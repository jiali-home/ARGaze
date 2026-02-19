import os
import math
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
from slowfast.models import MODEL_REGISTRY
from slowfast.utils import logging

logger = logging.get_logger(__name__)


class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder block without causal masking.

    Self-attention models interactions between history heatmap tokens and
    current query tokens; cross-attention fuses visual features for the
    current frame.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt, memory):
        # Self-attention over history + query tokens (no causal mask)
        tgt2, _ = self.self_attn(tgt, tgt, tgt)
        tgt = self.norm1(tgt + self.dropout1(tgt2))

        # Cross-attention to visual features
        tgt2, _ = self.cross_attn(tgt, memory, memory)
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        # Feed-forward network
        tgt2 = self.ffn(tgt)
        tgt = self.norm3(tgt + tgt2)
        return tgt


@MODEL_REGISTRY.register()
class DINOv3_ARHeatmapGaze(nn.Module):
    """
    Autoregressive heatmap-based gaze estimation.

    P(G_t | G_{t-N:t-1}, I_t) where G_t is a dense gaze heatmap. Temporal
    autoregression is driven by feeding the previous heatmap window as input
    tokens; no causal mask is used in the decoder (aligned with ARTrack).
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        model_name = getattr(
            cfg.MODEL, "DINOV3_MODEL_NAME", "facebook/dinov3-vits16-pretrain-lvd1689m"
        )

        # =========================
        # 1. DINOv3 Encoder
        # =========================
        self.processor = None
        hf_kwargs = {}
        env_token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
        if env_token:
            hf_kwargs["use_auth_token"] = env_token
        self.processor = AutoImageProcessor.from_pretrained(model_name, **hf_kwargs)

        self.use_multiscale = getattr(cfg.MODEL, "USE_MULTISCALE_FEATURES", True)
        if self.use_multiscale:
            try:
                self.encoder = AutoModel.from_pretrained(
                    model_name, output_hidden_states=True, **hf_kwargs
                )
            except TypeError:
                self.encoder = AutoModel.from_pretrained(
                    model_name, output_hidden_states=True, token=True
                )
            self.multiscale_layers = getattr(cfg.MODEL, "MULTISCALE_LAYERS", [-3, -2, -1])
        else:
            self.encoder = AutoModel.from_pretrained(model_name, **hf_kwargs)

        freeze_encoder = getattr(cfg.MODEL, "FREEZE_ENCODER", True)
        unfreeze_last_k = getattr(cfg.MODEL, "UNFREEZE_LAST_K_LAYERS", 0)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        if unfreeze_last_k > 0 and hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
            for block in self.encoder.encoder.layer[-unfreeze_last_k:]:
                for param in block.parameters():
                    param.requires_grad = True
        if freeze_encoder and unfreeze_last_k == 0:
            self.encoder.eval()
        else:
            self.encoder.train()

        self.hidden_dim = self.encoder.config.hidden_size
        self.drop = getattr(cfg.MODEL, "DROPOUT_RATE", 0.1)
        self.heatmap_size = getattr(cfg.MODEL, "HEATMAP_SIZE", 64)
        self.history_length = getattr(cfg.MODEL, "HISTORY_LENGTH", 3)

        # =========================
        # 2. Multi-scale Feature Fusion
        # =========================
        if self.use_multiscale:
            num_scales = len(self.multiscale_layers)
            self.multiscale_proj = nn.ModuleList(
                [nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(num_scales)]
            )
            self.pixel_base_dim = self.hidden_dim * num_scales
        else:
            self.pixel_base_dim = self.hidden_dim

        self.feature_proj = nn.Sequential(
            nn.Linear(self.pixel_base_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        # =========================
        # 3. Heatmap Tokenization
        # =========================
        self.patch_h = 14
        self.patch_w = 14
        self.hm_downsample = nn.Upsample(
            size=(self.patch_h, self.patch_w),
            mode="bilinear",
            align_corners=False,
        )
        self.hm_to_tokens = nn.Sequential(
            nn.Conv2d(1, self.hidden_dim, kernel_size=1),
            nn.LayerNorm([self.hidden_dim, self.patch_h, self.patch_w]),
        )

        # Temporal and token-type embeddings
        self.temporal_pos_embed = nn.Embedding(self.history_length, self.hidden_dim)
        self.token_type_embed = nn.Embedding(2, self.hidden_dim)  # 0=history, 1=query

        # Learnable query tokens for current heatmap (one per patch)
        self.num_query_tokens = self.patch_h * self.patch_w
        self.query_tokens = nn.Parameter(
            torch.randn(1, self.num_query_tokens, self.hidden_dim) * 0.02
        )

        # =========================
        # 4. Transformer Decoder
        # =========================
        self.num_decoder_layers = getattr(cfg.MODEL, "NUM_DECODER_LAYERS", 3)
        self.nhead = getattr(cfg.MODEL, "NUM_ATTENTION_HEADS", 8)
        self.dim_feedforward = getattr(cfg.MODEL, "DIM_FEEDFORWARD", self.hidden_dim * 4)

        self.decoder_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model=self.hidden_dim,
                    nhead=self.nhead,
                    dim_feedforward=self.dim_feedforward,
                    dropout=self.drop,
                )
                for _ in range(self.num_decoder_layers)
            ]
        )

        # =========================
        # 5. Conv Decoder to Heatmap
        # =========================
        self.pixel_decoder_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(self.drop),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.pixel_decoder_conv2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(self.drop),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        if self.heatmap_size != 56:
            if self.heatmap_size >= 112:
                self.pixel_final_upsample = nn.Sequential(
                    nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
                )
            else:
                self.pixel_final_upsample = nn.Upsample(
                    size=(self.heatmap_size, self.heatmap_size),
                    mode="bilinear",
                    align_corners=False,
                )
        else:
            self.pixel_final_upsample = nn.Identity()

        self.pixel_final_proj = nn.Conv2d(128, 1, kernel_size=1)

        # =========================
        # 6. Positional Encoding Cache
        # =========================
        self.pos_encoding_cache = {}
        self.register_buffer(
            "init_heatmap",
            torch.zeros(1, 1, self.heatmap_size, self.heatmap_size),
            persistent=False,
        )

        logger.info("DINOv3_ARHeatmapGaze initialized")
        logger.info(f"  - Visual encoder: {model_name}")
        logger.info(f"  - Hidden dim: {self.hidden_dim}")
        logger.info(f"  - Heatmap size: {self.heatmap_size}")
        logger.info(f"  - History length: {self.history_length}")
        logger.info(f"  - Decoder layers: {self.num_decoder_layers}")

    def _preprocess_frames(self, frames):
        if frames.max() <= 1.0:
            frames = frames * 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=frames.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=frames.device).view(1, 3, 1, 1)
        frames = (frames / 255.0 - mean) / std
        return frames

    def _get_2d_pos_encoding(self, height, width, device):
        cache_key = (height, width)
        if cache_key in self.pos_encoding_cache:
            return self.pos_encoding_cache[cache_key].to(device)

        pos_encoding = torch.zeros(height, width, self.hidden_dim, device=device)
        d_model = self.hidden_dim
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device).float()
            * -(math.log(10000.0) / d_model)
        )

        pos_h = torch.arange(height, device=device).unsqueeze(1).float()
        div_term_h = div_term[: d_model // 4].unsqueeze(0)
        pos_h_encoded = pos_h * div_term_h
        pos_encoding[:, :, 0 : d_model // 2 : 2] = (
            torch.sin(pos_h_encoded).unsqueeze(1).repeat(1, width, 1)
        )
        pos_encoding[:, :, 1 : d_model // 2 : 2] = (
            torch.cos(pos_h_encoded).unsqueeze(1).repeat(1, width, 1)
        )

        pos_w = torch.arange(width, device=device).unsqueeze(1).float()
        div_term_w = div_term[: d_model // 4].unsqueeze(0)
        pos_w_encoded = pos_w * div_term_w
        pos_encoding[:, :, d_model // 2 :: 2] = (
            torch.sin(pos_w_encoded).unsqueeze(0).repeat(height, 1, 1)
        )
        pos_encoding[:, :, d_model // 2 + 1 :: 2] = (
            torch.cos(pos_w_encoded).unsqueeze(0).repeat(height, 1, 1)
        )

        pos_encoding = pos_encoding.view(1, height * width, self.hidden_dim)
        self.pos_encoding_cache[cache_key] = pos_encoding.cpu()
        return pos_encoding

    def _heatmaps_to_tokens(self, heatmaps, pos_encoding):
        """
        Convert a list of heatmaps into patch tokens with positional encodings.
        """
        tokens = []
        hist_type = self.token_type_embed(torch.tensor(0, device=pos_encoding.device))
        for idx, hm in enumerate(heatmaps):
            hm_down = self.hm_downsample(hm)  # (B, 1, patch_h, patch_w)
            hm_tok = self.hm_to_tokens(hm_down)  # (B, hidden_dim, patch_h, patch_w)
            hm_tok = hm_tok.flatten(2).permute(0, 2, 1)  # (B, num_patches, hidden_dim)

            # Positional encodings: spatial + temporal
            temporal_embed = self.temporal_pos_embed(
                torch.tensor(idx, device=hm_tok.device)
            )
            hm_tok = (
                hm_tok
                + pos_encoding
                + temporal_embed.view(1, 1, -1)
                + hist_type.view(1, 1, -1)
            )
            tokens.append(hm_tok)

        if len(tokens) == 0:
            return None
        return torch.cat(tokens, dim=1)  # (B, history_len*num_patches, hidden_dim)

    def forward(self, x, gt_heatmap=None, train_ar=True, ss_prob=0.0):
        """
        Args:
            x: (B, C, T, H, W) or list with tensor as first element.
            gt_heatmap: (B, 1, T, Hm, Wm) or (B, T, Hm, Wm); used for teacher forcing.
            train_ar: If True, autoregressive mode with teacher forcing.
            ss_prob: Scheduled sampling prob (only when train_ar and gt provided).
        Returns:
            heatmap logits of shape (B, 1, T, heatmap_size, heatmap_size)
        """
        x = x[0] if isinstance(x, list) else x
        if x.dim() == 4:
            x = x.unsqueeze(2)

        B, C, T, H_in, W_in = x.shape

        if gt_heatmap is not None and gt_heatmap.dim() == 4:
            gt_heatmap = gt_heatmap.unsqueeze(1)

        frames = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H_in, W_in)

        encoder_params_grad = any(param.requires_grad for param in self.encoder.parameters())
        context_manager = torch.enable_grad() if encoder_params_grad else torch.no_grad()

        with context_manager:
            frames = self._preprocess_frames(frames)
            outputs = self.encoder(pixel_values=frames)

            if self.use_multiscale:
                hidden_states = outputs.hidden_states
                inter_features = []
                num_spatial_patches = 196
                for layer_idx in self.multiscale_layers:
                    layer_tokens = hidden_states[layer_idx][:, 1 : 1 + num_spatial_patches, :]
                    inter_features.append(layer_tokens)

                multiscale_features = []
                for feat, proj in zip(inter_features, self.multiscale_proj):
                    multiscale_features.append(proj(feat))

                patch_tokens = torch.cat(multiscale_features, dim=-1)
            else:
                num_spatial_patches = 196
                patch_tokens = outputs.last_hidden_state[:, 1 : 1 + num_spatial_patches, :]

        patch_tokens = self.feature_proj(patch_tokens)  # (B*T, num_patches, hidden_dim)
        patch_tokens = patch_tokens.view(B, T, num_spatial_patches, self.hidden_dim)

        pos_encoding = self._get_2d_pos_encoding(self.patch_h, self.patch_w, patch_tokens.device)
        patch_tokens = patch_tokens + pos_encoding.unsqueeze(0)

        return self._autoregressive_decode(
            visual_features=patch_tokens,
            gt_heatmap=gt_heatmap,
            B=B,
            T=T,
            train_ar=train_ar,
            ss_prob=ss_prob,
            pos_encoding=pos_encoding,
        )

    def _get_history_heatmaps(self, t, gt_heatmap, predicted_heatmaps, B, train_ar, ss_prob):
        history = []
        for idx in range(self.history_length):
            src_t = t - self.history_length + idx
            if src_t < 0:
                history.append(self.init_heatmap.expand(B, -1, -1, -1))
                continue

            use_pred = False
            if gt_heatmap is not None:
                if train_ar and len(predicted_heatmaps) > src_t and torch.rand(1).item() < ss_prob:
                    use_pred = True
            else:
                use_pred = len(predicted_heatmaps) > src_t

            if use_pred and len(predicted_heatmaps) > src_t:
                hm = predicted_heatmaps[src_t].detach()
            else:
                if gt_heatmap is not None:
                    hm = gt_heatmap[:, :, src_t, :, :]
                elif len(predicted_heatmaps) > src_t:
                    hm = predicted_heatmaps[src_t].detach()
                else:
                    hm = self.init_heatmap.expand(B, -1, -1, -1)
            history.append(hm)
        return history

    def _autoregressive_decode(
        self, visual_features, gt_heatmap, B, T, train_ar, ss_prob, pos_encoding
    ):
        heatmap_outputs = []
        predicted_heatmaps = []

        query_type = self.token_type_embed(torch.tensor(1, device=visual_features.device))

        for t in range(T):
            history_heatmaps = self._get_history_heatmaps(
                t, gt_heatmap, predicted_heatmaps, B, train_ar, ss_prob
            )
            hist_tokens = self._heatmaps_to_tokens(history_heatmaps, pos_encoding)

            query_tokens = self.query_tokens.expand(B, -1, -1)
            query_tokens = query_tokens + pos_encoding + query_type.view(1, 1, -1)

            if hist_tokens is not None:
                tgt = torch.cat([hist_tokens, query_tokens], dim=1)
            else:
                tgt = query_tokens

            memory = visual_features[:, t, :, :]  # (B, num_patches, hidden_dim)

            for layer in self.decoder_layers:
                tgt = layer(tgt, memory)

            heatmap_tokens = tgt[:, -self.num_query_tokens :, :]  # (B, num_query_tokens, hidden_dim)
            feat = heatmap_tokens.permute(0, 2, 1).reshape(
                B, self.hidden_dim, self.patch_h, self.patch_w
            )

            feat = self.pixel_decoder_conv1(feat)
            feat = self.pixel_decoder_conv2(feat)
            feat = self.pixel_final_upsample(feat)
            heatmap_t = self.pixel_final_proj(feat)  # (B, 1, Hm, Wm)

            heatmap_outputs.append(heatmap_t)
            predicted_heatmaps.append(heatmap_t.detach())

        heatmap = torch.stack(heatmap_outputs, dim=2)  # (B, 1, T, Hm, Wm)
        return heatmap
