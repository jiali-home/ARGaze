import os
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

from slowfast.models import MODEL_REGISTRY
from slowfast.utils import logging

logger = logging.get_logger(__name__)

_DINOV3_MODEL_ALIASES = {
    "vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "vit-s16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "vit-b16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "vit-l16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}


class TransformerDecoderLayer(nn.Module):
    """Decoder block without causal mask (temporal AR via fed history)."""

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
        tgt2, _ = self.self_attn(tgt, tgt, tgt)
        tgt = self.norm1(tgt + self.dropout1(tgt2))

        tgt2, _ = self.cross_attn(tgt, memory, memory)
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        tgt2 = self.ffn(tgt)
        tgt = self.norm3(tgt + tgt2)
        return tgt


@MODEL_REGISTRY.register()
class DINOv3_ARHeatmapGazeTemplate(nn.Module):
    """
    AR heatmap gaze with dynamic template crops from previous gaze (tracking-style cue).

    At each timestep t:
      - Crop previous frame around gaze_{t-1} (GT for teacher forcing, prediction for inference)
      - Encode crop(s) with DINOv3 -> template tokens
      - Cross-attn memory = [template tokens, current frame tokens]
      - Decode current heatmap autoregressively (history heatmap tokens + query tokens)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        model_name = getattr(
            cfg.MODEL, "DINOV3_MODEL_NAME", "facebook/dinov3-vits16-pretrain-lvd1689m"
        )
        resolved_name = _DINOV3_MODEL_ALIASES.get(model_name, model_name)
        if resolved_name != model_name:
            logger.info(f"Resolved DINOv3 model alias '{model_name}' -> '{resolved_name}'")
        model_name = resolved_name

        # 1) DINOv3 encoder
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
        self.input_size = getattr(cfg.DATA, "TRAIN_CROP_SIZE", 224)
        self.use_roi_prompt = getattr(cfg.MODEL, "USE_ROI_PROMPT", False)
        self.use_template_tokens = getattr(cfg.MODEL, "USE_TEMPLATE_TOKENS", True)
        self.use_original_template_tokens = getattr(
            cfg.MODEL, "USE_ORIGINAL_TEMPLATE_TOKENS", False
        )
        self.roi_scale = float(getattr(cfg.MODEL, "ROI_SCALE", 0.25))
        self.roi_grid_size = int(getattr(cfg.MODEL, "ROI_GRID_SIZE", 16))

        # Template crop scales (fraction of min(H, W))
        self.template_scales: List[float] = getattr(cfg.MODEL, "TEMPLATE_SCALES", [0.25])
        if isinstance(self.template_scales, (float, int)):
            self.template_scales = [float(self.template_scales)]

        # Number of historical frames to use as templates
        self.template_history_length = getattr(cfg.MODEL, "TEMPLATE_HISTORY_LENGTH", 1)

        # Whether to use gaze position as crop center for templates
        self.template_use_gaze_center = getattr(cfg.MODEL, "TEMPLATE_USE_GAZE_CENTER", True)

        # Backward compatibility: USE_FULL_FRAME_TEMPLATE takes precedence
        use_full_frame_template = getattr(cfg.MODEL, "USE_FULL_FRAME_TEMPLATE", False)
        if use_full_frame_template:
            self.template_use_gaze_center = False
        self.use_full_frame_template = use_full_frame_template  # Keep for logging

        # Use token type embeddings to distinguish history/query/template tokens
        self.use_token_type_embed = getattr(cfg.MODEL, "USE_TOKEN_TYPE_EMBED", True)

        # Use temporal embeddings to indicate time order of history tokens
        self.use_temporal_embed = getattr(cfg.MODEL, "USE_TEMPORAL_EMBED", True)

        # Ablation: Use ROI prompt instead of template encoding
        # When True, replaces template crops with ROI marking on current frame
        # Memory becomes: current_tokens (with ROI embed at gaze_{t-1} location)
        self.use_roi_instead_of_template = getattr(cfg.MODEL, "USE_ROI_INSTEAD_OF_TEMPLATE", False)

        # 2) Multi-scale feature fusion
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

        # 3) Heatmap tokenization
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

        # Embeddings: 0=history, 1=query, 2=template
        self.token_type_embed = nn.Embedding(3, self.hidden_dim)
        self.temporal_pos_embed = nn.Embedding(self.history_length, self.hidden_dim)

        # Query tokens (one per patch)
        self.num_query_tokens = self.patch_h * self.patch_w
        self.query_tokens = nn.Parameter(
            torch.randn(1, self.num_query_tokens, self.hidden_dim) * 0.02
        )

        # 4) Transformer decoder
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

        # 5) Conv decoder to heatmap
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

        # ROI prompt embedding (added to patch tokens)
        self.roi_prompt_embed = nn.Parameter(torch.zeros(self.hidden_dim))

        # Positional encoding cache and init heatmap
        self.pos_encoding_cache = {}
        self.register_buffer(
            "init_heatmap",
            torch.zeros(1, 1, self.heatmap_size, self.heatmap_size),
            persistent=False,
        )

        logger.info("DINOv3_ARHeatmapGazeTemplate initialized")
        logger.info(f"  - Use ROI instead of template: {self.use_roi_instead_of_template}")
        if self.use_roi_instead_of_template:
            logger.info(f"  - ROI prompt replaces template encoding (ablation mode)")
            logger.info(f"  - ROI scale: {self.roi_scale}, grid size: {self.roi_grid_size}")
        else:
            logger.info(f"  - Template history length: {self.template_history_length}")
            logger.info(f"  - Template use gaze center: {self.template_use_gaze_center}")
            if self.template_use_gaze_center:
                logger.info(f"  - Template scales (gaze-centered): {self.template_scales}")
                logger.info(f"  - Template center affected by scheduled sampling: True")
            else:
                logger.info(f"  - Template mode: Full frame (no gaze-centered crop)")
        logger.info(f"  - Use token type embeddings: {self.use_token_type_embed}")
        logger.info(f"  - Use temporal embeddings: {self.use_temporal_embed}")
        logger.info(f"  - Use original template tokens: {self.use_original_template_tokens}")

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
        tokens = []
        # Conditionally add token type embedding for history tokens
        if self.use_token_type_embed:
            hist_type = self.token_type_embed(torch.tensor(0, device=pos_encoding.device))
        else:
            hist_type = 0.0

        for idx, hm in enumerate(heatmaps):
            hm_down = self.hm_downsample(hm)
            hm_tok = self.hm_to_tokens(hm_down)
            hm_tok = hm_tok.flatten(2).permute(0, 2, 1)

            # Conditionally add temporal embedding for history tokens
            if self.use_temporal_embed:
                temporal_embed = self.temporal_pos_embed(torch.tensor(idx, device=hm_tok.device))
            else:
                temporal_embed = 0.0

            # Add embeddings based on flags
            # - hist_type is 0.0 if USE_TOKEN_TYPE_EMBED=False
            # - temporal_embed is 0.0 if USE_TEMPORAL_EMBED=False
            if self.use_token_type_embed and self.use_temporal_embed:
                hm_tok = (
                    hm_tok
                    + pos_encoding
                    + temporal_embed.view(1, 1, -1)
                    + hist_type.view(1, 1, -1)
                )
            elif self.use_token_type_embed and not self.use_temporal_embed:
                hm_tok = (
                    hm_tok
                    + pos_encoding
                    + hist_type.view(1, 1, -1)
                )
            elif not self.use_token_type_embed and self.use_temporal_embed:
                hm_tok = (
                    hm_tok
                    + pos_encoding
                    + temporal_embed.view(1, 1, -1)
                )
            else:  # Both disabled
                hm_tok = hm_tok + pos_encoding

            tokens.append(hm_tok)
        if len(tokens) == 0:
            return None
        return torch.cat(tokens, dim=1)

    def _heatmap_to_coords(self, heatmap):
        """Soft-argmax to get normalized coords from heatmap (B,1,H,W)."""
        if heatmap.dim() == 4:
            hm = heatmap
        else:
            hm = heatmap.unsqueeze(1)
        B, _, H, W = hm.shape
        hm_flat = hm.view(B, -1)
        prob = F.softmax(hm_flat, dim=-1).view(B, H, W)

        y_coords = torch.linspace(0, 1, steps=H, device=hm.device).view(1, H, 1)
        x_coords = torch.linspace(0, 1, steps=W, device=hm.device).view(1, 1, W)
        y_expect = (prob * y_coords).sum(dim=(1, 2))
        x_expect = (prob * x_coords).sum(dim=(1, 2))
        coords = torch.stack([x_expect, y_expect], dim=-1)
        return coords

    def _crop_and_resize(self, frame, center_norm, scales: List[float]):
        """Crop around center (normalized) with given scales; return tensor of crops."""
        _, H, W = frame.shape
        cx = float(center_norm[0]) * (W - 1)
        cy = float(center_norm[1]) * (H - 1)
        crops = []
        for s in scales:
            crop_size = max(1, int(round(s * min(H, W))))
            half = crop_size // 2
            x0 = int(round(cx)) - half
            y0 = int(round(cy)) - half
            x1 = x0 + crop_size
            y1 = y0 + crop_size

            pad_left = max(0, -x0)
            pad_top = max(0, -y0)
            pad_right = max(0, x1 - W)
            pad_bottom = max(0, y1 - H)

            x0_clamped = max(0, x0)
            y0_clamped = max(0, y0)
            x1_clamped = min(W, x1)
            y1_clamped = min(H, y1)

            patch = frame[:, y0_clamped:y1_clamped, x0_clamped:x1_clamped]
            if any(p > 0 for p in [pad_left, pad_right, pad_top, pad_bottom]):
                patch = F.pad(
                    patch,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="reflect",
                )
            patch = F.interpolate(
                patch.unsqueeze(0),
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            crops.append(patch)
        return crops

    def _build_patch_sampling_grid(self, centers, scale, device):
        """Build grid for sampling patch tokens at a given scale."""
        B = centers.shape[0]
        H = self.patch_h
        W = self.patch_w
        crop_size = scale * min(H, W)
        grids = []
        for b in range(B):
            cx = centers[b, 0] * (W - 1)
            cy = centers[b, 1] * (H - 1)
            x = torch.linspace(
                cx - crop_size / 2.0,
                cx + crop_size / 2.0,
                steps=W,
                device=device,
            )
            y = torch.linspace(
                cy - crop_size / 2.0,
                cy + crop_size / 2.0,
                steps=H,
                device=device,
            )
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            grid = torch.stack((xx, yy), dim=-1)
            grid[..., 0] = grid[..., 0] / (W - 1) * 2.0 - 1.0
            grid[..., 1] = grid[..., 1] / (H - 1) * 2.0 - 1.0
            grids.append(grid)
        return torch.stack(grids, dim=0)

    def _sample_template_from_embeddings(self, frame_tokens, centers, scales: List[float]):
        """Sample template tokens from precomputed patch embeddings."""
        B = frame_tokens.shape[0]
        tokens_map = (
            frame_tokens.view(B, self.patch_h, self.patch_w, self.hidden_dim)
            .permute(0, 3, 1, 2)
        )
        all_scales = []
        for scale in scales:
            grid = self._build_patch_sampling_grid(centers, scale, frame_tokens.device)
            sampled = F.grid_sample(
                tokens_map,
                grid,
                mode="bilinear",
                padding_mode="reflection",
                align_corners=True,
            )
            sampled = sampled.permute(0, 2, 3, 1).reshape(
                B, self.num_query_tokens, self.hidden_dim
            )
            all_scales.append(sampled)
        return torch.cat(all_scales, dim=1)

    def _encode_template(
        self,
        frames_history,
        centers_history,
        context_manager,
        template_frame_tokens=None,
    ):
        """
        Encode multiple historical frames as templates.

        Args:
            frames_history: list of (B, C, H, W) tensors, length = template_history_length
            centers_history: list of (B, 2) tensors or None, length = template_history_length
                             Each element is the gaze center for the corresponding frame
            context_manager: torch context manager for encoder
            template_frame_tokens: list of (B, num_patches, hidden_dim) if reusing original embeddings

        Returns:
            template_tokens: (B, total_template_tokens, hidden_dim)
                where total_template_tokens = template_history_length * num_scales * num_patches
                (or template_history_length * num_patches if using full frame)
        """
        B = frames_history[0].shape[0]
        all_template_tokens = []

        # Process each historical frame
        for frame_idx, (frame, center) in enumerate(zip(frames_history, centers_history)):
            if template_frame_tokens is not None:
                frame_tokens = template_frame_tokens[frame_idx]
                if not self.template_use_gaze_center:
                    patch_tokens = frame_tokens
                else:
                    if center is None:
                        center = torch.full((B, 2), 0.5, device=frame_tokens.device)
                    patch_tokens = self._sample_template_from_embeddings(
                        frame_tokens, center, self.template_scales
                    )
                all_template_tokens.append(patch_tokens)
                continue

            C, H, W = frame.shape[1:]
            crops = []

            # Two modes for template extraction:
            # 1. template_use_gaze_center=False: always use full frame
            # 2. template_use_gaze_center=True: use gaze-centered crops
            if not self.template_use_gaze_center:
                # Full-frame template mode: use entire frame for all samples
                for b in range(B):
                    patch = F.interpolate(
                        frame[b].unsqueeze(0),
                        size=(self.input_size, self.input_size),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                    crops.append(patch)
            else:
                # Gaze-centered crop mode (default)
                for b in range(B):
                    if center is None:
                        # No gaze center available: use center of frame as default
                        # Create num_scales crops (all centered at [0.5, 0.5])
                        default_center = torch.tensor([0.5, 0.5], device=frame.device)
                        crops.extend(self._crop_and_resize(frame[b], default_center, self.template_scales))
                    else:
                        crops.extend(self._crop_and_resize(frame[b], center[b], self.template_scales))

            crops = torch.stack(crops, dim=0)  # (B*num_scales or B, C, input_size, input_size)
            with context_manager:
                crops = self._preprocess_frames(crops)
                outputs = self.encoder(pixel_values=crops)
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

            patch_tokens = self.feature_proj(patch_tokens)

            # Reshape logic depends on mode
            if not self.template_use_gaze_center:
                # Full-frame mode: each sample gets 1 template -> (B, num_patches, hidden_dim)
                # No reshaping needed, already (B, num_patches, hidden_dim)
                pass
            else:
                # Gaze-centered mode: reshape to (B, num_scales*num_patches, hidden_dim)
                num_scales = len(self.template_scales)
                patch_tokens = patch_tokens.view(num_scales, B, num_spatial_patches, self.hidden_dim)
                patch_tokens = patch_tokens.permute(1, 0, 2, 3).reshape(B, num_scales * num_spatial_patches, self.hidden_dim)

            all_template_tokens.append(patch_tokens)

        # Concatenate all template tokens from different historical frames
        # Result: (B, template_history_length * num_scales * num_patches, hidden_dim)
        template_tokens = torch.cat(all_template_tokens, dim=1)

        return template_tokens

    def encode_single_frame(self, frame, context_manager=None):
        """Encode a single frame into patch tokens (B, num_patches, hidden_dim)."""
        encoder_params_grad = any(param.requires_grad for param in self.encoder.parameters())
        if context_manager is None:
            context_manager = torch.enable_grad() if encoder_params_grad else torch.no_grad()

        with context_manager:
            frames = self._preprocess_frames(frame)
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

        patch_tokens = self.feature_proj(patch_tokens)
        return patch_tokens

    def encode_template_crops(self, frames_history, centers_history, context_manager=None):
        """Encode template crops from historical frames."""
        encoder_params_grad = any(param.requires_grad for param in self.encoder.parameters())
        if context_manager is None:
            context_manager = torch.enable_grad() if encoder_params_grad else torch.no_grad()
        return self._encode_template(
            frames_history=frames_history,
            centers_history=centers_history,
            context_manager=context_manager,
            template_frame_tokens=None,
        )

    def encode_template_from_tokens(self, template_frame_tokens, centers_history):
        """Encode template tokens from cached frame embeddings."""
        B = template_frame_tokens[0].shape[0]
        all_template_tokens = []
        for frame_tokens, center in zip(template_frame_tokens, centers_history):
            if not self.template_use_gaze_center:
                patch_tokens = frame_tokens
            else:
                if center is None:
                    center = torch.full((B, 2), 0.5, device=frame_tokens.device)
                patch_tokens = self._sample_template_from_embeddings(
                    frame_tokens, center, self.template_scales
                )
            all_template_tokens.append(patch_tokens)
        return torch.cat(all_template_tokens, dim=1)

    def streaming_decode_step(
        self,
        t,
        current_tokens,
        template_tokens,
        predicted_heatmaps,
        pos_encoding,
    ):
        """Decode a single streaming step using cached predicted heatmaps."""
        B = current_tokens.shape[0]
        history_heatmaps = self._get_history_heatmaps(
            t, None, predicted_heatmaps, B, train_ar=False, ss_prob=0.0
        )
        hist_tokens = self._heatmaps_to_tokens(history_heatmaps, pos_encoding)

        query_tokens = self.query_tokens.expand(B, -1, -1)
        if self.use_token_type_embed:
            query_type = self.token_type_embed(torch.tensor(1, device=current_tokens.device))
            template_type = self.token_type_embed(torch.tensor(2, device=current_tokens.device))
            query_tokens = query_tokens + pos_encoding + query_type.view(1, 1, -1)
        else:
            template_type = 0.0
            query_tokens = query_tokens + pos_encoding

        if hist_tokens is not None:
            tgt = torch.cat([hist_tokens, query_tokens], dim=1)
        else:
            tgt = query_tokens

        current_tokens = current_tokens + pos_encoding

        if self.use_roi_instead_of_template:
            center_xy = self._get_roi_center(t, None, predicted_heatmaps, B, train_ar=False, ss_prob=0.0)
            roi_mask = self._build_roi_mask(center_xy, self.roi_grid_size, current_tokens.device)
            if (self.roi_grid_size, self.roi_grid_size) != (self.patch_h, self.patch_w):
                roi_mask = F.interpolate(
                    roi_mask,
                    size=(self.patch_h, self.patch_w),
                    mode="nearest",
                )
            roi_mask = roi_mask.view(B, 1, -1)
            current_tokens = current_tokens + roi_mask.transpose(1, 2) * self.roi_prompt_embed.view(1, 1, -1)
            template_tokens = None
            memory = current_tokens
        else:
            if template_tokens is not None:
                num_template_tokens = template_tokens.shape[1]
                num_pos_repeats = num_template_tokens // pos_encoding.shape[1]
                if num_pos_repeats > 1:
                    template_pos_encoding = pos_encoding.repeat(1, num_pos_repeats, 1)
                else:
                    template_pos_encoding = pos_encoding
                if self.use_token_type_embed:
                    template_tokens = template_tokens + template_pos_encoding + template_type.view(1, 1, -1)
                else:
                    template_tokens = template_tokens + template_pos_encoding

            if self.use_roi_prompt:
                center_xy = self._get_roi_center(t, None, predicted_heatmaps, B, train_ar=False, ss_prob=0.0)
                roi_mask = self._build_roi_mask(center_xy, self.roi_grid_size, current_tokens.device)
                if (self.roi_grid_size, self.roi_grid_size) != (self.patch_h, self.patch_w):
                    roi_mask = F.interpolate(
                        roi_mask,
                        size=(self.patch_h, self.patch_w),
                        mode="nearest",
                    )
                roi_mask = roi_mask.view(B, 1, -1)
                current_tokens = current_tokens + roi_mask.transpose(1, 2) * self.roi_prompt_embed.view(1, 1, -1)

            if template_tokens is None:
                memory = current_tokens
            else:
                memory = torch.cat([template_tokens, current_tokens], dim=1)

        for layer in self.decoder_layers:
            tgt = layer(tgt, memory)

        heatmap_tokens = tgt[:, -self.num_query_tokens :, :]
        feat = heatmap_tokens.permute(0, 2, 1).reshape(
            B, self.hidden_dim, self.patch_h, self.patch_w
        )
        feat = self.pixel_decoder_conv1(feat)
        feat = self.pixel_decoder_conv2(feat)
        feat = self.pixel_final_upsample(feat)
        heatmap_t = self.pixel_final_proj(feat)
        return heatmap_t

    def forward(self, x, gt_heatmap=None, train_ar=True, ss_prob=0.0, gt_heatmap_center=None):
        x = x[0] if isinstance(x, list) else x
        if x.dim() == 4:
            x = x.unsqueeze(2)

        B, C, T, H_in, W_in = x.shape
        if gt_heatmap is not None and gt_heatmap.dim() == 4:
            gt_heatmap = gt_heatmap.unsqueeze(1)
        if gt_heatmap_center is not None and gt_heatmap_center.dim() == 4:
            gt_heatmap_center = gt_heatmap_center.unsqueeze(1)

        # Prepare raw frames for template cropping
        raw_frames = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)

        # Encode all frames once for main visual tokens
        frames_flat = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H_in, W_in)
        encoder_params_grad = any(param.requires_grad for param in self.encoder.parameters())
        context_manager = torch.enable_grad() if encoder_params_grad else torch.no_grad()

        with context_manager:
            frames_flat = self._preprocess_frames(frames_flat)
            outputs = self.encoder(pixel_values=frames_flat)
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

        patch_tokens = self.feature_proj(patch_tokens)
        patch_tokens = patch_tokens.view(B, T, num_spatial_patches, self.hidden_dim)
        patch_tokens_base = patch_tokens
        pos_encoding = self._get_2d_pos_encoding(self.patch_h, self.patch_w, patch_tokens.device)
        patch_tokens = patch_tokens + pos_encoding.unsqueeze(0)

        return self._autoregressive_decode(
            visual_features=patch_tokens,
            visual_features_base=patch_tokens_base,
            raw_frames=raw_frames,
            gt_heatmap=gt_heatmap,
            gt_heatmap_center=gt_heatmap_center,
            B=B,
            T=T,
            train_ar=train_ar,
            ss_prob=ss_prob,
            pos_encoding=pos_encoding,
            context_manager=context_manager,
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

    def _get_template_center(
        self,
        t,
        gt_heatmap,
        predicted_heatmaps,
        B,
        default_full_frame=False,
        gt_heatmap_center=None,
    ):
        """Get gaze center for a single frame (legacy method, kept for backward compatibility)."""
        if t == 0 and default_full_frame:
            # Signal to skip cropping by returning None
            return None
        src_t = t - 1
        if gt_heatmap_center is not None:
            hm = gt_heatmap_center[:, :, src_t, :, :]
            return self._heatmap_to_coords(hm)
        if gt_heatmap is not None:
            hm = gt_heatmap[:, :, src_t, :, :]
            return self._heatmap_to_coords(hm)
        if len(predicted_heatmaps) > src_t:
            return self._heatmap_to_coords(predicted_heatmaps[src_t])
        return torch.full((B, 2), 0.5, device=self.init_heatmap.device)

    def _get_template_frames_and_centers(
        self,
        t,
        raw_frames,
        gt_heatmap,
        predicted_heatmaps,
        B,
        T,
        train_ar=True,
        ss_prob=0.0,
        gt_heatmap_center=None,
    ):
        """
        Get multiple historical frames and their corresponding gaze centers for template.

        Args:
            t: current time step
            raw_frames: (B, T, C, H, W) all raw frames
            gt_heatmap: (B, 1, T, H, W) ground truth heatmaps (or None)
            gt_heatmap_center: (B, 1, T, H, W) GT heatmaps used only for template center (or None)
            predicted_heatmaps: list of predicted heatmaps
            B: batch size
            T: total time steps
            train_ar: whether in AR training mode (enables scheduled sampling)
            ss_prob: scheduled sampling probability (0.0-1.0)

        Returns:
            frames_history: list of (B, C, H, W), length = template_history_length
            centers_history: list of (B, 2) or None, length = template_history_length
            frame_indices: list of int indices into T
        """
        frames_history = []
        centers_history = []
        frame_indices = []

        for idx in range(self.template_history_length):
            src_t = t - self.template_history_length + idx

            # Get the frame
            if src_t < 0:
                # Before sequence start: use first frame
                frame = raw_frames[:, 0]
                frame_indices.append(0)
            else:
                frame = raw_frames[:, src_t]
                frame_indices.append(src_t)
            frames_history.append(frame)

            # Get the gaze center for this frame
            if not self.template_use_gaze_center:
                # Not using gaze-centered crops, no need for centers
                centers_history.append(None)
            else:
                # Using gaze-centered crops, need gaze positions
                if src_t < 0:
                    # Before sequence start: use default center or None
                    centers_history.append(None)
                else:
                    # Get gaze center from GT or prediction (with scheduled sampling)
                    use_pred_center = False
                    center_gt = gt_heatmap_center if gt_heatmap_center is not None else gt_heatmap
                    if center_gt is not None:
                        # Apply scheduled sampling to template center (same as history heatmaps)
                        if train_ar and len(predicted_heatmaps) > src_t and torch.rand(1).item() < ss_prob:
                            use_pred_center = True
                    else:
                        use_pred_center = len(predicted_heatmaps) > src_t

                    if use_pred_center and len(predicted_heatmaps) > src_t:
                        # Use predicted heatmap for template center
                        center = self._heatmap_to_coords(predicted_heatmaps[src_t])
                    else:
                        # Use GT heatmap for template center (teacher forcing)
                        if center_gt is not None:
                            hm = center_gt[:, :, src_t, :, :]
                            center = self._heatmap_to_coords(hm)
                        elif len(predicted_heatmaps) > src_t:
                            center = self._heatmap_to_coords(predicted_heatmaps[src_t])
                        else:
                            # No GT or prediction available: use center of frame
                            center = torch.full((B, 2), 0.5, device=self.init_heatmap.device)
                    centers_history.append(center)

        return frames_history, centers_history, frame_indices

    def _build_roi_mask(self, center_xy, grid_size, device):
        """
        Build binary ROI mask on a grid from normalized centers.
        center_xy: (B, 2) in [0, 1], order (x, y)
        Returns: (B, 1, grid_size, grid_size)
        """
        B = center_xy.size(0)
        cx = center_xy[:, 0] * grid_size
        cy = center_xy[:, 1] * grid_size
        half = (self.roi_scale * grid_size) / 2.0

        x_min = (cx - half).clamp(0.0, grid_size)
        x_max = (cx + half).clamp(0.0, grid_size)
        y_min = (cy - half).clamp(0.0, grid_size)
        y_max = (cy + half).clamp(0.0, grid_size)

        xs = torch.arange(grid_size, device=device).view(1, 1, 1, grid_size)
        ys = torch.arange(grid_size, device=device).view(1, 1, grid_size, 1)
        xs = xs.expand(B, 1, grid_size, grid_size)
        ys = ys.expand(B, 1, grid_size, grid_size)

        x_min = x_min.view(B, 1, 1, 1)
        x_max = x_max.view(B, 1, 1, 1)
        y_min = y_min.view(B, 1, 1, 1)
        y_max = y_max.view(B, 1, 1, 1)

        mask = (xs >= x_min) & (xs < x_max) & (ys >= y_min) & (ys < y_max)
        return mask.float()

    def _get_roi_center(self, t, gt_heatmap, predicted_heatmaps, B, train_ar, ss_prob):
        if t <= 0:
            return torch.full((B, 2), 0.5, device=self.init_heatmap.device)
        src_t = t - 1
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
        return self._heatmap_to_coords(hm)

    def _autoregressive_decode(
        self,
        visual_features,
        visual_features_base,
        raw_frames,
        gt_heatmap,
        gt_heatmap_center,
        B,
        T,
        train_ar,
        ss_prob,
        pos_encoding,
        context_manager,
    ):
        heatmap_outputs = []
        predicted_heatmaps = []

        # Conditionally get token type embeddings for query and template
        if self.use_token_type_embed:
            query_type = self.token_type_embed(torch.tensor(1, device=visual_features.device))
            template_type = self.token_type_embed(torch.tensor(2, device=visual_features.device))
        else:
            query_type = 0.0
            template_type = 0.0

        for t in range(T):
            history_heatmaps = self._get_history_heatmaps(
                t, gt_heatmap, predicted_heatmaps, B, train_ar, ss_prob
            )
            hist_tokens = self._heatmaps_to_tokens(history_heatmaps, pos_encoding)

            # Query tokens
            query_tokens = self.query_tokens.expand(B, -1, -1)
            if self.use_token_type_embed:
                query_tokens = query_tokens + pos_encoding + query_type.view(1, 1, -1)
            else:
                query_tokens = query_tokens + pos_encoding

            if hist_tokens is not None:
                tgt = torch.cat([hist_tokens, query_tokens], dim=1)
            else:
                tgt = query_tokens

            # Get current frame tokens
            current_tokens = visual_features[:, t, :, :]

            # Two modes for providing template information:
            # Mode 1 (ablation): ROI prompt replaces template encoding
            # Mode 2 (original): Template crop encoding + optional ROI prompt
            if self.use_roi_instead_of_template:
                # Ablation mode: Use ROI prompt on current frame instead of template encoding
                # Get gaze center from previous frame (t-1)
                center_xy = self._get_roi_center(t, gt_heatmap, predicted_heatmaps, B, train_ar, ss_prob)
                roi_mask = self._build_roi_mask(center_xy, self.roi_grid_size, current_tokens.device)
                if (self.roi_grid_size, self.roi_grid_size) != (self.patch_h, self.patch_w):
                    roi_mask = F.interpolate(
                        roi_mask,
                        size=(self.patch_h, self.patch_w),
                        mode="nearest",
                    )
                roi_mask = roi_mask.view(B, 1, -1)
                current_tokens = current_tokens + roi_mask.transpose(1, 2) * self.roi_prompt_embed.view(1, 1, -1)

                # Memory is just current tokens (with ROI marking)
                template_tokens = None
                memory = current_tokens

            else:
                # Original mode: Template encoding from historical frames
                if self.use_template_tokens:
                    frames_history, centers_history, frame_indices = self._get_template_frames_and_centers(
                        t,
                        raw_frames,
                        gt_heatmap,
                        predicted_heatmaps,
                        B,
                        T,
                        train_ar,
                        ss_prob,
                        gt_heatmap_center,
                    )
                    template_frame_tokens = None
                    if self.use_original_template_tokens:
                        template_frame_tokens = [
                            visual_features_base[:, idx, :, :] for idx in frame_indices
                        ]
                    template_tokens = self._encode_template(
                        frames_history=frames_history,
                        centers_history=centers_history,
                        context_manager=context_manager,
                        template_frame_tokens=template_frame_tokens,
                    )

                    # Expand pos_encoding to match template_tokens shape
                    # template_tokens: (B, num_template_tokens, hidden_dim)
                    # pos_encoding: (1, 196, hidden_dim)
                    # Need to repeat pos_encoding for each template frame/scale
                    num_template_tokens = template_tokens.shape[1]
                    num_pos_repeats = num_template_tokens // pos_encoding.shape[1]
                    if num_pos_repeats > 1:
                        # Repeat spatial positional encoding for multiple template frames/scales
                        template_pos_encoding = pos_encoding.repeat(1, num_pos_repeats, 1)
                    else:
                        template_pos_encoding = pos_encoding

                    if self.use_token_type_embed:
                        template_tokens = template_tokens + template_pos_encoding + template_type.view(1, 1, -1)
                    else:
                        template_tokens = template_tokens + template_pos_encoding
                else:
                    template_tokens = None

                # Optional ROI prompt on current frame tokens (binary mask on patch grid)
                if self.use_roi_prompt:
                    center_xy = self._get_roi_center(t, gt_heatmap, predicted_heatmaps, B, train_ar, ss_prob)
                    roi_mask = self._build_roi_mask(center_xy, self.roi_grid_size, current_tokens.device)
                    if (self.roi_grid_size, self.roi_grid_size) != (self.patch_h, self.patch_w):
                        roi_mask = F.interpolate(
                            roi_mask,
                            size=(self.patch_h, self.patch_w),
                            mode="nearest",
                        )
                    roi_mask = roi_mask.view(B, 1, -1)
                    current_tokens = current_tokens + roi_mask.transpose(1, 2) * self.roi_prompt_embed.view(1, 1, -1)

                # Cross-attn memory: [template tokens (optional), current frame tokens]
                if template_tokens is None:
                    memory = current_tokens
                else:
                    memory = torch.cat([template_tokens, current_tokens], dim=1)

            for layer in self.decoder_layers:
                tgt = layer(tgt, memory)

            heatmap_tokens = tgt[:, -self.num_query_tokens :, :]
            feat = heatmap_tokens.permute(0, 2, 1).reshape(
                B, self.hidden_dim, self.patch_h, self.patch_w
            )
            feat = self.pixel_decoder_conv1(feat)
            feat = self.pixel_decoder_conv2(feat)
            feat = self.pixel_final_upsample(feat)
            heatmap_t = self.pixel_final_proj(feat)

            heatmap_outputs.append(heatmap_t)
            predicted_heatmaps.append(heatmap_t.detach())

        heatmap = torch.stack(heatmap_outputs, dim=2)
        return heatmap
