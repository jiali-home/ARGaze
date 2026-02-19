#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple
from torch.nn.init import trunc_normal_
from functools import partial
from fairscale.nn.checkpoint import checkpoint_wrapper

import slowfast.utils.weight_init_helper as init_helper
from slowfast.models.batchnorm_helper import get_norm
from slowfast.models.attention import MultiScaleBlock, MultiScaleDecoderBlock
from slowfast.models.causal_utils import convert_conv3d_to_causal
from slowfast.models.global_attention import GlobalLocalBlock
from slowfast.models.utils import round_width, validate_checkpoint_wrapper_import
from . import head_helper, resnet_helper, stem_helper
from .build import MODEL_REGISTRY

"""A More Flexible Video models."""

# ResNet_Baseline with maxpool: ['2D', '3D_full', '3D_a', '3D_b', '3D_c', '3D_d', '3D_e', '3D_cd']
# ResNet_3D with conv: ['2D_conv', '3D_a_conv', '3D_b_conv', '3D_c_conv', '3D_d_conv', '3D_e_conv', '3D_dev1', '3D_dev2', '3D_de_conv', '3D_full_conv']

model = '3D_full_56'

##############################################################
# Number of blocks for different stages given the model depth.
_MODEL_STAGE_DEPTH = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}

# Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    "c2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "c2d_nopool": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "i3d": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "i3d_nopool": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "slow": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[3]],  # res4 temporal kernel.
        [[3]],  # res5 temporal kernel.
    ],
    "slowfast": [
        [[1], [5]],  # conv1 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res2 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res3 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res4 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res5 temporal kernel for slow and fast pathway.
    ],
    "x3d": [
        [[5]],  # conv1 temporal kernels.
        [[3]],  # res2 temporal kernels.
        [[3]],  # res3 temporal kernels.
        [[3]],  # res4 temporal kernels.
        [[3]],  # res5 temporal kernels.
    ],
}

_POOL1 = {
    "c2d": [[2, 1, 1]],
    "c2d_nopool": [[1, 1, 1]],
    "i3d": [[2, 1, 1]],
    "i3d_nopool": [[1, 1, 1]],
    "slow": [[1, 1, 1]],
    "slowfast": [[1, 1, 1], [1, 1, 1]],
    "x3d": [[1, 1, 1]],
}



@MODEL_REGISTRY.register()
class GLC_Gaze_Causal(nn.Module):
    """
    Multiscale Vision Transformers with Global-Local Correlation for Egocentric Gaze Estimation
    """

    def __init__(self, cfg):
        super().__init__()
        # Get parameters.
        assert cfg.DATA.TRAIN_CROP_SIZE == cfg.DATA.TEST_CROP_SIZE

        # Defrost config if it's immutable (needed for modifying POOL_KV_STRIDE later)
        was_frozen = cfg.is_frozen()
        if was_frozen:
            cfg.defrost()

        self.cfg = cfg
        pool_first = cfg.MVIT.POOL_FIRST  # False

        # Prepare input.
        spatial_size = cfg.DATA.TRAIN_CROP_SIZE
        temporal_size = cfg.DATA.NUM_FRAMES  # 8
        in_chans = cfg.DATA.INPUT_CHANNEL_NUM[0]
        use_2d_patch = cfg.MVIT.PATCH_2D  # default false
        self.patch_stride = cfg.MVIT.PATCH_STRIDE
        if use_2d_patch:
            self.patch_stride = [1] + self.patch_stride

        # Prepare output.
        num_classes = cfg.MODEL.NUM_CLASSES
        embed_dim = cfg.MVIT.EMBED_DIM  # 96

        # Prepare backbone
        num_heads = cfg.MVIT.NUM_HEADS
        mlp_ratio = cfg.MVIT.MLP_RATIO
        qkv_bias = cfg.MVIT.QKV_BIAS  # True
        self.drop_rate = cfg.MVIT.DROPOUT_RATE  # 0
        depth = cfg.MVIT.DEPTH
        drop_path_rate = cfg.MVIT.DROPPATH_RATE  # 0.2
        mode = cfg.MVIT.MODE
        self.cls_embed_on = cfg.MVIT.CLS_EMBED_ON
        self.global_embed_on = cfg.MVIT.GLOBAL_EMBED_ON
        self.global_embed_num = 1
        self.sep_pos_embed = cfg.MVIT.SEP_POS_EMBED
        if cfg.MVIT.NORM == "layernorm":
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        else:
            raise NotImplementedError("Only supports layernorm.")
        self.num_classes = num_classes
        # Input embedding
        self.patch_embed = stem_helper.PatchEmbed(
            dim_in=in_chans,
            dim_out=embed_dim,
            kernel=cfg.MVIT.PATCH_KERNEL,
            stride=cfg.MVIT.PATCH_STRIDE,
            padding=cfg.MVIT.PATCH_PADDING,
            conv_2d=use_2d_patch,
        )
        if not use_2d_patch:
            # Prevent future leakage inside the initial temporal convolution.
            self.patch_embed.proj = convert_conv3d_to_causal(self.patch_embed.proj)
        self.input_dims = [temporal_size, spatial_size, spatial_size]
        assert self.input_dims[1] == self.input_dims[2]
        self.patch_dims = [self.input_dims[i] // self.patch_stride[i] for i in range(len(self.input_dims))]
        num_patches = math.prod(self.patch_dims)
        # Global embedding
        if self.global_embed_on:
            self.global_embed = stem_helper.CasGlobalEmbed(
                dim_in=embed_dim,
                dim_embed=embed_dim,
                conv_2d=use_2d_patch,
            )
            if not use_2d_patch:
                self._make_module_temporal_causal(self.global_embed)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if self.cls_embed_on:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # size (1, 1, 96)
            pos_embed_dim = num_patches + 1
        else:
            pos_embed_dim = num_patches

        if self.sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(torch.zeros(1, self.patch_dims[1] * self.patch_dims[2], embed_dim))
            self.pos_embed_temporal = nn.Parameter(torch.zeros(1, self.patch_dims[0], embed_dim))
            if self.cls_embed_on:
                self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, pos_embed_dim, embed_dim))

        if self.drop_rate > 0.0:
            self.pos_drop = nn.Dropout(p=self.drop_rate)

        dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
        for i in range(len(cfg.MVIT.DIM_MUL)):
            dim_mul[cfg.MVIT.DIM_MUL[i][0]] = cfg.MVIT.DIM_MUL[i][1]
        for i in range(len(cfg.MVIT.HEAD_MUL)):
            head_mul[cfg.MVIT.HEAD_MUL[i][0]] = cfg.MVIT.HEAD_MUL[i][1]

        pool_q = [[] for i in range(cfg.MVIT.DEPTH)]
        pool_kv = [[] for i in range(cfg.MVIT.DEPTH)]
        stride_q = [[] for i in range(cfg.MVIT.DEPTH)]
        stride_kv = [[] for i in range(cfg.MVIT.DEPTH)]
        self.stage_end = []  # save the number of blocks before downsampling

        for i in range(len(cfg.MVIT.POOL_Q_STRIDE)):
            stride_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_Q_STRIDE[i][1:]
            self.stage_end.append(cfg.MVIT.POOL_Q_STRIDE[i][0]-1)
            if cfg.MVIT.POOL_KVQ_KERNEL is not None:
                pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = [s + 1 if s > 1 else s for s in cfg.MVIT.POOL_Q_STRIDE[i][1:]]

        # If POOL_KV_STRIDE_ADAPTIVE is not None, initialize POOL_KV_STRIDE.
        if cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE is not None:
            _stride_kv = cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE
            cfg.MVIT.POOL_KV_STRIDE = []
            for i in range(cfg.MVIT.DEPTH):
                if len(stride_q[i]) > 0:  # if there's a stride in q
                    _stride_kv = [max(_stride_kv[d] // stride_q[i][d], 1) for d in range(len(_stride_kv))]
                cfg.MVIT.POOL_KV_STRIDE.append([i] + _stride_kv)

        # Refreeze config if it was originally frozen
        if was_frozen:
            cfg.freeze()

        for i in range(len(cfg.MVIT.POOL_KV_STRIDE)):
            stride_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = cfg.MVIT.POOL_KV_STRIDE[i][1:]
            if cfg.MVIT.POOL_KVQ_KERNEL is not None:
                pool_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = [s + 1 if s > 1 else s for s in cfg.MVIT.POOL_KV_STRIDE[i][1:]]
        self.norm_stem = norm_layer(embed_dim) if cfg.MVIT.NORM_STEM else None  # None

        self.blocks = nn.ModuleList()

        if cfg.MODEL.ACT_CHECKPOINT:  # False
            validate_checkpoint_wrapper_import(checkpoint_wrapper)

        for i in range(depth):
            num_heads = round_width(num_heads, head_mul[i])
            embed_dim = round_width(embed_dim, dim_mul[i], divisor=num_heads)
            dim_out = round_width(embed_dim, dim_mul[i + 1], divisor=round_width(num_heads, head_mul[i + 1]),)
            attention_block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=self.drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                kernel_q=pool_q[i] if len(pool_q) > i else [],
                kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                stride_q=stride_q[i] if len(stride_q) > i else [],
                stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                mode=mode,
                has_cls_embed=self.cls_embed_on,
                has_global_embed=self.global_embed_on,
                global_embed_num=self.global_embed_num,
                pool_first=pool_first,
                causal=True,
            )
            if cfg.MODEL.ACT_CHECKPOINT:
                attention_block = checkpoint_wrapper(attention_block)
            self.blocks.append(attention_block)

        self.global_fuse = GlobalLocalBlock(
            dim=self.blocks[15].dim_out,
            dim_out=self.blocks[15].dim_out,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=self.drop_rate,
            drop_path=0.4,
            norm_layer=norm_layer,
            kernel_q=[3, 3, 3],
            kernel_kv=[3, 3, 3],
            stride_q=[1, 1, 1],
            stride_kv=[1, 1, 1],
            mode=mode,
            has_cls_embed=self.cls_embed_on,
            has_global_embed=self.global_embed_on,
            global_embed_num=self.global_embed_num,
            pool_first=pool_first,
            causal=True,
        )

        embed_dim = dim_out
        self.norm = norm_layer(embed_dim * 2)

        # TransDecoder
        decode_dim_in = [768*2, 768, 384, 192]
        decode_dim_out = [768, 384, 192, 96]
        decode_num_heads = [8, 4, 4, 2]
        decode_kernel_q = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
        decode_kernel_kv = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
        decode_stride_q = [[1, 2, 2], [1, 2, 2], [1, 2, 2], [2, 1, 1]]  # upsample stride
        decode_stride_kv = [[1, 2, 2], [1, 4, 4], [1, 8, 8], [1, 16, 16]]
        for i in range(len(decode_dim_in)):
            decoder_block = MultiScaleDecoderBlock(
                dim=decode_dim_in[i],
                dim_out=decode_dim_out[i],
                num_heads=decode_num_heads[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=self.drop_rate,
                drop_path=0,
                norm_layer=norm_layer,
                kernel_q=decode_kernel_q[i] if len(decode_kernel_q) > i else [],
                kernel_kv=decode_kernel_kv[i] if len(decode_kernel_kv) > i else [],
                stride_q=decode_stride_q[i] if len(decode_stride_q) > i else [],
                stride_kv=decode_stride_kv[i] if len(decode_stride_kv) > i else [],
                mode=mode,
                has_cls_embed=self.cls_embed_on,
                has_global_embed=self.global_embed_on,
                global_embed_num=self.global_embed_num,
                pool_first=pool_first,
            )

            setattr(self, f'decode_block{i+1}', decoder_block)

        self.classifier = nn.Conv3d(96, 1, kernel_size=1)

        # Initialization
        if self.sep_pos_embed:
            trunc_normal_(self.pos_embed_spatial, std=0.02)
            trunc_normal_(self.pos_embed_temporal, std=0.02)
            if self.cls_embed_on:
                trunc_normal_(self.pos_embed_class, std=0.02)
        else:
            trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_embed_on:
            trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _make_module_temporal_causal(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Conv3d):
                setattr(module, name, convert_conv3d_to_causal(child))
            else:
                GLC_Gaze_Causal._make_module_temporal_causal(child)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        if self.cfg.MVIT.ZERO_DECAY_POS_CLS:
            if self.sep_pos_embed:
                if self.cls_embed_on:
                    return {"pos_embed_spatial", "pos_embed_temporal", "pos_embed_class", "cls_token"}
                else:
                    return {"pos_embed_spatial", "pos_embed_temporal", "pos_embed_class"}
            else:
                if self.cls_embed_on:
                    return {"pos_embed", "cls_token"}
                else:
                    return {"pos_embed"}
        else:
            return {}

    def forward(self, x, return_glc=False):
        inpt = x[0]  # size (B, 3, 8, 256, 256)
        x = self.patch_embed(inpt)  # size (B, 16384, 96)  16384 = 4*64*64

        T = self.cfg.DATA.NUM_FRAMES // self.patch_stride[0]  # 4
        H = self.cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride[1]  # 64
        W = self.cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride[2]  # 64
        B, N, C = x.shape  # B, 16384, 96

        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

        if self.global_embed_on:
            # share the first conv with patch embedding, followed by multi-conv (best now)
            global_tokens = x.view(B, T, H, W, C).permute(0, 4, 1, 2, 3)
            global_tokens = self.global_embed(global_tokens)

        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(1, self.patch_dims[0], 1) \
                        + torch.repeat_interleave(self.pos_embed_temporal, self.patch_dims[1] * self.patch_dims[2], dim=1)
            if self.cls_embed_on:
                pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
            x = x + pos_embed
        else:
            x = x + self.pos_embed

        if self.global_embed_on:
            x = torch.cat((global_tokens, x), dim=1)

        if self.drop_rate:
            x = self.pos_drop(x)

        if self.norm_stem:
            x = self.norm_stem(x)

        thw = [T, H, W]
        inter_feat = [[x, thw]]  # record features to be integrated in decoder
        for i, blk in enumerate(self.blocks):
            x, thw = blk(x, thw)

            if i in self.stage_end:
                inter_feat.append([x, thw])

            if i == 15:
                if not return_glc:
                    x_fuse, thw = self.global_fuse(x, thw)
                else:
                    x_fuse, thw, glc = self.global_fuse(x, thw, return_glc=True)
                x = torch.cat([x, x_fuse], dim=2)

        x = self.norm(x)  # x size [B, 256, 768]

        # Decoder (Transformer)
        feat, thw = self.decode_block1(x, thw)  # (B, 1024, 768)  1024 = 4*16*16
        feat = feat + inter_feat[-1][0]

        feat, thw = self.decode_block2(feat, thw)  # (B, 4096, 384)  4096 = 4*32*32
        feat = feat + inter_feat[-2][0]

        feat, thw = self.decode_block3(feat, thw)  # (B, 16384, 192)  16384 = 4*64*64
        feat = feat + inter_feat[-3][0]

        feat, thw = self.decode_block4(feat, thw)  # (B, 32768, 96)  16384 = 8*64*64
        if self.global_embed_on:
            feat = feat[:, self.global_embed_num:, :]
        feat = feat.reshape(feat.size(0), *thw, feat.size(2)).permute(0, 4, 1, 2, 3)
        en_feat, thw = inter_feat[-4]
        if self.global_embed_on:
            en_feat = en_feat[:, self.global_embed_num:, :]
        en_feat = en_feat.reshape(en_feat.size(0), *thw, en_feat.size(2)).permute(0, 4, 1, 2, 3)
        feat = feat + F.interpolate(en_feat, size=(thw[0]*2, thw[1], thw[2]), mode='trilinear')

        feat = self.classifier(feat)

        if not return_glc:
            return feat
        else:
            return [feat, glc]
