#!/usr/bin/env python3
"""
Count model parameters for all models in test.sh
This script instantiates each model and counts total/trainable parameters.
"""

import os
import sys
import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from slowfast.config.defaults import get_cfg
from slowfast.models.build import MODEL_REGISTRY

# Import all model modules to register them
# import slowfast.models.dinov3_glc_decoder  # This will register DINOv3 GLC models
# import slowfast.models.dinov3_model_builder  # Other DINOv3 models
# import slowfast.models.dinov3_enc_dec  # DINOv3 Encoder-Decoder models
# import slowfast.models.dinov3_query_decoder  # DINOv3 Query Decoder models
# import slowfast.models.dinov3_query_focuser  # DINOv3 Query Focuser models
# import slowfast.models.dinov3_global_query_focuser  # DINOv3 Global Query Focuser models
# import slowfast.models.dinov3_T_S_attention  # DINOv3 Divided Space-Time Attention models
# # import slowfast.models.dinov3_ARgaze  # DINOv3 Autoregressive Gaze models
# import slowfast.models.custom_video_model_builder  # GLC_Gaze model
import slowfast.model.dinov3_glc_decoder 

def count_parameters(model):
    """Count total and trainable parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params
    }


def format_number(num):
    """Format large numbers with K/M/B suffixes."""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)


def create_model_config(model_name, config_file="configs/Egtea/DINOV3_VITS16.yaml", temporal_mode=None,
                       query_fusion=None, use_query_attention=False, ts_causal=False, ts_win_t=None, num_ts_layers=3,
                       use_confidence=False, num_gaze_queries=1):
    """Create config for a specific model."""
    cfg = get_cfg()
    cfg.merge_from_file(config_file)

    # Model-specific settings
    cfg.MODEL.MODEL_NAME = model_name
    cfg.MODEL.NUM_CLASSES = 400
    cfg.DATA.NUM_FRAMES = 8
    cfg.DATA.TRAIN_CROP_SIZE = 224
    cfg.DATA.TEST_CROP_SIZE = 224
    cfg.MODEL.ARCH = "mvit"

    # IMPORTANT: Disable distributed training for parameter counting
    cfg.NUM_GPUS = 1
    cfg.NUM_SHARDS = 1

    # DINOv3-specific settings
    if "DINOv3" in model_name:
        cfg.MODEL.DINOV3_MODEL_NAME = "facebook/dinov2-small-imagenet1k-1-layer"
        cfg.MODEL.FREEZE_ENCODER = True
        cfg.MODEL.UNFREEZE_LAST_K_LAYERS = 0
        cfg.MODEL.USE_MULTISCALE_FEATURES = True
        cfg.MODEL.MULTISCALE_LAYERS = [-4, -3, -2, -1]
        cfg.MODEL.HEATMAP_SIZE = 64
        cfg.MODEL.DROPOUT_RATE = 0.1
        cfg.MODEL.TEMPORAL_MODE = temporal_mode if temporal_mode else "none"

        # GRU settings (for FrameWise_GRU model)
        if "GRU" in model_name:
            cfg.MODEL.GRU_NUM_LAYERS = 1
            cfg.MODEL.GRU_BIDIRECTIONAL = True

        # Query Decoder settings
        if "QueryDecoder" in model_name:
            cfg.MODEL.USE_QUERY = True
            cfg.MODEL.NUM_QUERIES = 1
            cfg.MODEL.QUERY_DIM = 384
            cfg.MODEL.QUERY_FUSION = query_fusion if query_fusion else "dot"
            cfg.MODEL.USE_QUERY_ATTENTION = use_query_attention
            cfg.MODEL.GRU_NUM_LAYERS = 1
            cfg.MODEL.GRU_BIDIRECTIONAL = True

        # Query Focuser settings (both static and CLS-initialized)
        if "QueryFocuser" in model_name or "GlobalQueryFocuser" in model_name:
            cfg.MODEL.NUM_QUERIES = 1
            cfg.MODEL.QUERY_TEMPERATURE = 1.0
            cfg.MODEL.GRU_NUM_LAYERS = 1
            cfg.MODEL.GRU_BIDIRECTIONAL = False

        # Divided Space-Time Attention settings
        if "T_S_Attention" in model_name:
            cfg.MODEL.NUM_TS_LAYERS = num_ts_layers
            cfg.MODEL.TS_CAUSAL = ts_causal
            cfg.MODEL.TS_WIN_T = ts_win_t
            cfg.MODEL.NUM_QUERIES = 1
            cfg.MODEL.QUERY_TEMPERATURE = 1.0
            cfg.MODEL.MAX_TEMPORAL_LEN = 32

        # Autoregressive Gaze settings (DINOv3_ARgaze)
        if "ARgaze" in model_name:
            cfg.MODEL.NUM_GAZE_QUERIES = num_gaze_queries
            cfg.MODEL.USE_CONFIDENCE_HEAD = use_confidence
            cfg.TRAIN.SS_PROB_START = 0.0
            cfg.TRAIN.SS_PROB_END = 0.3
            cfg.TRAIN.SS_PROB_RAMP_EPOCHS = 10

    return cfg


def analyze_model(model_name, config_file="configs/Egtea/DINOV3_VITS16.yaml", temporal_mode=None,
                 query_fusion=None, use_query_attention=False, ts_causal=False, ts_win_t=None, num_ts_layers=3,
                 use_confidence=False, num_gaze_queries=1):
    """Analyze a single model's parameters."""
    model_label = model_name
    if temporal_mode:
        model_label = f"{model_name} (temporal_mode={temporal_mode})"
    if query_fusion:
        model_label = f"{model_name} (fusion={query_fusion})"
    if use_query_attention:
        model_label = f"{model_label} + Attention"

    try:
        # Create config
        cfg = create_model_config(model_name, config_file, temporal_mode, query_fusion, use_query_attention,
                                 ts_causal, ts_win_t, num_ts_layers, use_confidence, num_gaze_queries)

        # Build model directly without DistributedDataParallel wrapper
        print(f"Building {model_label}...", end=' ', flush=True)
        model = MODEL_REGISTRY.get(model_name)(cfg)

        # Move to CPU for parameter counting (avoid GPU memory issues)
        model = model.cpu()

        # Count parameters
        params = count_parameters(model)
        print("✓")

        return params

    except Exception as e:
        print(f"✗")
        print(f"Error analyzing {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function to analyze all models."""
    print("="*80)
    print("Model Parameter Analysis")
    print("="*80)
    print()

    # List of models from test.sh and train.sh
    models = [
        {
            'name': 'DINOv3GazeModel_V0_DecoderOnly',
            'config': 'configs/Egtea/DINOV3_VITS16.yaml',
            'temporal_mode': 'none',
            'description': 'DecoderOnly (temporal=none)'
        },
        {
            'name': 'DINOv3GazeModel_V0_DecoderOnly',
            'config': 'configs/Egtea/DINOV3_VITS16.yaml',
            'temporal_mode': 'causal',
            'description': 'DecoderOnly (temporal=causal)'
        },
        {
            'name': 'DINOv3GazeModel_V0_DecoderOnly',
            'config': 'configs/Egtea/DINOV3_VITS16.yaml',
            'temporal_mode': 'offline',
            'description': 'DecoderOnly (temporal=offline)'
        },
        {
            'name': 'DINOv3GazeModel_V0_GLCDecoderOnly',
            'config': 'configs/Egtea/DINOV3_VITS16.yaml',
            'temporal_mode': 'none',
            'description': 'GLCDecoderOnly (no skip)'
        },
        {
            'name': 'DINOv3GazeModel_V0_GLCDecoderOnly_SkipCon',
            'config': 'configs/Egtea/DINOV3_VITS16.yaml',
            'temporal_mode': 'none',
            'description': 'GLCDecoderOnly_SkipCon (with skip)'
        },
        {
            'name': 'DINOv3_FrameWise_GRU',
            'config': 'configs/Egtea/DINOV3_VITS16.yaml',
            'temporal_mode': None,
            'description': 'FrameWise_GRU'
        },
        {
            'name': 'DINOv3_QueryDecoder',
            'config': 'configs/Egtea/DINOV3_VITS16.yaml',
            'temporal_mode': None,
            'query_fusion': 'dot',
            'use_query_attention': False,
            'description': 'QueryDecoder (dot only)'
        },
        {
            'name': 'DINOv3_QueryDecoder',
            'config': 'configs/Egtea/DINOV3_VITS16.yaml',
            'temporal_mode': None,
            'query_fusion': 'dot+conv',
            'use_query_attention': False,
            'description': 'QueryDecoder (dot+conv)'
        },
        {
            'name': 'DINOv3_QueryDecoder',
            'config': 'configs/Egtea/DINOV3_VITS16.yaml',
            'temporal_mode': None,
            'query_fusion': 'dot+conv',
            'use_query_attention': True,
            'description': 'QueryDecoder (dot+conv+attn)'
        },
        {
            'name': 'DINOv3_QueryFocuser',
            'config': 'configs/Egtea/DINOV3_VITS16.yaml',
            'temporal_mode': None,
            'description': 'QueryFocuser (static query)'
        },
        {
            'name': 'DINOv3_GlobalQueryFocuser',
            'config': 'configs/Egtea/DINOV3_VITS16.yaml',
            'temporal_mode': None,
            'description': 'GlobalQueryFocuser (CLS-init)'
        },
        {
            'name': 'DINOv3_T_S_Attention',
            'config': 'configs/Egtea/DINOV3_VITS16.yaml',
            'temporal_mode': None,
            'ts_causal': False,
            'ts_win_t': None,
            'num_ts_layers': 3,
            'description': 'T+S Attention (offline, 3 layers)'
        },
        {
            'name': 'DINOv3_T_S_Attention',
            'config': 'configs/Egtea/DINOV3_VITS16.yaml',
            'temporal_mode': None,
            'ts_causal': True,
            'ts_win_t': None,
            'num_ts_layers': 3,
            'description': 'T+S Attention (causal, 3 layers)'
        },
        {
            'name': 'DINOv3_T_S_Attention',
            'config': 'configs/Egtea/DINOV3_VITS16.yaml',
            'temporal_mode': None,
            'ts_causal': True,
            'ts_win_t': 8,
            'num_ts_layers': 3,
            'description': 'T+S Attention (causal+windowed, 3 layers)'
        },
        {
            'name': 'DINOv3_ARgaze',
            'config': 'configs/Egtea/DINOV3_VITS16.yaml',
            'temporal_mode': None,
            'description': 'ARgaze (Autoregressive, 1 query, no confidence)'
        },
        {
            'name': 'DINOv3_ARgaze',
            'config': 'configs/Egtea/DINOV3_VITS16.yaml',
            'temporal_mode': None,
            'use_confidence': True,
            'description': 'ARgaze (Autoregressive, 1 query, with confidence)'
        },
        {
            'name': 'DINOv3_ARgaze',
            'config': 'configs/Egtea/DINOV3_VITS16.yaml',
            'temporal_mode': None,
            'num_gaze_queries': 3,
            'description': 'ARgaze (Autoregressive, 3 queries, no confidence)'
        },
        {
            'name': 'GLC_Gaze',
            'config': 'configs/Egtea/MVIT_B_16x4_CONV.yaml',
            'temporal_mode': None,
            'description': 'GLC_Gaze (MViT baseline)'
        },
    ]

    results = {}

    for model_info in models:
        params = analyze_model(
            model_info['name'],
            model_info['config'],
            model_info.get('temporal_mode'),
            model_info.get('query_fusion'),
            model_info.get('use_query_attention', False),
            model_info.get('ts_causal', False),
            model_info.get('ts_win_t'),
            model_info.get('num_ts_layers', 3),
            model_info.get('use_confidence', False),
            model_info.get('num_gaze_queries', 1)
        )
        if params:
            # Create unique key for results
            result_key = model_info['description']
            results[result_key] = params

    # Summary table
    print(f"\n{'='*100}")
    print("PARAMETER COMPARISON TABLE")
    print(f"{'='*100}")
    print(f"{'Model':<45} {'Total':<15} {'Trainable':<15} {'Frozen':<15} {'Train%':<10}")
    print(f"{'-'*100}")

    for model_desc, params in results.items():
        train_pct = params['trainable']/params['total']*100 if params['total'] > 0 else 0
        print(f"{model_desc:<45} "
              f"{format_number(params['total']):<15} "
              f"{format_number(params['trainable']):<15} "
              f"{format_number(params['frozen']):<15} "
              f"{train_pct:>6.2f}%")

    print(f"{'='*100}\n")


if __name__ == "__main__":
    # Set environment variables to avoid downloading models
    os.environ['TRANSFORMERS_OFFLINE'] = '0'

    # Suppress HuggingFace warnings
    import warnings
    warnings.filterwarnings('ignore')

    main()
