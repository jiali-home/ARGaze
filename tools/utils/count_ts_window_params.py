#!/usr/bin/env python3
"""Count parameters for DINOv3_T_S_wind_attn model."""

import os
import sys
import torch

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from slowfast.config.defaults import get_cfg
from slowfast.models.build import MODEL_REGISTRY
import slowfast.models.dinov3_T_S_window_attention

def count_parameters(model):
    """Count total and trainable parameters."""
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

def create_model_config(ts_causal=False, ts_win_t=None, num_ts_layers=3):
    """Create config for DINOv3_T_S_wind_attn model."""
    cfg = get_cfg()
    cfg.merge_from_file("configs/Egtea/DINOV3_VITS16.yaml")

    # Model settings
    cfg.MODEL.MODEL_NAME = "DINOv3_T_S_wind_attn"
    cfg.MODEL.NUM_CLASSES = 400
    cfg.DATA.NUM_FRAMES = 8
    cfg.DATA.TRAIN_CROP_SIZE = 224
    cfg.DATA.TEST_CROP_SIZE = 224

    # Disable distributed training
    cfg.NUM_GPUS = 1
    cfg.NUM_SHARDS = 1

    # DINOv3 settings
    cfg.MODEL.DINOV3_MODEL_NAME = "facebook/dinov2-small"
    cfg.MODEL.FREEZE_ENCODER = True
    cfg.MODEL.UNFREEZE_LAST_K_LAYERS = 0
    cfg.MODEL.USE_MULTISCALE_FEATURES = True
    cfg.MODEL.MULTISCALE_LAYERS = [-4, -3, -2, -1]
    cfg.MODEL.HEATMAP_SIZE = 64
    cfg.MODEL.DROPOUT_RATE = 0.1

    # T+S Attention settings
    cfg.MODEL.NUM_TS_LAYERS = num_ts_layers
    cfg.MODEL.TS_CAUSAL = ts_causal
    cfg.MODEL.TS_WIN_T = ts_win_t
    cfg.MODEL.NUM_QUERIES = 1
    cfg.MODEL.QUERY_TEMPERATURE = 1.0
    cfg.MODEL.MAX_TEMPORAL_LEN = 32
    cfg.MODEL.USE_QUERY_FOCUSER = True

    return cfg

def main():
    """Main function to analyze DINOv3_T_S_wind_attn model."""
    print("="*80)
    print("DINOv3_T_S_wind_attn Model Parameter Analysis")
    print("="*80)
    print()

    # Test different configurations
    configs = [
        {
            'ts_causal': False,
            'ts_win_t': None,
            'num_ts_layers': 3,
            'description': 'T+S Attention (offline, 3 layers, full temporal)'
        },
        {
            'ts_causal': True,
            'ts_win_t': None,
            'num_ts_layers': 3,
            'description': 'T+S Attention (causal, 3 layers, all past)'
        },
        {
            'ts_causal': True,
            'ts_win_t': 8,
            'num_ts_layers': 3,
            'description': 'T+S Attention (causal, 3 layers, windowed)'
        },
    ]

    results = {}

    for config in configs:
        print(f"Building {config['description']}...", end=' ', flush=True)
        try:
            cfg = create_model_config(
                ts_causal=config['ts_causal'],
                ts_win_t=config['ts_win_t'],
                num_ts_layers=config['num_ts_layers']
            )

            # Build model
            model = MODEL_REGISTRY.get("DINOv3_T_S_wind_attn")(cfg)
            model = model.cpu()

            # Count parameters
            params = count_parameters(model)
            results[config['description']] = params
            print("✓")

        except Exception as e:
            print("✗")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()

    # Print summary table
    print(f"\n{'='*100}")
    print("PARAMETER SUMMARY")
    print(f"{'='*100}")
    print(f"{'Model Configuration':<50} {'Total':<15} {'Trainable':<15} {'Frozen':<15} {'Train%':<10}")
    print(f"{'-'*100}")

    for desc, params in results.items():
        train_pct = params['trainable']/params['total']*100 if params['total'] > 0 else 0
        print(f"{desc:<50} "
              f"{format_number(params['total']):<15} "
              f"{format_number(params['trainable']):<15} "
              f"{format_number(params['frozen']):<15} "
              f"{train_pct:>6.2f}%")

    print(f"{'='*100}\n")

    # Print detailed breakdown for the first config
    if results:
        first_config = list(configs)[0]
        print("\nDetailed Component Breakdown (Offline mode):")
        print("-" * 60)

        cfg = create_model_config(
            ts_causal=first_config['ts_causal'],
            ts_win_t=first_config['ts_win_t'],
            num_ts_layers=first_config['num_ts_layers']
        )
        model = MODEL_REGISTRY.get("DINOv3_T_S_wind_attn")(cfg)

        # Count parameters by component
        components = {
            'Encoder (DINOv3-S)': model.encoder,
            'Multi-scale Projection': model.multiscale_proj,
            'Multiscale to Hidden': model.multiscale_to_hidden,
            'Temporal Pos Encoding': [model.temporal_pos_encoding],
            'T+S Blocks': model.ts_blocks,
            'Query Token & Projection': [model.query_token_prior, model.cls_to_query_proj] if hasattr(model, 'query_token_prior') else [],
            'Conv Decoder': nn.ModuleList([
                model.pixel_decoder_conv1,
                model.pixel_decoder_conv2,
                model.pixel_final_proj
            ]) if hasattr(model, 'pixel_decoder_conv1') else []
        }

        import torch.nn as nn
        for name, component in components.items():
            if isinstance(component, list):
                params = sum(p.numel() for item in component for p in item.parameters() if isinstance(item, (nn.Module, nn.Parameter)) or hasattr(item, 'numel'))
                trainable = sum(p.numel() for item in component for p in (item.parameters() if hasattr(item, 'parameters') else [item]) if (hasattr(p, 'requires_grad') and p.requires_grad) or (isinstance(item, nn.Parameter) and item.requires_grad))
            else:
                params = sum(p.numel() for p in component.parameters())
                trainable = sum(p.numel() for p in component.parameters() if p.requires_grad)

            print(f"{name:<30}: {format_number(params):>10} ({format_number(trainable):>10} trainable)")

if __name__ == "__main__":
    # Suppress warnings
    import warnings
    warnings.filterwarnings('ignore')

    # Set offline mode to avoid downloading
    os.environ['TRANSFORMERS_OFFLINE'] = '0'

    main()
