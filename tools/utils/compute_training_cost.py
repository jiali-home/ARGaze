#!/usr/bin/env python3
"""
Compute training/inference cost for NeurIPS submission.

This script calculates:
1. Model parameters
2. FLOPs per forward pass
3. Total training compute (FLOPs)
4. Total inference compute (FLOPs)

Based on the formula provided in NeurIPS checklist Section 6.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.abspath('.'))

from slowfast.config.defaults import get_cfg
from slowfast.models.build import MODEL_REGISTRY
from slowfast.utils.misc import params_count, get_model_stats
import slowfast.models.dinov3_T_S_window_attention


def format_number(num):
    """Format large numbers in scientific notation."""
    if num >= 1e15:
        return f"{num:.2e}"
    elif num >= 1e12:
        return f"{num/1e12:.2f}T"
    elif num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(int(num))


def create_config(dataset='egtea'):
    """Create configuration for model."""
    cfg = get_cfg()

    if dataset == 'egtea':
        cfg.merge_from_file("configs/Egtea/DINOV3_VITS16.yaml")
    elif dataset == 'ego4d':
        cfg.merge_from_file("configs/Ego4d/DINOV3_VITS16.yaml")
    elif dataset == 'holoassist':
        cfg.merge_from_file("configs/HoloAssist/DINOV3_VITS16.yaml")

    # Model settings
    cfg.MODEL.MODEL_NAME = "DINOv3_T_S_wind_attn"
    cfg.DATA.NUM_FRAMES = 8
    cfg.DATA.TRAIN_CROP_SIZE = 224
    cfg.DATA.TEST_CROP_SIZE = 224

    # Disable distributed
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

    # T+S Attention
    cfg.MODEL.NUM_TS_LAYERS = 3
    cfg.MODEL.TS_CAUSAL = False  # Offline mode
    cfg.MODEL.TS_WIN_T = None
    cfg.MODEL.NUM_QUERIES = 1
    cfg.MODEL.USE_QUERY_FOCUSER = True
    cfg.MODEL.MAX_TEMPORAL_LEN = 32

    return cfg


def compute_training_flops(flops_per_forward, num_samples, num_epochs, batch_size):
    """
    Compute total training FLOPs.

    Formula from NeurIPS checklist Section 6:
    Total FLOPs = FLOPs_per_forward × (num_samples / batch_size) × num_epochs × 3

    The factor of 3 accounts for:
    - 1x forward pass
    - 2x backward pass (roughly 2x forward)
    """
    iterations_per_epoch = num_samples / batch_size
    total_iterations = iterations_per_epoch * num_epochs
    # Each iteration: 1 forward + 2 backward ≈ 3× forward FLOPs
    total_flops = flops_per_forward * total_iterations * 3
    return total_flops


def compute_inference_flops(flops_per_forward, num_test_samples):
    """
    Compute total inference FLOPs.

    Inference only requires forward pass (no backward).
    """
    return flops_per_forward * num_test_samples


def main():
    print("="*80)
    print("Training/Inference Compute Cost Analysis")
    print("For NeurIPS Checklist Submission")
    print("="*80)
    print()

    # Dataset configurations
    datasets = {
        'egtea': {
            'train_samples': 15310,  # From train_ego4d_gaze_stride8.csv after expansion
            'test_samples': 200,  # Approximate test set size
            'epochs': 30,
            'batch_size': 32,
        },
        'ego4d': {
            'train_samples': 183721,  # From train_ego4d_gaze_stride8.csv
            'test_samples': 10000,  # Approximate
            'epochs': 20,
            'batch_size': 24,
        },
        'holoassist': {
            'train_samples': 50000,  # Approximate
            'test_samples': 5000,
            'epochs': 25,
            'batch_size': 16,
        }
    }

    # Analyze each dataset
    for dataset_name, dataset_info in datasets.items():
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name.upper()}")
        print(f"{'='*80}")

        try:
            # Build model
            cfg = create_config(dataset_name)
            print(f"Building model...", end=' ', flush=True)
            model = MODEL_REGISTRY.get("DINOv3_T_S_wind_attn")(cfg)
            model = model.cuda() if torch.cuda.is_available() else model.cpu()
            print("✓")

            # 1. Count parameters
            total_params = params_count(model)
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            print(f"\n1. Model Parameters:")
            print(f"   Total parameters: {format_number(total_params)} ({total_params:,})")
            print(f"   Trainable parameters: {format_number(trainable_params)} ({trainable_params:,})")

            # 2. Compute FLOPs per forward pass
            print(f"\n2. FLOPs per Forward Pass:")
            print(f"   Computing with input: (1, 3, {cfg.DATA.NUM_FRAMES}, {cfg.DATA.TRAIN_CROP_SIZE}, {cfg.DATA.TRAIN_CROP_SIZE})")

            try:
                flops_giga = get_model_stats(model, cfg, "flop", use_train_input=True)
                flops_per_forward = flops_giga * 1e9  # Convert GFLOPs to FLOPs
                print(f"   FLOPs per forward: {format_number(flops_per_forward)} ({flops_per_forward:.2e})")
            except Exception as e:
                print(f"   Warning: Could not compute FLOPs automatically: {e}")
                print(f"   Please compute manually using fvcore or similar tools")
                flops_per_forward = 0

            # 3. Training compute
            if flops_per_forward > 0:
                print(f"\n3. Training Configuration:")
                print(f"   Training set size: {dataset_info['train_samples']:,} samples")
                print(f"   Batch size: {dataset_info['batch_size']}")
                print(f"   Number of epochs: {dataset_info['epochs']}")

                iterations_per_epoch = dataset_info['train_samples'] / dataset_info['batch_size']
                total_iterations = iterations_per_epoch * dataset_info['epochs']
                print(f"   Total iterations: {total_iterations:,.0f}")

                training_flops = compute_training_flops(
                    flops_per_forward,
                    dataset_info['train_samples'],
                    dataset_info['epochs'],
                    dataset_info['batch_size']
                )

                print(f"\n   Total Training FLOPs:")
                print(f"   {format_number(training_flops)} ({training_flops:.2e})")
                print(f"   Formula: {format_number(flops_per_forward)} × {total_iterations:,.0f} iterations × 3 (forward+backward)")

                # 4. Inference compute
                print(f"\n4. Inference Configuration:")
                print(f"   Test set size: {dataset_info['test_samples']:,} samples")

                inference_flops = compute_inference_flops(
                    flops_per_forward,
                    dataset_info['test_samples']
                )

                print(f"\n   Total Inference FLOPs:")
                print(f"   {format_number(inference_flops)} ({inference_flops:.2e})")
                print(f"   FLOPs per 1000 instances: {format_number(flops_per_forward * 1000)} ({flops_per_forward * 1000:.2e})")

        except Exception as e:
            print(f"\n✗ Error analyzing {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}")
    print("Summary for NeurIPS Checklist:")
    print(f"{'='*80}")
    print("""
Fill in Section 2.3 (Compute for Reported Results):

1. Select Compute Metric: FLOPs (Floating-Point Operations)

2. Model size: [Use value from "Total parameters" above]

3. FLOPs per forward pass: [Use value from "FLOPs per forward" above]

4. Training set size: [Use from training configuration]

5. Number of epochs: [Use from training configuration]

6. Batch size: [Use from training configuration]

7. Total training FLOPs: [Use from "Total Training FLOPs" above]

8. Test set size: [Use from inference configuration]

9. FLOPs per 1000 inference instances: [Use calculated value above]

10. Total inference FLOPs: [Use from "Total Inference FLOPs" above]

11. FLOPs calculation tool: fvcore (https://github.com/facebookresearch/fvcore)
    """)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    os.environ['TRANSFORMERS_OFFLINE'] = '0'

    main()
