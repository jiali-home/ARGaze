#!/usr/bin/env python3
"""
Visualize error cases at the video clip level (temporal context).
Shows 8 consecutive frames with GT and predicted gaze points.
Highlights high-error frames with yellow borders.
"""
import numpy as np
import os
import pickle
import torch
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

import sys
sys.path.append("/mnt/data1/jiali/GLC")

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.utils import frame_softmax
from scipy.spatial.distance import cdist

logger = logging.get_logger(__name__)


# ============================================================================
# Utility Functions
# ============================================================================

def load_hand_mask(hand_mask_dir, video_name, frame_idx):
    """Load hand mask for a specific frame."""
    if not hand_mask_dir:
        return False, None

    mask_path = os.path.join(hand_mask_dir, video_name, f"{int(frame_idx):06d}.png")

    if not os.path.exists(mask_path):
        return False, None

    try:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return False, None

        mask = (mask > 0).astype(np.uint8)
        has_hand = np.any(mask > 0)
        return has_hand, mask
    except Exception as e:
        return False, None


def parse_clip_start_frame(video_path):
    """
    Parse the global start frame index encoded in the cropped clip filename.

    Filenames follow ...-F<start>-F<end>.mp4. We only need <start>.
    """
    clip_name = os.path.basename(video_path)
    parts = clip_name[:-4].split("-")
    if len(parts) < 2:
        return None
    start_token = parts[-2]
    if not start_token.startswith("F"):
        return None
    try:
        return int(start_token[1:])
    except Exception:
        return None


def load_frame_image(frames_dir, video_name, frame_idx, frame_ext="jpg",
                     fallback_video_path=None, fallback_clip_start=None,
                     warn_missing=True):
    """
    Load the original frame image.

    If the pre-extracted frame is missing, optionally fall back to decoding
    a single frame from the cropped clip video.
    """
    possible_paths = [
        os.path.join(frames_dir, video_name, f"{int(frame_idx):06d}.{frame_ext}"),
        os.path.join(frames_dir, video_name, f"frame_{int(frame_idx):010d}.{frame_ext}"),
        os.path.join(frames_dir, video_name, f"{int(frame_idx):010d}.{frame_ext}"),
        os.path.join(frames_dir, video_name, f"{int(frame_idx):05d}.{frame_ext}"),
        os.path.join(frames_dir, video_name, f"{int(frame_idx):08d}.{frame_ext}"),
        os.path.join(frames_dir, video_name, f"{int(frame_idx)}.{frame_ext}"),
        os.path.join(frames_dir, video_name, f"img_{int(frame_idx):06d}.{frame_ext}"),
        os.path.join(frames_dir, video_name, f"{int(frame_idx)+1:06d}.{frame_ext}"),
    ]

    for img_path in possible_paths:
        if os.path.exists(img_path):
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception:
                pass

    # Fallback: decode from the cropped clip video if available
    if fallback_video_path and fallback_clip_start is not None:
        rel_idx = max(int(frame_idx) - int(fallback_clip_start), 0)
        try:
            cap = cv2.VideoCapture(fallback_video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, rel_idx)
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception:
            pass

    if warn_missing:
        logger.warning(f"Could not load frame: {video_name}/{frame_idx}, tried {len(possible_paths)} paths")
        logger.warning(f"Possible paths: {possible_paths}")
    return None


def compute_distance_to_hand_pixels(point, hand_mask):
    """Compute minimum distance from point to any hand mask pixel."""
    H, W = hand_mask.shape
    hand_coords = np.argwhere(hand_mask > 0)

    if hand_coords.shape[0] == 0:
        return np.inf

    hand_coords_norm = hand_coords.astype(np.float32)
    hand_coords_norm[:, 0] /= H
    hand_coords_norm[:, 1] /= W
    hand_coords_xy = hand_coords_norm[:, [1, 0]]

    point_xy = np.array([[point[0], point[1]]])
    distances = cdist(point_xy, hand_coords_xy, metric='euclidean')

    return distances.min()


def get_predicted_gaze_point(pred_heatmap):
    """Extract predicted gaze point from heatmap (argmax)."""
    H, W = pred_heatmap.shape
    flat_idx = pred_heatmap.argmax()
    row = flat_idx // W
    col = flat_idx % W

    x = col / W
    y = row / H

    return x, y


# ============================================================================
# Clip-level Visualization
# ============================================================================

def visualize_error_clip(frames, hand_masks, gt_points, pred_points,
                         errors, frame_indices, video_name,
                         error_threshold=0.3, clip_idx=0, case_type=""):
    """
    Visualize a clip of 8 frames with temporal context.

    Args:
        frames: list of T images (H, W, 3) or None
        hand_masks: list of T hand masks (H, W)
        gt_points: list of T tuples (x, y, is_valid)
        pred_points: list of T tuples (x, y)
        errors: list of T error values
        frame_indices: list of T frame indices
        video_name: video name
        error_threshold: threshold for highlighting high-error frames
        clip_idx: clip index for naming
        case_type: type of error case

    Returns:
        fig: matplotlib figure
    """
    T = len(errors)

    # Create figure with T subplots in 2 rows
    n_rows = 2
    n_cols = T // n_rows

    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))

    for t in range(T):
        ax = fig.add_subplot(n_rows, n_cols, t + 1)

        # Get frame or create placeholder
        frame_missing = False
        if frames[t] is not None:
            img = frames[t]
        else:
            # Frame failed to load - create gray placeholder with text
            frame_missing = True
            if hand_masks[t] is not None:
                H, W = hand_masks[t].shape
            else:
                H, W = 224, 224
            # Create gray background instead of black
            img = np.ones((H, W, 3), dtype=np.uint8) * 128

        H, W = img.shape[:2]

        # Display image
        ax.imshow(img)

        # If frame is missing, add text overlay
        if frame_missing:
            ax.text(W/2, H/2, 'Frame\nNot Found',
                   ha='center', va='center', fontsize=14,
                   color='white', weight='bold',
                   bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))

        # Overlay hand mask
        if hand_masks[t] is not None:
            hand_mask_resized = cv2.resize(hand_masks[t], (W, H), interpolation=cv2.INTER_NEAREST)
            hand_overlay = np.zeros((H, W, 4))
            hand_overlay[:, :, 0] = hand_mask_resized  # Red channel
            hand_overlay[:, :, 3] = hand_mask_resized * 0.3  # Alpha
            ax.imshow(hand_overlay)

        # Plot GT point (green circle) - only if valid
        gt_x, gt_y, is_valid = gt_points[t]
        if is_valid:
            gt_x_px = gt_x * W
            gt_y_px = gt_y * H
            ax.plot(gt_x_px, gt_y_px, 'o', color='lime', markersize=10,
                   markeredgecolor='white', markeredgewidth=2, label='GT')
        else:
            # Add "No GT" text for invalid frames
            ax.text(W/2, H - 20, 'No GT',
                   ha='center', va='top', fontsize=10,
                   color='white', weight='bold',
                   bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))

        # Plot predicted point (red X)
        pred_x, pred_y = pred_points[t]
        pred_x_px = pred_x * W
        pred_y_px = pred_y * H
        ax.plot(pred_x_px, pred_y_px, 'x', color='red', markersize=12,
               markeredgewidth=3, label='Pred')

        # Draw arrow from pred to GT if valid
        if is_valid:
            ax.annotate('', xy=(gt_x_px, gt_y_px), xytext=(pred_x_px, pred_y_px),
                       arrowprops=dict(arrowstyle='->', color='yellow', lw=2, alpha=0.8))

        # Title with frame info
        error_val = errors[t]

        # Handle NaN errors (invalid GT)
        if np.isnan(error_val):
            title = f"Frame {int(frame_indices[t])}\nNo GT"
            title_color = 'gray'
        else:
            title = f"Frame {int(frame_indices[t])}\nError: {error_val:.3f}"

            # Highlight high-error frames with yellow border
            if error_val > error_threshold:
                title_color = 'red'
                # Add yellow border
                rect = patches.Rectangle((0, 0), W-1, H-1, linewidth=4,
                                        edgecolor='yellow', facecolor='none')
                ax.add_patch(rect)
            else:
                title_color = 'black'

        ax.set_title(title, fontsize=10, color=title_color, fontweight='bold')
        ax.axis('off')

    # Overall title - filter NaN values
    valid_errors = [e for e in errors if not np.isnan(e)]
    if len(valid_errors) > 0:
        max_error = max(valid_errors)
        mean_error = np.mean(valid_errors)
        high_error_count = sum(1 for e in valid_errors if e > error_threshold)
        num_valid = len(valid_errors)
    else:
        max_error = 0.0
        mean_error = 0.0
        high_error_count = 0
        num_valid = 0

    suptitle = (f"{case_type} - Clip #{clip_idx}\n"
               f"Video: {video_name} | "
               f"Max Error: {max_error:.3f} | Mean Error: {mean_error:.3f} | "
               f"High-Error Frames: {high_error_count}/{num_valid} valid")

    plt.suptitle(suptitle, fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig


# ============================================================================
# Error Clip Collector
# ============================================================================

class ErrorClipCollector:
    """Collects error clips (temporal sequences) for visualization."""

    def __init__(self, error_threshold=0.3, max_clips=100):
        self.error_threshold = error_threshold
        self.max_clips = max_clips

        self.clips = {
            'single_spike': [],         # One high-error frame in the clip
            'multiple_errors': [],      # Multiple high-error frames (temporal consistency issue)
            'gradual_degradation': [],  # Error gradually increases over time
            'all_errors': []            # All clips with any high error (for sorting later)
        }

    def add_clip(self, frames, hand_masks, gt_points, pred_points,
                 errors, frame_indices, video_name):
        """
        Add a clip if it contains interesting temporal error patterns.

        Args:
            frames: list of T images
            hand_masks: list of T masks
            gt_points: list of T (x, y, is_valid) tuples
            pred_points: list of T (x, y) tuples
            errors: list of T error values
            frame_indices: list of T frame indices
            video_name: video name
        """
        T = len(errors)

        # Filter out NaN values (invalid GT frames)
        valid_errors = [e for e in errors if not np.isnan(e)]

        # Skip if no valid errors
        if len(valid_errors) == 0:
            return

        # Compute clip-level statistics (only on valid frames)
        max_error = max(valid_errors)
        mean_error = np.mean(valid_errors)
        high_error_count = sum(1 for e in valid_errors if e > self.error_threshold)

        # Skip if no significant errors
        if max_error < self.error_threshold:
            return

        clip_data = {
            'frames': frames,
            'hand_masks': hand_masks,
            'gt_points': gt_points,
            'pred_points': pred_points,
            'errors': errors,
            'frame_indices': frame_indices,
            'video_name': video_name,
            'max_error': max_error,
            'mean_error': mean_error,
            'high_error_count': high_error_count
        }

        # Add to all_errors for later sorting
        if len(self.clips['all_errors']) < self.max_clips:
            self.clips['all_errors'].append(clip_data)

        # Categorize by temporal error pattern
        # Type 1: Single spike - only 1 high-error frame
        if high_error_count == 1:
            if len(self.clips['single_spike']) < self.max_clips // 3:
                self.clips['single_spike'].append(clip_data)

        # Type 2: Multiple errors - 2 or more high-error frames
        elif high_error_count >= 2:
            if len(self.clips['multiple_errors']) < self.max_clips // 3:
                self.clips['multiple_errors'].append(clip_data)

        # Type 3: Gradual degradation - error increases over time
        # Check if errors show an increasing trend
        first_half_mean = np.mean(errors[:T//2])
        second_half_mean = np.mean(errors[T//2:])
        if second_half_mean > first_half_mean * 1.5:  # 50% increase
            if len(self.clips['gradual_degradation']) < self.max_clips // 3:
                self.clips['gradual_degradation'].append(clip_data)

    def visualize_all_clips(self, output_dir, top_n=50):
        """
        Generate visualizations for all collected clips.

        Args:
            output_dir: output directory
            top_n: number of top error clips to visualize from all_errors
        """
        os.makedirs(output_dir, exist_ok=True)

        type_names = {
            'single_spike': 'Single Error Spike',
            'multiple_errors': 'Multiple Error Frames',
            'gradual_degradation': 'Gradual Error Increase',
            'all_errors': 'Top Error Clips'
        }

        for clip_type, display_name in type_names.items():
            clips = self.clips[clip_type]

            if len(clips) == 0:
                logger.info(f"No clips found for {display_name}")
                continue

            # For all_errors, only take top N by max error
            if clip_type == 'all_errors':
                clips_sorted = sorted(clips, key=lambda x: x['max_error'], reverse=True)[:top_n]
                logger.info(f"Visualizing top {len(clips_sorted)} of {len(clips)} {display_name}...")
            else:
                # For other types, sort and take all
                clips_sorted = sorted(clips, key=lambda x: x['max_error'], reverse=True)
                logger.info(f"Visualizing {len(clips_sorted)} {display_name} clips...")

            # Create subdirectory
            type_dir = os.path.join(output_dir, clip_type)
            os.makedirs(type_dir, exist_ok=True)

            for idx, clip in enumerate(clips_sorted):
                fig = visualize_error_clip(
                    clip['frames'],
                    clip['hand_masks'],
                    clip['gt_points'],
                    clip['pred_points'],
                    clip['errors'],
                    clip['frame_indices'],
                    clip['video_name'],
                    error_threshold=self.error_threshold,
                    clip_idx=idx + 1,
                    case_type=display_name
                )

                save_path = os.path.join(
                    type_dir,
                    f"{clip_type}_{idx+1:02d}_maxerr{clip['max_error']:.3f}_"
                    f"meanerr{clip['mean_error']:.3f}_{clip['video_name']}.png"
                )
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)

            logger.info(f"Saved {len(clips_sorted)} clips to {type_dir}")


# ============================================================================
# Main Evaluation Loop
# ============================================================================

@torch.no_grad()
def collect_error_clips(test_loader, model, hand_mask_dir, frames_dir, frame_ext, cfg, use_hand_masks=True):
    """Collect error clips for visualization."""
    model.eval()

    collector = ErrorClipCollector(error_threshold=0.3, max_clips=500)

    show_progress = du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS)
    test_iterator = tqdm(test_loader, desc="Collecting error clips",
                        disable=not show_progress, ncols=100)

    for cur_iter, (inputs, labels, labels_hm, video_idx, meta) in enumerate(test_iterator):
        # Move to GPU
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
            labels_hm = labels_hm.cuda()

        # Forward pass
        preds = model(inputs)
        preds = frame_softmax(preds, temperature=2)

        # Move to CPU
        if cfg.NUM_GPUS:
            preds = preds.cpu()
            labels = labels.cpu()
            labels_hm = labels_hm.cpu()

        # Handle shape
        if preds.dim() == 5 and preds.size(1) == 1:
            preds = preds.squeeze(1)

        if labels.shape[-1] > 3:
            labels = labels[..., :3]

        B, T = preds.shape[:2]
        H, W = preds.shape[-2:]

        # Process each clip (batch element)
        for b in range(B):
            video_path = meta['path'][b]
            video_name = video_path.split('/')[-2]
            clip_start = parse_clip_start_frame(video_path)

            if len(meta['index'].shape) == 1:
                frame_indices = meta['index'].reshape(B, 1)
            else:
                frame_indices = meta['index']

            # Collect data for all T frames in this clip
            frames = []
            hand_masks = []
            gt_points = []
            pred_points = []
            errors = []
            indices = []

            for t in range(T):
                frame_idx = frame_indices[b, t].item()
                indices.append(frame_idx)

                # Debug: Print path info for first clip
                if cur_iter == 0 and b == 0 and t == 0:
                    logger.info(f"Debug - First frame info:")
                    logger.info(f"  Video: {video_name}")
                    logger.info(f"  Frame index: {frame_idx}")
                    logger.info(f"  Frames dir: {frames_dir}")
                    test_path = os.path.join(frames_dir, video_name, f"{int(frame_idx):06d}.{frame_ext}")
                    logger.info(f"  Test path: {test_path}")
                    logger.info(f"  Exists: {os.path.exists(test_path)}")

                    # List actual files in the directory
                    video_dir = os.path.join(frames_dir, video_name)
                    if os.path.exists(video_dir):
                        files = sorted(os.listdir(video_dir))[:5]  # First 5 files
                        logger.info(f"  First 5 files in dir: {files}")

                # Load hand mask
                if use_hand_masks:
                    has_hand, hand_mask = load_hand_mask(hand_mask_dir, video_name, frame_idx)
                    if not has_hand or hand_mask is None:
                        hand_mask = np.zeros((H, W), dtype=np.uint8)
                else:
                    hand_mask = np.zeros((H, W), dtype=np.uint8)

                hand_mask_resized = cv2.resize(hand_mask, (W, H), interpolation=cv2.INTER_NEAREST)
                hand_masks.append(hand_mask_resized)

                # Load frame image
                img = load_frame_image(
                    frames_dir,
                    video_name,
                    frame_idx,
                    frame_ext,
                    fallback_video_path=video_path,
                    fallback_clip_start=clip_start,
                    warn_missing=False,
                )
                if img is None:
                    # Use a gray placeholder if both frame dir and fallback video fail
                    img = np.ones((H, W, 3), dtype=np.uint8) * 128
                frames.append(img)

                # Get GT gaze (Ego4D gaze_type semantics: 0/1/2 valid, 3/4 invalid)
                gt_x, gt_y, gaze_type = labels[b, t].numpy()
                is_valid = gaze_type not in [3, 4]
                gt_points.append((gt_x, gt_y, float(is_valid)))

                # Get predicted gaze
                pred_heatmap = preds[b, t].numpy()
                pred_x, pred_y = get_predicted_gaze_point(pred_heatmap)
                pred_points.append((pred_x, pred_y))

                # Compute error
                if is_valid:
                    error = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
                else:
                    # Use NaN for invalid GT (will be filtered out in statistics)
                    error = np.nan

                errors.append(error)

            # Add clip to collector
            collector.add_clip(
                frames, hand_masks, gt_points, pred_points,
                errors, indices, video_name
            )

        if show_progress and cur_iter % 10 == 0:
            total_clips = sum(len(v) for v in collector.clips.values())
            test_iterator.set_postfix({'Clips': total_clips})

    return collector


def test(cfg):
    """Main test function."""
    # Set up environment
    du.init_distributed_training(cfg)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    logging.setup_logging(cfg.OUTPUT_DIR)

    # Build and load model
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)
    cu.load_test_checkpoint(cfg, model)

    # Create test loader
    test_loader = loader.construct_loader(cfg, "test")
    logger.info(f"Testing model for {len(test_loader)} iterations")

    # Get directories
    hand_mask_dir = getattr(cfg.DATA, "HAND_MASK_DIR", None)
    frames_dir = cfg.DATA.FRAMES_DIR
    frame_ext = getattr(cfg.DATA, 'FRAME_EXT', 'jpg')

    use_hand_masks = bool(hand_mask_dir) and os.path.exists(hand_mask_dir)
    if hand_mask_dir and not os.path.exists(hand_mask_dir):
        logger.warning(f"Hand mask directory not found: {hand_mask_dir}. Proceeding without hand masks.")
    if use_hand_masks:
        logger.info(f"Hand masks: {hand_mask_dir}")
    else:
        logger.info("Hand masks disabled or directory missing; proceeding without hand mask overlays.")
    logger.info(f"Frames: {frames_dir}")

    # Collect error clips
    collector = collect_error_clips(test_loader, model, hand_mask_dir,
                                    frames_dir, frame_ext, cfg,
                                    use_hand_masks=use_hand_masks)

    # Log statistics
    logger.info("=" * 80)
    logger.info("Error Clip Collection Summary")
    logger.info("=" * 80)
    logger.info(f"Total clips with errors:     {len(collector.clips['all_errors'])} clips")
    logger.info(f"Single error spike:          {len(collector.clips['single_spike'])} clips")
    logger.info(f"Multiple error frames:       {len(collector.clips['multiple_errors'])} clips")
    logger.info(f"Gradual error increase:      {len(collector.clips['gradual_degradation'])} clips")
    logger.info("=" * 80)

    # Generate visualizations
    vis_dir = os.path.join(cfg.OUTPUT_DIR, 'error_clip_visualizations')
    logger.info(f"\nGenerating clip visualizations in {vis_dir}...")
    logger.info(f"Will create top 50 clips from all_errors category...")
    collector.visualize_all_clips(vis_dir, top_n=50)

    logger.info("\nVisualization complete!")
    logger.info(f"Results saved to: {vis_dir}")

    return collector


if __name__ == "__main__":
    import slowfast.utils.parser as parser

    args = parser.parse_args()
    cfg = parser.load_config(args)

    test(cfg)
