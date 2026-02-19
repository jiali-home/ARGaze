#!/usr/bin/env python3
"""
Streaming Gaze Estimation Demo for DINOv3_ARHeatmapGazeTemplate

This script provides real-time gaze estimation from webcam or video input.
Output: Frame with gaze heatmap overlay and predicted gaze point marker.

Usage:
    # Webcam input
    python demo_streaming.py --config demo_config.yaml --checkpoint checkpoint.pyth --camera 0

    # Video file input
    python demo_streaming.py --config demo_config.yaml --checkpoint checkpoint.pyth --video input.mp4

    # Video file with output
    python demo_streaming.py --config demo_config.yaml --checkpoint checkpoint.pyth --video input.mp4 --output output.mp4
"""

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Add parent directory to path for slowfast imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from slowfast.config.defaults import get_cfg
from slowfast.models.build import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Streaming Gaze Estimation Demo")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--camera", type=int, default=None, help="Camera device index (e.g., 0)")
    parser.add_argument("--video", type=str, default=None, help="Path to input video file")
    parser.add_argument("--output", type=str, default=None, help="Path to output video file (optional)")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--show", action="store_true", default=True, help="Display output window")
    parser.add_argument("--no-show", dest="show", action="store_false", help="Disable display window")
    parser.add_argument("--heatmap-alpha", type=float, default=0.4, help="Heatmap overlay alpha (0-1)")
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    return cfg


def load_model(cfg, checkpoint_path, device):
    """Build model and load checkpoint."""
    # Ensure custom models are registered.
    try:
        import slowfast.models.custom_video_model_builder  # noqa: F401
    except Exception:
        pass
    model = build_model(cfg)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Handle different checkpoint formats
    if "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present (from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print(f"Model loaded from {checkpoint_path}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model


def preprocess_frame(frame, target_size=224):
    """
    Preprocess a single frame for model input.

    Args:
        frame: BGR frame from OpenCV (H, W, 3)
        target_size: Target size for model input

    Returns:
        tensor: (3, target_size, target_size) in [0, 1] range
    """
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize to target size
    frame_resized = cv2.resize(frame_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    # Convert to tensor and normalize to [0, 1]
    tensor = torch.from_numpy(frame_resized).float() / 255.0

    # HWC -> CHW
    tensor = tensor.permute(2, 0, 1)

    return tensor


def heatmap_to_coords(heatmap):
    """
    Convert heatmap to (x, y) coordinates using soft-argmax.

    Args:
        heatmap: (B, 1, H, W) or (H, W) tensor

    Returns:
        coords: (x, y) normalized coordinates in [0, 1]
    """
    if heatmap.dim() == 2:
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)
    elif heatmap.dim() == 3:
        heatmap = heatmap.unsqueeze(0)

    B, _, H, W = heatmap.shape
    hm_flat = heatmap.view(B, -1)
    prob = F.softmax(hm_flat, dim=-1).view(B, H, W)

    y_coords = torch.linspace(0, 1, steps=H, device=heatmap.device).view(1, H, 1)
    x_coords = torch.linspace(0, 1, steps=W, device=heatmap.device).view(1, 1, W)

    y_expect = (prob * y_coords).sum(dim=(1, 2))
    x_expect = (prob * x_coords).sum(dim=(1, 2))

    return x_expect.item(), y_expect.item()


def overlay_heatmap(frame, heatmap, alpha=0.4):
    """
    Overlay heatmap on frame.

    Args:
        frame: BGR frame (H, W, 3)
        heatmap: (H_hm, W_hm) numpy array
        alpha: Overlay transparency

    Returns:
        blended: BGR frame with heatmap overlay
    """
    h, w = frame.shape[:2]

    # Resize heatmap to frame size
    hm_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)

    # Normalize to [0, 255]
    hm_min, hm_max = hm_resized.min(), hm_resized.max()
    if hm_max > hm_min:
        hm_norm = ((hm_resized - hm_min) / (hm_max - hm_min) * 255).astype(np.uint8)
    else:
        hm_norm = np.zeros_like(hm_resized, dtype=np.uint8)

    # Apply colormap (JET)
    hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)

    # Blend with original frame
    blended = cv2.addWeighted(frame, 1.0, hm_color, alpha, 0)

    return blended


def draw_gaze_marker(frame, x, y, color=(0, 255, 0), radius=10, thickness=2):
    """
    Draw gaze point marker on frame.

    Args:
        frame: BGR frame (H, W, 3)
        x, y: Normalized coordinates in [0, 1]
        color: BGR color tuple
        radius: Marker radius
        thickness: Line thickness

    Returns:
        frame: Frame with marker drawn
    """
    h, w = frame.shape[:2]
    px = int(x * w)
    py = int(y * h)

    # Draw crosshair
    cv2.circle(frame, (px, py), radius, color, thickness)
    cv2.line(frame, (px - radius - 5, py), (px + radius + 5, py), color, thickness)
    cv2.line(frame, (px, py - radius - 5), (px, py + radius + 5), color, thickness)

    return frame


def draw_info_overlay(frame, fps, gaze_x, gaze_y, buffering=False):
    """Draw information overlay on frame."""
    h, w = frame.shape[:2]

    # Semi-transparent background for text
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (280, 100), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    # Text info
    font = cv2.FONT_HERSHEY_SIMPLEX
    if buffering:
        cv2.putText(frame, "Buffering...", (20, 40), font, 0.7, (255, 255, 0), 2)
    else:
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), font, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Gaze: ({gaze_x:.3f}, {gaze_y:.3f})", (20, 70), font, 0.6, (255, 255, 255), 2)

    return frame


class GazeEstimator:
    """Streaming gaze estimation wrapper."""

    def __init__(self, model, cfg, device):
        self.model = model
        self.cfg = cfg
        self.device = device

        self.num_frames = cfg.DATA.NUM_FRAMES  # 8
        self.input_size = cfg.DATA.TRAIN_CROP_SIZE  # 224

        # Frame buffer (sliding window)
        self.frame_buffer = deque(maxlen=self.num_frames)

    def reset(self):
        """Reset the frame buffer."""
        self.frame_buffer.clear()

    def process_frame(self, frame):
        """
        Process a single frame and return gaze prediction.

        Args:
            frame: BGR frame from OpenCV

        Returns:
            gaze_x, gaze_y: Normalized gaze coordinates (or None if buffering)
            heatmap: Output heatmap (or None if buffering)
        """
        # Preprocess and add to buffer
        tensor = preprocess_frame(frame, self.input_size)
        self.frame_buffer.append(tensor)

        # Check if buffer is full
        if len(self.frame_buffer) < self.num_frames:
            return None, None, None

        # Stack frames: (T, C, H, W)
        frames = torch.stack(list(self.frame_buffer), dim=0)

        # Add batch dimension and rearrange: (B, C, T, H, W)
        frames = frames.unsqueeze(0).permute(0, 2, 1, 3, 4)
        frames = frames.to(self.device)

        # Run inference
        with torch.no_grad():
            # gt_heatmap=None, train_ar=True, ss_prob=0.0 for autoregressive inference
            heatmaps = self.model(frames, gt_heatmap=None, train_ar=True, ss_prob=0.0)
            # heatmaps: (B, 1, T, H_hm, W_hm)

        # Extract last frame's heatmap
        last_heatmap = heatmaps[0, 0, -1]  # (H_hm, W_hm)

        # Convert to coordinates
        gaze_x, gaze_y = heatmap_to_coords(last_heatmap)

        # Convert heatmap to numpy for visualization
        heatmap_np = last_heatmap.cpu().numpy()

        return gaze_x, gaze_y, heatmap_np


def main():
    args = parse_args()

    # Validate input source
    if args.camera is None and args.video is None:
        print("Error: Must specify either --camera or --video")
        sys.exit(1)

    # Load config and model
    print("Loading configuration...")
    cfg = load_config(args.config)

    print("Loading model...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(cfg, args.checkpoint, device)

    # Initialize gaze estimator
    estimator = GazeEstimator(model, cfg, device)

    # Open video source
    if args.camera is not None:
        print(f"Opening camera {args.camera}...")
        cap = cv2.VideoCapture(args.camera)
    else:
        print(f"Opening video {args.video}...")
        cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        print("Error: Could not open video source")
        sys.exit(1)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_source = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print(f"Video source: {frame_width}x{frame_height} @ {fps_source:.1f} FPS")

    # Setup output writer if specified
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps_source, (frame_width, frame_height))
        print(f"Writing output to {args.output}")

    # Processing loop
    print("\nStarting gaze estimation...")
    print("Press 'q' to quit, 'r' to reset buffer")

    fps_counter = deque(maxlen=30)
    frame_count = 0

    try:
        while True:
            t_start = time.time()

            ret, frame = cap.read()
            if not ret:
                if args.video:
                    print("End of video")
                break

            # Process frame
            gaze_x, gaze_y, heatmap = estimator.process_frame(frame)

            # Create output visualization
            if gaze_x is not None:
                # Overlay heatmap
                vis_frame = overlay_heatmap(frame.copy(), heatmap, alpha=args.heatmap_alpha)

                # Draw gaze marker
                vis_frame = draw_gaze_marker(vis_frame, gaze_x, gaze_y, color=(0, 255, 0))

                # Calculate FPS
                t_elapsed = time.time() - t_start
                fps_counter.append(1.0 / max(t_elapsed, 1e-6))
                current_fps = np.mean(fps_counter)

                # Draw info overlay
                vis_frame = draw_info_overlay(vis_frame, current_fps, gaze_x, gaze_y)
            else:
                # Still buffering
                vis_frame = frame.copy()
                vis_frame = draw_info_overlay(vis_frame, 0, 0, 0, buffering=True)
                buffer_progress = len(estimator.frame_buffer) / estimator.num_frames
                cv2.putText(vis_frame, f"Buffer: {len(estimator.frame_buffer)}/{estimator.num_frames}",
                           (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Write to output
            if writer:
                writer.write(vis_frame)

            # Display
            if args.show:
                cv2.imshow("Gaze Estimation Demo", vis_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    estimator.reset()
                    print("Buffer reset")

            frame_count += 1

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    print(f"\nProcessed {frame_count} frames")


if __name__ == "__main__":
    main()
