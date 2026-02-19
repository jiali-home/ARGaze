#!/usr/bin/env python3
import argparse
import time
import os
from pathlib import Path
import torch
from tqdm import tqdm
import numpy as np
import cv2

from dataloader import build_gaze_dataloader


def main():
    ap = argparse.ArgumentParser(description="Quick test for Ego-Exo Gaze DataLoader")
    ap.add_argument("--frames-root-dir", default="/mnt/sdc1/jiali/data/ego-exo/takes_frames", help="Root that mirrors .../takes/<take>/frame_aligned_videos/... frames")
    ap.add_argument("--gaze-data-dir", default="/mnt/sdc1/jiali/data/ego-exo/gaze_data", help="Root containing takes/<take>/eye_gaze/{personalized,general}_eye_gaze_2d.csv")
    ap.add_argument("--split-csv", default="./ood/split_assignments.csv", help="Path to split_assignments.csv (e.g., explore/ood/split_assignments.csv)")
    ap.add_argument("--split", default="test_ood_participant", help="train|val|test_iid|test_ood_task|test_ood_site|test_ood_participant")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-frames", type=int, default=8)
    ap.add_argument("--target-fps", type=int, default=10)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--heatmap-sigma", type=float, default=None)
    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--vis-dir", default="explore/gaze_previews")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--num-batches", type=int, default=1, help="Iterate this many batches to measure throughput and show progress")
    ap.add_argument("--mps-enabled", action="store_true")
    # Visualization of raw frames with GT point/heatmap
    ap.add_argument("--dump-examples", action="store_true", help="Save example overlays from the first batch")
    ap.add_argument("--dump-dir", default="explore/gaze_samples", help="Directory to save example overlays")
    ap.add_argument("--dump-samples", type=int, default=4, help="How many samples from the first batch to save")
    ap.add_argument("--dump-t-index", type=int, default=-1, help="Temporal index to visualize (-1 for last frame)")
    ap.add_argument("--overlay-heatmap", action="store_true", help="Also overlay GT heatmap on the raw frame")
    args = ap.parse_args()

    loader = build_gaze_dataloader(
        frames_root_dir=args.frames_root_dir,
        gaze_data_dir=args.gaze_data_dir,
        split_assignments_csv=args.split_csv,
        split=args.split,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        target_fps=args.target_fps,
        image_size=args.image_size,
        stride=args.stride,
        heatmap_sigma=args.heatmap_sigma,
        visualize=args.visualize,
        vis_dir=args.vis_dir,
        num_workers=args.num_workers,
        mps_enabled=args.mps_enabled,
    )

    print(f"Built loader for split={args.split}. Dataset size: {len(loader.dataset)} samples. Batches incoming...")
    it = iter(loader)
    start = time.time()
    first = None
    for _ in tqdm(range(max(1, args.num_batches)), desc="Loading batches"):
        try:
            batch = next(it)
        except StopIteration:
            print("Dataset is empty or exhausted.")
            return 0
        if first is None:
            first = batch
    elapsed = time.time() - start
    if elapsed > 0:
        print(f"Throughput: {args.num_batches/elapsed:.2f} batches/sec with num_workers={args.num_workers}")
    imgs, poses, gazes, heatmaps = first

    # Shapes
    print("img_tensor:", tuple(imgs.shape), "(B,C,T,H,W)" if imgs.dim() == 5 else "(C,T,H,W)")
    print("pose_9d_tensor:", tuple(poses.shape), "(B,T,9)")
    print("gaze_tensor:", tuple(gazes.shape), "(B,T,3)")
    print("gaze_heatmap_tensor:", tuple(heatmaps.shape), "(B,T,H',W')")

    # Basic checks
    print("img range:", float(imgs.min()), "to", float(imgs.max()))
    nan_pose = torch.isnan(poses).any(dim=-1).sum()
    print("pose NaN frames:", int(nan_pose))
    nan_mask = torch.isnan(gazes[..., :2])
    print("gaze NaNs (x or y):", int(nan_mask.any(dim=-1).sum()))
    print("heatmap stats: min", float(heatmaps.min()), "max", float(heatmaps.max()))

    # Peek a few gaze points from the first sample
    b0 = 0
    t_idxs = list(range(min(3, gazes.shape[1])))
    for t in t_idxs:
        gx, gy, vis = gazes[b0, t].tolist()
        print(f"[sample0 t={t}] gaze=({gx:.3f},{gy:.3f}), vis={vis}")

    print("OK. If --visualize was set, previews were saved to:", args.vis_dir)

    # Optionally dump a few overlays from the first batch using raw frames.
    if args.dump_examples:
        out_dir = Path(args.dump_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ds = loader.dataset
        B, _, T, H, W = imgs.shape
        sel = list(range(min(B, max(1, int(args.dump_samples)))))
        t_vis = args.dump_t_index if (0 <= args.dump_t_index < T) else (T - 1)
        print(f"Saving overlays for samples {sel} at t={t_vis} into {str(out_dir)}")

        for b in sel:
            # Resolve sample info to fetch raw frame
            try:
                s = ds.samples[b]
                frames_dir = Path(s['frames_dir'])
                frame_idx = int(s['frame_indices'][t_vis])
                take = s.get('take_name', frames_dir.parent.name)
            except Exception:
                # Fallback if ds.samples not accessible
                frames_dir = None
                frame_idx = None
                take = f"sample_{b:04d}"

            # Load raw image
            if frames_dir is not None and frame_idx is not None:
                fpath = frames_dir / f"frame_{frame_idx:05d}.jpg"
                img = cv2.imread(str(fpath))
            else:
                # Use resized tensor frame as fallback
                img = (imgs[b, :, t_vis].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if img is None:
                continue
            img = np.ascontiguousarray(img)
            Ih, Iw = img.shape[:2]

            # Prepare GT heatmap upsampled to the image size (align with training W&B logic)
            hm = heatmaps[b, t_vis].unsqueeze(0).unsqueeze(0)  # [1,1,h,w]
            hm = torch.nn.functional.interpolate(hm, size=(Ih, Iw), mode='bilinear', align_corners=False)[0, 0]
            hm = hm.detach().cpu().numpy()
            hm = (hm - np.min(hm)) / (np.max(hm) - np.min(hm) + 1e-6)  # [0,1]

            # Build red (pred) and green (gt) overlays, mirroring train_gaze_net W&B
            pred_rgb = np.stack([hm, np.zeros_like(hm), np.zeros_like(hm)], axis=-1)  # R
            gt_rgb = np.stack([np.zeros_like(hm), hm, np.zeros_like(hm)], axis=-1)    # G
            overlay_pred = (0.4 * (pred_rgb * 255.0) + img.astype(np.float32)).clip(0, 255).astype(np.uint8)
            overlay_gt = (0.4 * (gt_rgb * 255.0) + img.astype(np.float32)).clip(0, 255).astype(np.uint8)

            # Draw GT point (normalized [0,1]) on all three panels for clarity
            gx, gy, vis = gazes[b, t_vis].tolist()
            if np.isfinite(gx) and np.isfinite(gy) and vis > 0.0:
                px = int(round(float(gx) * (Iw - 1)))
                py = int(round(float(gy) * (Ih - 1)))
                for panel in (img, overlay_pred, overlay_gt):
                    cv2.circle(panel, (px, py), 7, (255, 255, 255), thickness=3)
                    cv2.circle(panel, (px, py), 3, (0, 0, 0), thickness=-1)

            # Save triptych: [raw | pred-overlay | gt-overlay]
            panel = np.concatenate([img, overlay_pred, overlay_gt], axis=1)
            base = f"{take}_b{b:03d}_t{t_vis:02d}_f{frame_idx if frame_idx is not None else -1:05d}.jpg"
            cv2.imwrite(str(out_dir / base), panel)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
