#!/usr/bin/env python3
"""Resize Ego4D frames down to 224x224 for only the samples used in training.

The original script blindly processed every frame under ``DATA.FRAMES_DIR``.  For
the fixed-window training setup we only need the frames that actually appear in
``data/train_ego4d_gaze_stride8.csv`` (or any CSV with ``clip_path,start_frame,
end_frame``).  This version reads that CSV, enumerates the exact frame indices
referenced by each sample (``start_frame + k * sampling_rate`` for
``k=0..NUM_FRAMES-1``), and resizes only those images.  The result is a much
smaller preprocessed frame set and faster preprocessing.

Example:

```
python resize_ego4d_frames.py \
    --csv data/train_ego4d_gaze_stride8.csv \
    --num-frames 8 --sampling-rate 8 --fps 30 \
    --src-dir /mnt/data1/jiali/data/gaze/ego4d/v2/frames \
    --dst-dir /mnt/data1/jiali/data/gaze/ego4d/v2/frames_stride8_224
```

Options:
    --mode: Resize mode ('crop' keeps aspect ratio, 'squash' direct resize)
    --workers: Number of parallel workers
    --quality: JPEG quality (1-100)
    --csv: CSV enumerating samples with frame ranges
    --num-frames / --sampling-rate / --fps: match training config so we pick
        the same frame indices used during training.
"""

import os
import csv
import argparse
from collections import defaultdict
from pathlib import Path
from multiprocessing import Pool
from typing import Dict, List, Set

from tqdm import tqdm
from PIL import Image


def resize_image_crop(img, target_size=224):
    """Resize with aspect ratio preservation and center crop."""
    # Resize so that the shorter side is target_size
    w, h = img.size
    if w < h:
        new_w = target_size
        new_h = int(h * target_size / w)
    else:
        new_h = target_size
        new_w = int(w * target_size / h)

    img = img.resize((new_w, new_h), Image.LANCZOS)

    # Center crop to target_size x target_size
    left = (new_w - target_size) // 2
    top = (new_h - target_size) // 2
    right = left + target_size
    bottom = top + target_size

    img = img.crop((left, top, right, bottom))
    return img


def resize_image_squash(img, target_size=224):
    """Directly resize to target_size x target_size (may distort)."""
    return img.resize((target_size, target_size), Image.LANCZOS)


def process_image(args):
    """Process a single image file."""
    src_path, dst_path, mode, quality = args

    try:
        # Create output directory if needed
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip if already exists
        if dst_path.exists():
            return True, f"Skipped (exists): {dst_path}"

        # Load image
        img = Image.open(src_path)

        # Convert to RGB if needed (some images might be grayscale or RGBA)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize based on mode
        if mode == 'crop':
            img_resized = resize_image_crop(img, target_size=224)
        else:  # squash
            img_resized = resize_image_squash(img, target_size=224)

        # Save
        img_resized.save(dst_path, 'JPEG', quality=quality)

        return True, None

    except Exception as e:
        return False, f"Error processing {src_path}: {e}"


def parse_sample_csv(csv_path: Path) -> List[List[str]]:
    """Return rows from CSV, skipping optional header."""
    rows: List[List[str]] = []
    with csv_path.open("r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0].startswith("#"):
                continue
            rows.append(row)
    # Drop header if present (non-numeric second column)
    if rows and rows[0][0] == "clip_path":
        rows = rows[1:]
    return rows


def collect_required_frames(
    rows: List[List[str]],
    num_frames: int,
    sampling_rate: int,
    coverage: str,
) -> Dict[str, Set[int]]:
    """Map each video to the exact frame indices we must resize.

    When ``coverage`` is ``dense`` we materialize *every* frame in
    ``[start_frame, end_frame)`` for the clip window.  This matches the
    dataloader's behavior when sampling from pre-extracted frames (it may pick
    any frame inside the window, not just evenly spaced points), preventing
    missing-file fallbacks during training.  ``sampled`` retains the previous
    behavior of only touching ``start_frame + k * sampling_rate``.
    """

    required: Dict[str, Set[int]] = defaultdict(set)
    for row in rows:
        if len(row) < 3:
            raise ValueError("CSV must have at least three columns: clip_path,start_frame,end_frame")
        rel_path, start_frame, end_frame = row[0], int(row[1]), int(row[2])
        video_dir = rel_path.split("/", 1)[0]

        if coverage == "sampled":
            for k in range(num_frames):
                frame_idx = start_frame + k * sampling_rate
                if frame_idx >= end_frame:
                    break
                required[video_dir].add(frame_idx)
        else:  # dense coverage over the full window
            required[video_dir].update(range(start_frame, end_frame))
    return required


def main():
    parser = argparse.ArgumentParser(description='Resize Ego4D frames to 224x224 (sample-aware)')
    parser.add_argument('--csv', type=str, required=True,
                        help='CSV listing clip_path,start_frame,end_frame (e.g. data/train_ego4d_gaze_stride8.csv)')
    parser.add_argument('--num-frames', type=int, default=8,
                        help='Number of frames per sample (DATA.NUM_FRAMES)')
    parser.add_argument('--sampling-rate', type=int, default=8,
                        help='Sampling rate between successive frames (DATA.SAMPLING_RATE)')
    parser.add_argument('--mode', type=str, default='crop', choices=['crop', 'squash'],
                        help='Resize mode: crop (preserve aspect ratio) or squash (direct resize)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers')
    parser.add_argument('--quality', type=int, default=95,
                        help='JPEG quality (1-100)')
    parser.add_argument('--src-dir', type=str, default='/mnt/data1/jiali/data/gaze/ego4d/v2/frames',
                        help='Directory with original frames (video_id/000001.jpg)')
    parser.add_argument('--dst-dir', type=str, default='/mnt/data1/jiali/data/gaze/ego4d/v2/frames_224',
                        help='Destination directory for resized frames')
    parser.add_argument('--frame-ext', type=str, default='jpg',
                        help='Frame file extension (default: jpg)')
    parser.add_argument('--coverage', type=str, default='dense', choices=['dense', 'sampled'],
                        help='dense = resize every frame in each window (safe for dataloader), '
                             'sampled = only resize evenly spaced frames (may miss some)')

    args = parser.parse_args()

    src_root = Path(args.src_dir)
    dst_root = Path(args.dst_dir)
    csv_path = Path(args.csv)

    if not src_root.exists():
        print(f"Error: Source directory does not exist: {src_root}")
        return
    if not csv_path.exists():
        print(f"Error: CSV not found: {csv_path}")
        return

    rows = parse_sample_csv(csv_path)
    if not rows:
        print(f"No rows found in {csv_path}")
        return

    frame_dict = collect_required_frames(rows, args.num_frames, args.sampling_rate, args.coverage)
    total_needed = sum(len(v) for v in frame_dict.values())

    print(f"Sample CSV: {csv_path}")
    print(f"Videos: {len(frame_dict)} | Frames needed: {total_needed}")
    print(f"Source: {src_root}")
    print(f"Destination: {dst_root}")
    print(f"Mode: {args.mode} | Workers: {args.workers} | Quality: {args.quality}")

    dst_root.mkdir(parents=True, exist_ok=True)

    tasks = []
    for video_dir, frame_indices in frame_dict.items():
        for frame_idx in sorted(frame_indices):
            filename = f"{frame_idx:06d}.{args.frame_ext}"
            src_path = src_root / video_dir / filename
            dst_path = dst_root / video_dir / filename
            tasks.append((src_path, dst_path, args.mode, args.quality))

    filtered_tasks = []
    missing = []
    for tup in tasks:
        if tup[0].exists():
            filtered_tasks.append(tup)
        else:
            missing.append(str(tup[0]))

    if missing:
        print(f"Warning: {len(missing)} source frames are missing. Example: {missing[0]}")

    tasks = filtered_tasks

    print(f"\nProcessing {len(tasks)} frames...")
    success_count = 0
    error_count = 0

    with Pool(processes=args.workers) as pool:
        results = list(tqdm(
            pool.imap(process_image, tasks),
            total=len(tasks),
            desc="Resizing images",
            unit="img"
        ))

    for success, msg in results:
        if success:
            success_count += 1
        else:
            error_count += 1
            if msg:
                print(msg)

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Needed frames: {len(tasks)}")
    print(f"  Success: {success_count}")
    print(f"  Errors: {error_count}")
    print(f"  Output: {dst_root}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
