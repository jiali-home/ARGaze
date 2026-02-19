#!/usr/bin/env python3
"""
Extract frames for videos listed in a split CSV so the gaze dataloader can consume them.

Output layout matches the loader expectation:
  <frames_root_dir>/<rel_to_video_root>/<video_basename_without_ext>/frame_00000.jpg

Examples:
  python extract_frames.py \
    --csv ./ood/split_assignments.csv \
    --frames-root-dir /mnt/sdc1/jiali/data/ego-exo/takes_frames \
    --video-root-dir /mnt/sdc1/jiali/data/ego-exo/takes

Notes:
  - Safe resume: if interrupted, rerunning resumes from the last written frame.
  - By default, treats a folder as complete only when all frames exist; otherwise resumes.
  - Indexing starts at 0 for each video (frame_00000.jpg, frame_00001.jpg, ...).
  - Resizing is optional; the loader resizes at read time, so keeping native is OK.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import pandas as pd
from tqdm import tqdm


def derive_video_root(sample_video_path: str) -> str:
    if "/takes/" in sample_video_path:
        return sample_video_path.split("/takes/")[0]
    # Fallback: use three levels up
    return str(Path(sample_video_path).parents[3])


def extract_one_video(
    video_path: str,
    frames_root_dir: Path,
    video_root_dir: Path,
    resize_wh: Optional[Tuple[int, int]] = None,
    overwrite: bool = False,
) -> Tuple[str, int]:
    from pathlib import Path

    vp = Path(video_path)

    if not vp.exists():
        return (f"Video not found: {vp}", 0)
    try:
        rel = vp.relative_to(video_root_dir)
    except ValueError:
        return (f"Video {vp} is not under video_root_dir {video_root_dir}", 0)



    # old_root = Path("/mnt/sdc1/jiali/data/ego-exo")
    # new_root = Path("/mnt/data2/jiali/data/egoexo4d")

    # rel = vp.relative_to(old_root)
    # parts = list(rel.parts)
    # i = parts.index("frame_aligned_videos")  # will raise if not present

    # new_rel = parts[:i+1] + ["downscaled", "448"] + parts[i+1:]
    # vp_new = new_root.joinpath(*new_rel)

    # if not vp_new.exists():
    #     return (f"Video not found: {vp_new}", 0)

    # try:
    #     rel = vp_new.relative_to(video_root_dir)
    # except Exception:
    #     # Try to derive root automatically
    #     auto_root = derive_video_root(str(vp_new))
    #     rel = vp_new.relative_to(auto_root)

    out_dir = frames_root_dir / rel.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine resume point if any frames already exist
    start_idx = 0
    existing_indices = []
    non_contiguous_note = None
    if not overwrite:
        for p in out_dir.glob("frame_*.jpg"):
            name = p.stem  # frame_00012
            try:
                idx = int(name.split("_")[-1])
                existing_indices.append(idx)
            except Exception:
                continue
        if existing_indices:
            max_existing = max(existing_indices)
            min_existing = min(existing_indices)
            if not (min_existing == 0 and len(set(existing_indices)) == (max_existing + 1)):
                non_contiguous_note = " (warning: non-contiguous existing frames detected)"
            start_idx = max_existing + 1

    # cap = cv2.VideoCapture(str(vp_new))
    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        return (f"Cannot open video: {video_path}", 0)

    # If we already have all frames, skip (when frame count is reliable)
    total_frames_prop = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not overwrite and existing_indices and total_frames_prop > 0:
        if max(existing_indices) >= total_frames_prop - 1:
            cap.release()
            return (f"Already complete: {out_dir} ({len(set(existing_indices))} frames)", 0)

    # Seek to start index if resuming
    if start_idx > 0:
        ok_seek = cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        if not ok_seek:
            # Fallback: manually skip frames
            skipped = 0
            while skipped < start_idx:
                ok, _ = cap.read()
                if not ok:
                    break
                skipped += 1

    # Iterate frames from start_idx
    written = 0
    frame_idx = start_idx
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if resize_wh is not None:
            w, h = resize_wh
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        out_path = out_dir / f"frame_{frame_idx:05d}.jpg"
        cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        frame_idx += 1
        written += 1
    cap.release()

    if start_idx > 0 and written == 0:
        return (f"Resume check: no new frames; likely already complete at {out_dir}", 0)

    prefix = "Resumed and wrote" if start_idx > 0 else "Wrote"
    note = non_contiguous_note or ""
    return (f"{prefix} {written} frames -> {out_dir}{note}", written)


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract frames for videos in a split CSV")
    ap.add_argument("--csv", required=True, help="CSV with video_path column (e.g., explore/ood/split_assignments.csv)")
    ap.add_argument("--frames-root-dir", required=True, help="Root directory to write frames mirroring video tree")
    ap.add_argument("--video-root-dir", default=None, help="Root prefix to strip from video_path to get relative path (auto if omitted)")
    ap.add_argument("--resize-width", type=int, default=256, help="Resize output frames to this width (optional)")
    ap.add_argument("--resize-height", type=int, default=256, help="Resize output frames to this height (optional)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing frames")
    ap.add_argument(
        "--num-workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) // 2),
        help="Parallel video workers (processes). Default ~half of CPUs.",
    )
    ap.add_argument("--max-videos", type=int, default=None, help="Limit number of unique videos to process (for quick tests)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "video_path" not in df.columns:
        print("ERROR: CSV must contain 'video_path' column")
        return 2

    videos = df["video_path"].dropna().astype(str).unique().tolist()
    if args.max_videos is not None:
        videos = videos[: args.max_videos]

    frames_root_dir = Path(args.frames_root_dir)
    frames_root_dir.mkdir(parents=True, exist_ok=True)
    if args.video_root_dir:
        video_root_dir = Path(args.video_root_dir)
    else:
        video_root_dir = Path(derive_video_root(videos[0]))

    resize_wh = None
    if args.resize_width is not None and args.resize_height is not None:
        resize_wh = (int(args.resize_width), int(args.resize_height))

    total = 0

    # If only one worker requested, run sequentially for simpler logs.
    if args.num_workers <= 1:
        for vp in tqdm(videos, desc="Extracting"):
            msg, n = extract_one_video(
                vp,
                frames_root_dir=frames_root_dir,
                video_root_dir=video_root_dir,
                resize_wh=resize_wh,
                overwrite=args.overwrite,
            )
            total += n
            tqdm.write(msg)
    else:
        # Use a process pool to parallelize per-video extraction. To avoid CPU oversubscription,
        # keep OpenCV running single-threaded inside workers.
        def _init_worker():
            try:
                import cv2 as _cv2
                _cv2.setNumThreads(1)
            except Exception:
                pass

        with ProcessPoolExecutor(max_workers=args.num_workers, initializer=_init_worker) as ex:
            futures = [
                ex.submit(
                    extract_one_video,
                    vp,
                    frames_root_dir,
                    video_root_dir,
                    resize_wh,
                    args.overwrite,
                )
                for vp in videos
            ]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Extracting"):
                msg, n = fut.result()
                total += n
                tqdm.write(msg)

    print(f"Done. Total frames written: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# python extract_frames.py --csv ./ood/split_assignments.csv --frames-root-dir /mnt/data2/jiali/data/egoexo4d/takes_frames --video-root-dir /mnt/data2/jiali/data/egoexo4d/takes