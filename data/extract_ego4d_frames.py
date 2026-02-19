#!/usr/bin/env python3
"""
Extract frames from Ego4D gaze videos to speed up data loading.

Usage:
    python extract_ego4d_frames.py \
        --video_root /mnt/data1/jiali/data/gaze/ego4d/v2/clips.gaze \
        --output_dir /mnt/data1/jiali/data/gaze/ego4d/v2/frames \
        --ext jpg \
        --quality 95 \
        --num_workers 8

Frame naming convention:
    {video_name}/{frame_idx:06d}.jpg
    Example: video_001/000001.jpg, video_001/000002.jpg, ...

This script will:
1. Find all .mp4 files in video_root
2. Extract ALL frames from each video
3. Save frames with zero-padded 6-digit indices
4. Support parallel processing with multiple workers
"""

import os
import argparse
import cv2
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
import time


def extract_frames_from_video(args_tuple):
    """Extract frames from a single video clip.

    Args:
        args_tuple: (video_path, output_dir, ext, quality)

    Returns:
        Tuple of (video_name, num_frames_extracted, success, message)
    """
    video_path, output_dir, ext, quality = args_tuple

    # Get video name from path (parent directory name)
    video_name = os.path.basename(os.path.dirname(video_path))
    clip_name = os.path.basename(video_path)

    # Create output directory for this video
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    try:
        # Open video to get properties
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return (video_name, 0, False, f"Failed to open video: {clip_name}")

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Parse clip start and end time from filename (e.g., "video_t0_t5.mp4")
        try:
            parts = clip_name.split('_')
            clip_tstart = int(parts[-2][1:])  # Remove 't' prefix, e.g., "t0" -> 0
            clip_tend = int(parts[-1].split('.')[0][1:])  # Remove 't' and extension, e.g., "t5.mp4" -> 5

            # Calculate global frame offset (assuming TARGET_FPS, usually 30)
            # This should match the logic in ego4d_gaze.py line 184 and 219
            target_fps = 30  # Default FPS used in ego4d dataset
            # ego4d_gaze.py: clip_fstart = clip_tstart * TARGET_FPS
            # ego4d_gaze.py: frames_global_idx = idx.numpy() + int(clip_fstart) - 1
            # where idx is 0-based within the clip, so:
            # frame 0 in clip -> clip_fstart - 1
            # frame 1 in clip -> clip_fstart
            # frame 2 in clip -> clip_fstart + 1
            frame_offset = int(clip_tstart * target_fps) - 1  # Subtract 1 to match ego4d_gaze.py logic
        except Exception as e:
            return (video_name, 0, False, f"Failed to parse clip times from {clip_name}: {e}")

        # Check if this specific clip's frames already exist
        # Expected frames: [frame_offset, frame_offset+1, ..., frame_offset+total_frames-1]
        expected_frame_indices = list(range(frame_offset, frame_offset + total_frames))
        existing_frames = [
            os.path.join(video_output_dir, f"{idx:06d}.{ext}")
            for idx in expected_frame_indices
        ]

        if all(os.path.exists(f) for f in existing_frames):
            cap.release()
            return (video_name, total_frames, True, f"Already extracted (clip: {clip_name})")

        # Extract frames
        frame_idx = 0
        frames_saved = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Global frame index matching ego4d_gaze.py line 219
            # ego4d_gaze.py: frames_global_idx = idx.numpy() + int(clip_fstart) - 1
            # where idx is 0-based within the clip (0, 1, 2, ...)
            # so for frame_idx=0 in clip: frame_offset + 0 = clip_fstart - 1
            global_frame_idx = frame_offset + frame_idx

            # Save frame with zero-padded filename
            frame_path = os.path.join(video_output_dir, f"{global_frame_idx:06d}.{ext}")

            # Skip if frame already exists (in case of interrupted extraction)
            if not os.path.exists(frame_path):
                if ext == 'jpg':
                    cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                elif ext == 'png':
                    cv2.imwrite(frame_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 9 - (quality // 10)])
                else:
                    cv2.imwrite(frame_path, frame)
                frames_saved += 1

            frame_idx += 1

        cap.release()

        return (video_name, frames_saved, True, f"Extracted {frames_saved} frames from {clip_name}")

    except Exception as e:
        return (video_name, 0, False, f"Error: {str(e)}")


def find_all_videos(video_root):
    """Find all video files recursively."""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_paths = []

    for root, _dirs, files in os.walk(video_root):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_paths.append(os.path.join(root, file))

    return sorted(video_paths)


def main():
    parser = argparse.ArgumentParser(description="Extract frames from Ego4D gaze videos")
    parser.add_argument('--video_root', type=str, required=True,
                        help='Root directory containing video files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for extracted frames')
    parser.add_argument('--ext', type=str, default='jpg', choices=['jpg', 'png'],
                        help='Image format (default: jpg)')
    parser.add_argument('--quality', type=int, default=95,
                        help='JPEG quality (1-100) or PNG compression level (0-90)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of parallel workers (default: 8)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print video paths without extracting frames')

    args = parser.parse_args()

    # Find all videos
    print(f"Searching for videos in {args.video_root}...")
    video_paths = find_all_videos(args.video_root)
    print(f"Found {len(video_paths)} videos")

    if args.dry_run:
        print("\nDry run - Videos to process:")
        for i, path in enumerate(video_paths[:10], 1):
            print(f"  {i}. {path}")
        if len(video_paths) > 10:
            print(f"  ... and {len(video_paths) - 10} more")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare arguments for parallel processing
    tasks = [(video_path, args.output_dir, args.ext, args.quality)
             for video_path in video_paths]

    # Process videos in parallel
    print(f"\nExtracting frames using {args.num_workers} workers...")
    start_time = time.time()

    with Pool(processes=args.num_workers) as pool:
        results = list(tqdm(
            pool.imap(extract_frames_from_video, tasks),
            total=len(tasks),
            desc="Processing videos"
        ))

    # Summarize results
    total_time = time.time() - start_time
    successful = sum(1 for _, _, success, _ in results if success)
    failed = len(results) - successful
    total_frames = sum(frames for _, frames, _, _ in results)

    print(f"\n{'='*60}")
    print(f"Frame extraction completed!")
    print(f"{'='*60}")
    print(f"Total videos processed: {len(results)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"Total frames extracted: {total_frames:,}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average: {total_time/len(results):.2f} seconds/video")
    print(f"Output directory: {args.output_dir}")

    # Print failed videos
    if failed > 0:
        print(f"\nFailed videos:")
        for video_name, _, success, msg in results:
            if not success:
                print(f"  - {video_name}: {msg}")


if __name__ == "__main__":
    main()
