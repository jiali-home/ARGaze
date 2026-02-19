#!/usr/bin/env python3

import torch
import torch.utils.data

import slowfast.utils.logging as logging

from .build import DATASET_REGISTRY
from . import utils as utils

logger = logging.get_logger(__name__)


import os
import math
import json
import hashlib
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from collections import OrderedDict

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.distributed as dist

try:
    from .build import DATASET_REGISTRY
except ImportError:
    DATASET_REGISTRY = None

logger = logging.getLogger(__name__)
_DEBUG = os.environ.get("EGOEXO_GAZE_DEBUG", "0") == "1"

if "cv2" in globals():
    cv2.setNumThreads(1)

def _str_label_alias(split: str) -> str:
    return {
        'train': 'TRAIN', 'val': 'VAL', 'test': 'TEST_IID',
        'test_iid': 'TEST_IID', 'test_ood_task': 'TEST_OOD_TASK',
        'test_ood_site': 'TEST_OOD_SITE', 'test_ood_participant': 'TEST_OOD_PARTICIPANT',
    }.get(split.lower(), split)

class EgoExoGazeClipDataset(Dataset):
    def __init__(self, cfg, split: str):
        self.cfg = cfg
        self.split_name = split
        self._init_params()
        self._init_paths()
        
        self.annotations = self._load_annotations()
        
        self._init_caches()
        self.samples: List[Dict[str, Any]] = []

        if not self._maybe_load_index_cache():
            is_rank0 = self._is_rank0()
            if is_rank0:
                self._precompute_samples()
                self._save_index_cache()
            self._barrier_if_dist()
            if not is_rank0 and not self._maybe_load_index_cache():
                self._precompute_samples()

    def _init_params(self):
        """Initialize dataset parameters from config."""
        data_cfg = self.cfg.DATA
        self.sequence_length = int(data_cfg.NUM_FRAMES)
        self.target_fps = int(data_cfg.TARGET_FPS)
        self.image_size = (int(data_cfg.TRAIN_CROP_SIZE), int(data_cfg.TRAIN_CROP_SIZE))
        self.heatmap_sigma = getattr(data_cfg, 'HEATMAP_SIGMA', None)
        self.fill_nan_center = bool(getattr(data_cfg, 'FILL_NAN_CENTER', False))
        self.interpolate_nan = bool(getattr(data_cfg, 'INTERPOLATE_NAN', True))
        self.window_stride = int(getattr(data_cfg, 'WINDOW_STRIDE', 1))
        self.skip_frame_exists_check = bool(getattr(data_cfg, 'SKIP_FRAME_EXISTS_CHECK', False))
        self.skip_missing_frames_dir = bool(getattr(data_cfg, 'SKIP_MISSING_FRAMES_DIR', False))
        self.subsample_fraction = float(getattr(data_cfg, f'SUBSAMPLE_{self.split_name.upper()}_FRACTION', 1.0))

    def _init_paths(self):
        data_cfg = self.cfg.DATA
        self.frames_root_dir = Path(data_cfg.FRAMES_DIR)
        self.video_root_dir = Path(data_cfg.VIDEO_ROOT_DIR)
        self.gaze_data_dir = Path(data_cfg.GAZE_DATA_DIR) if hasattr(data_cfg, 'GAZE_DATA_DIR') and data_cfg.GAZE_DATA_DIR else None
        self.cache_dir = Path(getattr(data_cfg, 'CACHE_DIR', self.frames_root_dir / '.gaze_cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.visualize = bool(getattr(data_cfg, 'VISUALIZE', False))
        if self.visualize:
            self.vis_dir = Path(getattr(data_cfg, 'VIS_DIR', 'explore/gaze_previews'))
            self.vis_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.vis_dir = None

    def _load_annotations(self):
        """Load and filter annotations from CSV file."""
        annotations_path = self._get_annotations_path(self.cfg, self.split_name)
        logger.info(f"Loading annotations from: {annotations_path}")
        annotations_original = pd.read_csv(annotations_path)
        logger.info(f"Loaded {len(annotations_original)} total annotations")

        annotations = annotations_original
        if 'split' in annotations.columns:
            split_label = _str_label_alias(self.split_name)
            logger.info(f"Filtering for split: {self.split_name} -> {split_label}")
            annotations = annotations[annotations['split'] == split_label].reset_index(drop=True)
            logger.info(f"After filtering: {len(annotations)} annotations for split '{split_label}'")

            if len(annotations) == 0:
                available_splits = sorted(annotations_original['split'].unique().tolist())
                logger.error(f"No annotations found for split '{split_label}'!")
                logger.error(f"Available splits in CSV: {available_splits}")
                raise ValueError(f"No data found for split '{split_label}'. Available: {available_splits}")

        if 0.0 < self.subsample_fraction < 1.0 and not annotations.empty:
            seed = int(getattr(self.cfg, 'RNG_SEED', 0))
            n_before = len(annotations)
            by_parent = bool(getattr(self.cfg.DATA, 'SUBSAMPLE_BY_PARENT_TASK', False))
            if by_parent and 'parent_task_name' in annotations.columns:
                def _sample_group(df: pd.DataFrame) -> pd.DataFrame:
                    n_keep = max(1, int(math.ceil(len(df) * self.subsample_fraction)))
                    return df.sample(n=n_keep, random_state=seed)
                annotations = (
                    annotations.groupby('parent_task_name', group_keys=False)
                    .apply(_sample_group)
                    .reset_index(drop=True)
                )
                logger.info(
                    f"Subsampled by parent_task_name from {n_before} to {len(annotations)} "
                    f"({self.subsample_fraction*100:.1f}%)"
                )
            else:
                annotations = annotations.sample(frac=self.subsample_fraction, random_state=seed).reset_index(drop=True)
                logger.info(f"Subsampled from {n_before} to {len(annotations)} ({self.subsample_fraction*100:.1f}%)")

            if bool(getattr(self.cfg.DATA, 'SUBSAMPLE_WRITE_CSV', False)):
                base_path = getattr(self.cfg.DATA, 'SPLIT_ASSIGNMENTS_CSV', None)
                save_path = getattr(self.cfg.DATA, 'SUBSAMPLE_SAVE_CSV_PATH', None)
                if not save_path:
                    suffix = "by_parent_task" if by_parent else "random"
                    if base_path:
                        p = Path(base_path)
                        save_path = str(p.with_suffix(f".subsample_{self.split_name}_{suffix}_{self.subsample_fraction:.3f}.csv"))
                    else:
                        save_path = f"subsample_{self.split_name}_{suffix}_{self.subsample_fraction:.3f}.csv"
                try:
                    annotations.to_csv(save_path, index=False)
                    logger.info(f"Saved subsampled annotations CSV to: {save_path}")
                except Exception as e:
                    logger.warning(f"Failed to save subsampled CSV to {save_path}: {e}")

        return annotations

    def _init_caches(self):
        """Initialize in-memory caches for frequently accessed data."""
        self._take_ts_cache: Dict[str, np.ndarray] = {}
        self._take_aria_cache: Dict[str, int] = {}
        self._gaze_cache: Dict[str, Dict[str, Any]] = {}
        self._frame_hw_cache: Dict[str, Tuple[int, int]] = {}
        self.skip_index_cache_load = os.environ.get("EGOEXO_GAZE_SKIP_INDEX_CACHE_LOAD", "0") == "1" or getattr(self.cfg.DATA, 'SKIP_INDEX_CACHE_LOAD', False)
        self.skip_take_cache_load = os.environ.get("EGOEXO_GAZE_SKIP_TAKE_CACHE_LOAD", "0") == "1" or getattr(self.cfg.DATA, 'SKIP_TAKE_CACHE_LOAD', False)

    def _get_annotations_path(self, cfg, split: str) -> str:
        if hasattr(cfg.DATA, 'SPLIT_ASSIGNMENTS_CSV') and cfg.DATA.SPLIT_ASSIGNMENTS_CSV:
            return cfg.DATA.SPLIT_ASSIGNMENTS_CSV
        return {
            "train": cfg.DATA.PATH_TO_DATA_DIR,
            "val": cfg.DATA.PATH_TO_VAL_DIR,
            "test": cfg.DATA.PATH_TO_TEST_DIR,
        }[split]

    def _video_to_frames_dir(self, video_path: str) -> Path:
        """
        Convert video path to frames directory path.
        Handles cases where video_path and frames are on different mount points.

        Example:
            video: /mnt/data3/.../takes/cmu_bike01_7/frame_aligned_videos/aria01_214-1.mp4
            frames: /mnt/data1/.../takes_frames/cmu_bike01_7/frame_aligned_videos/aria01_214-1/
        """
        p = Path(video_path)

        # Extract the relative path after 'takes' directory
        parts = p.parts
        if 'takes' in parts:
            idx = parts.index('takes')
            # Get path after 'takes': e.g., cmu_bike01_7/frame_aligned_videos/aria01_214-1.mp4
            rel_parts = parts[idx + 1:]
            rel_path = Path(*rel_parts).with_suffix('')  # Remove .mp4
        else:
            # Fallback: try relative_to if both paths share the same root
            try:
                rel_path = p.relative_to(self.video_root_dir).with_suffix('')
            except ValueError:
                # If relative_to fails, just use video stem as last resort
                rel_path = Path(p.stem)

        return self.frames_root_dir / rel_path

    def _ensure_take_context(self, video_path: str, frames_dir: Path):
        _, take_name = find_parts_after_takes(Path(video_path))
        ego_exo_root = Path(self.cfg.DATA.VIDEO_ROOT_DIR) 
        
        cache_path = self.cache_dir / f"{take_name}.npz"
        if not self.skip_take_cache_load and cache_path.exists():
            try:
                with np.load(cache_path, allow_pickle=False) as dat:
                    self._frame_hw_cache[take_name] = (int(dat['frame_h']), int(dat['frame_w']))
                    self._take_aria_cache[take_name] = int(dat['aria_num'])
                    self._take_ts_cache[take_name] = dat['take_ts_ns']
                    self._gaze_cache[take_name] = {'ts': dat['gaze_ts'], 'x': dat['gaze_x'], 'y': dat['gaze_y'], 'fmt': str(dat['gaze_fmt'])}
                    return take_name, ego_exo_root, self._take_ts_cache[take_name], self._gaze_cache[take_name]['ts'], np.stack([self._gaze_cache[take_name]['x'], self._gaze_cache[take_name]['y']], axis=0), self._frame_hw_cache[take_name], get_video_wh(video_path), self._gaze_cache[take_name]['fmt']
            except Exception:
                pass

        if take_name not in self._frame_hw_cache: self._frame_hw_cache[take_name] = first_frame_hw(frames_dir, default_hw=(1080, 1920))
        if take_name not in self._take_aria_cache:
            aria_num = find_aria_number(ego_exo_root, take_name) or infer_aria_number_from_timesync(ego_exo_root, take_name)
            if aria_num is None: raise FileNotFoundError(f"Cannot resolve aria number for take '{take_name}'")
            self._take_aria_cache[take_name] = aria_num
        if take_name not in self._take_ts_cache: self._take_ts_cache[take_name] = load_take_frame_timestamps_ns(ego_exo_root, take_name, self._take_aria_cache[take_name])

        if take_name not in self._gaze_cache:
            gdf = load_gaze_df_for_take_cached(ego_exo_root, take_name, self.gaze_data_dir)
            if gdf is None or gdf.empty:
                self._gaze_cache[take_name] = {'ts': np.array([], dtype=np.int64), 'x': np.array([], dtype=np.float32), 'y': np.array([], dtype=np.float32), 'fmt': 'unknown'}
            else:
                self._gaze_cache[take_name] = {
                    'ts': gdf['timestamp_ns'].to_numpy(np.int64),
                    'x': gdf['x'].to_numpy(np.float32),
                    'y': gdf['y'].to_numpy(np.float32),
                    'fmt': 'pixels' # Simplified for brevity
                }

        context = self._gaze_cache[take_name]
        if context['ts'].size > 0:
            try:
                # Save gaze and frame timestamp arrays to per-take cache
                np.savez(
                    cache_path,
                    frame_h=self._frame_hw_cache[take_name][0],
                    frame_w=self._frame_hw_cache[take_name][1],
                    aria_num=self._take_aria_cache[take_name],
                    take_ts_ns=self._take_ts_cache[take_name],
                    gaze_ts=context['ts'],
                    gaze_x=context['x'],
                    gaze_y=context['y'],
                    gaze_fmt=context['fmt'],
                )
            except Exception:
                pass

        return take_name, ego_exo_root, self._take_ts_cache[take_name], context['ts'], np.stack([context['x'], context['y']]), self._frame_hw_cache[take_name], get_video_wh(video_path), context['fmt']

    def _precompute_samples(self):
        from tqdm import tqdm
        for _, row in tqdm(self.annotations.iterrows(), total=len(self.annotations), desc="Indexing samples"):
            self._process_row(row)
        logger.info(f"Prepared {len(self.samples)} samples from {len(self.annotations)} clips.")

    def _process_row(self, row: pd.Series):
        video_path = row['video_path']
        # frames_dir = Path(self.cfg.DATA.FRAMES_DIR)  # Wrong: uses same dir for all videos
        frames_dir = self._video_to_frames_dir(video_path)
        if not frames_dir.exists():
            if self.skip_missing_frames_dir: return
            raise FileNotFoundError(f"Frames dir not found: {frames_dir}")

        take_name, ego_exo_root, take_ts_ns, gaze_ts, gaze_xy, frame_hw, video_wh, gaze_fmt = self._ensure_take_context(video_path, frames_dir)
        if gaze_ts.size == 0: return

        fps_native = 1e9 / np.median(np.diff(take_ts_ns)) if take_ts_ns.size > 1 else 30.0
        start_idx = max(0, min(int(round(row['clip_start_time'] * fps_native)), len(take_ts_ns) - 1))
        end_idx = max(0, min(int(round(row['clip_end_time'] * fps_native)), len(take_ts_ns)))
        
        native_indices = list(range(start_idx, end_idx))
        stride = max(1, int(round(fps_native / self.target_fps))) if fps_native > self.target_fps else 1
        clip_indices = native_indices[::stride]

        if len(clip_indices) < self.sequence_length: return

        clip_gaze = self._gaze_for_frames_via_ts(clip_indices, take_ts_ns, gaze_ts, gaze_xy, video_wh)
        
        for s in range(0, len(clip_indices) - self.sequence_length + 1, self.window_stride):
            window_indices = clip_indices[s:s + self.sequence_length]
            window_gaze = self._process_gaze_window(clip_gaze[s:s + self.sequence_length])
            
            sample = {
                'frames_dir': frames_dir,
                'frame_indices': window_indices,
                'frame_ts_ns': take_ts_ns[window_indices].astype(np.int64),
                'take_name': take_name,
                'ego_exo_root': str(ego_exo_root),
                'gaze_sequence': [g[:2] for g in window_gaze],
                'gaze_label': window_gaze,
                'task_name': row.get('task_name', ''),
                'parent_task_name': row.get('parent_task_name', ''),
            }

            self.samples.append(sample)

    def _process_gaze_window(self, window_gaze_norm):
        if self.interpolate_nan:
            valid_gaze = [(i, g) for i, g in enumerate(window_gaze_norm) if not any(math.isnan(c) for c in g)]
            if valid_gaze:
                valid_indices, valid_points = zip(*valid_gaze)
                for i, (x, y) in enumerate(window_gaze_norm):
                    if math.isnan(x):
                        j = min(valid_indices, key=lambda k: abs(k - i))
                        window_gaze_norm[i] = window_gaze_norm[j]
        
        labels = []
        for gx, gy in window_gaze_norm:
            is_nan = math.isnan(gx) or math.isnan(gy)
            if is_nan and self.fill_nan_center:
                labels.append([0.5, 0.5, 0])
            elif is_nan:
                labels.append([float('nan'), float('nan'), 0])
            else:
                labels.append([gx, gy, 1])
        return labels

    def _gaze_for_frames_via_ts(self, frame_indices, take_ts_ns, gaze_ts_ns, gaze_xy, video_wh):
        half_frame_ns = 0.5 * (1e9 / self.target_fps)
        order = np.argsort(gaze_ts_ns)
        gts, gx, gy = gaze_ts_ns[order], gaze_xy[0][order], gaze_xy[1][order]
        
        frame_timestamps = take_ts_ns[frame_indices]
        positions = np.searchsorted(gts, frame_timestamps)
        
        out = []
        for i, pos in enumerate(positions):
            ts = frame_timestamps[i]
            cand_indices = [c for c in [pos - 1, pos, pos + 1] if 0 <= c < len(gts)]
            if not cand_indices:
                out.append((float('nan'), float('nan')))
                continue
            
            best_idx = min(cand_indices, key=lambda idx: abs(gts[idx] - ts))
            
            if abs(gts[best_idx] - ts) <= half_frame_ns:
                x_norm = np.clip(gx[best_idx] / max(1, video_wh[0]), 0.0, 1.0)
                y_norm = np.clip(gy[best_idx] / max(1, video_wh[1]), 0.0, 1.0)
                out.append((x_norm, y_norm))
            else:
                out.append((float('nan'), float('nan')))
        return out
    
    def _index_signature(self):
        """Generate a unique signature for cache invalidation."""
        return {
            'split': self.split_name,
            'csv_path': str(getattr(self, 'annotations_path', '')),
            'csv_mtime': os.path.getmtime(getattr(self, 'annotations_path', '')) if getattr(self, 'annotations_path', '') else 0,
            'sequence_length': self.sequence_length,
            'target_fps': self.target_fps,
            'subsample_fraction': self.subsample_fraction,
        }

    def _cache_file_path(self):
        sig_hash = hashlib.md5(json.dumps(self._index_signature(), sort_keys=True).encode()).hexdigest()
        return self.cache_dir / f"samples_{self.split_name}_{sig_hash}.pt"

    def _maybe_load_index_cache(self):
        if self.skip_index_cache_load:
            logger.info("Skipping index cache load (SKIP_INDEX_CACHE_LOAD or env override).")
            return False
        p = self._cache_file_path()
        if not p.exists(): return False
        try:
            obj = torch.load(str(p), map_location='cpu')
            if obj['signature'] == self._index_signature():
                self.samples = obj['samples']
                logger.info(f"Loaded {len(self.samples)} samples from cache: {p.name}")
                return True
        except Exception:
            pass
        return False

    def _save_index_cache(self):
        p = self._cache_file_path()
        try:
            torch.save({'signature': self._index_signature(), 'samples': self.samples}, p)
            logger.info(f"Saved sample index cache with {len(self.samples)} samples to {p.name}")
        except Exception as e:
            if _DEBUG: logger.warning(f"Failed to save index cache: {e}")

    def _is_rank0(self):
        try:
            return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
        except Exception:
            return True

    def _barrier_if_dist(self):
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            frames_tensor: [C, T, H, W] RGB frames normalized to [0, 1]
            gaze_tensor: [T, 3] with (x_norm, y_norm, visibility)
            labels_hm: [T, Hh, Wh] Gaussian heatmaps
        """
        s = self.samples[idx]
        frames = self._load_frames(s['frames_dir'], s['frame_indices'])
        frames_tensor = torch.from_numpy(np.stack(frames).transpose(3, 0, 1, 2)).float() / 255.0
        gaze_tensor = torch.tensor(s['gaze_label'], dtype=torch.float32)

        # Generate Gaussian heatmaps
        # TODO: Make heatmap size configurable (currently hardcoded to 64x64)
        heatmap_size = (64, 64)
        labels_hm = create_sequence_heatmaps(s['gaze_sequence'], heatmap_size, self.heatmap_sigma)

        if self.visualize:
            self._maybe_save_preview(frames, s['gaze_sequence'], labels_hm, s.get('parent_task_name', ''))

        return frames_tensor, gaze_tensor, labels_hm

    def _load_frames(self, frames_dir, frame_indices):
        frames = []
        placeholder = np.zeros((*self.image_size, 3), dtype=np.uint8)
        for fi in frame_indices:
            fpath = frames_dir / f"frame_{fi:05d}.jpg"
            img = cv2.imread(str(fpath))
            if img is None:
                frames.append(placeholder)
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[:2] != self.image_size:
                img = cv2.resize(img, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_AREA)
            frames.append(img)
        return frames

    def _maybe_save_preview(self, frames, gaze_seq, labels_hm, parent_task_name: str):
        if not self.vis_dir or not frames: return
        mid = len(frames) // 2
        img, gaze, hm = frames[mid].copy(), gaze_seq[mid], labels_hm[mid].cpu().numpy()
        h, w = img.shape[:2]
        
        hm_resized = cv2.resize(hm, (w, h))
        hm_color = cv2.applyColorMap((255 * (hm_resized - hm_resized.min()) / (hm_resized.max() - hm_resized.min() + 1e-6)).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB), 0.4, img, 0.6, 0)
        
        if not any(math.isnan(c) for c in gaze):
            px, py = int(round(gaze[0] * w)), int(round(gaze[1] * h))
            # GT gaze: red
            cv2.circle(overlay, (px, py), 6, (255, 0, 0), 2)
            cv2.circle(overlay, (px, py), 3, (0, 0, 0), -1)

        # Mark heatmap argmax (high point)
        try:
            hm_idx = int(np.nanargmax(hm))
            hm_h, hm_w = hm.shape
            hm_y, hm_x = divmod(hm_idx, hm_w)
            px_hm = int(round(hm_x / max(1, hm_w - 1) * (w - 1)))
            py_hm = int(round(hm_y / max(1, hm_h - 1) * (h - 1)))
            # Heatmap argmax: green
            cv2.circle(overlay, (px_hm, py_hm), 6, (0, 255, 0), 2)
            cv2.circle(overlay, (px_hm, py_hm), 3, (0, 0, 0), -1)
        except Exception:
            pass

        # Create parent task subdir if provided
        safe_parent = (parent_task_name or "unknown").strip()
        safe_parent = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in safe_parent)
        out_dir = self.vis_dir / safe_parent
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"preview_{os.getpid()}_{np.random.randint(1e9)}.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

if DATASET_REGISTRY:
    DATASET_REGISTRY.register()(EgoExoGazeClipDataset)


import math
from typing import Tuple, List, Optional

import torch

# Cache coordinate grids per (H, W, dtype, device) to avoid rebuilding every call
_COORD_CACHE = {}


def _coords(H: int, W: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    key = (H, W, dtype, device)
    cached = _COORD_CACHE.get(key)
    if cached is not None:
        return cached
    y_coords = torch.arange(H, dtype=dtype, device=device)
    x_coords = torch.arange(W, dtype=dtype, device=device)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')  # [H,W], [H,W]
    coords = torch.stack([xx, yy], dim=-1)  # [H,W,2]
    _COORD_CACHE[key] = coords
    return coords


def create_sequence_heatmaps(
    gaze_sequence: List[Tuple[float, float]],
    spatial_size: Tuple[int, int],
    sigma: Optional[float] = None
) -> torch.Tensor:
    """
    Build a stack of Gaussian heatmaps from normalized gaze points (x_norm, y_norm) in [0,1].
    NaNs are placed at (0.5, 0.5) (center).
    Returns tensor of shape [T, H, W].
    """
    H, W = spatial_size
    T = len(gaze_sequence)
    sigma = sigma if sigma is not None else min(H, W) * 0.05

    pts = []
    for x, y in gaze_sequence:
        if x is None or y is None or math.isnan(x) or math.isnan(y):
            pts.append((0.5, 0.5))
        else:
            pts.append((float(x), float(y)))
    device = torch.device('cpu')
    dtype = torch.float32
    gaze_points = torch.tensor(pts, dtype=dtype, device=device)  # [T,2]
    gaze_pixels = gaze_points * torch.tensor([W, H], dtype=dtype, device=device)  # [T,2]
    coords = _coords(H, W, dtype, device)  # [H,W,2]

    # (T,1,1,2) - (1,H,W,2) -> (T,H,W,2)
    dist_sq = torch.sum((gaze_pixels.view(T, 1, 1, 2) - coords.unsqueeze(0)) ** 2, dim=-1)
    heatmaps = torch.exp(-dist_sq / (2 * (sigma ** 2)))

    max_vals = torch.amax(heatmaps, dim=(1, 2), keepdim=True)
    max_vals[max_vals == 0] = 1.0
    return heatmaps / max_vals


import os
import re
import json
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from threading import Lock

import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Global per-worker cache for gaze DataFrames
# This avoids repeated CSV parsing for samples from the same take
_GAZE_DF_CACHE: Dict[Tuple, Optional[pd.DataFrame]] = {}
_GAZE_DF_CACHE_LOCK = Lock()


def find_parts_after_takes(p: Path) -> Tuple[Path, str]:
    parts = p.parts
    if 'takes' not in parts:
        raise ValueError(f"'takes' not found in path: {p}")
    idx = parts.index('takes')
    take_name = parts[idx + 1]
    ego_exo_root = Path(*parts[:idx])
    return ego_exo_root, take_name


def find_aria_number(ego_exo_root: Path, take_name: str) -> Optional[int]:
    vrs_dir = ego_exo_root / 'imu_data' / 'takes' / take_name
    if not vrs_dir.exists():
        return None
    pat = re.compile(r'aria0(\d+).*\.vrs')
    for root, _, files in os.walk(vrs_dir):
        for f in files:
            m = pat.match(f)
            if m:
                return int(m.group(1))
    return None


def infer_aria_number_from_timesync(ego_exo_root: Path, take_name: str) -> Optional[int]:
    """Infer an ARIA number by scanning the timesync.csv headers.

    Looks for columns like 'aria0XX_...timestamp_ns' and returns the first match.
    Returns None if timesync not found or no matching columns.
    """
    try:
        timesync_path, _, _ = resolve_timesync_window(ego_exo_root, take_name)
    except Exception:
        return None
    try:
        df = pd.read_csv(timesync_path, nrows=1)
    except Exception:
        return None
    pat = re.compile(r"aria0(\d+).*(timestamp_ns|_ts|timestamp)", re.IGNORECASE)
    for c in df.columns:
        m = pat.search(c)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                continue
    return None


def load_takes_json(ego_exo_root: Path) -> List[dict]:
    tj = ego_exo_root / 'takes.json'
    if not tj.exists():
        raise FileNotFoundError(f"takes.json not found at {tj}")
    return json.loads(tj.read_text())


def resolve_timesync_window(
    ego_exo_root: Path,
    take_name: str
) -> Tuple[Path, int, int]:
    takes = load_takes_json(ego_exo_root)
    rec = next((t for t in takes if t.get('take_name') == take_name), None)
    if rec is None:
        raise FileNotFoundError(f"take '{take_name}' not found in takes.json at {ego_exo_root}")

    capture_name = re.sub(r"_\d+$", "", take_name)
    timesync_path = ego_exo_root / 'captures'  / capture_name / 'timesync.csv'
    if not timesync_path.exists():
        raise FileNotFoundError(f"timesync.csv not found: {timesync_path}")

    start_idx = int(rec.get('timesync_start_idx', 0)) + 1
    end_idx = int(rec.get('timesync_end_idx', 0))
    return timesync_path, start_idx, end_idx


def load_take_frame_timestamps_ns(
    ego_exo_root: Path,
    take_name: str,
    aria_num: int
) -> np.ndarray:
    timesync_path, start_idx, end_idx = resolve_timesync_window(ego_exo_root, take_name)
    df = pd.read_csv(timesync_path, low_memory=False)

    col = f"aria0{aria_num}_214-1_capture_timestamp_ns"
    if col not in df.columns:
        candidates = [c for c in df.columns if f"aria0{aria_num}" in c and 'timestamp_ns' in c]
        if not candidates:
            raise KeyError(f"No timestamp_ns column for aria0{aria_num} in {timesync_path}")
        col = candidates[0]

    ts = df[col].astype('Int64')
    ts = ts.iloc[start_idx:min(end_idx, len(df))].dropna().astype(np.int64).to_numpy()
    if ts.size == 0:
        raise ValueError(f"No timestamps extracted for {take_name} using [{start_idx}, {end_idx})")
    return ts


def load_gaze_df_for_take(ego_exo_root: Path, take_name: str, gaze_data_root: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """Load gaze CSV for a take (original uncached version).

    This function reads and parses the CSV file every time it's called.
    For better performance, use load_gaze_df_for_take_cached() instead.
    """
    base_root = gaze_data_root if gaze_data_root is not None else (ego_exo_root / 'gaze_data')
    base = base_root / 'takes' / take_name / 'eye_gaze'
    for gtype in ['personalized', 'general']:
        path = base / f"{gtype}_eye_gaze_2d.csv"
        if path.exists():
            df = pd.read_csv(path)
            ts_col = None
            if 'timestamp_ns' in df.columns:
                ts_col = 'timestamp_ns'
            else:
                cand = [c for c in df.columns if 'timestamp' in c.lower()]
                if cand:
                    ts_col = cand[0]
            if ts_col is None or 'x' not in df.columns or 'y' not in df.columns:
                logger.warning(f"Gaze CSV missing required cols at {path} (need timestamp & x/y).")
                return None
            out = pd.DataFrame()
            out['x'] = df['x']
            out['y'] = df['y']
            col_lower = ts_col.lower()
            vals = df[ts_col].astype('float64').to_numpy()
            if col_lower.endswith('_ns') or col_lower == 'timestamp_ns':
                ts_ns = vals
            elif col_lower.endswith('_us') or 'micro' in col_lower:
                ts_ns = vals * 1_000.0
            elif col_lower.endswith('_ms'):
                ts_ns = vals * 1_000_000.0
            else:
                ts_ns = vals * 1_000_000_000.0
            out['timestamp_ns'] = ts_ns.astype(np.int64)
            return out
    return None


def load_gaze_df_for_take_cached(ego_exo_root: Path, take_name: str, gaze_data_root: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """Load gaze CSV for a take with per-worker caching.

    This function caches parsed DataFrames in memory to avoid repeated CSV
    reading and parsing for samples from the same take. Each DataLoader worker
    maintains its own cache.

    Performance: ~3-5x faster for large datasets with repeated take access.

    Args:
        ego_exo_root: Root directory of EgoExo4D dataset
        take_name: Name of the take (e.g., 'cmu_bike01_1')
        gaze_data_root: Optional override for gaze data directory

    Returns:
        DataFrame with columns ['x', 'y', 'timestamp_ns'] or None if not found
    """
    # Create cache key from input parameters
    cache_key = (str(ego_exo_root), take_name, str(gaze_data_root) if gaze_data_root else None)

    # Check cache first (thread-safe)
    with _GAZE_DF_CACHE_LOCK:
        if cache_key in _GAZE_DF_CACHE:
            return _GAZE_DF_CACHE[cache_key]

    # Cache miss - load from disk using original function
    df = load_gaze_df_for_take(ego_exo_root, take_name, gaze_data_root)

    # Store in cache (thread-safe)
    with _GAZE_DF_CACHE_LOCK:
        _GAZE_DF_CACHE[cache_key] = df

    return df

def load_gaze3d_df_for_take(ego_exo_root: Path, take_name: str, gaze_data_root: Optional[Path] = None) -> Optional[pd.DataFrame]:
    base_root = gaze_data_root if gaze_data_root is not None else (ego_exo_root / 'gaze_data')
    base = base_root / 'takes' / take_name / 'eye_gaze'
    for gtype in ['personalized', 'general']:
        path = base / f"{gtype}_eye_gaze.csv"
        if path.exists():
            df = pd.read_csv(path)
            ts_col = None
            if 'tracking_timestamp_ns' in df.columns:
                ts_col = 'tracking_timestamp_ns'
            else:
                cand = [c for c in df.columns if 'timestamp' in c.lower()]
                if cand:
                    ts_col = cand[0]
            if ts_col is None or 'left_yaw' not in df.columns or 'right_yaw' not in df.columns or 'pitch' not in df.columns:
                logger.warning(f"Gaze CSV missing required cols at {path} (need timestamp & left_yaw/right_yaw/pitch).")
                return None
            out = pd.DataFrame()
            out['left_yaw'] = df['left_yaw']
            out['right_yaw'] = df['right_yaw']
            out['pitch'] = df['pitch']
            out['depth'] = df['depth']
            col_lower = ts_col.lower()
            vals = df[ts_col].astype('float64').to_numpy()
            if col_lower.endswith('_ns') or col_lower == 'tracking_timestamp_ns':
                ts_ns = vals
            elif col_lower.endswith('_us') or 'micro' in col_lower:
                ts_ns = vals * 1_000.0
            elif col_lower.endswith('_ms'):
                ts_ns = vals * 1_000_000.0
            else:
                ts_ns = vals * 1_000_000_000.0
            out['tracking_timestamp_ns'] = ts_ns.astype(np.int64)
            return out
    return None

def first_frame_hw(frames_dir: Path, default_hw: Tuple[int, int]) -> Tuple[int, int]:
    imgs = sorted(frames_dir.glob("frame_*.jpg"))
    if not imgs:
        return default_hw
    img = cv2.imread(str(imgs[0]))
    if img is None:
        return default_hw
    h, w = img.shape[:2]
    return h, w


def get_video_wh(video_path: str) -> Tuple[int, int]:
    """
    Return video resolution for gaze coordinate normalization.

    NOTE: This function returns a hardcoded resolution and does NOT
    require the actual video file to exist. The video_path parameter
    is kept for API compatibility but is not used.

    This allows the dataset to work with only extracted frames,
    saving disk space by not requiring the original video files.
    """
    # Hardcoded resolution for EgoExo4D (1404x1404)
    # This is the resolution used for gaze coordinate normalization
    w, h = (1404, 1404)
    return (w, h)

    

@DATASET_REGISTRY.register()
class Egoexo4dgaze(torch.utils.data.Dataset):
    """
    SlowFast dataset wrapper for Ego-Exo4D gaze estimation.

    Provides a unified API consistent with Egteagaze, Ego4dgaze, and Holoassistgaze datasets.
    Returns: (frames, labels_xyv, labels_hm, index, extra_data)
    """

    def __init__(self, cfg, mode, num_retries: int = 0):
        """
        Args:
            cfg (CfgNode): Configuration node
            mode (str): Dataset split - "train", "val", "test", or test variants
            num_retries (int): Number of retries for failed samples (unused, for API consistency)
        """
        # Map "test" to "test_iid" for backward compatibility with other datasets
        original_mode = mode
        if mode == "test":
            mode = "test_iid"
            logger.info(f"Mode '{original_mode}' mapped to '{mode}' for EgoExo4D dataset")

        assert mode in ["train", "val", "test_iid", "test_ood_site", "test_ood_task", "test_ood_participant"], \
            f"Unsupported split: {mode}"
        self.cfg = cfg
        self.mode = mode
        self._num_retries = int(num_retries)

        logger.info(f"=" * 80)
        logger.info(f"USING TEST SPLIT: {mode}")
        logger.info(f"=" * 80)
        logger.info(f"Constructing EgoExo4D gaze dataset ({mode})...")
        # Core dataset handles indexing, caching, frame/gaze/heatmap loading
        self.base = EgoExoGazeClipDataset(cfg, split=mode)

    def __len__(self):
        return len(self.base)

    @property
    def num_videos(self):
        """For API consistency with other gaze datasets."""
        return len(self.base)

    def __getitem__(self, index: int):
        """
        Returns:
            frames: List of pathway tensors [C, T, H, W] (single pathway for MViT)
            labels_xyv: Tensor [T, 3] with (x_norm, y_norm, visibility)
            labels_hm: Tensor [T, Hh, Wh] with Gaussian heatmaps
            index: Sample index
            extra_data: Dict with {'path': str, 'index': np.ndarray of frame indices}
        """
        # Get data from base dataset (now returns 3 values)
        frames_tensor, gaze_tensor, labels_hm = self.base[index]

        # Pack into pathway output (single pathway for MViT/backbones)
        frames = utils.pack_pathway_output(self.cfg, frames_tensor)

        # Build extra metadata dict matching other datasets' format
        extra = {'path': '', 'index': np.array([])}
        try:
            s = self.base.samples[index]
            # Construct path string similar to other datasets
            video_path = f"{s.get('take_name', 'unknown')}"
            frame_indices = s.get('frame_indices', [])
            extra = {
                'path': video_path,
                'index': np.array(frame_indices, dtype=np.int64) if frame_indices else np.array([]),
                'task_name': s.get('task_name', ''),
                'parent_task_name': s.get('parent_task_name', ''),
            }
        except Exception:
            pass

        return frames, gaze_tensor, labels_hm, index, extra
