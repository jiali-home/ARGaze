"""Thin shim that re-exports the split gaze loader and dataset.

This keeps backward-compat imports while moving implementation into
the `gaze/` package so this file stays small and readable.
"""

import os
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

# Public API re-exports (point to local package)
from gaze_dataloader.heatmap import create_sequence_heatmaps
from gaze_dataloader.dataset import EgoExoGazeClipDataset as _EgoExoGazeClipDataset
from gaze_dataloader.loader import build_gaze_dataloader

# Backward-compat alias (original class name)
Egoexogazeclipdataset = _EgoExoGazeClipDataset
