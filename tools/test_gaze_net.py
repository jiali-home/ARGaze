#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""
import csv
import numpy as np
import os
import pickle
import time
import torch
import torch.nn.functional as F
import cv2
from tqdm import tqdm

import sys
sys.path.append("/mnt/data1/jiali/GLC")

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.utils.metrics as metrics
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.env import pathmgr
from slowfast.utils.meters import AVAMeter, TestMeter, TestGazeMeter
from slowfast.utils.utils import frame_softmax
from slowfast.utils import baseline_utils

logger = logging.get_logger(__name__)


def _overlay_heatmap_bgr(frame_bgr, heatmap, alpha=0.4):
    """Overlay a heatmap (H, W) onto a BGR frame."""
    h, w = frame_bgr.shape[:2]
    hm_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    hm_min, hm_max = float(hm_resized.min()), float(hm_resized.max())
    if hm_max > hm_min:
        hm_norm = ((hm_resized - hm_min) / (hm_max - hm_min) * 255.0).astype(np.uint8)
    else:
        hm_norm = np.zeros((h, w), dtype=np.uint8)
    hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame_bgr, 1.0, hm_color, float(alpha), 0)


def _draw_gaze_marker_bgr(frame_bgr, x, y, color, radius=6, thickness=2):
    """Draw a normalized (x, y) gaze marker on a BGR frame."""
    h, w = frame_bgr.shape[:2]
    px = int(round(x * (w - 1)))
    py = int(round(y * (h - 1)))
    cv2.circle(frame_bgr, (px, py), radius, color, thickness)
    cv2.line(frame_bgr, (px - radius - 3, py), (px + radius + 3, py), color, thickness)
    cv2.line(frame_bgr, (px, py - radius - 3), (px, py + radius + 3), color, thickness)
    return frame_bgr


def _load_frame_image(frames_dir, video_name, frame_idx, frame_ext):
    """Load a frame image from disk, returns BGR or None."""
    if not frames_dir or not video_name:
        return None
    candidates = [
        os.path.join(frames_dir, video_name, f"{int(frame_idx):06d}.{frame_ext}"),
        os.path.join(frames_dir, video_name, f"frame_{int(frame_idx):010d}.{frame_ext}"),
        os.path.join(frames_dir, video_name, f"{int(frame_idx):010d}.{frame_ext}"),
        os.path.join(frames_dir, video_name, f"{int(frame_idx):05d}.{frame_ext}"),
        os.path.join(frames_dir, video_name, f"{int(frame_idx):08d}.{frame_ext}"),
        os.path.join(frames_dir, video_name, f"{int(frame_idx)}.{frame_ext}"),
        os.path.join(frames_dir, video_name, f"img_{int(frame_idx):06d}.{frame_ext}"),
        os.path.join(frames_dir, video_name, f"{int(frame_idx)+1:06d}.{frame_ext}"),
    ]
    for img_path in candidates:
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                return img
    return None


def _get_video_name(video_path):
    if not video_path:
        return ""
    return os.path.basename(os.path.dirname(video_path))


def _get_valid_frame_mask(labels, dataset, edge_threshold):
    """Return a (B, T) boolean mask of valid frames."""
    if labels.size(-1) > 3:
        labels = labels[..., :3]
    labels_flat = labels.view(labels.size(0) * labels.size(1), labels.size(2))
    dataset_name = (dataset or "").lower()
    if dataset_name in ["holoassistgaze", "egoexo4dgaze"]:
        tracked_idx = torch.where(labels_flat[:, 2] >= 0.5)[0]
    else:
        fixation_idx = 1 if dataset_name == "egteagaze" else 0
        tracked_idx = torch.where(labels_flat[:, 2] == fixation_idx)[0]
    if tracked_idx.numel() == 0:
        return torch.zeros(labels_flat.size(0), dtype=torch.bool).view(labels.size(0), labels.size(1))
    if edge_threshold > 0.0:
        labels_tracked = labels_flat.index_select(0, tracked_idx)
        edge_valid_idx = metrics.filter_edge_frames(labels_tracked, edge_threshold)
        tracked_idx = tracked_idx[edge_valid_idx]
    valid_mask_flat = torch.zeros(labels_flat.size(0), dtype=torch.bool)
    valid_mask_flat[tracked_idx] = True
    return valid_mask_flat.view(labels.size(0), labels.size(1))


def _heatmap_to_coords(hm, mode="argmax"):
    """Convert heatmap to normalized (x, y)."""
    H, W = hm.shape[-2], hm.shape[-1]
    mode = (mode or "argmax").lower()
    if mode == "expectation":
        weights = torch.nan_to_num(hm, nan=0.0, posinf=0.0, neginf=0.0)
        total = float(weights.sum())
        if total <= 0.0:
            return 0.5, 0.5
        ys = torch.arange(H, dtype=torch.float64)
        xs = torch.arange(W, dtype=torch.float64)
        wy = weights.double().sum(dim=1)
        wx = weights.double().sum(dim=0)
        pred_y = float((wy * ys).sum() / total)
        pred_x = float((wx * xs).sum() / total)
        pred_x = pred_x / (W - 1) if W > 1 else 0.5
        pred_y = pred_y / (H - 1) if H > 1 else 0.5
        return pred_x, pred_y
    flat_idx = int(torch.nan_to_num(hm, nan=-1e6).argmax().item())
    row = flat_idx // W
    col = flat_idx % W
    pred_x = col / (W - 1) if W > 1 else 0.5
    pred_y = row / (H - 1) if H > 1 else 0.5
    return pred_x, pred_y


def _compute_per_frame_stats(preds, labels_hm, labels, dataset, edge_threshold, l2_mode, threshold):
    """Compute per-frame metrics and predicted coordinates."""
    if labels.size(-1) > 3:
        labels = labels[..., :3]
    preds_sw = preds.squeeze(1)
    B, T, Hp, Wp = preds_sw.shape
    Hl, Wl = labels_hm.size(-2), labels_hm.size(-1)
    if (Hp, Wp) != (Hl, Wl):
        preds_sw = F.interpolate(
            preds_sw.view(B * T, 1, Hp, Wp),
            size=(Hl, Wl),
            mode="bilinear",
            align_corners=False,
        ).view(B, T, Hl, Wl)

    labels_hm_sw = labels_hm.squeeze(1)
    if labels_hm_sw.dim() == 5:
        labels_hm_sw = labels_hm_sw.squeeze(1)
    if labels_hm_sw.shape[-2:] != (Hl, Wl):
        labels_hm_sw = F.interpolate(
            labels_hm_sw.view(B * T, 1, labels_hm_sw.size(-2), labels_hm_sw.size(-1)),
            size=(Hl, Wl),
            mode="bilinear",
            align_corners=False,
        ).view(B, T, Hl, Wl)

    binary_preds = (preds_sw > float(threshold)).int()
    binary_labels = (labels_hm_sw > 0.001).int()
    tp = (binary_preds * binary_labels).sum(dim=(2, 3)).float()
    fg_labels = binary_labels.sum(dim=(2, 3)).float()
    fg_preds = binary_preds.sum(dim=(2, 3)).float()
    recall = tp / (fg_labels + 1e-6)
    precision = tp / (fg_preds + 1e-6)
    f1 = (2.0 * recall * precision) / (recall + precision + 1e-6)

    valid_mask = _get_valid_frame_mask(labels, dataset, edge_threshold)
    invalid_mask = ~valid_mask
    f1 = f1.masked_fill(invalid_mask, float("nan"))
    recall = recall.masked_fill(invalid_mask, float("nan"))
    precision = precision.masked_fill(invalid_mask, float("nan"))

    pred_x = torch.full((B, T), float("nan"))
    pred_y = torch.full((B, T), float("nan"))
    l2 = torch.full((B, T), float("nan"))
    for b in range(B):
        for t in range(T):
            if not valid_mask[b, t]:
                continue
            hm = preds_sw[b, t]
            px, py = _heatmap_to_coords(hm, mode=l2_mode)
            pred_x[b, t] = px
            pred_y[b, t] = py
            gt_x = float(labels[b, t, 0])
            gt_y = float(labels[b, t, 1])
            l2[b, t] = ((px - gt_x) ** 2 + (py - gt_y) ** 2) ** 0.5

    return {
        "f1": f1.numpy(),
        "recall": recall.numpy(),
        "precision": precision.numpy(),
        "l2": l2.numpy(),
        "pred_x": pred_x.numpy(),
        "pred_y": pred_y.numpy(),
        "valid": valid_mask.numpy(),
    }


@torch.no_grad()
def _pad_streaming_window(window, target_len, warm_start):
    """Pad temporal window for streaming inference during warm-up phase."""
    if window.size(2) >= target_len:
        return window
    pad_len = target_len - window.size(2)
    if warm_start == "replicate":
        pad_frame = window[:, :, :1].expand(-1, -1, pad_len, -1, -1)
    elif warm_start == "zeros":
        pad_frame = torch.zeros_like(window[:, :, :1]).expand(-1, -1, pad_len, -1, -1)
    else:
        raise ValueError(f"Unknown TEST.STREAMING_WARM_START: {warm_start}")
    return torch.cat([pad_frame, window], dim=2)


@torch.no_grad()
def _streaming_glc_forward(model, inputs, cfg, streaming_times=None):
    """
    Streaming inference for GLC models (GLC_Gaze, GLC_Gaze_Causal).
    Processes video frame-by-frame with sliding window for fair latency comparison.
    """
    if isinstance(inputs, (list, tuple)):
        if len(inputs) != 1:
            raise ValueError("Streaming GLC expects a single-pathway input.")
        input_tensor = inputs[0]
        wrap_input = True
    else:
        input_tensor = inputs
        wrap_input = False

    batch_size, channels, num_frames, height, width = input_tensor.shape
    if batch_size != 1:
        raise ValueError(
            f"Streaming test requires batch_size=1 for fair latency measurement, got {batch_size}. "
            "Set TEST.BATCH_SIZE 1 in config."
        )

    window_size = cfg.DATA.NUM_FRAMES
    warm_start = cfg.TEST.STREAMING_WARM_START

    warmup_frames = getattr(cfg.TEST, "STREAMING_METRICS_WARMUP", 0)
    record_timing = (
        getattr(cfg.TEST, "STREAMING_METRICS_ENABLE", False) and streaming_times is not None
    )

    # Pre-allocate output tensor for memory efficiency
    # First, get output shape from a single forward pass
    window = input_tensor[:, :, :1]
    window = _pad_streaming_window(window, window_size, warm_start)
    model_input = [window] if wrap_input else window
    if record_timing and 0 >= warmup_frames:
        if cfg.NUM_GPUS:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
    preds_t = model(model_input)
    if record_timing and 0 >= warmup_frames:
        if cfg.NUM_GPUS:
            torch.cuda.synchronize()
        streaming_times.append(time.perf_counter() - t0)
    if preds_t.dim() == 5:
        preds_t = preds_t[:, :, -1]  # [B, C, H, W]

    # Pre-allocate output: [B, C, T, H, W]
    preds = torch.empty(
        batch_size, preds_t.size(1), num_frames, preds_t.size(2), preds_t.size(3),
        dtype=preds_t.dtype, device=preds_t.device
    )
    preds[:, :, 0] = preds_t

    # Process remaining frames
    for t in range(1, num_frames):
        start = max(0, t - window_size + 1)
        window = input_tensor[:, :, start : t + 1]
        window = _pad_streaming_window(window, window_size, warm_start)
        model_input = [window] if wrap_input else window
        if record_timing and t >= warmup_frames:
            if cfg.NUM_GPUS:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
        preds_t = model(model_input)
        if record_timing and t >= warmup_frames:
            if cfg.NUM_GPUS:
                torch.cuda.synchronize()
            streaming_times.append(time.perf_counter() - t0)
        if preds_t.dim() == 5:
            preds_t = preds_t[:, :, -1]
        preds[:, :, t] = preds_t

    return preds


@torch.no_grad()
def _streaming_ar_template_forward(model, inputs, cfg, streaming_times=None):
    """
    Streaming inference for AR template models with per-frame encode + template crop.
    """
    if isinstance(inputs, (list, tuple)):
        if len(inputs) != 1:
            raise ValueError("Streaming AR template expects a single-pathway input.")
        input_tensor = inputs[0]
    else:
        input_tensor = inputs

    model_fn = model.module if hasattr(model, "module") else model
    batch_size, _, num_frames, _, _ = input_tensor.shape
    if batch_size != 1:
        raise ValueError(
            f"Streaming test requires batch_size=1 for fair latency measurement, got {batch_size}. "
            "Set TEST.BATCH_SIZE 1 in config."
        )

    pos_encoding = model_fn._get_2d_pos_encoding(
        model_fn.patch_h, model_fn.patch_w, input_tensor.device
    )

    preds_seq = []
    predicted_heatmaps = []
    cached_frame_tokens = []
    warmup_frames = getattr(cfg.TEST, "STREAMING_METRICS_WARMUP", 0)
    record_timing = (
        getattr(cfg.TEST, "STREAMING_METRICS_ENABLE", False) and streaming_times is not None
    )
    for t in range(num_frames):
        frame_t = input_tensor[:, :, t]
        if record_timing and t >= warmup_frames:
            if cfg.NUM_GPUS:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
        current_tokens = model_fn.encode_single_frame(frame_t)

        frames_history = []
        centers_history = []
        frame_tokens_history = []
        for idx in range(model_fn.template_history_length):
            src_t = t - model_fn.template_history_length + idx
            if src_t < 0:
                frame_hist = input_tensor[:, :, 0]
            else:
                frame_hist = input_tensor[:, :, src_t]
            frames_history.append(frame_hist)

            if not model_fn.template_use_gaze_center:
                centers_history.append(None)
            else:
                if src_t < 0 or len(predicted_heatmaps) <= src_t:
                    center = torch.full((batch_size, 2), 0.5, device=input_tensor.device)
                else:
                    center = model_fn._heatmap_to_coords(predicted_heatmaps[src_t])
                centers_history.append(center)

            if src_t < 0:
                if cached_frame_tokens:
                    frame_tokens_history.append(cached_frame_tokens[0])
                else:
                    frame_tokens_history.append(current_tokens)
            else:
                if src_t < len(cached_frame_tokens):
                    frame_tokens_history.append(cached_frame_tokens[src_t])
                else:
                    frame_tokens_history.append(current_tokens)

        template_tokens = None
        if model_fn.use_template_tokens:
            if getattr(cfg.TEST, "STREAMING_TEMPLATE_FROM_TOKENS", False) and model_fn.use_original_template_tokens:
                template_tokens = model_fn.encode_template_from_tokens(
                    template_frame_tokens=frame_tokens_history,
                    centers_history=centers_history,
                )
            else:
                template_tokens = model_fn.encode_template_crops(
                    frames_history=frames_history,
                    centers_history=centers_history,
                )

        pred_t = model_fn.streaming_decode_step(
            t,
            current_tokens,
            template_tokens,
            predicted_heatmaps,
            pos_encoding,
        )
        if record_timing and t >= warmup_frames:
            if cfg.NUM_GPUS:
                torch.cuda.synchronize()
            streaming_times.append(time.perf_counter() - t0)
        preds_seq.append(pred_t)
        predicted_heatmaps.append(pred_t.detach())
        cached_frame_tokens.append(current_tokens.detach())

    preds = torch.stack(preds_seq, dim=2)
    return preds


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg, writer=None, efficiency_metrics=None):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestGazeMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode (skip if using baseline)
    if model is not None:
        model.eval()
    test_meter.iter_tic()

    # Create progress bar for testing
    # Only show progress bar on the main process in distributed training
    show_progress = du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS)
    test_iterator = tqdm(test_loader, desc="Testing", disable=not show_progress, ncols=100)

    streaming_times = []
    collect_streaming_times = (
        getattr(cfg.TEST, "STREAMING_ENABLE", False)
        and getattr(cfg.TEST, "STREAMING_METRICS_ENABLE", False)
    )
    vis_enabled = getattr(cfg.TEST, "VISUALIZE", False) and du.is_master_proc()
    save_frame_metrics = getattr(cfg.TEST, "SAVE_PER_FRAME_METRICS", False) and du.is_master_proc()
    vis_dir = None
    frames_dir = getattr(cfg.DATA, "FRAMES_DIR", None) or getattr(cfg.DATA, "FRAME_DIR", None)
    frame_ext = getattr(cfg.DATA, "FRAME_EXT", "jpg")
    vis_max_frames = int(getattr(cfg.TEST, "VIS_MAX_FRAMES", -1))
    vis_saved = 0
    per_frame_rows = []
    per_task_stats = {}
    task_key = getattr(cfg.TEST, "TASK_CATEGORY_KEY", "parent_task_name")
    report_by_idx = getattr(cfg.TEST, "REPORT_METRICS_BY_CLIP_INDEX", False)
    only_last_frame = getattr(cfg.TEST, "ONLY_LAST_FRAME", False)
    per_idx_stats = None
    if vis_enabled or save_frame_metrics:
        vis_dir = os.path.join(cfg.OUTPUT_DIR, getattr(cfg.TEST, "VIS_DIR", "visulization"))
        os.makedirs(vis_dir, exist_ok=True)
    if vis_enabled and (not frames_dir or not os.path.exists(frames_dir)):
        logger.warning("Frame visualization enabled, but DATA.FRAMES_DIR is invalid: %s", frames_dir)
        vis_enabled = False

    def _normalize_task_values(task_values, batch_size):
        if task_values is None:
            return None
        if isinstance(task_values, torch.Tensor):
            task_values = task_values.tolist()
        if isinstance(task_values, np.ndarray):
            task_values = task_values.tolist()
        if not isinstance(task_values, (list, tuple)) or len(task_values) != batch_size:
            return None
        return [str(v) if v is not None else "" for v in task_values]

    def _count_valid_frames(labels_subset, dataset_name, edge_thresh):
        labels_flat = labels_subset.view(-1, labels_subset.size(-1))
        if dataset_name in ["Holoassistgaze", "holoassistgaze", "Egoexo4dgaze", "egoexo4dgaze"]:
            tracked_idx = torch.where(labels_flat[:, 2] >= 0.5)[0]
        else:
            fixation_idx = 1 if dataset_name == "egteagaze" else 0
            tracked_idx = torch.where(labels_flat[:, 2] == fixation_idx)[0]
        if edge_thresh > 0.0 and tracked_idx.numel() > 0:
            labels_tracked = labels_flat.index_select(0, tracked_idx)
            edge_valid_idx = metrics.filter_edge_frames(labels_tracked, edge_thresh)
            tracked_idx = tracked_idx[edge_valid_idx]
        return int(tracked_idx.numel())

    for cur_iter, (inputs, labels, labels_hm, video_idx, meta) in enumerate(test_iterator):
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            # Transfer the data to the current GPU device.
            labels = labels.cuda()
            labels_hm = labels_hm.cuda()
            video_idx = video_idx.cuda()

        test_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            preds = preds.detach().cpu() if cfg.NUM_GPUS else preds.detach()
            ori_boxes = (
                ori_boxes.detach().cpu() if cfg.NUM_GPUS else ori_boxes.detach()
            )
            metadata = (
                metadata.detach().cpu() if cfg.NUM_GPUS else metadata.detach()
            )

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(preds, ori_boxes, metadata)
            test_meter.log_iter_stats(None, cur_iter)
        else:
            # Check if we should use a baseline instead of model            
            baseline_type = getattr(cfg.MODEL, "BASELINE_TYPE", "none")

            if baseline_type != "none":
                # Generate baseline predictions
                batch_size, num_frames = labels_hm.shape[:2]
                height, width = labels_hm.shape[-2:]
                sigma = getattr(cfg.MODEL, "BASELINE_GAUSSIAN_STD", 5.0)
                device = labels_hm.device

                if baseline_type == "random":
                    preds = baseline_utils.generate_random_baseline_heatmaps(
                        batch_size, num_frames, height, width, sigma, device
                    )
                elif baseline_type == "center":
                    preds = baseline_utils.generate_center_baseline_heatmaps(
                        batch_size, num_frames, height, width, sigma, device
                    )
                elif baseline_type == "dataset_prior":
                    prior_path = cfg.MODEL.DATASET_PRIOR_PATH
                    if not prior_path or not os.path.exists(prior_path):
                        raise ValueError(f"DATASET_PRIOR_PATH must be set and exist for baseline_type='dataset_prior'")

                    prior = baseline_utils.load_dataset_prior(prior_path, cfg.TEST.DATASET)
                    preds = baseline_utils.generate_dataset_prior_heatmaps(
                        batch_size, num_frames, height, width, sigma,
                        prior['mean_x'], prior['mean_y'], device
                    )
                else:
                    raise ValueError(f"Unknown baseline type: {baseline_type}")

                logger.info(f"Using baseline: {baseline_type}")
            else:
                # Perform the forward pass with actual model
                # Check if autoregressive inference should be used (for AR models)
                if cfg.MODEL.MODEL_NAME in [
                    "DINOv3_AR_HP",
                    "DINOv3_CrossAttnGazeAR",
                    "DINOv3_TransformerAR",
                    "DINOv3_ARHeatmapGaze",
                    "DINOv3_ARHeatmapGazeTemplate",
                ]:
                    use_autoregressive_test = getattr(cfg.MODEL, "AUTOREGRESSIVE_TEST", True)
                    use_gt_as_prompt = getattr(cfg.MODEL, "USE_GT_AS_PROMPT", False)
                    use_gt_as_template_center = getattr(
                        cfg.MODEL, "USE_GT_AS_TEMPLATE_CENTER", False
                    )

                    # Handle DINOv3_AR_HP model
                    if cfg.MODEL.MODEL_NAME == "DINOv3_AR_HP" and getattr(cfg.MODEL, "USE_HEAD_PROMPTING", False):
                        if use_gt_as_prompt:
                            # EXP 7: GT as Prompt (Oracle Upper Bound)
                            logger.info("[EXP 7 - AR_HP] Using GT heatmap as prompt (Oracle mode)")
                            preds = model([inputs, labels_hm]) if hasattr(model, 'module') else model([inputs, labels_hm])
                        elif use_autoregressive_test:
                            # Autoregressive mode: sequential frame-by-frame prediction
                            preds = model.module.forward_autoregressive(inputs) if hasattr(model, 'module') else model.forward_autoregressive(inputs)
                        else:
                            # Standard forward (may use zero prompts if no GT provided)
                            preds = model(inputs)

                    # Handle DINOv3_CrossAttnGazeAR model
                    elif cfg.MODEL.MODEL_NAME == "DINOv3_CrossAttnGazeAR":
                        if use_gt_as_prompt:
                            # EXP 7: GT as Prompt (Oracle Upper Bound)
                            logger.info("[EXP 7 - CrossAttnGazeAR] Using GT heatmap as prompt (Oracle mode)")
                            # CrossAttnGazeAR uses teacher forcing mode with gt_heatmap
                            if hasattr(model, 'module'):
                                preds = model.module(inputs, gt_heatmap=labels_hm, train_ar=False, ss_prob=0.0)
                            else:
                                preds = model(inputs, gt_heatmap=labels_hm, train_ar=False, ss_prob=0.0)
                            # Handle confidence output if present
                            if isinstance(preds, tuple):
                                preds = preds[0]  # Extract heatmap, ignore confidence
                        else:
                            # Standard autoregressive inference: use predicted heatmaps
                            if hasattr(model, 'module'):
                                preds = model.module(inputs, gt_heatmap=None, train_ar=True, ss_prob=0.0)
                            else:
                                preds = model(inputs, gt_heatmap=None, train_ar=True, ss_prob=0.0)
                            # Handle confidence output if present
                            if isinstance(preds, tuple):
                                preds = preds[0]  # Extract heatmap, ignore confidence

                    # Handle DINOv3_TransformerAR model
                    elif cfg.MODEL.MODEL_NAME == "DINOv3_TransformerAR":
                        if use_gt_as_prompt:
                            # EXP 11: GT as Prompt (Oracle Upper Bound for Transformer)
                            logger.info("[EXP 11 - TransformerAR] Using GT heatmap as prompt (Oracle mode)")
                            # TransformerAR uses training mode with GT heatmap for teacher forcing
                            if hasattr(model, 'module'):
                                preds = model.module(inputs, gt_heatmap=labels_hm, train_ar=True, ss_prob=0.0)
                            else:
                                preds = model(inputs, gt_heatmap=labels_hm, train_ar=True, ss_prob=0.0)
                        else:
                            # Standard autoregressive inference: use predicted heatmaps
                            logger.info("[TransformerAR] Using autoregressive inference mode")
                            if hasattr(model, 'module'):
                                preds = model.module(inputs, gt_heatmap=None, train_ar=False, ss_prob=0.0)
                            else:
                                preds = model(inputs, gt_heatmap=None, train_ar=False, ss_prob=0.0)
                    # Handle DINOv3_ARHeatmapGaze / DINOv3_ARHeatmapGazeTemplate models
                    elif cfg.MODEL.MODEL_NAME in ["DINOv3_ARHeatmapGaze", "DINOv3_ARHeatmapGazeTemplate"]:
                        if use_gt_as_prompt:
                            logger.info(
                                "=== ORACLE TEST ENABLED (ARHeatmapGaze): Using GT heatmap as prompt ==="
                            )
                            if hasattr(model, 'module'):
                                preds = model.module(inputs, gt_heatmap=labels_hm, train_ar=False, ss_prob=0.0)
                            else:
                                preds = model(inputs, gt_heatmap=labels_hm, train_ar=False, ss_prob=0.0)
                        elif use_gt_as_template_center and cfg.MODEL.MODEL_NAME == "DINOv3_ARHeatmapGazeTemplate":
                            logger.info(
                                "=== ORACLE CENTER TEST (ARHeatmapGazeTemplate): GT center, predicted history ==="
                            )
                            if getattr(cfg.TEST, "STREAMING_ENABLE", False):
                                logger.warning(
                                    "Streaming test ignores USE_GT_AS_TEMPLATE_CENTER; "
                                    "using predicted centers only."
                                )
                                preds = _streaming_ar_template_forward(
                                    model,
                                    inputs,
                                    cfg,
                                    streaming_times if collect_streaming_times else None,
                                )
                            else:
                                if hasattr(model, 'module'):
                                    preds = model.module(
                                        inputs,
                                        gt_heatmap=None,
                                        train_ar=True,
                                        ss_prob=0.0,
                                        gt_heatmap_center=labels_hm,
                                    )
                                else:
                                    preds = model(
                                        inputs,
                                        gt_heatmap=None,
                                        train_ar=True,
                                        ss_prob=0.0,
                                        gt_heatmap_center=labels_hm,
                                    )
                        else:
                            # Standard autoregressive inference: use predicted heatmaps
                            if getattr(cfg.TEST, "STREAMING_ENABLE", False):
                                if cfg.MODEL.MODEL_NAME != "DINOv3_ARHeatmapGazeTemplate":
                                    logger.warning(
                                        "TEST.STREAMING_ENABLE is set but MODEL.MODEL_NAME=%s; "
                                        "falling back to standard inference.",
                                        cfg.MODEL.MODEL_NAME,
                                    )
                                    preds = model(inputs)
                                else:
                                    preds = _streaming_ar_template_forward(
                                        model,
                                        inputs,
                                        cfg,
                                        streaming_times if collect_streaming_times else None,
                                    )
                            else:
                                if hasattr(model, 'module'):
                                    preds = model.module(inputs, gt_heatmap=None, train_ar=True, ss_prob=0.0)
                                else:
                                    preds = model(inputs, gt_heatmap=None, train_ar=True, ss_prob=0.0)
                    else:
                        # AR_HP without head prompting - standard forward
                        preds = model(inputs)
                elif getattr(cfg.TEST, "STREAMING_ENABLE", False):
                    # Streaming only supported for GLC and ARHeatmapGazeTemplate models
                    if cfg.MODEL.MODEL_NAME not in ["GLC_Gaze", "GLC_Gaze_Causal"]:
                        raise ValueError(
                            f"TEST.STREAMING_ENABLE is set but MODEL.MODEL_NAME={cfg.MODEL.MODEL_NAME} "
                            "does not support streaming. Supported models: GLC_Gaze, GLC_Gaze_Causal, "
                            "DINOv3_ARHeatmapGazeTemplate (handled above)."
                        )
                    preds = _streaming_glc_forward(
                        model,
                        inputs,
                        cfg,
                        streaming_times if collect_streaming_times else None,
                    )
                else:
                    # Standard models: use regular forward pass
                    preds = model(inputs)
                # preds, glc = model(inputs, return_glc=True)  # used to visualization glc correlation

            preds = frame_softmax(preds, temperature=2)  # KLDiv

            # Gather all the predictions across all the devices to perform ensemble.
            if cfg.NUM_GPUS > 1:
                preds, labels, labels_hm, video_idx = du.all_gather([preds, labels, labels_hm, video_idx])

            # PyTorch
            if cfg.NUM_GPUS:  # compute on cpu
                preds = preds.cpu()
                labels = labels.cpu()
                labels_hm = labels_hm.cpu()
                video_idx = video_idx.cpu()

            if only_last_frame:
                preds = preds[:, :, -1:].contiguous()
                if labels_hm.dim() == 5:
                    labels_hm = labels_hm[:, :, -1:].contiguous()
                else:
                    labels_hm = labels_hm[:, -1:].contiguous()
                labels = labels[:, -1:].contiguous()

            preds_rescale = preds.detach().view(preds.size()[:-2] + (preds.size(-1) * preds.size(-2),))
            preds_rescale = (preds_rescale - preds_rescale.min(dim=-1, keepdim=True)[0]) / (preds_rescale.max(dim=-1, keepdim=True)[0] - preds_rescale.min(dim=-1, keepdim=True)[0] + 1e-6)
            preds_rescale = preds_rescale.view(preds.size())

            # Get edge filtering parameters from config
            edge_threshold = cfg.TEST.EDGE_THRESHOLD if cfg.TEST.FILTER_EDGE_GAZE else 0.0

            f1, recall, precision, threshold = metrics.adaptive_f1(preds_rescale, labels_hm, labels, dataset=cfg.TEST.DATASET, edge_threshold=edge_threshold)
            auc = metrics.auc(preds_rescale, labels_hm, labels, dataset=cfg.TEST.DATASET, edge_threshold=edge_threshold)
            l2 = metrics.l2_distance(
                preds_rescale,
                labels_hm,
                labels,
                dataset=cfg.TEST.DATASET,
                edge_threshold=edge_threshold,
                l2_mode=cfg.TEST.L2_MODE,
            )

            if report_by_idx:
                T = preds_rescale.size(2)
                if per_idx_stats is None or len(per_idx_stats) < T:
                    per_idx_stats = []
                    for _ in range(T):
                        per_idx_stats.append({
                            "count": 0,
                            "f1_sum": 0.0,
                            "recall_sum": 0.0,
                            "precision_sum": 0.0,
                            "auc_sum": 0.0,
                            "entropy_sum": 0.0,
                        })
                for t in range(T):
                    preds_t = preds_rescale[:, :, t:t + 1]
                    if labels_hm.dim() == 5:
                        labels_hm_t = labels_hm[:, :, t:t + 1]
                    else:
                        labels_hm_t = labels_hm[:, t:t + 1]
                    labels_t = labels[:, t:t + 1]
                    valid_frames = _count_valid_frames(labels_t, cfg.TEST.DATASET, edge_threshold)
                    if valid_frames <= 0:
                        continue
                    f1_t, recall_t, precision_t, _ = metrics.adaptive_f1(
                        preds_t, labels_hm_t, labels_t, dataset=cfg.TEST.DATASET, edge_threshold=edge_threshold
                    )
                    auc_t = metrics.auc(
                        preds_t, labels_hm_t, labels_t, dataset=cfg.TEST.DATASET, edge_threshold=edge_threshold
                    )
                    entropy_t = -(preds[:, :, t:t + 1] * torch.log(preds[:, :, t:t + 1] + 1e-10)).sum(dim=(-1, -2))
                    entropy_t = float(entropy_t.mean().item())
                    stats = per_idx_stats[t]
                    stats["count"] += valid_frames
                    stats["f1_sum"] += float(f1_t) * valid_frames
                    stats["recall_sum"] += float(recall_t) * valid_frames
                    stats["precision_sum"] += float(precision_t) * valid_frames
                    stats["auc_sum"] += float(auc_t) * valid_frames
                    stats["entropy_sum"] += entropy_t * valid_frames

            frame_stats = None
            if vis_enabled or save_frame_metrics:
                frame_stats = _compute_per_frame_stats(
                    preds_rescale,
                    labels_hm,
                    labels,
                    cfg.TEST.DATASET,
                    edge_threshold,
                    cfg.TEST.L2_MODE,
                    threshold,
                )

            if (vis_enabled or save_frame_metrics) and frame_stats is not None:
                meta_paths = meta.get("path", [])
                frame_indices = meta.get("index", None)
                if isinstance(frame_indices, torch.Tensor):
                    frame_indices = frame_indices.cpu().numpy()
                if frame_indices is not None and len(frame_indices.shape) == 1:
                    frame_indices = frame_indices.reshape(frame_indices.shape[0], 1)

                B, T = preds_rescale.size(0), preds_rescale.size(2)
                for b in range(B):
                    video_path = meta_paths[b] if b < len(meta_paths) else ""
                    video_name = _get_video_name(str(video_path))
                    for t in range(T):
                        frame_idx = int(frame_indices[b, t]) if frame_indices is not None else int(t)
                        gt_x = float(labels[b, t, 0])
                        gt_y = float(labels[b, t, 1])
                        valid = bool(frame_stats["valid"][b, t])
                        pred_x = float(frame_stats["pred_x"][b, t])
                        pred_y = float(frame_stats["pred_y"][b, t])
                        f1_frame = float(frame_stats["f1"][b, t])
                        recall_frame = float(frame_stats["recall"][b, t])
                        precision_frame = float(frame_stats["precision"][b, t])
                        l2_frame = float(frame_stats["l2"][b, t])

                        if save_frame_metrics:
                            per_frame_rows.append({
                                "video_name": video_name,
                                "frame_idx": frame_idx,
                                "clip_index": int(video_idx[b].item()) if video_idx is not None else -1,
                                "batch_index": b,
                                "frame_offset": t,
                                "f1": f1_frame,
                                "recall": recall_frame,
                                "precision": precision_frame,
                                "l2": l2_frame,
                                "pred_x": pred_x,
                                "pred_y": pred_y,
                                "gt_x": gt_x,
                                "gt_y": gt_y,
                                "valid": int(valid),
                                "threshold": float(threshold),
                            })

                        if vis_enabled and (vis_max_frames < 0 or vis_saved < vis_max_frames):
                            frame = _load_frame_image(frames_dir, video_name, frame_idx, frame_ext)
                            if frame is None:
                                continue
                            heatmap_np = preds_rescale[b, 0, t].numpy()
                            vis_frame = _overlay_heatmap_bgr(frame, heatmap_np, alpha=getattr(cfg.TEST, "VIS_ALPHA", 0.4))
                            if np.isfinite(pred_x) and np.isfinite(pred_y):
                                vis_frame = _draw_gaze_marker_bgr(vis_frame, pred_x, pred_y, color=(0, 255, 0))
                            if valid:
                                vis_frame = _draw_gaze_marker_bgr(vis_frame, gt_x, gt_y, color=(0, 0, 255))
                            out_dir = os.path.join(vis_dir, video_name) if video_name else vis_dir
                            os.makedirs(out_dir, exist_ok=True)
                            out_path = os.path.join(out_dir, f"{frame_idx:06d}_b{b:02d}_t{t:02d}.jpg")
                            cv2.imwrite(out_path, vis_frame)
                            vis_saved += 1

            test_meter.iter_toc()

            # Update and log stats.
            test_meter.update_stats(f1, recall, precision, auc, l2, preds=preds_rescale, labels_hm=labels_hm, labels=labels)  # If running  on CPU (cfg.NUM_GPUS == 0), use 1 to represent 1 CPU.
            test_meter.log_iter_stats(cur_iter)

            # Update progress bar with current metrics
            if show_progress:
                test_iterator.set_postfix({
                    'F1': f'{f1:.4f}',
                    'AUC': f'{auc:.4f}',
                    'L2': f'{l2:.4f}'
                })

            task_values = _normalize_task_values(meta.get(task_key), preds_rescale.size(0))
            if task_values is None and task_key != "task_name":
                task_values = _normalize_task_values(meta.get("task_name"), preds_rescale.size(0))

            if task_values is not None:
                for task in sorted(set(task_values)):
                    task_label = task if task else "unknown"
                    idx = [i for i, v in enumerate(task_values) if v == task]
                    if not idx:
                        continue
                    idx_tensor = torch.tensor(idx, dtype=torch.long)
                    preds_task = preds_rescale.index_select(0, idx_tensor)
                    labels_hm_task = labels_hm.index_select(0, idx_tensor)
                    labels_task = labels.index_select(0, idx_tensor)

                    task_f1, task_recall, task_precision, _ = metrics.adaptive_f1(
                        preds_task, labels_hm_task, labels_task, dataset=cfg.TEST.DATASET, edge_threshold=edge_threshold
                    )
                    task_auc = metrics.auc(
                        preds_task, labels_hm_task, labels_task, dataset=cfg.TEST.DATASET, edge_threshold=edge_threshold
                    )
                    task_l2 = metrics.l2_distance(
                        preds_task,
                        labels_hm_task,
                        labels_task,
                        dataset=cfg.TEST.DATASET,
                        edge_threshold=edge_threshold,
                        l2_mode=cfg.TEST.L2_MODE,
                    )
                    valid_frames = _count_valid_frames(labels_task, cfg.TEST.DATASET, edge_threshold)
                    if valid_frames == 0:
                        continue

                    stats = per_task_stats.setdefault(task_label, {
                        "frames": 0,
                        "clips": 0,
                        "f1_sum": 0.0,
                        "recall_sum": 0.0,
                        "precision_sum": 0.0,
                        "auc_sum": 0.0,
                        "l2_sum": 0.0,
                    })
                    stats["frames"] += valid_frames
                    stats["clips"] += len(idx)
                    stats["f1_sum"] += task_f1 * valid_frames
                    stats["recall_sum"] += task_recall * valid_frames
                    stats["precision_sum"] += task_precision * valid_frames
                    stats["auc_sum"] += task_auc * valid_frames
                    stats["l2_sum"] += task_l2 * valid_frames

        test_meter.iter_tic()

    # Log epoch stats and print the final testing results.
    if not cfg.DETECTION.ENABLE:
        all_preds = test_meter.video_preds.clone().detach()
        all_labels = test_meter.video_labels
        if cfg.NUM_GPUS:
            all_preds = all_preds.cpu()
            all_labels = all_labels.cpu()
        if writer is not None:
            writer.plot_eval(preds=all_preds, labels=all_labels)

        if cfg.TEST.SAVE_RESULTS_PATH != "":
            save_path = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_RESULTS_PATH)

            if du.is_root_proc():
                with pathmgr.open(save_path, "wb") as f:
                    pickle.dump([all_preds, all_labels], f)

            logger.info("Successfully saved prediction results to {}".format(save_path))

    if save_frame_metrics and per_frame_rows and vis_dir is not None:
        metrics_path = os.path.join(
            vis_dir, getattr(cfg.TEST, "PER_FRAME_METRICS_FILE", "per_frame_metrics.csv")
        )
        fieldnames = list(per_frame_rows[0].keys())
        with open(metrics_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_frame_rows)
        logger.info("Saved per-frame metrics to %s", metrics_path)

    test_meter.finalize_metrics(efficiency_metrics=efficiency_metrics)
    if report_by_idx and per_idx_stats and du.is_master_proc():
        rows = []
        for t, stats in enumerate(per_idx_stats):
            count = stats["count"]
            if count <= 0:
                continue
            rows.append({
                "frame_index": t,
                "frames": count,
                "f1": stats["f1_sum"] / count,
                "recall": stats["recall_sum"] / count,
                "precision": stats["precision_sum"] / count,
                "auc": stats["auc_sum"] / count,
                "entropy": stats["entropy_sum"] / count,
            })
        if rows:
            logging.log_json_stats({"split": "test_clip_index", "metrics_by_index": rows})
    if per_task_stats and du.is_master_proc():
        metrics_path = os.path.join(
            cfg.OUTPUT_DIR,
            getattr(cfg.TEST, "TASK_METRICS_FILE", "task_metrics.csv"),
        )
        rows = []
        for task, stats in per_task_stats.items():
            frames = stats["frames"]
            if frames <= 0:
                continue
            rows.append({
                "task_key": task_key,
                "task": task,
                "frames": frames,
                "clips": stats["clips"],
                "f1": stats["f1_sum"] / frames,
                "recall": stats["recall_sum"] / frames,
                "precision": stats["precision_sum"] / frames,
                "auc": stats["auc_sum"] / frames,
                "l2": stats["l2_sum"] / frames,
            })

        rows = sorted(rows, key=lambda r: (-r["frames"], r["task"]))
        if rows:
            with open(metrics_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            logger.info("Saved per-task metrics (%s) to %s", task_key, metrics_path)
            for row in rows:
                logger.info(
                    "[Task %s] frames=%d clips=%d f1=%.4f recall=%.4f precision=%.4f auc=%.4f l2=%.4f",
                    row["task"],
                    row["frames"],
                    row["clips"],
                    row["f1"],
                    row["recall"],
                    row["precision"],
                    row["auc"],
                    row["l2"],
                )

    if collect_streaming_times:
        if streaming_times:
            times = np.array(streaming_times, dtype=np.float64)
            total_time = float(times.sum())
            fps = float(len(times) / total_time) if total_time > 0.0 else 0.0
            stats = {
                "split": "test_final_streaming",
                "streaming_frames": int(len(times)),
                "streaming_latency_ms_mean": float(times.mean() * 1000.0),
                "streaming_latency_ms_p50": float(np.percentile(times, 50) * 1000.0),
                "streaming_latency_ms_p95": float(np.percentile(times, 95) * 1000.0),
                "streaming_fps": fps,
            }
            logging.log_json_stats(stats)
        else:
            logger.warning("Streaming metrics enabled but no timing data was collected.")
    return test_meter


def test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    # logger.info("Test with config:")
    # logger.info(cfg)

    # Log test split configuration
    test_split = getattr(cfg.TEST, "SPLIT", "test")
    logger.info("=" * 80)
    logger.info(f"TEST CONFIGURATION:")
    logger.info(f"  Dataset: {cfg.TEST.DATASET}")
    logger.info(f"  Split: {test_split}")
    logger.info("=" * 80)

    # Log edge filtering configuration
    if cfg.TEST.FILTER_EDGE_GAZE:
        logger.info(f"Edge gaze filtering ENABLED: threshold={cfg.TEST.EDGE_THRESHOLD:.3f} (filtering gazes within {cfg.TEST.EDGE_THRESHOLD*100:.1f}% of frame edges)")
    else:
        logger.info("Edge gaze filtering DISABLED")
    logger.info(f"L2 mode: {cfg.TEST.L2_MODE}")

    # Check if using baseline (skip model building if so)
    baseline_type = getattr(cfg.MODEL, "BASELINE_TYPE", "none")
    use_baseline = baseline_type != "none"

    if use_baseline:
        logger.info(f"Using baseline method: {baseline_type}")
        logger.info("Skipping model building and checkpoint loading for baseline testing")
        model = None  # No model needed for baselines
    else:
        # Build the video model and print model statistics.
        model = build_model(cfg)
        if du.is_master_proc() and cfg.LOG_MODEL_INFO:
            misc.log_model_info(model, cfg, use_train_input=False)

        cu.load_test_checkpoint(cfg, model)

    # Compute efficiency metrics
    if not use_baseline:
        logger.info("Computing efficiency metrics...")
        num_params = metrics.count_parameters(model)
        logger.info(f"Model Parameters: {num_params:.2f}M")
    else:
        num_params = 0.0

    # Initialize efficiency metrics dictionary
    efficiency_metrics = {
        "parameters_M": num_params,
        "gflops": -1.0,
        "throughput_fps": -1.0,
        "latency_ms": -1.0,
        "peak_memory_mb": -1.0,
        "activation_memory_mb": -1.0
    }

    # Get a sample input for FLOPs, throughput, and memory computation
    # Skip for baselines since they don't use a model
    if not use_baseline:
        test_loader = loader.construct_loader(cfg, "test")

        # Compute efficiency metrics with sample input
        try:
            sample_inputs, sample_labels, sample_labels_hm, sample_video_idx, sample_meta = next(iter(test_loader))

            if cfg.NUM_GPUS:
                if isinstance(sample_inputs, (list,)):
                    for i in range(len(sample_inputs)):
                        sample_inputs[i] = sample_inputs[i].cuda(non_blocking=True)
                else:
                    sample_inputs = sample_inputs.cuda(non_blocking=True)

            # Compute FLOPs
            flops = metrics.compute_flops(model, sample_inputs, cfg)
            if flops > 0:
                efficiency_metrics["gflops"] = flops
                logger.info(f"Model GFLOPs: {flops:.2f}")

            # Compute throughput and latency
            throughput, latency = metrics.measure_throughput_and_latency(model, sample_inputs, cfg)
            efficiency_metrics["throughput_fps"] = throughput
            efficiency_metrics["latency_ms"] = latency
            logger.info(f"Throughput: {throughput:.2f} FPS")
            logger.info(f"Latency: {latency:.2f} ms")

            # Compute memory footprint
            memory_stats = metrics.measure_memory_footprint(model, sample_inputs, cfg)
            if memory_stats["peak_memory_mb"] > 0:
                efficiency_metrics["peak_memory_mb"] = memory_stats["peak_memory_mb"]
                efficiency_metrics["activation_memory_mb"] = memory_stats["activation_memory_mb"]
                logger.info(f"Peak Memory: {memory_stats['peak_memory_mb']:.2f} MB")
                logger.info(f"Activation Memory: {memory_stats['activation_memory_mb']:.2f} MB")

        except Exception as e:
            logger.warning(f"Failed to compute efficiency metrics: {e}")

    # Create/recreate test loader for actual testing
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    if cfg.DETECTION.ENABLE:
        assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE or cfg.NUM_GPUS == 0
        test_meter = AVAMeter(len(test_loader), cfg, mode="test")
    else:
        assert (test_loader.dataset.num_videos % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS) == 0)
        # Create meters for multi-view testing.
        edge_threshold = cfg.TEST.EDGE_THRESHOLD if cfg.TEST.FILTER_EDGE_GAZE else 0.0
        test_meter = TestGazeMeter(
            num_videos=test_loader.dataset.num_videos // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
            num_clips=cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
            num_cls=cfg.MODEL.NUM_CLASSES,
            overall_iters=len(test_loader),
            dataset=cfg.TEST.DATASET,
            store_heatmaps=cfg.TEST.STORE_HEATMAPS,
            edge_threshold=edge_threshold,
            l2_mode=cfg.TEST.L2_MODE,
        )

    # Set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Perform multi-view test on the entire dataset.
    test_meter = perform_test(test_loader, model, test_meter, cfg, writer, efficiency_metrics=efficiency_metrics)
    if writer is not None:
        writer.close()

    logger.info("Testing finished!")
