#!/bin/bash

# Testing script for DINOv3_ARHeatmapGazeTemplate on Ego4D
# Default settings: Scale=0.35, History=3

# ================= Configuration =================
# Path to your trained checkpoint
CHECKPOINT_PATH="path/to/your/checkpoint.pyth"

# Directory to save test results
OUTPUT_DIR="./output/ego4d/test_results"

# Path to dataset
# DATA_DIR="/path/to/ego4d"
# =================================================

export CUDA_VISIBLE_DEVICES=0

python tools/run_net.py \
  --cfg configs/Ego4d/DINOV3_ARHeatmapGazeTemplate.yaml \
  TRAIN.ENABLE False \
  TEST.ENABLE True \
  NUM_GPUS 1 \
  MODEL.HISTORY_LENGTH 3 \
  MODEL.TEMPLATE_SCALES [0.35] \
  TEST.CHECKPOINT_FILE_PATH ${CHECKPOINT_PATH} \
  OUTPUT_DIR ${OUTPUT_DIR} \
  TEST.VISUALIZE True \
  TEST.SAVE_PER_FRAME_METRICS True \
  DATA.HEATMAP_SIGMA -1.0
