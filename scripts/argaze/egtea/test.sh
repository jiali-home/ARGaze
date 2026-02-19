#!/bin/bash

# Testing script for DINOv3_ARHeatmapGazeTemplate
# Default settings: Scale=0.35, History=3

# ================= Configuration =================
# Path to your trained checkpoint
CHECKPOINT_PATH="path/to/your/checkpoint.pyth"

# Directory to save test results and visualizations
OUTPUT_DIR="./output/test_results"

# Path to the dataset frames
DATA_DIR="/path/to/egtea/cropped_frames"
# =================================================

export CUDA_VISIBLE_DEVICES=0

python tools/run_net.py \
  --cfg configs/Egtea/DINOV3_ARHeatmapGazeTemplate.yaml \
  TRAIN.ENABLE False \
  TEST.ENABLE True \
  NUM_GPUS 1 \
  MODEL.HISTORY_LENGTH 3 \
  MODEL.TEMPLATE_SCALES [0.35] \
  TEST.CHECKPOINT_FILE_PATH ${CHECKPOINT_PATH} \
  OUTPUT_DIR ${OUTPUT_DIR} \
  TEST.VISUALIZE True \
  TEST.SAVE_PER_FRAME_METRICS True \
  DATA.FRAMES_DIR ${DATA_DIR} \
  DATA.FRAME_EXT jpg
