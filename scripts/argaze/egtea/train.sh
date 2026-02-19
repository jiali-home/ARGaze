#!/bin/bash

# Training script for DINOv3_ARHeatmapGazeTemplate
# Default settings: Scale=0.35, History=3

# ================= Configuration =================
# Directory to save checkpoints and logs
OUTPUT_DIR="./output/train_results"

# WANDB Run Name (Optional)
WANDB_NAME="ARHeatmapGazeTemplate_Scale035_Train"

# Path to dataset frames (Set in config or override here)
# DATA_DIR="/path/to/egtea/cropped_frames"
# =================================================

export CUDA_VISIBLE_DEVICES=0

python tools/run_net.py \
  --cfg configs/Egtea/DINOV3_ARHeatmapGazeTemplate.yaml \
  TRAIN.ENABLE True \
  TEST.ENABLE False \
  NUM_GPUS 1 \
  MODEL.HISTORY_LENGTH 3 \
  MODEL.TEMPLATE_SCALES [0.35] \
  OUTPUT_DIR ${OUTPUT_DIR} \
  WANDB.RUN_NAME ${WANDB_NAME}
