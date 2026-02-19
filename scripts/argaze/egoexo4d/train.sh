#!/bin/bash

# Training script for DINOv3_ARHeatmapGazeTemplate on EgoExo4D
# Default settings: Scale=0.35, History=3

# ================= Configuration =================
# Directory to save checkpoints and logs
OUTPUT_DIR="./output/egoexo4d/train_results"

# WANDB Run Name (Optional)
WANDB_NAME="EgoExo4d_ARHeatmapGazeTemplate_Scale035_Train"
# =================================================

export CUDA_VISIBLE_DEVICES=0

python tools/run_net.py \
  --cfg configs/Egoexo4d/DINOV3_ARHeatmapGazeTemplate.yaml \
  TRAIN.ENABLE True \
  TEST.ENABLE False \
  NUM_GPUS 1 \
  MODEL.HISTORY_LENGTH 3 \
  MODEL.TEMPLATE_SCALES [0.35] \
  OUTPUT_DIR ${OUTPUT_DIR} \
  WANDB.RUN_NAME ${WANDB_NAME}
