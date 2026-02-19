#!/bin/bash

# Testing script for DINOv3_ARHeatmapGazeTemplate on EgoExo4D
# Default settings: Scale=0.35, History=3
# Evaluates on all 4 splits: test_iid, test_ood_site, test_ood_task, test_ood_participant

# ================= Configuration =================
# Path to your trained checkpoint
CHECKPOINT_PATH="path/to/your/checkpoint.pyth"

# Base directory to save test results (subfolders will be created for each split)
OUTPUT_BASE_DIR="./output/egoexo4d/test_results"
# =================================================

export CUDA_VISIBLE_DEVICES=0

SPLITS=("test_iid" "test_ood_site" "test_ood_task" "test_ood_participant")

for split in "${SPLITS[@]}"; do
  echo "Testing on split: ${split}"

  python tools/run_net.py \
    --cfg configs/Egoexo4d/DINOV3_ARHeatmapGazeTemplate.yaml \
    TRAIN.ENABLE False \
    TEST.ENABLE True \
    TEST.SPLIT ${split} \
    TEST.STORE_HEATMAPS False \
    TEST.TASK_CATEGORY_KEY parent_task_name \
    NUM_GPUS 1 \
    MODEL.HISTORY_LENGTH 3 \
    MODEL.TEMPLATE_SCALES [0.35] \
    TEST.CHECKPOINT_FILE_PATH ${CHECKPOINT_PATH} \
    OUTPUT_DIR "${OUTPUT_BASE_DIR}/${split}" \
    WANDB.RUN_NAME "EgoExo4D_Template_Test_${split}"
done
