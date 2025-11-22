#!/bin/bash

# Training script for two-stage training of Simba project

# Stage 1: Train SymmGT model
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/dist_train.sh 2 13232 \
    --config ./cfgs/ShapeNet55_models/SymmGT.yaml \
    --exp_name SymmGT_stage_1

# Automatically retrieve the best model path from stage 1
STAGE1_MODEL_PATH="./experiment/SymmGT_stage_1/best_model.pth"  # Adjust path as needed

# Stage 2: Train Simba model using the pretrained weights from stage 1
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/dist_train.sh 2 13232 \
    --config ./cfgs/ShapeNet55_models/Simba.yaml \
    --exp_name Simba_stage_2 \
    --pretrain $STAGE1_MODEL_PATH