#!/bin/bash

# Set CUDA environment
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

# Set Aphrodite environment
export APHRODITE_HOME=/home/gitlab/ocr
export APHRODITE_CACHE_DIR=/home/gitlab/.cache/aphrodite

# Run model
aphrodite run \
    --model erax-ai/EraX-VL-2B-V1.5 \
    --tensor-parallel-size 2 \
    --max-model-len 2048 \
    --rope-scaling-type linear \
    --rope-scaling-factor 2.0 \
    --trust-remote-code \
    --dtype float16