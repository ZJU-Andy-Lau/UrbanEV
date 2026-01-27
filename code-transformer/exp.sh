#!/bin/bash

# 设置变量
EPOCH=1

# 运行 TimeXer 模型
python run.py \
    --seq_len 12 \
    --label_len 12 \
    --epoch $EPOCH \
    --model TimeXer \
    --use_npy

# 运行 TimesNet 模型
python run.py \
    --seq_len 12 \
    --label_len 12 \
    --epoch $EPOCH \
    --model TimesNet \
    --use_npy

echo "✅ All tasks completed!"
