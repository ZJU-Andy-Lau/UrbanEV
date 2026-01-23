#!/bin/bash

# 设置变量
models=(ar lo arima fcnn lstm gcn gcnlstm astgcn)
pred_lens=(3 6 9 12)
folds=(1 2 3 4 5 6)
EPOCH=20

# 嵌套循环执行实验
for m in "${models[@]}"; do
    for l in "${pred_lens[@]}"; do
        for f in "${folds[@]}"; do
            python main.py --model "$m" --pred_len "$l" --fold "$f" --epoch "$EPOCH" --use_npy
        done
    done
done

echo "✅ All experiments completed."
