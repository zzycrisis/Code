#!/bin/bash
# =============================================================
# Phase 1: 微调 Qwen2.5-7B 防御模型 (大约 1-2 小时/每个)
# =============================================================
# 先跑 direct，再跑 intent。两个都必须跑完才能进入 Phase 2。

echo "=== Phase 1: Fine-tuning direct defense model ==="
python fine_tuning.py --base-model qwen2.5 --prompt-type direct --batch-size 8 --num-epochs 1

echo "=== Phase 1: Fine-tuning intent defense model ==="
python fine_tuning.py --base-model qwen2.5 --prompt-type intent --batch-size 8 --num-epochs 1

echo "=== Fine-tuning done! LoRA weights saved to: ==="
echo "  checkpoint/qwen2.5-lora-direct/"
echo "  checkpoint/qwen2.5-lora-intent/"
