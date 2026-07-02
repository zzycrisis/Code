#!/bin/bash
# =============================================================
# Phase 1: 微调 Qwen2.5-7B 防御模型
# =============================================================
# A10 (22.5GB) 优化配置:
#   --batch-size 4   - 降低显存占用 (8 → ~18GB, 4 → ~16GB)
#   gradient_checkpointing 已启用, 进一步节省 ~30% 显存
#
# 如果仍然 OOM, 把 --batch-size 改成 2
# 如果显存富余, 可改成 6 加快训练
#
# 断点续训: 加 --resume 即可
# =============================================================

echo "=== Fine-tuning direct defense model ==="
python fine_tuning.py --base-model qwen2.5 --prompt-type direct --batch-size 4 --num-epochs 1

echo "=== Fine-tuning intent defense model ==="
python fine_tuning.py --base-model qwen2.5 --prompt-type intent --batch-size 4 --num-epochs 1

echo "=== Done! LoRA weights saved to: ==="
echo "  checkpoint/qwen2.5-lora-direct/"
echo "  checkpoint/qwen2.5-lora-intent/"
