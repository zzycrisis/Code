#!/bin/bash
# =============================================================
# Phase 2: 批量评估 SelfDefend-tuning 防御效果
# =============================================================
# 前置条件: 已完成 Phase 1 微调，LoRA 权重在 checkpoint/ 下
#
# 每个 jailbreak 结果文件用 direct 和 intent 两种 prompt 各测一次
# 结果保存到 results/defense/<target_model>/SelfDefend-tuning-<prompt>/

TARGET="qwen2.5-7b"
DEFENSE="SelfDefend-tuning"

# Step 1 产出的攻击结果文件
RESULT_FILES=(
    "results/primary/GCG_individual_llama2-7b_${TARGET}.json"
    "results/primary/AutoDAN-GA_llama2-7b_${TARGET}.json"
    "results/primary/PAIR_llama-2-7b_${TARGET}.json"
    "results/primary/DrAttack_llama-2-7b_${TARGET}.json"
    "results/primary/LLM-Fuzzer_llama-2-7b_${TARGET}.json"
    "results/primary/TAP_llama-2-13b_${TARGET}.json"
    "results/primary/RLbreaker_llama-2-7b_${TARGET}.json"
    "results/primary/Puzzler_${TARGET}.json"
)

for f in "${RESULT_FILES[@]}"; do
    if [ ! -f "$f" ]; then
        echo "SKIP: $f (not found)"
        continue
    fi

    for prompt in direct intent; do
        echo "========================================"
        echo "Evaluating: $(basename $f)"
        echo "Defense:    ${DEFENSE}-${prompt}"
        echo "========================================"
        python evaluate.py \
            --jailbreak-path "$f" \
            --defense-method "$DEFENSE" \
            --defense-prompt "$prompt"
        echo ""
    done
done

echo "========================================"
echo "All evaluations completed!"
echo "Results saved to: results/defense/${TARGET}/"
echo "========================================"
