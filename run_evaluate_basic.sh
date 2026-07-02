#!/bin/bash
# =============================================================
# Phase 2a: 批量评估 SelfDefend-basic 防御效果
# =============================================================
# 不需要微调，不需要 API，直接用目标模型自身做影子检测
#
# SelfDefend-basic 原理:
#   同一个 Qwen2.5-7B 同时承担两个角色:
#     - 目标栈: 正常回答问题
#     - 影子栈: 检测 prompt 是否违规 (带 few-shot 示例)
#   两个栈并行运行，延迟 = max(目标延迟, 影子延迟)
# =============================================================

TARGET="qwen2.5-7b"
DEFENSE="SelfDefend-basic"

RESULT_FILES=(
    "results/primary/GCG_individual_llama2-7b_${TARGET}.json"
    "results/primary/AutoDAN-GA_llama2-7b_${TARGET}.json"
    "results/primary/PAIR_llama-2-7b_${TARGET}.json"
    "results/primary/DrAttack_llama-2-7b_${TARGET}.json"
    "results/primary/LLM-Fuzzer_llama-2-7b_${TARGET}.json"
    "results/primary/TAP_llama-2-13b_${TARGET}.json"
    "results/primary/RLbreaker_llama-2-7b_${TARGET}.json"
    "results/primary/Puzzler_${TARGET}.json"
    "results/manual/DAN_${TARGET}.json"
    "results/multilingual/MultiJail_unintentional_bn_${TARGET}.json"
    "results/normal/AlpacaEval_instruction_${TARGET}.json"   # 正常数据 → 测误报率
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
echo "All SelfDefend-basic evaluations completed!"
echo "Results saved to: results/defense/${TARGET}/"
echo "========================================"
