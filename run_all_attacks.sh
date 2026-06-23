#!/bin/bash
# 一键运行所有 jailbreak 方法攻击 qwen2.5-7b

TARGET="qwen2.5-7b"

ATTACK_FILES=(
    "attack/results/primary/GCG_individual_llama2-7b.json"
    "attack/results/primary/AutoDAN-GA_llama2-7b.json"
    "attack/results/primary/PAIR_llama-2-7b.json"
    "attack/results/primary/DrAttack_llama-2-7b.json"
    "attack/results/primary/LLM-Fuzzer_llama-2-7b.json"
    "attack/results/primary/TAP_llama-2-13b.json"
    "attack/results/primary/RLbreaker_llama-2-7b.json"
    "attack/results/primary/Puzzler.json"
    "attack/results/manual/DAN.json"
    "attack/results/multilingual/MultiJail_unintentional_bn.json"
)

for f in "${ATTACK_FILES[@]}"; do
    echo "========================================"
    echo "Running: $f"
    echo "========================================"
    python jailbreaking.py --jailbreak-path "$f" --target-model "$TARGET"
    echo ""
done

echo "All attacks completed!"
