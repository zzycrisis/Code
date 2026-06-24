#!/bin/bash
# 一键运行所有 jailbreak 方法攻击 qwen2.5-7b
# 模型只加载一次，所有 attack 文件依次处理，避免重复加载导致 CUDA OOM

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

python jailbreaking.py --jailbreak-path "${ATTACK_FILES[@]}" --target-model "$TARGET"

echo "All attacks completed!"
