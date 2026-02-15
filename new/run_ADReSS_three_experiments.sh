#!/bin/bash
# ADReSS：pathology（端到端可学习权重 / 注意力，GPU）+ random，无 Optuna
# 1) pathology：端到端联合训练 detector 权重（或 attention） + WavLM
# 2) random：随机掩码对照

set -e
export CUDA_VISIBLE_DEVICES=3
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

DATA_ROOT="${DATA_ROOT:-/mnt/lv2/data/ad/ADReSS}"
MODEL_PATH="${MODEL_PATH:-/home/sunqi/models/wavlm-base}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/new/update_results_adress_auto4}"

echo "=============================================="
echo "ADReSS: pathology(e2e) + random (no Optuna)"
echo "=============================================="
echo "Data:   $DATA_ROOT"
echo "Model:  $MODEL_PATH"
echo "Output: $OUTPUT_ROOT"
echo "=============================================="

mkdir -p "$OUTPUT_ROOT"

run_one() {
    local strategy=$1
    shift || true
    echo ""
    echo ">>> Running mask_strategy=$strategy ..."
    PYTHONPATH="$ROOT" python "$ROOT/new/long_wavlm_disease_mask_experiment.py" \
        --data_root "$DATA_ROOT" \
        --model_name_or_path "$MODEL_PATH" \
        --output_root "$OUTPUT_ROOT" \
        --mask_strategy="$strategy" \
        --binary "$@"
    echo "<<< Done mask_strategy=$strategy"
}

# pathology：端到端 frame-level attention 加权
run_one pathology
run_one random

echo ""
echo "=============================================="
echo "全部完成。实验结果: $OUTPUT_ROOT"
echo "=============================================="
