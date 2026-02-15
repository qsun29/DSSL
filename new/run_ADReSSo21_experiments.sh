#!/bin/bash
# ADReSSo21 diagnosis：pathology（端到端 attention）+ random
# 数据：/mnt/lv2/data/ad/ADReSS/ADReSSo21（diagnosis/train/audio/cn, ad）

set -e
export CUDA_VISIBLE_DEVICES=2
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

DATA_ROOT="${DATA_ROOT:-/mnt/lv2/data/ad/ADReSS/ADReSSo21}"
MODEL_PATH="${MODEL_PATH:-/home/sunqi/models/wavlm-base}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/new/results_adresso21}"

echo "=============================================="
echo "ADReSSo21 diagnosis: pathology(e2e) + random"
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

run_one pathology
run_one random

echo ""
echo "=============================================="
echo "全部完成。实验结果: $OUTPUT_ROOT"
echo "=============================================="
