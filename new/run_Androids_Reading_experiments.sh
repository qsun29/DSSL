#!/bin/bash
# Androids-Corpus Reading-Task：pathology（端到端 attention）+ random
# 数据：/mnt/lv2/data/ad/Androids-Corpus/Reading-Task/audio（HC, PT）

set -e
export CUDA_VISIBLE_DEVICES=6
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

DATA_ROOT="${DATA_ROOT:-/mnt/lv2/data/ad/Androids-Corpus}"
MODEL_PATH="${MODEL_PATH:-/home/sunqi/models/wavlm-base}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/new/results_androids_reading}"

echo "=============================================="
echo "Androids-Corpus Reading-Task: pathology(e2e) + random"
echo "=============================================="
echo "Data:   $DATA_ROOT (Reading-Task/audio)"
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
