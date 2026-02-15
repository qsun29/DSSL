#!/bin/bash
# NeuroVoz（帕金森 vs 健康人）：pathology（端到端 attention）+ random
# 数据：/mnt/lv3/sunqi/data（解压 NeuroVoz.zip 后含 audios/、metadata/；audios 下 HC_*.wav, PD_*.wav）
# 使用 num_workers=4 重叠数据加载与 GPU 计算；若仍慢，因 detector 内特征提取含 GPU->CPU(numpy)

set -e
export CUDA_VISIBLE_DEVICES=6
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

DATA_ROOT="${DATA_ROOT:-/mnt/lv3/sunqi/data}"
MODEL_PATH="${MODEL_PATH:-/home/sunqi/models/wavlm-base}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/new/results_neurovoz}"

echo "=============================================="
echo "NeuroVoz (PD vs HC): pathology(e2e) + random"
echo "=============================================="
echo "Data:   $DATA_ROOT (audios: HC_*.wav, PD_*.wav)"
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
        --num_workers 4 \
        --binary "$@"
    echo "<<< Done mask_strategy=$strategy"
}

run_one pathology
run_one random

echo ""
echo "=============================================="
echo "全部完成。实验结果: $OUTPUT_ROOT"
echo "=============================================="
