#!/bin/bash
# 在 NCMMSC2021_AD 数据集上跑三种掩码策略：none / pathology / random
# 数据：/mnt/lv2/data/ad/NCMMSC2021_AD/AD_dataset_long/train/{AD,HC}
# 与 long_wavlm_disease_mask_experiment.py 同款代码，仅 data_root 与 output_root 不同

set -e
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

DATA_ROOT="${DATA_ROOT:-/mnt/lv2/data/ad/NCMMSC2021_AD}"
MODEL_PATH="${MODEL_PATH:-/home/sunqi/models/wavlm-base}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/new/results_ncmmsc_auto_tmp}"

echo "=============================================="
echo "NCMMSC2021_AD 三组实验 (none / pathology / random)"
echo "=============================================="
echo "Data:   $DATA_ROOT"
echo "Model:  $MODEL_PATH"
echo "Output: $OUTPUT_ROOT"
echo "=============================================="

# 确保输出根目录存在
mkdir -p "$OUTPUT_ROOT"

run_one() {
    local strategy=$1
    echo ""
    echo ">>> Running mask_strategy=$strategy ..."
    PYTHONPATH="$ROOT" python "$ROOT/new/long_wavlm_disease_mask_experiment.py" \
        --data_root "$DATA_ROOT" \
        --model_name_or_path "$MODEL_PATH" \
        --output_root "$OUTPUT_ROOT" \
        --mask_strategy="$strategy" \
        --binary
    echo "<<< Done mask_strategy=$strategy"
}

# 1) 基线：无掩码
run_one none

# 2) 疾病帧智能掩码
run_one pathology

# 3) 随机掩码对照
run_one random

echo ""
echo "=============================================="
echo "全部完成。结果目录: $OUTPUT_ROOT"
echo "=============================================="

# 若要在终端分别运行，可复制下面三条命令（先 cd 到项目根）：
# cd /home/sunqi/pd
# PYTHONPATH=/home/sunqi/pd python new/long_wavlm_disease_mask_experiment.py --data_root /mnt/lv2/data/ad/NCMMSC2021_AD --output_root new/results_ncmmsc --mask_strategy none --binary
# PYTHONPATH=/home/sunqi/pd python new/long_wavlm_disease_mask_experiment.py --data_root /mnt/lv2/data/ad/NCMMSC2021_AD --output_root new/results_ncmmsc --mask_strategy pathology --binary
# PYTHONPATH=/home/sunqi/pd python new/long_wavlm_disease_mask_experiment.py --data_root /mnt/lv2/data/ad/NCMMSC2021_AD --output_root new/results_ncmmsc --mask_strategy random --binary
