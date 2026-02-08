#!/bin/bash
# NCMMSC2021 AD: 先做病理权重搜索，再用最优权重跑 pathology 实验
# 使用 GPU 1

set -e
export CUDA_VISIBLE_DEVICES=1
cd /home/sunqi/pd/new

echo "=============================================="
echo "NCMMSC2021 AD: 1) 权重搜索 (Optuna)"
echo "=============================================="
PYTHONPATH=/home/sunqi/pd python search_pathology_weights.py \
  --data_root /mnt/lv2/data/ad/NCMMSC2021_AD \
  --output_dir ./results_pathology_weight_search_ncmm \
  --n_trials 30 \
  --epochs_per_trial 5

echo ""
echo "=============================================="
echo "NCMMSC2021 AD: 2) Pathology 实验（使用搜索得到的最优权重）"
echo "=============================================="
cd /home/sunqi/pd
PYTHONPATH=/home/sunqi/pd python new/long_wavlm_disease_mask_experiment.py \
  --data_root /mnt/lv2/data/ad/NCMMSC2021_AD \
  --output_root new/results_ncmmsc_auto \
  --mask_strategy pathology \
  --pathology_weight_logits new/results_pathology_weight_search_ncmm/best_logits.pt \
  --binary

echo ""
echo "=============================================="
echo "NCMMSC 完成。权重搜索结果: new/results_pathology_weight_search_ncmm/"
echo "Pathology 实验结果: new/results_ncmmsc_auto/"
echo "=============================================="
