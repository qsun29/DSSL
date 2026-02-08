#!/bin/bash
# 多语料库预训练脚本
# 使用疾病帧掩码预训练
# 记录完整的 tmux 输入输出日志

# 设置错误时退出
set -e

# 数据目录
DATA_DIRS=(
    /mnt/lv3/sunqi/EWA-DB/
    /mnt/lv2/data/ad/PD-TUFH/audio
    /mnt/lv2/data/ser/IEMOCAP
    /mnt/lv2/data/ad/ADReSS
    /mnt/lv2/data/ad/Androids-Corpus
    /mnt/lv2/data/ad/NCMMSC2021_AD
)

# 脚本目录
SCRIPT_DIR="/home/sunqi/pd/wavlm-download"
cd "$SCRIPT_DIR"

# 创建日志目录
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# 生成日志文件名（包含时间戳）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/pretrain_multi_corpus_disease_${TIMESTAMP}.log"

# 记录开始时间
echo "==========================================" | tee -a "$LOG_FILE"
echo "Pretraining with DISEASE MASK" | tee -a "$LOG_FILE"
echo "Start time: $(date)" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 记录执行的完整命令
echo "Command:" | tee -a "$LOG_FILE"
echo "python pretrain_with_disease_mask_update.py \\" | tee -a "$LOG_FILE"
echo "    --data_dirs ${DATA_DIRS[*]} \\" | tee -a "$LOG_FILE"
echo "    --output_dir ./checkpoints_pretrain_multi_corpus_disease \\" | tee -a "$LOG_FILE"
echo "    --batch_size 32 \\" | tee -a "$LOG_FILE"
echo "    --num_epochs 10 \\" | tee -a "$LOG_FILE"
echo "    --learning_rate 1e-5 \\" | tee -a "$LOG_FILE"
echo "    --use_disease_mask \\" | tee -a "$LOG_FILE"
echo "    --pathology_threshold 75.0 \\" | tee -a "$LOG_FILE"
echo "    --device cuda \\" | tee -a "$LOG_FILE"
echo "    --gpu_id 5" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 使用 script 命令记录完整的交互式会话（包括输入和输出）
# 或者使用 tee 记录所有输出
python pretrain_with_disease_mask_update.py \
    --data_dirs "${DATA_DIRS[@]}" \
    --output_dir ./checkpoints_pretrain_multi_corpus_disease \
    --batch_size 32 \
    --num_epochs 10 \
    --learning_rate 1e-5 \
    --use_disease_mask \
    --pathology_threshold 75.0 \
    --device cuda \
    --gpu_id 5 2>&1 | tee -a "$LOG_FILE"

# 记录结束时间和退出状态
EXIT_CODE=${PIPESTATUS[0]}
echo "" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "End time: $(date)" | tee -a "$LOG_FILE"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Status: SUCCESS" | tee -a "$LOG_FILE"
else
    echo "Status: FAILED (exit code: $EXIT_CODE)" | tee -a "$LOG_FILE"
fi
echo "Log file saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

exit $EXIT_CODE
