# DSSL
使用示例：

## 目前阶段就关注从头预训练阶段loss正常即可，运行文件参考两个sh，run_pretrainD.sh是疾病帧掩码从头预训练，run_pretrainR.sh是随机掩码从头预训练

下面是预训练+微调完整流程
## 1. Group4（HuggingFace里WavLM预训练模型 + 标准微调）:
   cd /home/sunqi/pd/wavlm-download
   CUDA_VISIBLE_DEVICES=2 python finetune_with_disease_mask.py \
       --data_root /home/sunqi/pd/NCMMSC2021_AD_intelligent_masked \
       --pretrained_model_path /home/sunqi/pd/wavlm-base-plus \
       --output_dir checkpoints_finetune_group4 \
       --batch_size 2 \
       --num_epochs 10 \
       --learning_rate 2e-5 \
       --device cuda


## 2. Group1（疾病帧掩码从头预训练模型 + 标准微调）:
    python /home/sunqi/pd/wavlm-download/pretrain_with_disease_mask.py \
        --data_dir /home/sunqi/pd/NCMMSC2021_AD_intelligent_masked/AD_dataset_6s/traindata \
        --output_dir ./checkpoints_pretrain_disease \
        --batch_size 2 \
        --num_epochs 10 \
        --learning_rate 1e-5 \
        --use_disease_mask \
        --pathology_threshold 75.0 \
        --device cuda \
        --gpu_id 0

    
   CUDA_VISIBLE_DEVICES=2 python finetune_with_disease_mask.py \
       --data_root /home/sunqi/pd/NCMMSC2021_AD_intelligent_masked \
       --pretrained_model_path checkpoints_pretrain_disease/pretrained_wavlm_disease_mask.pt \
       --output_dir checkpoints_finetune_group1 \
       --batch_size 16 \
       --num_epochs 10 \
       --learning_rate 2e-5 \
       --device cuda

## 3. Baseline（随机掩码从头预训练模型 + 标准微调）:
    python /home/sunqi/pd/wavlm-download/pretrain_with_disease_mask.py \
        --data_dir /home/sunqi/pd/NCMMSC2021_AD_intelligent_masked/AD_dataset_6s/traindata \
        --output_dir ./checkpoints_pretrain_random \
        --batch_size 2 \
        --num_epochs 10 \
        --learning_rate 1e-5 \
        --pathology_threshold 75.0 \
        --device cuda \
        --gpu_id 1

    cd /home/sunqi/pd/wavlm-download
    CUDA_VISIBLE_DEVICES=1 python finetune_with_disease_mask.py \
       --data_root /home/sunqi/pd/NCMMSC2021_AD_intelligent_masked \
       --pretrained_model_path checkpoints_pretrain_random/pretrained_wavlm_disease_mask.pt \
       --output_dir checkpoints_finetune_baseline \
       --batch_size 16 \
       --num_epochs 10 \
       --learning_rate 2e-5 \
       --device cuda
