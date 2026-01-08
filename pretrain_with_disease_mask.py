"""
使用疾病帧掩码的 WavLM 预训练脚本
基于原始 WavLM 架构，在预训练阶段--use_disease_mask选择使用智能疾病帧掩码还是随机掩码

随机掩码从头预训练
python /home/sunqi/pd/wavlm-download/pretrain_with_disease_mask.py \
    --data_dir /home/sunqi/pd/NCMMSC2021_AD_intelligent_masked/AD_dataset_6s/traindata \
    --output_dir ./checkpoints_pretrain_random \
    --batch_size 2 \
    --num_epochs 10 \
    --learning_rate 1e-5 \
    --pathology_threshold 75.0 \
    --device cuda \
    --gpu_id 0

疾病帧掩码从头预训练
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
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
import soundfile as sf
import sys
import csv
import json

# 可视化相关（使用无界面后端，确保在服务器上也能画图）
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# 导入原始 WavLM 模块
sys.path.insert(0, '/home/sunqi/pd/wavlm')
from WavLM import WavLM, WavLMConfig

# 导入疾病帧检测模块（GPU 版本）
from pathology_detector_gpu import (
    GPUPathologyFeatureDetector,
    generate_disease_mask_for_wavlm_gpu
)

import logging
from datetime import datetime

# 配置日志（先使用基本配置，稍后在函数中配置文件输出）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class PretrainDataset(Dataset):
    """预训练数据集（无标签，用于自监督学习）
    
    支持：
    - 单个目录（字符串或 Path）
    - 多个目录（list/tuple/set of str/Path）
    
    会在给定目录下递归收集所有 `*.wav` / `*.flac` 音频文件，
    因此可以直接用于多语种 ASR/情感/AD 等多数据集联合预训练。
    """
    def __init__(self, data_dir, sample_rate=16000, max_duration_s=6.0, device=None):
        self.sample_rate = sample_rate
        self.max_length = int(sample_rate * max_duration_s)
        self.device = device

        # 统一成目录列表
        if isinstance(data_dir, (str, Path)):
            data_dirs = [Path(data_dir)]
        elif isinstance(data_dir, (list, tuple, set)):
            data_dirs = [Path(p) for p in data_dir]
        else:
            raise ValueError(f"Unsupported data_dir type: {type(data_dir)}. "
                             f"Expected str / Path / list / tuple / set.")

        self.data_dirs = data_dirs

        # 收集所有音频文件（不再强制要求 HC/AD/MCI 目录结构）
        self.audio_files = []
        for root in self.data_dirs:
            if not root.exists():
                logger.warning(f"Data directory does not exist and will be skipped: {root}")
                continue

            count_before = len(self.audio_files)
            for pattern in ("*.wav", "*.flac"):
                for audio_path in root.rglob(pattern):
                    self.audio_files.append(str(audio_path))
            count_after = len(self.audio_files)
            logger.info(f"Collected {count_after - count_before} audio files from: {root}")
        
        logger.info(f"Found {len(self.audio_files)} audio files for pretraining in total")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        
        try:
            waveform, sr = sf.read(audio_path)
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            
            # 裁剪或填充到固定长度
            if len(waveform) > self.max_length:
                start = np.random.randint(0, len(waveform) - self.max_length)
                waveform = waveform[start:start + self.max_length]
            elif len(waveform) < self.max_length:
                pad_len = self.max_length - len(waveform)
                waveform = np.pad(waveform, (0, pad_len), mode='constant')
            
            # 归一化
            waveform = waveform / (np.abs(waveform).max() + 1e-8)
            
            tensor = torch.FloatTensor(waveform)
            if self.device is not None:
                tensor = tensor.to(self.device, non_blocking=True)
            
            return tensor
        except Exception as e:
            logger.warning(f"Error loading {audio_path}: {e}")
            error_tensor = torch.zeros(self.max_length)
            if self.device is not None:
                error_tensor = error_tensor.to(self.device, non_blocking=True)
            return error_tensor


def pretrain_with_disease_mask(
    data_dir="/home/sunqi/pd/NCMMSC2021_AD_intelligent_masked/AD_dataset_6s/traindata",
    output_dir="./checkpoints_pretrain",
    batch_size=2,
    num_epochs=10,
    learning_rate=1e-4,
    use_disease_mask=True,
    pathology_threshold=75.0,
    device="cuda",
    gpu_id=0
):
    """
    使用疾病帧掩码进行 WavLM 预训练
    
    Args:
        data_dir: 训练数据目录，可以是：
            - 单个目录（str 或 Path）
            - 多个目录（list/tuple/set of str/Path），用于多数据集联合预训练
        output_dir: 模型保存目录
        use_disease_mask: 是否使用疾病帧掩码（True=疾病帧掩码，False=随机掩码）
        pathology_threshold: 病理特征检测阈值（百分位数）
    """
    # 设置 GPU 设备
    if device == "cuda" and torch.cuda.is_available():
        if gpu_id >= torch.cuda.device_count():
            logger.warning(f"GPU {gpu_id} not available, using GPU 0 instead")
            gpu_id = 0
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        logger.info(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device("cpu")
        logger.warning("Using CPU (will be very slow)")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 配置日志文件输出
    log_dir = output_dir.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成日志文件名（包含时间戳和配置信息）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mask_type = "disease" if use_disease_mask else "random"
    log_file_path = log_dir / f"pretrain_multi_corpus_{mask_type}_{timestamp}.log"
    
    # 创建文件处理器（避免重复添加）
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file_path) for h in logger.handlers):
        file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    logger.info(f"Logging to file: {log_file_path}")
    
    # 检查 GPU 使用情况
    if device.type == "cuda":
        gpu_id = device.index if device.index is not None else 0
        logger.info(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        torch.cuda.empty_cache()
    
    # 1. 创建数据集
    logger.info("Loading dataset...")
    dataset_device = device if device.type == "cuda" else None
    dataset = PretrainDataset(data_dir, device=dataset_device)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    # 2. 创建 WavLM 模型
    logger.info("Initializing WavLM model...")
    cfg = WavLMConfig()
    cfg.mask_prob = 0.65
    cfg.mask_length = 10
    model = WavLM(cfg)
    model = model.to(device)
    
    # 3. 创建病理特征检测器（如果使用疾病帧掩码）
    detector = None
    if use_disease_mask:
        logger.info(f"Using GPU-accelerated disease frame masking (threshold={pathology_threshold}th percentile)")
        detector = GPUPathologyFeatureDetector(
            sample_rate=16000,
            threshold_percentile=pathology_threshold,
            device=device
        )
    else:
        logger.info("Using random masking (traditional WavLM)")
    
    # 4. 优化器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # 预训练损失：掩码预测损失（MSE）
    def compute_loss(model_outputs, target_features, mask_indices):
        """计算掩码预测损失"""
        predictions = model_outputs['x']  # [B, T, D]
        targets = target_features  # [B, T, D]
        
        if mask_indices.dtype != torch.bool:
            mask_indices = mask_indices.bool()
        
        num_masked = mask_indices.sum().item()
        if num_masked == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
        loss = nn.functional.mse_loss(predictions, targets, reduction='none')  # [B, T, D]
        loss_per_position = loss.mean(dim=-1)  # [B, T]
        loss = (loss_per_position * mask_indices.float()).sum() / (num_masked + 1e-8)
        
        return loss
    
    # 5. 训练循环
    logger.info("Starting pretraining...")
    logger.info(f"Total batches per epoch: {len(dataloader)}")
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    
    # 初始化 loss 历史记录
    loss_history = []
    log_file = output_dir / "pretrain_loss_history.csv"
    
    # 创建 CSV 文件并写入表头
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'avg_loss', 'use_disease_mask'])
    logger.info(f"Loss history will be saved to: {log_file}")
    
    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, waveforms in enumerate(dataloader):
            if not waveforms.is_cuda and device.type == "cuda":
                waveforms = waveforms.to(device, non_blocking=True)
            B = waveforms.size(0)
            
            # 获取特征长度 T
            with torch.no_grad():
                temp_features, padding_mask = model.extract_features(
                    source=waveforms,
                    mask=False
                )
                B, T, D = temp_features.shape
            
            # 生成掩码
            if use_disease_mask and detector is not None:
                disease_mask = generate_disease_mask_for_wavlm_gpu(
                    waveforms=waveforms,
                    detector=detector,
                    target_seq_len=T,
                    device=device
                )
                
                # 对齐 mask 长度
                if disease_mask.shape[1] != T:
                    if disease_mask.shape[1] > T:
                        disease_mask = disease_mask[:, :T]
                    else:
                        pad_length = T - disease_mask.shape[1]
                        disease_mask = F.pad(disease_mask, (0, pad_length), mode='constant', value=False)
                
                features, padding_mask = model.extract_features(
                    source=waveforms,
                    mask=True,
                    disease_mask=disease_mask
                )
                mask_indices = disease_mask
            else:
                # 使用随机掩码（传统方法）
                # 需要手动生成随机掩码，因为 extract_features 不返回 mask_indices
                # 注意：这里直接从本地 WavLM 模块中导入，而不是 `wavlm.WavLM`
                from WavLM import compute_mask_indices
                padding_mask_np = padding_mask.cpu().numpy() if padding_mask is not None else None
                rand_mask_indices_np = compute_mask_indices(
                    (B, T),
                    padding_mask_np,
                    model.cfg.mask_prob,
                    model.cfg.mask_length,
                    model.cfg.mask_selection,
                    model.cfg.mask_other,
                    min_masks=2,
                    no_overlap=model.cfg.no_mask_overlap,
                    min_space=model.cfg.mask_min_space,
                )
                mask_indices = torch.from_numpy(rand_mask_indices_np).to(device)
                
                # 提取特征并应用随机掩码
                features, padding_mask = model.extract_features(
                    source=waveforms,
                    mask=True
                )
            
            # 获取目标特征（未掩码）
            with torch.no_grad():
                target_features, _ = model.extract_features(
                    source=waveforms,
                    mask=False
                )
                if device.type == "cuda" and not target_features.is_cuda:
                    target_features = target_features.to(device)
            
            # 确保所有张量在同一设备上
            features = features.to(device)
            target_features = target_features.to(device)
            mask_indices = mask_indices.to(device)
            
            # 计算损失
            if scaler:
                with torch.amp.autocast("cuda"):
                    loss = compute_loss({'x': features}, target_features, mask_indices)
            else:
                loss = compute_loss({'x': features}, target_features, mask_indices)
            
            # 反向传播
            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % 10 == 0:
                progress = (batch_idx + 1) / len(dataloader) * 100
                logger.info(
                    f"Epoch {epoch} | Batch {batch_idx+1}/{len(dataloader)} ({progress:.1f}%) | "
                    f"Loss: {loss.item():.4f}"
                )
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        logger.info(f"Epoch {epoch} Summary | Avg Loss: {avg_loss:.4f}")
        
        # 记录 loss 历史
        loss_history.append({
            'epoch': epoch,
            'avg_loss': avg_loss,
            'use_disease_mask': use_disease_mask
        })
        
        # 保存到 CSV
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{avg_loss:.6f}", use_disease_mask])
        
        # 保存检查点
        if epoch % 5 == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'use_disease_mask': use_disease_mask,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # 保存最终模型（只保存 .pt 文件）
    try:
        final_model_path = output_dir / "pretrained_wavlm_disease_mask.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': cfg.__dict__,
            'use_disease_mask': use_disease_mask,
        }, final_model_path)
        logger.info(f"Saved final model to {final_model_path}")
    except Exception as e:
        logger.warning(f"Failed to save final model: {e}. "
                       f"You may need to manually save or convert the config for fine-tuning.")
    
    # 保存完整的 loss 历史到 JSON（用于后续分析）
    json_file = output_dir / "pretrain_loss_history.json"
    with open(json_file, 'w') as f:
        json.dump(loss_history, f, indent=2)
    logger.info(f"Saved loss history to: {json_file}")

    # 绘制并保存 loss 曲线图
    if plt is not None and len(loss_history) > 0:
        try:
            epochs = [item["epoch"] for item in loss_history]
            losses = [item["avg_loss"] for item in loss_history]

            plt.figure(figsize=(6, 4))
            plt.plot(epochs, losses, marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("Average Loss")
            plt.title("Pretraining Loss Curve")
            plt.grid(True, linestyle="--", alpha=0.5)

            fig_path = output_dir / "pretrain_loss_curve.png"
            plt.tight_layout()
            plt.savefig(fig_path)
            plt.close()
            logger.info(f"Saved loss curve figure to: {fig_path}")
        except Exception as e:
            logger.warning(f"Failed to plot loss curve: {e}")
    else:
        if plt is None:
            logger.warning("matplotlib is not available; skip plotting loss curve.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="WavLM Pretraining with Disease Frame Masking")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/sunqi/pd/NCMMSC2021_AD_intelligent_masked/AD_dataset_6s/traindata",
        help="Single training data directory (backward compatible, ignored if --data_dirs is provided)",
    )
    parser.add_argument(
        "--data_dirs",
        type=str,
        nargs="+",
        default=None,
        help="Multiple training data directories for multi-corpus pretraining",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints_pretrain",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--use_disease_mask",
        action="store_true",
        help="Use disease frame masking (otherwise use random masking)",
    )
    parser.add_argument(
        "--pathology_threshold",
        type=float,
        default=75.0,
        help="Pathology detection threshold (percentile)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU ID to use (0, 1, 2, ...). Default: 0",
    )
    
    args = parser.parse_args()

    # 优先使用多目录设置
    if args.data_dirs is not None and len(args.data_dirs) > 0:
        data_roots = args.data_dirs
        logger.info(f"Using multiple data directories: {data_roots}")
    else:
        data_roots = args.data_dir
        logger.info(f"Using single data directory: {data_roots}")

    pretrain_with_disease_mask(
        data_dir=data_roots,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        use_disease_mask=args.use_disease_mask,
        pathology_threshold=args.pathology_threshold,
        device=args.device,
        gpu_id=args.gpu_id
    )

