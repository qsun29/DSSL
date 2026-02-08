"""
AD vs HC 分类 + 疾病帧掩码 SpecAugment 实验：

1. 基线：不做疾病掩码（不使用 SpecAugment）
2. 疾病掩码：先做疾病帧检测，再对疾病帧做智能掩码，作为 SpecAugment（数据增强），
   比较下游分类性能是否提升。

数据（默认）：ADReSS
    /mnt/lv2/data/ad/ADReSS/ADReSS-IS2020-data/train/Full_wave_enhanced_audio/
    - cc/  -> HC (label=1)
    - cd/  -> AD / 认知下降 (label=0)

模型：
    /home/sunqi/models/wavlm-base  (HuggingFace WavLM 预训练权重)

依赖（均位于 new/ 下）：
    - gpu_pathology_detector.py  (GPUPathologyFeatureDetector)
    - gpu_adaptive_masker.py    (GPUAdaptivePathologyMasker)
    说话人划分仍使用 NCMMSC2021_AD_experiment.intelligent_masked_dataset。
"""

import os
import sys
import json
import csv
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import torchaudio

# 保证可以导入 new 下模块（gpu_*）及 NCMMSC2021_AD_experiment（intelligent_masked_dataset）
NEW_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _dir in (NEW_DIR, PROJECT_ROOT):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

from transformers import WavLMForSequenceClassification, Wav2Vec2FeatureExtractor

from gpu_pathology_detector import GPUPathologyFeatureDetector
from gpu_adaptive_masker import GPUAdaptivePathologyMasker
from NCMMSC2021_AD_experiment.intelligent_masked_dataset import (
    get_subject_id,
    split_by_subject,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


DEFAULT_DATA_ROOT = "/mnt/lv2/data/ad/ADReSS"
DEFAULT_MODEL_PATH = "/home/sunqi/models/wavlm-base"
DEFAULT_OUTPUT_ROOT = str(PROJECT_ROOT / "new" / "results_long_wavlm_disease_mask")


def collect_adress_train_samples(data_root: str) -> List[Tuple[str, int]]:
    """
    从 ADReSS 训练集收集样本。

    ADReSS-IS2020-data/train/Full_wave_enhanced_audio/
    - cc/  -> 认知正常 (HC), label=1
    - cd/  -> 认知下降 (AD), label=0
    """
    root = Path(data_root)
    audio_root = root / "ADReSS-IS2020-data" / "train" / "Full_wave_enhanced_audio"
    if not audio_root.exists():
        raise FileNotFoundError(f"ADReSS train audio not found: {audio_root}")

    samples: List[Tuple[str, int]] = []
    cc_dir = audio_root / "cc"
    cd_dir = audio_root / "cd"

    if cd_dir.exists():
        for wav in sorted(cd_dir.glob("*.wav")):
            samples.append((str(wav.resolve()), 0))  # AD
    if cc_dir.exists():
        for wav in sorted(cc_dir.glob("*.wav")):
            samples.append((str(wav.resolve()), 1))  # HC

    if not samples:
        raise RuntimeError(f"No wav files under {audio_root}")

    logger.info(
        f"Collected {len(samples)} samples from ADReSS train "
        f"(cc={cc_dir.exists()}, cd={cd_dir.exists()})"
    )
    return samples


def collect_long_train_samples(
    data_root: str,
    binary: bool = True,
) -> List[Tuple[str, int]]:
    """
    从 NCMMSC2021 AD_dataset_long/train/{AD,HC(,MCI)} 收集样本（备用）。
    binary=True: AD=0, HC=1。
    """
    root = Path(data_root)
    train_root = root / "AD_dataset_long" / "train"
    if not train_root.exists():
        raise FileNotFoundError(f"Train root not found: {train_root}")

    samples: List[Tuple[str, int]] = []
    ad_dir = train_root / "AD"
    hc_dir = train_root / "HC"
    mci_dir = train_root / "MCI"

    if binary:
        if ad_dir.exists():
            for wav in sorted(ad_dir.glob("*.wav")):
                samples.append((str(wav.resolve()), 0))
        if hc_dir.exists():
            for wav in sorted(hc_dir.glob("*.wav")):
                samples.append((str(wav.resolve()), 1))
    else:
        if ad_dir.exists():
            for wav in sorted(ad_dir.glob("*.wav")):
                samples.append((str(wav.resolve()), 0))
        if mci_dir.exists():
            for wav in sorted(mci_dir.glob("*.wav")):
                samples.append((str(wav.resolve()), 1))
        if hc_dir.exists():
            for wav in sorted(hc_dir.glob("*.wav")):
                samples.append((str(wav.resolve()), 2))

    if not samples:
        raise RuntimeError(f"No samples found under {train_root}")
    logger.info(f"Collected {len(samples)} samples from NCMMSC long/train")
    return samples


class LongWavLMDataset(Dataset):
    """
    long/train AD vs HC 数据集。

    mask_strategy:
        "none"      - 不做掩码
        "pathology" - 使用 GPUAdaptivePathologyMasker 的 intelligent 波形
        "random"    - 使用 random_masked 波形（随机掩码）
    aug_prob: 每条样本以该概率应用掩码，否则用原始波形（混合增强）。
    """

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        model_name_or_path: str,
        mask_strategy: str = "none",
        masker: GPUAdaptivePathologyMasker = None,
        sample_rate: int = 16000,
        max_duration_s: float = 10.0,
        aug_prob: float = 1.0,
    ):
        self.samples = samples
        self.sample_rate = sample_rate
        self.max_length = int(sample_rate * max_duration_s)
        self.mask_strategy = mask_strategy
        self.masker = masker
        self.aug_prob = aug_prob
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name_or_path
        )

        if self.mask_strategy != "none" and self.masker is None:
            raise ValueError("mask_strategy != 'none' 但 masker 为 None")

    def __len__(self):
        return len(self.samples)

    def _load_waveform(self, path: str) -> np.ndarray:
        waveform, sr = sf.read(path)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        waveform = waveform.astype(np.float32)

        if sr != self.sample_rate:
            # 使用 torchaudio 重采样
            wav_tensor = torch.tensor(waveform, dtype=torch.float32)
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            wav_tensor = resampler(wav_tensor)
            waveform = wav_tensor.cpu().numpy()

        # 归一化到 [-1, 1] 附近
        max_abs = np.max(np.abs(waveform)) + 1e-8
        waveform = waveform / max_abs
        return waveform

    def _apply_mask(self, waveform: np.ndarray) -> np.ndarray:
        if self.mask_strategy == "none":
            return waveform
        if self.aug_prob < 1.0 and np.random.rand() >= self.aug_prob:
            return waveform
        intelligent, random_masked = self.masker.generate_waveform_mask(
            waveform, mute_value=0.0
        )
        if self.mask_strategy == "pathology":
            return intelligent
        elif self.mask_strategy == "random":
            return random_masked
        else:
            raise ValueError(f"Unknown mask_strategy: {self.mask_strategy}")

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        waveform = self._load_waveform(path)
        waveform = self._apply_mask(waveform)

        # 截断 / 填充到固定长度（秒）
        if len(waveform) > self.max_length:
            waveform = waveform[: self.max_length]
        elif len(waveform) < self.max_length:
            pad_len = self.max_length - len(waveform)
            waveform = np.pad(waveform, (0, pad_len), mode="constant")

        inputs = self.feature_extractor(
            waveform,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        return {
            "input_values": inputs.input_values.squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def train_one_epoch(
    model: WavLMForSequenceClassification,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
    epoch: int,
    *,
    use_amp: bool = False,
    scaler: "torch.amp.GradScaler | None" = None,
    gradient_accumulation_steps: int = 1,
    scheduler: "torch.optim.lr_scheduler.LRScheduler | None" = None,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    step = 0

    for batch in dataloader:
        input_values = batch["input_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(input_values=input_values)
            logits = outputs.logits
            loss = criterion(logits, labels) / gradient_accumulation_steps

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        step += 1
        if step % gradient_accumulation_steps == 0:
            if scaler is not None and use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item() * gradient_accumulation_steps * labels.size(0)
        preds = logits.detach().argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)

    avg_loss = total_loss / total_count
    avg_acc = total_correct / total_count
    logger.info(
        f"Epoch {epoch} TRAIN | loss={avg_loss:.4f}, acc={avg_acc:.4f}"
    )
    return avg_loss, avg_acc


def evaluate(
    model: WavLMForSequenceClassification,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
) -> float:
    model.eval()
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for batch in dataloader:
            input_values = batch["input_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(input_values=input_values)
                logits = outputs.logits
            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_count += labels.size(0)

    acc = total_correct / total_count if total_count > 0 else 0.0
    logger.info(f"VALIDATION | acc={acc:.4f}")
    return acc


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="NCMMSC2021_AD long/train + WavLM-base + 疾病掩码 SpecAugment 实验"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=DEFAULT_DATA_ROOT,
        help="数据根目录：ADReSS 用 /mnt/lv2/data/ad/ADReSS；NCMMSC 用含 AD_dataset_long 的根目录",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="WavLM-base 模型目录（HuggingFace 格式）",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help="实验结果根目录",
    )
    parser.add_argument(
        "--mask_strategy",
        type=str,
        default="none",
        choices=["none", "pathology", "random"],
        help="SpecAugment 策略：none=不掩码，pathology=疾病帧智能掩码，random=随机掩码",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        default=True,
        help="只做 AD vs HC 二分类（丢弃 MCI）",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="按说话人划分时的训练集比例",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="batch size",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=20,
        help="训练轮数",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="学习率",
    )
    parser.add_argument(
        "--max_duration_s",
        type=float,
        default=10.0,
        help="每段截断/填充到的秒数",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="设备",
    )
    parser.add_argument("--mask_prob", type=float, default=0.40, help="掩码比例（0~1）")
    parser.add_argument("--pathology_ratio", type=float, default=0.9, help="智能掩码中病理帧占比")
    parser.add_argument("--pathology_threshold", type=float, default=85.0, help="病理帧检测阈值（百分位）")
    parser.add_argument("--aug_prob", type=float, default=1.0, help="每条样本应用掩码的概率（0~1）")
    parser.add_argument(
        "--pathology_weight_logits",
        type=str,
        default=None,
        help="搜索得到的最优权重 logits 文件路径（如 best_logits.pt），用于 init_weight_logits",
    )
    # 前沿训练选项：混合精度、编译、学习率调度
    parser.add_argument(
        "--use_amp",
        action="store_true",
        default=True,
        help="使用自动混合精度 (AMP) 加速（默认开启，CUDA 时有效）",
    )
    parser.add_argument(
        "--no_amp",
        action="store_false",
        dest="use_amp",
        help="关闭 AMP",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="使用 torch.compile 编译模型（PyTorch 2.0+，可加速）",
    )
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="cosine",
        choices=["none", "cosine", "linear"],
        help="学习率调度：cosine 带 warmup（推荐）",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="warmup 占总步数比例（仅当 lr_schedule 非 none 时）",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="梯度累积步数，等效更大 batch",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader 进程数（0=主进程加载）",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    model_path = Path(args.model_name_or_path)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    exp_name = f"mask_{args.mask_strategy}_binary_{int(args.binary)}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / f"{exp_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "train.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    logger.info(f"Data root: {data_root}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Mask strategy: {args.mask_strategy}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    # 1) 收集训练样本（ADReSS 或 NCMMSC long）
    if (data_root / "ADReSS-IS2020-data").exists():
        raw_samples = collect_adress_train_samples(str(data_root))
        num_labels_override = 2  # ADReSS 仅 cc/cd 二分类
    else:
        raw_samples = collect_long_train_samples(str(data_root), binary=args.binary)
        num_labels_override = 2 if args.binary else 3

    # 2) 按说话人划分 train / val，避免同一人同时出现在训练和验证集中
    train_samples, val_samples = split_by_subject(
        raw_samples, train_ratio=args.train_ratio, seed=42
    )
    logger.info(
        f"Split by subject: train={len(train_samples)} samples, "
        f"val={len(val_samples)} samples"
    )

    # 3) 构建疾病掩码器（如果需要）
    masker = None
    if args.mask_strategy in ("pathology", "random"):
        init_logits = None
        if getattr(args, "pathology_weight_logits", None):
            p = Path(args.pathology_weight_logits)
            if p.exists():
                ckpt = torch.load(p, map_location="cpu", weights_only=True)
                init_logits = ckpt.get("weight_logits", ckpt)
                logger.info(f"Using pathology weights from {p}")
        detector = GPUPathologyFeatureDetector(
            sample_rate=16000,
            threshold_percentile=args.pathology_threshold,
            device=args.device,
            init_weight_logits=init_logits,
        )
        random_ratio = max(0.0, 1.0 - args.pathology_ratio)
        masker = GPUAdaptivePathologyMasker(
            detector=detector,
            mask_prob=args.mask_prob,
            pathology_ratio=args.pathology_ratio,
            random_ratio=random_ratio,
        )
        logger.info(
            f"Using masker: mask_prob={args.mask_prob}, pathology_ratio={args.pathology_ratio}, "
            f"pathology_threshold={args.pathology_threshold}, aug_prob={args.aug_prob}"
        )

    # 4) 构建数据集与 DataLoader
    train_ds = LongWavLMDataset(
        train_samples,
        model_name_or_path=str(model_path),
        mask_strategy=args.mask_strategy,
        masker=masker,
        max_duration_s=args.max_duration_s,
        aug_prob=args.aug_prob,
    )
    # 验证集不做掩码，保证评估一致
    val_ds = LongWavLMDataset(
        val_samples,
        model_name_or_path=str(model_path),
        mask_strategy="none",
        masker=None,
        max_duration_s=args.max_duration_s,
        aug_prob=1.0,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )

    # 5) 构建模型
    device = torch.device(args.device)
    num_labels = num_labels_override
    model = WavLMForSequenceClassification.from_pretrained(
        str(model_path),
        num_labels=num_labels,
        problem_type="single_label_classification",
    )
    model = model.to(device)

    use_amp = args.use_amp and device.type == "cuda"
    if getattr(torch, "compile", None) and args.compile:
        model = torch.compile(model, mode="reduce-overhead")
        logger.info("Model compiled with torch.compile(mode='reduce-overhead')")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = max(1, len(train_loader) // args.gradient_accumulation_steps)
    num_training_steps = steps_per_epoch * args.num_epochs
    warmup_steps = int(num_training_steps * args.warmup_ratio) if args.lr_schedule != "none" else 0
    if args.lr_schedule == "cosine" and (warmup_steps > 0 or num_training_steps > 0):
        from torch.optim.lr_scheduler import LinearLR, SequentialLR
        cosine_t_max = max(1, num_training_steps - warmup_steps)
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_t_max, eta_min=args.lr * 0.01
        )
        if warmup_steps > 0:
            warmup_sched = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
            scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[warmup_steps])
        else:
            scheduler = cosine_sched
        logger.info(f"LR schedule: cosine (total_steps={num_training_steps}, warmup={warmup_steps})")
    elif args.lr_schedule == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=0.01, total_iters=num_training_steps
        )
        logger.info("LR schedule: linear")
    else:
        scheduler = None

    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        logger.info("Training with AMP (autocast + GradScaler)")

    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    history_path = output_dir / "history.csv"
    with open(history_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_acc"])

    for epoch in range(1, args.num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            criterion,
            epoch,
            use_amp=use_amp,
            scaler=scaler,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            scheduler=scheduler,
        )
        val_acc = evaluate(model, val_loader, device, use_amp=use_amp)

        with open(history_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [epoch, f"{train_loss:.4f}", f"{train_acc:.4f}", f"{val_acc:.4f}"]
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = output_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "val_acc": best_val_acc,
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "model_name_or_path": str(model_path),
                        "num_labels": num_labels,
                        "mask_strategy": args.mask_strategy,
                    },
                },
                best_path,
            )
            logger.info(
                f"New best val_acc={best_val_acc:.4f} at epoch {epoch}, "
                f"saved to {best_path}"
            )

    # 保存运行信息
    run_info = {
        "data_root": str(data_root),
        "model_name_or_path": str(model_path),
        "output_dir": str(output_dir),
        "mask_strategy": args.mask_strategy,
        "mask_prob": args.mask_prob,
        "pathology_ratio": args.pathology_ratio,
        "pathology_threshold": args.pathology_threshold,
        "aug_prob": args.aug_prob,
        "binary": args.binary,
        "train_ratio": args.train_ratio,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "lr": args.lr,
        "max_duration_s": args.max_duration_s,
        "use_amp": use_amp,
        "compile": args.compile,
        "lr_schedule": args.lr_schedule,
        "warmup_ratio": args.warmup_ratio,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_workers": args.num_workers,
        "best_val_acc": best_val_acc,
        "num_train": len(train_samples),
        "num_val": len(val_samples),
    }
    with open(output_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)

    logger.info(f"Done. Best val_acc={best_val_acc:.4f}")
    logger.info(f"Results saved under {output_dir}")


if __name__ == "__main__":
    main()

