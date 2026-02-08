"""
用 Optuna 搜索病理特征组合的最优权重：以验证集分类准确率为目标，
在 6 维权重空间上做贝叶斯优化（TPE 采样 + 中位数剪枝），找到使下游 AD vs HC 表现最好的权重。

前沿设置：TPESampler(multivariate=True)、MedianPruner 早停、AMP 混合精度。

用法示例：
  cd /home/sunqi/pd/new
  python search_pathology_weights.py --data_root /mnt/lv2/data/ad/NCMMSC2021_AD \\
    --output_dir ./results_weight_search --n_trials 30 --epochs_per_trial 5 --use_amp

结果：--output_dir 下会生成 best_weights.json、best_logits.pt、optuna_study.db（可复现/可视化）。
"""

import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 保证可导入 new 与实验脚本
NEW_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = NEW_DIR.parent
for _d in (NEW_DIR, PROJECT_ROOT):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import optuna
from transformers import WavLMForSequenceClassification, Wav2Vec2FeatureExtractor

from gpu_pathology_detector import (
    GPUPathologyFeatureDetector,
    FEATURE_NAMES,
)
from gpu_adaptive_masker import GPUAdaptivePathologyMasker
from NCMMSC2021_AD_experiment.intelligent_masked_dataset import split_by_subject

from long_wavlm_disease_mask_experiment import (
    collect_adress_train_samples,
    collect_long_train_samples,
    LongWavLMDataset,
    train_one_epoch,
    evaluate,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _get_train_val_samples(data_root: str, binary: bool = True):
    """与 long_wavlm_disease_mask_experiment 一致的样本收集与划分。"""
    data_root = Path(data_root)
    if (data_root / "ADReSS-IS2020-data").exists():
        raw = collect_adress_train_samples(str(data_root))
        num_labels = 2
    else:
        raw = collect_long_train_samples(str(data_root), binary=binary)
        num_labels = 2 if binary else 3
    train_samples, val_samples = split_by_subject(
        raw, train_ratio=0.8, seed=42
    )
    return train_samples, val_samples, num_labels


def run_trial(
    weight_logits: torch.Tensor,
    train_samples,
    val_samples,
    num_labels: int,
    model_path: str,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    max_duration_s: float,
    pathology_threshold: float,
    mask_prob: float,
    pathology_ratio: float,
    aug_prob: float,
    use_amp: bool = True,
    trial: "optuna.Trial | None" = None,
) -> float:
    """
    用给定权重 logits 的 detector 做 pathology 掩码，训练 WavLM 分类器 epochs 轮，返回最佳验证准确率。
    若传入 trial，则每 epoch 上报中间值并支持 Optuna 剪枝（早停表现差的 trial）。
    """
    detector = GPUPathologyFeatureDetector(
        sample_rate=16000,
        threshold_percentile=pathology_threshold,
        device=device,
        init_weight_logits=weight_logits.to(device),
    )
    masker = GPUAdaptivePathologyMasker(
        detector=detector,
        mask_prob=mask_prob,
        pathology_ratio=pathology_ratio,
        random_ratio=max(0.0, 1.0 - pathology_ratio),
    )
    train_ds = LongWavLMDataset(
        train_samples,
        model_name_or_path=model_path,
        mask_strategy="pathology",
        masker=masker,
        max_duration_s=max_duration_s,
        aug_prob=aug_prob,
    )
    val_ds = LongWavLMDataset(
        val_samples,
        model_name_or_path=model_path,
        mask_strategy="none",
        masker=None,
        max_duration_s=max_duration_s,
        aug_prob=1.0,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    model = WavLMForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        problem_type="single_label_classification",
    )
    model = model.to(device)
    use_amp = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            criterion,
            epoch,
            use_amp=use_amp,
            scaler=scaler,
            gradient_accumulation_steps=1,
            scheduler=None,
        )
        val_acc = evaluate(model, val_loader, device, use_amp=use_amp)
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if trial is not None:
            trial.report(val_acc, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return best_val_acc


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Optuna 搜索病理特征组合权重（目标：验证集分类准确率）"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/mnt/lv2/data/ad/NCMMSC2021_AD",
        help="数据根目录（ADReSS 或 NCMMSC）",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="/home/sunqi/models/wavlm-base",
        help="WavLM 模型路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(NEW_DIR / "results_pathology_weight_search"),
        help="保存 best_weights.json、best_logits.pt 的目录",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=30,
        help="Optuna 试验次数",
    )
    parser.add_argument(
        "--epochs_per_trial",
        type=int,
        default=5,
        help="每次试验的训练轮数（少则快但噪声大）",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
    )
    parser.add_argument(
        "--max_duration_s",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--pathology_threshold",
        type=float,
        default=85.0,
    )
    parser.add_argument(
        "--mask_prob",
        type=float,
        default=0.40,
    )
    parser.add_argument(
        "--pathology_ratio",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--aug_prob",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        default=True,
        help="训练时使用混合精度 (AMP)",
    )
    parser.add_argument(
        "--no_amp",
        action="store_false",
        dest="use_amp",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        default=True,
        help="使用 MedianPruner 早停表现差的 trial",
    )
    parser.add_argument(
        "--no_prune",
        action="store_false",
        dest="prune",
    )
    parser.add_argument(
        "--study_storage",
        type=str,
        default=None,
        help="Optuna 数据库路径（如 optuna_study.db），用于持久化与复现",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data and splitting by subject...")
    train_samples, val_samples, num_labels = _get_train_val_samples(
        args.data_root, binary=args.binary
    )
    logger.info(
        f"Train={len(train_samples)}, Val={len(val_samples)}, num_labels={num_labels}"
    )

    # 固定随机性以便复现（每次 trial 内会重新初始化模型）
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    def objective(trial: optuna.Trial) -> float:
        # 6 维 logits，范围约 [-2, 2]，softmax 后不会过于极端
        logits = [
            trial.suggest_float(f"logit_{i}", -2.0, 2.0)
            for i in range(len(FEATURE_NAMES))
        ]
        weight_logits = torch.tensor(logits, dtype=torch.float32)
        return run_trial(
            weight_logits,
            train_samples,
            val_samples,
            num_labels,
            model_path=args.model_name_or_path,
            device=device,
            epochs=args.epochs_per_trial,
            batch_size=args.batch_size,
            lr=args.lr,
            max_duration_s=args.max_duration_s,
            pathology_threshold=args.pathology_threshold,
            mask_prob=args.mask_prob,
            pathology_ratio=args.pathology_ratio,
            aug_prob=args.aug_prob,
            use_amp=args.use_amp,
            trial=trial if args.prune else None,
        )

    # 前沿：TPE 多变量采样 + 中位数剪枝 + 可选 SQLite 持久化
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=min(10, args.n_trials),
        multivariate=True,
        seed=args.seed,
    )
    pruner = (
        optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=2,
        )
        if args.prune
        else optuna.pruners.NopPruner()
    )
    study_kw = dict(direction="maximize", sampler=sampler, pruner=pruner)
    if args.study_storage:
        study_kw["storage"] = f"sqlite:///{Path(args.study_storage).resolve()}"
        study_kw["study_name"] = "pathology_weight_search"
        study_kw["load_if_exists"] = True
    study = optuna.create_study(**study_kw)
    logger.info(
        f"Starting Optuna: n_trials={args.n_trials}, epochs_per_trial={args.epochs_per_trial}, "
        f"TPE(multivariate=True), prune={args.prune}"
    )
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # 保存最优权重（softmax 后的标量权重 + 用于初始化的 logits）
    best_logits = torch.tensor(
        [study.best_params[f"logit_{i}"] for i in range(len(FEATURE_NAMES))],
        dtype=torch.float32,
    )
    best_weights = torch.softmax(best_logits, dim=0).tolist()
    weights_dict = dict(zip(FEATURE_NAMES, best_weights))

    out_weights = out_dir / "best_weights.json"
    with open(out_weights, "w", encoding="utf-8") as f:
        json.dump(weights_dict, f, indent=2, ensure_ascii=False)
    out_logits = out_dir / "best_logits.pt"
    torch.save({"weight_logits": best_logits}, out_logits)

    logger.info(f"Best validation accuracy: {study.best_value:.4f}")
    logger.info(f"Best weights: {weights_dict}")
    logger.info(f"Saved to {out_weights} and {out_logits}")

    # 使用方式提示
    logger.info(
        "在实验中使用该权重：用 init_weight_logits=torch.load('best_logits.pt')['weight_logits'] 初始化 GPUPathologyFeatureDetector"
    )


if __name__ == "__main__":
    main()
