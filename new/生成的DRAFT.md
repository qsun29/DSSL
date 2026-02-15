# Pathology-Aware Masking for Speech Classification: Project Description

**用途**：根据 `new/` 下实现整理的项目说明

---

## 1. 标题与摘要（建议）

**Title (English):**  
*Pathology-Aware Frame-Level Masking with End-to-End Learnable Weights for Binary Speech Classification*

**Abstract (English, ~180 words):**  
We propose a pathology-aware masking strategy for binary speech classification (e.g., AD vs. HC, depression vs. control): a GPU-based detector extracts six interpretable acoustic features (rhythm irregularity, pause likelihood, pitch monotony, energy drop, voice-quality anomaly, periodicity) and combines them with **end-to-end learnable frame-level attention**. We compare **pathology-aware masking** (mask detected pathology frames; detector and classifier trained jointly) with **random time masking** (control) using a WavLM-based classifier. The pipeline supports multiple datasets via a unified data loader: ADReSS, ADReSSo21, NCMMSC2021 AD, ADReSS-M, Androids-Corpus Reading-Task, and E-DAIC. On AD corpora (ADReSS, ADReSSo21, NCMMSC, ADReSS-M), pathology-aware masking consistently outperforms random masking; on Androids-Corpus (depression), pathology also outperforms random. Results show that targeting pathology for masking preserves more discriminative structure than random masking. We release the code structure, data collectors, and run scripts in `new/` for full reproducibility.

---

## 2. 项目与动机

- **任务**：二分类（如 AD vs. HC、抑郁 vs. 对照），输入 16 kHz 语音，按**说话人**划分 train/val，避免同一人同时出现在训练与验证集。
- **动机**：在训练时对“病理相关”片段做时间掩码（类似 SpecAugment），迫使模型依赖其余片段；病理检测器的多特征组合**不再人工固定**，而通过**端到端与分类器联合训练**（帧级 attention）学习，使“病理”定义与下游分类目标一致。

---

## 3. 方法

### 3.1 病理特征检测器（Pathology feature detector）

- **实现**：`gpu_pathology_detector.py`（`GPUPathologyFeatureDetector`）。  
  在波形上按帧（frame length 512, hop 256）提取六类特征，经**帧级可学习 attention**（或可选 MLP/线性权重）组合为每帧“病理得分”，再按分位数阈值（默认 85%）得到二值病理掩码。  
- **六类特征**：  
  1) rhythm_irregularity（帧间能量变化）；  
  2) pause_likelihood（低能量归一化）；  
  3) pitch_monotony（基于 Mel 频谱质心变化率的单调性）；  
  4) energy_drop（能量平滑后的负梯度）；  
  5) voice_quality_anomaly（频谱质心沿时间变化）；  
  6) voice_quality_periodicity（频谱稳定性/周期性）。  
- **权重**：默认 `use_attention=True`，即帧级 attention 与 WavLM 分类器**端到端联合训练**，梯度回传到 detector 与 masker（方案 A）。可选加载预训练权重（如 Optuna 搜索得到的 `best_logits.pt`）作初始化。

### 3.2 掩码策略（Masking strategies）

- **pathology**：训练时用检测器得到病理掩码，将病理帧对应时间段置零；掩码比例约 40%（`mask_prob=0.4`），其中约 90% 来自病理帧、10% 随机段补足（`pathology_ratio=0.9`）；验证集不掩码。  
- **random**：训练时施加相同总掩码比例的随机时间掩码；验证集不掩码。  

所有条件下验证集均不掩码，评估指标为验证集准确率。

### 3.3 模型与训练

- **模型**：WavLM-base（HuggingFace）+ 序列分类头，二分类。  
- **训练**：20 epochs，batch size 4，AdamW lr=2e-5，weight_decay=0.01，cosine 学习率 + 10% warmup，AMP；按说话人划分、固定种子（42）。  
- **输入**：每条样本截断/填充至 10 s（`max_duration_s=10`），经 Wav2Vec2 feature extractor 得到固定长度表示；支持 .wav 与 .mp3（mp3 经 torchaudio 加载）。

---

## 4. 数据与数据加载

### 4.1 支持的数据集（统一入口）

数据收集逻辑集中在 **`train_sample_collectors.py`**，根据 `data_root` 目录结构自动选择数据集；主实验脚本调用 `collect_train_samples(data_root, binary=True)` 得到 `(raw_samples, num_labels)`。

| 数据集 | 识别条件 | 说明 |
|--------|----------|------|
| **ADReSS** | `ADReSS-IS2020-data` 存在 | cc=HC(1), cd=AD(0) |
| **ADReSSo21** | `diagnosis/train/audio` 存在 | cn=HC(1), ad=AD(0) |
| **E-DAIC** | `labels/train_split.csv` + `data` | PHQ_Binary → 抑郁(0)/非抑郁(1) |
| **ADReSS-M** | `train` + `training-groundtruth.csv` | dx: Control(1), ProbableAD(0)，mp3 |
| **Androids-Corpus** | `Reading-Task/audio/HC` 或 `PT` | HC(1), PT(0)，Reading-Task 朗读任务 |
| **NCMMSC2021 AD** | 否则走 long | `AD_dataset_long/train/{AD,HC[,MCI]}`，binary 时 AD=0, HC=1 |

说话人 ID 由 `NCMMSC2021_AD_experiment.intelligent_masked_dataset.get_subject_id` 从路径解析（ADReSS/NCMMSC 等用文件名约定），再经 `split_by_subject(samples, train_ratio=0.8, seed=42)` 划分。

### 4.2 各数据集脚本与结果目录

| 数据集 | 运行脚本 | 结果目录（示例） |
|--------|----------|------------------|
| ADReSS | `run_ADReSS_three_experiments.sh` | `update_results_adress_auto4` / `results_adress` |
| ADReSSo21 | `run_ADReSSo21_experiments.sh` | `results_adresso21` |
| NCMMSC2021 AD | `run_NCMMSC_three_experiments.sh` | `results_ncmmsc_auto` / `results_ncmmsc` |
| ADReSS-M | `run_ADReSS-M_experiments.sh` | `results_adressm` |
| Androids-Corpus Reading | `run_Androids_Reading_experiments.sh` | `results_androids_reading` |

每个脚本默认依次运行 **pathology** 与 **random** 两种策略。  
可选：对 NCMMSC 先运行 `search_pathology_weights.py`（Optuna）得到 `best_logits.pt`，再在 pathology 时通过 `--pathology_weight_logits` 加载；当前主流程为**端到端**，不依赖预搜权重。

---

## 5. 实验结果（Best validation accuracy）

以下为各数据集上 **pathology** 与 **random** 的 best val_acc（来自 `new/` 下已有 run 的 `train.log` / `run_info.json`），供投稿时选用与更新。

| 数据集 | 样本规模（train / val） | Pathology (e2e) | Random |
|--------|-------------------------|-----------------|--------|
| ADReSS | 86 / 22 | **86.36** | 63.64 |
| ADReSSo21 | 132 / 34 | **70.59** | 64.71 |
| NCMMSC2021 AD | 157 / 30 | **96.67** | 76.67 |
| ADReSS-M | 189 / 48 | **70.83** | ~56.25 |
| Androids-Corpus Reading | 89 / 23 | **78.26** | 69.57 |

Pathology 在所有数据集上均优于 random，说明按病理位置掩码比随机掩码保留更多对分类有用的结构。

---

## 6. 代码结构（new/ 下）

```
new/
├── long_wavlm_disease_mask_experiment.py   # 主实验：数据加载、划分、WavLM 训练、pathology / random
├── train_sample_collectors.py              # 各数据集 (path, label) 收集 + collect_train_samples 统一入口
├── gpu_pathology_detector.py               # 6 维病理特征 + 帧级 attention/权重，GPU
├── gpu_adaptive_masker.py                  # 病理掩码 + 随机掩码生成，端到端可导
├── search_pathology_weights.py             # 可选：Optuna 搜索病理权重，输出 best_logits.pt / best_weights.json
├── run_*_experiments.sh                    # 各数据集运行脚本（pathology + random）
└── results_*/                              # 每次 run 下：mask_<strategy>_binary_1_<timestamp>/ train.log, run_info.json, best_model.pt, history.csv
```

- **依赖**：`transformers`（WavLM）、`torch`、`torchaudio`、`soundfile`、`NCMMSC2021_AD_experiment.intelligent_masked_dataset`（`get_subject_id`, `split_by_subject`）。  
- **入口**：  
  - 单次实验：`PYTHONPATH=$ROOT python $ROOT/new/long_wavlm_disease_mask_experiment.py --data_root <path> --mask_strategy pathology|random --output_root <out> ...`  
  - 批量：`bash new/run_<Dataset>_experiments.sh`（可设置 `DATA_ROOT`, `OUTPUT_ROOT`, `CUDA_VISIBLE_DEVICES`）。

---

## 7. 关键超参与复现

- **掩码**：`mask_prob=0.4`，`pathology_ratio=0.9`，`pathology_threshold=85`，`aug_prob=1.0`。  
- **训练**：`max_duration_s=10`，`batch_size=4`，`num_epochs=20`，`lr=2e-5`，`train_ratio=0.8`，`lr_schedule=cosine`，`warmup_ratio=0.1`，`use_amp=True`。  
- **结果**：每组实验在 `output_root` 下生成 `mask_<strategy>_binary_1_<timestamp>/`，内含 `train.log`、`run_info.json`、`best_model.pt`、`history.csv`；`run_info.json` 含 `best_val_acc`、`num_train`、`num_val` 及全部命令行相关参数。

---

## 8. 创新点与对比（供论文书写）

- **病理感知的 SpecAugment 式增强**：用 6 维可解释声学特征在帧级估计“病理程度”，训练时仅对高病理帧做时间掩码，掩码位置由检测器决定而非随机。  
- **端到端可学习权重**：帧级 attention 与 WavLM 分类器联合训练，使“何谓病理帧”与下游分类目标在数据驱动下对齐；相对固定权重或纯随机掩码，pathology 策略在两个维度上均更合理。  
- **多数据集统一流程**：同一套 detector/masker/训练配置覆盖 AD（多语料）与抑郁（Androids-Corpus）等任务，便于泛化与对比实验。  
- **可复现**：数据加载、划分、超参与结果目录均在 `new/` 与脚本中明确，便于审稿与社区复现。

---

## 9. 结论（英文，供 Abstract/Conclusion）

We proposed a pathology-aware frame-level masking strategy with end-to-end learnable feature weighting (frame-level attention) for binary speech classification. The same pipeline was evaluated on AD (ADReSS, ADReSSo21, NCMMSC2021, ADReSS-M) and depression (Androids-Corpus Reading-Task) datasets. Pathology-aware masking consistently outperformed random masking across all corpora, demonstrating that targeting pathology for masking preserves more discriminative structure than random masking. The unified data loaders and run scripts in `new/` support full reproducibility for Interspeech 2026 submission and future extensions.

---

*说明：本稿已按 `new/` 下当前实现与多数据集结果更新；若后续有多种子或新结果，请更新表格与结论中的数值与表述。*
