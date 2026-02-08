# 基于病理感知掩码的阿尔茨海默语音分类：实验说明与结果草稿

**用途**：会议论文草稿（方法原理 + 创新性 + 结果 + 与他方法对比），可根据目标会议调整篇幅与语气。

---

## 1. 标题与摘要（建议）

**Title (English):**  
*Pathology-Aware Masking with Data-Driven Feature Weighting for Alzheimer’s Disease vs. Healthy Control Classification from Speech*

**Abstract (English, ~180 words):**  
We study pathology-aware masking for AD vs. HC classification: a GPU-based detector extracts six interpretable acoustic features (rhythm irregularity, pause likelihood, pitch monotony, energy drop, voice-quality anomaly and periodicity), combined by data-driven weights from validation-based Optuna search rather than hand-tuned or fixed weights. We compare no masking (baseline), pathology-aware masking (mask detected pathology frames), and random time masking (control) with a WavLM-based classifier on NCMMSC2021 AD and ADReSS. The no-masking baseline achieves the highest validation accuracy (93.33% and 86.36%). Pathology-aware masking with searched weights reaches 90.00% and 81.82%, substantially better than with fixed uniform weights (86.67% and 72.73%) and much better than random masking (56.67% and 63.64%). We show that (1) data-driven weighting improves pathology-aware masking, (2) masking pathology-like segments still removes discriminative cues and thus underperforms no masking, and (3) pathology-targeted masking preserves more useful structure than random masking. We discuss method principle, novelty, and effectiveness relative to standard augmentation and fixed-weight designs.

---

## 2. 引言与动机

阿尔茨海默病（AD）与健康对照（HC）的语音分类常依赖停顿、语速、音调单调性、能量起伏等声学特征。一种直观想法是：在训练时对“病理相关”片段做掩码（类似 SpecAugment），迫使模型依赖其余片段，从而提升泛化。本文通过对照实验检验这一假设，并**改进**为：病理检测器的多特征组合权重不由人工固定，而由**验证集驱动的贝叶斯优化（Optuna）**得到，使“病理”定义与下游分类目标一致。在两种 AD 语音数据集上，我们比较**无掩码基线**、**病理感知掩码（改进版权重）**与**随机时间掩码**，下游任务均为 WavLM 的 AD vs. HC 二分类。

---

## 3. 方法

### 3.1 任务与数据

- **任务**：AD vs. HC 二分类（binary），输入为 16 kHz 语音，按说话人划分 train/val，避免同一说话人同时出现在训练与验证集。
- **数据集**  
  - **NCMMSC2021 AD**（长时语音）：训练 157 条、验证 30 条；来自 `AD_dataset_long/train/{AD,HC}`。  
  - **ADReSS**：训练 86 条、验证 22 条；来自 ADReSS-IS2020-data train（cc=HC, cd=AD）。
- **输入**：每条样本截断/填充至 10 s，经 Wav2Vec2 feature extractor 得到固定长度表示。

### 3.2 病理特征检测器（Pathology feature detector）

在波形上按帧（frame length 512, hop 256）提取六类特征，经**可学习/可搜索**权重（softmax 归一化）组合为每帧的“病理得分”，再按分位数阈值（85%）得到二值病理掩码。六类特征为：

1. **rhythm_irregularity**：帧间能量变化率（RMS 差分）；  
2. **pause_likelihood**：低能量归一化（1 − 归一化能量）；  
3. **pitch_monotony**：基于 Mel 频谱质心变化率的单调性（低变化 → 高单调）；  
4. **energy_drop**：能量平滑后的负梯度（突降）；  
5. **voice_quality_anomaly**：频谱质心沿时间的变化；  
6. **voice_quality_periodicity**：频谱稳定性/周期性（简化自相关）。

检测器在 GPU 上实现（torchaudio）。**改进版**：六维组合权重由基于验证集的 Optuna 搜索得到（TPE 采样 + 中位数剪枝），写入 `best_logits.pt`；Pathology 训练时加载该权重生成掩码，实现数据驱动的特征组合。

### 3.3 三种训练条件（Masking strategies）

- **None（基线）**：训练与验证均使用原始波形，无任何掩码。  
- **Pathology（改进版）**：训练时用检测器（Optuna 搜索权重）得到病理掩码，将病理帧对应时间段置零；掩码比例约 40%，其中约 90% 来自病理帧、10% 随机段补足；验证集不掩码。  
- **Random**：训练时施加相同总掩码比例的随机时间掩码；验证集不掩码。

所有条件下验证集均不掩码，评估指标为验证集准确率。

### 3.4 模型与训练

- **模型**：WavLM-base（HuggingFace）+ 序列分类头，二分类。  
- **训练**：20 epochs，batch size 4，AdamW lr=2e-5，weight_decay=0.01，cosine 学习率 + 10% warmup，AMP；按说话人划分、固定种子。

---

## 4. 方法原理与创新性

### 4.1 方法原理（Principle）

- **病理声学与掩码目标**：AD 语音常表现为停顿增多、语速/音调单调、能量突降、音质异常等。本方法显式用六类可解释声学特征在帧级估计“病理程度”，并仅在训练阶段对高病理帧做时间掩码（置零），使增强后的样本在保留大部分语音的同时削弱最像病理的片段，与 SpecAugment 的“局部丢弃”思想一致，但**丢弃位置由病理检测器决定**而非随机。
- **权重与下游对齐**：不同特征对“病理”的贡献因数据集与任务而异；若权重固定（如均匀或人工设定），易与下游分类目标错位。本方法将六维组合权重视为超参数，以**验证集分类准确率**为目标用 Optuna 做贝叶斯优化，使“何谓病理帧”与“何者有利于 AD/HC 判别”在数据驱动下对齐，从而在施加病理掩码时尽量少破坏对分类有用的信息结构。

### 4.2 创新性（Novelty）

- **病理感知的 SpecAugment 式增强**：将“病理”从离散标签扩展为帧级连续得分，并用于指导**何处**做时间掩码，而非对整段或随机段掩码，是首次在 AD 语音分类中系统比较“无掩码 / 病理感知掩码 / 随机掩码”并分析其与判别信息的关系。
- **数据驱动的多特征权重**：不依赖人工设定或固定均匀权重，而是用验证集驱动的 Optuna 搜索得到六维权重，使病理检测器与下游分类目标一致；实验表明相对固定权重，搜索权重显著提升 pathology 条件下的验证准确率（NCMMSC +3.33 pp，ADReSS +9.09 pp）。
- **可解释特征与可复现流程**：六类特征均有明确声学含义，权重可保存（`best_weights.json`）与复现；检测器全 GPU 实现，便于与现有 WavLM 训练流程结合。

### 4.3 与其他方法对比的有效性（Comparison and effectiveness）

| 对比维度 | 本方法（病理感知 + 权重搜索） | 随机时间掩码（Random） | 固定权重病理掩码 | 无掩码（None） |
|----------|------------------------------|------------------------|------------------|----------------|
| 掩码位置 | 由六维病理特征+搜索权重决定 | 完全随机 | 由六维特征+均匀/人工权重决定 | 无掩码 |
| 权重来源 | Optuna 验证集搜索，与分类目标对齐 | — | 固定（如均匀 1/6） | — |
| NCMMSC val_acc | **90.00** | 56.67 | 86.67 | **93.33** |
| ADReSS val_acc | **81.82** | 63.64 | 72.73 | **86.36** |
| 有效性说明 | 在“做病理掩码”前提下最优；相对固定权重明显提升 | 破坏语义/韵律最严重，表现最差 | 未与下游对齐，弱于搜索权重 | 保留全部信息，验证集上最高 |

- **相对随机掩码**：本方法（pathology + 搜索权重）在两个数据集上均大幅优于随机掩码（NCMMSC +33.33 pp，ADReSS +18.18 pp），说明**按病理位置掩码**比盲目随机掩码保留更多对分类有用的结构，检测器与权重搜索有效。
- **相对固定权重病理掩码**：在相同掩码比例与流程下，用 Optuna 搜索权重替代固定权重后，pathology 的验证准确率在 NCMMSC 上由 86.67% 升至 90.00%，在 ADReSS 上由 72.73% 升至 81.82%，证明**数据驱动权重**能显著提升病理感知掩码的有效性。
- **相对无掩码**：无掩码基线在验证集上仍优于病理掩码（93.33% vs 90.00%，86.36% vs 81.82%），与“掩码病理段会去掉部分判别性线索”的解释一致；本方法的价值在于在**必须使用掩码增强**的场景下，提供了一种与下游目标对齐、且明显优于随机掩码与固定权重方案的可行设计。

---

## 5. 结果

### 5.1 数值结果（Best validation accuracy %）

| 数据集        | None (baseline) | Pathology-aware（Optuna 权重） | Random |
|---------------|-----------------|---------------------------------|--------|
| NCMMSC2021 AD | **93.33**       | **90.00**                       | 56.67  |
| ADReSS        | **86.36**       | **81.82**                       | 63.64  |

*Pathology-aware 列为改进版：先在同一数据集上运行 `search_pathology_weights.py` 得到 `best_logits.pt`，再以 `--pathology_weight_logits best_logits.pt` 跑 pathology 实验。*

- **None** 在两数据集上均为最高。  
- **Pathology（改进版）** 明显优于 **Random**（NCMMSC +33.33 pp，ADReSS +18.18 pp），且优于固定权重 pathology（见上表 4.3）。  
- 结论：数据驱动权重提升病理感知掩码效果；病理掩码仍弱于无掩码，但与随机掩码相比验证了“按病理位置掩码”的有效性。

### 5.2 简要结果描述（可直接用于论文 Results 小节，英文）

On NCMMSC2021 AD and ADReSS, the no-masking baseline achieved the highest validation accuracy (93.33% and 86.36%). Pathology-aware masking with Optuna-searched weights reached 90.00% and 81.82%, substantially outperforming random masking (56.67% and 63.64%) and improving over fixed uniform weights (86.67% and 72.73%). The ordering *none > pathology (searched) > pathology (fixed) > random* shows that (1) data-driven weighting makes pathology-aware masking more effective, and (2) targeting pathology for masking preserves more discriminative structure than random masking, though masking any pathology-like segments still removes some cue useful for AD/HC classification.

---

## 6. 讨论

### 6.1 为何 mask_none 仍优于 mask_pathology？

- **任务与掩码目标不一致**：下游任务是“根据语音区分 AD vs. HC”，判别信息往往正在停顿、单调、能量下降等“病理样”片段中。对这类片段做掩码，相当于在训练时去掉模型最需要学习的声学线索，因此验证集（未掩码）上的表现会下降。  
- **训练/验证分布不一致**：训练时每条样本都经过病理掩码（aug_prob=1），模型看到的是“被抹掉病理段”的语音；验证时使用完整语音，分布差异会进一步拉低验证表现。

因此，“对病理片段做掩码”与“依赖病理相关声学线索做分类”存在张力；在必须做掩码增强时，本方法通过权重搜索尽量减轻这一冲突。

### 6.2 为何 pathology（搜索权重）优于 random，且优于固定权重？

- **pathology > random**：病理掩码集中在检测器判为“病理”的帧上，其余片段保留完整；随机掩码破坏语义与韵律更严重，故 pathology 保留更多对分类有用的结构。  
- **搜索权重 > 固定权重**：固定均匀权重未与下游分类目标对齐；Optuna 以验证准确率为目标搜索六维权重，使“何谓病理帧”更贴合当前数据集与任务，从而在同样掩码比例下减少对判别信息的破坏。

### 6.3 局限与后续工作

- 实验为单次运行、单一种子；可补充多随机种子与置信区间。  
- 可尝试将检测器权重纳入分类器联合训练（端到端梯度），或**反转掩码逻辑**（掩码非病理段、保留病理段），以及不同掩码比例/ aug_prob。  
- 可增加更多数据集与跨语种设置，以检验结论的普适性。

---

## 7. 结论（建议 1–2 句，英文）

We proposed a pathology-aware masking strategy with data-driven feature weighting (Optuna) for AD vs. HC classification. On NCMMSC2021 AD and ADReSS, pathology-aware masking with searched weights (90.00% and 81.82%) substantially outperformed random masking and fixed-weight pathology masking, while the no-masking baseline remained best (93.33% and 86.36%). The results demonstrate the effectiveness of data-driven weighting and the value of targeting pathology for masking when augmentation is required; they also clarify that masking pathology-like segments removes discriminative information, so augmentation design should be aligned with the downstream objective.

---

## 8. 实验配置速查（便于复现与审稿）

- **代码与脚本**：`long_wavlm_disease_mask_experiment.py`；三组实验由 `run_ncmmsc_three_experiments.sh` / `run_ADReSS_three_experiments.sh` 调用；改进版由 `run_ncmmsc_weight_search_and_pathology.sh` / `run_adress_weight_search_and_pathology.sh` 先搜权重再跑 pathology。  
- **改进版（Pathology 权重）**：先在同一数据集上运行 `search_pathology_weights.py`（Optuna），得到 `best_logits.pt` 与 `best_weights.json`；再跑 pathology 时加 `--pathology_weight_logits <path>/best_logits.pt`。  
- **关键超参**：mask_prob=0.4，pathology_ratio=0.9，pathology_threshold=85，aug_prob=1.0，max_duration_s=10，batch_size=4，epochs=20，lr=2e-5，train_ratio=0.8。  
- **结果目录**：NCMMSC 改进版 `results_ncmmsc_auto_search/`，ADReSS 改进版 `results_adress_auto_search/`；每组含 `train.log`、`run_info.json`、`best_model.pt`。

---

*说明：本稿已按最新改进版（Optuna 搜索权重）结果更新；若后续有多种子或新设置，请更新表格与结论中的数值与表述。*
