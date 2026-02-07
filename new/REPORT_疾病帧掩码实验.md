# AD vs HC 分类 + 疾病帧掩码 SpecAugment 实验报告

## 1. 实验目的与设计

本实验在 **ADReSS** 数据集上，基于预训练 **WavLM-base** 做 AD（认知下降）与 HC（认知正常）二分类，并比较三种训练策略：

| 策略 | 说明 |
|------|------|
| **none（基线）** | 不做任何掩码，直接使用原始波形训练 |
| **pathology（疾病帧智能掩码）** | 先做疾病帧检测，再对检测到的病理相关片段做智能掩码，作为 SpecAugment 式数据增强 |
| **random（随机掩码）** | 使用相同掩码比例，但掩码位置随机，作为对照 |

核心问题：**将“疾病帧检测 + 针对疾病帧的掩码”作为一种数据增强，是否比不采用该策略（以及比随机掩码）在下游分类上效果更好？**

---

## 2. 代码结构解读

### 2.1 入口与配置

- **脚本**：`new/long_wavlm_disease_mask_experiment.py`
- **默认数据**：`/mnt/lv2/data/ad/ADReSS`（ADReSS-IS2020 train 段）
- **默认模型**：`/home/sunqi/models/wavlm-base`（HuggingFace WavLM）
- **输出目录**：`new/results_long_wavlm_disease_mask/`，每次运行生成子目录 `mask_{none|pathology|random}_binary_1_{timestamp}`

### 2.2 数据流程

1. **样本收集**  
   - ADReSS：`ADReSS-IS2020-data/train/Full_wave_enhanced_audio/`  
     - `cd/` → AD，label=0  
     - `cc/` → HC，label=1  
   - 若数据根下存在 `ADReSS-IS2020-data` 则走 ADReSS 分支，否则走 NCMMSC long 分支（备用）。

2. **按说话人划分**  
   - 使用 `get_subject_id(path)` 提取说话人 ID（ADReSS 下为文件名 stem，如 `S001`）。  
   - 使用 `split_by_subject(samples, train_ratio=0.8, seed=42)` 按**人**划分 train/val，避免同一人同时出现在训练集和验证集，减轻数据泄漏。

3. **音频预处理**  
   - 单声道、16 kHz、按 `max_duration_s`（默认 10s）截断或填充。  
   - 归一化后经 `Wav2Vec2FeatureExtractor` 得到 `input_values` 送入 WavLM。

### 2.3 疾病帧检测与掩码（pathology / random）

- **检测器**：`GPUPathologyFeatureDetector`（来自 `new/gpu_pathology_detector.py`）  
  - 基于节奏不规则性、停顿、音调单调性、能量骤降、音质异常等特征，在帧级别检测“病理相关”片段。  
  - 使用阈值百分位（默认由 `--pathology_threshold` 指定，如 85）得到二值病理掩码。

- **掩码器**：`GPUAdaptivePathologyMasker`（来自 `new/gpu_adaptive_masker.py`）  
  - 对每条波形调用 `generate_waveform_mask(waveform, mute_value=0.0)`，返回：  
    - `intelligent`：主要在**疾病帧**对应的时间段上做掩码（+ 少量随机掩码补足比例）。  
    - `random_masked`：相同总掩码比例，但位置完全随机。

- **策略与数据增强**  
  - `mask_strategy="pathology"`：训练时使用 `intelligent` 波形，相当于“针对疾病帧的 SpecAugment”。  
  - `mask_strategy="random"`：训练时使用 `random_masked` 波形，作为对照。  
  - **验证集始终使用原始波形（无掩码）**，保证评估一致、可比。

### 2.4 模型与训练

- **模型**：`WavLMForSequenceClassification.from_pretrained(model_path, num_labels=2)`，即加载预训练 WavLM-base 并接二分类头。
- **训练**：AdamW（lr=2e-5）、CrossEntropyLoss、梯度裁剪 1.0，默认 20 epoch，batch_size=4。
- **验证**：每个 epoch 在 val 上算准确率，保存 `best_model.pt`（按 val_acc 最优）。

### 2.5 输出文件

每个实验子目录下：

- `history.csv`：各 epoch 的 `train_loss`、`train_acc`、`val_acc`
- `run_info.json`：数据路径、mask_strategy、超参、`best_val_acc`、num_train/num_val
- `best_model.pt`：验证集上表现最好的模型权重与配置
- `train.log`：训练过程日志

---

## 3. 实验结果汇总

### 3.1 实验配置（三组一致）

| 项目 | 取值 |
|------|------|
| 数据 | ADReSS train（Full_wave_enhanced_audio） |
| 总样本 | 108（cc + cd） |
| 划分 | 按说话人 80% / 20%，train=86，val=22 |
| 模型 | WavLM-base，二分类 |
| batch_size | 4 |
| num_epochs | 20 |
| lr | 2e-5 |
| max_duration_s | 10 |

### 3.2 最佳验证准确率对比

| 策略 | best_val_acc | 对应目录 |
|------|--------------|----------|
| **none（基线）** | **68.18%** | mask_none_binary_1_20260207_124253 |
| **pathology（疾病帧掩码）** | **72.73%** | mask_pathology_binary_1_20260207_124649 |
| **random（随机掩码）** | **72.73%** | mask_random_binary_1_20260207_124720 |

### 3.3 训练曲线摘要

- **none**  
  - 训练损失随 epoch 明显下降，训练准确率后期接近 98%+，验证准确率在 15 轮达到最佳 68.18%，存在一定过拟合倾向。

- **pathology**  
  - 训练损失与训练准确率波动较大（约 0.45–0.58），验证准确率在 19 轮达到 72.73%。增强后模型更难拟合训练集，但验证集表现优于基线。

- **random**  
  - 训练准确率多数在 0.45–0.55，验证准确率在第 3 轮即达到 72.73%，之后在 0.45–0.59 之间波动；最佳验证准确率与 pathology 相同（72.73%），但曲线更不稳定。

### 3.4 数值对比小结

- 疾病帧掩码（pathology）与随机掩码（random）的 **best_val_acc 均高于基线 none**（72.73% vs 68.18%），约 +4.5 个百分点。
- pathology 与 random 在本轮实验中 **best_val_acc 相同**（72.73%），且 val 样本仅 22 条，差异需更多次运行或更大验证集进一步验证。

---

## 4. 结论与讨论

1. **相对基线**  
   在 ADReSS 上，使用 WavLM-base 做 AD/HC 二分类时，**无论疾病帧智能掩码还是随机掩码，作为 SpecAugment 均比不做掩码（none）的验证准确率更高**（72.73% vs 68.18%）。

2. **疾病帧 vs 随机掩码**  
   当前单次实验中，pathology 与 random 的最佳验证准确率相同；从曲线看，pathology 在后期更稳定达到 72.73%。要判断“疾病帧掩码是否优于随机掩码”，建议：  
   - 固定 seed 做多轮实验，比较 mean±std；  
   - 或使用官方 test set 做一次留出评估。

3. **数据与泛化**  
   按说话人划分后 val 仅 22 人，验证准确率方差会较大；若有机会在 ADReSS 官方测试集上评估，结论会更可靠。

4. **实现要点**  
   - 疾病帧检测 + 掩码仅用于**训练集**输入，验证/测试用原始波形。  
   - 按说话人划分避免同一人同时出现在 train 与 val，评估更合理。

---

## 5. 复现命令示例

```bash
cd /home/sunqi/pd

# 基线（无掩码）
python new/long_wavlm_disease_mask_experiment.py --data_root /mnt/lv2/data/ad/ADReSS --mask_strategy none

# 疾病帧智能掩码
python new/long_wavlm_disease_mask_experiment.py --data_root /mnt/lv2/data/ad/ADReSS --mask_strategy pathology

# 随机掩码对照
python new/long_wavlm_disease_mask_experiment.py --data_root /mnt/lv2/data/ad/ADReSS --mask_strategy random
```

指定 GPU 示例：`CUDA_VISIBLE_DEVICES=0 python new/long_wavlm_disease_mask_experiment.py ...`
