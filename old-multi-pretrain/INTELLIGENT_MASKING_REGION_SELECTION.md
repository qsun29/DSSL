# 智能掩码区域选择详解

## 📋 概述

智能掩码区域选择是一个**多步骤、多特征融合**的过程，通过检测6种病理特征，识别音频中包含病理信息的关键片段，然后只掩码这些片段，让模型必须学习病理信号的结构才能重建。

---

## 🔍 完整流程

```
音频波形
  ↓
提取6种病理特征
  ↓
特征归一化
  ↓
加权融合 → 病理分数
  ↓
百分位数阈值判断
  ↓
生成病理掩码（布尔数组）
  ↓
形态学平滑（闭运算+开运算）
  ↓
重采样到CNN特征长度
  ↓
生成最终掩码（80%病理 + 20%随机）
```

---

## 1️⃣ 病理特征提取

### 1.1 节奏异常（Rhythm Irregularity）

**检测方法**：
```python
# 计算能量包络
energy = librosa.feature.rms(
    y=waveform,
    frame_length=512,
    hop_length=256
)[0]

# 计算能量变化率（一阶差分）
energy_diff = np.abs(np.diff(energy))
```

**医学意义**：
- 帕金森患者：数字间隔不一致，节奏不规律
- 能量变化率大 → 节奏异常 → 病理特征

**权重**：0.2（20%）

---

### 1.2 停顿异常（Pause Abnormality）

**检测方法**：
```python
# 归一化能量
energy_normalized = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)

# 低能量 = 高停顿概率
pause_likelihood = 1.0 - energy_normalized
```

**医学意义**：
- 帕金森患者：停顿更多、更长、位置不规律
- 低能量区域 → 异常停顿 → 病理特征

**权重**：0.2（20%）

---

### 1.3 音调单调（Pitch Monotony）

**检测方法**：
```python
# 提取音调（使用pyin，最稳定）
pitches, voiced_flag, voiced_probs = librosa.pyin(
    waveform,
    fmin=librosa.note_to_hz('C2'),
    fmax=librosa.note_to_hz('C7'),
    sr=16000
)

# 计算音调变化率
pitch_variation = np.abs(np.diff(pitch_values))

# 单调性：变化率低 = 病理特征
pitch_monotony = 1.0 / (1.0 + pitch_variation)
```

**医学意义**：
- 帕金森患者：音调单调，变化减少（hypophonia）
- 音调变化率低 → 单调 → 病理特征

**权重**：0.2（20%）

**鲁棒性**：三层回退机制
1. 优先使用 `librosa.pyin()`（最稳定）
2. 如果失败，使用 `librosa.piptrack()`
3. 如果都失败，使用自相关方法

---

### 1.4 能量下降（Energy Drop）

**检测方法**：
```python
# 平滑能量包络
energy_smooth = signal.savgol_filter(energy, window_length=5, polyorder=2)

# 计算梯度
energy_gradient = np.gradient(energy_smooth)

# 负梯度 = 能量下降
energy_drop = np.maximum(0, -energy_gradient)
```

**医学意义**：
- 帕金森患者：能量变化模式异常，突然下降
- 负梯度大 → 能量下降 → 病理特征

**权重**：0.15（15%）

---

### 1.5 音质异常（Voice Quality Anomaly）

**检测方法**：
```python
# 计算STFT
stft = librosa.stft(waveform, n_fft=512, hop_length=256)
magnitude = np.abs(stft)

# 频谱质心（spectral centroid）
spectral_centroid = librosa.feature.spectral_centroid(
    S=magnitude,
    sr=16000
)[0]

# 频谱质心变化异常 = 音质问题
centroid_diff = np.abs(np.diff(spectral_centroid))
```

**医学意义**：
- 帕金森患者：音质下降，出现抖动和闪烁
- 频谱质心变化大 → 音质异常 → 病理特征

**权重**：0.15（15%）

---

### 1.6 周期性异常（Voice Quality Periodicity）

**检测方法**：
```python
# 自相关
autocorr = np.correlate(waveform, waveform, mode='full')
autocorr = autocorr[len(autocorr)//2:]

# 找到基频周期
peak_idx = signal.find_peaks(
    autocorr[sample_rate//200:],
    height=autocorr.max() * 0.1
)[0]

# 周期性低 = 音质差
periodicity = autocorr[period] / (autocorr[0] + 1e-8)
periodicity_score = 1.0 - np.clip(periodicity, 0, 1)
```

**医学意义**：
- 帕金森患者：周期性降低，音质下降
- 周期性低 → 音质差 → 病理特征

**权重**：0.1（10%）

---

## 2️⃣ 特征融合

### 2.1 归一化

每个特征都归一化到 [0, 1]：

```python
normalized = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
```

**目的**：确保不同特征在同一尺度上，可以公平地加权组合。

---

### 2.2 加权组合

```python
weights = {
    'rhythm_irregularity': 0.2,      # 20%
    'pause_likelihood': 0.2,         # 20%
    'pitch_monotony': 0.2,          # 20%
    'energy_drop': 0.15,            # 15%
    'voice_quality_anomaly': 0.15,  # 15%
    'voice_quality_periodicity': 0.1 # 10%
}

# 计算病理分数
pathology_score = np.zeros(n_frames)
for name, weight in weights.items():
    pathology_score += weight * normalized_features[name]
```

**结果**：每个帧都有一个病理分数（0-1之间），分数越高，病理特征越明显。

---

## 3️⃣ 阈值判断

### 3.1 百分位数阈值

```python
# 使用75百分位作为阈值
threshold = np.percentile(pathology_score, 75.0)

# 生成掩码：病理分数高于阈值的帧
pathology_mask = pathology_score >= threshold
```

**默认阈值**：75.0（75百分位）

**含义**：
- 只掩码病理分数最高的25%的帧
- 确保只掩码最明显的病理特征
- 避免掩码正常语音片段

**可调参数**：
- `--pathology_threshold 50.0`：掩码更多片段（包括轻微异常）
- `--pathology_threshold 75.0`：默认，掩码明显病理特征
- `--pathology_threshold 90.0`：只掩码最明显的病理特征

---

## 4️⃣ 掩码平滑

### 4.1 形态学操作

```python
from scipy.ndimage import binary_closing, binary_opening

# 闭运算：连接相近的片段
mask = binary_closing(mask, structure=np.ones(3))

# 开运算：移除太短的片段
mask = binary_opening(mask, structure=np.ones(3))
```

**目的**：
1. **连续性**：病理特征通常是连续的，确保掩码也是连续的
2. **去除噪声**：移除太短的片段（可能是误检）
3. **连接片段**：连接相近的片段（可能是同一个病理特征）

---

## 5️⃣ 掩码生成策略

### 5.1 重采样到CNN特征长度

```python
# CNN下采样率约为320
feature_len = len(waveform) // 320

# 将病理掩码重采样到特征长度
pathology_mask_resampled = _resample_mask(
    pathology_mask,
    target_len=feature_len
)
```

**重采样方法**：
- **下采样**：使用最大池化（保留任何病理特征）
- **上采样**：使用最近邻插值

---

### 5.2 混合掩码策略

```python
# 1. 病理片段掩码（主要部分，80%）
pathology_indices = np.where(pathology_mask_resampled)[0]
num_pathology_masks = int(len(pathology_indices) * 0.8 * 0.65)
selected_indices = np.random.choice(
    pathology_indices,
    size=num_pathology_masks,
    replace=False
)

# 生成span掩码（连续掩码）
for idx in selected_indices:
    span = np.random.randint(5, 21)  # 5-20帧
    start = max(0, idx - span // 2)
    end = min(feature_len, start + span)
    mask[start:end] = True

# 2. 随机掩码（补充，20%）
num_random_masks = int(feature_len * 0.65 * 0.2)
random_indices = np.random.choice(
    feature_len,
    size=num_random_masks,
    replace=False
)
```

**策略说明**：
- **80%病理掩码**：主要掩码病理片段，确保模型学习病理特征
- **20%随机掩码**：作为补充，确保覆盖和多样性
- **Span掩码**：连续掩码（5-20帧），模拟真实掩码场景

**掩码比例**：
- 总掩码比例：约65%（`mask_prob=0.65`）
- 病理掩码：65% × 80% = 52%
- 随机掩码：65% × 20% = 13%

---

## 6️⃣ 完整示例

### 6.1 代码示例

```python
from pathology_detector import PathologyFeatureDetector, AdaptivePathologyMasker
import librosa
import torch

# 加载音频
waveform, sr = librosa.load("sample.wav", sr=16000)

# 创建检测器
detector = PathologyFeatureDetector(
    sample_rate=16000,
    threshold_percentile=75.0
)

# 创建掩码生成器
masker = AdaptivePathologyMasker(
    detector=detector,
    pathology_ratio=0.8,  # 80%病理掩码
    random_ratio=0.2      # 20%随机掩码
)

# 生成掩码
mask = masker.generate_mask(
    waveform=waveform,
    target_seq_len=125,  # CNN特征长度
    device=torch.device("cuda")
)

# mask: [1, 125] 布尔掩码
# True表示该位置被掩码
```

### 6.2 可视化示例

```python
import matplotlib.pyplot as plt

# 检测病理特征
pathology_mask, features = detector.detect_pathology_segments(
    waveform,
    return_features=True
)

# 可视化
fig, axes = plt.subplots(len(features) + 2, 1, figsize=(12, 2 * (len(features) + 2)))

# 1. 波形
axes[0].plot(waveform)
axes[0].set_title("Waveform")
axes[0].set_ylabel("Amplitude")

# 2. 每个特征
for i, (name, feat) in enumerate(features.items(), 1):
    axes[i].plot(feat)
    axes[i].set_title(f"Feature: {name}")
    axes[i].set_ylabel("Score")

# 3. 病理分数（融合后）
pathology_score = np.zeros(len(features['rhythm_irregularity']))
weights = {...}  # 权重字典
for name, weight in weights.items():
    pathology_score += weight * features[name]
axes[-2].plot(pathology_score)
axes[-2].axhline(y=np.percentile(pathology_score, 75.0), color='r', linestyle='--', label='Threshold (75%)')
axes[-2].set_title("Pathology Score (Fused)")
axes[-2].set_ylabel("Score")
axes[-2].legend()

# 4. 最终掩码
axes[-1].plot(pathology_mask.astype(float))
axes[-1].set_title("Pathology Mask")
axes[-1].set_ylabel("Masked")
axes[-1].set_xlabel("Frame")

plt.tight_layout()
plt.savefig("pathology_analysis.png")
```

---

## 7️⃣ 参数调优

### 7.1 关键参数

| 参数 | 默认值 | 说明 | 调优建议 |
|------|--------|------|----------|
| `threshold_percentile` | 75.0 | 病理特征检测阈值 | 50-90，值越高越严格 |
| `pathology_ratio` | 0.8 | 病理掩码比例 | 0.6-0.9，值越高越专注 |
| `random_ratio` | 0.2 | 随机掩码比例 | 0.1-0.4，值越高越多样 |
| `mask_prob` | 0.65 | 总掩码比例 | 0.5-0.8，值越高掩码越多 |

### 7.2 调优策略

**如果掩码太少**：
- 降低 `threshold_percentile`（如50.0）
- 提高 `pathology_ratio`（如0.9）

**如果掩码太多**：
- 提高 `threshold_percentile`（如90.0）
- 降低 `pathology_ratio`（如0.6）

**如果需要更多多样性**：
- 提高 `random_ratio`（如0.3）

---

## 8️⃣ 优势总结

### 8.1 针对性学习

- ✅ **只掩码病理片段**：模型必须学习病理信号才能重建
- ✅ **解决核心问题**：直接解决"微弱特征捕捉不足"问题

### 8.2 可解释性

- ✅ **基于医学特征**：6种特征都有明确的医学意义
- ✅ **可追溯性**：可以解释为什么掩码某个区域
- ✅ **可视化证据**：可以可视化整个检测过程

### 8.3 鲁棒性

- ✅ **多层回退**：音调提取有3层回退机制
- ✅ **边界处理**：完善的边界和异常处理
- ✅ **平滑处理**：形态学操作确保掩码连续性

### 8.4 灵活性

- ✅ **可调参数**：多个参数可以调整
- ✅ **适应不同疾病**：可以调整特征权重和阈值
- ✅ **混合策略**：病理掩码+随机掩码，兼顾专注性和多样性

---

## 9️⃣ 与随机掩码的对比

### 随机掩码

```
随机选择位置 → 掩码
问题：可能掩码到正常语音片段
```

### 智能掩码

```
提取6种病理特征 → 融合 → 阈值判断 → 只掩码病理片段
优势：模型必须学习病理信号才能重建
```

---

## 🔟 总结

智能掩码区域选择是一个**多步骤、多特征融合**的过程：

1. **提取6种病理特征**：节奏、停顿、音调、能量、音质、周期性
2. **归一化和融合**：加权组合，得到病理分数
3. **阈值判断**：使用75百分位识别病理片段
4. **平滑处理**：形态学操作确保连续性
5. **混合掩码**：80%病理掩码 + 20%随机掩码

**核心优势**：只掩码含病理特征的关键片段，让模型必须学习病理信号的结构才能重建，从而解决"微弱特征捕捉不足"问题。

---

**详细技术文档已创建！** ✅

