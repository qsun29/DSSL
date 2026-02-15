"""
完全GPU加速的病理特征检测器
使用torchaudio替代librosa，所有操作在GPU上执行
权重为可学习参数，通过 softmax 归一化实现自适应组合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import numpy as np
from typing import Dict, Tuple, Optional
from scipy import signal
from scipy.stats import entropy

# 特征顺序，与可学习权重一一对应
FEATURE_NAMES = [
    'low_speech_rate',
    'low_articulation_rate',
    'short_speech_segments',
    'high_npvi_rhythm',
    'voice_breaks',
]

class GPUPathologyFeatureDetector(nn.Module):
    """
    GPU加速的病理特征检测器
    使用torchaudio在GPU上提取所有特征，替代librosa
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length: int = 512,
        hop_length: int = 256,
        threshold_percentile: float = 75.0,
        device: str = 'cuda',
        weight_temperature: float = 1.0,
        init_weight_logits: Optional[torch.Tensor] = None,
        use_mlp_head: bool = False,
        use_attention: bool = False,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.threshold_percentile = threshold_percentile
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.weight_temperature = weight_temperature  # softmax 温度，>1 更平滑
        self.use_mlp_head = use_mlp_head
        self.use_attention = use_attention
        num_f = len(FEATURE_NAMES)
        # 可学习权重 logits：线性加权时使用；attention 时仍保留用于 get_weights 的兼容
        if init_weight_logits is not None:
            self.weight_logits = nn.Parameter(
                init_weight_logits.clone().detach().float()
            )
        else:
            self.weight_logits = nn.Parameter(torch.zeros(num_f))
        if use_attention:
            # frame-level attention：对 6 维特征做注意力加权，每帧学习“哪些特征重要”
            self.attention_head = nn.Sequential(
                nn.Linear(num_f, 16),
                nn.Tanh(),
                nn.Linear(16, num_f),
            )
            self.mlp_head = None
        elif use_mlp_head:
            self.mlp_head = nn.Sequential(
                nn.Linear(num_f, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            )
            self.attention_head = None
        else:
            self.mlp_head = None
            self.attention_head = None

        # GPU上的音频处理变换
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=frame_length,
            hop_length=hop_length,
            n_mels=128
        ).to(self.device)
        
        self.mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=13,
            melkwargs={'n_fft': frame_length, 'hop_length': hop_length}
        ).to(self.device)
    
    def detect_pathology_segments(
        self,
        waveform,
        return_features: bool = False
    ) -> Tuple[np.ndarray, Dict]:
        """
        在GPU上检测包含病理特征的片段
        
        Args:
            waveform: 音频波形 [seq_len] (可以是numpy array或torch tensor)
            return_features: 是否返回特征字典
        
        Returns:
            pathology_mask: 布尔数组，True表示该帧包含病理特征
            features: 特征字典（如果return_features=True）
        """
        # 转换为tensor并移到GPU
        if isinstance(waveform, np.ndarray):
            waveform_tensor = torch.FloatTensor(waveform).to(self.device)
        elif isinstance(waveform, torch.Tensor):
            waveform_tensor = waveform.to(self.device)
        else:
            raise TypeError(f"Unsupported waveform type: {type(waveform)}")
        
        # 确保是单声道
        if waveform_tensor.dim() > 1:
            waveform_tensor = waveform_tensor.mean(dim=0)
        
        # 计算帧数
        n_frames = int((len(waveform_tensor) - self.frame_length) / self.hop_length) + 1
        
        # 在GPU上提取病理特征
        features = self._extract_pathology_features_gpu(waveform_tensor)
        
        # 基于特征生成病理掩码（使用可学习权重）
        pathology_mask, pathology_score = self._generate_pathology_mask(features, n_frames)

        if return_features:
            features['pathology_score_tensor'] = pathology_score  # 可用于反向传播与损失计算
            return pathology_mask, features
        return pathology_mask
    
    def _extract_pathology_features_gpu(self, waveform_tensor: torch.Tensor) -> Dict:
        """
        在GPU上提取病理特征 (针对 AD vs HC 差异优化)
        特征包括：
        1. low_speech_rate: 语速显著下降
        2. low_articulation_rate: 发音速率显著下降
        3. short_speech_segments: 连续话段时长较短
        4. high_npvi_rhythm: 节律变异 (nPVI) 增高
        5. voice_breaks: 声带中断/断裂增多
        """
        features = {}
        
        # 确保在GPU上
        if waveform_tensor.device != self.device:
            waveform_tensor = waveform_tensor.to(self.device)
        
        # 基础特征提取
        frame_length = self.frame_length
        hop_length = self.hop_length
        n_frames = int((len(waveform_tensor) - frame_length) / hop_length) + 1
        
        # B1. 计算RMS能量 (用于VAD和音节核检测)
        if len(waveform_tensor) >= frame_length:
            pad_len = (n_frames - 1) * hop_length + frame_length - len(waveform_tensor)
            if pad_len > 0:
                waveform_padded = F.pad(waveform_tensor, (0, pad_len))
            else:
                waveform_padded = waveform_tensor
            frames = waveform_padded.unfold(0, frame_length, hop_length)
            energy_tensor = torch.sqrt(torch.mean(frames ** 2, dim=1))
            energy = energy_tensor.cpu().numpy()
        else:
            energy_tensor = torch.tensor([0.0], device=self.device)
            energy = np.array([0.0])
            
        # 简单的VAD (基于能量阈值)
        threshold = np.percentile(energy, 30) if len(energy) > 0 else 0
        is_speech = energy > threshold
        
        # B2. 检测音节核 (Syllable Nuclei) - 近似为能量局部峰值
        # 使用 scipy.signal.find_peaks 找峰值
        # 为了模拟"音节"，我们寻找能量包络的显著峰
        peaks, _ = signal.find_peaks(energy, height=threshold, distance=max(1, int(0.1 * self.sample_rate / hop_length))) # 假设音节最小间隔100ms
        
        # 创建音节脉冲序列
        syllable_pulses = np.zeros_like(energy)
        syllable_pulses[peaks] = 1.0
        
        # 窗口大小 (用于计算局部速率) - 例如 2秒
        window_size_frames = int(2.0 * self.sample_rate / hop_length)
        if window_size_frames < 1: window_size_frames = 1
        window = np.ones(window_size_frames)
        
        # --- Feature 1: Low Speech Rate ---
        # 局部语速：窗口内音节数 / 窗口时间
        # 使用卷积计算滑动窗口内的音节总数
        syllable_count_smooth = np.convolve(syllable_pulses, window, mode='same')
        # 归一化为 "每秒音节数"
        seconds_per_window = window_size_frames * hop_length / self.sample_rate
        local_speech_rate = syllable_count_smooth / seconds_per_window
        
        # 并不是直接用 speech rate，而是 "low speech rate" 为病理特征
        # 假设正常语速约为 4-5 Hz，AD显著下降。我们取反或使用高斯核衡量 "过低"
        # 这里简单处理：越低越异常 (上限截断)
        features['low_speech_rate'] = np.maximum(0, 4.0 - local_speech_rate) # 假设低于4Hz开始算慢
        
        # --- Feature 2: Low Articulation Rate ---
        # 发音速率：窗口内音节数 / (窗口内发音时间)
        speech_frames_smooth = np.convolve(is_speech.astype(float), window, mode='same')
        # 避免除以零
        articulation_time = speech_frames_smooth * hop_length / self.sample_rate
        articulation_time[articulation_time < 0.1] = 0.1 # 防止除以极小值
        
        local_articulation_rate = syllable_count_smooth / articulation_time
        # 同样，越低越异常
        features['low_articulation_rate'] = np.maximum(0, 4.5 - local_articulation_rate) 
        
        # --- Feature 3: Short Speech Segments ---
        # 连续话段时长。AD患者说话片段更短。
        # 标记每个连续语音段及其长度
        labeled_speech, num_features = signal.label(is_speech)
        segment_lengths = np.zeros_like(energy)
        
        if num_features > 0:
            # 获取每个片段的长度
            for i in range(1, num_features + 1):
                mask = (labeled_speech == i)
                length_in_frames = np.sum(mask)
                length_in_seconds = length_in_frames * hop_length / self.sample_rate
                segment_lengths[mask] = length_in_seconds
        
        # 特征：时长越短越异常 (例如小于 2秒)
        # 这是一个 "每帧" 的特征，属于短片段的帧会有高值
        features['short_speech_segments'] = np.maximum(0, 1.5 - segment_lengths) * is_speech.astype(float)
        
        # --- Feature 4: High nPVI (Rhythm Variability) ---
        # nPVI 计算相邻音节间隔的差异。需基于音节间隔 (Inter-Syllable Intervals, ISI)
        # 这是一个稀疏特征（只在音节处定义），我们需要平滑它
        peak_indices = peaks
        npvi_local = np.zeros_like(energy)
        
        if len(peak_indices) > 1:
             # 计算相邻音节间隔 (ISI)
            isis = np.diff(peak_indices) * hop_length / self.sample_rate # 秒
            
            # 计算局部 PVI
            # nPVI_k = 100 * |d_k - d_{k+1}| / ((d_k + d_{k+1})/2)
            # 我们将其映射回每个音节位置
            for i in range(len(isis) - 1):
                d_k = isis[i]
                d_next = isis[i+1]
                mean_d = (d_k + d_next) / 2 + 1e-6
                val = 100 * np.abs(d_k - d_next) / mean_d
                # 将该值赋给对应的音节附近区域
                start_idx = peak_indices[i]
                end_idx = peak_indices[i+2]
                npvi_local[start_idx:end_idx] = val / 100.0 # 归一化大概范围
        
        # 使用平滑填充
        if len(npvi_local) > 0:
             # 前向填充+平滑
            features['high_npvi_rhythm'] = signal.savgol_filter(npvi_local, min(11, len(npvi_local)), 1)
        else:
            features['high_npvi_rhythm'] = np.zeros_like(energy)

        # --- Feature 5: Voice Breaks ---
        # 声带中断/断裂：在发音段内，音调或能量突然中断
        # 使用 pitch (F0) 或 倒谱峰值 (Cepstral Peak Prominence) 突变
        # 这里简化使用频谱质心或能量的突降，且必须在 "is_speech" 内部
        
        # 计算频谱平坦度 (Spectral Flatness) - GPU
        # 较高的平坦度通常意味着噪音/无声/气声 (Voice Break 特征)
        with torch.no_grad():
             # 使用 Mel 谱图近似计算
            mel_spec = self.mel_transform(waveform_tensor.unsqueeze(0)).squeeze()
            # 几何平均 / 算术平均
            gmean = torch.exp(torch.mean(torch.log(mel_spec + 1e-10), dim=0))
            amean = torch.mean(mel_spec, dim=0) + 1e-10
            flatness = (gmean / amean).cpu().numpy()
        
        # Voice Break: 在语音段内，平坦度突然升高 (变得像噪音)
        # 或者能量突然下降但未归零
        
        # 确保只要长度一致
        if len(flatness) != len(energy):
             # 简单的重采样或截断对齐
             target_len = len(energy)
             flatness = signal.resample(flatness, target_len)

        # 定义 Voice Break：语音段内 + 高平坦度
        voice_break_score = flatness * is_speech.astype(float)
        
        # 平滑处理
        features['voice_breaks'] = signal.medfilt(voice_break_score, kernel_size=3)
        
        return features
    
    def _generate_pathology_mask(
        self,
        features: Dict,
        n_frames: int
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """基于特征生成病理掩码（二值，用于非端到端推理）。"""
        pathology_score = self._compute_pathology_score_tensor(features, n_frames)
        threshold = torch.quantile(
            pathology_score.double(),
            self.threshold_percentile / 100.0
        )
        pathology_mask = (pathology_score >= threshold).detach().cpu().numpy()
        return pathology_mask, pathology_score

    def get_pathology_scores(self, waveform_tensor: torch.Tensor) -> torch.Tensor:
        """
        返回每帧的病理分数 (n_frames,)，对 weight_logits 可导，用于端到端软掩码。
        仅对权重可导（特征来自 numpy），不对输入波形求导。
        """
        if waveform_tensor.dim() > 1:
            waveform_tensor = waveform_tensor.mean(dim=0)
        waveform_tensor = waveform_tensor.to(self.device)
        n_frames = int((len(waveform_tensor) - self.frame_length) / self.hop_length) + 1
        features = self._extract_pathology_features_gpu(waveform_tensor)
        pathology_score = self._compute_pathology_score_tensor(features, n_frames)
        return pathology_score

    def _compute_pathology_score_tensor(self, features: Dict, n_frames: int) -> torch.Tensor:
        """从特征字典计算病理分数张量（可导 w.r.t. weight_logits），不生成二值掩码。"""
        min_len = min(len(f) for f in features.values() if len(f) > 0)
        n_frames = min(n_frames, min_len)
        normalized_features = {}
        for name, feat in features.items():
            feat_trimmed = feat[:n_frames]
            if len(feat_trimmed) > 0:
                feat_min = feat_trimmed.min()
                feat_max = feat_trimmed.max()
                if feat_max > feat_min:
                    normalized_features[name] = (feat_trimmed - feat_min) / (feat_max - feat_min)
                else:
                    normalized_features[name] = np.zeros_like(feat_trimmed)
            else:
                normalized_features[name] = np.zeros(n_frames)
        features_tensor = torch.stack([
            torch.from_numpy(normalized_features[name]).float().to(self.weight_logits.device)
            for name in FEATURE_NAMES
        ], dim=1)
        if self.attention_head is not None:
            # frame-level attention：每帧对 6 维特征学权重，再加权求和
            logits = self.attention_head(features_tensor)
            attn = F.softmax(logits / (self.weight_temperature + 1e-8), dim=-1)
            pathology_score = (attn * features_tensor).sum(dim=1)
        elif self.mlp_head is not None:
            pathology_score = self.mlp_head(features_tensor).squeeze(-1)
        else:
            weights = F.softmax(self.weight_logits / self.weight_temperature, dim=0)
            pathology_score = (features_tensor * weights.unsqueeze(0)).sum(dim=1)
        return pathology_score

    def get_weights(self) -> Dict[str, float]:
        """返回当前学习到的归一化权重（线性/MLP 为 softmax；attention 为最后一层对 6 维的贡献）"""
        with torch.no_grad():
            if self.attention_head is not None:
                # 用最后一层 (16 -> 6) 的权重范数作为各特征维度的相对重要性
                last_linear = self.attention_head[-1]
                w = last_linear.weight.abs().sum(dim=0).cpu().numpy()
                w = w / (w.sum() + 1e-8)
            else:
                w = F.softmax(self.weight_logits / self.weight_temperature, dim=0).cpu().numpy()
        return dict(zip(FEATURE_NAMES, w.tolist()))

    def get_pathology_segments(
        self,
        waveform: np.ndarray,
        return_indices: bool = False
    ) -> list:
        """获取病理片段的起止位置（样本索引）"""
        pathology_mask = self.detect_pathology_segments(waveform)
        
        # 找到连续的真值区域
        segments = []
        in_segment = False
        start = 0
        
        for i, is_pathology in enumerate(pathology_mask):
            if is_pathology and not in_segment:
                start = i
                in_segment = True
            elif not is_pathology and in_segment:
                segments.append((start, i))
                in_segment = False
        
        if in_segment:
            segments.append((start, len(pathology_mask)))
        
        if return_indices:
            # 转换为样本索引
            sample_segments = []
            for start_frame, end_frame in segments:
                start_sample = start_frame * self.hop_length
                end_sample = min(
                    end_frame * self.hop_length + self.frame_length,
                    len(waveform) if isinstance(waveform, np.ndarray) else len(waveform)
                )
                sample_segments.append((start_sample, end_sample))
            return sample_segments
        
        return segments

