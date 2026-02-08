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
    'rhythm_irregularity',
    'pause_likelihood',
    'pitch_monotony',
    'energy_drop',
    'voice_quality_anomaly',
    'voice_quality_periodicity',
]

# 原手工权重，用于初始化 logits，使「未训练时」= 原版表现；训练时可在此基础上微调
PRIOR_WEIGHTS = [0.2, 0.2, 0.2, 0.15, 0.15, 0.1]


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
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.threshold_percentile = threshold_percentile
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.weight_temperature = weight_temperature  # softmax 温度，>1 更平滑
        self.use_mlp_head = use_mlp_head
        # 可学习权重 logits：可传入自定义 logits，或用原手工权重的 log 初始化
        if init_weight_logits is not None:
            self.weight_logits = nn.Parameter(
                init_weight_logits.clone().detach().float()
            )
        else:
            prior = torch.tensor(PRIOR_WEIGHTS, dtype=torch.float32)
            self.weight_logits = nn.Parameter(torch.log(prior.clamp(min=1e-8)))
        # 可选：小型 MLP 头做非线性组合（前沿），替代线性加权
        if use_mlp_head:
            self.mlp_head = nn.Sequential(
                nn.Linear(len(FEATURE_NAMES), 32),
                nn.GELU(),
                nn.Linear(32, 1),
            )
        else:
            self.mlp_head = None

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
        """在GPU上提取病理特征"""
        features = {}
        
        # 确保在GPU上
        if waveform_tensor.device != self.device:
            waveform_tensor = waveform_tensor.to(self.device)
        
        # 1. 节奏特征（基于能量包络的变化率）- GPU
        # 计算RMS能量（完全在GPU上）
        frame_length = self.frame_length
        hop_length = self.hop_length
        n_frames = int((len(waveform_tensor) - frame_length) / hop_length) + 1
        
        # 使用unfold进行批量计算（GPU加速）
        if len(waveform_tensor) >= frame_length:
            # 填充到可以整除hop_length
            pad_len = (n_frames - 1) * hop_length + frame_length - len(waveform_tensor)
            if pad_len > 0:
                waveform_padded = F.pad(waveform_tensor, (0, pad_len))
            else:
                waveform_padded = waveform_tensor
            
            # 使用unfold提取帧
            frames = waveform_padded.unfold(0, frame_length, hop_length)
            # 计算每帧的RMS能量
            energy_tensor = torch.sqrt(torch.mean(frames ** 2, dim=1))
            energy = energy_tensor.cpu().numpy()
        else:
            energy = np.array([torch.sqrt(torch.mean(waveform_tensor ** 2)).item()])
        
        energy_diff = np.abs(np.diff(energy))
        features['rhythm_irregularity'] = energy_diff
        
        # 2. 停顿检测（低能量区域）
        if len(energy) > 0:
            energy_min = energy.min()
            energy_max = energy.max()
            if energy_max > energy_min:
                energy_normalized = (energy - energy_min) / (energy_max - energy_min)
            else:
                energy_normalized = np.ones_like(energy)
            features['pause_likelihood'] = 1.0 - energy_normalized
        else:
            features['pause_likelihood'] = np.array([])
        
        # 3. 音调特征（使用GPU频谱分析）- GPU
        with torch.no_grad():
            # 计算Mel频谱
            mel_spec = self.mel_transform(waveform_tensor.unsqueeze(0))
            mel_spec = mel_spec.squeeze()
            
            # 计算频谱质心（GPU）
            freqs = torch.linspace(0, self.sample_rate // 2, mel_spec.shape[-1]).to(self.device)
            spectral_centroid = torch.sum(freqs.unsqueeze(0) * mel_spec, dim=-1) / (torch.sum(mel_spec, dim=-1) + 1e-8)
            spectral_centroid = spectral_centroid.cpu().numpy()
            
            # 计算音调变化率（单调性：变化率低 = 单调）
            if len(spectral_centroid) > 1:
                pitch_variation = np.abs(np.diff(spectral_centroid))
                # 填充到与energy相同长度
                if len(pitch_variation) < len(energy):
                    pitch_variation = np.pad(
                        pitch_variation,
                        (0, len(energy) - len(pitch_variation)),
                        mode='edge'
                    )
                elif len(pitch_variation) > len(energy):
                    pitch_variation = pitch_variation[:len(energy)]
            else:
                pitch_variation = np.zeros(len(energy))
            
            # 单调性：变化率低 = 病理特征
            features['pitch_monotony'] = 1.0 / (1.0 + pitch_variation)
        
        # 4. 能量下降（突然的能量下降）
        if len(energy) >= 5:
            from scipy import signal
            energy_smooth = signal.savgol_filter(
                energy,
                window_length=min(5, len(energy)),
                polyorder=min(2, len(energy)-1)
            )
            energy_gradient = np.gradient(energy_smooth)
            features['energy_drop'] = np.maximum(0, -energy_gradient)
        else:
            features['energy_drop'] = np.zeros(len(energy))
        
        # 5. 音质特征（基于频谱特征，使用GPU计算的mel_spec）
        # 频谱质心变化异常 = 音质问题
        if len(spectral_centroid) > 1:
            spec_variation = np.abs(np.diff(spectral_centroid))
            spec_variation = np.pad(spec_variation, (0, 1), mode='edge')
            if len(spec_variation) < len(energy):
                spec_variation = np.pad(
                    spec_variation,
                    (0, len(energy) - len(spec_variation)),
                    mode='edge'
                )
            elif len(spec_variation) > len(energy):
                spec_variation = spec_variation[:len(energy)]
        else:
            spec_variation = np.zeros(len(energy))
        features['voice_quality_anomaly'] = spec_variation
        
        # 6. 周期性特征（简化版，基于频谱稳定性）
        # 计算频谱的周期性（使用自相关）
        if mel_spec.shape[-1] > 10:
            # 对频谱做时间维度的自相关
            spec_mean = mel_spec.mean(dim=0).cpu().numpy()
            if len(spec_mean) > 10:
                autocorr = np.correlate(spec_mean, spec_mean, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                if len(autocorr) > 5:
                    # 找到周期性峰值
                    peaks = signal.find_peaks(autocorr[1:min(50, len(autocorr))], height=autocorr.max()*0.1)[0]
                    if len(peaks) > 0:
                        periodicity_score = 1.0 - min(1.0, peaks[0] / 50.0)
                    else:
                        periodicity_score = 0.5
                else:
                    periodicity_score = 0.5
            else:
                periodicity_score = 0.5
        else:
            periodicity_score = 0.5
        
        # 将周期性分数扩展到所有帧
        features['voice_quality_periodicity'] = np.full(len(energy), periodicity_score)
        
        return features
    
    def _generate_pathology_mask(
        self,
        features: Dict,
        n_frames: int
    ) -> np.ndarray:
        """基于特征生成病理掩码"""
        # 确保所有特征长度一致
        min_len = min(len(f) for f in features.values() if len(f) > 0)
        n_frames = min(n_frames, min_len)
        
        # 归一化每个特征到[0, 1]
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
        
        # 组合特征：线性加权（softmax 权重）或 MLP 头（非线性）
        features_tensor = torch.stack([
            torch.from_numpy(normalized_features[name]).float().to(self.weight_logits.device)
            for name in FEATURE_NAMES
        ], dim=1)
        if self.mlp_head is not None:
            pathology_score = self.mlp_head(features_tensor).squeeze(-1)  # (n_frames,)
        else:
            weights = F.softmax(self.weight_logits / self.weight_temperature, dim=0)  # (6,)
            pathology_score = (features_tensor * weights.unsqueeze(0)).sum(dim=1)  # (n_frames,)

        # 使用阈值生成掩码
        threshold = torch.quantile(
            pathology_score.double(),
            self.threshold_percentile / 100.0
        )
        pathology_mask = (pathology_score >= threshold).detach().cpu().numpy()

        return pathology_mask, pathology_score

    def get_weights(self) -> Dict[str, float]:
        """返回当前学习到的归一化权重（softmax 后，和为 1）"""
        with torch.no_grad():
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

