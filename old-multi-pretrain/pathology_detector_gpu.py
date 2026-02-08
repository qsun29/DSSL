"""
GPU 加速的病理特征检测模块
使用 torchaudio 和 PyTorch 实现，全部在 GPU 上运行
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Optional
import logging

# torchaudio 是可选的，如果没有安装也不影响基本功能
try:
    import torchaudio
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False

logger = logging.getLogger(__name__)


class GPUPathologyFeatureDetector:
    """
    GPU 加速的病理特征检测器
    所有计算都在 GPU 上进行，大幅提升速度
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length: int = 512,
        hop_length: int = 256,
        threshold_percentile: float = 75.0,
        device: Optional[torch.device] = None,
    ):
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.threshold_percentile = threshold_percentile
        
        # 设置设备
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        logger.info(f"GPUPathologyFeatureDetector initialized on {device}")
    
    def detect_pathology_segments(
        self,
        waveform: torch.Tensor,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        检测包含病理特征的片段（GPU 版本）
        
        Args:
            waveform: 音频波形 [seq_len] 或 [1, seq_len]，在 GPU 上
            return_features: 是否返回特征字典
        
        Returns:
            pathology_mask: 布尔张量，True表示该帧包含病理特征 [n_frames]
            features: 特征字典（如果return_features=True）
            
        ⚠️ 注意：返回的是布尔掩码，不是数值特征。如果需要数值特征，请使用 return_features=True
        """
        # 确保波形在 GPU 上且是 1D
        if waveform.dim() > 1:
            waveform = waveform.squeeze()
        if not waveform.is_cuda:
            waveform = waveform.to(self.device)
        
        # 计算帧数
        n_frames = int((len(waveform) - self.frame_length) / self.hop_length) + 1
        
        # 提取各种病理特征（全部在 GPU 上）
        features = self._extract_pathology_features_gpu(waveform)
        
        # 基于特征生成病理掩码
        pathology_mask = self._generate_pathology_mask_gpu(features, n_frames)
        
        if return_features:
            return pathology_mask, features
        return pathology_mask
    
    def _extract_pathology_features_gpu(self, waveform: torch.Tensor) -> Dict:
        """在 GPU 上提取病理特征"""
        features = {}
        
        # 1. 节奏特征（基于能量包络的变化率）
        # 使用滑动窗口计算 RMS 能量
        energy = self._compute_rms_energy_gpu(waveform)  # [n_frames]
        energy_diff = torch.abs(torch.diff(energy))
        features['rhythm_irregularity'] = energy_diff
        
        # 2. 停顿检测（低能量区域）
        energy_min = energy.min()
        energy_max = energy.max()
        energy_normalized = (energy - energy_min) / (energy_max - energy_min + 1e-8)
        features['pause_likelihood'] = 1.0 - energy_normalized
        
        # 3. 音调特征（单调性）和音质特征（共享STFT计算）
        # 计算 STFT（GPU 版本）
        stft = torch.stft(
            waveform.unsqueeze(0),
            n_fft=self.frame_length,
            hop_length=self.hop_length,
            win_length=self.frame_length,
            window=torch.hann_window(self.frame_length, device=self.device),
            return_complex=True
        ).squeeze(0)  # [n_freq, n_frames]
        magnitude = torch.abs(stft)  # [n_freq, n_frames]
        
        # 提取音调（使用自相关方法，GPU 友好）
        pitch_values = self._extract_pitch_gpu(waveform)  # [n_frames] 或标量
        
        if pitch_values.numel() > 1:
            pitch_variation = torch.abs(torch.diff(pitch_values))
            # 填充到与 energy 相同长度
            if len(pitch_variation) < len(energy):
                pad_len = len(energy) - len(pitch_variation)
                # 对于1D张量，使用constant模式或手动复制最后一个值
                if len(pitch_variation) > 0:
                    last_val = pitch_variation[-1]
                    padding = last_val.repeat(pad_len)
                    pitch_variation_padded = torch.cat([pitch_variation, padding])
                else:
                    pitch_variation_padded = torch.zeros(len(energy), device=self.device)
            else:
                pitch_variation_padded = pitch_variation[:len(energy)]
            features['pitch_monotony'] = 1.0 / (1.0 + pitch_variation_padded)
        else:
            features['pitch_monotony'] = torch.ones(len(energy), device=self.device)
        
        # 4. 能量下降（突然的能量下降）
        if len(energy) >= 5:
            # 使用卷积实现 Savitzky-Golay 滤波（GPU 友好）
            energy_smooth = self._savgol_filter_gpu(energy, window_length=5, polyorder=2)
            # 计算梯度
            energy_gradient = torch.diff(energy_smooth)
            # 负梯度 = 能量下降
            features['energy_drop'] = torch.clamp(-energy_gradient, min=0.0)
            # 填充第一个元素
            features['energy_drop'] = F.pad(features['energy_drop'], (1, 0), mode='constant', value=0.0)
        else:
            features['energy_drop'] = torch.zeros(len(energy), device=self.device)
        
        # 5. 音质特征（基于频谱特征）
        # 频谱质心（spectral centroid）
        spectral_centroid = self._compute_spectral_centroid_gpu(magnitude)  # [n_frames]
        centroid_diff = torch.abs(torch.diff(spectral_centroid))
        # 手动填充最后一个值（因为1D张量的replicate模式可能不支持）
        if len(centroid_diff) > 0:
            last_val = centroid_diff[-1]
            centroid_diff_padded = torch.cat([centroid_diff, last_val.unsqueeze(0)])
        else:
            centroid_diff_padded = torch.zeros(len(energy), device=self.device)
        features['voice_quality_anomaly'] = centroid_diff_padded
        
        # 6. 周期性异常（基于自相关）
        periodicity_score = self._compute_periodicity_gpu(waveform)  # 标量
        features['voice_quality_periodicity'] = torch.full(
            (len(energy),), periodicity_score, device=self.device
        )
        
        return features
    
    def _compute_rms_energy_gpu(self, waveform: torch.Tensor) -> torch.Tensor:
        """在 GPU 上计算 RMS 能量（向量化实现）"""
        # 使用 unfold 进行高效的滑动窗口计算
        n_frames = int((len(waveform) - self.frame_length) / self.hop_length) + 1
        
        if n_frames <= 0:
            return torch.tensor([], device=self.device)
        
        # 填充波形以确保所有帧都有完整长度
        pad_length = (n_frames - 1) * self.hop_length + self.frame_length - len(waveform)
        if pad_length > 0:
            waveform_padded = F.pad(waveform, (0, pad_length), mode='constant')
        else:
            waveform_padded = waveform
        
        # 使用 unfold 提取所有帧（更高效）
        frames = waveform_padded.unfold(0, self.frame_length, self.hop_length)  # [n_frames, frame_length]
        
        # 计算每帧的 RMS（向量化）
        energy = torch.sqrt(torch.mean(frames ** 2, dim=1))
        
        return energy
    
    def _extract_pitch_gpu(self, waveform: torch.Tensor) -> torch.Tensor:
        """在 GPU 上提取音调（使用自相关方法）"""
        # 使用自相关估计基频
        # 为了效率，只使用部分波形
        max_len = min(len(waveform), self.sample_rate * 2)  # 最多2秒
        waveform_seg = waveform[:max_len]
        
        # 计算自相关（使用 FFT 加速）
        # 自相关 = IFFT(FFT(x) * conj(FFT(x)))
        fft_wav = torch.fft.rfft(waveform_seg)
        autocorr_fft = torch.fft.irfft(fft_wav * torch.conj(fft_wav))
        autocorr = autocorr_fft[:len(autocorr_fft)//2]  # 只取前半部分
        
        # 找到峰值（基频周期）
        if len(autocorr) > self.sample_rate // 50:  # 至少20ms
            # 跳过前5ms
            search_start = self.sample_rate // 200
            search_region = autocorr[search_start:]
            
            # 找到局部最大值（向量化实现）
            threshold = search_region.max() * 0.1
            
            # 使用 torch 的 diff 和比较操作找到峰值
            # 峰值条件：x[i] > x[i-1] and x[i] > x[i+1] and x[i] > threshold
            if len(search_region) >= 3:
                left_diff = search_region[1:] - search_region[:-1]  # [n-1]
                right_diff = search_region[:-1] - search_region[1:]  # [n-1]
                
                # 找到满足条件的索引（i 对应 search_region[1:-1]）
                peak_mask = (left_diff[:-1] > 0) & (right_diff[1:] > 0) & (search_region[1:-1] > threshold)
                peak_indices = torch.where(peak_mask)[0] + 1  # 调整索引
                
                if len(peak_indices) > 0:
                    period = peak_indices[0].item() + search_start
                    f0 = self.sample_rate / period
                else:
                    f0 = None
            else:
                f0 = None
            
            if f0 is not None:
                # 扩展到所有帧（简化处理）
                n_frames = int((len(waveform) - self.frame_length) / self.hop_length) + 1
                return torch.full((n_frames,), f0, device=self.device)
        
        # 如果找不到，返回默认值
        n_frames = int((len(waveform) - self.frame_length) / self.hop_length) + 1
        return torch.full((n_frames,), 100.0, device=self.device)  # 默认100Hz
    
    def _savgol_filter_gpu(self, x: torch.Tensor, window_length: int, polyorder: int) -> torch.Tensor:
        """GPU 版本的 Savitzky-Golay 滤波（简化实现）"""
        # 简化实现：使用移动平均
        if window_length > len(x):
            window_length = len(x)
        if window_length % 2 == 0:
            window_length += 1
        
        # 使用卷积实现移动平均
        kernel = torch.ones(window_length, device=self.device) / window_length
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        # 使用constant模式填充（对于1D张量）
        pad_size = window_length // 2
        x_2d = x.unsqueeze(0).unsqueeze(0)  # [1, 1, L]
        # 手动复制边界值进行填充
        if len(x) > 0:
            left_pad = x[0].repeat(pad_size)
            right_pad = x[-1].repeat(pad_size)
            x_padded = torch.cat([left_pad, x, right_pad]).unsqueeze(0).unsqueeze(0)
        else:
            x_padded = x_2d
        smoothed = F.conv1d(x_padded, kernel)
        return smoothed.squeeze()
    
    def _compute_spectral_centroid_gpu(self, magnitude: torch.Tensor) -> torch.Tensor:
        """在 GPU 上计算频谱质心"""
        # magnitude: [n_freq, n_frames]
        n_freq = magnitude.shape[0]
        freqs = torch.arange(n_freq, device=self.device, dtype=torch.float32)
        freqs = freqs * self.sample_rate / (2 * (n_freq - 1))
        freqs = freqs.unsqueeze(1)  # [n_freq, 1]
        
        # 计算加权平均
        magnitude_sum = magnitude.sum(dim=0, keepdim=True)  # [1, n_frames]
        centroid = (freqs * magnitude).sum(dim=0) / (magnitude_sum.squeeze(0) + 1e-8)
        
        return centroid
    
    def _compute_periodicity_gpu(self, waveform: torch.Tensor) -> torch.Tensor:
        """在 GPU 上计算周期性分数"""
        # 使用部分波形计算
        max_len = min(len(waveform), self.sample_rate * 2)
        waveform_seg = waveform[:max_len]
        
        # 自相关
        fft_wav = torch.fft.rfft(waveform_seg)
        autocorr_fft = torch.fft.irfft(fft_wav * torch.conj(fft_wav))
        autocorr = autocorr_fft[:len(autocorr_fft)//2]
        
        if len(autocorr) > self.sample_rate // 50:
            search_start = self.sample_rate // 200
            search_region = autocorr[search_start:]
            
            threshold = search_region.max() * 0.1
            
            # 使用向量化操作找到峰值
            if len(search_region) >= 3:
                left_diff = search_region[1:] - search_region[:-1]
                right_diff = search_region[:-1] - search_region[1:]
                
                peak_mask = (left_diff[:-1] > 0) & (right_diff[1:] > 0) & (search_region[1:-1] > threshold)
                peak_indices = torch.where(peak_mask)[0] + 1
                
                if len(peak_indices) > 0:
                    period = peak_indices[0].item() + search_start
                    periodicity = autocorr[period] / (autocorr[0] + 1e-8)
                    return 1.0 - torch.clamp(periodicity, 0, 1)
        
        return torch.tensor(0.5, device=self.device)
    
    def _generate_pathology_mask_gpu(
        self,
        features: Dict,
        n_frames: int
    ) -> torch.Tensor:
        """在 GPU 上生成病理掩码"""
        # 确保所有特征长度一致
        min_len = min(len(f) for f in features.values())
        n_frames = min(n_frames, min_len)
        
        # 归一化每个特征到[0, 1]
        normalized_features = {}
        for name, feat in features.items():
            feat_trimmed = feat[:n_frames]
            if len(feat_trimmed) > 0:
                feat_min = feat_trimmed.min()
                feat_max = feat_trimmed.max()
                if feat_max > feat_min:
                    normalized = (feat_trimmed - feat_min) / (feat_max - feat_min + 1e-8)
                else:
                    normalized = torch.zeros_like(feat_trimmed)
                normalized_features[name] = normalized
            else:
                normalized_features[name] = torch.zeros(n_frames, device=self.device)
        
        # 组合特征（加权平均）
        weights = {
            'rhythm_irregularity': 0.2,
            'pause_likelihood': 0.2,
            'pitch_monotony': 0.2,
            'energy_drop': 0.15,
            'voice_quality_anomaly': 0.15,
            'voice_quality_periodicity': 0.1,
        }
        
        pathology_score = torch.zeros(n_frames, device=self.device)
        for name, weight in weights.items():
            if name in normalized_features:
                pathology_score += weight * normalized_features[name]
        
        # ⚠️ 关键修复：确保 pathology_score 的 scale 正确
        # pathology_score 应该在 [0, 1] 范围内（因为 normalized_features 都是 [0, 1]）
        # 但为了安全，添加额外的归一化检查和诊断信息
        if len(pathology_score) > 0:
            score_min = pathology_score.min().item()
            score_max = pathology_score.max().item()
            score_mean = pathology_score.mean().item()
            score_std = pathology_score.std().item()
            
            # 诊断信息（仅在第一次或异常时输出）
            if score_max > 1.5 or score_std > 1.0:
                logger.warning(
                    f"⚠️  Pathology score scale check: "
                    f"min={score_min:.4f}, max={score_max:.4f}, mean={score_mean:.4f}, std={score_std:.4f}"
                )
            
            # 检查 scale 是否合理
            # 理论上应该在 [0, 1] 范围内，但允许小幅超出
            if score_max > 10.0 or score_std > 5.0:
                logger.error(
                    f"❌ CRITICAL: Pathology score scale is too large! "
                    f"min={score_min:.4f}, max={score_max:.4f}, mean={score_mean:.4f}, std={score_std:.4f}. "
                    f"This may cause numerical issues. Normalizing to [0, 1] range."
                )
                # 归一化到 [0, 1]（使用 min-max 归一化）
                if score_max > score_min:
                    pathology_score = (pathology_score - score_min) / (score_max - score_min + 1e-8)
                else:
                    pathology_score = torch.zeros_like(pathology_score)
            elif score_max > 1.0:
                # 如果稍微超出 [0, 1]，进行 clip
                logger.debug(f"Pathology score slightly exceeds [0, 1]: max={score_max:.4f}, clipping...")
                pathology_score = torch.clamp(pathology_score, 0.0, 1.0)
            
            # 确保最终范围在 [0, 1]
            pathology_score = torch.clamp(pathology_score, 0.0, 1.0)
        
        # 使用百分位数阈值
        threshold = torch.quantile(pathology_score, self.threshold_percentile / 100.0)
        
        # 生成掩码：病理分数高于阈值的帧
        pathology_mask = pathology_score >= threshold
        
        # 后处理：平滑掩码（使用形态学操作）
        pathology_mask = self._smooth_mask_gpu(pathology_mask)
        
        return pathology_mask
    
    def _smooth_mask_gpu(self, mask: torch.Tensor, min_span: int = 3) -> torch.Tensor:
        """在 GPU 上平滑掩码"""
        # 简化实现：使用膨胀和腐蚀
        # 膨胀：连接相近的片段
        kernel = torch.ones(min_span, device=self.device)
        # 对于1D张量，使用constant模式填充
        pad_size = min_span // 2
        mask_2d = mask.float().unsqueeze(0).unsqueeze(0)  # [1, 1, L]
        mask_expanded = F.pad(mask_2d, (pad_size, pad_size), mode='constant', value=0.0)
        dilated = F.conv1d(mask_expanded, kernel.unsqueeze(0).unsqueeze(0)) > 0
        dilated = dilated.squeeze()
        
        # 腐蚀：移除太短的片段
        dilated_2d = dilated.float().unsqueeze(0).unsqueeze(0)  # [1, 1, L]
        eroded_padded = F.pad(dilated_2d, (pad_size, pad_size), mode='constant', value=0.0)
        eroded = F.conv1d(eroded_padded, kernel.unsqueeze(0).unsqueeze(0)) >= min_span
        eroded = eroded.squeeze()
        
        return eroded.bool()


def generate_disease_mask_for_wavlm_gpu(
    waveforms: torch.Tensor,
    detector: GPUPathologyFeatureDetector,
    target_seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """
    GPU 版本的疾病帧掩码生成（批量处理，更快）
    
    ⚠️ 重要：此函数生成的 mask 长度基于 target_seq_len，但实际 WavLM encoder 输出
    的时间步数可能不同（由于 CNN subsampling、padding 等）。调用方必须在 forward
    中验证并修正 mask 长度，使用 encoder 输出的实际时间步数。
    
    Args:
        waveforms: [B, L] 音频波形，在 GPU 上
        detector: GPUPathologyFeatureDetector 实例
        target_seq_len: 目标序列长度（特征长度）- 注意：这应该是 encoder 的实际输出长度
        device: 设备
    
    Returns:
        disease_mask: [B, T] 布尔掩码，在 GPU 上（T = target_seq_len，但可能需要在 forward 中修正）
    """
    batch_size = waveforms.size(0)
    disease_masks = []
    
    # 批量处理（每个样本独立处理，但都在 GPU 上）
    for i in range(batch_size):
        waveform = waveforms[i]  # [L]，已经在 GPU 上
        
        # 检测疾病帧（全部在 GPU 上）
        # ⚠️ 注意：detect_pathology_segments 内部使用的 n_frames 计算基于 STFT 参数
        # 这个 n_frames 不一定等于 WavLM encoder 的实际输出时间步数
        # 但我们会通过重采样来对齐到 target_seq_len（encoder 的实际输出长度）
        pathology_mask = detector.detect_pathology_segments(waveform)  # [n_frames_stft]
        
        # 改进的重采样逻辑：使用线性插值确保时间对齐更准确
        if len(pathology_mask) == 0:
            resampled = torch.zeros(target_seq_len, dtype=torch.bool, device=device)
        elif len(pathology_mask) == target_seq_len:
            resampled = pathology_mask
        elif len(pathology_mask) > target_seq_len:
            # 下采样：使用平均池化（比最大池化更平滑，保持时间对齐）
            # 使用自适应平均池化（向量化实现，更高效）
            scale = len(pathology_mask) / target_seq_len
            
            # 计算每个目标位置对应的源位置
            target_indices = torch.arange(target_seq_len, device=device, dtype=torch.float32)
            source_start = (target_indices * scale).floor().long().clamp(0, len(pathology_mask) - 1)
            source_end = ((target_indices + 1) * scale).floor().long().clamp(0, len(pathology_mask))
            
            # 向量化计算每个区间的平均值
            mask_float = pathology_mask.float()
            resampled_float = torch.zeros(target_seq_len, device=device, dtype=torch.float32)
            
            for t in range(target_seq_len):
                start = source_start[t].item()
                end = source_end[t].item()
                if start < end:
                    resampled_float[t] = mask_float[start:end].mean()
            
            # 转换为 bool（阈值 0.5）
            resampled = resampled_float > 0.5
        else:
            # 上采样：使用线性插值（比最近邻更平滑）
            # 计算源位置到目标位置的映射
            scale = (len(pathology_mask) - 1) / (target_seq_len - 1) if target_seq_len > 1 else 0
            
            indices_float = torch.arange(target_seq_len, device=device, dtype=torch.float32) * scale
            indices_low = indices_float.floor().long().clamp(0, len(pathology_mask) - 1)
            indices_high = (indices_low + 1).clamp(0, len(pathology_mask) - 1)
            
            # 线性插值权重
            weight = indices_float - indices_low.float()
            
            # 对于 bool 类型，使用加权平均后阈值化
            mask_low = pathology_mask[indices_low].float()
            mask_high = pathology_mask[indices_high].float()
            resampled = ((1 - weight) * mask_low + weight * mask_high) > 0.5
        
        # 确保长度正确（双重检查）
        if len(resampled) != target_seq_len:
            if len(resampled) > target_seq_len:
                resampled = resampled[:target_seq_len]
            else:
                pad_needed = target_seq_len - len(resampled)
                resampled = F.pad(resampled, (0, pad_needed), mode='constant', value=False)
        
        disease_masks.append(resampled.unsqueeze(0))
    
    return torch.cat(disease_masks, dim=0)  # [B, T]

