"""
GPU版本的AdaptivePathologyMasker
所有操作在GPU上执行；支持端到端可导的软掩码（方案A）。
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

# 导入GPU病理检测器（同目录）
from gpu_pathology_detector import GPUPathologyFeatureDetector


class GPUAdaptivePathologyMasker(nn.Module):
    """
    GPU加速的自适应病理掩码生成器
    所有操作在GPU上执行；继承 nn.Module 以支持端到端可学习权重的软掩码。
    """
    
    def __init__(
        self,
        detector: GPUPathologyFeatureDetector,
        mask_prob: float = 0.65,
        min_mask_span: int = 5,
        max_mask_span: int = 20,
        pathology_ratio: float = 0.8,
        random_ratio: float = 0.2,
        soft_mask_temperature: float = 0.1,
    ):
        super().__init__()
        self.detector = detector
        self.mask_prob = mask_prob
        self.min_mask_span = min_mask_span
        self.max_mask_span = max_mask_span
        self.pathology_ratio = pathology_ratio
        self.random_ratio = random_ratio
        self.soft_mask_temperature = soft_mask_temperature
        # 可学习阈值，用于软掩码 sigmoid((score - threshold) / temp)，端到端训练时更新
        self.soft_threshold = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
    
    def generate_waveform_mask(
        self,
        waveform: np.ndarray,
        mute_value: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        在波形空间生成智能掩码和随机掩码
        
        Returns:
            intelligent: 智能掩码后的波形
            random_masked: 随机掩码后的波形
        """
        # 转换为tensor并移到GPU
        if isinstance(waveform, np.ndarray):
            waveform_tensor = torch.FloatTensor(waveform).to(self.detector.device)
        else:
            waveform_tensor = waveform.to(self.detector.device)
        
        # 确保是单声道
        if waveform_tensor.dim() > 1:
            waveform_tensor = waveform_tensor.mean(dim=0)
        
        # 在GPU上检测病理片段（完全在GPU上）
        pathology_mask, features = self.detector.detect_pathology_segments(
            waveform_tensor,  # 直接传入GPU tensor
            return_features=True
        )
        
        # 生成智能掩码
        intelligent = waveform_tensor.clone()
        mask_waveform = torch.zeros(len(waveform_tensor), dtype=torch.bool, device=self.detector.device)
        
        # 将病理掩码转换为采样点索引
        pathology_indices = np.where(pathology_mask)[0]
        pathology_sample_indices = []
        for idx in pathology_indices:
            start_sample = idx * self.detector.hop_length
            end_sample = min(len(waveform_tensor), start_sample + self.detector.frame_length)
            pathology_sample_indices.extend(range(start_sample, end_sample))
        
        pathology_sample_indices = torch.tensor(list(set(pathology_sample_indices)), device=self.detector.device)
        
        # 计算目标掩码采样点数量
        target_mask_samples = int(len(waveform_tensor) * self.mask_prob)
        
        # 1. 病理片段掩码（主要部分，80%）
        pathology_mask_samples = int(target_mask_samples * self.pathology_ratio)
        if len(pathology_sample_indices) > 0 and pathology_mask_samples > 0:
            # 计算需要多少个span掩码段
            avg_span = (self.min_mask_span + self.max_mask_span) // 2 * self.detector.hop_length
            num_pathology_spans = max(1, pathology_mask_samples // avg_span)
            
            # 从病理片段中随机选择位置
            num_selected = min(num_pathology_spans, len(pathology_sample_indices))
            selected_indices = pathology_sample_indices[
                torch.randperm(len(pathology_sample_indices), device=self.detector.device)[:num_selected]
            ]
            
            # 对选中的位置生成span掩码
            for idx in selected_indices:
                span_samples = torch.randint(
                    self.min_mask_span * self.detector.hop_length,
                    (self.max_mask_span + 1) * self.detector.hop_length,
                    (1,),
                    device=self.detector.device
                ).item()
                start = max(0, idx.item() - span_samples // 2)
                end = min(len(waveform_tensor), start + span_samples)
                mask_waveform[start:end] = True
        
        # 2. 随机掩码（补充，20%）
        random_mask_samples = target_mask_samples - mask_waveform.sum().item()
        if random_mask_samples > 0:
            # 计算需要多少个随机掩码段
            avg_span = (self.min_mask_span + self.max_mask_span) // 2 * self.detector.hop_length
            num_random_spans = max(1, random_mask_samples // avg_span)
            
            # 随机选择位置生成span掩码
            available_indices = torch.where(~mask_waveform)[0]
            if len(available_indices) > 0:
                num_random_spans = min(num_random_spans, len(available_indices))
                random_indices = available_indices[
                    torch.randperm(len(available_indices), device=self.detector.device)[:num_random_spans]
                ]
                for idx in random_indices:
                    span_samples = torch.randint(
                        self.min_mask_span * self.detector.hop_length,
                        (self.max_mask_span + 1) * self.detector.hop_length,
                        (1,),
                        device=self.detector.device
                    ).item()
                    start = max(0, idx.item() - span_samples // 2)
                    end = min(len(waveform_tensor), start + span_samples)
                    # 只掩码未被掩码的区域
                    mask_waveform[start:end] = True
        
        # 应用掩码
        intelligent[mask_waveform] = mute_value
        total_mask_samples = mask_waveform.sum().item()
        
        # 随机掩码：静音相同数量的采样点（完全随机）
        random_masked = waveform_tensor.clone()
        if total_mask_samples > 0:
            random_mask = torch.zeros(len(waveform_tensor), dtype=torch.bool, device=self.detector.device)
            random_indices = torch.randperm(len(waveform_tensor), device=self.detector.device)[:total_mask_samples]
            random_mask[random_indices] = True
            random_masked[random_mask] = mute_value
        
        return intelligent.cpu().numpy(), random_masked.cpu().numpy()

    def generate_soft_masked_waveform(self, waveform_tensor: torch.Tensor) -> torch.Tensor:
        """
        端到端可导的软掩码：用 pathology_score 生成连续 mask，梯度可回传到 detector.weight_logits。
        waveform_tensor: (T,) 或 (1, T)，同设备为 detector.device。
        Returns: (T,) 软掩码后的波形，与输入同 shape/device。
        """
        if waveform_tensor.dim() > 1:
            waveform_tensor = waveform_tensor.mean(dim=0)
        waveform_tensor = waveform_tensor.to(self.detector.device)
        pathology_score = self.detector.get_pathology_scores(waveform_tensor)  # (n_frames,)
        hop_length = self.detector.hop_length
        n_samples = len(waveform_tensor)
        # 帧级 score 按 hop_length 重复到采样级，再截断/填充到 n_samples
        score_expanded = pathology_score.repeat_interleave(hop_length)
        if score_expanded.size(0) < n_samples:
            score_expanded = torch.nn.functional.pad(
                score_expanded, (0, n_samples - score_expanded.size(0)), value=0.0
            )
        else:
            score_expanded = score_expanded[:n_samples]
        # 软掩码：sigmoid((score - threshold) / temp)，对 weight_logits 可导
        soft_mask = torch.sigmoid(
            (score_expanded - self.soft_threshold) / (self.soft_mask_temperature + 1e-8)
        )
        masked = waveform_tensor * (1.0 - soft_mask)
        return masked

