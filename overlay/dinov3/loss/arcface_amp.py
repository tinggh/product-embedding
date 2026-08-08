"""AMP 兼容的 ArcFaceLoss 包装。

旧 ArcFaceLoss 在 bf16 autocast 下，einsum/linear 产出的 bf16 cos_theta 与
fp32 phi 在 torch.scatter 处发生 dtype 冲突。此处不改上游 arcface_loss.py，
仅在 forward 时关闭 autocast 并强制 fp32 计算（角度 margin 本就需要 fp32 精度）。
"""

import torch

from dinov3.loss.arcface_loss import ArcFaceLoss


class ArcFaceLossAMP(ArcFaceLoss):
    def forward(self, embedding: torch.Tensor, ground_truth: torch.Tensor):
        with torch.autocast(device_type="cuda", enabled=False):
            return super().forward(embedding.float(), ground_truth)
