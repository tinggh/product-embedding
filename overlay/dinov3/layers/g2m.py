# G2M (Generalized Dual Mean) pooling over dense patch tokens.
#
# 并行双 GeM 结构：一路提取空间维度的广义均值描述子，另一路学习通道维度
# 分布权重，对前者做自适应校准，使模型聚焦最具判别度的局部图案
# （品牌 Icon、核心色块）。与 GeM 一样仅含 pow/mean/elementwise 算子，
# TensorRT 导出友好。

import torch
from torch import nn

from dinov3.layers.gem import GeM


class G2M(nn.Module):
    """广义双均值池化 (Generalized Dual Mean Pooling)。

    x: (B, N, C) patch tokens -> (B, C) 校准后的紧凑描述子。

    注：参考实现中的 gate (Linear(1,1)+Sigmoid) 在原 forward 中未接入；
    未使用的参数在 DDP 下永远拿不到梯度会导致 backward 挂死，因此这里把
    gate 接为对通道分布的全局标量校准，保留「自适应校准」的原始意图。
    """

    def __init__(self, p: float = 3.0):
        super().__init__()
        # 负责提取局部几何特征的空间 GeM
        self.spatial_gem = GeM(p=p)
        # 负责学习通道维度主成分分布的 GeM
        self.channel_gem = GeM(p=p)
        # 自适应校准映射层：从通道分布的全局水平学一个 (0,1) 标量门控
        self.gate = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C) patch tokens
        spatial_out = self.spatial_gem(x)    # (B, C) 空间描述子
        channel_dist = self.channel_gem(x)   # (B, C) 通道分布权重
        # 自适应校准：通道分布的全局门控 × 逐通道权重，增强空间输出判别度
        g = self.gate(channel_dist.mean(dim=1, keepdim=True))  # (B, 1)
        return spatial_out * channel_dist * g

    def extra_repr(self) -> str:
        return f"spatial={self.spatial_gem}, channel={self.channel_gem}"
