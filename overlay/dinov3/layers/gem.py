# GeM (Generalized Mean) pooling over dense patch tokens.
#
# 用可学习 p 的广义均值池化替代对单一 CLS token 的依赖，
# 放大高激活的判别性局部区域（品牌 Icon、核心色块）的贡献，
# 对局部遮挡/边缘裁剪更稳健。仅含 pow/mean 算子，TensorRT 导出友好。

import torch
import torch.nn.functional as F
from torch import nn


class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C) patch tokens -> (B, C)
        return x.clamp(min=self.eps).pow(self.p).mean(dim=1).pow(1.0 / self.p)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.item():.4f}, eps={self.eps})"
