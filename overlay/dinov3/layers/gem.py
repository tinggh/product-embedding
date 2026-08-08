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
        # bf16 下 x.pow(p) 的均值可能下溢为 0，外层 pow(1/p) 反向传播会产生 inf/NaN，
        # 因此池化强制 fp32 计算（关闭 autocast），并在 mean 后再次 clamp 保底。
        in_dtype = x.dtype
        with torch.autocast(device_type="cuda", enabled=False):
            p = self.p.float()
            out = (
                x.float()
                .clamp(min=self.eps)
                .pow(p)
                .mean(dim=1)
                .clamp(min=self.eps)
                .pow(1.0 / p)
            )
        return out.to(in_dtype)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.item():.4f}, eps={self.eps})"
