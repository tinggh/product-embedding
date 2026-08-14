"""
监督对比损失（SupCon，E1c）。

动机：Sub-center ArcFace 通过角度 margin 拉开类间，但对同 Product Line 不同变体
（P1 颜色/口味变体、P5 易混淆品）的细粒度判别仍不足。SupCon 以成对对比形式
直接监督嵌入空间：同类拉近、异类推远，与 ArcFace 的梯度信号互补。

变体对（同 product_line 不同 SKU）由 HardNegativeBatchSampler 混入 batch，
本损失无需显式知道 product_line——batch 内异类即被推远，硬负采样保证变体对
高频出现，从而对 P1/P5 失败 case 施加额外判别压力。

参考：Khosla et al. "Supervised Contrastive Learning" (NeurIPS 2020)。
"""

import torch
import torch.nn.functional as F
from torch import nn


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # embeddings: (N, D)，需已 L2 归一化；labels: (N,)
        device = embeddings.device
        embeddings = F.normalize(embeddings.float(), p=2, dim=-1)
        n = embeddings.size(0)
        if n < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        sim = embeddings @ embeddings.t() / self.temperature  # (N, N)
        # 数值稳定：减去每行最大值
        sim = sim - sim.detach().max(dim=1, keepdim=True).values
        sim = sim - torch.eye(n, device=device) * 1e9  # 屏蔽对角自相似

        labels = labels.view(-1)
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~torch.eye(
            n, dtype=torch.bool, device=device
        )

        # 没有正样本时退化为 0（避免 log(0)）
        has_pos = pos_mask.any(dim=1)
        if not has_pos.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        log_prob = sim - torch.log(torch.exp(sim).sum(dim=1, keepdim=True) + 1e-12)
        # 每个锚点对所有正样本取平均 log-softmax，再对有正样本的锚点取平均
        pos_count = pos_mask.float().sum(dim=1)
        mean_log_prob_pos = (log_prob * pos_mask.float()).sum(dim=1) / pos_count.clamp(min=1)
        loss = -mean_log_prob_pos[has_pos].mean()
        return loss
