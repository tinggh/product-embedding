"""
Patch 级 InfoNCE 损失（DenseCL 风格，E2c）。

动机：LocalGlobalConsistencyLoss 只在图像级嵌入上做一致性，patch 级的视角/尺度
不变性未被显式监督。P2（多视角）/P3（局部-整体）要求同一商品不同视角的对应
局部区域相似，故对 patch token 做密集对比：每个 patch 的正样本是另一视图同位置
patch，负样本是 batch 内其他图像的 patch。促使 patch 编码器对视角变化不变。

参考：Wang et al. "Dense Contrastive Learning for Self-Supervised Visual Pre-Training" (CVPR 2021)。
"""

import torch
import torch.nn.functional as F
from torch import nn


class PatchNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.1, num_samples: int = 32):
        super().__init__()
        self.temperature = temperature
        self.num_samples = num_samples

    def forward(self, patch_a: torch.Tensor, patch_b: torch.Tensor) -> torch.Tensor:
        # patch_a, patch_b: (B, N, C)，需逐 patch L2 归一化
        b, n, c = patch_a.shape
        device = patch_a.device
        m = min(self.num_samples, n)
        # 随机采样 M 个 patch 位置，降低 (B*M)² 矩阵开销
        idx = torch.randperm(n, device=device)[:m]
        a = patch_a[:, idx].reshape(b * m, c).float()       # (B*M, C)
        bb = patch_b[:, idx].reshape(b * m, c).float()      # (B*M, C)
        a = F.normalize(a, p=2, dim=-1)
        bb = F.normalize(bb, p=2, dim=-1)

        # 每个锚点 a[i] 的正样本 = bb[i]（同图同位置另一视图）
        # 负样本 = batch 内其他图的 patch（bb[j] 中 j 属于不同图）
        # 构造图归属：i 属于图 i // M
        img_id = torch.arange(b * m, device=device) // m    # (B*M,)
        sim_pos = (a * bb).sum(-1) / self.temperature        # (B*M,)
        # 全对比矩阵 a @ bb.t()，屏蔽同图（含正样本）作为负样本候选
        sim_all = a @ bb.t() / self.temperature              # (B*M, B*M)
        same_img = img_id.unsqueeze(0) == img_id.unsqueeze(1)  # (B*M, B*M)
        # 负样本分母：对每个锚点 i，取所有 j 中不同图的 sim
        neg = sim_all.masked_fill(same_img, 0.0)
        denom = sim_pos.exp() + neg.exp().sum(1) + 1e-12
        loss = -torch.log(sim_pos.exp() / denom + 1e-12).mean()
        return loss
