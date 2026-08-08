"""
局部-全局一致性损失（Multi-Similarity 变体）。

动机：解决「整体与部分相似度低」问题——训练时对同一图取全局视图与局部视图，
显式拉近二者嵌入，同时与 batch 内其他 SKU 的视图分离，使模型具备尺度一致性：
只看到商品局部时，嵌入仍指向该 SKU 的表征区域。

实现：把全局视图嵌入与局部视图嵌入拼接为同一批（2N, D），标签复制为 (2N)，
在拼接批次上计算 Multi-Similarity Loss（正面pair加权 + 难负pair加权）。
参考 Wang et al. "Multi-Similarity Loss with General Pair Weighting" (CVPR 2019)。
"""

import torch
from torch import nn


class LocalGlobalConsistencyLoss(nn.Module):
    def __init__(self, scale_pos: float = 2.0, scale_neg: float = 40.0, thresh: float = 0.5, margin: float = 0.1):
        super().__init__()
        self.scale_pos = scale_pos
        self.scale_neg = scale_neg
        self.thresh = thresh
        self.margin = margin

    def forward(self, global_emb: torch.Tensor, local_emb: torch.Tensor) -> torch.Tensor:
        emb = torch.cat([global_emb, local_emb], dim=0)  # (2N, D)，已归一化
        n = global_emb.size(0)
        label = torch.arange(n, device=emb.device).repeat(2)  # 同图两视图同标签

        sim = emb @ emb.t()  # (2N, 2N)
        label_eq = label.unsqueeze(0) == label.unsqueeze(1)
        pos_mask = label_eq & ~torch.eye(2 * n, dtype=torch.bool, device=emb.device)
        neg_mask = ~label_eq

        # 正样本对：挖掘 sim < thresh + max(neg sim) 的难正对
        pos_loss_sum = torch.tensor(0.0, device=emb.device)
        if pos_mask.any():
            neg_ceiling = sim.detach()[neg_mask].max() if neg_mask.any() else torch.tensor(0.0, device=emb.device)
            pos_pair_mask = pos_mask & (sim < self.thresh + neg_ceiling.clamp_min(0.0))
            if pos_pair_mask.any():
                pos_exp = torch.exp(-self.scale_pos * (sim[pos_pair_mask] - self.margin))
                pos_loss_sum = torch.log1p(pos_exp.sum())

        # 负样本对：挖掘 sim > min(pos sim) - margin 的难负对
        neg_loss_sum = torch.tensor(0.0, device=emb.device)
        if neg_mask.any():
            min_pos = sim.detach()[pos_mask].min() if pos_mask.any() else torch.tensor(1.0, device=emb.device)
            neg_pair_mask = neg_mask & (sim > min_pos - self.margin)
            if neg_pair_mask.any():
                neg_exp = torch.exp(self.scale_neg * (sim[neg_pair_mask] - self.margin))
                neg_loss_sum = torch.log1p(neg_exp.sum())

        return (pos_loss_sum + neg_loss_sum) / (2 * n)
