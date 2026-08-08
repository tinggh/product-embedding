"""
Sub-center ArcFace + Center Loss。

动机：同一 SKU 的正面/侧面/背面在视觉上是多模态分布，单中心 ArcFace 强行把
所有视角拉向同一个类别中心，导致特征坍塌、跨视角相似度低。Sub-center ArcFace
为每个类别维护 K 个子中心，样本只需逼近最近的一个子中心；Center Loss 作为
正则项收缩各子中心内部分布，防止类内过于松散。

forward 契约与 ArcFaceLoss 一致：返回 (loss, cos_theta)，cos_theta 为
max-over-K 的类别相似度（N, num_classes），eval_model 的 accuracy 逻辑可直接复用。
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class SubCenterArcFaceLoss(nn.Module):
    def __init__(
        self,
        embed_size: int,
        num_classes: int,
        num_subcenters: int = 3,
        scale: float = 64.0,
        margin: float = 0.2,
        easy_margin: bool = False,
    ):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.num_subcenters = num_subcenters
        self.easy_margin = easy_margin
        self.ce = nn.CrossEntropyLoss()
        # (num_classes, K, embed_size)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, num_subcenters, embed_size))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def subcenter_similarity(self, embedding: torch.Tensor) -> torch.Tensor:
        """返回 max-over-K 的类别余弦相似度，(N, num_classes)。"""
        x = F.normalize(embedding)
        w = F.normalize(self.weight, dim=-1)  # (C, K, D)
        # (N, C, K) -> max over K -> (N, C)
        cos_theta_all = torch.einsum("nd,ckd->nck", x, w)
        return cos_theta_all.max(dim=2).values.clamp(-1 + 1e-7, 1 - 1e-7)

    def forward(self, embedding: torch.Tensor, ground_truth: torch.Tensor):
        cos_theta = self.subcenter_similarity(embedding)
        pos = torch.gather(cos_theta, 1, ground_truth.view(-1, 1))
        sin_theta = torch.sqrt((1.0 - torch.pow(pos, 2)).clamp(-1 + 1e-7, 1 - 1e-7))
        phi = pos * self.cos_m - sin_theta * self.sin_m
        if self.easy_margin:
            phi = torch.where(pos > 0, phi, pos)
        else:
            phi = torch.where(pos > self.th, phi, pos - self.mm)
        output = torch.scatter(cos_theta, 1, ground_truth.view(-1, 1).long(), phi)
        output *= self.scale
        loss = self.ce(output, ground_truth)
        return loss, cos_theta


class CenterLoss(nn.Module):
    """Center Loss：惩罚样本与其类别最近子中心的欧氏距离，收缩类内分布。

    中心与 SubCenterArcFaceLoss 共享同一份 weight（按最近子中心取值），
    保证两个损失优化的是同一个超球面中心结构。
    """

    def __init__(self, subcenter_loss: SubCenterArcFaceLoss):
        super().__init__()
        self.subcenter_loss = subcenter_loss

    def forward(self, embedding: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        w = self.subcenter_loss.weight  # (C, K, D)
        centers = w[ground_truth]  # (N, K, D)
        x = F.normalize(embedding).unsqueeze(1)  # (N, 1, D)
        w_norm = F.normalize(centers, dim=-1)
        # 选余弦相似度最高的子中心作为活跃中心
        sim = (x * w_norm).sum(-1)  # (N, K)
        active = centers[torch.arange(embedding.size(0), device=embedding.device), sim.argmax(dim=1)]
        return ((embedding - active) ** 2).sum(dim=1).mean()


class SubCenterArcFaceWithCenterLoss(nn.Module):
    """L_total = L_sub_arc + lambda * L_center，forward 返回 (loss, cos_theta)。"""

    def __init__(self, embed_size: int, num_classes: int, num_subcenters: int = 3,
                 scale: float = 64.0, margin: float = 0.2, center_lambda: float = 0.5):
        super().__init__()
        self.sub_arc = SubCenterArcFaceLoss(
            embed_size, num_classes, num_subcenters=num_subcenters, scale=scale, margin=margin
        )
        self.center = CenterLoss(self.sub_arc)
        self.center_lambda = center_lambda

    def forward(self, embedding: torch.Tensor, ground_truth: torch.Tensor):
        arc_loss, cos_theta = self.sub_arc(embedding, ground_truth)
        loss = arc_loss + self.center_lambda * self.center(embedding, ground_truth)
        return loss, cos_theta
