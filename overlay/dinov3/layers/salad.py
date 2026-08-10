# SALAD (Sinkhorn Algorithm for Locally Aggregated Descriptors) aggregation.
#
# 适配自 https://github.com/serizba/salad (CVPR 2024, "Optimal Transport
# Aggregation for Visual Place Recognition", Izquierdo & Civera)，原实现见
# models/aggregators/salad.py（Sinkhorn 部分改编自 OpenGlue, MIT license）。
#
# 与官方版本的区别：官方输入为 conv 特征图 (B, C, H, W)（1x1 Conv2d），
# 这里直接作用于 ViT patch tokens (B, N, C)，等价地把 1x1 Conv2d 换成 Linear。
#
# 核心思想：用 Sinkhorn 最优传输把局部特征软分配到 M 个可学习聚簇
# （含 dustbin 吸收无判别力的 token），按簇聚合后与全局 CLS 分支拼接，
# 得到对遮挡/裁剪更鲁棒的局部聚合描述子。

import math

import torch
import torch.nn.functional as F
from torch import nn


# Code adapted from OpenGlue, MIT license
# https://github.com/ucuapps/OpenGlue/blob/main/models/superglue/optimal_transport.py
def log_otp_solver(log_a, log_b, M, num_iters: int = 20, reg: float = 1.0) -> torch.Tensor:
    r"""Sinkhorn matrix scaling algorithm for Differentiable Optimal Transport problem."""
    M = M / reg  # regularization

    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)

    for _ in range(num_iters):
        u = log_a - torch.logsumexp(M + v.unsqueeze(1), dim=2).squeeze()
        v = log_b - torch.logsumexp(M + u.unsqueeze(2), dim=1).squeeze()

    return M + u.unsqueeze(2) + v.unsqueeze(1)


# Code adapted from OpenGlue, MIT license
# https://github.com/ucuapps/OpenGlue/blob/main/models/superglue/superglue.py
def get_matching_probs(S, dustbin_score=1.0, num_iters=3, reg=1.0):
    """sinkhorn"""
    batch_size, m, n = S.size()
    # augment scores matrix
    S_aug = torch.empty(batch_size, m + 1, n, dtype=S.dtype, device=S.device)
    S_aug[:, :m, :n] = S
    S_aug[:, m, :] = dustbin_score

    # prepare normalized source and target log-weights
    norm = -torch.tensor(math.log(n + m), device=S.device)
    log_a, log_b = norm.expand(m + 1).contiguous(), norm.expand(n).contiguous()
    log_a[-1] = log_a[-1] + math.log(n - m)
    log_a, log_b = log_a.expand(batch_size, -1), log_b.expand(batch_size, -1)
    log_P = log_otp_solver(
        log_a,
        log_b,
        S_aug,
        num_iters=num_iters,
        reg=reg,
    )
    return log_P - norm


class SALAD(nn.Module):
    """Sinkhorn Algorithm for Locally Aggregated Descriptors.

    forward(patch_tokens, cls_token):
        patch_tokens: (B, N, C)
        cls_token:    (B, C)
        return:       (B, num_clusters*cluster_dim + token_dim)，已 L2 归一化
    """

    def __init__(
        self,
        num_channels: int = 1024,
        num_clusters: int = 64,
        cluster_dim: int = 128,
        token_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.num_channels = num_channels
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim

        dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # MLP for global scene token g
        self.token_features = nn.Sequential(
            nn.Linear(self.num_channels, 512),
            nn.ReLU(),
            nn.Linear(512, self.token_dim),
        )
        # MLP for local features f_i（官方为 1x1 Conv2d，token 输入下等价于 Linear）
        self.cluster_features = nn.Sequential(
            nn.Linear(self.num_channels, 512),
            dropout_layer,
            nn.ReLU(),
            nn.Linear(512, self.cluster_dim),
        )
        # MLP for score matrix S
        self.score = nn.Sequential(
            nn.Linear(self.num_channels, 512),
            dropout_layer,
            nn.ReLU(),
            nn.Linear(512, self.num_clusters),
        )
        # Dustbin parameter z
        self.dust_bin = nn.Parameter(torch.tensor(1.0))

    @property
    def descriptor_dim(self) -> int:
        return self.num_clusters * self.cluster_dim + self.token_dim

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C) patch tokens, t: (B, C) global cls token
        in_dtype = x.dtype

        f = self.cluster_features(x)           # (B, N, l)
        p = self.score(x).transpose(1, 2)      # (B, m, N)
        t = self.token_features(t)             # (B, g)

        # Sinkhorn 含 logsumexp 迭代，bf16 精度不足，强制 fp32 计算
        with torch.autocast(device_type="cuda", enabled=False):
            p = get_matching_probs(p.float(), self.dust_bin.float(), 3)
            p = torch.exp(p)
            # Normalize to maintain mass（去掉 dustbin 行）
            p = p[:, :-1, :]                   # (B, m, N)

            # 按簇加权聚合局部特征：(B, l, m)
            agg = torch.einsum("bnl,bmn->blm", f.float(), p)

            out = torch.cat(
                [
                    F.normalize(t.float(), p=2, dim=-1),
                    F.normalize(agg, p=2, dim=1).flatten(1),
                ],
                dim=-1,
            )
            out = F.normalize(out, p=2, dim=-1)

        return out.to(in_dtype)
