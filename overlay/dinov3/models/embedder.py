"""
商品嵌入模型封装：DINOv3 backbone + 池化头 + 投影。

将「特征提取方式」从裸 CLS token 升级为 CLS + GeM(patch tokens) 拼接后投影，
训练（finetune.py）、评测（app/probe_eval.py）、导出（model_convert.py）共用同一封装，
保证三处的 embedding 定义一致。

嵌入维度保持 1024，与下游 Milvus 底库 / 阈值体系兼容。
"""

import torch
import torch.nn.functional as F
from torch import nn

from dinov3.layers.g2m import G2M
from dinov3.layers.gem import GeM
from dinov3.layers.salad import SALAD
from dinov3.models.vision_transformer import vit_large

POOLING_CHOICES = ("cls", "gem", "cls+gem", "cls+g2m", "salad")


class ProductEmbedder(nn.Module):
    """DINOv3 backbone + 池化 + 投影头，输出 L2 归一化的商品嵌入。

    pooling:
        "cls"      — 仅 CLS token（等价于旧行为，投影头为 Identity）
        "gem"      — 仅 GeM(patch tokens)
        "cls+gem"  — 拼接 CLS 与 GeM(patch) 后经 Linear 投影回 embed_dim
        "cls+g2m"  — 拼接 CLS 与 G2M(patch) 后经 Linear 投影回 embed_dim
        "salad"    — SALAD 局部聚合(patch, cls) 后经 Linear 投影回 embed_dim
    """

    def __init__(self, backbone: nn.Module, embed_dim: int = 1024, pooling: str = "cls+gem"):
        super().__init__()
        assert pooling in POOLING_CHOICES, f"unknown pooling: {pooling}"
        self.backbone = backbone
        self.pooling = pooling
        self.gem = GeM(p=3.0) if "gem" in pooling else None
        self.g2m = G2M(p=3.0) if "g2m" in pooling else None
        self.salad = SALAD(num_channels=embed_dim) if pooling == "salad" else None
        if pooling in ("cls+gem", "cls+g2m"):
            in_dim = embed_dim * 2
        elif pooling == "salad":
            in_dim = self.salad.descriptor_dim
        else:
            in_dim = embed_dim
        self.proj = nn.Linear(in_dim, embed_dim) if pooling in ("cls+gem", "cls+g2m", "salad") else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x, is_training=True)
        cls_token = out["x_norm_clstoken"]
        patch_tokens = out["x_norm_patchtokens"]
        if self.pooling == "cls":
            feat = cls_token
        elif self.pooling == "gem":
            feat = self.gem(patch_tokens)
        elif self.pooling == "cls+g2m":
            feat = torch.cat([cls_token, self.g2m(patch_tokens)], dim=-1)
        elif self.pooling == "salad":
            feat = self.salad(patch_tokens, cls_token)
        else:
            feat = torch.cat([cls_token, self.gem(patch_tokens)], dim=-1)
        return F.normalize(self.proj(feat), p=2, dim=-1)

    def backbone_param_groups(self, base_lr: float, llrd: float, unfreeze_last: int):
        """按 layer-wise lr decay 生成 backbone 参数组（深层 lr 大、浅层 lr 小）。

        unfreeze_last: 只解冻最后 N 个 block（N >= 24 即全解冻）；冻结层不进参数组。
        llrd: 相邻 block 的学习率衰减系数（0.75~0.85 推荐）。
        """
        blocks = self.backbone.blocks
        n_blocks = len(blocks)
        first_trainable = max(0, n_blocks - unfreeze_last)

        for name, param in self.backbone.named_parameters():
            param.requires_grad = False

        groups = {}
        # 顶层 block lr = base_lr，每往浅一层乘 llrd
        for idx in range(first_trainable, n_blocks):
            lr = base_lr * (llrd ** (n_blocks - 1 - idx))
            for param in blocks[idx].parameters():
                param.requires_grad = True
            groups[lr] = groups.get(lr, []) + list(blocks[idx].parameters())

        # 全解冻时，patch embed / rope 等骨干前置参数用最低 lr 一并训练
        if first_trainable == 0:
            block_params = {id(p) for b in blocks for p in b.parameters()}
            stem_lr = base_lr * (llrd ** n_blocks)
            stem = []
            for name, param in self.backbone.named_parameters():
                if id(param) not in block_params:
                    param.requires_grad = True
                    stem.append(param)
            if stem:
                groups[stem_lr] = groups.get(stem_lr, []) + stem

        return [{"params": params, "lr": lr} for lr, params in sorted(groups.items())]

    def head_parameters(self):
        """池化头 + 投影头参数（新建参数，用较大 lr）。"""
        params = list(self.proj.parameters())
        if self.gem is not None:
            params += list(self.gem.parameters())
        if self.g2m is not None:
            params += list(self.g2m.parameters())
        if self.salad is not None:
            params += list(self.salad.parameters())
        return params


def build_product_embedder(pooling: str = "cls+gem", embed_dim: int = 1024) -> ProductEmbedder:
    """按 finetune 配置构建 embedder（224px, RoPE, 4 storage tokens）。"""
    backbone = vit_large(
        img_size=224,
        patch_size=16,
        pos_embed_rope_base=100,
        qkv_bias=True,
        layerscale_init=1e-5,
        norm_layer="layernorm",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        untie_cls_and_patch_norms=False,
        untie_global_and_local_cls_norm=False,
    )
    # mask_k_bias 的 bias_mask buffer 初始为 NaN，需等 ckpt 加载才有 [1,0,1] 掩码；
    # 若构建后未加载权重（如单元测试/调试），此处兜底初始化，避免前向产出 NaN。
    for m in backbone.modules():
        if hasattr(m, "bias_mask") and m.bias_mask is not None and torch.isnan(m.bias_mask).any():
            o = m.out_features
            m.bias_mask.fill_(1)
            m.bias_mask[o // 3 : 2 * o // 3].fill_(0)
    return ProductEmbedder(backbone, embed_dim=embed_dim, pooling=pooling)
