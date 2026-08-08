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

from dinov3.layers.gem import GeM
from dinov3.models.vision_transformer import vit_large


class ProductEmbedder(nn.Module):
    """DINOv3 backbone + 池化 + 投影头，输出 L2 归一化的商品嵌入。

    pooling:
        "cls"      — 仅 CLS token（等价于旧行为，投影头为 Identity）
        "gem"      — 仅 GeM(patch tokens)
        "cls+gem"  — 拼接 CLS 与 GeM(patch) 后经 Linear 投影回 embed_dim
    """

    def __init__(self, backbone: nn.Module, embed_dim: int = 1024, pooling: str = "cls+gem"):
        super().__init__()
        assert pooling in ("cls", "gem", "cls+gem"), f"unknown pooling: {pooling}"
        self.backbone = backbone
        self.pooling = pooling
        self.gem = GeM(p=3.0) if "gem" in pooling else None
        in_dim = embed_dim * 2 if pooling == "cls+gem" else embed_dim
        self.proj = nn.Linear(in_dim, embed_dim) if pooling == "cls+gem" else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x, is_training=True)
        cls_token = out["x_norm_clstoken"]
        if self.pooling == "cls":
            feat = cls_token
        elif self.pooling == "gem":
            feat = self.gem(out["x_norm_patchtokens"])
        else:
            feat = torch.cat([cls_token, self.gem(out["x_norm_patchtokens"])], dim=-1)
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
        return params


def build_product_embedder(pooling: str = "cls+gem", embed_dim: int = 1024) -> ProductEmbedder:
    """按 finetune.py 现有 vit_large 配置构建 embedder（224px, RoPE, 4 storage tokens）。"""
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
    return ProductEmbedder(backbone, embed_dim=embed_dim, pooling=pooling)
