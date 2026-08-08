# product-embedding

基于 DINOv3 的商品细粒度特征提取优化（度量学习微调框架）。
本仓库是 **overlay**：只包含对 [DINOv3](https://github.com/facebookresearch/dinov3) 官方代码的新增文件，
不 fork、不修改上游任何文件，通过 `install.sh` 覆盖安装到一份 dinov3-main 检出中使用。

## 优化内容

| 组件 | 文件 | 说明 |
|---|---|---|
| GeM 池化 | `dinov3/layers/gem.py` | 可学习 p 的广义均值池化，替代单一 CLS，抗遮挡（TRT 友好） |
| 嵌入封装 | `dinov3/models/embedder.py` | `ProductEmbedder` = backbone + GeM + 投影，输出 1024 维 L2 归一化嵌入；内含 LLRD 参数组 |
| Sub-center ArcFace | `dinov3/loss/subcenter_arcface_loss.py` | 每类 K=3 子中心 + Center Loss，解决多视角/多面问题 |
| 局部-全局一致性 | `dinov3/loss/local_global_consistency.py` | Multi-Similarity 变体，解决部分-整体相似度低 |
| 保色增强 | `dinov3/data/color_preserving_augs.py` | 小幅 hue 扰动（±0.02）+ 亮度/阴影/暗角/遮挡增强，颜色变体敏感且光照稳健 |
| 硬负样本采样 | `dinov3/data/hard_negative_sampler.py` | 同 Product Line 变体混入 batch，DDP 分片 |
| 训练入口 | `dinov3/train/finetune_v2.py` | 深解冻 + LLRD + AdamW + bf16 AMP + 梯度累积 |
| 层级标签 | `app/build_hierarchy.py` | 生成 hierarchy.json + split 泄漏校验 |
| 六维探针评测 | `app/probe_eval.py` | P1 颜色变体 / P2 多视角 / P3 部分-整体 / P4 遮挡 / P5 相似品 / P6 开集 |
| 数据集准备 | `scripts/prepare_sku_dataset.py` | sku100wdata 风格数据集（barcode 目录无 split）→ RetailProduct npy |

## 安装

```bash
git clone https://github.com/tinggh/product-embedding.git
./product-embedding/install.sh /path/to/dinov3-main
```

## 使用

详见 `overlay/app/RUNBOOK_finetune_v2.md`（消融实验矩阵、4090-24G 显存档位、评测门禁流程）。

快速开始（GPU 服务器）：

```bash
cd /path/to/dinov3-main
# 1. 数据集准备（生成 split npy，原地不动图片）
python /path/to/product-embedding/scripts/prepare_sku_dataset.py \
    --dataset_root /path/to/sku100wdata0324
# 2. 层级标签
python -m app.build_hierarchy --dataset_root /path/to/sku100wdata0324 --output hierarchy.json
# 3. 训练（单卡示例；多卡调 --nproc_per_node）
torchrun --nproc_per_node=1 dinov3/train/finetune_v2.py --train \
    --dataset_root /path/to/sku100wdata0324 \
    --ckpt_path /path/to/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
    --output_dir /path/to/runs/exp_e5 \
    --loss subcenter --pooling cls+gem --aug color_preserving \
    --consistency_lambda 0.5 --hierarchy_json hierarchy.json --hard_ratio 0.5
# 4. 六维探针评测
python -m app.probe_eval --probe_root /path/to/probe --ckpt /path/to/runs/exp_e5/best.pth --output report
```

## 设计背景

- 放弃 SSL 领域续训：DINO+iBOT 的不变性目标与细粒度检索目标错位，窄域续训导致通用表征漂移、泛化变差；
  保留的"预训练"是 DINOv3 官方 LVD-1689M 权重本身。
- 只微调最后 1 个 block 无法改变编码在早中层的颜色/纹理表征 → 深解冻 + LLRD。
- 颜色敏感与光照鲁棒的矛盾由「小幅 hue 扰动 + 监督信号」解决，而非 SSL 增强调参。
