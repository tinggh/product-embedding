# product-embedding

基于 DINOv3 的商品细粒度特征提取优化（度量学习微调框架）。
本仓库是 **overlay**：只包含对 [DINOv3](https://github.com/facebookresearch/dinov3) 官方代码的新增文件，
不 fork、不修改上游任何文件，通过 `install.sh` 覆盖安装到一份 dinov3-main 检出中使用。

## 优化内容

| 组件 | 文件 | 说明 |
|---|---|---|
| GeM 池化 | `dinov3/layers/gem.py` | 可学习 p 的广义均值池化，替代单一 CLS，抗遮挡（TRT 友好） |
| 嵌入封装 | `dinov3/models/embedder.py` | `ProductEmbedder` = backbone + 池化 + 投影，输出 1024 维 L2 归一化嵌入；内含 LLRD 参数组；支持 `cls+gem+salad` 双头池化（E1b）与 `forward(return_patch=)` 取 patch token |
| Sub-center ArcFace | `dinov3/loss/subcenter_arcface_loss.py` | 每类 K=3 子中心 + Center Loss，解决多视角/多面问题 |
| 局部-全局一致性 | `dinov3/loss/local_global_consistency.py` | Multi-Similarity 变体，解决部分-整体相似度低 |
| 监督对比损失 | `dinov3/loss/supcon_loss.py` | SupCon（E1c），同类拉近/异类推远，与 ArcFace 梯度互补，增强变体判别 |
| Patch 级对比损失 | `dinov3/loss/patch_nce_loss.py` | DenseCL 风格 PatchNCE（E2c），两视图同位置 patch 为正、跨图 patch 为负，提升多视角/局部一致性 |
| 保色增强 | `dinov3/data/color_preserving_augs.py` | 小幅 hue 扰动（±0.02）+ 亮度/阴影/暗角/遮挡增强；含 `DualViewTransform` 与 `TripleViewTransform`（E2b，全局+局部+零件三视图） |
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

容器/服务器一键流水线（`scripts/` 下，全部参数走环境变量，可直接迁移到其他 GPU 容器）：

```bash
# 1. 数据准备：metadata → hierarchy → 测试集选择(闭集+开集) → npy 切分(开集剔除)
PY=/path/to/python REPO=/path/dinov3-main PE=/path/product-embedding \
DATASET_ROOT=/path/sku100wdata FINGERPRINT=/path/fingerprint.csv \
WORK_DIR=/path/work bash scripts/00_prepare_data.sh

# 2. 训练（E5 全量配置；EXTRA_ARGS 可覆盖，如 EXTRA_ARGS="--max_epoch 30"）
PY=/path/to/python REPO=/path/dinov3-main \
DATASET_ROOT=/path/sku100wdata NPY_DIR=/path/work/npy \
CKPT=/path/dinov3_vitl16_pretrain_lvd1689m.pth OUTPUT_DIR=/path/runs/exp \
NGPU=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
HIERARCHY=/path/work/hierarchy.json bash scripts/01_train.sh

# 3. 六维探针评测
PY=/path/to/python REPO=/path/dinov3-main \
PROBE_ROOT=/path/work/testset/probe CKPT=/path/runs/exp/best.pth \
OUTPUT=/path/work/report bash scripts/02_probe_eval.sh
```

手动分步调用（旧方式）：

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

## 评估结果

六维探针评测（`work_sku100w/testset/probe`，50164 张图）：P1/P5 为 diff 型（相似度<0.7 通过率，越高越好），P2/P3/P4 为 same 型（相似度≥0.75 通过率，越高越好），P6 为开集 top1 命中率（≥0.9 通过）。overall PASS 需 6 维全通过（P1~P5 要求 100%）。

| experiment | P1 色变 | P2 多视角 | P3 局整 | P4 遮挡 | P5 易混 | P6 开集 | overall |
|---|---|---|---|---|---|---|---|
| baseline (v1.1.2) | 0.189 | 0.735 | 0.706 | 0.995 | 0.567 | 0.973 | FAIL |
| full_e40 (40ep) | 0.919 | 0.599 | 0.746 | 0.980 | 0.971 | 0.992 | FAIL |
| full (E5, 80ep) | 0.911 | 0.684 | 0.803 | 0.986 | 0.962 | 0.993 | FAIL |
| shallow (40ep) | 0.959 | 0.298 | 0.273 | 0.982 | 0.952 | 0.982 | FAIL |
| legacy_aug (40ep) | 0.895 | 0.610 | 0.352 | 0.995 | 0.848 | 0.988 | FAIL |
| arcface (40ep) | 0.922 | 0.341 | 0.519 | 0.977 | 0.986 | 0.990 | FAIL |
| cls_pool (40ep) | 0.816 | 0.417 | 0.522 | 0.987 | 0.948 | 0.987 | FAIL |
| no_hardneg (40ep) | 0.865 | 0.620 | 0.739 | 0.985 | 0.943 | 0.992 | FAIL |
| g2m (40ep) | 0.919 | 0.435 | 0.605 | 0.976 | 0.976 | 0.990 | FAIL |
| salad (40ep) | 0.662 | 0.901 | 0.930 | 0.996 | 0.791 | 0.989 | FAIL |
| **e1b 双头 (40ep)** | 0.835 | **0.838** | **0.908** | 0.998 | 0.929 | 0.991 | FAIL |
| e1b_patch 双头+PatchNCE | 0.692 | 0.911 | 0.972 | 1.000 | 0.829 | 0.989 | FAIL |
| e1c SupCon | 0.870 | 0.732 | 0.853 | 0.990 | 0.948 | 0.991 | FAIL |
| e1a salad+大margin | 0.065 | 0.995 | 1.000 | 1.000 | 0.100 | 0.947 | FAIL |

### 关键发现

- **e1b 双头（cls+gem+salad）是冠军**：P2/P3 从 0.60/0.75 跃升到 0.84/0.91（+0.24/+0.16），P1/P5 仅降 0.08/0.04——双头架构成功打破一致性与判别力的权衡。
- **e1b 天虹检索 0.85 阈值精度翻倍**：pegSection 0.33→0.70，StackBase 0.40→0.64；P2/P3 一致性提升直接转化为高阈值召回。
- **baseline → full 管线价值**：P1 色变 0.19→0.92（+0.73）、P5 易混 0.57→0.97（+0.40）、P6 已过 0.9 门禁。
- **训练长度效应**（full 40→80ep）：P2 +0.085、P3 +0.057（一致性随训练提升），P1 −0.008、P5 −0.010（判别略回退）。
- **核心权衡**：P2/P3（一致性）与 P1/P5（判别）对池化头要求相反。`salad` 强 P2/P3 但弱 P1/P5；`cls+gem` 折中；**e1b 双头打破此权衡**。
- **e1a 崩盘**：salad+大margin 判别坍塌（P1=0.065/P5=0.10），死路。**e4c 印证**：hard_ratio 0.7 让 P5↑但 P2/P3↓。

## 优化方向

full_e40 短板排序：**P2(0.60) > P3(0.75) > P1(0.92) > P5(0.97) > P4/P6（已饱和）**。e1b 双头已把 P2/P3 提到 0.84/0.91，下一步围绕 e1b 恢复判别力。

| 优先 | 实验 | 命令（e1b 双头基线叠加） | 攻克目标 |
|---|---|---|---|
| P0 | e1b+SupCon | `--pooling cls+gem+salad --supcon_lambda 0.1` | 拉回 P1/P5 保 P2/P3 |
| P0 | e1b 80ep | e1b 配置训 80ep | 一致性随训练再提升 |
| P1 | e1b+低PatchNCE | `--pooling cls+gem+salad --patch_consistency_lambda 0.03` | 微调 P2/P3 不伤判别 |
| P1 | e1b+hard_ratio | `--pooling cls+gem+salad --hard_ratio 0.7` | 硬负拉回 P5 |
| P2 | e1c SupCon（轻量备选） | `--supcon_lambda 0.1`（不改架构） | P2/P3=0.73/0.85 保判别 |

e1b 集成 7 项策略：DINOv3 ViT-L 深解冻+LLRD、**cls+gem+salad 双头池化**、Sub-center ArcFace+Center Loss、保色增强、双视图一致性、层级硬负采样、AdamW+bf16。第二轮 9 组实验已完成，第三轮 e1b 组合实验进行中。详见 `runs/rec/ablation/EVAL_REPORT.md`。
