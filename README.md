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
- **e1b 天虹检索 0.85 阈值总样本命中率显著提升（相对 full_e40）**：pegSection 0.33→0.70，StackBase 0.40→0.64。这里的数值是包含拒识样本的 accuracy，不是 accepted precision；新模型必须分别校准阈值。
- **baseline → full 管线价值**：P1 色变 0.19→0.92（+0.73）、P5 易混 0.57→0.97（+0.40）、P6 已过 0.9 门禁。
- **训练长度效应**（full 40→80ep）：P2 +0.085、P3 +0.057（一致性随训练提升），P1 −0.008、P5 −0.010（判别略回退）。
- **核心权衡**：P2/P3（一致性）与 P1/P5（判别）对池化头要求相反。`salad` 强 P2/P3 但弱 P1/P5；`cls+gem` 折中；**e1b 双头打破此权衡**。
- **e1a 崩盘**：salad+大margin 判别坍塌（P1=0.065/P5=0.10），死路。**e4c 印证**：hard_ratio 0.7 让 P5↑但 P2/P3↓。

## 第三轮组合实验

围绕 e1b 恢复判别力的三项实验均已完成，但没有超过纯 e1b 的综合表现。

| 实验 | P1 | P2 | P3 | P5 | 结论 |
|---|---:|---:|---:|---:|---|
| **e1b** | 0.835 | **0.838** | **0.908** | 0.929 | 当前综合最优 |
| e1b+SupCon 0.1 | **0.881** | 0.560 | 0.760 | **0.952** | 判别恢复，但一致性大幅回退 |
| e1b+PatchNCE 0.03 | 0.862 | 0.628 | 0.857 | **0.952** | 强于 0.1，但仍不及纯 e1b |
| e1b+hard_ratio 0.7 | 0.876 | 0.601 | 0.819 | 0.943 | 增加硬负比例没有拉回 P5 |

因此停止继续放大 SupCon、PatchNCE 或 hard ratio；当前验证方向改为 epoch 选择、后期关闭一致性损失，以及按业务场景进行双阶段融合和阈值校准。

e1b 集成 7 项策略：DINOv3 ViT-L 深解冻+LLRD、**cls+gem+salad 双头池化**、Sub-center ArcFace+Center Loss、保色增强、双视图一致性、层级硬负采样、AdamW+bf16。第二、三轮实验、双阶段检索及训练后期关闭一致性损失均已完成。详见 `runs/rec/ablation/EVAL_REPORT.md`。

## 最新补充实验（2026-08-20）

### e1b 不同 epoch

门禁改为 P1~P5 `min_pass_rate=0.95`（不再要求不现实的 100%）；下表仍展示原始通过率。epoch 0 的 P1/P5 为 0 是未训练投影头的预期结果。

| epoch | P1 色变 | P2 多视角 | P3 局整 | P4 遮挡 | P5 易混 | P6 开集 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.741 |
| 10 | 0.805 | 0.795 | 0.887 | 0.996 | 0.905 | 0.991 |
| 20 | 0.830 | 0.835 | 0.899 | 0.996 | 0.933 | 0.991 |
| **30** | **0.838** | **0.846** | 0.905 | 0.996 | **0.933** | **0.992** |
| 39 | 0.835 | 0.838 | **0.908** | **0.998** | 0.929 | 0.991 |

**结论**：epoch 30 是当前 Pareto 最优点；30→39 仅小幅提高 P3/P4，同时 P1/P2/P5 均回退。产物位于 `runs/rec/ablation/e1b_epochs/`。

### 两阶段检索（e1b 召回 + full_e40 重排）

- **StackBase 有收益**：`TopK=50, alpha(e1b)=0.5` 的留出集无阈值 Top-1 为 **0.9831**，高于 e1b(0.9759) 和 full_e40(0.9783)。若校准目标为 precision≥0.99，使用 e1b Top-50 + full_e40 重排（alpha=0），阈值 0.69，留出集 coverage=0.8277、precision=0.9937、accuracy=0.8224。
- **pegSection 无收益**：最优仍为纯 e1b；precision≥0.99 时校准阈值过高且留出集 coverage 仅约 6%，说明该场景不应与 StackBase 共用阈值或融合策略。

报告位于 `runs/rec/ablation/two_stage/`。

### 两个业务测试集的 0.60–0.70 阈值结果

本项目业务侧将 **precision** 定义为 `(Top-1 SKU 正确且相似度>阈值的数量) / 全量测试数据`，拒识计为未命中；评估脚本 JSON 中该值的字段名是 `accuracy`。每个单元格为 **业务 precision（全量分母）/ accepted precision（超阈值分母）**，后者计算 `(Top-1 正确且过阈值) / 过阈值数量`。完整 JSON（含 coverage）位于 `runs/rec/ablation/threshold_curves/`。

#### pegSection（527 张测试图）

| 模型 | 0.60 | 0.62 | 0.64 | 0.66 | 0.68 | 0.70 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | .8482/.8482 | .8444/.8492 | .8444/.8492 | .8387/.8500 | .8349/.8494 | .8330/.8508 |
| full_e40 | .8406/.8878 | .8330/.8887 | .8292/.8882 | .8216/.8891 | .8008/.8884 | .7799/.8858 |
| **e1b** | **.8520/.8552** | **.8501/.8550** | **.8482/.8596** | **.8463/.8762** | **.8406/.8755** | **.8368/.8785** |
| arcface | .8046/.8870 | .7913/.8872 | .7647/.8857 | .7476/.8894 | .7021/.8873 | .6698/.8869 |
| e1b_stop30@39 | .8444/.8525 | .8425/.8538 | .8406/.8585 | .8349/.8627 | .8330/.8676 | .8273/.8668 |

pegSection 上 e1b 在 0.60–0.70 全区间取得最高业务 precision；若优先总命中率推荐阈值 **0.60**，若更重视 accepted precision 可选 **0.66**。full_e40/arcface 的 accepted precision 较高，但分数整体偏低、拒识更多；arcface 无阈值 Top-1 为 0.8425。

#### StackBase（3689 张测试图）

| 模型 | 0.60 | 0.62 | 0.64 | 0.66 | 0.68 | 0.70 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | **.9705**/.9705 | **.9705**/.9705 | **.9705**/.9707 | **.9705**/.9710 | **.9705**/.9712 | **.9696**/.9715 |
| full_e40 | .9311/.9859 | .9149/.9874 | .8924/.9907 | .8696/.9923 | .8327/.9922 | .7970/.9926 |
| e1b | .9696/.9771 | .9672/.9770 | .9618/.9785 | .9523/.9791 | .9433/.9808 | .9287/.9825 |
| arcface | .9013/.9902 | .8756/.9911 | .8463/.9911 | .8075/.9920 | .7666/.9930 | .7232/.9940 |
| e1b_stop30@39 | .9686/.9765 | .9642/.9769 | .9604/.9771 | .9547/.9791 | .9442/.9795 | .9314/.9806 |

StackBase 上单模型若优先业务 precision，baseline 在该区间仍最佳；e1b 在阈值 0.60 时几乎保持 baseline 业务 precision，同时 accepted precision 提升至 0.9771。full_e40/arcface 适合高 accepted precision、允许低 coverage 的场景：full_e40@0.64 为业务 precision=0.8924、accepted precision=0.9907、coverage=0.9008；arcface@0.60 为 0.9013/0.9902/0.9103。arcface 无阈值 Top-1 为 0.9794。

### 训练后期关闭一致性损失

独立脚本：`runs/rec/ablation/run_e1b_stop30.sh`（仓库模板：`examples/run_e1b_stop30_h100.sh`）。实验从 e1b epoch30 恢复，自 epoch31 起 `consistency weight=0`，训练至 epoch39。

| checkpoint | P1 | P2 | P3 | P4 | P5 | P6 |
|---|---:|---:|---:|---:|---:|---:|
| 原 e1b@30 | 0.8378 | 0.8456 | 0.9051 | 0.9962 | 0.9333 | 0.9916 |
| 原 e1b@39 | 0.8351 | 0.8380 | 0.9076 | **0.9975** | 0.9286 | 0.9912 |
| stop30 best@38 | **0.8405** | 0.8519 | 0.9063 | 0.9962 | **0.9333** | **0.9916** |
| **stop30@39** | **0.8405** | **0.8532** | **0.9076** | 0.9962 | **0.9333** | **0.9916** |

**六维结论**：关闭一致性后，epoch39 相对原 e1b@39 的 P1/P2/P5 分别提升 +0.0054/+0.0152/+0.0047，P3 持平，P4 下降 0.0013；相对原 e1b@30 也小幅提升 P1/P2/P3。因此 stop30@39 是当前六维最均衡 checkpoint，但 P1/P2/P3/P5 仍未达到 0.95 门禁，overall 仍为 FAIL。

**业务检索结论**：stop30@39 的无阈值 Top-1 为 pegSection=0.8520、StackBase=0.9753，均未超过原 e1b（0.8539/0.9759）。pegSection 在 0.60–0.70 全区间也低于原 e1b；StackBase 仅在较高阈值略有全量 precision 增益（0.70：0.9314 vs 0.9287）。precision≥0.99 的独立校准在 StackBase 选择阈值 0.82，留出集 business precision=0.7194、accepted precision=0.9917、coverage=0.7254；pegSection 校准阈值 0.93 在留出集仅得到 accepted precision=0.8519、coverage=0.1484，不能满足目标。因此 stop30 可作为六维探针候选，但不替换当前业务检索 e1b，pegSection 也不应使用 0.99 precision 硬门禁。

产物：`e1b_stop30_report.{json,md}`（best@38）、`e1b_stop30_epoch39_report.{json,md}`、`threshold_curves/{peg,stack}_e1b_stop30_epoch39.{json,md}`。
