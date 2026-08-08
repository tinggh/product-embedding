# finetune_v2 运行手册（GPU 服务器，4090-24G）

环境：`/mnt/upfs/Hanshow/Env/miniconda3/envs/feature_extractor`（conda activate feature_extractor）
工作目录：仓库根 `dinov3-main/`（以 `python -m` 方式运行 app 脚本）。

## Step 0：层级标签与 split 泄漏校验

```bash
python -m app.build_hierarchy \
    --dataset_root /ya/Dataset/shelf/liuting/rec/tesco_skus_dataset \
    --output hierarchy.json
# 检查同目录输出的 hierarchy_stats.txt（含 ≥2 变体的 product_line 数，即硬负样本来源）
# 与 split_leakage.txt（同一 barcode 跨 split 的泄漏列表，必须为空或人工确认）
```

## Step 1：六维探针评测集

按以下结构整理（pairs.csv 无表头：`img_a,img_b,label,case_id`，label ∈ same/diff）：

```
probe_root/
  p1_color_variant/pairs.csv   # 同系列不同口味/颜色 SKU 对（diff）
  p2_multi_view/pairs.csv      # 同 SKU 正面-侧面-背面（same）
  p3_part_whole/pairs.csv      # 局部 crop vs 整体（same）
  p4_occlusion/pairs.csv       # 吊牌/阴影/模糊实拍（same）
  p5_similar_items/pairs.csv   # 已知易混淆对（diff，收录 Embedding 分析文档中的 case）
  p6_open_set/query/ gallery/  # 文件名 {class_name}__{后缀}.jpg
```

```bash
# 对任意 ckpt 出六维报告（自动兼容旧 best.pth 与新 ProductEmbedder ckpt）
python -m app.probe_eval --probe_root probe_root --ckpt best.pth --output report_e0
```

## Step 2：E0 基线复现

用旧脚本（未改动）：`torchrun --nproc_per_node=1 dinov3/train/finetune.py --train ...`，
或直接对现有 best.pth 跑 probe_eval，拿到六维初始报告作为对照。

## Step 3：消融矩阵（finetune_v2.py）

统一入口：

```bash
torchrun --nproc_per_node=1 dinov3/train/finetune_v2.py --train \
    --dataset_root /ya/Dataset/shelf/liuting/rec/tesco_skus_dataset \
    --ckpt_path /ya/Code/liuting/modelscope/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
    --output_dir /ya/Code/liuting/runs/rec/<exp_name> \
    [实验参数]
```

| 实验 | 参数 | 说明 |
|---|---|---|
| E1a 深解冻 4 层 | `--unfreeze_last 4 --loss arcface --pooling cls --aug legacy` | 隔离"解冻深度"单变量 |
| E1b 深解冻 12 层 | `--unfreeze_last 12 --loss arcface --pooling cls --aug legacy` | 同上 |
| E1c 全解冻 | `--unfreeze_last 24 --loss arcface --pooling cls --aug legacy` | 同上 |
| E2 保色增强 | E1 最优 + `--aug color_preserving` | hue 默认 0.02，P1 失败调小 / P4 失败调大（`--hue`） |
| E3 Sub-center | E2 + `--loss subcenter --num_subcenters 3 --center_lambda 0.5` | 多视角问题 |
| E4 一致性+GeM | E3 + `--consistency_lambda 0.5 --pooling cls+gem` | 部分-整体问题 |
| E5 硬负样本 | E4 + `--hierarchy_json hierarchy.json --hard_ratio 0.5` | 相似变体区分 |
| E6（可选） | E5 但 `--ckpt_path` 换成 SSL 续训 teacher ckpt | 终审 SSL 初始化残留价值 |

每档先用短 schedule（改 `opt.max_epoch=20~30`）跑，probe_eval 六维全绿再进入下一档；
任何一档 P6 回归变差 → 回退该变量并重定参数。定参后跑全量 100 epoch。

## 显存与吞吐（4090-24G，ViT-L/16 全解冻，bf16 AMP 已内置）

- 默认 `opt.batchsize=128`（per-GPU）+ `accum_steps=4`，有效 batch 512；
- OOM 时降到 `batchsize=64, accum_steps=8`；双视图（E4 起）显存翻倍，先降到 64；
- batch/学习率等训练超参在 `finetune_v2.py` 顶部 `opt` dataclass 中调整。

## 断点续训

```bash
--resume_path /ya/Code/liuting/runs/rec/<exp>/vitl_epoch_20.pth   # 自动恢复 optimizer/scheduler/epoch
```

## 部署注意（Phase 3 预告）

新 ckpt 为 ProductEmbedder 格式（key 含 `backbone./gem./proj.`），
`feature_extractor/dist/model_convert.py` 导出 ONNX 时需同步改为加载 ProductEmbedder
（GeM 仅 pow/mean 算子，TRT 友好）。**阈值体系需用 generate_threshold_report.py 重建，勿沿用旧阈值。**
