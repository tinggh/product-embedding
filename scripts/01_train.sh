#!/usr/bin/env bash
# 训练启动脚本（单容器多卡 torchrun）。
#
# 必设环境变量：
#   PY            python 可执行文件路径
#   REPO          dinov3-main 仓库根（overlay 已 install）
#   DATASET_ROOT  数据集目录（barcode 类目目录）
#   NPY_DIR       prepare_sku_dataset.py 的 npy 输出目录
#   CKPT          初始化权重（官方 LVD 或 SSL teacher ckpt）
#   OUTPUT_DIR    训练输出目录
#
# 可选：
#   NGPU          使用 GPU 数（默认 1）
#   CUDA_VISIBLE_DEVICES  GPU 选择（默认 0）
#   MASTER_PORT   分布式端口（默认 29511）
#   HIERARCHY     hierarchy.json 路径（提供则启用硬负样本采样）
#   EXTRA_ARGS    追加给 finetune_v2.py 的额外参数（如 "--max_epoch 1"）
#   LD_PRELOAD_LIB  若 torch 报 nvJitLink 符号错误，设为 env 内 libnvJitLink.so.12 路径
#
# 默认参数对应消融终点 E5 全量配置；调参见 overlay/app/RUNBOOK_finetune_v2.md。
set -euo pipefail

PY="${PY:?set PY}"
REPO="${REPO:?set REPO}"
DATASET_ROOT="${DATASET_ROOT:?set DATASET_ROOT}"
NPY_DIR="${NPY_DIR:?set NPY_DIR}"
CKPT="${CKPT:?set CKPT}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR}"

NGPU="${NGPU:-1}"
MASTER_PORT="${MASTER_PORT:-29511}"
HIERARCHY="${HIERARCHY:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$REPO"
export LD_PRELOAD="${LD_PRELOAD_LIB:-${LD_PRELOAD:-}}"

# 每次运行的日志独立落盘，避免共享代码库时多任务日志混杂：
#   $OUTPUT_DIR/app.log    —— 训练过程日志（run_logger，读 DINOV3_RUN_LOG）
#   $OUTPUT_DIR/train.log  —— 完整 stdout/stderr（含报错栈）
mkdir -p "$OUTPUT_DIR"
export DINOV3_RUN_LOG="$OUTPUT_DIR/app.log"
exec > >(tee -a "$OUTPUT_DIR/train.log") 2>&1

HARD_NEG=()
if [ -n "$HIERARCHY" ]; then
    HARD_NEG=(--hierarchy_json "$HIERARCHY" --hard_ratio 0.5)
fi

cd "$REPO"
# shellcheck disable=SC2086
"$PY" -m torch.distributed.run --nproc_per_node="$NGPU" --master_port="$MASTER_PORT" \
    dinov3/train/finetune_v2.py --train \
    --dataset_root "$DATASET_ROOT" \
    --dataset_extra "$NPY_DIR" \
    --ckpt_path "$CKPT" \
    --output_dir "$OUTPUT_DIR" \
    --loss subcenter --num_subcenters 3 --center_lambda 0.5 \
    --pooling cls+gem --unfreeze_last 24 \
    --aug color_preserving --hue 0.02 \
    --consistency_lambda 0.5 \
    "${HARD_NEG[@]}" \
    $EXTRA_ARGS
