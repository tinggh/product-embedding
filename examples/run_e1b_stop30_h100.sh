#!/usr/bin/env bash
# e1b epoch30 后关闭图像级一致性损失，续训至 epoch39，并自动评测。
# 直接执行会通过 setsid+nohup 转入后台；`run` 子命令用于后台子进程。
set -euo pipefail

ROOT=${ROOT:-/mnt/upfs/Hanshow}
L=$ROOT/Code/liuting
PY=${PY:-$ROOT/Env/miniconda3/envs/feature_extractor/bin/python}
REPO=${REPO:-$L/dinov3-main}
OUT=${OUT:-$L/runs/rec/ablation/e1b_stop30}
SOURCE_CKPT=${SOURCE_CKPT:-$L/runs/rec/ablation/e1b/vitl_epoch_30.pth}
PROBE_ROOT=${PROBE_ROOT:-$L/work_sku100w/testset/probe}
GPUS=${GPUS:-1,3}
MASTER_PORT=${MASTER_PORT:-29930}
LOG=$OUT/train.log

mkdir -p "$OUT"

if [[ "${1:-start}" != "run" ]]; then
    if pgrep -f "finetune_v2.py.*--output_dir $OUT" >/dev/null; then
        echo "e1b_stop30 is already running"
        exit 0
    fi
    nohup setsid env ROOT="$ROOT" L="$L" PY="$PY" REPO="$REPO" OUT="$OUT" \
        SOURCE_CKPT="$SOURCE_CKPT" PROBE_ROOT="$PROBE_ROOT" GPUS="$GPUS" \
        MASTER_PORT="$MASTER_PORT" bash "$0" run </dev/null >> "$OUT/launcher.log" 2>&1 &
    echo "started pid=$!; launcher=$OUT/launcher.log; train=$LOG"
    exit 0
fi

latest=$(find "$OUT" -maxdepth 1 -name 'vitl_epoch_*.pth' -printf '%f\n' \
    | sort -V | tail -1 || true)
if [[ -z "$latest" ]]; then
    cp "$SOURCE_CKPT" "$OUT/vitl_epoch_30.pth"
    latest=vitl_epoch_30.pth
fi
latest_epoch=${latest#vitl_epoch_}
latest_epoch=${latest_epoch%.pth}

if (( latest_epoch < 39 )); then
    ngpu=$(awk -F',' '{print NF}' <<< "$GPUS")
    export CUDA_VISIBLE_DEVICES="$GPUS"
    export PYTHONPATH="$REPO"
    export LD_PRELOAD=${LD_PRELOAD:-$ROOT/Env/miniconda3/envs/feature_extractor/lib/python3.10/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12}
    export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
    "$PY" -m torch.distributed.run --nproc_per_node="$ngpu" --master_port="$MASTER_PORT" \
        "$REPO/dinov3/train/finetune_v2.py" --train \
        --dataset_root "$ROOT/Code/konglingmei/Datasets/sku100wdata" \
        --dataset_extra "$L/work_sku100w/npy" \
        --ckpt_path "$L/modelscope/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth" \
        --resume_path "$OUT/$latest" --output_dir "$OUT" \
        --loss subcenter --num_subcenters 3 --center_lambda 0.5 \
        --pooling cls+gem+salad --unfreeze_last 24 \
        --aug color_preserving --hue 0.02 \
        --consistency_lambda 0.5 --consistency_stop_epoch 30 \
        --batchsize 96 --accum_steps 3 --num_workers 8 \
        --max_epoch 40 --lr_milestones 12,24,36 --save_interval 2 \
        --hierarchy_json "$L/work_sku100w/hierarchy.json" --hard_ratio 0.5 \
        >> "$LOG" 2>&1
fi

ckpt="$OUT/best.pth"
[[ -f "$ckpt" ]] || ckpt="$OUT/vitl_epoch_39.pth"
CUDA_VISIBLE_DEVICES=${PROBE_GPU:-${GPUS%%,*}} PYTHONPATH="$REPO" \
    "$PY" -m app.probe_eval --probe_root "$PROBE_ROOT" --ckpt "$ckpt" \
    --output "$L/runs/rec/ablation/e1b_stop30_report" \
    --pooling cls+gem+salad --min-pass-rate 0.95 --p6-min-acc 0.9 \
    > "$OUT/probe.log" 2>&1

