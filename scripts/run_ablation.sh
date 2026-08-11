#!/usr/bin/env bash
# 并行消融实验队列：按 GPU 数把实验分配到各卡并行执行，每个实验 = 短跑训练 + 六维探针评测。
#
# 实验矩阵（leave-one-out，与 full 基线单变量对照）：
#   full         完整配置（E5 终点）
#   shallow      只解冻最后 1 层        → 验证深解冻价值（E1）
#   legacy_aug   旧增强（无保色管道）    → 验证保色增强价值（E2）
#   arcface      单中心 ArcFace         → 验证 Sub-center+Center（E3）
#   cls_pool     无双视图一致性 + CLS   → 验证一致性损失与 GeM（E4）
#   no_hardneg   无硬负样本采样         → 验证层级硬负挖掘（E5）
#   g2m          CLS + G2M 双均值池化    → 对比 GeM 聚合头（E6）
#   salad        SALAD Sinkhorn 局部聚合 → 对比 GeM 聚合头（E7）
#
# 环境变量：
#   PY REPO DATASET_ROOT NPY_DIR CKPT HIERARCHY PROBE_ROOT WORK_DIR  必设
#   GPU_GROUPS  分号分隔的 GPU 组，组内逗号分隔（如 "0,1;2,3;4,5;6,7" = 4 组双卡）。
#               实验按组轮询分配、组间并行、组内串行（多出的实验排在该组后面）。
#               兼容旧用法 GPUS="0,1"（每卡一组）。
#   EPOCHS      训练 epoch 数（默认 100，完整训练）
#   ITERS       每 epoch 最大 iter（默认 0 = 完整 epoch；仅快速冒烟时设值）
#   MILESTONES  lr 衰减里程碑（默认 "30,60,90"，应随 EPOCHS 调整）
#   SAVE_INTERVAL 评估/保存间隔（默认 10）
#   SKIP        逗号分隔要跳过的实验名（如 "full"——已有正式全量跑时）
#   PORT_BASE   torchrun master_port 起始值（默认 29700，冲突时改）
#   LD_PRELOAD_LIB  nvJitLink 修复
#
# 产物：$WORK_DIR/ablation/<name>/（训练 ckpt + train.log + app.log）与 <name>_report.json/.md
set -euo pipefail

PY="${PY:?}"; REPO="${REPO:?}"; DATASET_ROOT="${DATASET_ROOT:?}"; NPY_DIR="${NPY_DIR:?}"
CKPT="${CKPT:?}"; PROBE_ROOT="${PROBE_ROOT:?}"; WORK_DIR="${WORK_DIR:?}"
HIERARCHY="${HIERARCHY:-}"
GPU_GROUPS="${GPU_GROUPS:-${GPUS:-0}}"
EPOCHS="${EPOCHS:-100}"
ITERS="${ITERS:-0}"
MILESTONES="${MILESTONES:-30,60,90}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
SKIP="${SKIP:-}"
export LD_PRELOAD="${LD_PRELOAD_LIB:-${LD_PRELOAD:-}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

NAMES=(full shallow legacy_aug arcface cls_pool no_hardneg g2m salad)

extra_args_for() {
    case "$1" in
        full)        echo "" ;;
        shallow)     echo "--unfreeze_last 1" ;;
        legacy_aug)  echo "--aug legacy" ;;
        arcface)     echo "--loss arcface" ;;
        cls_pool)    echo "--consistency_lambda 0 --pooling cls" ;;
        no_hardneg)  echo "" ;;
        g2m)         echo "--pooling cls+g2m" ;;
        salad)       echo "--pooling salad" ;;
    esac
}

# 探针评测的 pooling 必须与该实验训练时的 pooling 一致，否则 state_dict 结构不匹配
probe_pooling_for() {
    case "$1" in
        cls_pool)    echo "cls" ;;
        g2m)         echo "cls+g2m" ;;
        salad)       echo "salad" ;;
        *)           echo "cls+gem" ;;
    esac
}

run_one() {
    local name="$1" gpu="$2" port="$3"
    local out="$WORK_DIR/ablation/$name"
    local ngpu
    ngpu=$(awk -F',' '{print NF}' <<< "$gpu")
    mkdir -p "$out"
    local hard=()
    if [ -n "$HIERARCHY" ] && [ "$name" != "no_hardneg" ]; then
        hard=(--hierarchy_json "$HIERARCHY" --hard_ratio 0.5)
    fi
    echo "[gpu $gpu] start $name"
    cd "$REPO"
    # 断点续训：输出目录里已有 vitl_epoch_*.pth 时从最新一个恢复
    # （训练未完成时）；已完成 40 epoch 的实验应通过 SKIP 排除
    local resume=()
    local latest
    latest=$(ls -v "$out"/vitl_epoch_*.pth 2>/dev/null | tail -1)
    if [ -n "$latest" ]; then
        resume=(--resume_path "$latest")
        echo "[gpu $gpu] resume $name from $latest"
    fi
    export DINOV3_RUN_LOG="$out/app.log"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$REPO" "$PY" -m torch.distributed.run \
        --nproc_per_node="$ngpu" --master_port="$port" \
        dinov3/train/finetune_v2.py --train \
        --dataset_root "$DATASET_ROOT" --dataset_extra "$NPY_DIR" \
        --ckpt_path "$CKPT" --output_dir "$out" \
        --loss subcenter --num_subcenters 3 --center_lambda 0.5 \
        --pooling cls+gem --unfreeze_last 24 \
        --aug color_preserving --hue 0.02 --consistency_lambda 0.5 \
        --batchsize 96 --accum_steps 3 --num_workers 12 \
        --max_epoch "$EPOCHS" --max_iters_per_epoch "$ITERS" \
        --lr_milestones "$MILESTONES" --save_interval "$SAVE_INTERVAL" \
        "${hard[@]}" "${resume[@]}" $(extra_args_for "$name") \
        >> "$out/train.log" 2>&1
    echo "[gpu $gpu] train done $name, probing"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$REPO" "$PY" -m app.probe_eval \
        --probe_root "$PROBE_ROOT" --ckpt "$out/best.pth" \
        --output "$WORK_DIR/ablation/${name}_report" --pooling "$(probe_pooling_for "$name")" \
        > "$out/probe.log" 2>&1 || echo "[gpu $gpu] probe FAILED $name"
    echo "[gpu $gpu] done $name"
}

IFS=';' read -ra GPU_ARR <<< "$GPU_GROUPS"
IFS=',' read -ra SKIP_ARR <<< "$SKIP"
declare -A GPU_JOBS
port="${PORT_BASE:-29700}"
for name in "${NAMES[@]}"; do
    skip=0
    for s in "${SKIP_ARR[@]}"; do [ "$s" == "$name" ] && skip=1; done
    [ $skip -eq 1 ] && continue
    # 选当前任务最少的 GPU（按已分配任务字符串长度近似）
    best="${GPU_ARR[0]}"
    for g in "${GPU_ARR[@]}"; do
        cur="${GPU_JOBS[$g]:-}"
        best_jobs="${GPU_JOBS[$best]:-}"
        [ "${#cur}" -lt "${#best_jobs}" ] && best="$g"
    done
    GPU_JOBS[$best]="${GPU_JOBS[$best]:-} $name:$port"
    port=$((port + 1))
done

pids=()
for g in "${GPU_ARR[@]}"; do
    jobs="${GPU_JOBS[$g]:-}"
    [ -z "$jobs" ] && continue
    (
        for job in $jobs; do
            name="${job%%:*}"; p="${job##*:}"
            run_one "$name" "$g" "$p"
        done
    ) &
    pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done

echo "all ablations done, collecting reports"
"$PY" "$(dirname "${BASH_SOURCE[0]}")/collect_reports.py" \
    --report_glob "$WORK_DIR/ablation/*_report.json" \
    --output "$WORK_DIR/ablation/comparison.md"
echo "comparison -> $WORK_DIR/ablation/comparison.md"
