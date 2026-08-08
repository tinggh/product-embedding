#!/usr/bin/env bash
# h100 宿主机 8 卡并行消融实验启动脚本（实际运行版本，2026-08-08）
# 用法: sudo bash run.sh
# 说明: 容器内使用时把 ROOT 改为 /ya
ROOT=${ROOT:-/mnt/upfs/Hanshow}
L=$ROOT/Code/liuting

env PY=$ROOT/Env/miniconda3/envs/feature_extractor/bin/python \
    REPO=$L/dinov3-main \
    DATASET_ROOT=$ROOT/Code/konglingmei/Datasets/sku100wdata \
    NPY_DIR=$L/work_sku100w/npy \
    CKPT=$L/modelscope/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
    HIERARCHY=$L/work_sku100w/hierarchy.json \
    PROBE_ROOT=$L/work_sku100w/testset/probe \
    WORK_DIR=$L/runs/rec \
    LD_PRELOAD_LIB=$ROOT/Env/miniconda3/envs/feature_extractor/lib/python3.10/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12 \
    GPU_GROUPS="0,1;2,3;4,5;6,7" \
    EPOCHS=40 \
    MILESTONES=12,24,36 \
    SKIP=full \
    nohup bash $L/product-embedding/scripts/run_ablation.sh \
        > $L/runs/rec/ablation/ablation_main.log 2>&1 &

echo "ablation launched, main log: $L/runs/rec/ablation/ablation_main.log"
