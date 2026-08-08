#!/usr/bin/env bash
# 六维探针评测。
#
# 环境变量：
#   PY          python 可执行文件路径
#   REPO        dinov3-main 仓库根
#   PROBE_ROOT  探针集目录（build_test_set.py 输出的 testset/probe）
#   CKPT        待评测 ckpt（旧 best.pth 或新 ProductEmbedder 格式均可）
#   OUTPUT      报告 basename（生成 OUTPUT.md / OUTPUT.json）
#   POOLING     cls | gem | cls+gem（新 ckpt 用 cls+gem，旧 ckpt 忽略）
#   LD_PRELOAD_LIB  若 torch 报 nvJitLink 符号错误，设为 env 内 libnvJitLink.so.12 路径
set -euo pipefail

PY="${PY:?set PY}"
REPO="${REPO:?set REPO}"
PROBE_ROOT="${PROBE_ROOT:?set PROBE_ROOT}"
CKPT="${CKPT:?set CKPT}"
OUTPUT="${OUTPUT:?set OUTPUT}"
POOLING="${POOLING:-cls+gem}"
export LD_PRELOAD="${LD_PRELOAD_LIB:-${LD_PRELOAD:-}}"

cd "$REPO"
PYTHONPATH="$REPO" "$PY" -m app.probe_eval \
    --probe_root "$PROBE_ROOT" \
    --ckpt "$CKPT" \
    --output "$OUTPUT" \
    --pooling "$POOLING"
