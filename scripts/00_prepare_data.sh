#!/usr/bin/env bash
# 数据准备流水线：metadata → hierarchy → 测试集选择 → npy 切分（含开集留出）
#
# 在任意 GPU 容器/服务器上运行前，按环境设置以下变量（或用默认值）：
#   PY            python 可执行文件路径（默认 python3）
#   REPO          dinov3-main 仓库根（overlay 已 install）
#   PE            product-embedding 仓库根
#   DATASET_ROOT  数据集目录（root 下直接是 barcode 类目目录）
#   FINGERPRINT   ean,product_name CSV（可为空字符串跳过层级/测试集）
#   WORK_DIR      产物输出目录（npy、hierarchy、testset）
#
# 用法: PY=/path/to/python REPO=/path/dinov3-main PE=/path/product-embedding \
#       DATASET_ROOT=/path/sku100wdata FINGERPRINT=/path/fingerprint.csv \
#       WORK_DIR=/path/work bash 00_prepare_data.sh
set -euo pipefail

PY="${PY:-python3}"
REPO="${REPO:?set REPO to dinov3-main root}"
PE="${PE:?set PE to product-embedding root}"
DATASET_ROOT="${DATASET_ROOT:?set DATASET_ROOT}"
FINGERPRINT="${FINGERPRINT:-}"
WORK_DIR="${WORK_DIR:?set WORK_DIR}"

mkdir -p "$WORK_DIR"
HIERARCHY="$WORK_DIR/hierarchy.json"
SKUS_CSV="$WORK_DIR/testset/skus.csv"
OPEN_LIST="$WORK_DIR/testset/open_list.txt"

if [ -n "$FINGERPRINT" ]; then
    echo "==> [1/4] fingerprint -> metadata.csv"
    "$PY" - <<EOF
import csv
with open("$FINGERPRINT", encoding="utf-8-sig") as f, open("$WORK_DIR/metadata.csv", "w", encoding="utf-8", newline="") as out:
    w = csv.writer(out); w.writerow(["barcode", "name"])
    for row in csv.DictReader(f):
        w.writerow([row["ean"].strip(), row["product_name"].strip()])
print("metadata.csv written")
EOF

    echo "==> [2/4] build hierarchy"
    cd "$REPO"
    PYTHONPATH="$REPO" "$PY" -m app.build_hierarchy \
        --dataset_root "$DATASET_ROOT" \
        --metadata_csv "$WORK_DIR/metadata.csv" \
        --output "$HIERARCHY"
    mv -f "$REPO/hierarchy_stats.txt" "$WORK_DIR/hierarchy_stats.txt" 2>/dev/null || true
    mv -f "$REPO/split_leakage.txt" "$WORK_DIR/split_leakage.txt" 2>/dev/null || true

    echo "==> [3/4] select test skus + build probe set"
    "$PY" "$PE/scripts/build_test_set.py" \
        --dataset_root "$DATASET_ROOT" \
        --fingerprint_csv "$FINGERPRINT" \
        --hierarchy_json "$HIERARCHY" \
        --output_dir "$WORK_DIR/testset"
else
    echo "==> [1-3/4] FINGERPRINT 未提供，跳过 hierarchy 与测试集构建"
fi

echo "==> [4/4] prepare npy splits (holdout open skus if any)"
PREPARE_EXTRA=()
if [ -f "$OPEN_LIST" ]; then
    PREPARE_EXTRA=(--holdout_list "$OPEN_LIST")
fi
"$PY" "$PE/scripts/prepare_sku_dataset.py" \
    --dataset_root "$DATASET_ROOT" \
    --output "$WORK_DIR/npy" \
    "${PREPARE_EXTRA[@]}"

echo "DONE. artifacts in $WORK_DIR"
