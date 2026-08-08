"""将 sku100wdata 风格数据集（root 下直接是 barcode 类目目录，无 split）准备为
RetailProduct 可用的格式：直接生成 entries-*.npy / class-ids-*.npy / class-names-*.npy，
按"每类内部按比例切分图片"划分 train/val/test，无需移动/链接任何图片。

用法（在 dinov3-main 仓库根的同级运行，或把本脚本路径加入 PYTHONPATH）：
    python prepare_sku_dataset.py --dataset_root /path/sku100wdata0324
生成后训练：
    torchrun --nproc_per_node=1 dinov3/train/finetune_v2.py --train \
        --dataset_root /path/sku100wdata0324 ...
"""

import argparse
import os
import re
import sys

import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def scan_classes(dataset_root):
    classes = []
    for name in sorted(os.listdir(dataset_root)):
        dirpath = os.path.join(dataset_root, name)
        if not os.path.isdir(dirpath):
            continue
        images = [
            f
            for f in os.listdir(dirpath)
            if os.path.splitext(f)[1].lower() in IMG_EXTS
        ]
        if images:
            classes.append((name, sorted(images)))
    return classes


def split_images(images, val_ratio, test_ratio, rng):
    """每类内部切分；>=10 张按比例，3~9 张各取 1 张进 val/test，<3 张全进 train。"""
    idx = rng.permutation(len(images))
    n = len(images)
    if n >= 10:
        n_val = max(1, int(round(n * val_ratio)))
        n_test = max(1, int(round(n * test_ratio)))
    elif n >= 3:
        n_val = n_test = 1
    else:
        n_val = n_test = 0
    val_idx = idx[:n_val]
    test_idx = idx[n_val : n_val + n_test]
    train_idx = idx[n_val + n_test :]
    return train_idx, val_idx, test_idx


def build_entries(rows):
    """rows: list of (class_index, class_id, class_name, image_relpath)"""
    max_cn = max(len(r[2]) for r in rows)
    max_rp = max(len(r[3]) for r in rows)
    dtype = np.dtype(
        [
            ("class_index", "<u4"),
            ("class_id", "<u4"),
            ("class_name", f"U{max_cn}"),
            ("image_relpath", f"U{max_rp}"),
        ]
    )
    arr = np.empty(len(rows), dtype=dtype)
    for i, r in enumerate(rows):
        arr[i] = r
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument(
        "--output", default="",
        help="npy 输出目录（默认写回 dataset_root；数据集只读时指定，训练时配合 --dataset_extra 使用）",
    )
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    classes = scan_classes(args.dataset_root)
    assert classes, f"no class dirs found under {args.dataset_root}"
    print(f"classes: {len(classes)}")

    class_ids = np.arange(len(classes), dtype="<u4")
    class_names = np.array([c for c, _ in classes], dtype=f"U{max(len(c) for c, _ in classes)}")

    splits = {"TRAIN": [], "VAL": [], "TEST": []}
    for class_index, (class_name, images) in enumerate(classes):
        tr, va, te = split_images(images, args.val_ratio, args.test_ratio, rng)
        for split_name, split_idx in (("TRAIN", tr), ("VAL", va), ("TEST", te)):
            for i in split_idx:
                relpath = os.path.join(class_name, images[i])
                splits[split_name].append((class_index, class_index, class_name, relpath))

    out_dir = args.output or args.dataset_root
    os.makedirs(out_dir, exist_ok=True)
    for split_name, rows in splits.items():
        entries = build_entries(rows)
        np.save(os.path.join(out_dir, f"entries-{split_name}.npy"), entries)
        np.save(os.path.join(out_dir, f"class-ids-{split_name}.npy"), class_ids)
        np.save(os.path.join(out_dir, f"class-names-{split_name}.npy"), class_names)
        print(f"{split_name}: {len(rows)} images")

    print("done. npy written to:", out_dir)


if __name__ == "__main__":
    main()
