"""从 sku100wdata 构建评测测试集（闭集+开集混合）与六维探针集。

流程：
1. fingerprint（ean,product_name）× hierarchy（class_name→product_line）× 数据集类目取交集；
2. 变体组（同 product_line ≥2 个 SKU）优先选入 variant_quota 个 SKU，再随机补到 n_skus；
3. 选中 SKU 中随机 open_ratio 比例整体留出（开集，模拟新品），其余为闭集；
4. 输出 skus.csv、open_list.txt（供 prepare_sku_dataset.py --holdout_list）；
5. 生成 probe 目录（软链图片，P3/P4 为 PIL 生成的实图）：
   - p1_color_variant/pairs.csv  同 product_line 不同 SKU（diff）
   - p2_multi_view/pairs.csv     同 SKU 两张不同图（same）
   - p3_part_whole/pairs.csv     原图 vs 中心局部裁剪 40~60%（same）
   - p4_occlusion/pairs.csv      原图 vs 合成遮挡+亮度扰动（same，合成代理）
   - p5_similar_items/pairs.csv  同 product_line 内随机未选入 P1 的 SKU 对（diff）
   - p6_open_set/query|gallery   closed/open SKU 各半切分图片（{ean}__{idx}.jpg 软链）

用法：
    python build_test_set.py --dataset_root /path/sku100wdata \
        --fingerprint_csv /path/sku100wdata_product_fingerprint.csv \
        --hierarchy_json /path/hierarchy_sku100w.json \
        --output_dir /path/testset
"""

import argparse
import csv
import json
import os
import os.path as osp
import random
from collections import defaultdict

import numpy as np
from PIL import Image, ImageEnhance

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(class_dir):
    return sorted(
        f for f in os.listdir(class_dir) if osp.splitext(f)[1].lower() in IMG_EXTS
    )


def link(src, dst):
    if osp.lexists(dst):
        os.remove(dst)
    os.symlink(src, dst)


def select_skus(named_skus, line_to_skus, n_skus, variant_quota, rng):
    """变体组优先 + 随机补充。"""
    variant_groups = [sorted(v) for v in line_to_skus.values() if len(v) >= 2]
    variant_groups.sort(key=len, reverse=True)
    selected = []
    for group in variant_groups:
        if len(selected) >= variant_quota:
            break
        selected.extend(group)
    selected = selected[:variant_quota]
    remaining = [s for s in named_skus if s not in set(selected)]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, n_skus - len(selected))])
    return selected, variant_groups


def make_center_crop(src_path, dst_path, rng, scale_range=(0.4, 0.6)):
    img = Image.open(src_path).convert("RGB")
    w, h = img.size
    s = rng.uniform(*scale_range)
    cw, ch = int(w * s), int(h * s)
    x0 = rng.randint(0, w - cw)
    y0 = rng.randint(0, h - ch)
    img.crop((x0, y0, x0 + cw, y0 + ch)).save(dst_path, quality=95)


def make_occluded(src_path, dst_path, rng):
    img = Image.open(src_path).convert("RGB")
    w, h = img.size
    arr = np.asarray(img).copy()
    for _ in range(rng.randint(1, 3)):
        bw = rng.randint(w // 8, w // 3)
        bh = rng.randint(h // 8, h // 3)
        x0 = rng.randint(0, w - bw)
        y0 = rng.randint(0, h - bh)
        arr[y0 : y0 + bh, x0 : x0 + bw] = rng.randint(180, 255)
    img = Image.fromarray(arr)
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.7, 1.2))
    img.save(dst_path, quality=95)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--fingerprint_csv", required=True)
    parser.add_argument("--hierarchy_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_skus", type=int, default=1000)
    parser.add_argument("--variant_quota", type=int, default=700)
    parser.add_argument("--open_ratio", type=float, default=0.2)
    parser.add_argument("--max_p1_pairs", type=int, default=2000)
    parser.add_argument("--max_same_pairs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    with open(args.fingerprint_csv, encoding="utf-8-sig") as f:
        fingerprint = {
            row["ean"].strip(): row["product_name"].strip() for row in csv.DictReader(f)
        }
    with open(args.hierarchy_json, encoding="utf-8") as f:
        hierarchy = json.load(f)

    named_skus = sorted(
        s for s in fingerprint if osp.isdir(osp.join(args.dataset_root, s))
    )
    print(f"named skus in dataset: {len(named_skus)}")

    line_to_skus = defaultdict(list)
    for sku in named_skus:
        pl = hierarchy.get(sku, {}).get("product_line", sku)
        line_to_skus[pl].append(sku)

    selected, variant_groups = select_skus(
        named_skus, line_to_skus, args.n_skus, args.variant_quota, rng
    )
    selected_set = set(selected)
    n_open = int(len(selected) * args.open_ratio)
    open_skus = set(rng.sample(selected, n_open))
    print(
        f"selected: {len(selected)} (open: {len(open_skus)}, closed: {len(selected) - len(open_skus)}), "
        f"variant groups total: {len(variant_groups)}"
    )

    os.makedirs(args.output_dir, exist_ok=True)
    with open(osp.join(args.output_dir, "skus.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ean", "product_name", "product_line", "split"])
        for sku in sorted(selected):
            w.writerow(
                [
                    sku,
                    fingerprint[sku],
                    hierarchy.get(sku, {}).get("product_line", sku),
                    "open" if sku in open_skus else "closed",
                ]
            )
    with open(osp.join(args.output_dir, "open_list.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(open_skus)) + "\n")

    # 每个选中 SKU 的图片清单
    sku_images = {
        sku: list_images(osp.join(args.dataset_root, sku)) for sku in selected
    }
    sku_images = {k: v for k, v in sku_images.items() if len(v) >= 2}

    probe = osp.join(args.output_dir, "probe")

    def img_dir(name):
        d = osp.join(probe, name, "images")
        os.makedirs(d, exist_ok=True)
        return d

    def src_path(sku, fname):
        return osp.join(args.dataset_root, sku, fname)

    # ---- P6 open_set：query/gallery 各半 ----
    q_dir = osp.join(probe, "p6_open_set", "query")
    g_dir = osp.join(probe, "p6_open_set", "gallery")
    os.makedirs(q_dir, exist_ok=True)
    os.makedirs(g_dir, exist_ok=True)
    for sku, imgs in sku_images.items():
        idx = list(range(len(imgs)))
        rng.shuffle(idx)
        half = max(1, len(idx) // 2)
        for i, j in enumerate(idx):
            dst_dir = q_dir if i < half else g_dir
            link(src_path(sku, imgs[j]), osp.join(dst_dir, f"{sku}__{i:05d}.jpg"))

    # ---- P1 颜色/口味变体（同 line 不同 SKU，diff）----
    p1_dir = img_dir("p1_color_variant")
    p1_pairs = []
    sel_groups = [g for g in variant_groups if sum(s in selected_set for s in g) >= 2]
    rng.shuffle(sel_groups)
    case = 0
    for group in sel_groups:
        in_group = [s for s in group if s in sku_images]
        if len(in_group) < 2:
            continue
        for _ in range(min(3, len(in_group) - 1)):
            a, b = rng.sample(in_group, 2)
            fa, fb = rng.choice(sku_images[a]), rng.choice(sku_images[b])
            la, lb = f"{a}__{fa}", f"{b}__{fb}"
            link(src_path(a, fa), osp.join(p1_dir, la))
            link(src_path(b, fb), osp.join(p1_dir, lb))
            p1_pairs.append((f"images/{la}", f"images/{lb}", "diff", f"case{case:05d}"))
            case += 1
        if len(p1_pairs) >= args.max_p1_pairs:
            break
    p1_pairs = p1_pairs[: args.max_p1_pairs]

    # ---- P2 多视角（同 SKU 两图，same）----
    p2_dir = img_dir("p2_multi_view")
    p2_pairs = []
    p2_skus = [s for s in sku_images]
    rng.shuffle(p2_skus)
    for i, sku in enumerate(p2_skus[: args.max_same_pairs]):
        fa, fb = rng.sample(sku_images[sku], 2)
        la, lb = f"{sku}__{fa}", f"{sku}__{fb}"
        link(src_path(sku, fa), osp.join(p2_dir, la))
        link(src_path(sku, fb), osp.join(p2_dir, lb))
        p2_pairs.append((f"images/{la}", f"images/{lb}", "same", f"case{i:05d}"))

    # ---- P3 部分-整体（same，PIL 裁剪实图）----
    p3_dir = img_dir("p3_part_whole")
    p3_pairs = []
    p3_skus = [s for s in sku_images]
    rng.shuffle(p3_skus)
    for i, sku in enumerate(p3_skus[: args.max_same_pairs]):
        fa = rng.choice(sku_images[sku])
        la = f"{sku}__{fa}"
        lc = f"{sku}__crop_{fa}"
        link(src_path(sku, fa), osp.join(p3_dir, la))
        make_center_crop(src_path(sku, fa), osp.join(p3_dir, lc), rng)
        p3_pairs.append((f"images/{la}", f"images/{lc}", "same", f"case{i:05d}"))

    # ---- P4 遮挡（same，合成代理）----
    p4_dir = img_dir("p4_occlusion")
    p4_pairs = []
    p4_skus = [s for s in sku_images]
    rng.shuffle(p4_skus)
    for i, sku in enumerate(p4_skus[: args.max_same_pairs]):
        fa = rng.choice(sku_images[sku])
        la = f"{sku}__{fa}"
        lc = f"{sku}__occ_{fa}"
        link(src_path(sku, fa), osp.join(p4_dir, la))
        make_occluded(src_path(sku, fa), osp.join(p4_dir, lc), rng)
        p4_pairs.append((f"images/{la}", f"images/{lc}", "same", f"case{i:05d}"))

    # ---- P5 相似品（同 line 不同 SKU 的另一批 diff 对，与 P1 不重叠）----
    p5_dir = img_dir("p5_similar_items")
    p5_pairs = []
    case = 0
    for group in sel_groups:
        in_group = [s for s in group if s in sku_images]
        if len(in_group) < 2:
            continue
        a, b = rng.sample(in_group, 2)
        fa = rng.choice(sku_images[a])
        fb = rng.choice(sku_images[b])
        la, lb = f"{a}__{fa}", f"{b}__{fb}"
        link(src_path(a, fa), osp.join(p5_dir, la))
        link(src_path(b, fb), osp.join(p5_dir, lb))
        p5_pairs.append((f"images/{la}", f"images/{lb}", "diff", f"case{case:05d}"))
        case += 1
        if len(p5_pairs) >= args.max_same_pairs:
            break

    for name, pairs in (
        ("p1_color_variant", p1_pairs),
        ("p2_multi_view", p2_pairs),
        ("p3_part_whole", p3_pairs),
        ("p4_occlusion", p4_pairs),
        ("p5_similar_items", p5_pairs),
    ):
        with open(osp.join(probe, name, "pairs.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerows(pairs)
        print(f"{name}: {len(pairs)} pairs")

    print("done ->", args.output_dir)


if __name__ == "__main__":
    main()
