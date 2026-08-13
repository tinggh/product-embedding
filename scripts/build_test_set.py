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

P1 相似度筛选（--p1_select sim）：
    随机抽样的 P1 对很多差异来自角度/特征面而非颜色本身。sim 模式调用
    feature_extractor 的 FastAPI 服务（POST /api/v1/predict，见
    feature_extractor/app/tests/test_fastapi.py）提取特征，同 product_line 内
    按 SKU 质心余弦相似度降序选"最难区分"的 diff 对，图片取跨 SKU 最相似的一对。
    服务不在本机时可用 ssh 反向隧道：
        ssh -R 18080:127.0.0.1:8080 <gpu_host>
        --feature_url http://127.0.0.1:18080/api/v1/predict

仅重建 P1（复用已有 skus.csv，不动其他维度）：
    python build_test_set.py --only_p1 --p1_select sim \
        --dataset_root /path/sku100wdata --output_dir /path/testset \
        --feature_url http://127.0.0.1:18080/api/v1/predict
"""

import argparse
import base64
import csv
import json
import os
import os.path as osp
import random
import shutil
import time
from collections import defaultdict

import numpy as np
import requests
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


def _foreground_mask(arr, border=6):
    """用四周边缘像素估计背景色，前景 = 与背景颜色差异显著的区域。

    商品图背景通常是大面积纯色（白/灰/货架），边缘取 6px 带估计背景均值与
    离散度，逐像素与背景的色差超过阈值即判为前景。返回 bool 掩码 (H, W)。
    """
    b = np.concatenate([
        arr[:border].reshape(-1, 3), arr[-border:].reshape(-1, 3),
        arr[:, :border].reshape(-1, 3), arr[:, -border:].reshape(-1, 3),
    ]).astype(np.float32)
    bg = b.mean(axis=0)
    bg_spread = float(b.std(axis=0).mean())
    diff = np.abs(arr.astype(np.float32) - bg).mean(axis=2)
    return diff > max(3.0 * bg_spread, 20.0)


def _pick_occl_box(fg, bw, bh, rng, tries=30, min_overlap=0.4):
    """在前景上选遮挡框位置：候选框锚定在随机前景像素上（保证压住商品），
    取前景覆盖率最高者，达到 min_overlap 即提前返回；无前景时退回中心区。"""
    h, w = fg.shape
    ys, xs = np.nonzero(fg)
    if len(xs) == 0:
        cx = max(0, min(w - bw, (w - bw) // 2 + rng.randint(-w // 8, w // 8)))
        cy = max(0, min(h - bh, (h - bh) // 2 + rng.randint(-h // 8, h // 8)))
        return cx, cy
    best, best_ov = None, -1.0
    for _ in range(tries):
        i = rng.randrange(len(xs))
        x0 = min(max(int(xs[i]) - rng.randint(0, bw - 1), 0), w - bw)
        y0 = min(max(int(ys[i]) - rng.randint(0, bh - 1), 0), h - bh)
        ov = float(fg[y0 : y0 + bh, x0 : x0 + bw].mean())
        if ov > best_ov:
            best, best_ov = (x0, y0), ov
        if ov >= min_overlap:
            return best
    return best


def make_occluded(src_path, dst_path, rng):
    """原图 + 前景感知遮挡 + 亮度扰动（P4 合成代理）。

    遮挡块要求落在商品前景上（与背景色差大的区域），避免随机遮挡只盖住
    背景、对特征提取没有实际挑战的问题。
    """
    img = Image.open(src_path).convert("RGB")
    w, h = img.size
    arr = np.asarray(img).copy()
    fg = _foreground_mask(arr)
    for _ in range(rng.randint(1, 3)):
        bw = rng.randint(w // 8, w // 3)
        bh = rng.randint(h // 8, h // 3)
        x0, y0 = _pick_occl_box(fg, bw, bh, rng)
        arr[y0 : y0 + bh, x0 : x0 + bw] = rng.randint(180, 255)
    img = Image.fromarray(arr)
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.7, 1.2))
    img.save(dst_path, quality=95)


def _to_jpeg_b64(path):
    """读图并重编码为 JPEG base64。

    服务端用 turbojpeg 解码，只认 JPEG；数据集里混有 PNG/WebP，
    直接透传原始字节会让整批推理失败，这里统一用 PIL 重编码。
    """
    from io import BytesIO
    img = Image.open(path).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def extract_features(feature_url, images, req_batch=64, timeout=600):
    """调用 feature_extractor FastAPI 服务批量提特征。

    images: [(key, abs_path), ...]，返回 {key: np.ndarray}。
    接口见 feature_extractor/app/tests/test_fastapi.py。
    解码失败的图片跳过（不计入返回）。
    """
    feats = {}
    pending = []
    n_bad = 0

    with requests.Session() as session:
        def flush():
            if not pending:
                return
            payload = {
                "id": f"p1build-{int(time.time() * 1000)}",
                "image_slices": [b for _, b in pending],
                "shelf_info": {"shelf_code": "na", "customer": "na", "store": "na"},
            }
            resp = session.post(feature_url, json=payload, timeout=timeout)
            resp.raise_for_status()
            body = resp.json()
            f_list = (body.get("data") or {}).get("features")
            if not f_list or len(f_list) != len(pending):
                raise RuntimeError(
                    f"feature service 返回特征数异常: "
                    f"{0 if not f_list else len(f_list)}/{len(pending)}, body={str(body)[:200]}"
                )
            for (k, _), fv in zip(pending, f_list):
                feats[k] = np.asarray(fv, dtype=np.float32)
            pending.clear()
            print(f"  features: {len(feats)}/{len(images)}", flush=True)

        for key, path in images:
            try:
                b64 = _to_jpeg_b64(path)
            except Exception as e:
                n_bad += 1
                print(f"  skip undecodable image {path}: {e}", flush=True)
                continue
            pending.append((key, b64))
            if len(pending) >= req_batch:
                flush()
        flush()
    if n_bad:
        print(f"  skipped {n_bad} undecodable images")
    return feats


def _l2norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def build_p1_pairs_random(sel_groups, selected_set, sku_images, src_path, p1_dir, args, rng):
    """随机抽样 P1：同 line 内随机 SKU 对 + 随机图片（原方案）。"""
    pairs = []
    case = 0
    for group in sel_groups:
        in_group = [s for s in group if s in selected_set and s in sku_images]
        if len(in_group) < 2:
            continue
        for _ in range(min(3, len(in_group) - 1)):
            a, b = rng.sample(in_group, 2)
            fa, fb = rng.choice(sku_images[a]), rng.choice(sku_images[b])
            la, lb = f"{a}__{fa}", f"{b}__{fb}"
            link(src_path(a, fa), osp.join(p1_dir, la))
            link(src_path(b, fb), osp.join(p1_dir, lb))
            pairs.append((f"images/{la}", f"images/{lb}", "diff", f"case{case:05d}"))
            case += 1
        if len(pairs) >= args.max_p1_pairs:
            break
    return pairs[: args.max_p1_pairs]


def build_p1_pairs_sim(sel_groups, selected_set, sku_images, src_path, p1_dir, args, rng):
    """相似度筛选 P1：同 line 内按 SKU 质心相似度降序选最难区分的 diff 对。

    随机抽样选出的对很多差异来自角度/特征面而非颜色本身，代表性差；
    这里取最相似的 SKU 对，且图片取跨 SKU 最相似的一对（ hardest case ）。
    """
    cand_skus = sorted(
        {s for g in sel_groups for s in g if s in selected_set and s in sku_images}
    )
    sampled = {}
    todo = []
    for sku in cand_skus:
        imgs = sku_images[sku]
        pick = rng.sample(imgs, min(args.p1_images_per_sku, len(imgs)))
        sampled[sku] = pick
        todo.extend((f"{sku}__{fn}", src_path(sku, fn)) for fn in pick)
    print(f"p1 sim: 提取 {len(cand_skus)} 个 SKU / {len(todo)} 张图的特征 ...")
    img_feat = {k: _l2norm(v) for k, v in extract_features(
        args.feature_url, todo, args.req_batch).items()}

    centroid = {}
    for sku, pick in sampled.items():
        vs = [img_feat[f"{sku}__{fn}"] for fn in pick if f"{sku}__{fn}" in img_feat]
        if vs:
            centroid[sku] = _l2norm(np.mean(vs, axis=0))

    pairs = []
    case = 0
    for group in sel_groups:
        in_group = [s for s in group if s in centroid]
        if len(in_group) < 2:
            continue
        scored = sorted(
            (
                (float(centroid[a] @ centroid[b]), a, b)
                for i, a in enumerate(in_group)
                for b in in_group[i + 1 :]
            ),
            key=lambda t: -t[0],
        )
        for sim, a, b in scored[:3]:
            best, ba, bb = -2.0, None, None
            for fa in sampled[a]:
                va = img_feat.get(f"{a}__{fa}")
                if va is None:
                    continue
                for fb in sampled[b]:
                    vb = img_feat.get(f"{b}__{fb}")
                    if vb is None:
                        continue
                    s = float(va @ vb)
                    if s > best:
                        best, ba, bb = s, fa, fb
            if ba is None:
                continue
            la, lb = f"{a}__{ba}", f"{b}__{bb}"
            link(src_path(a, ba), osp.join(p1_dir, la))
            link(src_path(b, bb), osp.join(p1_dir, lb))
            pairs.append((f"images/{la}", f"images/{lb}", "diff", f"case{case:05d}"))
            case += 1
        if len(pairs) >= args.max_p1_pairs:
            break
    return pairs[: args.max_p1_pairs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--fingerprint_csv")
    parser.add_argument("--hierarchy_json")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_skus", type=int, default=1000)
    parser.add_argument("--variant_quota", type=int, default=700)
    parser.add_argument("--open_ratio", type=float, default=0.2)
    parser.add_argument("--max_p1_pairs", type=int, default=2000)
    parser.add_argument("--max_same_pairs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--p1_select", choices=["random", "sim"], default="random",
                        help="P1 选对方案：random=随机；sim=按特征相似度选最难区分的对")
    parser.add_argument("--feature_url", default="http://127.0.0.1:8080/api/v1/predict",
                        help="feature_extractor FastAPI 服务地址（p1_select=sim 时使用）")
    parser.add_argument("--p1_images_per_sku", type=int, default=8,
                        help="sim 模式下每个 SKU 抽样提特征的图片数")
    parser.add_argument("--req_batch", type=int, default=64,
                        help="单次 predict 请求的图片数")
    parser.add_argument("--only_p1", action="store_true",
                        help="仅重建 output_dir/probe/p1_color_variant（复用已有 skus.csv）")
    parser.add_argument("--redo_p4", action="store_true",
                        help="按已有 p4 pairs.csv 重新合成遮挡图（img_a 不变，仅重写 img_b）")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # ---- redo_p4：按已有 pairs.csv 重新合成遮挡图 ----
    if args.redo_p4:
        p4_root = osp.join(args.output_dir, "probe", "p4_occlusion")
        with open(osp.join(p4_root, "pairs.csv"), encoding="utf-8-sig") as f:
            rows = [r for r in csv.reader(f) if r and r[0].startswith("images/")]
        for img_a, img_b, *_ in rows:
            src = osp.join(p4_root, img_a)  # images/ 下软链，指向数据集原图
            dst = osp.join(p4_root, img_b)  # images/ 下之前合成的遮挡图（实文件，直接覆盖）
            make_occluded(src, dst, rng)
        print(f"p4_occlusion: 重新合成 {len(rows)} 张遮挡图（前景感知） -> {p4_root}/images")
        return

    # ---- only_p1：复用已有 skus.csv，仅重建 P1 ----
    if args.only_p1:
        with open(osp.join(args.output_dir, "skus.csv"), encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        selected_set = {r["ean"] for r in rows}
        line_to_skus = defaultdict(list)
        for r in rows:
            line_to_skus[r["product_line"]].append(r["ean"])
        sel_groups = [sorted(v) for v in line_to_skus.values() if len(v) >= 2]
        rng.shuffle(sel_groups)
        sku_images = {
            sku: list_images(osp.join(args.dataset_root, sku)) for sku in selected_set
        }
        sku_images = {k: v for k, v in sku_images.items() if len(v) >= 2}

        p1_root = osp.join(args.output_dir, "probe", "p1_color_variant")
        if osp.isdir(p1_root):
            shutil.rmtree(p1_root)
        p1_dir = osp.join(p1_root, "images")
        os.makedirs(p1_dir, exist_ok=True)

        def src_path(sku, fname):
            return osp.join(args.dataset_root, sku, fname)

        if args.p1_select == "sim":
            p1_pairs = build_p1_pairs_sim(
                sel_groups, selected_set, sku_images, src_path, p1_dir, args, rng)
        else:
            p1_pairs = build_p1_pairs_random(
                sel_groups, selected_set, sku_images, src_path, p1_dir, args, rng)
        with open(osp.join(p1_root, "pairs.csv"), "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerows(p1_pairs)
        print(f"p1_color_variant: {len(p1_pairs)} pairs ({args.p1_select}) -> {p1_root}")
        return

    if not args.fingerprint_csv or not args.hierarchy_json:
        parser.error("完整构建模式需要 --fingerprint_csv 和 --hierarchy_json")

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
    sel_groups = [g for g in variant_groups if sum(s in selected_set for s in g) >= 2]
    rng.shuffle(sel_groups)
    if args.p1_select == "sim":
        p1_pairs = build_p1_pairs_sim(
            sel_groups, selected_set, sku_images, src_path, p1_dir, args, rng)
    else:
        p1_pairs = build_p1_pairs_random(
            sel_groups, selected_set, sku_images, src_path, p1_dir, args, rng)

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
