#! python3
# -*- encoding: utf-8 -*-
"""
@Describe : 六维探针评测脚本，作为所有训练实验的门禁。
            P1 颜色/口味变体(diff) / P2 多角度(same) / P3 局部-整体(same) /
            P4 遮挡模糊(same) / P5 易混淆对(diff) / P6 开放集回归(top-1)。
@Usage    : python -m app.probe_eval --probe_root /path/to/probe \
                --ckpt /path/best.pth --output report [--pooling cls+gem]
"""

import argparse
import csv
import json
import os
import os.path as osp
import re
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from app.log_module import logger
from app.general_model import ImageProcessor
from dinov3.models.embedder import build_product_embedder
from dinov3.models.vision_transformer import vit_large

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# 维度名 -> (probe 子目录名, pair 类型)
PROBE_PAIRS = OrderedDict([
    ("P1_color_variant", ("p1_color_variant", "diff")),
    ("P2_multi_view", ("p2_multi_view", "same")),
    ("P3_part_whole", ("p3_part_whole", "same")),
    ("P4_occlusion", ("p4_occlusion", "same")),
    ("P5_similar_items", ("p5_similar_items", "diff")),
])
P6_DIR = "p6_open_set"

QUANTILES = [5, 25, 50, 75, 95]


class ProbeFeatureExtractor:
    """加载 ckpt 并批量提取 L2 归一化 embedding。

    ckpt 兼容逻辑：state_dict 的 key 带 `backbone.` 前缀时按 ProductEmbedder
    加载（新格式，输出 backbone + 池化 + 投影后的 embedding）；否则按裸
    vit_large backbone 加载（旧格式 best.pth），embedding 取 CLS token。
    """

    def __init__(self, ckpt_path, pooling="cls+gem", device="cuda:0"):
        self.ckpt_path = ckpt_path
        self.device = device
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )
        self.load(pooling)

    def load(self, pooling):
        logger.info(f"Loading model from {self.ckpt_path}")
        state_dict = torch.load(self.ckpt_path, map_location="cpu")
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        if isinstance(state_dict, dict) and "teacher" in state_dict:
            state_dict = state_dict["teacher"]
        state_dict = OrderedDict(
            (re.sub(r"^module\.", "", k), v) for k, v in state_dict.items()
        )

        if any(k.startswith("backbone.") for k in state_dict):
            logger.info(f"checkpoint detected as ProductEmbedder format, pooling={pooling}")
            model = build_product_embedder(pooling=pooling)
            try:
                msg = model.load_state_dict(state_dict, strict=True)
            except RuntimeError as e:
                logger.warning(f"strict load failed ({e}), fallback to strict=False")
                msg = model.load_state_dict(state_dict, strict=False)
            logger.info(f"ProductEmbedder loaded with msg: {msg}")
            self.model = model
            self.forward_fn = model
        else:
            logger.info("checkpoint detected as bare backbone format, use CLS token")
            backbone = vit_large(
                img_size=224,
                patch_size=16,
                pos_embed_rope_base=100,
                qkv_bias=True,
                layerscale_init=1e-5,
                norm_layer="layernorm",
                ffn_layer="mlp",
                ffn_bias=True,
                proj_bias=True,
                n_storage_tokens=4,
                mask_k_bias=True,
                untie_cls_and_patch_norms=False,
                untie_global_and_local_cls_norm=False,
            )
            msg = backbone.load_state_dict(state_dict, strict=False)
            logger.info(f"bare backbone loaded with msg: {msg}")
            self.model = backbone

            def forward_cls(x):
                out = backbone(x, is_training=True)
                return F.normalize(out["x_norm_clstoken"], p=2, dim=-1)

            self.forward_fn = forward_cls

        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def extract(self, image_paths, batch_size=64):
        """对一组图片路径批量提取特征，返回 {path: np.ndarray}。"""
        features = {}
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            pil_imgs = [ImageProcessor.preprocess_image(p, isb64=False) for p in batch_paths]
            input_tensor = torch.stack([self.transform(p) for p in pil_imgs]).to(self.device)
            if self.device.startswith("cuda"):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    feats = self.forward_fn(input_tensor)
            else:
                feats = self.forward_fn(input_tensor)
            feats = F.normalize(feats.float(), p=2, dim=-1).cpu().numpy()
            for path, feat in zip(batch_paths, feats):
                features[path] = feat
            logger.info(f"extracted features {min(i + batch_size, len(image_paths))}/{len(image_paths)}")
        return features


def read_pairs(pairs_csv):
    """读取 pairs.csv（img_a,img_b,label,case_id），容忍表头行。"""
    rows = []
    with open(pairs_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 4:
                continue
            if row[0].strip().lower() == "img_a":
                continue
            rows.append({
                "img_a": row[0].strip(),
                "img_b": row[1].strip(),
                "label": row[2].strip().lower(),
                "case_id": row[3].strip(),
            })
    return rows


def cosine(feat_a, feat_b):
    return float(np.dot(feat_a, feat_b))


def eval_pairs(probe_dir, kind, features, diff_thresh, same_thresh, min_pass_rate):
    """评估 P1~P5 的 pair 型维度。diff 型：sim > diff_thresh 为失败；same 型：sim < same_thresh 为失败。"""
    pairs_csv = osp.join(probe_dir, "pairs.csv")
    rows = read_pairs(pairs_csv)

    sims = []
    failures = defaultdict(list)  # case_id -> [(img_a, img_b, sim)]
    for row in rows:
        path_a = osp.join(probe_dir, row["img_a"])
        path_b = osp.join(probe_dir, row["img_b"])
        if path_a not in features or path_b not in features:
            logger.warning(f"missing feature for pair: {row['img_a']}, {row['img_b']}")
            continue
        sim = cosine(features[path_a], features[path_b])
        sims.append(sim)
        failed = (sim > diff_thresh) if kind == "diff" else (sim < same_thresh)
        if failed:
            failures[row["case_id"]].append((row["img_a"], row["img_b"], round(sim, 4)))

    sims = np.array(sims) if sims else np.array([0.0])
    n_total = len(rows)
    n_fail = sum(len(v) for v in failures.values())
    pass_rate = (n_total - n_fail) / n_total if n_total else 0.0

    if kind == "diff":
        stats = {"mean": float(sims.mean()), "p95": float(np.percentile(sims, 95)), "max": float(sims.max())}
        thresh = diff_thresh
    else:
        stats = {"mean": float(sims.mean()), "p5": float(np.percentile(sims, 5)), "min": float(sims.min())}
        thresh = same_thresh

    return {
        "type": kind,
        "n_pairs": n_total,
        "stats": {k: round(v, 4) for k, v in stats.items()},
        "threshold": thresh,
        "n_failed": n_fail,
        "pass_rate": round(pass_rate, 4),
        "failures": {cid: fs for cid, fs in sorted(failures.items())},
        "passed": pass_rate >= min_pass_rate,
    }


def parse_class_from_filename(filename):
    """p6 文件名含 class_name 前缀，约定 {class_name}__{任意后缀}.jpg；无分隔符时取整个 stem。"""
    stem = osp.splitext(osp.basename(filename))[0]
    return stem.split("__")[0]


def eval_open_set(probe_dir, features, p6_min_acc):
    """评估 P6：query 对 gallery 的 top-1 命中率 + same/diff 相似度分布。"""
    query_dir = osp.join(probe_dir, "query")
    gallery_dir = osp.join(probe_dir, "gallery")

    def collect(d):
        items = []
        for r, _, files in os.walk(d):
            for f in sorted(files):
                if f.lower().endswith(IMG_EXTS):
                    path = osp.join(r, f)
                    items.append((path, parse_class_from_filename(f)))
        return items

    queries = collect(query_dir)
    gallery = collect(gallery_dir)
    logger.info(f"P6: {len(queries)} queries, {len(gallery)} gallery images")

    gallery_feats = np.stack([features[p] for p, _ in gallery])
    gallery_classes = [c for _, c in gallery]

    n_hit = 0
    same_sims, diff_sims = [], []
    misses = []
    for q_path, q_class in queries:
        sims = gallery_feats @ features[q_path]
        best_idx = int(np.argmax(sims))
        hit = gallery_classes[best_idx] == q_class
        n_hit += int(hit)
        if not hit:
            misses.append((osp.relpath(q_path, probe_dir), q_class,
                           gallery_classes[best_idx], round(float(sims[best_idx]), 4)))
        for g_class, s in zip(gallery_classes, sims):
            (same_sims if g_class == q_class else diff_sims).append(float(s))

    top1 = n_hit / len(queries) if queries else 0.0

    def dist(values):
        if not values:
            return {}
        q = np.percentile(np.array(values), QUANTILES)
        return {f"p{p}": round(float(v), 4) for p, v in zip(QUANTILES, q)}

    return {
        "type": "open_set",
        "n_query": len(queries),
        "n_gallery": len(gallery),
        "top1_acc": round(top1, 4),
        "threshold": p6_min_acc,
        "same_sim_dist": dist(same_sims),
        "diff_sim_dist": dist(diff_sims),
        "misses": misses,
        "passed": top1 >= p6_min_acc,
    }


def collect_image_paths(probe_root):
    """汇总所有 probe 维度涉及的唯一图片路径。"""
    paths = set()
    for _, (sub_dir, _) in PROBE_PAIRS.items():
        pairs_csv = osp.join(probe_root, sub_dir, "pairs.csv")
        if not osp.isfile(pairs_csv):
            continue
        for row in read_pairs(pairs_csv):
            paths.add(osp.join(probe_root, sub_dir, row["img_a"]))
            paths.add(osp.join(probe_root, sub_dir, row["img_b"]))
    p6_dir = osp.join(probe_root, P6_DIR)
    for sub in ("query", "gallery"):
        d = osp.join(p6_dir, sub)
        if not osp.isdir(d):
            continue
        for r, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith(IMG_EXTS):
                    paths.add(osp.join(r, f))
    return sorted(paths)


def render_markdown(report, args):
    lines = []
    lines.append("# Probe Evaluation Report")
    lines.append("")
    lines.append(f"- ckpt: `{args.ckpt}`")
    lines.append(f"- probe_root: `{args.probe_root}`")
    lines.append(f"- pooling: `{args.pooling}`")
    lines.append(f"- overall: **{report['overall']}**")
    lines.append("")
    lines.append("| dim | type | n | key stats | pass_rate/acc | threshold | result |")
    lines.append("|---|---|---|---|---|---|---|")
    for dim, res in report["dimensions"].items():
        if res.get("skipped"):
            lines.append(f"| {dim} | - | - | - | - | - | SKIPPED |")
            continue
        stats_str = ", ".join(f"{k}={v}" for k, v in res["stats"].items()) if "stats" in res else \
            f"top1={res['top1_acc']}"
        rate = res.get("pass_rate", res.get("top1_acc"))
        result = "PASS" if res["passed"] else "FAIL"
        lines.append(f"| {dim} | {res['type']} | {res.get('n_pairs', res.get('n_query'))} "
                     f"| {stats_str} | {rate:.4f} | {res['threshold']} | {result} |")
    lines.append("")

    for dim, res in report["dimensions"].items():
        if res.get("skipped"):
            continue
        lines.append(f"## {dim}")
        if "failures" in res:
            if not res["failures"]:
                lines.append("")
                lines.append("no failures")
            for case_id, fs in res["failures"].items():
                lines.append("")
                lines.append(f"- case `{case_id}`:")
                for img_a, img_b, sim in fs:
                    lines.append(f"    - `{img_a}` vs `{img_b}` sim={sim}")
        elif res.get("misses") is not None:
            lines.append("")
            lines.append(f"- same sim dist: {res['same_sim_dist']}")
            lines.append(f"- diff sim dist: {res['diff_sim_dist']}")
            if res["misses"]:
                lines.append("- misses:")
                for q, q_cls, g_cls, sim in res["misses"]:
                    lines.append(f"    - `{q}` ({q_cls}) -> {g_cls} sim={sim}")
            else:
                lines.append("- no misses")
        lines.append("")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="six-dimension probe evaluation")
    parser.add_argument("--probe_root", required=True, help="probe root directory")
    parser.add_argument("--ckpt", required=True, help="model checkpoint path")
    parser.add_argument("--output", default="report", help="output report basename (writes .md/.json)")
    parser.add_argument("--pooling", default="cls+gem", choices=["cls", "gem", "cls+gem", "cls+g2m", "salad"],
                        help="pooling mode for ProductEmbedder checkpoints")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--diff-thresh", type=float, default=0.7,
                        help="diff-type pairs with sim above this are failures")
    parser.add_argument("--same-thresh", type=float, default=0.75,
                        help="same-type pairs with sim below this are failures")
    parser.add_argument("--min-pass-rate", type=float, default=1.0,
                        help="min pair pass rate for P1~P5 to be judged PASS")
    parser.add_argument("--p6-min-acc", type=float, default=0.9,
                        help="min top-1 accuracy for P6 to be judged PASS")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(f"probe eval start: probe_root={args.probe_root}, ckpt={args.ckpt}")

    image_paths = collect_image_paths(args.probe_root)
    logger.info(f"total unique images to extract: {len(image_paths)}")

    extractor = ProbeFeatureExtractor(args.ckpt, pooling=args.pooling, device=args.device)
    features = extractor.extract(image_paths, batch_size=args.batch_size)

    dimensions = OrderedDict()
    for dim, (sub_dir, kind) in PROBE_PAIRS.items():
        probe_dir = osp.join(args.probe_root, sub_dir)
        if not osp.isfile(osp.join(probe_dir, "pairs.csv")):
            logger.warning(f"{dim}: pairs.csv not found under {probe_dir}, skipped")
            dimensions[dim] = {"skipped": True, "passed": True}
            continue
        logger.info(f"evaluating {dim} ({kind})")
        dimensions[dim] = eval_pairs(probe_dir, kind, features,
                                     args.diff_thresh, args.same_thresh, args.min_pass_rate)

    p6_dir = osp.join(args.probe_root, P6_DIR)
    if osp.isdir(osp.join(p6_dir, "query")) and osp.isdir(osp.join(p6_dir, "gallery")):
        logger.info("evaluating P6_open_set")
        dimensions["P6_open_set"] = eval_open_set(p6_dir, features, args.p6_min_acc)
    else:
        logger.warning(f"P6_open_set: query/gallery dirs not found under {p6_dir}, skipped")
        dimensions["P6_open_set"] = {"skipped": True, "passed": True}

    evaluated = [r for r in dimensions.values() if not r.get("skipped")]
    overall = "PASS" if all(r["passed"] for r in evaluated) else "FAIL"
    report = {"overall": overall, "ckpt": args.ckpt, "dimensions": dimensions}

    out_base = args.output
    out_dir = osp.dirname(osp.abspath(out_base))
    os.makedirs(out_dir, exist_ok=True)
    json_path = out_base + ".json"
    md_path = out_base + ".md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(report, args))
    logger.info(f"report written to {md_path} and {json_path}")
    logger.info(f"overall: {overall}")


if __name__ == "__main__":
    main()
