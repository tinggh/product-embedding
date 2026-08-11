# -*- coding: utf-8 -*-
"""六维探针集 P1~P5 pair 可视化。

每个 pair 的两张图并排放在一个单元格中，按维度分页拼成 contact sheet；
可选加载 ckpt 计算相似度并按 pass/fail 着色（绿=通过，红=失败），
用于人工检查探针集质量与直观对比模型在各维度上的表现。

输出：
    <output_dir>/<dim>_sheet_01.jpg ...   分页 contact sheet
    <output_dir>/pairs/<dim>/<case_id>.jpg  （--per_pair 时）每个 pair 单独一张

@Usage    : python -m app.visualize_probe --probe_root /path/to/probe \
                --output_dir /path/to/vis [--ckpt best.pth --pooling cls+gem]
"""

import argparse
import math
import os
import os.path as osp

from PIL import Image, ImageDraw, ImageFont

from app.log_module import logger
from app.probe_eval import PROBE_PAIRS, ProbeFeatureExtractor, cosine, read_pairs

PASS_COLOR = (46, 160, 67)    # 绿
FAIL_COLOR = (218, 54, 51)    # 红
NA_COLOR = (110, 118, 129)    # 灰（无相似度）
BG_COLOR = (255, 255, 255)
CAPTION_COLOR = (30, 30, 30)
CAPTION_H = 40                # 每个 pair 单元格的标题高度
HEADER_H = 44                 # 每页 sheet 的页头高度
GAP = 8


def load_font(size):
    for name in ("DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        for d in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/liberation", ""):
            try:
                return ImageFont.truetype(osp.join(d, name), size)
            except (OSError, IOError):
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # 老版本 Pillow 不支持 size 参数
        return ImageFont.load_default()


def make_thumb(path, thumb):
    """等比缩放并居中贴到 thumb×thumb 白底；图片缺失返回 None。"""
    try:
        img = Image.open(path).convert("RGB")
    except (OSError, FileNotFoundError):
        return None
    img.thumbnail((thumb, thumb), Image.LANCZOS)
    canvas = Image.new("RGB", (thumb, thumb), BG_COLOR)
    canvas.paste(img, ((thumb - img.width) // 2, (thumb - img.height) // 2))
    return canvas


def judge_pass(label, sim, diff_thresh, same_thresh):
    """与 probe_eval 一致：diff 型 sim > diff_thresh 失败；same 型 sim < same_thresh 失败。"""
    if label == "diff":
        return sim <= diff_thresh
    return sim >= same_thresh


def draw_pair_cell(row, probe_dir, thumb, sim, diff_thresh, same_thresh, font):
    """画一个 pair 单元格：两张缩略图并排 + 标题行，返回 (cell, passed|None)。"""
    cell_w = thumb * 2 + GAP
    cell = Image.new("RGB", (cell_w, thumb + CAPTION_H), BG_COLOR)
    draw = ImageDraw.Draw(cell)

    img_a = make_thumb(osp.join(probe_dir, row["img_a"]), thumb)
    img_b = make_thumb(osp.join(probe_dir, row["img_b"]), thumb)
    if img_a is None or img_b is None:
        draw.text((GAP, thumb // 2), "missing image", fill=FAIL_COLOR, font=font)
    else:
        cell.paste(img_a, (0, CAPTION_H))
        cell.paste(img_b, (thumb + GAP, CAPTION_H))

    passed = None
    if sim is not None:
        passed = judge_pass(row["label"], sim, diff_thresh, same_thresh)
        color = PASS_COLOR if passed else FAIL_COLOR
        verdict = "PASS" if passed else "FAIL"
        caption = f"{row['case_id']} [{row['label']}] sim={sim:.3f} {verdict}"
        # 边框着色，快速定位失败 pair
        draw.rectangle([0, 0, cell_w - 1, thumb + CAPTION_H - 1], outline=color, width=3)
    else:
        color = NA_COLOR
        caption = f"{row['case_id']} [{row['label']}]"
    draw.rectangle([0, 0, cell_w - 1, CAPTION_H - 1], fill=(245, 245, 245))
    draw.text((GAP, 4), caption, fill=color if sim is not None else CAPTION_COLOR, font=font)
    draw.text((GAP, 22), row["img_a"].split("/")[-1][:34], fill=CAPTION_COLOR, font=font)
    return cell, passed


def visualize_dim(dim, probe_dir, args, features, font, header_font):
    rows = read_pairs(osp.join(probe_dir, "pairs.csv"))
    if args.max_pairs > 0:
        rows = rows[: args.max_pairs]
    if not rows:
        logger.warning(f"{dim}: no pairs, skipped")
        return

    # 相似度（可选）
    sims = [None] * len(rows)
    if features is not None:
        for i, row in enumerate(rows):
            fa = features.get(osp.join(probe_dir, row["img_a"]))
            fb = features.get(osp.join(probe_dir, row["img_b"]))
            if fa is not None and fb is not None:
                sims[i] = cosine(fa, fb)

    cell_w = args.thumb * 2 + GAP
    cell_h = args.thumb + CAPTION_H
    pairs_per_row = args.cols
    rows_per_sheet = args.rows
    pairs_per_sheet = pairs_per_row * rows_per_sheet
    n_sheets = math.ceil(len(rows) / pairs_per_sheet)
    n_pass = 0

    for sheet_idx in range(n_sheets):
        chunk = rows[sheet_idx * pairs_per_sheet : (sheet_idx + 1) * pairs_per_sheet]
        chunk_sims = sims[sheet_idx * pairs_per_sheet : (sheet_idx + 1) * pairs_per_sheet]
        n_rows = math.ceil(len(chunk) / pairs_per_row)
        sheet_w = GAP + pairs_per_row * (cell_w + GAP)
        sheet_h = HEADER_H + n_rows * (cell_h + GAP)
        sheet = Image.new("RGB", (sheet_w, sheet_h), BG_COLOR)
        draw = ImageDraw.Draw(sheet)
        draw.text(
            (GAP, 12),
            f"{dim}  page {sheet_idx + 1}/{n_sheets}  pairs {sheet_idx * pairs_per_sheet + 1}"
            f"-{sheet_idx * pairs_per_sheet + len(chunk)}/{len(rows)}"
            + (f"  (diff_thresh={args.diff_thresh}, same_thresh={args.same_thresh})" if features else ""),
            fill=CAPTION_COLOR, font=header_font,
        )
        for i, (row, sim) in enumerate(zip(chunk, chunk_sims)):
            cell, passed = draw_pair_cell(
                row, probe_dir, args.thumb, sim, args.diff_thresh, args.same_thresh, font
            )
            if passed:
                n_pass += 1
            r, c = divmod(i, pairs_per_row)
            sheet.paste(cell, (GAP + c * (cell_w + GAP), HEADER_H + r * (cell_h + GAP)))
            if args.per_pair:
                pair_dir = osp.join(args.output_dir, "pairs", dim)
                os.makedirs(pair_dir, exist_ok=True)
                cell.save(osp.join(pair_dir, f"{row['case_id']}.jpg"), quality=92)

        out_path = osp.join(args.output_dir, f"{dim}_sheet_{sheet_idx + 1:02d}.jpg")
        sheet.save(out_path, quality=90)
        logger.info(f"{dim}: sheet saved -> {out_path}")

    if features is not None:
        logger.info(f"{dim}: {n_pass}/{len(rows)} pairs PASS")
    logger.info(f"{dim}: done, {n_sheets} sheet(s)")


def main():
    parser = argparse.ArgumentParser(description="probe P1~P5 pair visualization")
    parser.add_argument("--probe_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ckpt", default=None, help="可选：加载模型计算相似度并着色")
    parser.add_argument("--pooling", default="cls+gem",
                        choices=["cls", "gem", "cls+gem", "cls+g2m", "salad"])
    parser.add_argument("--dims", default=None, help="逗号分隔维度子集，如 P1,P5；默认全部 P1~P5")
    parser.add_argument("--max_pairs", type=int, default=0, help="每维度最多可视化的 pair 数（0=全部）")
    parser.add_argument("--per_pair", action="store_true", help="每个 pair 额外单独导出一张图")
    parser.add_argument("--cols", type=int, default=3, help="每行 pair 数")
    parser.add_argument("--rows", type=int, default=10, help="每页 sheet 的行数")
    parser.add_argument("--thumb", type=int, default=224, help="单图缩略边长")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--diff_thresh", type=float, default=0.7)
    parser.add_argument("--same_thresh", type=float, default=0.85)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    font = load_font(13)
    header_font = load_font(16)

    features = None
    if args.ckpt:
        # 只收集 P1~P5 pair 涉及的图片（不带 P6 的 query/gallery，量大用不上）
        paths = set()
        for _dim, (sub_dir, _kind) in PROBE_PAIRS.items():
            pairs_csv = osp.join(args.probe_root, sub_dir, "pairs.csv")
            if not osp.isfile(pairs_csv):
                continue
            for row in read_pairs(pairs_csv):
                paths.add(osp.join(args.probe_root, sub_dir, row["img_a"]))
                paths.add(osp.join(args.probe_root, sub_dir, row["img_b"]))
        extractor = ProbeFeatureExtractor(args.ckpt, pooling=args.pooling, device=args.device)
        features = extractor.extract(sorted(paths), batch_size=args.batch_size)

    for dim, (sub_dir, _kind) in PROBE_PAIRS.items():
        if args.dims and dim.split("_")[0] not in [d.strip() for d in args.dims.split(",")]:
            continue
        probe_dir = osp.join(args.probe_root, sub_dir)
        if not osp.isfile(osp.join(probe_dir, "pairs.csv")):
            logger.warning(f"{dim}: pairs.csv not found under {probe_dir}, skipped")
            continue
        visualize_dim(dim, probe_dir, args, features, font, header_font)

    logger.info(f"all done -> {args.output_dir}")


if __name__ == "__main__":
    main()
