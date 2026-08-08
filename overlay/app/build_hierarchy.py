#! python3
# -*- encoding: utf-8 -*-
"""
@Describe : 从 ImageFolder 数据集生成层级标签文件 hierarchy.json，
            供训练时的同系列（product_line）硬负样本采样使用。
            同时输出 product_line 统计与 split 泄漏校验结果。
@Usage    : python -m app.build_hierarchy --dataset_root /path/to/dataset \
                --output hierarchy.json [--metadata_csv meta.csv]
"""

import argparse
import csv
import json
import os
import os.path as osp
import re
from collections import defaultdict

from app.log_module import logger

SPLITS = ["train", "val", "test"]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# 常见口味/香型词（中英文，匹配时不区分大小写），用于从商品名尾部剥离以得到 product_line
FLAVOR_WORDS = [
    "原味", "无糖", "低糖", "零糖", "抹茶", "草莓", "巧克力", "黑巧", "香草", "蓝莓",
    "西瓜", "苹果", "香橙", "橙子", "橙", "葡萄", "柠檬", "青柠", "牛奶", "咖啡",
    "芝士", "奶酪", "榛子", "榛果", "椰子", "芒果", "桃子", "水蜜桃", "香蕉", "焦糖",
    "海盐", "薄荷", "黄瓜", "番茄", "烧烤", "香辣", "麻辣", "酸辣", "蜂蜜", "奶油",
    "蓝莓味", "草莓味", "巧克力味", "香草味", "柠檬味", "橙子味", "苹果味", "葡萄味",
    "blueberry", "strawberry", "chocolate", "vanilla", "lemon", "orange", "apple",
    "grape", "original", "mint", "caramel", "coconut", "hazelnut", "coffee", "matcha",
    "peach", "mango", "banana", "milk", "cheese", "honey", "sea salt", "sugar free",
]
FLAVOR_SET = {w.lower() for w in FLAVOR_WORDS}
# 尾部口味词剥离（长词优先，避免“蓝莓”先于“蓝莓味”命中；容忍“黄瓜味/黄瓜口味”形式）
FLAVOR_SUFFIX_RE = re.compile(
    r"(?:%s)(?:味|口味)?$" % "|".join(sorted((re.escape(w) for w in FLAVOR_SET), key=len, reverse=True)),
    re.IGNORECASE,
)
# 规格：500g / 1.5L / 250ml / 135克 / 500毫升 / 1升 / x6 / 12x500ml 等
SPEC_TOKEN_RE = re.compile(
    r"^(\d+(\.\d+)?\s*(kg|g|ml|l|克|毫升|升)|[xX×]\s*\d+|\d+\s*[xX×]\s*\d+(\.\d+)?\s*(kg|g|ml|l|克|毫升|升)?)$",
    re.IGNORECASE,
)
SPEC_SUFFIX_RE = re.compile(
    r"(\d+(\.\d+)?\s*(kg|g|ml|l|克|毫升|升)([xX×]\s*\d+)?|[xX×]\s*\d+)$",
    re.IGNORECASE,
)
# 包装量词（箱/袋/盒/瓶/包/罐/条），仅作尾部剥离
PACK_SUFFIX_RE = re.compile(r"(箱|袋|盒|瓶|包|罐|条|装)$")


def split_class_name(class_name):
    """class_name 形如 {id}_{barcode}_{商品名}，返回 (id, barcode, name)。"""
    parts = class_name.split("_", 2)
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return parts[0], parts[1], ""
    return class_name, "", ""


def parse_brand(product_name):
    """brand = 商品名第一个 token；中文商品名无空格时置空字符串。"""
    tokens = product_name.split()
    if len(tokens) >= 2:
        return tokens[0]
    return ""


def strip_product_name(product_name):
    """从商品名尾部迭代剥离口味/规格/包装词，得到产品系名（仍含 brand 前缀）。"""
    name = product_name.strip()
    changed = True
    while changed and name:
        changed = False
        if " " in name:
            tokens = name.split()
            while len(tokens) > 1 and (
                tokens[-1].lower() in FLAVOR_SET
                or SPEC_TOKEN_RE.match(tokens[-1])
                or PACK_SUFFIX_RE.search(tokens[-1])
            ):
                tokens.pop()
                changed = True
            new_name = " ".join(tokens)
        else:
            new_name = FLAVOR_SUFFIX_RE.sub("", name)
            new_name = SPEC_SUFFIX_RE.sub("", new_name)
            new_name = PACK_SUFFIX_RE.sub("", new_name)
        if new_name != name:
            changed = True
            name = new_name.strip()
    return name


def load_metadata(metadata_csv):
    """读取权威元数据（列：barcode,brand,category,product_line[,name]），按 barcode 索引。"""
    metadata = {}
    with open(metadata_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            barcode = (row.get("barcode") or "").strip()
            if barcode:
                metadata[barcode] = {
                    "brand": (row.get("brand") or "").strip(),
                    "category": (row.get("category") or "").strip(),
                    "product_line": (row.get("product_line") or "").strip(),
                    "name": (row.get("name") or "").strip(),
                }
    logger.info(f"Loaded metadata for {len(metadata)} barcodes from {metadata_csv}")
    return metadata


def scan_dataset(dataset_root):
    """扫描 train/val/test split，返回 {split: {class_name: n_images}}。

    若不存在 split 子目录（如 sku100wdata 风格：root 下直接是类目目录，
    split 由 prepare_sku_dataset.py 生成的 npy 虚拟划分），退化为扫描 root
    本身并整体记为 train split。
    """
    has_splits = any(osp.isdir(osp.join(dataset_root, s)) for s in SPLITS)
    scan_roots = (
        {s: osp.join(dataset_root, s) for s in SPLITS}
        if has_splits
        else {"train": dataset_root}
    )
    split_classes = {}
    for split, split_dir in scan_roots.items():
        classes = {}
        if not osp.isdir(split_dir):
            logger.warning(f"split dir not found, skipped: {split_dir}")
            split_classes[split] = classes
            continue
        for class_name in sorted(os.listdir(split_dir)):
            class_dir = osp.join(split_dir, class_name)
            if not osp.isdir(class_dir):
                continue
            n_images = 0
            for _, _, files in os.walk(class_dir):
                n_images += sum(1 for f in files if f.lower().endswith(IMG_EXTS))
            classes[class_name] = n_images
        logger.info(f"split={split}: {len(classes)} classes, {sum(classes.values())} images")
        split_classes[split] = classes
    return split_classes


def build_record(class_name, metadata):
    """生成单条层级记录。"""
    _, barcode, product_name = split_class_name(class_name)
    variant_id = barcode if barcode else class_name

    meta = metadata.get(barcode) if barcode else None
    if meta is not None and not product_name:
        # 类目名为纯 barcode 时，用元数据中的商品名做 brand/product_line 推导
        product_name = meta.get("name", "")

    brand = parse_brand(product_name)
    category = ""
    if meta is not None:
        brand = meta["brand"] or brand
        category = meta["category"]
        product_line = meta["product_line"]
    else:
        product_line = ""

    if not product_line:
        stripped = strip_product_name(product_name)
        if stripped:
            product_line = stripped
        else:
            product_line = brand if brand else class_name

    return {
        "brand": brand,
        "category": category,
        "product_line": product_line,
        "variant_id": variant_id,
    }


def check_split_leakage(split_classes):
    """同一 variant_id 出现在多个 split 即为泄漏，返回 [(variant_id, [splits])]。"""
    variant_splits = defaultdict(set)
    for split, classes in split_classes.items():
        for class_name in classes:
            _, barcode, _ = split_class_name(class_name)
            variant_id = barcode if barcode else class_name
            variant_splits[variant_id].add(split)
    leakage = [(vid, sorted(splits)) for vid, splits in variant_splits.items() if len(splits) > 1]
    return sorted(leakage)


def write_stats(hierarchy, stats_path):
    """输出 product_line 统计：总数、多 variant 系列数、top 20。"""
    line_variants = defaultdict(set)
    for class_name, record in hierarchy.items():
        line_variants[record["product_line"]].add(record["variant_id"])

    multi_variant = {pl: vids for pl, vids in line_variants.items() if len(vids) >= 2}
    top20 = sorted(line_variants.items(), key=lambda kv: len(kv[1]), reverse=True)[:20]

    lines = []
    lines.append(f"total classes: {len(hierarchy)}")
    lines.append(f"total product_lines: {len(line_variants)}")
    lines.append(f"product_lines with >= 2 variants (hard negative source): {len(multi_variant)}")
    lines.append("")
    lines.append("top 20 largest product_lines (by variant count):")
    for pl, vids in top20:
        lines.append(f"  {len(vids):4d}  {pl}")

    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"hierarchy stats written to {stats_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="build hierarchy labels from dataset")
    parser.add_argument("--dataset_root", required=True, help="dataset root with train/val/test splits")
    parser.add_argument("--output", default="hierarchy.json", help="output hierarchy json path")
    parser.add_argument("--metadata_csv", default=None,
                        help="optional CSV with barcode,brand,category,product_line columns")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(f"Building hierarchy from {args.dataset_root}")

    metadata = load_metadata(args.metadata_csv) if args.metadata_csv else {}
    split_classes = scan_dataset(args.dataset_root)

    # 合并所有 split 的 class_name（同一 class 可能同时出现在多个 split，记录唯一）
    all_classes = sorted({c for classes in split_classes.values() for c in classes})
    hierarchy = {class_name: build_record(class_name, metadata) for class_name in all_classes}
    logger.info(f"total classes collected: {len(hierarchy)}")

    output_dir = osp.dirname(osp.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(hierarchy, f, ensure_ascii=False, indent=2)
    logger.info(f"hierarchy written to {args.output}")

    write_stats(hierarchy, osp.join(output_dir, "hierarchy_stats.txt"))

    leakage = check_split_leakage(split_classes)
    leakage_path = osp.join(output_dir, "split_leakage.txt")
    with open(leakage_path, "w", encoding="utf-8") as f:
        if leakage:
            f.write("WARNING: the following variant_ids appear in multiple splits:\n")
            for vid, splits in leakage:
                f.write(f"  {vid}: {', '.join(splits)}\n")
        else:
            f.write("no split leakage found\n")
    if leakage:
        logger.warning(f"split leakage detected for {len(leakage)} variant_ids, see {leakage_path}")
    else:
        logger.info("no split leakage found")


if __name__ == "__main__":
    main()
