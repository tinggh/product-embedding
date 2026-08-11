"""为 SKU 名称补充规格（如 500ml / 90g），解决大量同名商品无法区分的问题。

从商品主数据构建 ean -> 规格 查找表（多数据源按优先级合并），
对名称中尚不含规格的商品，把规格追加到名称末尾：
    依云天然矿泉水 -> 依云天然矿泉水500ml

数据源（后者优先）：
    1. 2186.csv                              (ean, specification)
    2. goods_dataset_..._addattr.csv         (商品条码, 规格/价签规格/ItemSpecification)
    3. goods_info.csv                        (商品条码, 规格)
    4. goods_info_0324.csv                   (商品条码, 规格)

目标文件（就地更新，首改前自动备份 .bak）：
    dinov3-main/app/sku100wdata_product_fingerprint.csv  (ean, product_name)
    work_sku100w/testset/skus.csv                        (ean, product_name, ...)
    work_sku100w/metadata.csv                            (barcode, name)

用法：
    python add_spec_to_sku_names.py --datasets_root /path/Datasets \
        --targets /path/a.csv:ean:product_name /path/b.csv:barcode:name
    加 --dry_run 只统计不写文件。
"""

import argparse
import csv
import os
import re
import shutil
import sys

csv.field_size_limit(sys.maxsize)

# 名称/规格中“数字+单位”视为已含规格（容量/重量/长度 + 计数/包装单位）
SPEC_RE = re.compile(
    r"\d+(\.\d+)?\s*(ml|毫升|mL|Ml|ML|l|L|升|g|G|克|kg|KG|千克|斤|mm|cm"
    r"|片|支|粒|包|只|个|条|瓶|罐|盒|袋|枚|颗|套|杯|卷|提|箱|听|桶|板"
    r"|刀头|入|把|块|双|对|副|组)"
)
# 无意义的规格占位值
JUNK_SPEC = {"", "详见包装", "见包装", "无", "-", "/", "暂无", "以实物为准"}


def normalize_spec(spec: str) -> str:
    """去空白，统一常见单位写法；非法/占位规格返回空串。"""
    s = re.sub(r"\s+", "", (spec or "").strip())
    if s in JUNK_SPEC or not SPEC_RE.search(s):
        return ""
    return s


def load_spec_lut(path, ean_col, spec_cols):
    lut = {}
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            ean = (row.get(ean_col) or "").strip()
            if not ean:
                continue
            for col in spec_cols:
                spec = normalize_spec(row.get(col))
                if spec:
                    lut[ean] = spec
                    break
    return lut


def build_merged_lut(datasets_root):
    """按优先级从低到高合并（后 load 的覆盖先 load 的）。"""
    sources = [
        ("2186.csv", "ean", ["specification"]),
        ("csvfiles/goods_dataset_0431_clean_all_3_0402_clean_addattr.csv",
         "商品条码", ["规格", "价签规格", "ItemSpecification"]),
        ("goods_info.csv", "商品条码", ["规格"]),
        ("goods_info_0324.csv", "商品条码", ["规格"]),
    ]
    merged = {}
    for rel, ean_col, spec_cols in sources:
        path = os.path.join(datasets_root, rel)
        if not os.path.isfile(path):
            print(f"[warn] spec source missing, skipped: {path}")
            continue
        lut = load_spec_lut(path, ean_col, spec_cols)
        merged.update(lut)
        print(f"spec source {rel}: {len(lut)} entries (merged total {len(merged)})")
    return merged


def update_csv(path, ean_col, name_col, spec_lut, dry_run=False):
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    n_named = n_had = n_filled = 0
    for r in rows:
        name = (r.get(name_col) or "").strip()
        if not name:
            continue
        n_named += 1
        if SPEC_RE.search(name):
            n_had += 1
            continue
        spec = spec_lut.get((r.get(ean_col) or "").strip(), "")
        if spec:
            r[name_col] = name + spec
            n_filled += 1

    print(
        f"{os.path.basename(path)}: named={n_named} 名称已有规格={n_had} "
        f"本次补充={n_filled} 仍无规格={n_named - n_had - n_filled}"
    )
    if dry_run or n_filled == 0:
        return
    if not os.path.isfile(path + ".bak"):
        shutil.copy2(path, path + ".bak")
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> written (backup at {path}.bak)")


def main():
    parser = argparse.ArgumentParser(description="为 SKU 名称补充规格")
    parser.add_argument("--datasets_root", required=True, help="konglingmei Datasets 目录")
    parser.add_argument(
        "--targets", nargs="+", required=True,
        help="目标 csv，格式 path:ean_col:name_col",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    spec_lut = build_merged_lut(args.datasets_root)
    for t in args.targets:
        path, ean_col, name_col = t.rsplit(":", 2)
        update_csv(path, ean_col, name_col, spec_lut, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
