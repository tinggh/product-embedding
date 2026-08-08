"""将商品主数据（天虹 goods.csv 等）转换为 build_hierarchy.py 需要的 metadata CSV。

输出列：barcode,brand,category,name
- barcode: 商品条码（与数据集类目目录名一致）
- brand: 品牌（goods.csv 中常为空，build_hierarchy 会回退到从 name 解析）
- category: 三级分类名称（没有则退二级/一级）
- name: 商品名称（供 build_hierarchy 推导 product_line）

用法：
    python goods_csv_to_metadata.py --goods_csv /path/goods.csv --output metadata.csv
"""

import argparse
import csv
import sys

csv.field_size_limit(sys.maxsize)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--goods_csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    n_in = n_out = 0
    with open(args.goods_csv, "r", encoding="utf-8-sig", errors="replace") as f, open(
        args.output, "w", encoding="utf-8", newline=""
    ) as out:
        reader = csv.DictReader(f)
        writer = csv.writer(out)
        writer.writerow(["barcode", "brand", "category", "name"])
        for row in reader:
            n_in += 1
            barcode = (row.get("商品条码") or row.get("商品编码") or "").strip().strip("\t").strip()
            name = (row.get("商品名称") or "").strip().strip("\t").strip()
            if not barcode or not name:
                continue
            brand = (row.get("品牌") or "").strip().strip("\t").strip()
            category = (
                (row.get("三级分类名称") or "").strip()
                or (row.get("二级分类名称") or "").strip()
                or (row.get("一级分类名称") or "").strip()
            )
            writer.writerow([barcode, brand, category, name])
            n_out += 1
    print(f"read {n_in} rows, wrote {n_out} rows -> {args.output}")


if __name__ == "__main__":
    main()
