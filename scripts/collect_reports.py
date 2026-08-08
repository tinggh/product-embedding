"""汇总多份 probe_eval 报告（*_report.json）为一张对比表。

用法：
    python collect_reports.py --report_glob "/path/ablation/*_report.json" --output comparison.md
"""

import argparse
import glob
import json
import os

DIM_ORDER = [
    "P1_color_variant",
    "P2_multi_view",
    "P3_part_whole",
    "P4_occlusion",
    "P5_similar_items",
    "P6_open_set",
]


def dim_metric(dim):
    """提取每维的关键数值：same/diff 型用 pass_rate，open_set 用 top1/acc。"""
    if "pass_rate" in dim:
        return dim["pass_rate"]
    for key in ("top1", "acc", "accuracy", "top1_acc"):
        if key in dim:
            return dim[key]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report_glob", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = {}
    for path in sorted(glob.glob(args.report_glob)):
        name = os.path.basename(path).replace("_report.json", "")
        with open(path, encoding="utf-8") as f:
            report = json.load(f)
        dims = report.get("dimensions", {})
        row = {}
        for dim_name in DIM_ORDER:
            if dim_name in dims:
                row[dim_name] = dim_metric(dims[dim_name])
        row["overall"] = report.get("overall", "?")
        rows[name] = row

    lines = ["# 消融实验对比", ""]
    header = "| experiment | " + " | ".join(DIM_ORDER) + " | overall |"
    lines.append(header)
    lines.append("|" + "---|" * (len(DIM_ORDER) + 2))
    for name, row in rows.items():
        cells = []
        for dim_name in DIM_ORDER:
            v = row.get(dim_name)
            cells.append(f"{v:.4f}" if isinstance(v, (int, float)) else "-")
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {row['overall']} |")
    lines.append("")
    lines.append("说明：P1/P5 为 diff 型（变体/相似品区分，pass_rate=相似度低于阈值的比例，越高越好）；")
    lines.append("P2/P3/P4 为 same 型（多视角/部分-整体/遮挡一致性，pass_rate=相似度高于阈值的比例，越高越好）；")
    lines.append("P6 为开集 top1 命中率。")

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
