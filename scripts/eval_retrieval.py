#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""商品检索 top1 精度评测：对给定特征库(gallery)与测试集(test)做 top1 检索，
正确 = (top1 相似度 > 阈值) AND (top1 SKU == 真 SKU)。

与 app/probe_eval 共用 ProbeFeatureExtractor，保证特征提取与训练/探针一致。

用法（需先 cd dinov3-main 并设 PYTHONPATH）：
  python -m scripts.eval_retrieval --repo /path/dinov3-main \
      --ckpt /path/best.pth --pooling cls+gem \
      --gallery /path/featsImgs --test /path/testImgs \
      --thresholds 0.7,0.85 [--output report]
"""
import argparse
import glob
import os
import sys
import time

import numpy as np


def collect(root):
    """收集 root 下每个 SKU 子目录的图片，返回 (paths, skus)。"""
    paths, skus = [], []
    for sku in sorted(os.listdir(root)):
        d = os.path.join(root, sku)
        if not os.path.isdir(d):
            continue
        for img in sorted(glob.glob(os.path.join(d, "*.jpg")) +
                          glob.glob(os.path.join(d, "*.png"))):
            paths.append(img)
            skus.append(sku)
    return paths, np.array(skus)


def main():
    ap = argparse.ArgumentParser(description="top1 检索精度评测")
    ap.add_argument("--repo", default=os.environ.get("REPO", "."),
                    help="dinov3-main 仓库根（用于 import app/dinov3）")
    ap.add_argument("--ckpt", required=True, help="待评测 ckpt")
    ap.add_argument("--pooling", default="cls+gem",
                    choices=["cls", "gem", "cls+gem", "cls+g2m", "salad", "cls+gem+salad"])
    ap.add_argument("--gallery", required=True, help="特征库目录（每 SKU 一个子目录）")
    ap.add_argument("--test", required=True, help="测试集目录（每 SKU 一个子目录）")
    ap.add_argument("--thresholds", default="0.7,0.85",
                    help="逗号分隔的相似度阈值列表")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", default="cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") is not None else "cuda:0")
    ap.add_argument("--output", default="", help="可选，输出报告 basename（写 .md/.json）")
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.repo))
    # 不 chdir 到 repo：app.log_module 会写 "logs/app.log"（相对 CWD），
    # repo/logs/app.log 可能被历史 sudo 运行改为 root 属主导致权限错误。
    # 改用临时可写目录，让日志落在可写位置。
    import tempfile
    _tmpd = tempfile.mkdtemp(prefix="eval_retrieval_")
    os.chdir(_tmpd)
    os.makedirs("logs", exist_ok=True)
    from app.probe_eval import ProbeFeatureExtractor  # noqa: E402

    thrs = [float(t) for t in args.thresholds.split(",") if t.strip()]

    ext = ProbeFeatureExtractor(args.ckpt, pooling=args.pooling, device=args.device)

    # ---- gallery ----
    g_paths, g_skus = collect(args.gallery)
    print(f"[gallery] {len(g_paths)} imgs, {len(set(g_skus))} skus")
    t0 = time.time()
    g_feats = ext.extract(g_paths, batch_size=args.batch_size)
    g_mat = np.stack([g_feats[p] for p in g_paths]).astype(np.float32)
    g_mat /= np.linalg.norm(g_mat, axis=1, keepdims=True).clip(min=1e-12)
    print(f"[gallery] extracted {g_mat.shape} in {time.time()-t0:.1f}s")

    # ---- test ----
    t_paths, t_skus = collect(args.test)
    print(f"[test] {len(t_paths)} imgs, {len(set(t_skus))} skus")
    t0 = time.time()
    t_feats = ext.extract(t_paths, batch_size=args.batch_size)
    t_mat = np.stack([t_feats[p] for p in t_paths]).astype(np.float32)
    t_mat /= np.linalg.norm(t_mat, axis=1, keepdims=True).clip(min=1e-12)
    print(f"[test] extracted {t_mat.shape} in {time.time()-t0:.1f}s")

    # ---- top1 检索 ----
    sims = t_mat @ g_mat.T
    top1_idx = sims.argmax(axis=1)
    top1_sim = sims[np.arange(len(t_mat)), top1_idx]
    top1_sku = g_skus[top1_idx]
    correct_sku = top1_sku == t_skus
    n = len(t_skus)

    # ---- 缺失 SKU ----
    g_sku_set = set(g_skus.tolist())
    missing = [s for s in sorted(set(t_skus.tolist())) if s not in g_sku_set]

    print("\n========== 结果 ==========")
    print(f"gallery: {len(g_paths)} imgs / {len(g_sku_set)} skus | "
          f"test: {len(t_paths)} imgs / {len(set(t_skus))} skus | "
          f"test 缺失 SKU: {len(missing)}")
    print(f"top1 SKU 正确率（无阈值）: {correct_sku.mean():.4f} ({int(correct_sku.sum())}/{n})")
    rows = []
    for thr in thrs:
        correct = correct_sku & (top1_sim > thr)
        acc = correct.mean()
        above = int((top1_sim > thr).sum())
        reject = n - above
        prec = correct.sum() / above if above else 0.0
        print(f"thr={thr}: 正确率={acc:.4f} ({int(correct.sum())}/{n})  "
              f"超阈值={above}  拒识={reject}  超阈值正确率(prec)={prec:.4f}")
        rows.append({"threshold": thr, "accuracy": round(float(acc), 4),
                     "correct": int(correct.sum()), "total": n,
                     "above_threshold": above, "rejected": reject,
                     "precision": round(float(prec), 4)})

    if missing:
        miss_mask = np.array([s in set(missing) for s in t_skus])
        print(f"\n[gallery 缺失 test SKU] {len(missing)}: {missing}")
        if miss_mask.any():
            print(f"  缺失 SKU test 图={int(miss_mask.sum())}, "
                  f"top1_sim 均值={top1_sim[miss_mask].mean():.4f}, "
                  f"max={top1_sim[miss_mask].max():.4f}")

    # ---- 错误样例 ----
    print(f"\n[thr={thrs[-1]} 错误样例（top1 错或 sim<=阈值）]")
    thr_last = thrs[-1]
    wrong = (~correct_sku) | (top1_sim <= thr_last)
    for i in np.where(wrong)[0][:12]:
        print(f"  true={t_skus[i][:38]}  pred={top1_sku[i][:38]}  sim={top1_sim[i]:.4f}")

    # ---- 可选写报告 ----
    if args.output:
        import json
        report = {
            "ckpt": args.ckpt, "pooling": args.pooling,
            "gallery": args.gallery, "test": args.test,
            "gallery_imgs": len(g_paths), "gallery_skus": len(g_sku_set),
            "test_imgs": len(t_paths), "test_skus": len(set(t_skus)),
            "missing_test_skus": missing,
            "top1_acc_no_threshold": round(float(correct_sku.mean()), 4),
            "thresholds": rows,
        }
        with open(args.output + ".json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        with open(args.output + ".md", "w", encoding="utf-8") as f:
            f.write(f"# 检索精度评测\n\n- ckpt: `{args.ckpt}`\n- pooling: `{args.pooling}`\n"
                    f"- gallery: `{args.gallery}` ({len(g_paths)} imgs / {len(g_sku_set)} skus)\n"
                    f"- test: `{args.test}` ({len(t_paths)} imgs / {len(set(t_skus))} skus)\n"
                    f"- test 缺失 SKU: {len(missing)}\n\n")
            f.write(f"- top1 正确率（无阈值）: **{correct_sku.mean():.4f}** ({int(correct_sku.sum())}/{n})\n\n")
            f.write("| 阈值 | 正确率 | 正确/总数 | 超阈值 | 拒识 | 超阈值正确率 |\n|---|---|---|---|---|---|\n")
            for r in rows:
                f.write(f"| {r['threshold']} | {r['accuracy']:.4f} | {r['correct']}/{r['total']} | "
                        f"{r['above_threshold']} | {r['rejected']} | {r['precision']:.4f} |\n")
        print(f"\nreport -> {args.output}.md / {args.output}.json")


if __name__ == "__main__":
    main()
