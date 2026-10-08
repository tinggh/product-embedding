#!/usr/bin/env python3
"""两阶段商品检索：模型 A 召回 Top-K，模型 B 以加权余弦相似度重排。"""
import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np

from eval_retrieval import collect, stable_calibration_mask, threshold_curve


def parse_csv(value, cast=float):
    return [cast(x) for x in value.split(",") if x.strip()]


def extract_matrix(extractor, paths, batch_size):
    feats = extractor.extract(paths, batch_size=batch_size)
    matrix = np.stack([feats[p] for p in paths]).astype(np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True).clip(min=1e-12)


def main():
    ap = argparse.ArgumentParser(description="two-stage retrieval evaluation")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--recall-ckpt", required=True)
    ap.add_argument("--recall-pooling", default="cls+gem+salad")
    ap.add_argument("--rerank-ckpt", required=True)
    ap.add_argument("--rerank-pooling", default="cls+gem")
    ap.add_argument("--gallery", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--topk", default="5,10,20,50")
    ap.add_argument("--alphas", default="0,0.25,0.5,0.65,0.8,1")
    ap.add_argument("--threshold-grid", default="0.50:0.95:0.01")
    ap.add_argument("--target-precision", type=float, default=0.99)
    ap.add_argument("--calibration-fraction", type=float, default=0.3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.repo))
    os.chdir(tempfile.mkdtemp(prefix="eval_two_stage_"))
    os.makedirs("logs", exist_ok=True)
    from app.probe_eval import ProbeFeatureExtractor

    g_paths, g_skus = collect(args.gallery)
    t_paths, t_skus = collect(args.test)
    matrices = []
    for ckpt, pooling in ((args.recall_ckpt, args.recall_pooling),
                          (args.rerank_ckpt, args.rerank_pooling)):
        started = time.time()
        extractor = ProbeFeatureExtractor(ckpt, pooling=pooling, device=args.device)
        matrices.append((extract_matrix(extractor, g_paths, args.batch_size),
                         extract_matrix(extractor, t_paths, args.batch_size)))
        print(f"extracted {pooling} in {time.time() - started:.1f}s")

    (g_recall, t_recall), (g_rerank, t_rerank) = matrices
    recall_sim = t_recall @ g_recall.T
    rerank_sim = t_rerank @ g_rerank.T
    calibration_mask = stable_calibration_mask(t_paths, args.calibration_fraction)
    evaluation_mask = ~calibration_mask
    start, stop, step = (float(x) for x in args.threshold_grid.split(":"))
    grid = np.arange(start, stop + step / 2, step)
    results = []

    for k in parse_csv(args.topk, int):
        k = min(k, len(g_paths))
        candidates = np.argpartition(recall_sim, -k, axis=1)[:, -k:]
        a = np.take_along_axis(recall_sim, candidates, axis=1)
        b = np.take_along_axis(rerank_sim, candidates, axis=1)
        for alpha in parse_csv(args.alphas):
            fused = alpha * a + (1.0 - alpha) * b
            local_best = fused.argmax(axis=1)
            best_idx = candidates[np.arange(len(t_paths)), local_best]
            score = fused[np.arange(len(t_paths)), local_best]
            correct = g_skus[best_idx] == t_skus
            curve = threshold_curve(score[calibration_mask], correct[calibration_mask], grid)
            eligible = [r for r in curve if r["precision"] >= args.target_precision]
            chosen = max(eligible, key=lambda r: (r["coverage"], -r["threshold"])) if eligible else max(
                curve, key=lambda r: (r["precision"], r["coverage"])
            )
            heldout = threshold_curve(score[evaluation_mask], correct[evaluation_mask],
                                      [chosen["threshold"]])[0]
            results.append({
                "topk": k,
                "alpha_recall": alpha,
                "top1_acc_no_threshold": round(float(correct[evaluation_mask].mean()), 4),
                "selected_threshold": chosen["threshold"],
                "calibration_metrics": chosen,
                "heldout_metrics": heldout,
            })

    results.sort(key=lambda r: (
        r["heldout_metrics"]["accuracy"], r["heldout_metrics"]["precision"],
        r["top1_acc_no_threshold"]), reverse=True)
    report = {
        "recall": {"ckpt": args.recall_ckpt, "pooling": args.recall_pooling},
        "rerank": {"ckpt": args.rerank_ckpt, "pooling": args.rerank_pooling},
        "gallery": args.gallery, "test": args.test,
        "gallery_imgs": len(g_paths), "test_imgs": len(t_paths),
        "calibration_fraction": args.calibration_fraction,
        "target_precision": args.target_precision,
        "best": results[0], "results": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output + ".json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(args.output + ".md", "w", encoding="utf-8") as f:
        f.write("# Two-stage retrieval evaluation\n\n")
        f.write("| K | alpha(recall) | top1 | threshold | coverage | precision | accuracy |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in results:
            h = r["heldout_metrics"]
            f.write(f"| {r['topk']} | {r['alpha_recall']:.2f} | {r['top1_acc_no_threshold']:.4f} | "
                    f"{r['selected_threshold']:.4f} | {h['coverage']:.4f} | "
                    f"{h['precision']:.4f} | {h['accuracy']:.4f} |\n")
    print(json.dumps(report["best"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
