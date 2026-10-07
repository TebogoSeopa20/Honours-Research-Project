#!/usr/bin/env python
"""Paired bootstrap comparison of two scored runs on the same items.

COSMOS:      --cosmos A_cosmos_scores.jsonl B_cosmos_scores.jsonl
MMFakeBench: --mmfb   A_mmfakebench_scores.jsonl B_mmfakebench_scores.jsonl --n 2000

Thresholds are calibrated once on the val split (COSMOS) or calibration half
(MMFakeBench) and held fixed; only the evaluation items are resampled."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from calibrate_mmfakebench import stratified_sample
from calibrate_threshold import find_best_threshold
from misinfodet.utils import read_jsonl


def load_scores(path):
    return {r["id"]: r["score"] for r in read_jsonl(path)}


def cosmos_items(path_a, path_b):
    rows = list(read_jsonl(path_a))
    a, b = load_scores(path_a), load_scores(path_b)
    calib = [r for r in rows if r["split"] == "val"]
    evals = [r for r in rows if r["split"] == "test"]
    return calib, evals, a, b


def mmfb_items(path_a, path_b, n, seed, data_dir):
    records = read_jsonl(Path(data_dir) / "mmfakebench_test.jsonl")
    calib, evals = stratified_sample(records, n, seed)
    return calib, evals, load_scores(path_a), load_scores(path_b)


def acc(scores, labels, t):
    return float(np.mean((scores >= t).astype(int) == labels))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--cosmos", nargs=2, metavar=("A", "B"))
    g.add_argument("--mmfb", nargs=2, metavar=("A", "B"))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--iters", type=int, default=5000)
    args = ap.parse_args()

    if args.cosmos:
        path_a, path_b = args.cosmos
        calib, evals, a, b = cosmos_items(path_a, path_b)
    else:
        path_a, path_b = args.mmfb
        calib, evals, a, b = mmfb_items(path_a, path_b, args.n, args.seed, args.data_dir)

    t_a, _ = find_best_threshold([(a[r["id"]], r["label"]) for r in calib])
    t_b, _ = find_best_threshold([(b[r["id"]], r["label"]) for r in calib])

    y = np.array([r["label"] for r in evals])
    sa = np.array([a[r["id"]] for r in evals])
    sb = np.array([b[r["id"]] for r in evals])

    rng = np.random.default_rng(args.seed)
    d_auc, d_acc = [], []
    for _ in range(args.iters):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        d_auc.append(roc_auc_score(y[idx], sa[idx]) - roc_auc_score(y[idx], sb[idx]))
        d_acc.append(acc(sa[idx], y[idx], t_a) - acc(sb[idx], y[idx], t_b))

    print(f"A = {path_a}  (threshold {t_a:.2f})")
    print(f"B = {path_b}  (threshold {t_b:.2f})")
    print(f"n evaluation items = {len(y)}, bootstrap iterations = {len(d_auc)}\n")
    for name, point, diffs in (
        ("AUROC", roc_auc_score(y, sa) - roc_auc_score(y, sb), np.array(d_auc)),
        ("Accuracy", acc(sa, y, t_a) - acc(sb, y, t_b), np.array(d_acc)),
    ):
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        p = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
        verdict = "significant" if lo > 0 or hi < 0 else "not significant"
        print(f"{name:9s} A-B = {point:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   p~{min(p, 1.0):.3f}   ({verdict})")


if __name__ == "__main__":
    main()
