#!/usr/bin/env python
"""CPU-only analyses on saved COSMOS score files (outputs/predictions/*_cosmos_scores.jsonl).

  fusion       SNIFFER-style combination of the internal check (verdict score without retrieval)
               and the external check (score with retrieved evidence). Combiners are fitted on val only.
  calibration  Expected calibration error, Brier score and reliability diagram of the raw verdict
               probabilities, before and after Platt scaling fitted on val (Thibault et al. 2024)."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

PRED = Path("outputs/predictions")


def load(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    return {r["id"]: r for r in rows}


def split_arrays(d, split):
    ids = sorted((k for k, r in d.items() if r.get("split", k.split("_")[0]) == split), key=lambda k: int(k.split("_")[-1]))
    return ids, np.array([d[k]["score"] for k in ids], float), np.array([d[k]["label"] for k in ids], int)


def macro_f1(y, p):
    f = []
    for c in (0, 1):
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        f.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(f))


def threshold(sv, yv):
    cands = np.unique(np.quantile(sv, np.linspace(0, 1, 401)))
    return max(cands, key=lambda t: macro_f1(yv, (sv >= t).astype(int)))


def bootstrap(y, a, b, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        d.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    d = np.array(d)
    return roc_auc_score(y, a) - roc_auc_score(y, b), np.percentile(d, [2.5, 97.5]), min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fusion(args):
    internal = load(PRED / f"{args.run}_cosmos_scores.jsonl")
    print(f"internal check: {args.run}_cosmos_scores.jsonl")
    for ext in args.external:
        path = PRED / f"{args.run}_{ext}_cosmos_scores.jsonl"
        if not path.exists():
            print(f"\n(skipping {path.name}: not found)")
            continue
        external = load(path)
        common = set(internal) & set(external)
        I = {k: internal[k] for k in common}
        E = {k: external[k] for k in common}
        _, iv, yv = split_arrays(I, "val")
        _, it, yt = split_arrays(I, "test")
        _, ev, _ = split_arrays(E, "val")
        _, et, _ = split_arrays(E, "test")
        print(f"\n=== external check: {path.name}  (val n={len(yv)}, test n={len(yt)}) ===")
        if len(yv) == 0:
            print("no val items in the external file; combiners cannot be fitted without leaking test data")
            continue
        Xv = np.stack([logit(iv), logit(ev)], 1)
        Xt = np.stack([logit(it), logit(et)], 1)
        clf = LogisticRegression().fit(Xv, yv)
        variants = {
            "internal only": (iv, it),
            "external only": (ev, et),
            "mean of the two": ((iv + ev) / 2, (it + et) / 2),
            "max of the two": (np.maximum(iv, ev), np.maximum(it, et)),
            "logistic fusion (fit on val)": (clf.predict_proba(Xv)[:, 1], clf.predict_proba(Xt)[:, 1]),
        }
        print(f"{'method':30s} {'AUROC':>6s} {'acc':>6s} {'mF1':>6s}   vs internal only")
        for name, (sv, st) in variants.items():
            t = threshold(sv, yv)
            p = (st >= t).astype(int)
            line = f"{name:30s} {roc_auc_score(yt, st):.3f} {np.mean(p == yt):.3f} {macro_f1(yt, p):.3f}"
            if name != "internal only":
                d, ci, pv = bootstrap(yt, st, it)
                line += f"   {d:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] p={pv:.3f}"
            print(line)
        print(f"   logistic weights: internal={clf.coef_[0][0]:+.2f}  external={clf.coef_[0][1]:+.2f}")


def ece(p, y, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    tot, table = 0.0, []
    for b in range(bins):
        m = idx == b
        if m.any():
            conf, acc = p[m].mean(), y[m].mean()
            tot += m.mean() * abs(acc - conf)
            table.append((edges[b], edges[b + 1], int(m.sum()), conf, acc))
    return tot, table


def calibration(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect calibration")
    print(f"{'run':28s} {'AUROC':>6s} {'ECE raw':>8s} {'Brier raw':>9s} {'mean p':>7s} {'OOC rate':>8s} {'ECE Platt':>9s} {'Brier Platt':>11s}")
    for run in args.runs:
        path = PRED / f"{run}_cosmos_scores.jsonl"
        if not path.exists():
            print(f"{run:28s} (not found)")
            continue
        d = load(path)
        _, sv, yv = split_arrays(d, "val")
        _, st, yt = split_arrays(d, "test")
        e_raw, table = ece(st, yt)
        platt = LogisticRegression().fit(logit(sv)[:, None], yv)
        pt = platt.predict_proba(logit(st)[:, None])[:, 1]
        e_cal, _ = ece(pt, yt)
        print(f"{run:28s} {roc_auc_score(yt, st):.3f} {e_raw:8.3f} {np.mean((st - yt) ** 2):9.3f} {st.mean():7.3f} {yt.mean():8.3f} "
              f"{e_cal:9.3f} {np.mean((pt - yt) ** 2):11.3f}")
        if args.table and run == args.runs[0]:
            print("   reliability (raw, test):  bin  n  mean predicted P(OOC)  observed OOC rate")
            for lo, hi, n, c, a in table:
                print(f"   [{lo:.2f},{hi:.2f})  {n:4d}  {c:.3f}  {a:.3f}")
        ax.plot([r[3] for r in table], [r[4] for r in table], "o-", label=f"{run} (ECE {e_raw:.3f})")
    ax.set_xlabel("predicted P(out-of-context)")
    ax.set_ylabel("observed out-of-context rate")
    ax.set_title("Reliability of raw verdict probabilities (COSMOS test)")
    ax.legend(fontsize=8)
    out = PRED / "reliability_cosmos.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nreliability diagram saved to {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fusion")
    p.add_argument("--run", default="diag_stage1_only")
    p.add_argument("--external", nargs="+", default=["retrieval_evonly", "retrieval", "retrieval_template_only"])
    p = sub.add_parser("calibration")
    p.add_argument("--runs", nargs="+", default=["diag_stage1_only", "base_llava_eval", "stage2_eval_lmonly"])
    p.add_argument("--table", action="store_true")
    args = ap.parse_args()
    {"fusion": fusion, "calibration": calibration}[args.cmd](args)


if __name__ == "__main__":
    main()
