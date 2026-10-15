#!/usr/bin/env python
"""Learned fusion of the single verdict score and the MMD-Agent-style checks on MMFakeBench.
Logistic regression fitted on the calibration half only, evaluated on the evaluation half.
No GPU: reads saved *_mmfakebench_scores.jsonl and *_mmfakebench_hierarchical.jsonl."""
import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from hierarchical_mmfakebench import CLASSES, macro_f1, paired_bootstrap, stratified_sample, to_class
from misinfodet.utils import Config, read_jsonl

FEATURE_SETS = {
    "single verdict only": ["single"],
    "three checks only": ["p_text_false", "p_visual_fake", "p_inconsistent"],
    "single + three checks": ["single", "p_text_false", "p_visual_fake", "p_inconsistent"],
    "single + three checks + 4-way": ["single", "p_text_false", "p_visual_fake", "p_inconsistent", "s0", "s1", "s2", "s3"],
}


def features(rows, keys):
    return np.array([[r[k] for k in keys] for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config)

    pred = Path(cfg.output_dir) / "predictions"
    single = {r["id"]: r["score"] for r in read_jsonl(pred / f"{cfg.run_name}_mmfakebench_scores.jsonl")}
    hier = {r["id"]: r for r in read_jsonl(pred / f"{cfg.run_name}_mmfakebench_hierarchical.jsonl")}

    records = read_jsonl(Path(cfg.data_dir) / "mmfakebench_test.jsonl")
    calib, evals = stratified_sample(records, args.n, args.seed)

    def rows(recs):
        out = []
        for r in recs:
            h = hier[r["id"]]
            out.append({"single": single[r["id"]], "p_text_false": h["p_text_false"], "p_visual_fake": h["p_visual_fake"],
                        "p_inconsistent": h["p_inconsistent"], **{f"s{i}": v for i, v in enumerate(h["p_standard"])}})
        return out

    cr, er = rows(calib), rows(evals)
    cy = np.array([0 if to_class(r["forgery_type"]) == "real" else 1 for r in calib])
    ey = np.array([0 if to_class(r["forgery_type"]) == "real" else 1 for r in evals])
    etype = [to_class(r["forgery_type"]) for r in evals]
    base = np.array([r["single"] for r in er])
    print(f"Run: {cfg.run_name}  calibration n={len(cr)}  evaluation n={len(er)}")

    print("\n=== Binary real vs fake, logistic fusion fitted on calibration half (evaluation-half AUROC) ===")
    print(f"{'features':32s} {'AUROC':>6s}   vs single verdict")
    for name, keys in FEATURE_SETS.items():
        clf = LogisticRegression(max_iter=2000).fit(features(cr, keys), cy)
        s = clf.predict_proba(features(er, keys))[:, 1]
        line = f"{name:32s} {roc_auc_score(ey, s):.3f}"
        if keys != ["single"]:
            d, ci, p = paired_bootstrap(ey, s, base)
            line += f"   {d:+.3f}  95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]  p={p:.3f}"
        print(line)
        if name == "single + three checks":
            best_s = s
            print("   weights: " + "  ".join(f"{k}={w:+.2f}" for k, w in zip(keys, clf.coef_[0])))

    print("\n=== Per type vs real (AUROC): single verdict -> fused (single + three checks) ===")
    for cls in ("textual", "visual", "mismatch"):
        m = [i for i, t in enumerate(etype) if t in (cls, "real")]
        y = ey[m]
        print(f"{cls:9s} {roc_auc_score(y, base[m]):.3f} -> {roc_auc_score(y, best_s[m]):.3f}")

    print("\n=== 4-class, multinomial logistic fusion fitted on calibration half ===")
    cg = [to_class(r["forgery_type"]) for r in calib]
    keys = FEATURE_SETS["single + three checks + 4-way"]
    clf = LogisticRegression(max_iter=5000).fit(features(cr, keys), cg)
    p4 = list(clf.predict(features(er, keys)))
    f, per = macro_f1(etype, p4, CLASSES)
    acc = np.mean([a == b for a, b in zip(etype, p4)])
    print(f"macro F1={f * 100:.1f}  accuracy={acc * 100:.1f}  per-class F1: " + "  ".join(f"{c}={x * 100:.1f}" for c, x in zip(CLASSES, per)))


if __name__ == "__main__":
    main()
