#!/usr/bin/env python
"""MMD-Agent-style hierarchical detection on MMFakeBench (Liu et al., ICLR 2025).
Scores three sub-checks (textual veracity, visual veracity, cross-modal consistency)
plus a standard single 4-way prompt, using next-token probabilities. Same stratified
sample (seed 42, n 2000) as calibrate_mmfakebench.py, so items pair with earlier runs."""
import argparse
import json
import random
import time
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
from PIL import Image

from misinfodet.models.model_factory import load_llava
from misinfodet.utils import Config, read_jsonl

CLASSES = ["real", "textual", "visual", "mismatch"]

Q_TEXT = ('Here is a news caption: "{caption}"\n'
          "Based on your knowledge, does this caption contain false, fabricated or misleading information? "
          "Answer Yes or No.")
Q_VISUAL = ("Look carefully at this image. Has it been digitally edited, manipulated or AI-generated, "
            "or does it show something physically implausible? Answer Yes or No.")
Q_CONSIST = ('Here is a news caption: "{caption}"\n'
             "Does the image match the caption, showing the same people, place and event? Answer Yes or No.")
Q_STANDARD = ('Here is a news image and its caption: "{caption}"\n'
              "Which option best describes this image-caption pair?\n"
              "A. Real: the caption is true and matches the image.\n"
              "B. The caption contains false information.\n"
              "C. The image has been edited or AI-generated.\n"
              "D. The caption does not match the image.\n"
              "Answer with the letter only.")


def to_class(ftype):
    f = ftype.lower()
    if f in ("original", "real", "pristine"):
        return "real"
    if "text" in f:
        return "textual"
    if "visual" in f or "image" in f:
        return "visual"
    if "mismatch" in f or "consist" in f or "cross" in f:
        return "mismatch"
    raise ValueError(f"unmapped forgery_type: {ftype}")


def stratified_sample(records, n, seed):
    by_type = defaultdict(list)
    for r in records:
        by_type[r["forgery_type"]].append(r)
    rng = random.Random(seed)
    calib, evals = [], []
    for ftype, items in sorted(by_type.items()):
        k = round(n * len(items) / len(records))
        picked = rng.sample(items, k)
        half = k // 2
        calib += picked[:half]
        evals += picked[half:]
    rng.shuffle(calib)
    rng.shuffle(evals)
    return calib, evals


def token_ids(tok, words):
    out = []
    for w in words:
        ids = {tok.convert_tokens_to_ids(t) for t in (w, "▁" + w)}
        out.append([i for i in ids if i is not None and i != tok.unk_token_id])
    return out


def option_probs(model, processor, image, question, groups):
    import torch

    prompt = f"USER: <image>\n{question} ASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1].float()
    p = torch.softmax(logits, dim=-1)
    mass = torch.stack([p[g].sum() for g in groups])
    return (mass / mass.sum()).tolist()


def score_all(model, processor, records, out_path):
    import torch

    tok = processor.tokenizer
    yn = token_ids(tok, ["Yes", "No"])
    abcd = token_ids(tok, ["A", "B", "C", "D"])
    done = {r["id"]: r for r in read_jsonl(out_path)} if out_path.exists() else {}
    todo = [r for r in records if r["id"] not in done]
    print(f"{len(done)} already scored, {len(todo)} to go")
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as f:
        for i, rec in enumerate(todo):
            img = Image.open(rec["image_path"]).convert("RGB")
            row = {
                "id": rec["id"],
                "p_text_false": option_probs(model, processor, img, Q_TEXT.format(caption=rec["caption"]), yn)[0],
                "p_visual_fake": option_probs(model, processor, img, Q_VISUAL, yn)[0],
                "p_inconsistent": option_probs(model, processor, img, Q_CONSIST.format(caption=rec["caption"]), yn)[1],
                "p_standard": option_probs(model, processor, img, Q_STANDARD.format(caption=rec["caption"]), abcd),
            }
            done[rec["id"]] = row
            f.write(json.dumps(row) + "\n")
            f.flush()
            if (i + 1) % 20 == 0:
                torch.cuda.empty_cache()
                rate = (time.time() - t0) / (i + 1)
                print(f"  {i + 1}/{len(todo)}  ({rate:.1f}s/item, ~{rate * (len(todo) - i - 1) / 60:.0f} min left)")
    return done


def macro_f1(gold, pred, labels):
    f1s = []
    for c in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == c and p == c)
        fp = sum(1 for g, p in zip(gold, pred) if g != c and p == c)
        fn = sum(1 for g, p in zip(gold, pred) if g == c and p != c)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(f1s)), f1s


def cascade(rows, t1, t2, t3):
    out = []
    for r in rows:
        if r["p_text_false"] >= t1:
            out.append("textual")
        elif r["p_visual_fake"] >= t2:
            out.append("visual")
        elif r["p_inconsistent"] >= t3:
            out.append("mismatch")
        else:
            out.append("real")
    return out


def tune_cascade(rows, gold):
    a = np.array([r["p_text_false"] for r in rows])
    b = np.array([r["p_visual_fake"] for r in rows])
    c = np.array([r["p_inconsistent"] for r in rows])
    g = np.array([CLASSES.index(x) for x in gold])
    grid = [i / 20 for i in range(1, 20)]
    best = (-1, None)
    for t1, t2, t3 in product(grid, repeat=3):
        pred = np.where(a >= t1, 1, np.where(b >= t2, 2, np.where(c >= t3, 3, 0)))
        f1s = []
        for k in range(4):
            tp = np.sum((g == k) & (pred == k))
            fp = np.sum((g != k) & (pred == k))
            fn = np.sum((g == k) & (pred != k))
            f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
        f = float(np.mean(f1s))
        if f > best[0]:
            best = (f, (t1, t2, t3))
    return best


def auroc(y, s):
    from sklearn.metrics import roc_auc_score

    return roc_auc_score(y, s)


def paired_bootstrap(y, a, b, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    y, a, b = map(np.asarray, (y, a, b))
    base = auroc(y, a) - auroc(y, b)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        diffs.append(auroc(y[idx], a[idx]) - auroc(y[idx], b[idx]))
    diffs = np.array(diffs)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return base, np.percentile(diffs, [2.5, 97.5]), min(p, 1.0)


def analyse(calib, evals, scores, baseline_path):
    cr = [scores[r["id"]] for r in calib]
    er = [scores[r["id"]] for r in evals]
    cg = [to_class(r["forgery_type"]) for r in calib]
    eg = [to_class(r["forgery_type"]) for r in evals]
    print(f"\nEvaluation half: n={len(evals)}  " + "  ".join(f"{c}={eg.count(c)}" for c in CLASSES))

    std_pred = [CLASSES[int(np.argmax(r["p_standard"]))] for r in er]
    f, per = macro_f1(eg, std_pred, CLASSES)
    acc = np.mean([g == p for g, p in zip(eg, std_pred)])
    print("\n=== 4-class: standard prompting (zero-shot, argmax) ===")
    print(f"macro F1={f * 100:.1f}  accuracy={acc * 100:.1f}  per-class F1: " + "  ".join(f"{c}={x * 100:.1f}" for c, x in zip(CLASSES, per)))

    cf, (t1, t2, t3) = tune_cascade(cr, cg)
    h_pred = cascade(er, t1, t2, t3)
    f, per = macro_f1(eg, h_pred, CLASSES)
    acc = np.mean([g == p for g, p in zip(eg, h_pred)])
    print(f"\n=== 4-class: hierarchical (MMD-Agent-style), thresholds from calibration half: text {t1:.2f}, visual {t2:.2f}, consistency {t3:.2f} ===")
    print(f"macro F1={f * 100:.1f}  accuracy={acc * 100:.1f}  per-class F1: " + "  ".join(f"{c}={x * 100:.1f}" for c, x in zip(CLASSES, per)))

    y = [0 if g == "real" else 1 for g in eg]
    s_h = [max(r["p_text_false"], r["p_visual_fake"], r["p_inconsistent"]) for r in er]
    s_std = [1 - r["p_standard"][0] for r in er]
    print("\n=== Binary real vs fake AUROC (evaluation half) ===")
    print(f"standard 4-way prompt (1 - P(real)): {auroc(y, s_std):.3f}")
    print(f"hierarchical (max of three checks):  {auroc(y, s_h):.3f}")

    print("\n=== Each check on its own target type vs real (AUROC) ===")
    for key, cls in (("p_text_false", "textual"), ("p_visual_fake", "visual"), ("p_inconsistent", "mismatch")):
        sub = [(1 if g == cls else 0, r[key]) for g, r in zip(eg, er) if g in (cls, "real")]
        print(f"{key:15s} {cls:9s} vs real: {auroc([a for a, _ in sub], [b for _, b in sub]):.3f}")

    if baseline_path and Path(baseline_path).exists():
        base = {r["id"]: r["score"] for r in read_jsonl(Path(baseline_path))}
        keep = [i for i, r in enumerate(evals) if r["id"] in base]
        yb = [y[i] for i in keep]
        sb = [base[evals[i]["id"]] for i in keep]
        print(f"\n=== Paired bootstrap vs single verdict prompt ({Path(baseline_path).name}, n={len(keep)}) ===")
        print(f"single verdict AUROC: {auroc(yb, sb):.3f}")
        for name, s in (("hierarchical", s_h), ("standard 4-way", s_std)):
            d, ci, p = paired_bootstrap(yb, [s[i] for i in keep], sb)
            print(f"{name:15s} - single verdict: {d:+.3f}  95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]  p={p:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--baseline-scores", default=None, help="earlier *_mmfakebench_scores.jsonl for a paired comparison")
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--smoke", type=int, default=0, help="score only this many items, print them, and stop")
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config)

    records = read_jsonl(Path(cfg.data_dir) / "mmfakebench_test.jsonl")
    print("forgery types:", sorted({r["forgery_type"] for r in records}))
    for t in {r["forgery_type"] for r in records}:
        to_class(t)
    calib, evals = stratified_sample(records, args.n, args.seed)
    print(f"Sample: {len(calib)} calibration + {len(evals)} evaluation")

    out_path = Path(cfg.output_dir) / "predictions" / f"{cfg.run_name}_mmfakebench_hierarchical.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.analyse_only:
        scores = {r["id"]: r for r in read_jsonl(out_path)}
    else:
        print("Loading model...")
        model, processor = load_llava(cfg, trainable=False)
        model.eval()
        if args.smoke:
            smoke_path = out_path.with_name(out_path.stem + "_smoke.jsonl")
            smoke_path.unlink(missing_ok=True)
            for rec in (calib + evals)[: args.smoke]:
                row = score_all(model, processor, [rec], smoke_path)[rec["id"]]
                print(rec["forgery_type"], {k: v for k, v in row.items() if k != "id"})
            return
        scores = score_all(model, processor, calib + evals, out_path)
    analyse(calib, evals, scores, args.baseline_scores)


if __name__ == "__main__":
    main()
