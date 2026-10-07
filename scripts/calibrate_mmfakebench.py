#!/usr/bin/env python
"""MMFakeBench generalisation eval via calibrated verdict scoring (same method as
calibrate_threshold.py). Stratified sample, split into a calibration half and an
evaluation half. Scores are appended to disk as they are produced, so a crashed
or timed-out run resumes where it stopped."""
import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image

from calibrate_threshold import find_best_threshold, report
from misinfodet.models.model_factory import get_verdict_score, load_llava
from misinfodet.pipeline.prompts import VERDICT_PROMPT
from misinfodet.utils import Config, read_jsonl


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


def score_all(model, processor, records, cfg, out_path):
    import torch

    done = {}
    if out_path.exists():
        for r in read_jsonl(out_path):
            done[r["id"]] = r["score"]
    todo = [r for r in records if r["id"] not in done]
    print(f"{len(done)} already scored, {len(todo)} to go")
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as f:
        for i, rec in enumerate(todo):
            image = Image.open(rec["image_path"]).convert("RGB")
            prompt = VERDICT_PROMPT.format(caption=rec["caption"])
            score = get_verdict_score(model, processor, image, prompt, cfg)
            done[rec["id"]] = score
            f.write(json.dumps({"id": rec["id"], "score": score}) + "\n")
            f.flush()
            if (i + 1) % 20 == 0:
                torch.cuda.empty_cache()
                rate = (time.time() - t0) / (i + 1)
                print(f"  {i + 1}/{len(todo)} done  ({rate:.1f}s/item, ~{rate * (len(todo) - i - 1) / 60:.0f} min left)")
    return done


def auroc(scored):
    from sklearn.metrics import roc_auc_score

    golds = [g for _, g in scored]
    if len(set(golds)) < 2:
        return float("nan")
    return roc_auc_score(golds, [s for s, _ in scored])


def per_type(records, scores, threshold):
    print(f"\n--- Per forgery type (threshold={threshold:.2f}) ---")
    originals = [(scores[r["id"]], 0) for r in records if r["forgery_type"] == "original"]
    for ftype in sorted({r["forgery_type"] for r in records}):
        items = [r for r in records if r["forgery_type"] == ftype]
        flagged = sum(1 for r in items if scores[r["id"]] >= threshold)
        if ftype == "original":
            print(f"{ftype:32s} n={len(items):4d}  correctly passed as real: {1 - flagged / len(items):.3f}")
        else:
            pair = [(scores[r["id"]], 1) for r in items] + originals
            print(f"{ftype:32s} n={len(items):4d}  detected as fake: {flagged / len(items):.3f}  AUROC vs original: {auroc(pair):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cosmos-threshold", type=float, default=0.85)
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config)

    records = read_jsonl(Path(cfg.data_dir) / "mmfakebench_test.jsonl")
    calib, evals = stratified_sample(records, args.n, args.seed)
    print(f"Sample: {len(calib)} calibration + {len(evals)} evaluation")

    out_path = Path(cfg.output_dir) / "predictions" / f"{cfg.run_name}_mmfakebench_scores.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model, processor = load_llava(cfg, trainable=False)
    scores = score_all(model, processor, calib + evals, cfg, out_path)

    calib_scored = [(scores[r["id"]], r["label"]) for r in calib]
    eval_scored = [(scores[r["id"]], r["label"]) for r in evals]

    print(f"\nAUROC (threshold-free) - calibration half: {auroc(calib_scored):.4f}  evaluation half: {auroc(eval_scored):.4f}")

    report(eval_scored, 0.5, "EVAL @ 0.5 (uncalibrated)")
    report(eval_scored, args.cosmos_threshold, f"EVAL @ COSMOS threshold {args.cosmos_threshold:.2f} (zero-shot transfer)")

    best_t, best_f1 = find_best_threshold(calib_scored)
    print(f"\n>>> Best threshold on MMFakeBench calibration half: {best_t:.2f} (macro_f1={best_f1:.4f})")
    report(eval_scored, best_t, f"EVAL @ MMFakeBench-calibrated threshold {best_t:.2f}")

    per_type(evals, scores, best_t)


if __name__ == "__main__":
    main()
