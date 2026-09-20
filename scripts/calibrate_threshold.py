#!/usr/bin/env python
"""Calibrate the out-of-context/consistent decision threshold using the
model's actual token-level confidence, rather than trusting whichever word
it happens to generate first in free text.

Standard technique: force the model to the point right after "VERDICT:"
and read its probability for the next token starting "consistent" vs
"out-of-context" — a continuous score in [0,1] instead of a hard binary
guess. That score gets thresholded using the validation set (never
trained on), which can correct the class bias seen across every
configuration tested so far without any further training.

IMPORTANT: run with --debug first, on one example, to confirm the
tokenization assumption this relies on actually holds for this model,
before trusting a full calibration run.
"""
import argparse
from pathlib import Path

from PIL import Image

from misinfodet.models.model_factory import (
    debug_verdict_tokenization,
    get_verdict_score,
    load_llava,
)
from misinfodet.pipeline.prompts import PAIRED_VERDICT_PROMPT
from misinfodet.utils import Config, read_jsonl


def score_dataset(model, processor, records, cfg, label: str = ""):
    """Returns [(score, gold_label), ...] — score = P(out-of-context).

    Frees GPU memory every 20 examples and prints progress — a silent
    OOM crash partway through 500+ forward passes (confirmed: process
    disappeared entirely, no traceback, GPU memory back to 0MiB) is the
    real failure mode here, not a slow-but-alive computation.
    """
    import torch

    results = []
    for i, rec in enumerate(records):
        image = Image.open(rec["image_path"]).convert("RGB")
        prompt = PAIRED_VERDICT_PROMPT.format(caption1=rec["caption1"], caption2=rec["caption2"])
        score = get_verdict_score(model, processor, image, prompt, cfg)
        results.append((score, rec["label"]))
        if (i + 1) % 20 == 0:
            torch.cuda.empty_cache()
            print(f"  {label}{i + 1}/{len(records)} done")
    return results


def _macro_f1(preds, golds):
    f1s = []
    for cls in (0, 1):
        tp = sum(1 for p, g in zip(preds, golds) if p == cls and g == cls)
        fp = sum(1 for p, g in zip(preds, golds) if p == cls and g != cls)
        fn = sum(1 for p, g in zip(preds, golds) if p != cls and g == cls)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s)


def find_best_threshold(scored):
    """Sweep thresholds 0.05-0.95, return the one maximizing macro F1."""
    best_t, best_f1 = 0.5, -1.0
    golds = [g for _, g in scored]
    for i in range(5, 96):
        t = i / 100
        preds = [1 if s >= t else 0 for s, _ in scored]
        f1 = _macro_f1(preds, golds)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def report(scored, threshold, label):
    preds = [1 if s >= threshold else 0 for s, _ in scored]
    golds = [g for _, g in scored]
    acc = sum(1 for p, g in zip(preds, golds) if p == g) / len(golds)
    f1 = _macro_f1(preds, golds)
    cm = [[0, 0], [0, 0]]  # rows=gold, cols=pred
    for p, g in zip(preds, golds):
        cm[g][p] += 1
    print(f"\n--- {label} (threshold={threshold:.2f}) ---")
    print(f"n={len(golds)}  accuracy={acc:.4f}  macro_f1={f1:.4f}")
    print(f"predicted consistent: {preds.count(0)}  |  predicted out-of-context: {preds.count(1)}")
    print(f"actual    consistent: {golds.count(0)}  |  actual    out-of-context: {golds.count(1)}")
    print(f"confusion_matrix (rows=gold, cols=pred): {cm}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--debug", action="store_true",
                    help="run tokenization sanity check on 1 example, then exit")
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config)

    print("Loading model...")
    model, processor = load_llava(cfg, trainable=False)

    if args.debug:
        val_records = read_jsonl(Path(cfg.data_dir) / "cosmos_labeled_val.jsonl")
        rec = val_records[0]
        image = Image.open(rec["image_path"]).convert("RGB")
        prompt = PAIRED_VERDICT_PROMPT.format(caption1=rec["caption1"], caption2=rec["caption2"])
        debug_verdict_tokenization(model, processor, image, prompt, cfg)
        return

    val_records = read_jsonl(Path(cfg.data_dir) / "cosmos_labeled_val.jsonl")
    test_records = read_jsonl(Path(cfg.data_dir) / "cosmos_labeled_test.jsonl")

    print(f"Scoring {len(val_records)} validation examples...")
    val_scored = score_dataset(model, processor, val_records, cfg, label="val ")
    report(val_scored, 0.5, "Validation @ default 0.5 threshold")

    best_t, best_f1 = find_best_threshold(val_scored)
    print(f"\n>>> Best threshold found on validation: {best_t:.2f} (val macro_f1={best_f1:.4f})")

    print(f"\nScoring {len(test_records)} test examples...")
    test_scored = score_dataset(model, processor, test_records, cfg, label="test ")
    report(test_scored, 0.5, "TEST @ default 0.5 (uncalibrated)")
    report(test_scored, best_t, f"TEST @ calibrated threshold {best_t:.2f}")


if __name__ == "__main__":
    main()