#!/usr/bin/env python
"""Calibrated verdict scoring on the NewsCLIPpings merged_balanced subset built by
prepare_newsclippings_subset.py. Threshold chosen on the val subset, applied unchanged to test.
Same prompt and scoring as calibrate_mmfakebench.py, so results compare with MMFakeBench and COSMOS."""
import argparse
from pathlib import Path

from calibrate_mmfakebench import auroc, score_all
from calibrate_threshold import find_best_threshold, report
from misinfodet.models.model_factory import load_llava
from misinfodet.utils import Config, read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cosmos-threshold", type=float, default=None)
    ap.add_argument("--blank-image", action="store_true")
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config)

    val = read_jsonl(Path(cfg.data_dir) / "newsclippings_val.jsonl")
    test = read_jsonl(Path(cfg.data_dir) / "newsclippings_test.jsonl")
    print(f"NewsCLIPpings subset: val n={len(val)}  test n={len(test)}")

    out = Path(cfg.output_dir) / "predictions" / f"{cfg.run_name}_newsclippings{'_blank' if args.blank_image else ''}_scores.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    model, processor = load_llava(cfg, trainable=False)
    scores = score_all(model, processor, val + test, cfg, out, args.blank_image)

    vs = [(scores[r["id"]], r["label"]) for r in val]
    ts = [(scores[r["id"]], r["label"]) for r in test]
    print(f"\nAUROC  val: {auroc(vs):.4f}  test: {auroc(ts):.4f}")
    report(ts, 0.5, "TEST @ 0.5 (uncalibrated)")
    if args.cosmos_threshold is not None:
        report(ts, args.cosmos_threshold, f"TEST @ COSMOS threshold {args.cosmos_threshold:.2f}")
    t, f1 = find_best_threshold(vs)
    print(f"\n>>> Best threshold on NewsCLIPpings val: {t:.2f} (val macro_f1={f1:.4f})")
    report(ts, t, f"TEST @ NewsCLIPpings-calibrated threshold {t:.2f}")


if __name__ == "__main__":
    main()
