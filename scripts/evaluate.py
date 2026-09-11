#!/usr/bin/env python
"""Evaluate a predictions JSONL: classification metrics + optional
explanation-quality judging and human-eval CSV export."""
import argparse
import json
from pathlib import Path

from misinfodet.evaluation.explanation_quality import export_human_eval_sample
from misinfodet.evaluation.metrics import evaluate_predictions
from misinfodet.utils import Config, read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--dataset-jsonl", default=None,
                    help="Source dataset JSONL (needed for human-eval export captions)")
    ap.add_argument("--export-human-eval", type=int, default=0,
                    help="If >0, export this many items for human annotation")
    args = ap.parse_args()

    preds = read_jsonl(args.predictions)
    report = evaluate_predictions(preds)
    out = Path(args.predictions).with_name(Path(args.predictions).stem + "_metrics.json")
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("Metrics written to", out)

    if args.export_human_eval and args.dataset_jsonl:
        cfg = Config.from_yaml(args.config)
        captions = {
            r["id"]: r["caption"] if "caption" in r
            else f'Caption 1: "{r["caption1"]}" / Caption 2: "{r["caption2"]}"'
            for r in read_jsonl(args.dataset_jsonl)
        }
        csv_path = Path(cfg.output_dir) / "human_eval_sample.csv"
        export_human_eval_sample(preds, captions, csv_path, n=args.export_human_eval, seed=cfg.seed)


if __name__ == "__main__":
    main()
