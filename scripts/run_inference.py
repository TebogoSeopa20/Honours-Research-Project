#!/usr/bin/env python
"""Full-system (or ablation) inference: config toggles decide the variant."""
import argparse
from pathlib import Path

from misinfodet.models.model_factory import load_llava
from misinfodet.pipeline.detector import Detector
from misinfodet.utils import Config, iter_jsonl, set_seed, write_run_meta, append_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--dataset", default="cosmos", choices=["cosmos", "newsclippings", "mmfakebench"])
    ap.add_argument("--resume", action="store_true", help="skip ids already predicted")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    if not cfg.use_instruction_tuning:
        cfg.stage1_adapter = None
        cfg.stage2_adapter = None
    set_seed(cfg.seed)

    # Same held-out split as run_baseline.py — see the comment there.
    filename = f"cosmos_labeled_{args.split}.jsonl" if args.dataset == "cosmos" else f"{args.dataset}_{args.split}.jsonl"
    data_path = Path(cfg.data_dir) / filename
    out_path = Path(cfg.output_dir) / "predictions" / f"{cfg.run_name}_{args.dataset}_{args.split}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if args.resume and out_path.exists():
        done = {r["id"] for r in iter_jsonl(out_path)}
        print(f"Resuming: {len(done)} already predicted")
    else:
        out_path.unlink(missing_ok=True)
    write_run_meta(cfg, out_path.parent)

    model, processor = load_llava(cfg)
    detector = Detector(model, processor, cfg)

    n = 0
    for rec in iter_jsonl(data_path):
        if rec["id"] in done:
            continue
        if cfg.max_samples and n >= cfg.max_samples:
            break
        result = detector.detect(rec).to_dict()
        result["forgery_type"] = rec.get("forgery_type")
        append_jsonl(result, out_path)
        n += 1
        if n % 50 == 0:
            print(f"{n} done")
    print("Predictions written to", out_path)


if __name__ == "__main__":
    main()
