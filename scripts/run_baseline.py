#!/usr/bin/env python
"""Phase 2: zero-shot LVLM baseline (no tuning, no retrieval)."""
import argparse
from pathlib import Path

from misinfodet.models.model_factory import load_llava
from misinfodet.pipeline.detector import Detector
from misinfodet.utils import Config, iter_jsonl, set_seed, write_run_meta, append_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--dataset", default="cosmos", choices=["cosmos", "newsclippings", "mmfakebench"])
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    cfg.use_retrieval = False
    cfg.use_reranking = False
    cfg.stage1_adapter = None
    cfg.stage2_adapter = None
    set_seed(cfg.seed)

    # COSMOS's only labeled data is the 70/15/15 split scripts/prepare_cosmos.py
    # carved out of its ~1,700 labeled images — always eval on that same split
    # (not the full unsplit pool) so baseline and tuned-model numbers are
    # comparable to each other later.
    filename = f"cosmos_labeled_{args.split}.jsonl" if args.dataset == "cosmos" else f"{args.dataset}_{args.split}.jsonl"
    data_path = Path(cfg.data_dir) / filename
    out_path = Path(cfg.output_dir) / "predictions" / f"baseline_{args.dataset}_{args.split}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.unlink(missing_ok=True)
    write_run_meta(cfg, out_path.parent)

    model, processor = load_llava(cfg)
    detector = Detector(model, processor, cfg)

    for i, rec in enumerate(iter_jsonl(data_path)):
        if cfg.max_samples and i >= cfg.max_samples:
            break
        result = detector.detect(rec).to_dict()
        result["forgery_type"] = rec.get("forgery_type")
        append_jsonl(result, out_path)
        if (i + 1) % 50 == 0:
            print(f"{i + 1} done")
    print("Predictions written to", out_path)


if __name__ == "__main__":
    main()
