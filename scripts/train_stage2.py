#!/usr/bin/env python
"""Phase 4: Stage 2 out-of-context detection fine-tuning (loads Stage 1)."""
import argparse
from pathlib import Path

from misinfodet.data.instruction_builder import build_stage2_examples_paired
from misinfodet.models.model_factory import load_llava
from misinfodet.models.training import train
from misinfodet.utils import Config, read_jsonl, set_seed, write_run_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage2_ooc_finetune.yaml")
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config)
    set_seed(cfg.seed)

    # Only ~1,700 COSMOS test images are labeled at all (see docs/DATASETS.md);
    # cosmos_labeled_train.jsonl is the 70% split scripts/prepare_cosmos.py
    # carved out for this stage. Never train on cosmos_labeled_val/test.jsonl.
    records = read_jsonl(Path(cfg.data_dir) / "cosmos_labeled_train.jsonl")
    examples = list(build_stage2_examples_paired(records))
    if cfg.max_samples:
        examples = examples[: cfg.max_samples]
    print(f"Stage 2: {len(examples)} OOC detection examples")

    save_dir = Path(cfg.output_dir) / "checkpoints" / "stage2"
    write_run_meta(cfg, save_dir)
    model, processor = load_llava(cfg, trainable=True)  # cfg.stage1_adapter applied inside
    train(model, processor, examples, cfg, save_dir)


if __name__ == "__main__":
    main()
