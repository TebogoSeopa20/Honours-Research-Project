#!/usr/bin/env python
"""Phase 3: Stage 1 news-domain entity alignment tuning."""
import argparse
from pathlib import Path

from misinfodet.data.instruction_builder import build_stage1_examples
from misinfodet.models.model_factory import load_llava
from misinfodet.models.training import train
from misinfodet.utils import Config, read_jsonl, set_seed, write_run_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage1_entity_alignment.yaml")
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config)
    set_seed(cfg.seed)

    records = read_jsonl(Path(cfg.data_dir) / "cosmos_train.jsonl")
    examples = list(build_stage1_examples(records, seed=cfg.seed))
    if cfg.max_samples:
        examples = examples[: cfg.max_samples]
    print(f"Stage 1: {len(examples)} entity-alignment examples")

    save_dir = Path(cfg.output_dir) / "checkpoints" / "stage1"
    write_run_meta(cfg, save_dir)
    model, processor = load_llava(cfg, trainable=True)
    train(model, processor, examples, cfg, save_dir)


if __name__ == "__main__":
    main()
