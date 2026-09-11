#!/usr/bin/env python
"""Prepare COSMOS train/val (unlabeled) and test (labeled) into unified JSONL."""
import argparse
from pathlib import Path

from misinfodet.data.cosmos import prepare_all
from misinfodet.utils import Config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config)
    counts = prepare_all(cfg.cosmos_dir, cfg.data_dir, max_samples=cfg.max_samples)
    for split, n in counts.items():
        print(f"{split}: {n} records")


if __name__ == "__main__":
    main()
