#!/usr/bin/env python
"""Prepare NewsCLIPpings Merged/Balanced splits into unified JSONL."""
import argparse

from misinfodet.data.newsclippings import prepare_all
from misinfodet.utils import Config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config)
    counts = prepare_all(
        cfg.newsclippings_dir,
        cfg.visualnews_dir,
        cfg.data_dir,
        max_samples=args.max_samples or cfg.max_samples,
    )
    print("Prepared:", counts)


if __name__ == "__main__":
    main()
