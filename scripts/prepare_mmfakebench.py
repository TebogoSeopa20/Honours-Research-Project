#!/usr/bin/env python
"""Prepare MMFakeBench test split into unified JSONL (evaluation only)."""
import argparse
from pathlib import Path

from misinfodet.data.mmfakebench import prepare
from misinfodet.utils import Config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config)
    n = prepare(cfg.mmfakebench_dir, Path(cfg.data_dir) / "mmfakebench_test.jsonl")
    print(f"Prepared {n} MMFakeBench records")


if __name__ == "__main__":
    main()
