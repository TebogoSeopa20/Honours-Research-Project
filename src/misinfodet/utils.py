"""Shared utilities: config loading, reproducibility, logging, JSONL IO."""
from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

import yaml


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    """Flat-ish experiment config loaded from YAML.

    Unknown keys are kept in `extra` so configs can evolve without code churn.
    """

    # Experiment identity
    run_name: str = "run"
    seed: int = 42
    output_dir: str = "outputs"
    wandb: bool = False
    wandb_project: str = "lvlm-misinfo-detection"

    # Data
    data_dir: str = "data"
    newsclippings_dir: str = "data/newsclippings"
    visualnews_dir: str = "data/visualnews"
    mmfakebench_dir: str = "data/mmfakebench"
    cosmos_dir: str = "data/cosmos"
    max_samples: int | None = None  # subsample for smoke tests

    # Model
    base_model: str = "llava-hf/llava-1.5-7b-hf"
    load_in_4bit: bool = True
    stage1_adapter: str | None = None
    stage2_adapter: str | None = None
    max_new_tokens: int = 256
    temperature: float = 0.0

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Training
    epochs: int = 1
    batch_size: int = 4
    grad_accum: int = 4
    save_every_steps: int = 50
    lr: float = 2e-4
    warmup_ratio: float = 0.03

    # System toggles (used by ablations)
    use_instruction_tuning: bool = True
    use_retrieval: bool = True
    use_reranking: bool = True
    generate_explanations: bool = True
    retrieval_top_k: int = 8
    rerank_keep_k: int = 3
    search_provider: str = "serper"  # serper | bing | none(offline corpus)
    search_api_key_env: str = "SEARCH_API_KEY"

    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        kwargs = {k: v for k, v in raw.items() if k in known}
        extra = {k: v for k, v in raw.items() if k not in known}
        cfg = cls(**kwargs)
        cfg.extra = extra
        return cfg

    def dump(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(yaml.safe_dump(asdict(self), sort_keys=False))


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "no-git"


def write_run_meta(cfg: Config, out_dir: str | Path) -> None:
    """Persist exact experiment provenance next to its outputs."""
    meta = {
        "run_name": cfg.run_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": git_commit(),
        "python": sys.version,
        "config": asdict(cfg),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def get_logger(name: str = "misinfodet") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# --------------------------------------------------------------------------- #
# JSONL IO
# --------------------------------------------------------------------------- #
def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(records: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(record: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")