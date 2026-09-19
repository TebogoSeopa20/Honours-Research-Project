#!/usr/bin/env python
"""Phase 3 validation: does Stage 1 tuning actually improve entity alignment?

Picks real COSMOS images whose captions name specific people/places/orgs,
generates a description from base LLaVA and from the Stage 1-adapted model
with the same prompt, and prints both side by side against the real caption
so you can judge whether the tuned model names the right entities where the
base model gives something generic ("a man in a suit").

This is a qualitative check, not a metric — read the outputs yourself.
"""
import argparse
import json
from pathlib import Path

from PIL import Image

from misinfodet.models.model_factory import generate, load_llava
from misinfodet.utils import Config

DESCRIBE_PROMPT = (
    "Describe this image in one or two sentences, naming any specific "
    "people, places, or organizations you recognize."
)


def _looks_like_it_names_an_entity(caption: str) -> bool:
    """Rough heuristic: a capitalized multi-word phrase, e.g. 'Julian
    Castro', 'San Antonio' — good enough to pick interesting examples for
    a qualitative check, not used for anything that needs to be precise."""
    words = caption.split()
    for i in range(0, len(words) - 1):
        if words[i][:1].isupper() and words[i + 1][:1].isupper():
            return True
    return False


def pick_examples(train_jsonl: Path, n: int) -> list[dict]:
    examples = []
    with open(train_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if _looks_like_it_names_an_entity(rec["caption"]):
                examples.append(rec)
            if len(examples) >= n:
                break
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--stage1_adapter", default="outputs/checkpoints/stage1")
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    examples = pick_examples(Path(cfg.data_dir) / "cosmos_train.jsonl", args.n)
    print(f"Picked {len(examples)} examples with likely named entities\n")

    print("Loading base model (no adapter)...")
    cfg.stage1_adapter = None
    base_model, base_processor = load_llava(cfg, trainable=False)
    base_outputs = []
    for ex in examples:
        image = Image.open(ex["image_path"]).convert("RGB")
        base_outputs.append(generate(base_model, base_processor, image, DESCRIBE_PROMPT, cfg))
    del base_model
    import torch
    torch.cuda.empty_cache()

    print("Loading Stage 1-adapted model...")
    cfg.stage1_adapter = args.stage1_adapter
    tuned_model, tuned_processor = load_llava(cfg, trainable=False)
    tuned_outputs = []
    for ex in examples:
        image = Image.open(ex["image_path"]).convert("RGB")
        tuned_outputs.append(generate(tuned_model, tuned_processor, image, DESCRIBE_PROMPT, cfg))

    for ex, base_out, tuned_out in zip(examples, base_outputs, tuned_outputs):
        print("=" * 80)
        print(f"REAL CAPTION : {ex['caption']}")
        print(f"BASE MODEL   : {base_out}")
        print(f"STAGE1 MODEL : {tuned_out}")
    print("=" * 80)


if __name__ == "__main__":
    main()
