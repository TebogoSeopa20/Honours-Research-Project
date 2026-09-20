#!/usr/bin/env python
"""Diagnose the gap between get_verdict_score's uncalibrated result and
real generation, on a small sample — cheap, before committing real GPU
budget to another full validation+test run.

For each example, compares three things:
  1. get_verdict_score's bucketed decision (sums prob across all "out"/
     "cons"-prefixed tokens in top-200)
  2. the literal single top-1 token's bucket — this is what greedy
     generation ACTUALLY uses for the first token, no summing
  3. the real generated text's actual first word

If (1) and (2) disagree often, the bucket-summing approach is the
problem. If (2) and (3) disagree often, something downstream of the
first token changes the effective verdict (rarer, but possible).
"""
import argparse
from pathlib import Path

from PIL import Image

from misinfodet.models.model_factory import generate, load_llava
from misinfodet.pipeline.prompts import PAIRED_VERDICT_PROMPT
from misinfodet.utils import Config, read_jsonl


def _top1_bucket(next_token_logits, tokenizer) -> str:
    """The single highest-logit token's bucket — exactly what greedy
    decoding uses, no summing over multiple candidates."""
    idx = int(next_token_logits.argmax())
    text = tokenizer.decode([idx]).strip().lower()
    if text.startswith("out"):
        return "out"
    if text.startswith("cons"):
        return "consistent"
    return f"OTHER:{text!r}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=25)
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config)

    import torch
    from misinfodet.models.model_factory import _bucket_probs

    print("Loading model...")
    model, processor = load_llava(cfg, trainable=False)

    test_records = read_jsonl(Path(cfg.data_dir) / "cosmos_labeled_test.jsonl")[: args.n]

    mismatches_1_vs_2 = 0
    mismatches_2_vs_3 = 0
    for i, rec in enumerate(test_records):
        image = Image.open(rec["image_path"]).convert("RGB")
        prompt = PAIRED_VERDICT_PROMPT.format(caption1=rec["caption1"], caption2=rec["caption2"])
        conversation = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
        ]
        prompt_text = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        next_token_logits = outputs.logits[0, -1, :]

        p_out, p_consistent = _bucket_probs(next_token_logits, processor.tokenizer)
        bucket_decision = "out" if p_out >= p_consistent else "consistent"
        top1 = _top1_bucket(next_token_logits, processor.tokenizer)

        raw = generate(model, processor, image, prompt, cfg)
        actual_first_word = raw.strip().split()[0].lower() if raw.strip() else ""
        actual_bucket = ("out" if actual_first_word.startswith("out")
                         else "consistent" if actual_first_word.startswith("cons")
                         else f"OTHER:{actual_first_word!r}")

        m12 = bucket_decision != top1
        m23 = top1 != actual_bucket
        mismatches_1_vs_2 += m12
        mismatches_2_vs_3 += m23
        flag = "  <-- MISMATCH" if (m12 or m23) else ""
        print(f"[{i}] gold={rec['label']}  bucket_sum={bucket_decision}  "
              f"top1={top1}  actual={actual_bucket}{flag}")
        if (i + 1) % 20 == 0:
            torch.cuda.empty_cache()

    n = len(test_records)
    print(f"\nbucket_sum vs top1 disagreements: {mismatches_1_vs_2}/{n}")
    print(f"top1 vs actual generation disagreements: {mismatches_2_vs_3}/{n}")


if __name__ == "__main__":
    main()
