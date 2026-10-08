#!/usr/bin/env python
"""Two-pass explanation generation for COSMOS test items.

Pass 1 (already done): verdicts come from the final system's saved calibrated scores.
Pass 2 (this script): an explanation model (base LLaVA by default) is shown the image,
both captions and the verdict, and asked to justify it.

Also exports a stratified sample with its images for human rating."""
import argparse
import csv
import json
import random
import shutil
from pathlib import Path

from PIL import Image

from calibrate_threshold import find_best_threshold
from misinfodet.models.model_factory import generate, load_llava
from misinfodet.utils import Config, read_jsonl

EXPLAIN_PROMPT = (
    "You are a fact-checking assistant analysing a news image and two captions "
    "that have each been associated with it.\n"
    'Caption 1: "{caption1}"\n'
    'Caption 2: "{caption2}"\n\n'
    "A detection system has judged this pairing to be {verdict}.\n\n"
    "In 2-4 sentences, explain the specific visual and textual evidence that "
    "supports this judgement. Refer to concrete details visible in the image and "
    "to the people, places, events, or dates named in the captions."
)
VERDICT_TEXT = {
    1: "OUT-OF-CONTEXT: the two captions describe different events, so at most one can genuinely apply to this image",
    0: "CONSISTENT: both captions can genuinely apply to this image",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base_llava_eval.yaml", help="explanation model")
    ap.add_argument("--scores", default="outputs/predictions/diag_stage1_only_cosmos_scores.jsonl",
                    help="saved scores of the system whose verdicts are explained")
    ap.add_argument("--n-rate", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config)
    cfg.max_new_tokens = args.max_new_tokens

    rows = read_jsonl(args.scores)
    threshold, _ = find_best_threshold([(r["score"], r["label"]) for r in rows if r["split"] == "val"])
    scores = {r["id"]: r["score"] for r in rows if r["split"] == "test"}
    test = read_jsonl(Path(cfg.data_dir) / "cosmos_labeled_test.jsonl")
    print(f"Verdict threshold (from validation): {threshold:.2f}")

    out_dir = Path(cfg.output_dir) / "explanations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "explanations_cosmos_test.jsonl"
    done = {r["id"]: r for r in read_jsonl(out_path)} if out_path.exists() else {}

    print("Loading explanation model...")
    model, processor = load_llava(cfg, trainable=False)

    import torch

    with open(out_path, "a", encoding="utf-8") as f:
        for i, rec in enumerate(test):
            rid = f"test_{i}"
            if rid in done:
                continue
            verdict = int(scores[rid] >= threshold)
            image = Image.open(rec["image_path"]).convert("RGB")
            prompt = EXPLAIN_PROMPT.format(caption1=rec["caption1"], caption2=rec["caption2"], verdict=VERDICT_TEXT[verdict])
            text = generate(model, processor, image, prompt, cfg)
            row = {"id": rid, "image_path": rec["image_path"], "caption1": rec["caption1"],
                   "caption2": rec["caption2"], "gold": rec["label"], "verdict": verdict,
                   "correct": int(verdict == rec["label"]), "explanation": text}
            done[rid] = row
            f.write(json.dumps(row) + "\n")
            f.flush()
            if (i + 1) % 10 == 0:
                torch.cuda.empty_cache()
                print(f"  {i + 1}/{len(test)} done")

    all_rows = [done[f"test_{i}"] for i in range(len(test))]
    words = [len(r["explanation"].split()) for r in all_rows]
    print(f"\nExplanations: {len(all_rows)}  mean length {sum(words) / len(words):.1f} words  "
          f"empty {sum(1 for w in words if w == 0)}")

    rng = random.Random(args.seed)
    sample = []
    for label in (0, 1):
        pool = [r for r in all_rows if r["gold"] == label]
        sample += rng.sample(pool, min(args.n_rate // 2, len(pool)))
    rng.shuffle(sample)

    img_dir = out_dir / "rating_images"
    img_dir.mkdir(exist_ok=True)
    with open(out_dir / "rating_sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "image_file", "caption1", "caption2", "gold", "verdict", "correct",
                    "explanation", "factual_accuracy_1to5", "coherence_1to5", "relevance_1to5", "notes"])
        for k, r in enumerate(sample, 1):
            img_name = f"{k:02d}{Path(r['image_path']).suffix}"
            shutil.copy(r["image_path"], img_dir / img_name)
            w.writerow([k, img_name, r["caption1"], r["caption2"],
                        "out-of-context" if r["gold"] else "consistent",
                        "out-of-context" if r["verdict"] else "consistent",
                        r["correct"], r["explanation"], "", "", "", ""])
    print(f"Rating sheet: {out_dir / 'rating_sheet.csv'} ({len(sample)} items), images in {img_dir}")


if __name__ == "__main__":
    main()
