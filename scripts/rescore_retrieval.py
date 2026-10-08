#!/usr/bin/env python
"""Re-score the cached retrieval run with the evidence kept but the model's initial
assessment removed from the final prompt. Uses the evidence saved by
calibrate_retrieval.py, so no searches or generation are repeated."""
import argparse
import json
from pathlib import Path

from PIL import Image

from calibrate_threshold import auroc, find_best_threshold, report
from misinfodet.models.model_factory import get_verdict_score, load_llava
from misinfodet.pipeline import detector as D
from misinfodet.pipeline.prompts import PAIRED_FINAL_VERDICT_PROMPT
from misinfodet.utils import Config, read_jsonl

PLACEHOLDER = "(not provided)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config)

    pred_dir = Path(cfg.output_dir) / "predictions"
    cached = {r["id"]: r for r in read_jsonl(pred_dir / f"{cfg.run_name}_retrieval_cosmos_scores.jsonl")}
    out_path = pred_dir / f"{cfg.run_name}_retrieval_evonly_cosmos_scores.jsonl"
    done = {r["id"]: r for r in read_jsonl(out_path)} if out_path.exists() else {}

    print("Loading model...")
    model, processor = load_llava(cfg, trainable=False)

    import torch

    scored = {}
    with open(out_path, "a", encoding="utf-8") as f:
        for split in ("val", "test"):
            recs = read_jsonl(Path(cfg.data_dir) / f"cosmos_labeled_{split}.jsonl")
            scored[split] = []
            for i, rec in enumerate(recs):
                rid = f"{split}_{i}"
                if rid not in done:
                    image = Image.open(rec["image_path"]).convert("RGB")
                    prompt = PAIRED_FINAL_VERDICT_PROMPT.format(
                        caption1=rec["caption1"], caption2=rec["caption2"],
                        initial_reasoning=PLACEHOLDER,
                        evidence=D.format_evidence(cached[rid]["evidence"]),
                    )
                    score = get_verdict_score(model, processor, image, prompt, cfg)
                    done[rid] = {"id": rid, "split": split, "score": score, "label": rec["label"]}
                    f.write(json.dumps(done[rid]) + "\n")
                    f.flush()
                    if (i + 1) % 20 == 0:
                        torch.cuda.empty_cache()
                        print(f"  {split} {i + 1}/{len(recs)} done")
                scored[split].append((done[rid]["score"], rec["label"]))

    best_t, best_f1 = find_best_threshold(scored["val"])
    print(f"\n>>> Best threshold found on validation: {best_t:.2f} (val macro_f1={best_f1:.4f})")
    print(f"\nTEST AUROC (threshold-free): {auroc(scored['test']):.4f}")
    report(scored["test"], best_t, f"TEST @ calibrated threshold {best_t:.2f} (evidence, no initial assessment)")
    print(f"\nScores written to {out_path}")


if __name__ == "__main__":
    main()
