#!/usr/bin/env python
"""COSMOS evaluation of the retrieval-augmented pipeline under calibrated scoring.

Runs the same steps as Detector.detect (initial assessment -> query generation ->
retrieval -> optional re-ranking), then scores the evidence-grounded final prompt
with get_verdict_score instead of free generation. Threshold is calibrated on val.

Each item's queries, evidence and score are appended to disk as soon as they are
produced, so a crashed run resumes without re-spending search credits."""
import argparse
import json
from pathlib import Path

from PIL import Image

from calibrate_threshold import auroc, find_best_threshold, report
from misinfodet.models.model_factory import get_verdict_score, load_llava
from misinfodet.pipeline import detector as D
from misinfodet.pipeline.prompts import PAIRED_FINAL_VERDICT_PROMPT, PAIRED_VERDICT_PROMPT
from misinfodet.retrieval.evidence_retriever import EvidenceRetriever
from misinfodet.utils import Config, read_jsonl


def run_item(model, processor, retriever, rec, cfg):
    image = Image.open(rec["image_path"]).convert("RGB")
    c1, c2 = rec["caption1"], rec["caption2"]
    blob = f'Caption 1: "{c1}" | Caption 2: "{c2}"'

    initial_raw = D.generate(model, processor, image, PAIRED_VERDICT_PROMPT.format(caption1=c1, caption2=c2), cfg)
    initial_reasoning = D._extract_field(initial_raw, "REASONING") or initial_raw

    queries = D.generate_queries(model, processor, blob, initial_reasoning, cfg)
    evidence = retriever.retrieve(queries)
    if cfg.use_reranking:
        evidence = D.rerank(model, processor, blob, evidence, cfg)
    else:
        evidence = evidence[: cfg.rerank_keep_k]

    prompt = PAIRED_FINAL_VERDICT_PROMPT.format(
        caption1=c1, caption2=c2, initial_reasoning=initial_reasoning, evidence=D.format_evidence(evidence)
    )
    score = get_verdict_score(model, processor, image, prompt, cfg)
    return score, queries, evidence


def score_split(model, processor, retriever, records, split, cfg, out_path, done):
    import torch

    with open(out_path, "a", encoding="utf-8") as f:
        for i, rec in enumerate(records):
            rid = f"{split}_{i}"
            if rid in done:
                continue
            score, queries, evidence = run_item(model, processor, retriever, rec, cfg)
            row = {"id": rid, "split": split, "score": score, "label": rec["label"],
                   "queries": queries, "n_evidence": len(evidence), "evidence": evidence}
            done[rid] = row
            f.write(json.dumps(row) + "\n")
            f.flush()
            if (i + 1) % 10 == 0:
                torch.cuda.empty_cache()
                print(f"  {split} {i + 1}/{len(records)} done")
    return [(done[f"{split}_{i}"]["score"], rec["label"]) for i, rec in enumerate(records)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config)

    val = read_jsonl(Path(cfg.data_dir) / "cosmos_labeled_val.jsonl")
    test = read_jsonl(Path(cfg.data_dir) / "cosmos_labeled_test.jsonl")

    out_path = Path(cfg.output_dir) / "predictions" / f"{cfg.run_name}_retrieval_cosmos_scores.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"]: r for r in read_jsonl(out_path)} if out_path.exists() else {}
    print(f"{len(done)} already scored, {len(val) + len(test) - len(done)} to go")
    print(f"reranking={'on' if cfg.use_reranking else 'off'}, provider={cfg.search_provider}")

    print("Loading model...")
    model, processor = load_llava(cfg, trainable=False)
    retriever = EvidenceRetriever(cfg)

    val_scored = score_split(model, processor, retriever, val, "val", cfg, out_path, done)
    test_scored = score_split(model, processor, retriever, test, "test", cfg, out_path, done)

    empty = sum(1 for r in done.values() if r["n_evidence"] == 0)
    print(f"\nItems with no evidence retrieved: {empty}/{len(done)}")

    best_t, best_f1 = find_best_threshold(val_scored)
    print(f">>> Best threshold found on validation: {best_t:.2f} (val macro_f1={best_f1:.4f})")
    print(f"\nTEST AUROC (threshold-free): {auroc(test_scored):.4f}")
    report(test_scored, best_t, f"TEST @ calibrated threshold {best_t:.2f} (with retrieval)")
    print(f"\nScores written to {out_path}")


if __name__ == "__main__":
    main()
