"""Explanation quality assessment (Gap 3 / Gap 4 of the proposal).

Two channels, matching Section 3.5:
  1. Automated LLM-as-judge scoring with a fixed rubric
     (factual accuracy, coherence, relevance; 1-5 each).
  2. Export of a stratified sample to CSV for the small human evaluation
     study, with a blind column order so annotators don't see gold labels.
"""
from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path

from ..pipeline.prompts import EXPLANATION_JUDGE_PROMPT
from ..utils import Config, get_logger

log = get_logger(__name__)

CRITERIA = ("factual_accuracy", "coherence", "relevance")


def build_judge_prompt(pred: dict, caption: str) -> str:
    return EXPLANATION_JUDGE_PROMPT.format(
        caption=caption,
        gold_label="out-of-context" if pred.get("gold_label") == 1 else "consistent",
        pred_label="out-of-context" if pred.get("pred_label") == 1 else "consistent",
        explanation=pred.get("explanation", ""),
    )


def parse_judge_response(raw: str) -> dict | None:
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        scores = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    if not all(c in scores for c in CRITERIA):
        return None
    return {c: max(1, min(5, int(scores[c]))) for c in CRITERIA}


def judge_explanations(model, processor, predictions: list[dict],
                       captions: dict[str, str], cfg: Config) -> dict:
    """Score all explanations with the LVLM-as-judge; returns mean scores."""
    from ..models.model_factory import generate_text_only

    all_scores = {c: [] for c in CRITERIA}
    failures = 0
    for p in predictions:
        if not p.get("explanation"):
            continue
        raw = generate_text_only(
            model, processor, build_judge_prompt(p, captions.get(p["id"], "")), cfg
        )
        scores = parse_judge_response(raw)
        if scores is None:
            failures += 1
            continue
        for c in CRITERIA:
            all_scores[c].append(scores[c])
    n = max(len(all_scores[CRITERIA[0]]), 1)
    return {
        "n_scored": len(all_scores[CRITERIA[0]]),
        "judge_parse_failures": failures,
        **{f"mean_{c}": round(sum(v) / max(len(v), 1), 3) for c, v in all_scores.items()},
    }


def export_human_eval_sample(predictions: list[dict], captions: dict[str, str],
                             out_csv: str | Path, n: int = 50, seed: int = 42) -> int:
    """Stratified sample (correct/incorrect x real/fake) for human annotation."""
    rng = random.Random(seed)
    strata: dict[tuple, list[dict]] = {}
    for p in predictions:
        if not p.get("explanation") or p.get("gold_label") is None:
            continue
        key = (p["gold_label"], int(p["pred_label"] == p["gold_label"]))
        strata.setdefault(key, []).append(p)
    per_stratum = max(1, n // max(len(strata), 1))
    sample = []
    for members in strata.values():
        rng.shuffle(members)
        sample.extend(members[:per_stratum])
    rng.shuffle(sample)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "item_id", "caption", "system_verdict", "explanation",
            "factual_accuracy_1to5", "coherence_1to5", "relevance_1to5", "annotator_notes",
        ])
        for p in sample[:n]:
            verdict = "out-of-context" if p["pred_label"] == 1 else "consistent"
            writer.writerow([p["id"], captions.get(p["id"], ""), verdict,
                             p["explanation"], "", "", "", ""])
    log.info("Exported %d items for human evaluation to %s", min(len(sample), n), out_csv)
    return min(len(sample), n)
