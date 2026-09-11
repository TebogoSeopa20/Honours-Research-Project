"""Classification metrics for both benchmarks.

Macro-F1, precision, recall, accuracy, per-class breakdown, confusion matrix,
and (for MMFakeBench) a per-forgery-type breakdown for the mixed-source
analysis required by the proposal's evaluation framework.
"""
from __future__ import annotations

from collections import defaultdict

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

LABEL_NAMES = {0: "consistent/real", 1: "out-of-context/fake"}


def classification_report(golds: list[int], preds: list[int]) -> dict:
    if not golds:
        return {"error": "no samples"}
    p, r, f1, support = precision_recall_fscore_support(
        golds, preds, labels=[0, 1], zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        golds, preds, average="macro", zero_division=0
    )
    return {
        "n": len(golds),
        "accuracy": round(accuracy_score(golds, preds), 4),
        "macro_f1": round(macro_f1, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "per_class": {
            LABEL_NAMES[i]: {
                "precision": round(p[i], 4),
                "recall": round(r[i], 4),
                "f1": round(f1[i], 4),
                "support": int(support[i]),
            }
            for i in (0, 1)
        },
        "confusion_matrix": confusion_matrix(golds, preds, labels=[0, 1]).tolist(),
    }


def evaluate_predictions(predictions: list[dict]) -> dict:
    """Evaluate JSONL prediction records (from Detector.to_dict())."""
    usable = [p for p in predictions if p.get("gold_label") is not None]
    golds = [int(p["gold_label"]) for p in usable]
    preds = [int(p["pred_label"]) for p in usable]
    report = classification_report(golds, preds)
    report["parse_failure_rate"] = round(
        sum(bool(p.get("parse_failed")) for p in usable) / max(len(usable), 1), 4
    )

    # Per-forgery-type breakdown (MMFakeBench mixed-source analysis)
    by_type = defaultdict(lambda: {"golds": [], "preds": []})
    for p in usable:
        ft = p.get("forgery_type")
        if ft:
            by_type[ft]["golds"].append(int(p["gold_label"]))
            by_type[ft]["preds"].append(int(p["pred_label"]))
    if by_type:
        report["per_forgery_type"] = {
            ft: {
                "n": len(v["golds"]),
                "accuracy": round(accuracy_score(v["golds"], v["preds"]), 4),
                "f1_macro": round(
                    f1_score(v["golds"], v["preds"], average="macro", zero_division=0), 4
                ),
            }
            for ft, v in sorted(by_type.items())
        }
    return report
