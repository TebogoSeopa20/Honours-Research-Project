"""MMFakeBench preparation (secondary benchmark, evaluation only).

Expected raw layout after download (see docs/DATASETS.md):

    data/mmfakebench/MMFakeBench_test/...      (images)
    data/mmfakebench/MMFakeBench_test.json     (annotations)

Each raw annotation has: image_path, text, gt_answers ("True"/"Fake"),
fake_cls (one of 12 sub-types, e.g. "original", "mismatch", "text_veracity...").

Unified schema mirrors NewsCLIPpings, with the fine-grained forgery type kept
for the mixed-source breakdown analysis in the final report:
    {"id", "image_path", "caption", "label", "forgery_type", "source"}
label: 0 = real, 1 = fake (any forgery source).
"""
from __future__ import annotations

import json
from pathlib import Path

from ..utils import get_logger, write_jsonl

log = get_logger(__name__)


def prepare(
    mmfakebench_dir: str | Path,
    out_path: str | Path,
    split: str = "test",
    max_samples: int | None = None,
) -> int:
    root = Path(mmfakebench_dir)
    ann_path = root / f"MMFakeBench_{split}.json"
    if not ann_path.exists():
        raise FileNotFoundError(
            f"MMFakeBench annotations not found at {ann_path} (see docs/DATASETS.md)."
        )
    with open(ann_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    records = []
    for i, ann in enumerate(annotations):
        if max_samples is not None and len(records) >= max_samples:
            break
        gt = str(ann.get("gt_answers", "")).strip().lower()
        label = 0 if gt in {"true", "real"} else 1
        records.append(
            {
                "id": f"mmfb_{split}_{i}",
                "image_path": str(root / ann["image_path"].lstrip("./")),
                "caption": ann["text"],
                "label": label,
                "forgery_type": ann.get("fake_cls", "unknown"),
                "source": "mmfakebench",
            }
        )

    write_jsonl(records, out_path)
    log.info("MMFakeBench %s: wrote %d records to %s", split, len(records), out_path)
    return len(records)
