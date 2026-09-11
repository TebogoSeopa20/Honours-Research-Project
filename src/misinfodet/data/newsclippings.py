"""NewsCLIPpings (Merged/Balanced) preparation and loading.

NewsCLIPpings annotations reference images and captions from VisualNews.
Raw layout expected (see docs/DATASETS.md):

    data/newsclippings/news_clippings/data/merged_balanced/{train,val,test}.json
    data/visualnews/origin/...              (images)
    data/visualnews/data.json               (id -> caption/image_path map)

Output: unified JSONL per split with schema
    {"id", "image_path", "caption", "label", "source"}
label: 0 = pristine (matched), 1 = falsified (out-of-context).
"""
from __future__ import annotations

import json
from pathlib import Path

from ..utils import get_logger, write_jsonl

log = get_logger(__name__)

SPLITS = ("train", "val", "test")


def _load_visualnews_index(visualnews_dir: str | Path) -> dict[int, dict]:
    """Map VisualNews article id -> {caption, image_path}."""
    data_json = Path(visualnews_dir) / "data.json"
    if not data_json.exists():
        raise FileNotFoundError(
            f"VisualNews data.json not found at {data_json}. "
            "Request access via the VisualNews repository first (see docs/DATASETS.md)."
        )
    with open(data_json, "r", encoding="utf-8") as f:
        entries = json.load(f)
    return {int(e["id"]): e for e in entries}


def prepare_split(
    newsclippings_dir: str | Path,
    visualnews_dir: str | Path,
    split: str,
    out_path: str | Path,
    max_samples: int | None = None,
) -> int:
    """Convert one Merged/Balanced split into unified JSONL. Returns count."""
    ann_path = (
        Path(newsclippings_dir)
        / "news_clippings"
        / "data"
        / "merged_balanced"
        / f"{split}.json"
    )
    if not ann_path.exists():
        raise FileNotFoundError(
            f"NewsCLIPpings annotation not found at {ann_path} (see docs/DATASETS.md)."
        )

    with open(ann_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)["annotations"]

    vn = _load_visualnews_index(visualnews_dir)
    records, skipped = [], 0
    for i, ann in enumerate(annotations):
        if max_samples is not None and len(records) >= max_samples:
            break
        cap_entry = vn.get(int(ann["id"]))
        img_entry = vn.get(int(ann["image_id"]))
        if cap_entry is None or img_entry is None:
            skipped += 1
            continue
        image_path = str(Path(visualnews_dir) / img_entry["image_path"].lstrip("./"))
        records.append(
            {
                "id": f"nc_{split}_{i}",
                "image_path": image_path,
                "caption": cap_entry["caption"],
                "label": int(bool(ann["falsified"])),
                "source": "newsclippings_merged_balanced",
            }
        )

    write_jsonl(records, out_path)
    log.info(
        "%s: wrote %d records to %s (skipped %d missing VisualNews entries)",
        split,
        len(records),
        out_path,
        skipped,
    )
    return len(records)


def prepare_all(
    newsclippings_dir: str | Path,
    visualnews_dir: str | Path,
    out_dir: str | Path,
    max_samples: int | None = None,
) -> dict[str, int]:
    out_dir = Path(out_dir)
    counts = {}
    for split in SPLITS:
        counts[split] = prepare_split(
            newsclippings_dir,
            visualnews_dir,
            split,
            out_dir / f"newsclippings_{split}.jsonl",
            max_samples=max_samples,
        )
    return counts
