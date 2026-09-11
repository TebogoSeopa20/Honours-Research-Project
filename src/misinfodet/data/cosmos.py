"""COSMOS (Aneja et al., 2021) preparation and loading.

Raw layout expected after downloading and unzipping cosmos_anns.zip,
images_train.zip, images_val.zip, images_test.zip (see docs/DATASETS.md):

    data/cosmos/train_data.json
    data/cosmos/val_data.json
    data/cosmos/test_data.json
    data/cosmos/train/...           (images)
    data/cosmos/val/...             (images)
    data/cosmos/test/...            (images)

IMPORTANT — this dataset has two incompatible shapes, not one:

train_data.json / val_data.json: one image with a LIST of articles, each
{"caption", "article_url", "caption_modified", "entity_list"}. No label —
COSMOS's own method is self-supervised and never labels these splits.
Output here is one unlabeled (image, caption) record per article, for
Stage 1 news-domain entity-alignment tuning only (no OOC supervision needed).
Schema: {"id", "image_path", "caption", "source"}

test_data.json: one image with exactly TWO candidate captions (caption1,
caption2) and a single label for whether pairing them with this image is
out-of-context. This is a genuinely different task shape from
NewsCLIPpings/MMFakeBench's one-image-one-caption schema — it is NOT
flattened into two separate single-caption records here, since the label
describes the pair, not either caption individually. Prompting/instruction
code that consumes this needs a two-caption template, not the existing
single-caption one.
Schema: {"id", "image_path", "caption1", "caption2", "label", "source"}
label: 0 = not out-of-context, 1 = out-of-context.

Only ~1,700 test images carry any label at all — see docs/DATASETS.md for
why this caps how much labeled train/val/test data is actually available.
prepare_all splits this pool 70/15/15 (stratified, seeded) into
cosmos_labeled_{train,val,test}.jsonl for Stage 2 fine-tuning and held-out
evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..utils import get_logger, write_jsonl

log = get_logger(__name__)


def _image_path(cosmos_dir: str | Path, img_local_path: str) -> str:
    return str(Path(cosmos_dir) / img_local_path.lstrip("./"))


def prepare_unlabeled_split(
    cosmos_dir: str | Path,
    split: str,
    out_path: str | Path,
    max_samples: int | None = None,
) -> int:
    """train or val -> one record per (image, article caption). No labels."""
    if split not in ("train", "val"):
        raise ValueError("prepare_unlabeled_split is for 'train' or 'val' only")

    ann_path = Path(cosmos_dir) / f"{split}_data.json"
    if not ann_path.exists():
        raise FileNotFoundError(
            f"COSMOS {split}_data.json not found at {ann_path} (see docs/DATASETS.md)."
        )
    with open(ann_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    records = []
    for i, entry in enumerate(entries):
        image_path = _image_path(cosmos_dir, entry["img_local_path"])
        for j, article in enumerate(entry.get("articles", [])):
            if max_samples is not None and len(records) >= max_samples:
                break
            records.append(
                {
                    "id": f"cosmos_{split}_{i}_{j}",
                    "image_path": image_path,
                    "caption": article["caption"],
                    "source": "cosmos",
                }
            )
        if max_samples is not None and len(records) >= max_samples:
            break

    write_jsonl(records, out_path)
    log.info("COSMOS %s: wrote %d unlabeled records to %s", split, len(records), out_path)
    return len(records)


def _to_binary_label(raw) -> int:
    if isinstance(raw, str):
        return 1 if raw.strip().lower() in {"ooc", "1", "out-of-context"} else 0
    return int(bool(raw))


def _load_test_records(cosmos_dir: str | Path, max_samples: int | None = None) -> list[dict]:
    ann_path = Path(cosmos_dir) / "test_data.json"
    if not ann_path.exists():
        raise FileNotFoundError(
            f"COSMOS test_data.json not found at {ann_path} (see docs/DATASETS.md)."
        )
    with open(ann_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    records = []
    for i, entry in enumerate(entries):
        if max_samples is not None and len(records) >= max_samples:
            break
        records.append(
            {
                "id": f"cosmos_test_{i}",
                "image_path": _image_path(cosmos_dir, entry["img_local_path"]),
                "caption1": entry["caption1"],
                "caption2": entry["caption2"],
                "label": _to_binary_label(entry["label"]),
                "source": "cosmos",
            }
        )
    return records


def prepare_test(
    cosmos_dir: str | Path,
    out_path: str | Path,
    max_samples: int | None = None,
) -> int:
    """test -> one record per image, preserving the native two-caption pair.

    Writes the FULL labeled pool, unsplit. For Stage 2 training + held-out
    evaluation, use prepare_all's train/val/test split instead — training
    and testing on overlapping records here would silently inflate results.
    """
    records = _load_test_records(cosmos_dir, max_samples)
    write_jsonl(records, out_path)
    log.info("COSMOS test: wrote %d labeled records to %s", len(records), out_path)
    return len(records)


def split_labeled(
    records: list[dict], seed: int = 42, train_frac: float = 0.7, val_frac: float = 0.15
) -> tuple[list[dict], list[dict], list[dict]]:
    """Stratified 70/15/15 split by label. ~1,700 total records only, so
    this is the one place the split happens — see prepare_all."""
    from sklearn.model_selection import train_test_split

    labels = [r["label"] for r in records]
    train, rest = train_test_split(
        records, train_size=train_frac, random_state=seed, stratify=labels
    )
    rest_labels = [r["label"] for r in rest]
    val_of_rest = val_frac / (1 - train_frac)
    val, test = train_test_split(
        rest, train_size=val_of_rest, random_state=seed, stratify=rest_labels
    )
    return train, val, test


def prepare_all(
    cosmos_dir: str | Path,
    out_dir: str | Path,
    max_samples: int | None = None,
    seed: int = 42,
) -> dict[str, int]:
    """Writes unlabeled Stage 1 splits, plus a stratified 70/15/15 split of
    the ~1,700 labeled test images into train/val/test for Stage 2 and
    final evaluation. These three come from ONE small pool, so the split is
    seeded and done exactly once here — do not re-split ad hoc elsewhere,
    or different scripts will silently draw overlapping data.
    """
    out_dir = Path(out_dir)
    counts = {
        "train": prepare_unlabeled_split(cosmos_dir, "train", out_dir / "cosmos_train.jsonl", max_samples),
        "val": prepare_unlabeled_split(cosmos_dir, "val", out_dir / "cosmos_val.jsonl", max_samples),
    }

    all_labeled = list(_load_test_records(cosmos_dir, max_samples))
    train, val, test = split_labeled(all_labeled, seed=seed)
    write_jsonl(train, out_dir / "cosmos_labeled_train.jsonl")
    write_jsonl(val, out_dir / "cosmos_labeled_val.jsonl")
    write_jsonl(test, out_dir / "cosmos_labeled_test.jsonl")
    counts["labeled_train"] = len(train)
    counts["labeled_val"] = len(val)
    counts["labeled_test"] = len(test)
    log.info(
        "COSMOS labeled split (seed=%d): %d train / %d val / %d test",
        seed, len(train), len(val), len(test),
    )
    return counts
