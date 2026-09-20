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

Real data check (18,000 images, 360,758 raw records) found ~26% of these
records are EXACT (image, caption) duplicates — the same syndicated wire
caption appearing under several articles pointing at the same photo.
prepare_unlabeled_split drops exact repeats while keeping every genuinely
different caption for the same image, since that's real diversity worth
keeping, not waste.

test_data.json: one image with exactly TWO candidate captions (caption1,
caption2) and a single label for whether pairing them with this image is
out-of-context. This is a genuinely different task shape from
NewsCLIPpings/MMFakeBench's one-image-one-caption schema — it is NOT
flattened into two separate single-caption records here, since the label
describes the pair, not either caption individually.
Schema: {"id", "image_path", "caption1", "caption2", "label", "source"}
label: 0 = not out-of-context, 1 = out-of-context.

Despite the ".json" extension, these annotation files are actually JSONL —
one complete JSON object per line, not a single wrapping array. Confirmed
directly against real downloaded files, not assumed.

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


def _read_jsonl(path: Path) -> list[dict]:
    """COSMOS's *_data.json files are JSONL despite the extension — one
    JSON object per line, not a single array. Do not swap this for a plain
    json.load(f); that fails with "Extra data" past the first line."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _image_path(cosmos_dir: str | Path, img_local_path: str) -> str:
    return str(Path(cosmos_dir) / img_local_path.lstrip("./"))


def _get_spacy_nlp():
    """Loaded lazily and only when min_entity_freq > 1 actually needs it —
    the default path (min_entity_freq=1) has no spaCy dependency at all."""
    import spacy
    try:
        return spacy.load(
            "en_core_web_sm",
            disable=["lemmatizer", "tagger", "parser", "attribute_ruler"],
        )
    except OSError as e:
        raise RuntimeError(
            "min_entity_freq > 1 needs spaCy's English model. Install with:\n"
            "  pip install spacy\n"
            "  python -m spacy download en_core_web_sm"
        ) from e


def prepare_unlabeled_split(
    cosmos_dir: str | Path,
    split: str,
    out_path: str | Path,
    max_samples: int | None = None,
    seed: int = 42,
    min_entity_freq: int = 1,
) -> int:
    """train or val -> one record per (image, article caption). No labels.

    When max_samples caps the output, images are shuffled first (seeded,
    reproducible) so the subset is a random draw across the whole split,
    not just whichever images happen to come first in the annotation file.

    Exact (image, caption) duplicates are dropped — confirmed to be ~26%
    of raw records, almost all syndicated captions repeated verbatim
    across several articles about the same photo. A genuinely different
    caption for the same image is kept; only the exact repeat is waste.

    min_entity_freq > 1 filters to captions containing at least one PERSON
    named at least that many times across the WHOLE split (via spaCy NER,
    not a crude capitalized-word heuristic — an earlier bigram-matching
    version was confirmed to false-match phrases like "House Intelligence
    Committee" and "Alexander the Great" as if they were repeated people,
    which let ~77% of captions through regardless of the threshold set).
    Real data check found 44-74% of named people appear exactly once — no
    amount of training teaches a one-shot face-to-name mapping, so a
    uniform random sample mostly wastes budget on unlearnable examples.
    Filtering first concentrates a fixed training budget on people the
    model has a real chance of learning from repeated exposure, at the
    same total example count and GPU cost as an unfiltered sample.
    """
    if split not in ("train", "val"):
        raise ValueError("prepare_unlabeled_split is for 'train' or 'val' only")

    ann_path = Path(cosmos_dir) / f"{split}_data.json"
    if not ann_path.exists():
        raise FileNotFoundError(
            f"COSMOS {split}_data.json not found at {ann_path} (see docs/DATASETS.md)."
        )
    entries = _read_jsonl(ann_path)

    if min_entity_freq > 1:
        nlp = _get_spacy_nlp()

        # Flatten to (entry_idx, article_idx, caption) for one efficient
        # batch NER pass — calling spaCy per-caption in a loop is much
        # slower than nlp.pipe() over the whole list at once.
        flat = [
            (i, j, article["caption"])
            for i, entry in enumerate(entries)
            for j, article in enumerate(entry.get("articles", []))
        ]
        captions = [c for _, _, c in flat]

        person_ents_by_index: list[list[str]] = []
        for doc in nlp.pipe(captions, batch_size=200):
            person_ents_by_index.append(
                [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
            )

        def _surname_key(name: str) -> str:
            # News captions vary how they refer to the same person across
            # articles ("Donald Trump" / "President Trump" / "Mr. Trump") —
            # counting exact spans undercounts real repeats. Last token is
            # a rough but effective normalization for this pattern.
            return name.split()[-1].lower()

        person_counts: dict[str, int] = {}
        for persons in person_ents_by_index:
            for name in persons:
                key = _surname_key(name)
                person_counts[key] = person_counts.get(key, 0) + 1

        # entry_idx -> does ANY of its captions name a person appearing
        # min_entity_freq+ times anywhere in the split?
        entry_qualifies = [False] * len(entries)
        for (i, _j, _c), persons in zip(flat, person_ents_by_index):
            if entry_qualifies[i]:
                continue
            if any(person_counts.get(_surname_key(name), 0) >= min_entity_freq for name in persons):
                entry_qualifies[i] = True

        before = len(entries)
        entries = [e for i, e in enumerate(entries) if entry_qualifies[i]]
        log.info(
            "COSMOS %s: min_entity_freq=%d (spaCy PERSON) kept %d/%d images "
            "with a repeated-person caption", split, min_entity_freq, len(entries), before,
        )

    if max_samples is not None:
        import random
        random.Random(seed).shuffle(entries)

    records = []
    seen_pairs = set()
    for i, entry in enumerate(entries):
        image_path = _image_path(cosmos_dir, entry["img_local_path"])
        for j, article in enumerate(entry.get("articles", [])):
            if max_samples is not None and len(records) >= max_samples:
                break
            pair_key = (image_path, article["caption"])
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
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
    """Loads COSMOS's labeled test_data.json.

    A handful of entries reference images that don't actually exist in the
    image archive — confirmed case: "test/231.jpg:small", a stray macOS
    artifact file present in the ORIGINAL COSMOS release's images_test.zip,
    which got legitimately excluded when that archive was rebuilt to fix a
    forbidden-character upload error (see docs/DATASETS.md), but the
    annotation file was never regenerated to match. Any record whose image
    doesn't exist on disk is skipped here rather than crashing mid-training.
    """
    ann_path = Path(cosmos_dir) / "test_data.json"
    if not ann_path.exists():
        raise FileNotFoundError(
            f"COSMOS test_data.json not found at {ann_path} (see docs/DATASETS.md)."
        )
    entries = _read_jsonl(ann_path)

    records = []
    skipped_missing_image = 0
    for i, entry in enumerate(entries):
        if max_samples is not None and len(records) >= max_samples:
            break
        image_path = _image_path(cosmos_dir, entry["img_local_path"])
        if not Path(image_path).exists():
            skipped_missing_image += 1
            continue
        # Real field is "context_label", not "label" as originally assumed —
        # confirmed against actual downloaded data. Fallback to "label" kept
        # in case either name shows up across different COSMOS releases.
        if "context_label" in entry:
            label_raw = entry["context_label"]
        elif "label" in entry:
            label_raw = entry["label"]
        else:
            raise KeyError(
                f"No 'context_label' or 'label' field on entry {entry.get('img_local_path')}"
            )
        records.append(
            {
                "id": f"cosmos_test_{i}",
                "image_path": image_path,
                "caption1": entry["caption1"],
                "caption2": entry["caption2"],
                "label": _to_binary_label(label_raw),
                "source": "cosmos",
            }
        )
    if skipped_missing_image:
        log.info(
            "COSMOS test: skipped %d records with missing image files "
            "(known artifact, see docstring)", skipped_missing_image,
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
    min_entity_freq: int = 1,
) -> dict[str, int]:
    """Writes unlabeled Stage 1 splits, plus a stratified 70/15/15 split of
    the ~1,700 labeled test images into train/val/test for Stage 2 and
    final evaluation. These three come from ONE small pool, so the split is
    seeded and done exactly once here — do not re-split ad hoc elsewhere,
    or different scripts will silently draw overlapping data.
    """
    out_dir = Path(out_dir)
    counts = {
        "train": prepare_unlabeled_split(
            cosmos_dir, "train", out_dir / "cosmos_train.jsonl", max_samples,
            seed=seed, min_entity_freq=min_entity_freq,
        ),
        "val": prepare_unlabeled_split(
            cosmos_dir, "val", out_dir / "cosmos_val.jsonl", max_samples,
            seed=seed, min_entity_freq=min_entity_freq,
        ),
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