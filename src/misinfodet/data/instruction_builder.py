"""Build instruction-tuning examples for the two SNIFFER-style stages.

Stage 1 (news-domain entity alignment): train the LVLM to describe news
images in terms of named entities, using matched (pristine) image-caption
pairs as weak supervision for entity-grounded description.

Stage 2 (out-of-context fine-tuning): train the LVLM to judge whether an
image-caption pair is consistent and to explain its reasoning, using both
pristine and falsified pairs.

Each builder yields dicts of {"image_path", "prompt", "target"} which the
training scripts collate into LLaVA conversation format.
"""
from __future__ import annotations

import random
from typing import Iterable, Iterator

from ..pipeline.prompts import (
    STAGE1_ALIGNMENT_PROMPTS,
    STAGE2_PAIRED_TARGET_FAKE,
    STAGE2_PAIRED_TARGET_REAL,
    STAGE2_TARGET_FAKE,
    STAGE2_TARGET_REAL,
    PAIRED_VERDICT_PROMPT,
    VERDICT_PROMPT,
)


def build_stage1_examples(
    records: Iterable[dict], seed: int = 42
) -> Iterator[dict]:
    """Entity alignment: only pristine pairs (label 0, or no label at all).

    The caption is treated as the entity-grounded description target. This
    teaches the model to move from generic descriptions ("a man in a suit")
    to news-specific ones ("President X at the G20 summit"), closing the
    domain gap identified by COSMOS and SNIFFER.

    Records with no "label" key (e.g. COSMOS train/val, which are entirely
    unlabeled real image-caption pairs) are treated as usable, same as an
    explicit label 0 — Stage 1 never needs OOC ground truth.
    """
    rng = random.Random(seed)
    for rec in records:
        if rec.get("label", 0) != 0:
            continue
        prompt = rng.choice(STAGE1_ALIGNMENT_PROMPTS)
        yield {
            "id": rec["id"],
            "image_path": rec["image_path"],
            "prompt": prompt,
            "target": rec["caption"],
        }


def build_stage2_examples(records: Iterable[dict]) -> Iterator[dict]:
    """OOC detection: both classes, verdict + short reasoning template.

    Targets follow a fixed structure (VERDICT: ... / REASONING: ...) so the
    verdict can be parsed deterministically at inference time.

    Expects single-caption records ({"caption", "label"}) — NewsCLIPpings/
    MMFakeBench shape. For COSMOS's paired-caption test schema, use
    build_stage2_examples_paired instead.
    """
    for rec in records:
        prompt = VERDICT_PROMPT.format(caption=rec["caption"])
        target_tpl = STAGE2_TARGET_FAKE if rec["label"] == 1 else STAGE2_TARGET_REAL
        yield {
            "id": rec["id"],
            "image_path": rec["image_path"],
            "prompt": prompt,
            "target": target_tpl,
        }


def build_stage2_examples_paired(records: Iterable[dict]) -> Iterator[dict]:
    """OOC detection for COSMOS's test schema: one image, two captions, one
    label for whether pairing them together is out-of-context.

    Expects records shaped {"caption1", "caption2", "label", ...}, as
    produced by misinfodet.data.cosmos.prepare_test.
    """
    for rec in records:
        prompt = PAIRED_VERDICT_PROMPT.format(
            caption1=rec["caption1"], caption2=rec["caption2"]
        )
        target_tpl = (
            STAGE2_PAIRED_TARGET_FAKE if rec["label"] == 1 else STAGE2_PAIRED_TARGET_REAL
        )
        yield {
            "id": rec["id"],
            "image_path": rec["image_path"],
            "prompt": prompt,
            "target": target_tpl,
        }


def parse_verdict(text: str) -> int | None:
    """Parse a generated response into a label. Returns None if unparseable.

    Robust to the model deviating slightly from the template: looks for the
    VERDICT line first, then falls back to keyword scanning.
    """
    lowered = text.lower()
    for line in lowered.splitlines():
        line = line.strip()
        if line.startswith("verdict"):
            if "out-of-context" in line or "fake" in line or "mismatch" in line:
                return 1
            if "consistent" in line or "real" in line or "match" in line:
                return 0
    # Fallback keyword scan over the whole response
    fake_hits = sum(k in lowered for k in ("out-of-context", "fake", "mismatch", "falsified"))
    real_hits = sum(k in lowered for k in ("consistent", "genuine", "real", "matches the image"))
    if fake_hits > real_hits:
        return 1
    if real_hits > fake_hits:
        return 0
    return None
