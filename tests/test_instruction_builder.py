from misinfodet.data.instruction_builder import (
    build_stage1_examples,
    build_stage2_examples,
    build_stage2_examples_paired,
    parse_verdict,
)

RECORDS = [
    {"id": "a", "image_path": "img/a.jpg", "caption": "President at summit", "label": 0},
    {"id": "b", "image_path": "img/b.jpg", "caption": "Flood in city", "label": 1},
]

UNLABELED_RECORDS = [
    {"id": "c", "image_path": "img/c.jpg", "caption": "Mayor opens bridge"},
]

PAIRED_RECORDS = [
    {"id": "d", "image_path": "img/d.jpg", "caption1": "Flood in coastal town",
     "caption2": "Drought in farming region", "label": 1},
    {"id": "e", "image_path": "img/e.jpg", "caption1": "Summit opens", "caption2": "Summit opens", "label": 0},
]


def test_stage1_uses_only_pristine():
    ex = list(build_stage1_examples(RECORDS))
    assert len(ex) == 1
    assert ex[0]["id"] == "a"
    assert ex[0]["target"] == "President at summit"


def test_stage1_treats_unlabeled_records_as_usable():
    """COSMOS train/val have no 'label' key at all — must not crash or be
    silently dropped, since Stage 1 doesn't need OOC ground truth."""
    ex = list(build_stage1_examples(UNLABELED_RECORDS))
    assert len(ex) == 1
    assert ex[0]["target"] == "Mayor opens bridge"


def test_stage2_covers_both_classes():
    ex = list(build_stage2_examples(RECORDS))
    assert len(ex) == 2
    assert "out-of-context" in ex[1]["target"]
    assert "consistent" in ex[0]["target"]
    assert "Flood in city" in ex[1]["prompt"]


def test_stage2_paired_covers_both_classes():
    ex = list(build_stage2_examples_paired(PAIRED_RECORDS))
    assert len(ex) == 2
    assert "out-of-context" in ex[0]["target"]
    assert "consistent" in ex[1]["target"]
    assert "Flood in coastal town" in ex[0]["prompt"]
    assert "Drought in farming region" in ex[0]["prompt"]


def test_parse_verdict_template():
    assert parse_verdict("VERDICT: out-of-context\nREASONING: mismatch.") == 1
    assert parse_verdict("VERDICT: consistent\nREASONING: fine.") == 0


def test_parse_verdict_fallback_and_failure():
    assert parse_verdict("This pair looks fake, a clear mismatch.") == 1
    assert parse_verdict("The caption genuinely matches the image.") == 0
    assert parse_verdict("I cannot decide.") is None
