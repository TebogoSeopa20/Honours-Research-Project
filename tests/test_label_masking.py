"""Confirms the collate_fn label mask isolates exactly the target text.

Everything else — image tokens, prompt tokens, padding — must be -100.
Downloads the LLaVA processor config on first run (a few MB, not the
14GB model weights). Skips cleanly if offline.
"""
from __future__ import annotations

import os
import tempfile

import pytest
from PIL import Image


def test_labels_isolate_target_text():
    from transformers import AutoProcessor

    from misinfodet.models.training import make_collate_fn

    try:
        processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    except Exception as e:
        pytest.skip(f"processor unavailable (offline?): {e}")

    target = "REASONING: test target text.\nVERDICT: consistent"

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "dummy.png")
        Image.new("RGB", (336, 336), color=(128, 128, 128)).save(path)
        example = {"image_path": path, "prompt": "Describe this image.", "target": target}
        batch = make_collate_fn(processor)([example])

    labels, input_ids = batch["labels"][0], batch["input_ids"][0]
    decoded = processor.tokenizer.decode(
        input_ids[labels != -100], skip_special_tokens=True
    )
    assert decoded.strip() == target.strip(), f"mask boundary is off: got {decoded!r}"
