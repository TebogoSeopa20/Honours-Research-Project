"""LEMMA-style targeted query generation.

The model decides what it needs to know before searching, keeping retrieved
evidence focused rather than noisy (Xuan et al., 2024).
"""
from __future__ import annotations

from ..pipeline.prompts import QUERY_GENERATION_PROMPT
from ..utils import Config


def build_query_prompt(caption: str, initial_reasoning: str, n_queries: int = 3) -> str:
    return QUERY_GENERATION_PROMPT.format(
        caption=caption, initial_reasoning=initial_reasoning, n_queries=n_queries
    )


def parse_queries(raw: str, n_queries: int = 3, caption: str = "") -> list[str]:
    """Parse model output into clean query strings, with a safe fallback."""
    queries = []
    for line in raw.splitlines():
        line = line.strip().strip("-*").strip()
        # Strip accidental numbering like "1." or "2)"
        if line[:2].rstrip(".)").isdigit():
            line = line.split(".", 1)[-1].split(")", 1)[-1].strip()
        if line and len(line) > 5 and not line.lower().startswith(("here", "sure")):
            queries.append(line)
    if not queries and caption:
        queries = [caption[:120]]  # degenerate fallback: search the caption itself
    return queries[:n_queries]


def generate_queries(model, processor, caption: str, initial_reasoning: str, cfg: Config) -> list[str]:
    from ..models.model_factory import generate_text_only

    prompt = build_query_prompt(caption, initial_reasoning, n_queries=3)
    raw = generate_text_only(model, processor, prompt, cfg)
    return parse_queries(raw, n_queries=3, caption=caption)
