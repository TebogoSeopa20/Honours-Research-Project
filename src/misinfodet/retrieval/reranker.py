"""LVLM-based evidence re-ranking (Tahmasebi et al., 2024).

Each retrieved snippet is scored 0-10 for probative value against the claim;
only the top rerank_keep_k snippets reach the final verdict prompt.
"""
from __future__ import annotations

import re

from ..pipeline.prompts import RERANK_PROMPT
from ..utils import Config


def parse_score(raw: str) -> float:
    m = re.search(r"\d+(?:\.\d+)?", raw)
    if not m:
        return 0.0
    return min(max(float(m.group()), 0.0), 10.0)


def rerank(model, processor, caption: str, snippets: list[dict], cfg: Config) -> list[dict]:
    if not snippets:
        return []
    from ..models.model_factory import generate_text_only

    scored = []
    for s in snippets:
        text = f"{s.get('title','')}\n{s.get('snippet','')}\nSource: {s.get('url','')}"
        prompt = RERANK_PROMPT.format(caption=caption, snippet=text)
        raw = generate_text_only(model, processor, prompt, cfg)
        scored.append((parse_score(raw), s))
    scored.sort(key=lambda x: x[0], reverse=True)
    kept = []
    for score, s in scored[: cfg.rerank_keep_k]:
        s = dict(s)
        s["rerank_score"] = score
        kept.append(s)
    return kept


def format_evidence(snippets: list[dict]) -> str:
    if not snippets:
        return "(no external evidence available)"
    blocks = []
    for i, s in enumerate(snippets, 1):
        blocks.append(
            f"[{i}] {s.get('title','')}\n{s.get('snippet','')}\nSource: {s.get('url','')}"
        )
    return "\n\n".join(blocks)
