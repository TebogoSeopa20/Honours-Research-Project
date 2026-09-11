"""Full detection pipeline: the unified system from Chapter 3 of the proposal.

Flow for one image-text pair:
  1. Internal cross-modal assessment with the (optionally tuned) LVLM.
  2. If use_retrieval: generate targeted queries from the initial reasoning,
     retrieve web evidence, and (if use_reranking) re-rank it.
  3. Final verdict combining image + initial reasoning + external evidence,
     with confidence and a human-readable explanation.

Every toggle in Config (use_instruction_tuning, use_retrieval, use_reranking,
generate_explanations) corresponds to one row of the ablation study, so the
exact same code path is used for every configuration.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from PIL import Image

from ..data.instruction_builder import parse_verdict
from ..models.model_factory import generate
from ..retrieval.evidence_retriever import EvidenceRetriever
from ..retrieval.query_generator import generate_queries
from ..retrieval.reranker import format_evidence, rerank
from ..utils import Config, get_logger
from .prompts import (
    FINAL_VERDICT_PROMPT,
    PAIRED_FINAL_VERDICT_PROMPT,
    PAIRED_VERDICT_PROMPT,
    VERDICT_PROMPT,
)

log = get_logger(__name__)


@dataclass
class DetectionResult:
    id: str
    pred_label: int  # 0 = consistent/real, 1 = out-of-context/fake
    gold_label: int | None
    confidence: str
    explanation: str
    initial_reasoning: str
    queries: list[str] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    raw_response: str = ""
    parse_failed: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "pred_label": self.pred_label,
            "gold_label": self.gold_label,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "initial_reasoning": self.initial_reasoning,
            "queries": self.queries,
            "evidence": self.evidence,
            "raw_response": self.raw_response,
            "parse_failed": self.parse_failed,
        }


def _extract_field(text: str, name: str) -> str:
    m = re.search(rf"{name}\s*:\s*(.+?)(?:\n[A-Z]+\s*:|\Z)", text, re.S | re.I)
    return m.group(1).strip() if m else ""


class Detector:
    def __init__(self, model, processor, cfg: Config):
        self.model = model
        self.processor = processor
        self.cfg = cfg
        self.retriever = EvidenceRetriever(cfg) if cfg.use_retrieval else None

    def detect(self, record: dict) -> DetectionResult:
        image = Image.open(record["image_path"]).convert("RGB")
        paired = "caption1" in record and "caption2" in record
        if paired:
            caption1, caption2 = record["caption1"], record["caption2"]
            # Used only for retrieval query generation, which needs one text
            # blob — not a task-format assumption, see query_generator.py.
            caption_for_retrieval = f'Caption 1: "{caption1}" | Caption 2: "{caption2}"'
            initial_prompt = PAIRED_VERDICT_PROMPT.format(caption1=caption1, caption2=caption2)
        else:
            caption = record["caption"]
            caption_for_retrieval = caption
            initial_prompt = VERDICT_PROMPT.format(caption=caption)

        # ------------------------------------------------------------------ #
        # Step 1: internal cross-modal assessment
        # ------------------------------------------------------------------ #
        initial_raw = generate(self.model, self.processor, image, initial_prompt, self.cfg)
        initial_reasoning = _extract_field(initial_raw, "REASONING") or initial_raw

        queries: list[str] = []
        evidence: list[dict] = []
        final_raw = initial_raw

        # ------------------------------------------------------------------ #
        # Step 2: targeted retrieval (+ optional re-ranking)
        # ------------------------------------------------------------------ #
        if self.cfg.use_retrieval and self.retriever is not None:
            queries = generate_queries(
                self.model, self.processor, caption_for_retrieval, initial_reasoning, self.cfg
            )
            evidence = self.retriever.retrieve(queries)
            if self.cfg.use_reranking:
                evidence = rerank(self.model, self.processor, caption_for_retrieval, evidence, self.cfg)
            else:
                evidence = evidence[: self.cfg.rerank_keep_k]

            # ------------------------------------------------------------- #
            # Step 3: evidence-grounded final verdict
            # ------------------------------------------------------------- #
            if paired:
                final_prompt = PAIRED_FINAL_VERDICT_PROMPT.format(
                    caption1=caption1,
                    caption2=caption2,
                    initial_reasoning=initial_reasoning,
                    evidence=format_evidence(evidence),
                )
            else:
                final_prompt = FINAL_VERDICT_PROMPT.format(
                    caption=caption,
                    initial_reasoning=initial_reasoning,
                    evidence=format_evidence(evidence),
                )
            final_raw = generate(self.model, self.processor, image, final_prompt, self.cfg)

        pred = parse_verdict(final_raw)
        parse_failed = pred is None
        if parse_failed:
            # Conservative default: unparseable output counts as the majority
            # class 0 and is flagged, so parse failures are visible in analysis
            # rather than silently boosting metrics.
            pred = 0
            log.warning("Unparseable verdict for %s", record.get("id"))

        confidence = _extract_field(final_raw, "CONFIDENCE").lower() or "unknown"
        explanation = (
            (_extract_field(final_raw, "REASONING") or final_raw)
            if self.cfg.generate_explanations
            else ""
        )

        return DetectionResult(
            id=record.get("id", ""),
            pred_label=pred,
            gold_label=record.get("label"),
            confidence=confidence,
            explanation=explanation,
            initial_reasoning=initial_reasoning,
            queries=queries,
            evidence=evidence,
            raw_response=final_raw,
            parse_failed=parse_failed,
        )
