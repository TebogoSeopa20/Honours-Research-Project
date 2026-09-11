"""Web evidence retrieval behind a provider-agnostic interface.

Providers:
  - serper: Google results via serper.dev (cheap, good for research use)
  - bing:   Bing Web Search API
  - offline: pre-indexed local corpus (for fully reproducible ablations and
             for running without API keys / network)

Every snippet is returned as {"title", "snippet", "url", "query"}.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import requests

from ..utils import Config, get_logger

log = get_logger(__name__)


def _search_serper(query: str, api_key: str, k: int) -> list[dict]:
    resp = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        json={"q": query, "num": k},
        timeout=20,
    )
    resp.raise_for_status()
    organic = resp.json().get("organic", [])[:k]
    return [
        {"title": r.get("title", ""), "snippet": r.get("snippet", ""), "url": r.get("link", "")}
        for r in organic
    ]


def _search_bing(query: str, api_key: str, k: int) -> list[dict]:
    resp = requests.get(
        "https://api.bing.microsoft.com/v7.0/search",
        headers={"Ocp-Apim-Subscription-Key": api_key},
        params={"q": query, "count": k},
        timeout=20,
    )
    resp.raise_for_status()
    pages = resp.json().get("webPages", {}).get("value", [])[:k]
    return [
        {"title": r.get("name", ""), "snippet": r.get("snippet", ""), "url": r.get("url", "")}
        for r in pages
    ]


class OfflineCorpus:
    """TF-IDF retrieval over a local JSONL corpus of {title, snippet, url}.

    Guarantees identical evidence across reruns, which matters for the
    ablation study and for the final report's reproducibility claims.
    """

    def __init__(self, corpus_path):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.docs = []
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.docs.append(json.loads(line))
        texts = [f"{d.get('title','')} {d.get('snippet','')}" for d in self.docs]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(texts) if texts else None

    def search(self, query: str, k: int) -> list[dict]:
        if self.matrix is None:
            return []
        from sklearn.metrics.pairwise import cosine_similarity

        sims = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        top = sims.argsort()[::-1][:k]
        return [self.docs[i] for i in top if sims[i] > 0]


class EvidenceRetriever:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.provider = cfg.search_provider
        self.api_key = os.environ.get(cfg.search_api_key_env, "")
        self.offline = None
        corpus = Path(cfg.data_dir) / "evidence_corpus.jsonl"
        if self.provider == "offline" or (not self.api_key and corpus.exists()):
            if corpus.exists():
                self.offline = OfflineCorpus(corpus)
                self.provider = "offline"
                log.info("Using offline evidence corpus at %s", corpus)

    def retrieve(self, queries: list[str]) -> list[dict]:
        per_query = max(1, self.cfg.retrieval_top_k // max(len(queries), 1))
        results, seen = [], set()
        for q in queries:
            try:
                if self.provider == "serper":
                    hits = _search_serper(q, self.api_key, per_query)
                elif self.provider == "bing":
                    hits = _search_bing(q, self.api_key, per_query)
                elif self.provider == "offline" and self.offline:
                    hits = self.offline.search(q, per_query)
                else:
                    hits = []
            except Exception as e:  # network failures must not kill a run
                log.warning("Retrieval failed for query %r: %s", q, e)
                hits = []
            for h in hits:
                key = h.get("url") or h.get("snippet", "")[:80]
                if key and key not in seen:
                    seen.add(key)
                    h["query"] = q
                    results.append(h)
        return results[: self.cfg.retrieval_top_k]
