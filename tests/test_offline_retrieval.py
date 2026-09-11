import json

from misinfodet.retrieval.evidence_retriever import OfflineCorpus


def test_offline_corpus_search(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    docs = [
        {"title": "G20 summit in Rio", "snippet": "World leaders met in Rio de Janeiro.", "url": "u1"},
        {"title": "Flooding in Durban", "snippet": "Heavy rains caused flooding in KwaZulu-Natal.", "url": "u2"},
    ]
    corpus.write_text("\n".join(json.dumps(d) for d in docs))
    oc = OfflineCorpus(corpus)
    hits = oc.search("Durban flooding rains", k=1)
    assert hits and hits[0]["url"] == "u2"
