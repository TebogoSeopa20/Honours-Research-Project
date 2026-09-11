from misinfodet.retrieval.query_generator import parse_queries
from misinfodet.retrieval.reranker import parse_score


def test_parse_queries_strips_numbering():
    raw = "1. President X G20 2024 summit location\n2) flood city name date\n- third query here"
    qs = parse_queries(raw, n_queries=3)
    assert len(qs) == 3
    assert qs[0].startswith("President X")


def test_parse_queries_fallback_to_caption():
    qs = parse_queries("", caption="Some caption text about an event")
    assert qs == ["Some caption text about an event"]


def test_parse_score():
    assert parse_score("8") == 8.0
    assert parse_score("Score: 7.5/10") == 7.5
    assert parse_score("irrelevant") == 0.0
    assert parse_score("15") == 10.0  # clamped
