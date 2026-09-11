import json

from misinfodet.data.cosmos import prepare_test, prepare_unlabeled_split, split_labeled

TRAIN_ENTRY = [
    {
        "img_local_path": "train/1.jpg",
        "articles": [
            {"caption": "President visits factory", "article_url": "http://a"},
            {"caption": "Factory tour by the president", "article_url": "http://b"},
        ],
    }
]

TEST_ENTRY = [
    {
        "img_local_path": "test/1.jpg",
        "caption1": "Flood hits coastal town",
        "caption2": "Drought hits farming region",
        "label": 1,
    },
    {
        "img_local_path": "test/2.jpg",
        "caption1": "Summit opens in Geneva",
        "caption2": "Summit opens in Geneva",
        "label": "not-ooc",
    },
]


def test_unlabeled_split_expands_articles(tmp_path):
    (tmp_path / "train_data.json").write_text(json.dumps(TRAIN_ENTRY))
    out = tmp_path / "out.jsonl"
    n = prepare_unlabeled_split(tmp_path, "train", out)
    records = [json.loads(l) for l in out.read_text().splitlines()]
    assert n == 2
    assert "label" not in records[0]
    assert records[0]["caption"] == "President visits factory"
    assert records[1]["image_path"].endswith("train/1.jpg")


def test_test_split_keeps_two_captions_and_numeric_label(tmp_path):
    (tmp_path / "test_data.json").write_text(json.dumps(TEST_ENTRY))
    out = tmp_path / "out.jsonl"
    n = prepare_test(tmp_path, out)
    records = [json.loads(l) for l in out.read_text().splitlines()]
    assert n == 2
    assert records[0]["caption1"] == "Flood hits coastal town"
    assert records[0]["caption2"] == "Drought hits farming region"
    assert records[0]["label"] == 1
    assert records[1]["label"] == 0  # "not-ooc" string correctly parsed


def test_split_labeled_is_stratified_and_covers_everything():
    records = [
        {"id": f"r{i}", "label": 1 if i % 3 == 0 else 0} for i in range(60)
    ]
    train, val, test = split_labeled(records, seed=42)
    assert len(train) + len(val) + len(test) == 60
    # no overlap between splits
    ids = [r["id"] for r in train + val + test]
    assert len(ids) == len(set(ids))
    # roughly 70/15/15
    assert 38 <= len(train) <= 46
    assert 5 <= len(val) <= 13
    assert 5 <= len(test) <= 13
