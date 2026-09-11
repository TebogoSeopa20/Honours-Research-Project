from misinfodet.evaluation.metrics import classification_report, evaluate_predictions


def test_perfect_predictions():
    r = classification_report([0, 1, 0, 1], [0, 1, 0, 1])
    assert r["accuracy"] == 1.0 and r["macro_f1"] == 1.0


def test_report_shapes():
    r = classification_report([0, 0, 1, 1], [0, 1, 1, 1])
    assert r["n"] == 4
    assert len(r["confusion_matrix"]) == 2
    assert "out-of-context/fake" in r["per_class"]


def test_evaluate_predictions_with_forgery_types():
    preds = [
        {"id": "1", "gold_label": 0, "pred_label": 0, "parse_failed": False, "forgery_type": "original"},
        {"id": "2", "gold_label": 1, "pred_label": 1, "parse_failed": False, "forgery_type": "mismatch"},
        {"id": "3", "gold_label": 1, "pred_label": 0, "parse_failed": True, "forgery_type": "mismatch"},
    ]
    r = evaluate_predictions(preds)
    assert r["n"] == 3
    assert r["parse_failure_rate"] == round(1 / 3, 4)
    assert r["per_forgery_type"]["mismatch"]["n"] == 2
