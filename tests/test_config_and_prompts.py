from misinfodet.pipeline.prompts import (
    EXPLANATION_JUDGE_PROMPT,
    FINAL_VERDICT_PROMPT,
    QUERY_GENERATION_PROMPT,
    VERDICT_PROMPT,
)
from misinfodet.utils import Config


def test_prompts_format_cleanly():
    assert "test caption" in VERDICT_PROMPT.format(caption="test caption")
    FINAL_VERDICT_PROMPT.format(caption="c", initial_reasoning="r", evidence="e")
    QUERY_GENERATION_PROMPT.format(caption="c", initial_reasoning="r", n_queries=3)
    out = EXPLANATION_JUDGE_PROMPT.format(
        caption="c", gold_label="consistent", pred_label="consistent", explanation="e"
    )
    assert '"factual_accuracy"' in out


def test_config_roundtrip(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("run_name: smoke\nuse_retrieval: true\nsome_future_key: 1\n")
    cfg = Config.from_yaml(p)
    assert cfg.run_name == "smoke" and cfg.use_retrieval is True
    assert cfg.extra == {"some_future_key": 1}
    cfg.dump(tmp_path / "out.yaml")
