# Experiment log

Record every run here. The final report's results tables should trace back to
rows in this file plus the run_meta.json next to each prediction file.

| Date | Run name | Config | Dataset/split | n | Acc | Macro-F1 | Notes |
|---|---|---|---|---|---|---|---|
| | baseline | base.yaml | newsclippings/test | | | | zero-shot LLaVA-1.5-7B |
| | | | | | | | |

## Reference numbers from the literature (to compare against)

- SNIFFER (Qi et al., 2024): ~88.4% accuracy, NewsCLIPpings Merged/Balanced
- CLIP baseline: ~66% (from NewsCLIPpings paper)
- MMFakeBench zero-shot LVLMs: 25.7%-~60% F1 depending on model
- LEMMA gains: +7% Twitter, +13% Fakeddit over strongest LVLM baseline
