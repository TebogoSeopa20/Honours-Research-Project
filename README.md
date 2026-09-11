# LVLM-Based Image-Text Misinformation Detection

Research codebase for the Wits BSc Honours (Big Data Analytics) project:
**"A Large Vision Language Model (LVLM)-Based System for Detecting Image-Text Misinformation"**
Tebogo Seopa (2563912), supervised by Dr Seun Olukanmi.

The system combines four components, each mapped to a gap identified in the
literature review:

| Component | Motivated by | Gap addressed |
|---|---|---|
| Two-stage instruction tuning (news-entity alignment, then OOC fine-tuning) | SNIFFER (Qi et al., 2024) | Domain gap in general LVLMs |
| Targeted query-driven evidence retrieval | LEMMA (Xuan et al., 2024) | LVLMs reason poorly without grounding |
| LVLM-based evidence re-ranking | LVLM4FV (Tahmasebi et al., 2024) | Noisy retrieval hurts accuracy |
| Explanation generation + explanation-quality evaluation | MOCHEG (Yao et al., 2023) | Explainability as first-class goal |

Primary benchmark: **COSMOS** (Aneja et al., 2021) — switched from NewsCLIPpings
after the VisualNews image archive proved impractical to download reliably
(see docs/DATASETS.md). Approved by Dr Olukanmi. Secondary, zero-training
generalisation benchmark: **MMFakeBench**, unchanged.

---

## Repository layout

```
configs/                  YAML configs for every experiment (incl. ablations)
scripts/                  Entry points: prepare data, train, infer, evaluate
src/misinfodet/
  data/                   Dataset loaders + instruction example builders
  models/                 LLaVA loading, LoRA wrapping
  retrieval/              Query generation, web evidence retrieval, re-ranking
  pipeline/               Full detector orchestration + prompt templates
  evaluation/             Classification metrics, explanation quality, ablation report
tests/                    CPU-only unit tests (run before every experiment)
docs/                     Experiment log template, dataset access notes
outputs/                  Checkpoints, predictions, logs (gitignored)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"         # installs the misinfodet package + pytest only
pytest tests/ -q                 # sanity check (no GPU needed)
```

Use `pip install -e ".[dev]"` instead if you also want Weights & Biases
tracking (needs a working Go + Rust toolchain to build from source on
platforms without a prebuilt wheel — this bit Intel Mac earlier).

GPU training uses 4-bit quantised LLaVA-1.5-7B + LoRA, which fits on a single
24 GB GPU (RTX 3090/4090, A10G, L4). Colab Pro / Kaggle T4x2 works for
inference and small runs; use gradient accumulation for training.

## Data

### COSMOS (primary)
Access is gated behind a form; the confirmation page links directly to the
four download files (no separate approval wait). See `docs/DATASETS.md` for
the exact link and file layout, then run:

```bash
python scripts/prepare_cosmos.py --config configs/base.yaml
```

This produces `cosmos_train.jsonl` / `cosmos_val.jsonl` (unlabeled,
`{id, image_path, caption, source}` — for Stage 1 only) and a stratified
70/15/15 split of the ~1,700 labeled test images into
`cosmos_labeled_train.jsonl` / `cosmos_labeled_val.jsonl` /
`cosmos_labeled_test.jsonl` (labeled, `{id, image_path, caption1, caption2,
label, source}` — note the two-caption shape, different from NewsCLIPpings/
MMFakeBench). Stage 2 trains on `cosmos_labeled_train.jsonl` only; baseline
and full-system evaluation both use `cosmos_labeled_test.jsonl`, so their
numbers stay comparable. **Only these ~1,700 images carry any label**, since
COSMOS's own method is self-supervised — this caps Stage 2 fine-tuning and
evaluation data well below the 35,536 NewsCLIPpings originally provided;
see docs/DATASETS.md for the full trade-off already discussed with Dr
Olukanmi.

### NewsCLIPpings (superseded, kept for reference)
NewsCLIPpings is built on top of **VisualNews** (a 91GB single-file
archive), which repeatedly failed to download intact. If ever revisited:

1. Request VisualNews: https://github.com/FuxiaoLiu/VisualNews-Repository
2. Download NewsCLIPpings annotation JSONs: https://github.com/g-luo/news_clippings
3. Place them as described in `docs/DATASETS.md`, then run:

```bash
python scripts/prepare_newsclippings.py --config configs/base.yaml
```

### MMFakeBench (secondary, evaluation only)
```bash
python scripts/prepare_mmfakebench.py --config configs/base.yaml
```
Follow `docs/DATASETS.md` for the download link (HuggingFace). No training is
ever done on MMFakeBench, matching the proposal.

## Experiment pipeline (matches Chapters 3–4 of the proposal)

```bash
# Phase 2: zero-shot baseline
python scripts/run_baseline.py --config configs/base.yaml --split test

# Phase 3: Stage 1 news-domain entity alignment
python scripts/train_stage1.py --config configs/stage1_entity_alignment.yaml

# Phase 4: Stage 2 OOC fine-tuning (loads Stage 1 adapter)
python scripts/train_stage2.py --config configs/stage2_ooc_finetune.yaml

# Full-system inference (tuned model + retrieval + re-rank + explanations)
python scripts/run_inference.py --config configs/full_system.yaml --split test

# Evaluation: classification metrics + explanation quality + report
python scripts/evaluate.py --predictions outputs/predictions/full_system_test.jsonl

# Phase 5: ablations (no-tuning / no-retrieval / no-rerank / no-explanation)
bash scripts/run_ablations.sh
```

Every run writes a timestamped JSONL of predictions plus a `run_meta.json`
capturing the exact config, git commit, and seed, so results in the final
report are fully reproducible. Weights & Biases logging is on by default
(`wandb: true` in configs) and can be disabled.

## Evaluation framework

- **Classification**: macro-F1, precision, recall, accuracy, per-class breakdown,
  confusion matrix (`misinfodet.evaluation.metrics`).
- **Generalisation**: identical inference on MMFakeBench with zero training.
- **Explanation quality**: automated LLM-as-judge rubric (factual accuracy,
  coherence, relevance, 1–5 each) plus export of a stratified sample to CSV
  for the small human evaluation study (`misinfodet.evaluation.explanation_quality`).
- **Ablations**: `scripts/run_ablations.sh` runs the four configurations from
  Section 3.5 of the proposal and `evaluation/ablation_report.py` builds a
  LaTeX-ready comparison table.

## Reproducibility and academic integrity

- All randomness is seeded via `misinfodet.utils.set_seed`.
- Remember to update the Wits AI declaration for the final research report:
  this codebase is AI-assisted, so items **3 (Methods and Experiment Design)**,
  **4 (Data Analysis)** and **6 (Code Development)** should be ticked, and the
  methodology chapter should state how generative AI was used, as the
  declaration requires.

## Citation anchors

Baseline numbers from the literature, reported on **NewsCLIPpings**
(CLIP ~66%, SNIFFER ~88.4% accuracy) and **MMFakeBench** zero-shot
(strongest LVLMs ~50–60% F1, weakest ~25.7%) are **no longer a direct,
apples-to-apples comparison** now that COSMOS is primary — those papers
don't report on COSMOS. Comparison strategy going forward: the ablation
study (Section 3.5) becomes the primary evidence for each component's
contribution, and COSMOS's own reported self-supervised numbers (Aneja et
al., 2021) are the closest available external reference point on this
dataset. Confirm this framing with Dr Olukanmi before writing up results.
Log every reproduced number in `docs/EXPERIMENT_LOG.md` regardless.
