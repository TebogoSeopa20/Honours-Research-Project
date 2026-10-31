# LVLM-based image-text misinformation detection

Code for my Wits BSc Honours (Big Data Analytics) project, *A Large Vision Language Model (LVLM)-Based System for Detecting Image-Text Misinformation*.
Tebogo Seopa (2563912), supervised by Dr Seun Olukanmi.

Short version: I fine-tuned LLaVA-1.5-7B (4-bit, LoRA) on a free Kaggle GPU to spot out-of-context news photos, meaning a real photo paired with a caption from a different event.

## What actually worked

- **Final system:** base LLaVA plus Stage 1 news-domain tuning. It gets 83.2% accuracy and 0.910 AUROC on the COSMOS test split.
- **The big win was how I read the answer, not the training.** Parsing the model's written verdict gave 50%, basically a coin flip. Reading its probability for "out-of-context" and picking the cut-off on validation data (0.81) gave 78.5% with no training at all.
- **What didn't help:** Stage 2 task tuning (only 1,189 labelled examples) and the built-in web search pipeline.
- **Caveat:** it mostly reads the captions. Swapping every photo for a grey square barely changes the score.

The full write-up is in the report. The proposal is `Research_Proposal.pdf`.

## What's in here

```
configs/      one YAML per run (diag_stage1_only.yaml is the final system)
scripts/      one script per experiment
src/misinfodet/
  data/       COSMOS, MMFakeBench, NewsCLIPpings loaders
  models/     LLaVA loading, LoRA, training loop
  retrieval/  query generation, web search, re-ranking
  pipeline/   detector + all the prompts
  evaluation/ metrics, explanation checks
tests/        unit tests, CPU only
docs/         dataset notes and my experiment log
Proof_of_Research_Notebook.ipynb   the Kaggle notebook I ran everything in
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
pytest tests/ -q
```

Everything ran on Kaggle's free T4 (16 GB). On Kaggle you'll also need:

```bash
pip install -U "bitsandbytes>=0.46.1" peft
```

## Data

- **COSMOS** is the main dataset. You fill in a form and get the download links straight away. `docs/DATASETS.md` has the details. Only about 1,700 images are labelled, and I split those 70/15/15 into train, validation and test.
- **MMFakeBench** is test only. I never trained on it. The download is on Hugging Face.
- **NewsCLIPpings** was the original plan, but its images come as one 91 GB file I couldn't download in one piece. I only ran a CLIP baseline on it, using the embeddings the authors released.

```bash
python scripts/prepare_cosmos.py --config configs/base.yaml
python scripts/prepare_mmfakebench.py --config configs/base.yaml
```

## Running things

Training:

```bash
python scripts/train_stage1.py --config configs/stage1_entity_alignment.yaml
python scripts/train_stage2.py --config configs/stage2_ooc_finetune_lmonly.yaml
```

Main results (calibrated scoring; picks the cut-off on validation, then scores test once):

```bash
python scripts/calibrate_threshold.py --config configs/diag_stage1_only.yaml
python scripts/calibrate_threshold.py --config configs/diag_stage1_only.yaml --blank-image
python scripts/calibrate_mmfakebench.py --config configs/diag_stage1_only.yaml
```

The rest of the checks:

| What | Script |
|---|---|
| significance (paired bootstrap) | `compare_runs.py` |
| train/test overlap | `check_leakage.py` |
| CLIP baseline | `clip_baseline.py` |
| SBERT baseline, explanation checks | `final_checks.py` |
| separate questions on MMFakeBench | `hierarchical_mmfakebench.py`, `fuse_mmfakebench.py` |
| calibration and evidence fusion | `fusion_calibration.py` |
| web search pipeline | `calibrate_retrieval.py`, `rescore_retrieval.py` |
| explanations | `explain_verdicts.py` |

The old free-text scripts (`run_baseline.py`, `run_inference.py`, `evaluate.py`) still work. They're how I found the 50% problem.

## Gotchas

- If Stage 2 seems to do nothing, check you've got the fixed `model_factory.py`. An older version merged Stage 1 before attaching Stage 2, which silently zeroed Stage 2's weights.
- On Kaggle, set `PYTHONUNBUFFERED=1` or the progress output looks frozen.
- Kaggle sessions die after 12 hours, so back up anything in `outputs/` to a Kaggle dataset.
