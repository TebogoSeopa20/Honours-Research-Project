#!/usr/bin/env bash
# Phase 5: ablation study (Section 3.5 of the proposal).
# Each config differs from full_system.yaml by exactly one toggle.
set -euo pipefail
SPLIT="${1:-test}"

for CFG in configs/full_system.yaml \
           configs/ablations/no_instruction_tuning.yaml \
           configs/ablations/no_retrieval.yaml \
           configs/ablations/no_reranking.yaml \
           configs/ablations/no_explanations.yaml; do
  echo "=== Running $CFG on COSMOS $SPLIT ==="
  python scripts/run_inference.py --config "$CFG" --split "$SPLIT" --dataset cosmos --resume
  NAME=$(python - "$CFG" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["run_name"])
PY
)
  python scripts/evaluate.py --predictions "outputs/predictions/${NAME}_cosmos_${SPLIT}.jsonl"
done

# Generalisation: full system on MMFakeBench, zero training on it
echo "=== Full system on MMFakeBench (generalisation) ==="
python scripts/run_inference.py --config configs/full_system.yaml --split test --dataset mmfakebench --resume
python scripts/evaluate.py --predictions outputs/predictions/full_system_mmfakebench_test.jsonl

python -m misinfodet.evaluation.ablation_report
