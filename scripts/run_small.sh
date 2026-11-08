#!/usr/bin/env bash
# CLEANSPILL small tier: the frozen probe on six models, then the frozen scorer.
# Models and order are frozen in frozen/CLEANSPILL_PREREG.md. Run from anywhere; outputs go to results/.
# POOLS_DIR: the folder made by data/curate.py (one subfolder per text kind with pool.jsonl).
set -u
PY="${PYTHON:-python}"
POOLS_DIR="${POOLS_DIR:-data/pools}"
cd "$(dirname "$0")/.."
for spec in "allenai/OLMo-2-0425-1B olmo2-1b" "meta-llama/Llama-3.2-1B llama32-1b" "Qwen/Qwen3-0.6B-Base qwen3-0.6b" \
            "HuggingFaceTB/SmolLM2-1.7B smollm2-1.7b" "google/gemma-3-1b-pt gemma3-1b" "Qwen/Qwen3.5-0.8B-Base qwen35-0.8b"; do
  set -- $spec
  "$PY" frozen/cleanspill_probe.py --model "$1" --tag "$2" --data-dir "$POOLS_DIR" --out "results/cleanspill_$2.json" > "results/cleanspill_$2.log" 2>&1
  echo "[run_small] $2 rc=$? $(date '+%H:%M:%S')"
done
"$PY" frozen/sum_cleanspill.py results/cleanspill_olmo2-1b.json results/cleanspill_llama32-1b.json \
  results/cleanspill_qwen3-0.6b.json results/cleanspill_smollm2-1.7b.json results/cleanspill_gemma3-1b.json \
  results/cleanspill_qwen35-0.8b.json --out results/cleanspill_summary.json
echo "[run_small] scored rc=$? $(date '+%H:%M:%S')"
