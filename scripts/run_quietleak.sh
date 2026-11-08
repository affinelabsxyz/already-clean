#!/usr/bin/env bash
# QUIETLEAK: the frozen probe on CLEANSPILL's units for each model, then the frozen scorer.
# Models and order are frozen in frozen/QUIETLEAK_PREREG.md. Needs results/cleanspill_<tag>.json from run_small.sh.
set -u
PY="${PYTHON:-python}"
POOLS_DIR="${POOLS_DIR:-data/pools}"
cd "$(dirname "$0")/.."
for spec in "allenai/OLMo-2-0425-1B olmo2-1b" "meta-llama/Llama-3.2-1B llama32-1b" "Qwen/Qwen3-0.6B-Base qwen3-0.6b" \
            "HuggingFaceTB/SmolLM2-1.7B smollm2-1.7b" "google/gemma-3-1b-pt gemma3-1b" "Qwen/Qwen3.5-0.8B-Base qwen35-0.8b"; do
  set -- $spec
  "$PY" frozen/quietleak_probe.py --model "$1" --tag "$2" --units "results/cleanspill_$2.json" --data-dir "$POOLS_DIR" \
    --out "results/quietleak_$2.json" > "results/quietleak_$2.log" 2>&1
  echo "[run_quietleak] $2 rc=$? $(date '+%H:%M:%S')"
done
"$PY" frozen/sum_quietleak.py --tags olmo2-1b llama32-1b qwen3-0.6b smollm2-1.7b gemma3-1b qwen35-0.8b \
  --out results/quietleak_summary.json
echo "[run_quietleak] scored rc=$? $(date '+%H:%M:%S')"
