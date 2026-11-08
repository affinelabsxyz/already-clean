#!/usr/bin/env bash
# RLLEAK: the frozen probe on 19 GRPO-LoRA runs, then the frozen scorer. Runs and order are frozen in
# frozen/RLLEAK_PREREG.md. RUNS_DIR: a folder with one subfolder per run, each holding adapter/ (see README).
set -u
PY="${PYTHON:-python}"
POOLS_DIR="${POOLS_DIR:-data/pools}"
export RUNS_DIR="${RUNS_DIR:-runs_rl}"
cd "$(dirname "$0")/.."
R="$PY scripts/rlleak_run.py"
$R --model allenai/OLMo-2-0425-1B --tag olmo2-1b --data-dir "$POOLS_DIR" --out results/rlleak_olmo2-1b.json \
  --runs sf_olmo2-1b_early_s0 sf_olmo2-1b_early_s1 sf_olmo2-1b_late_s0 sf_olmo2-1b_late_s1 > results/rlleak_olmo2-1b.log 2>&1
echo "[run_rlleak] olmo2-1b rc=$? $(date '+%H:%M:%S')"
$R --model Qwen/Qwen3-0.6B-Base --tag qwen3-0.6b --data-dir "$POOLS_DIR" --out results/rlleak_qwen3-0.6b.json \
  --runs sf_qwen3-0.6b_early_s0 sf_qwen3-0.6b_early_s1 sf_qwen3-0.6b_late_s0 sf_qwen3-0.6b_late_s1 > results/rlleak_qwen3-0.6b.log 2>&1
echo "[run_rlleak] qwen3-0.6b rc=$? $(date '+%H:%M:%S')"
$R --model Qwen/Qwen3.5-0.8B-Base --tag qwen35-0.8b --data-dir "$POOLS_DIR" --out results/rlleak_qwen35-0.8b.json \
  --runs math_easy e3_floors0-7_s0 e3_floors0-7_s1 e3_floors8-15_s0 e3_floors8-15_s1 e3_floors16-23_s0 e3_floors16-23_s1 \
         sf_qwen35-0.8b_early_s0 sf_qwen35-0.8b_early_s1 sf_qwen35-0.8b_late_s0 sf_qwen35-0.8b_late_s1 > results/rlleak_qwen35-0.8b.log 2>&1
echo "[run_rlleak] qwen35-0.8b rc=$? $(date '+%H:%M:%S')"
"$PY" frozen/sum_rlleak.py --tags olmo2-1b qwen3-0.6b qwen35-0.8b --out results/rlleak_summary.json
echo "[run_rlleak] scored rc=$? $(date '+%H:%M:%S')"
