# already-clean

Code, pre-registrations and results behind the Affine Labs note
[We tested Nick Cammarata's "Neural networks are already clean" on modern models](https://affinelabs.xyz/research/already-clean/).

**The question.** If you edit one MLP unit, does a cleaner unit change less of everything else?

**The answer, in six open models up to 1.7B.** Yes. Clean units cause 4–9× less off-target change at the same edit
size. A forward-only "leak" score, how much a unit responds where it should be off, predicts that side effect best.

## Results

Three pre-registered studies, all run in fp32 on one GPU. The verdicts are the ones the frozen scorers print.

| Study | Question | Verdict as frozen |
|---|---|---|
| CLEANSPILL | Does a unit's cleanliness, or its depth, set an edit's collateral? | **H2 CLEAN-HELPS in 6 of 6 models** (partial ρ −0.64 to −0.75). H1 and H3 (depth, residual growth) fail. |
| QUIETLEAK | Does "clean helps" come from one meaning, or from being quiet elsewhere? | **A: SEMANTIC, 6 of 6.** Our prediction (QUIET) was wrong. **B1: LEAK-ADDS, 6 of 6** (leak ρ +0.73 to +0.84). B2: no. |
| RLLEAK | In 19 short maths RL runs, does the off-domain part of the update sit on leaky units? | **C2: NO**, the 2× bar was missed. The order leaky > random > clean holds in 19 of 19 runs: the maths part sits on clean units, about 2:1. |

Full write-ups, with every number and caveat: [`claims/`](claims/).

Models: OLMo-2-0425-1B, Llama-3.2-1B, Qwen3-0.6B-Base, Qwen3.5-0.8B-Base, SmolLM2-1.7B, Gemma-3-1B-pt.
RLLEAK uses 19 existing GRPO-LoRA runs on OLMo-2-1B (4), Qwen3-0.6B (4) and Qwen3.5-0.8B (11).

## What was frozen, and when

Each study's design, probe and scorer were frozen before any scored run. Every result JSON records the sha256 of the
probe that wrote it.

| Study | Frozen | Pre-registration section | Probe | Scorer |
|---|---|---|---|---|
| CLEANSPILL | 2026-10-22, about 02:45 KST | `0fbf0dca…` (99 lines) | `ed8a7d67…` | `4a76138f…` |
| QUIETLEAK | 2026-10-22, before the 08:56 KST run | `30075e63…` (72 lines) | `bc0e8251…` | `e9ba0e19…` |
| RLLEAK | 2026-10-22, before the 09:39 KST run | `b96f8551…` (61 lines) | `0fbfc4d7…` | `8ab186a4…` |

Check them:

```
python scripts/check_hashes.py
```

Things to know:

- **The timestamps are our record.** The pre-registrations were committed to our private research repository before
  each scored run. That history is not public, so you cannot check the times yourself. You can check that the files
  here are the ones the results came from: each result JSON stores its probe's sha256.
- **Line endings matter.** Four scripts were hashed with Windows line endings (CRLF) and two with LF. `frozen/` keeps
  each file in the form it was hashed in, and `.gitattributes` stops git from converting them. Pre-registration
  sections are hashed by `frozen/prereg_hash.py`, which normalises line endings.
- **The frozen files are unchanged, so they contain the original machine's paths** (the default `--data-dir`, and
  `RUNS_DIR` in `rlleak_probe.py`). The run scripts pass your own paths instead.
- **The pre-registrations are as written**, including our study names and a few names of other internal studies.
- Smoke runs, which checked the plumbing before freezing, are listed in each pre-registration and not included here.

## Re-score without a GPU

The per-model results are included. The frozen scorers rebuild the three summaries from them; the output is
identical to the files in `results/`:

```
python frozen/sum_cleanspill.py results/cleanspill_olmo2-1b.json results/cleanspill_llama32-1b.json \
  results/cleanspill_qwen3-0.6b.json results/cleanspill_smollm2-1.7b.json results/cleanspill_gemma3-1b.json \
  results/cleanspill_qwen35-0.8b.json --out results/cleanspill_summary.json
python frozen/sum_quietleak.py --tags olmo2-1b llama32-1b qwen3-0.6b smollm2-1.7b gemma3-1b qwen35-0.8b --out results/quietleak_summary.json
python frozen/sum_rlleak.py --tags olmo2-1b qwen3-0.6b qwen35-0.8b --out results/rlleak_summary.json
git diff --stat results/   # nothing should change
```

Run these from the repository root. The per-model JSONs also hold the per-unit values (cleanliness, leak, edit
collateral, intended effect), if you want to plot them yourself.

## Re-run from scratch

1. `pip install -r requirements.txt`. Llama-3.2 and Gemma-3 are gated on Hugging Face, so accept their licences first.
2. Build the seven text pools (maths easy and hard, code, instruction following, logic, medical, finance). WikiText is
   loaded directly.
   ```
   python data/curate.py --out data/pools
   ```
   `frozen/CLEANSPILL_text_pools.sha256` has the sha256 of the first 4,000 lines of each `pool.jsonl` we used. The
   source datasets can change over time; if yours differ, the run still works, but say so when you report it.
3. Run the studies. Outputs go to `results/`, so `git diff` shows what changed.
   ```
   bash scripts/run_small.sh        # CLEANSPILL, about 10–17 minutes per model on an RTX 5070 Ti
   bash scripts/run_quietleak.sh    # QUIETLEAK, about 40 minutes in total; needs CLEANSPILL's results
   RUNS_DIR=runs_rl bash scripts/run_rlleak.sh   # RLLEAK, needs the LoRA adapters (below)
   ```
   Set `PYTHON` to choose the interpreter and `POOLS_DIR` if the pools are not in `data/pools`.

RLLEAK needs the 19 LoRA adapters. They are too large for git, so they are in a
[Google Drive folder](https://drive.google.com/drive/folders/1BSMQwRCX6Xy6_q8aFueuQQwYCSoPxqKG?usp=sharing):
`already-clean_rlleak_adapters.zip` (641 MB, sha256 `d93de611…`). Unzip it in the repository root to get
`runs_rl/<run>/adapter/`. `data/adapters_SHA256SUMS.txt` lists the hash of every file inside.

## Files

| Path | What |
|---|---|
| `frozen/` | Pre-registrations, probes and scorers exactly as frozen, `prereg_hash.py`, and the text-pool hashes |
| `scripts/` | Run scripts, `rlleak_run.py` (runs the frozen RLLEAK probe with your adapter folder), `check_hashes.py` |
| `results/` | Per-model results and the three summaries |
| `claims/` | The write-up of each study's result |
| `data/curate.py` | Rebuilds the text pools from public datasets |
| `data/adapters_SHA256SUMS.txt` | Hashes of the RLLEAK adapter files in the Drive zip |

## Not included

- The text pools themselves: their source datasets have their own licences. `data/curate.py` rebuilds them.
- The LoRA adapters for RLLEAK: in the Google Drive folder above, not in git.
- Smoke outputs and run logs.

## Licence

Code: Apache-2.0 (see `LICENSE`). The models are used under their own licences.

Questions or collaboration: contact@affinelabs.xyz
