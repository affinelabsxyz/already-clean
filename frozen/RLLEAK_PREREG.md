# RLLEAK prereg: does an RL update's off-domain collateral come from the part that lands on leaky MLP units? (hypothesis C, step 1)

Frozen by Seungju on 2026-10-22, before any scored RLLEAK run. Lean: none.

**Why.**
- CLEANSPILL and QUIETLEAK found that a single-unit edit's off-target collateral is set by how much of the off-target
  text the edit touches. Leakage, a forward-only score, was the best predictor of that collateral.
- RL and fine-tuning are many small edits. If the same rule holds for them, the part of an RL update that lands on leaky
  units should carry most of the change on other domains, and little of the change on the trained domain.
- That would be the mechanism behind "lifts the target but shifts other abilities". It would also be the first step
  towards predicting it before training: from where a pre-RL gradient lands, in a later step.

## Frozen design

**Code.**
- Probe: `scripts/cleanspill/rlleak_probe.py`, sha256 `0fbfc4d7b4d7258fa218704866b524fe3784a3acfeb34e4335ed1c73bb01c330`.
  It uses CLEANSPILL's probe (ed8a7d67) as a library for text windows, plumbing and constants.
- Scorer: `scripts/cleanspill/sum_rlleak.py`, sha256 `8ab186a4b57f689683425b35f1e318e1ff5404111d05b0a402652d992b145e87`.
- Environment as CLEANSPILL: local RTX 5070 Ti, fp32, TF32 off.

**Runs.** These are all existing GRPO-LoRA runs on this machine (`researches/data-curation/runs_rl/*/adapter`).
- Every run uses r = 16 and α = 32, on all attention and MLP linears (the E3 runs on one third of the floors only).
- Every run was trained on GSM8K-style maths (the math_easy pool or its SHAPEFAIL subsets).
- Adapter sha256 prefixes:

| model | runs (adapter sha256 prefix) |
|---|---|
| Qwen3.5-0.8B-Base (11) | math_easy d0321d24 (120 steps, all floors); e3_floors0-7_s0 c0c9ffcb, _s1 1cd786cf; e3_floors8-15_s0 66336060, _s1 5ab546de; e3_floors16-23_s0 707e5740, _s1 18fb4957; sf_qwen35-0.8b_early_s0 2706f778, _s1 4efb5479; _late_s0 4237154d, _s1 afbdad57 |
| OLMo-2-0425-1B (4) | sf_olmo2-1b_early_s0 9e4d5ed5, _s1 ef748ab7; _late_s0 be8cdc08, _s1 edb3f662 |
| Qwen3-0.6B-Base (4) | sf_qwen3-0.6b_early_s0 7880d28e, _s1 e59e1fe1; _late_s0 6eeab982, _s1 c915e35a |

**Measures.**
- **Leakage of each MLP unit** relative to the trained domain. On window A of the base model,
  λ_i = (mean over the six non-maths kinds of mean s_i²) ÷ (mean over math_easy and math_hard of mean s_i²).
- **Per run**, merge the adapter. A unit's relative update energy is m_i = Σ over gate row, up row and down column of
  ‖ΔW‖² / ‖W_base‖². The run's update-weighted leakage is Λ_run = Σ m_i log λ_i / Σ m_i over updated units.
- **Collateral and target change.** On window C (CLEANSPILL's: 4 sequences × 8 kinds), take the mean next-token
  KL(base ‖ model) over positions 1…T−2:
  - K_off: over the six non-maths kinds;
  - K_on: over the two maths kinds;
  - CR = K_off / K_on.
- **Reverts** (the causal test).
  - Set the MLP weights of a unit set back to base: its gate and up rows and its down column. Attention updates are kept.
  - Sets are built by walking units in order until they hold a share q of the run's total MLP update energy:
    - **leaky**: highest λ first;
    - **clean**: lowest λ first;
    - **rand0–2**: three random orders, seed 0.
  - q = 0.2 is primary; q = 0.5 is reported.
  - Also measured: reverting the whole MLP update, to see how much of K_off and K_on is the MLP's at all.
- **Efficiency of a set:** f_off = (K_off(RL) − K_off(revert)) / K_off(RL), and f_on likewise. eff = f_off / max(f_on,
  0.01): the off-domain change removed per unit of on-domain change removed.

**Hypotheses and bars.**
- **C2 (primary, causal).**
  - A run supports it if, at q = 0.2, eff(leaky) ≥ 2 × eff(clean). If eff(clean) ≤ 0, the run supports it when
    eff(leaky) > 0.
  - A model holds if ≥ 2/3 of its runs support it: Qwen3.5 8 of 11, OLMo-2 3 of 4, Qwen3 3 of 4.
  - Verdict: **LEAK-DRIVES** if all three models hold; **PARTIAL** if 1–2 do; **NO** otherwise.
- **C2-rand (secondary).** The same rule with eff(leaky) ≥ 1.5 × the median eff of the three random sets.
- **C1 (descriptive).** Spearman(Λ_run, CR) within each model. A pooled version uses within-model z-scores. Runs are few
  and similar, so there is no bar.

**My prediction** (Seungju; before the run): C2 LEAK-DRIVES, with low-to-medium confidence. The single-unit result may not
carry to dense low-rank updates spread over all units.

**Smoke before freezing** (`results/rlleak_smoke/`; one run per model, window C thinned to 8 sequences; never scored).
- It checked that every model loads, the PEFT merge works (including Qwen3.5's linear-attention projections), the E3
  runs update only their floors (28,672 = 8 × 3,584 units), and the unit sets are built.
- The smoke log printed only unit counts, no K values.

**Run plan.** `scripts/cleanspill/run_rlleak.sh` runs OLMo-2, then Qwen3, then Qwen3.5, then the scorer. Outputs:
`results/rlleak_<tag>.json` and `results/rlleak_summary.json`.

## Freeze hash

sha256 `b96f8551e58efbc1d21e5adbaefa6d9772e148cc183e4ee142cea8c47e35f236` (61 lines) of the section "## Frozen design" from `scripts/prereg_hash.py`. Committed and pushed before the first scored RLLEAK run.
