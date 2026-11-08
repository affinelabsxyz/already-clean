### RLLEAK does an RL update's off-domain change come from the part that lands on leaky MLP units? (prereg notes/RLLEAK_PREREG.md, sha256 b96f8551…; probe 0fbfc4d7, scorer 8ab186a4; Lean none), 19 existing GSM8K GRPO-LoRA runs (Qwen3.5-0.8B ×11, OLMo-2-1B ×4, Qwen3-0.6B ×4), fp32, 2026-10-22: C2 NO (bars missed); the ordering leaky > random > clean holds in 19 of 19 runs

**Runs.** `scripts/cleanspill/run_rlleak.sh`, 09:39–09:48 +0900, all rc 0. Every result carries the frozen probe hash.

**Verdict, as frozen: C2 NO; C2-rand holds in no model.** At q = 0.2, eff = (share of the off-domain KL removed) ÷ (share
of the on-domain KL removed) when that energy share of the MLP update is reverted.

| model | runs with eff(leaky) ≥ 2× eff(clean) (needed) | eff(leaky) / eff(clean), min / median / max | eff(leaky) / median eff(random) | leaky > random > clean |
|---|---|---|---|---|
| OLMo-2-1B | 0 of 4 (3) | 1.55 / 1.65 / 1.75 | 1.15–1.19 | 4 of 4 |
| Qwen3-0.6B | 2 of 4 (3) | 1.85 / 2.10 / 2.65 | 1.23–1.38 | 4 of 4 |
| Qwen3.5-0.8B | 7 of 11 (8) | 1.60 / 2.15 / 2.42 | 1.03–1.50 | 11 of 11 |

**Descriptive.**
- **The ordering is perfectly consistent; the size is modest.** Reverting the leaky units' share removes off- and
  on-domain change about equally (eff 0.84–1.42, median about 1.1). The same energy on clean units removes about twice
  as much maths change as off-domain change (eff 0.45–0.85). Random sets sit between.
- At q = 0.5 the gaps shrink: median eff leaky / clean / random is 1.17 / 0.92 / 1.04 (OLMo-2), 0.96 / 0.86 / 0.94
  (Qwen3) and 1.00 / 0.73 / 0.86 (Qwen3.5).
- **The MLP carries most of each run's change.** Reverting the whole MLP update removes a median 88% (OLMo-2), 63–69%
  (Qwen3) and 68–76% (Qwen3.5) of both off- and on-domain KL. The rest is the attention update.
- **C1** (run-level Λ_run against off/on ratio): OLMo-2 −0.80 and Qwen3 +0.80 with n = 4, Qwen3.5 +0.33, pooled 0.11.
  Nothing.

**Reading.**
- The single-unit rule (CLEANSPILL, QUIETLEAK) does not carry over to RL as "collateral lives on leaky units": reverting
  them does not remove off-domain change more cheaply than on-domain change.
- What does carry over is the other side. The part of an RL update that sits on the maths-selective (clean) units
  carries the maths change preferentially, about 2:1 over the off-domain change. Random and leaky shares are close to
  proportional.
- So the targeted part of an RL update is localised on clean units, and its untargeted part is spread over everything.
- For hypothesis D (RL restricted to clean units), this is directional support, not proof. Reverting is not training.

**Caveats.**
- KL(base ‖ model) measures change, not harm. The runs are short (60–120 steps) LoRA runs on small models, all on
  GSM8K-style maths.
- Leakage is from the base model, relative to the maths kinds.
- The prereg's 2× bar was set from the single-unit effect sizes. The RL effect is real in sign but smaller.
