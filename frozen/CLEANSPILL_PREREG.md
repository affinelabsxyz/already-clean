# CLEANSPILL prereg (small tier, local): is a unit edit's collateral set by cleanliness or by depth?

Frozen by Seungju on 2026-10-22, about 02:45 +0900, before any scored run. The design follows the proposal
(`notes/proposals/2026-10-22_cleanspill_prereg.md`). Changes from it are listed under "Changes from the proposal". The
lab may audit this freeze and add a 7–32B tier as its own amendment. Lean: none (bookkeeping and measurement only).

## Frozen design (small tier)

**Code.**
- Probe: `scripts/cleanspill/cleanspill_probe.py`, sha256 `ed8a7d6700934882a3317bcc9bbc644c97511be9597990440830c9267acb3ebd`.
- Scorer: `scripts/cleanspill/sum_cleanspill.py`, sha256 `4a76138f80e15bd5dc5f587357878fde91497af1f77478d56bc896dc0929c7a8`.
- Every result JSON records the probe's sha256. A run with a different hash is not scored.

**Environment.**
- Runs on Seungju's RTX 5070 Ti (16 GB) under Windows.
- torch 2.11.0+cu128, transformers 5.20.0.
- fp32 weights and compute, TF32 off, default attention kernels.
- Qwen3.5 uses transformers' reference causal-conv1d fallback.

**Models** (residual growth ground → roof from the atlas snapshot, website commit b459bed; frozen here):

| tag | model | growth | group |
|---|---|---|---|
| olmo2-1b | allenai/OLMo-2-0425-1B | 10× | thin (anchor) |
| qwen35-0.8b | Qwen/Qwen3.5-0.8B-Base | 16× | thin |
| llama32-1b | meta-llama/Llama-3.2-1B | 27× | thin |
| qwen3-0.6b | Qwen/Qwen3-0.6B-Base | 692× | fat |
| smollm2-1.7b | HuggingFaceTB/SmolLM2-1.7B | 748× | fat |
| gemma3-1b | google/gemma-3-1b-pt | 850× | fat |

**Text.** There are eight kinds:
- **wiki:** WikiText-103 validation, non-empty lines joined.
- **math_easy, math_hard, code, medical, finance, logic, if:** the first 4,000 items of each curated `pool.jsonl` from
  PR 3's `curate.py` (seed 0), with question and answer joined.

Each kind's stream is tokenised once and cut into 256-token sequences. A BOS token is prepended where the tokenizer adds
one by default (Gemma only). Position 0 is excluded everywhere. Windows, by chunk index per kind:
- **A** (selectivity): chunks 0–15, 128 sequences;
- **B** (downstream gain): chunks 16–17, 16 sequences;
- **C** (edits): chunks 18–21, 32 sequences.

**Switch-on (GAUGE-R's rule B).** At each token, unit i of floor l is on if |s_i|·‖o_i^eff‖ is in the floor's top 2%.
- s_i is the input to `down_proj`.
- o_i^eff is the down-projection column. For OLMo-2 and Gemma it is scaled by the post-feedforward norm gain
  (Gemma: 1 + w).

**Units.** Per model, 6 equal floor bins. Each bin's pool is the units that are on at 0.5–20% of window-A tokens. Take the
20 with the lowest selectivity S and the 20 with the highest (ties broken by a seed-0 shuffle), so 240 units per model.

**Per unit:**
- **S (cleanliness)** = 1 − H(p)/log 8.
  - p is the unit's switch-on rate per kind, normalised over kinds, on window A.
  - S_h0 and S_h1 are the same on even and odd chunks.
  - The unit's home kind is the argmax of p.
- **G_dir (depth, primary predictor).**
  - The push: at floor l's output, add δ = 0.02·‖x_l‖·ô_i at every window-B position.
  - G_dir = the median over tokens of ‖Δx_final⊥‖ / ‖x_final‖ / 0.02. Δx_final⊥ is the change of the pre-final-norm state,
    orthogonal to the clean final state.
  - In words: the change in the direction the final norm reads, per unit of relative push.
  - G_dir_h0 and G_dir_h1 are the same on even and odd chunks.
  - Also recorded: total, side and κ of the push, and the same for 40 random directions per floor (seed 1000 + l).
- **The edit (window C).**
  - Scale s_i by (1 + c) at every position.
  - c is set so the mean change of the floor's write (after any post-norm), on home-kind positions, equals
    T = 0.02·median‖x_l‖ over those positions, where x_l is the residual entering floor l.
  - The first c is linear. If the realised mean differs from T by more than 5%, one secant correction is made.
- **Outcomes.**
  - **K_off (primary)** = mean KL(base ‖ edited) of the next-token distribution, over positions of the other kinds where
    the unit is off.
  - **E_on** = mean change of log p of the unit's top-5 promoted tokens, over home-kind positions where it is on. The
    promoted tokens are the logit lens of sign(mean s)·o_i^eff through the final-norm gain.
  - **R** = K_off / E_on, when E_on > 0.

**Gates** (per model). A model that fails V1 or V2 is not scored for H1 or H2. One that fails V3 is not scored for H2.
- **V1:** ≥ 95% of units have a realised write change within 5% of T.
- **V2:** split-half Spearman of G_dir ≥ 0.8.
- **V3:** split-half Spearman of S ≥ 0.7.

**Hypotheses and bars** (scored only on units that pass V1 and have a finite K_off).
- **H1 (depth predicts collateral).**
  - Bar: Spearman(G_dir, K_off) ≥ 0.5 within a model.
  - PASS if the bar holds in ≥ 4 of the gated models, olmo2-1b among them.
  - FAIL if at least 4 models are gated and the PASS rule is not met. Otherwise NOT SCORED.
- **H2 (cleanliness beyond depth).**
  - Compute the partial Spearman of S with K_off, controlling for G_dir, floor index and log(usage) (rank-OLS
    residuals). Each model gets a label:
    - DEPTH-SUFFICES if |r| ≤ 0.10;
    - CLEAN-HELPS if r < −0.10;
    - CLEAN-HURTS if r > +0.10.
  - The verdict is the label held by ≥ 4 gated models, olmo2-1b among them. Otherwise MIXED. With fewer than 4 gated
    models, NOT SCORED.
  - The version without the usage control, and the version with R in place of K_off, are reported, not scored.
- **H3 (shape).**
  - Over the 6 models, Spearman(log growth, median K_off) ≥ 0.6, and the median of the fat group's median K_off ≥ 2×
    the thin group's.
  - PASS if both hold. Needs all 6 models to pass V1.
- **H2b (descriptive).** Spearman of S against G_dir minus the floor's random-direction median G_dir.
- **Intervals.** 95% bootstrap over units, 2,000 resamples, seed 0.

**Run plan.**
- One process per model, in this order: olmo2-1b, llama32-1b, qwen3-0.6b, smollm2-1.7b, gemma3-1b, qwen35-0.8b.
- Outputs go to `results/cleanspill_<tag>.json`. A log goes beside each.
- Estimated 1.5–2.5 h in total.
- Scoring: `python scripts/cleanspill/sum_cleanspill.py results/cleanspill_*.json --out results/cleanspill_summary.json`.

## Changes from the proposal (made before freezing, after a smoke run)

A smoke run used 4 units per bin and 2 random directions per floor, on all six models
(`results/cleanspill_smoke/`, marked `smoke`, never scored). It showed three validity problems, fixed here:

1. **Usage confound.** With all units on at ≥ 0.5% in the pool, the most selective units were the rarest (1–9% usage)
   and the least selective were near-ubiquitous (up to 99%). Fixes: the pool is capped at 20% usage, and log(usage) is
   a control in H2.
2. **Inflation confound.** The proposal's G (the final displacement over the push) mostly measured how much the state
   grows between floor l and the roof: 20–70× in Qwen3-0.6B at early floors. It is replaced by G_dir, the change in
   the final *direction* per unit of *relative* push, which is what the read sees.
3. **Sign and precision.**
   - The intended effect used promoted tokens regardless of the unit's activation sign, so it is now signed.
   - The base log-probs were cached in fp16, which gave tiny negative KLs, so they are now fp32.

Also: the text kinds are the eight available locally rather than the S47 genres. The bins, ε, the 2% switch-on and the
bars are as proposed. **Disclosure:** the smoke outputs were looked at to find these problems, including a few per-unit
K_off values. No bar was changed after that. The H2 bands (±0.10) and the anchor rule are as in the proposal.

## Freeze hash

sha256 `0fbf0dca421b55add2e866a4a53667501a91a3e4ad20a4fbbb3f5d9b2d317412` (99 lines) of the section "## Frozen design (small tier)" from `scripts/prereg_hash.py`. Committed and pushed before the first scored run (`scripts/cleanspill/run_small.sh`).
