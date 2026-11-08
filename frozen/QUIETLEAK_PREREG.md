# QUIETLEAK prereg: does "clean helps" come from one meaning, or from being quiet elsewhere? (follow-up to CLEANSPILL)

Frozen by Seungju on 2026-10-22, before any scored QUIETLEAK run. Lean: none.

Background: CLEANSPILL's small tier (`notes/CLEANSPILL_claims.md`) found CLEAN-HELPS in 6 of 6 models. At matched edit
size, clean units cause 4–9× less off-target KL, beyond downstream gain, floor and usage. That edit scaled the unit at
every position. A generic unit is weakly active below the switch-on line on other text, so the edit writes there
directly. QUIETLEAK separates the two readings.

## Frozen design

**Code.**
- Probe: `scripts/cleanspill/quietleak_probe.py`, sha256 `bc0e8251ffbb2ef1e16b68ec8e8e108bdd0cc7038283d0dee61132bd42b02ce4`.
  It uses CLEANSPILL's probe (ed8a7d67) unchanged as a library.
- Scorer: `scripts/cleanspill/sum_quietleak.py`, sha256 `e9ba0e197d13cf9751153115a8c8f7968063079461cfcbc049f7d4d50553df1e`.
  It uses CLEANSPILL's scorer (4a76138f) unchanged as a library.

**Models, units, windows.** CLEANSPILL's six local models.
- The 240 units per model are read from `results/cleanspill_<tag>.json` (the frozen run). The probe checks that file's
  probe hash.
- Text windows, switch-on rule, ε = 0.02, precision and environment are CLEANSPILL's.

**(A) On-only edit.**
- Scale s_i by (1 + c) only at the window-C positions where the unit is on (rule B, from the clean pass); nowhere else.
- Matched size: the mean change of the floor's write over the edited home-kind positions (home ∧ on) equals
  T = 0.02·median‖x_l‖ over home-kind positions. The first c is linear, followed by up to three secant corrections.
- Units with fewer than 10 home ∧ on positions in window C are not edited. That excludes 4–12 of 240 per model, counted
  from CLEANSPILL's saved counts before freezing.
- **K_off_onl:** mean KL(base ‖ edited) over positions of the other kinds where the unit is off. Those positions are not
  edited, so any collateral there arrives through context or depth.
- **E_on_onl:** as CLEANSPILL's E_on.

**(B) Leakage.**
- For each unit, on window A: λ = (mean of s_i² over positions of the other kinds, averaged over those kinds) ÷ (mean of
  s_i² over home-kind positions).
- All positions ≥ 1 count, on or off.
- It is a forward-pass-only score.

**Gates** (per model).
- V1: ≥ 95% of edited units within 5% of T.
- Not more than 25% of units skipped.
- Gated models are scored for A and B.
- A unit is scored if it passed CLEANSPILL's V1, passes V1 here, and has a finite K_off in both runs.

**Hypotheses and bars.** Partial Spearman uses CLEANSPILL's rank-OLS. ρ_all is CLEANSPILL's statistic recomputed on the
same scored units.
- **A, per model:** ρ_onl = partial ρ(S, K_off_onl | G_dir, floor, log use). Each model gets one label:
  - **QUIET** if |ρ_onl| ≤ 0.10. The clean advantage disappears when the edit stays where the unit is on, so it came
    from leakage.
  - **SEMANTIC** if ρ_onl < −0.10 and |ρ_onl| ≥ 0.5·|ρ_all|. It survives.
  - **REVERSED** if ρ_onl > +0.10.
  - **PARTIAL** otherwise.
  - Verdict: the label held by ≥ 4 gated models, olmo2-1b among them; otherwise MIXED.
- **B1 (leakage adds to cleanliness).**
  - Bar: partial ρ(log λ, K_off | S, G_dir, floor, log use) ≥ +0.20, using CLEANSPILL's all-position K_off.
  - LEAK-ADDS if the bar holds in ≥ 4 gated models, olmo2-1b among them.
- **B2 (leakage explains cleanliness).**
  - Bar: partial ρ(S, K_off | log λ, G_dir, floor, log use) within ±0.10.
  - LEAK-EXPLAINS if it holds in ≥ 4 gated models, olmo2-1b among them.
- **Reported, not scored:**
  - medians of K_off and K_off_onl for the clean and generic halves;
  - raw ρ(log λ, K_off) and ρ(S, K_off);
  - ρ(S, log λ).
- **Intervals:** 95% bootstrap over units, 2,000 resamples.

**My predictions** (Seungju; written before the run): A = QUIET (medium confidence), B1 = LEAK-ADDS (medium),
B2 = LEAK-EXPLAINS (low).

**Run plan.**
- `scripts/cleanspill/run_quietleak.sh`: the same six models in CLEANSPILL's order, then the scorer.
- Outputs: `results/quietleak_<tag>.json` and `results/quietleak_summary.json`.

**Smoke before freezing** (`results/quietleak_smoke/`; 8 units on Qwen3-0.6B, Gemma-3-1B and OLMo-2-1B; never scored):
- The first version matched the edit size over *all* home-kind positions. With the edit confined to on positions,
  Gemma's post-feedforward norm caps a single unit's write, so the target could not be reached for units on at few
  positions. Example: floor 3 #1706 saturated near 7.5 against a target of 16.9.
- The target was therefore moved to the edited positions (home ∧ on), and secant retries were raised from 1 to 3.
- After the change, every edited smoke unit met V1.
- Only the validity counts (skips, V1, finite values) and the calibration traces were looked at, no K_off_onl values.
- CLEANSPILL's outcomes were already known when this prereg was written.

## Freeze hash

sha256 `30075e63e3e64aede466194e1dbce1298b8a8b89843a4a563bc67977cbcd0a4a` (72 lines) of the section "## Frozen design" from `scripts/prereg_hash.py`. Committed and pushed before the first scored QUIETLEAK run.
