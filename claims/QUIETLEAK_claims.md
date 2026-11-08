### QUIETLEAK does "clean helps" come from one meaning or from being quiet elsewhere? (prereg notes/QUIETLEAK_PREREG.md, sha256 30075e63…; probe bc0e8251, scorer e9ba0e19; Lean none), CLEANSPILL's six local models and units, fp32, 2026-10-22: A SEMANTIC (6/6); B1 LEAK-ADDS (6/6); B2 NO

**Runs.**
- `scripts/cleanspill/run_quietleak.sh`, 08:56–09:32 +0900. Every model rc 0. Each result JSON carries the frozen probe hash.
- Gates: every model passes. V1 is 1.00 everywhere. Skips are 4–12 of 240.

| model | ρ_all | ρ_onl [95%] | A label | B1 partial ρ(log λ, K_off \| S, …) [95%] | B2 partial ρ(S, K_off \| log λ, …) | K_off clean / generic, all-position edit | K_off clean / generic, on-only edit |
|---|---|---|---|---|---|---|---|
| olmo2-1b | −0.68 | −0.64 [−0.72, −0.54] | SEMANTIC | +0.46 [0.35, 0.56] | −0.21 | 4.6e-5 / 2.3e-4 | 8.9e-8 / 5.1e-6 |
| llama32-1b | −0.66 | −0.62 [−0.70, −0.53] | SEMANTIC | +0.66 [0.56, 0.74] | −0.23 | 1.5e-5 / 1.2e-4 | 1.2e-8 / 2.8e-6 |
| qwen3-0.6b | −0.64 | −0.63 [−0.72, −0.52] | SEMANTIC | +0.55 [0.45, 0.64] | −0.14 | 2.5e-5 / 2.2e-4 | 2.1e-8 / 5.6e-6 |
| smollm2-1.7b | −0.75 | −0.71 [−0.78, −0.62] | SEMANTIC | +0.53 [0.43, 0.63] | −0.24 | 2.6e-5 / 2.1e-4 | 4.2e-8 / 2.5e-6 |
| gemma3-1b | −0.61 | −0.62 [−0.70, −0.54] | SEMANTIC | +0.60 [0.47, 0.71] | −0.25 | 2.0e-4 / 1.6e-3 | 8.4e-9 / 3.9e-5 |
| qwen35-0.8b | −0.65 | −0.62 [−0.73, −0.51] | SEMANTIC | +0.43 [0.30, 0.56] | −0.17 | 3.1e-5 / 1.4e-4 | 2.9e-8 / 3.3e-6 |

**Verdicts.**
- **A: SEMANTIC (6 of 6).** My prediction was QUIET, and it was wrong. Confining the edit to the positions where the unit
  is on leaves the clean advantage almost unchanged (ρ_onl −0.62 to −0.71 against ρ_all −0.61 to −0.75). So
  sub-threshold leakage does not explain CLEAN-HELPS.
- **B1: LEAK-ADDS (6 of 6).** Leakage, a forward-only score, predicts collateral beyond cleanliness. As a single
  predictor it is stronger: raw ρ(log λ, K_off) is +0.73 to +0.84, against ρ(S, K_off) of −0.57 to −0.69.
- **B2: NO.** Cleanliness keeps a small part beyond leakage (−0.14 to −0.25).

**Descriptive.**
- The on-only edit cuts off-target collateral by a median 200–750× against the all-position edit at the same matched
  size on home tokens. Median K_off(all) / K_off(on-only): OLMo-2 297×, Llama 385×, Qwen3 264×, SmolLM2 310×,
  Gemma 751×, Qwen3.5 199×.
- Clean units' on-only collateral is about 50–5,000× below generic units'.

**Exploratory, after scoring (not pre-registered).**
- The A label separates sub-threshold leakage from firing above the line. It cannot separate "one meaning" from
  "fires in fewer places". The units' on-only collateral on other kinds tracks how many positions of those kinds the unit
  fires on (ρ +0.84 to +0.87), and that count is almost the inverse of cleanliness (ρ(S, ·) −0.80 to −0.87).
- Adding log(number of other-kind positions where the unit is on) as a further control shrinks the clean effect from
  −0.62…−0.71 to −0.18…−0.42. It stays negative in all six; in Gemma it stays strongest.
- So most of CLEAN-HELPS is "touches less of the other text", above or below the switch-on line. A smaller residual
  remains that this test cannot attribute.

**Reading.**
- For a single-unit edit, the collateral is set by how much of the off-target text the edit touches.
- At the unit level:
  - **leakage**, a forward-only score, is the best predictor;
  - **restricting the edit to where the unit fires** cuts collateral by two to three orders of magnitude;
  - cleanliness matters mainly as a proxy for both.
- This is a practical edit-risk recipe that needs no depth measurement. It sits with CLEANSPILL's null for depth
  (H1, H3).

**Caveats.**
- Six ≤ 1.7B models, one edit type (scaling), eight local text kinds.
- Sequences are single-kind, so on-only collateral on other kinds comes from edits at the unit's own other-kind firing
  positions, carried to their neighbours.
- The exploratory control is post hoc.
