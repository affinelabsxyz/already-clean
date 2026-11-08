### CLEANSPILL small tier: is a unit edit's collateral set by cleanliness or by depth? (prereg notes/CLEANSPILL_PREREG.md, sha256 0fbf0dca…; probe ed8a7d67, scorer 4a76138f; Lean none), six local ≤ 1.7B models, fp32, 2026-10-22: H1 FAIL (depth gain does not predict collateral); H2 CLEAN-HELPS (6 of 6); H3 FAIL (shape does not predict collateral)

**Runs.**
- `scripts/cleanspill/run_small.sh` on Seungju's RTX 5070 Ti, 02:42–03:59 +0900.
- Every model rc 0. Each result JSON carries probe sha256 ed8a7d67…, the frozen hash.
- Per model 10–17 min: olmo2-1b 648 s, llama32-1b 631 s, qwen3-0.6b 584 s, smollm2-1.7b 1,020 s, gemma3-1b 1,005 s,
  qwen35-0.8b 661 s.
- Scored by the frozen scorer → `results/cleanspill_summary.json`.

**Gates.** All six models pass:
- **V1** (edit size within 5% of the target): 1.00 everywhere except gemma3-1b, which is 0.954. Its 11 units that missed
  are excluded, as the prereg rule says.
- **V2** (split-half of G_dir): 0.99–1.00.
- **V3** (split-half of S): 0.83–0.92.

| model | growth | H1 ρ(G_dir, K_off) [95%] | H2 partial ρ(S, K_off \| G_dir, floor, log use) [95%] | label | median K_off |
|---|---|---|---|---|---|
| olmo2-1b | 10× | −0.38 [−0.49, −0.25] | −0.68 [−0.74, −0.59] | CLEAN-HELPS | 1.4e-4 |
| qwen35-0.8b | 16× | −0.09 [−0.23, +0.05] | −0.65 [−0.72, −0.56] | CLEAN-HELPS | 8.4e-5 |
| llama32-1b | 27× | +0.03 [−0.10, +0.16] | −0.67 [−0.73, −0.59] | CLEAN-HELPS | 7.2e-5 |
| qwen3-0.6b | 692× | −0.20 [−0.33, −0.07] | −0.64 [−0.72, −0.53] | CLEAN-HELPS | 1.2e-4 |
| smollm2-1.7b | 748× | −0.25 [−0.36, −0.13] | −0.75 [−0.80, −0.70] | CLEAN-HELPS | 9.0e-5 |
| gemma3-1b | 850× | +0.17 [+0.04, +0.30] | −0.65 [−0.72, −0.56] | CLEAN-HELPS | 8.6e-4 |

**Verdicts.**
- **H1: FAIL.** The bar was ≥ 0.5 in 4 models. ρ is ≤ 0.17 in all six, and negative in OLMo-2 and three others.
- **H2: CLEAN-HELPS** in all six, OLMo-2 included.
- **H3: FAIL.**
  - ρ(log growth, median K_off) is 0.31; the bar was 0.6.
  - The fat group's median over the thin group's is 1.41×; the bar was 2×.
  - Gemma's collateral is the highest by about 6–10×, but the other two fat models sit with the thin ones.

**Descriptive (reported in the prereg, not scored).**
- **Clean vs generic units at matched edit size.** The selected high-S units (median S 0.77–0.95) are compared with the
  low-S units (S ≈ 0).
  - Off-target KL is **4–9× smaller** for the clean units. By model: OLMo-2 5.0×, Llama 7.8×, Qwen3 9.3×, SmolLM2 7.9×,
    Gemma 8.3×, Qwen3.5 4.3×.
  - The intended effect (E_on, the log-prob gain of the unit's promoted tokens where it is on) is **2–4× larger** for the
    clean units.
- **Robustness of H2.**
  - Without the usage control the partial ρ is −0.56 to −0.71.
  - With R = K_off / E_on in place of K_off it is −0.39 to −0.62.
- **H2b.** Clean units' write directions are amplified downstream *more* than random directions at the same floor
  (ρ(S, G_dir − floor random median) +0.24 to +0.57). Yet they cause less collateral. So the depth gain of a unit's
  direction is not what sets its collateral.

**Reading.**
- At matched edit size, how clean a unit is decides its side effect. How much the floors above amplify its write
  direction does not, in any of six models from five labs, thin or fat.
- The likely mechanism is local, not downstream. The edit scales the unit wherever it is active, and a generic unit
  stays weakly active below the switch-on line on the other kinds of text (SLOTGRAPH: "weak, not silent"). So it writes
  there directly. A clean unit is nearly silent off its domain.
- So in practice "clean" here means "quiet elsewhere". The test cannot separate one meaning per unit from silence on
  other text.
- For Nick's and Minh's question: at the unit level, cleanliness is an asset for editing. Clean units give more
  intended effect and less collateral.
- For our depth programme: unit-level edit collateral is not where depth shows up.
- This sits beside OPTLOCAL's H2w (a random mask did as well as a concentrated one). There, units were picked by an
  optimiser, not by cleanliness, so the two results do not conflict.

**Caveats.**
- One edit type: scaling a unit's activation by a matched amount (ε = 0.02 of the residual norm on home tokens).
- "Clean" is text-kind selectivity over eight local kinds. Units are the extremes (20 lowest and 20 highest S per
  depth bin) within 0.5–20% usage.
- The high-S units are still rarer: median usage 1.4–2.5% against 5.7–12.8%. Usage is controlled in H2.
- Small tier only (≤ 1.7B). The 7–32B tier is the lab's, as an amendment.
- Smoke outputs (`results/cleanspill_smoke/`) were looked at before freezing. The design changes they caused are listed
  in the prereg.
