"""Score QUIETLEAK (notes/QUIETLEAK_PREREG.md) from results/cleanspill_<tag>.json + results/quietleak_<tag>.json.

python sum_quietleak.py --tags olmo2-1b llama32-1b qwen3-0.6b smollm2-1.7b gemma3-1b qwen35-0.8b --out results/quietleak_summary.json
"""
import argparse, json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sum_cleanspill import spear, partial_spear, boot_ci   # frozen CLEANSPILL scorer (sha256 4a76138f), as a library

BAND, NEED, ANCHOR, V1_SHARE, MAX_SKIP, B1_BAR = 0.10, 4, "olmo2-1b", 0.95, 0.25, 0.20


def score(tag):
    a = json.load(open(f"results/cleanspill_{tag}.json")); b = json.load(open(f"results/quietleak_{tag}.json"))
    assert not b["smoke"]
    q = {(r["floor"], r["unit"]): r for r in b["rows"]}
    rows = []
    for u in a["units"]:
        r = q[(u["floor"], u["unit"])]
        rows.append((u, r))
    n = len(rows)
    skipped = sum(r.get("skipped", False) for _, r in rows)
    edited = [(u, r) for u, r in rows if not r.get("skipped", False)]
    v1 = sum(r["V1_ok"] for _, r in edited) / max(len(edited), 1)
    ok = [(u, r) for u, r in edited if u["V1_ok"] and r["V1_ok"] and u["K_off"] == u["K_off"] and r["K_off_onl"] == r["K_off_onl"]]
    S = np.array([u["S"] for u, _ in ok]); G = np.array([u["G_dir"] for u, _ in ok]); FL = np.array([u["floor"] for u, _ in ok], float)
    LU = np.log([u["use"] for u, _ in ok]); KA = np.array([u["K_off"] for u, _ in ok]); KO = np.array([r["K_off_onl"] for _, r in ok])
    LK = np.log(np.array([r["leak"] for _, r in ok]).clip(1e-30))
    rho_all = partial_spear(KA, S, [G, FL, LU]); rho_onl = partial_spear(KO, S, [G, FL, LU])
    ci_onl = boot_ci(lambda i: partial_spear(KO[i], S[i], [G[i], FL[i], LU[i]]), len(ok))
    if abs(rho_onl) <= BAND:
        lab = "QUIET"
    elif rho_onl < -BAND and abs(rho_onl) >= 0.5 * abs(rho_all):
        lab = "SEMANTIC"
    elif rho_onl > BAND:
        lab = "REVERSED"
    else:
        lab = "PARTIAL"
    b1 = partial_spear(KA, LK, [S, G, FL, LU]); b2 = partial_spear(KA, S, [LK, G, FL, LU])
    b1_ci = boot_ci(lambda i: partial_spear(KA[i], LK[i], [S[i], G[i], FL[i], LU[i]]), len(ok))
    hi = [i for i, (u, _) in enumerate(ok) if u["side"] == "high"]; lo = [i for i, (u, _) in enumerate(ok) if u["side"] == "low"]
    med = lambda v, idx: float(np.median(v[idx])) if idx else float("nan")
    return {"n": n, "skipped": skipped, "skip_share": skipped / n, "n_scored": len(ok), "V1_onl_share": v1,
            "gated": v1 >= V1_SHARE and skipped / n <= MAX_SKIP,
            "A_rho_all": rho_all, "A_rho_onl": rho_onl, "A_ci_onl": ci_onl, "A_label": lab,
            "B1_partial_leak": b1, "B1_ci": b1_ci, "B2_partial_S_given_leak": b2,
            "raw_rho_leak_K": spear(LK, KA), "raw_rho_S_K": spear(S, KA), "rho_S_leak": spear(S, LK),
            "K_all_med_clean": med(KA, hi), "K_all_med_generic": med(KA, lo),
            "K_onl_med_clean": med(KO, hi), "K_onl_med_generic": med(KO, lo)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    M = {t: score(t) for t in a.tags}
    g = [t for t in a.tags if M[t]["gated"]]
    A = "NOT SCORED"
    if len(g) >= NEED:
        A = "MIXED"
        for name in ("QUIET", "SEMANTIC", "PARTIAL", "REVERSED"):
            hold = [t for t in g if M[t]["A_label"] == name]
            if len(hold) >= NEED and ANCHOR in hold:
                A = name
    b1 = [t for t in g if M[t]["B1_partial_leak"] >= B1_BAR]
    b2 = [t for t in g if abs(M[t]["B2_partial_S_given_leak"]) <= BAND]
    B1 = ("LEAK-ADDS" if len(b1) >= NEED and ANCHOR in b1 else "NO") if len(g) >= NEED else "NOT SCORED"
    B2 = ("LEAK-EXPLAINS" if len(b2) >= NEED and ANCHOR in b2 else "NO") if len(g) >= NEED else "NOT SCORED"
    out = {"models": M, "gated": g, "A": A, "B1": B1, "B1_hold": b1, "B2": B2, "B2_hold": b2}
    json.dump(out, open(a.out, "w"), indent=1)
    for t in a.tags:
        m = M[t]
        print(f"{t:13s} n {m['n_scored']:3d} skip {m['skipped']:2d} V1 {m['V1_onl_share']:.2f} | A all {m['A_rho_all']:+.2f} onl {m['A_rho_onl']:+.2f} {m['A_label']:9s} | "
              f"B1 {m['B1_partial_leak']:+.2f} B2 {m['B2_partial_S_given_leak']:+.2f} | K clean/gen all {m['K_all_med_clean']:.1e}/{m['K_all_med_generic']:.1e} onl {m['K_onl_med_clean']:.1e}/{m['K_onl_med_generic']:.1e}")
    print("A", A, "| B1", B1, "| B2", B2)


if __name__ == "__main__":
    main()
