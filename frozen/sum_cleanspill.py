"""Score CLEANSPILL (notes/CLEANSPILL_PREREG.md). Reads results/cleanspill_<tag>.json (non-smoke), writes
results/cleanspill_summary.json and prints the verdicts.

python sum_cleanspill.py results/cleanspill_*.json --out results/cleanspill_summary.json
"""
import argparse, json, math, random
import numpy as np

# residual growth ground -> roof, frozen in the prereg (atlas snapshot b459bed)
GROWTH = {"olmo2-1b": 10, "qwen35-0.8b": 16, "llama32-1b": 27, "qwen3-0.6b": 692, "smollm2-1.7b": 748, "gemma3-1b": 850}
THIN, FAT = ["olmo2-1b", "qwen35-0.8b", "llama32-1b"], ["qwen3-0.6b", "smollm2-1.7b", "gemma3-1b"]
H1_BAR, H2_BAND, V1_SHARE, V2_BAR, V3_BAR = 0.5, 0.10, 0.95, 0.8, 0.7
NEED, ANCHOR = 4, "olmo2-1b"


def rank(x):
    x = np.asarray(x, float); o = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); r[o] = np.arange(len(x))
    # average ties
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spear(a, b):
    a, b = rank(a), rank(b)
    return float(np.corrcoef(a, b)[0, 1])


def partial_spear(y, x, zs):
    ry, rx = rank(y), rank(x)
    Z = np.column_stack([np.ones(len(y))] + [rank(z) for z in zs])
    res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    return float(np.corrcoef(res(ry), res(rx))[0, 1])


def boot_ci(fn, n, B=2000, seed=0):
    rng = np.random.default_rng(seed); vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        try:
            v = fn(idx)
            if v == v:
                vals.append(v)
        except Exception:
            pass
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [float("nan")] * 2


def score_model(d):
    U = [u for u in d["units"]]
    ok = [u for u in U if u["V1_ok"] and u["K_off"] == u["K_off"]]
    v1 = sum(u["V1_ok"] for u in U) / len(U)
    v2 = spear([u["G_dir_h0"] for u in U], [u["G_dir_h1"] for u in U])
    v3 = spear([u["S_h0"] for u in U], [u["S_h1"] for u in U])
    G = np.array([u["G_dir"] for u in ok]); K = np.array([u["K_off"] for u in ok])
    S = np.array([u["S"] for u in ok]); FL = np.array([u["floor"] for u in ok], float)
    LU = np.log(np.array([u["use"] for u in ok]))
    h1 = spear(G, K)
    h1_ci = boot_ci(lambda i: spear(G[i], K[i]), len(ok))
    h2 = partial_spear(K, S, [G, FL, LU])
    h2_ci = boot_ci(lambda i: partial_spear(K[i], S[i], [G[i], FL[i], LU[i]]), len(ok))
    h2_nouse = partial_spear(K, S, [G, FL])
    Rm = [u for u in ok if u["R"] == u["R"]]
    h2_R = partial_spear([u["R"] for u in Rm], [u["S"] for u in Rm], [[u["G_dir"] for u in Rm], [u["floor"] for u in Rm],
                         [math.log(u["use"]) for u in Rm]]) if len(Rm) > 10 else float("nan")
    # H2b (descriptive): unit gain minus the floor's random-direction median, against S
    rf = {int(k): v for k, v in d["rand_floor"].items()}
    ex = np.array([u["G_dir"] - rf[u["floor"]]["G_dir_med"] for u in ok])
    h2b = spear(S, ex)
    by_bin = {}
    for u in ok:
        by_bin.setdefault(u["bin"], []).append(u["K_off"])
    return {"n_units": len(U), "n_scored": len(ok), "V1_share": v1, "V2_G_split": v2, "V3_S_split": v3,
            "V1_pass": v1 >= V1_SHARE, "V2_pass": v2 >= V2_BAR, "V3_pass": v3 >= V3_BAR,
            "H1_rho": h1, "H1_ci": h1_ci, "H2_partial": h2, "H2_ci": h2_ci, "H2_partial_no_use": h2_nouse, "H2_partial_R": h2_R, "H2b_rho": h2b,
            "K_off_median": float(np.median(K)), "K_off_median_by_bin": {b: float(np.median(v)) for b, v in sorted(by_bin.items())},
            "S_median_high": float(np.median([u["S"] for u in ok if u["side"] == "high"])),
            "S_median_low": float(np.median([u["S"] for u in ok if u["side"] == "low"]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    M = {}
    for f in a.files:
        d = json.load(open(f))
        if d.get("smoke"):
            continue
        M[d["tag"]] = score_model(d)
    tags = [t for t in GROWTH if t in M]
    gated12 = [t for t in tags if M[t]["V1_pass"] and M[t]["V2_pass"]]
    gated3 = [t for t in gated12 if M[t]["V3_pass"]]
    h1_hold = [t for t in gated12 if M[t]["H1_rho"] >= H1_BAR]
    H1 = "PASS" if len(h1_hold) >= NEED and ANCHOR in h1_hold else ("FAIL" if len(gated12) >= NEED else "NOT SCORED")
    lab = lambda r: "DEPTH-SUFFICES" if abs(r) <= H2_BAND else ("CLEAN-HELPS" if r < -H2_BAND else "CLEAN-HURTS")
    labs = {t: lab(M[t]["H2_partial"]) for t in gated3}
    H2 = "NOT SCORED"
    if len(gated3) >= NEED:
        H2 = "MIXED"
        for name in ("DEPTH-SUFFICES", "CLEAN-HELPS", "CLEAN-HURTS"):
            hold = [t for t in gated3 if labs[t] == name]
            if len(hold) >= NEED and ANCHOR in hold:
                H2 = name
    h3_tags = [t for t in tags if M[t]["V1_pass"]]
    H3 = "NOT SCORED"; h3 = {}
    if len(h3_tags) == len(GROWTH):
        r = spear([math.log(GROWTH[t]) for t in h3_tags], [M[t]["K_off_median"] for t in h3_tags])
        fat = np.median([M[t]["K_off_median"] for t in FAT]); thin = np.median([M[t]["K_off_median"] for t in THIN])
        h3 = {"rho_growth_Koff": r, "fat_over_thin": float(fat / thin)}
        H3 = "PASS" if r >= 0.6 and fat / thin >= 2 else "FAIL"
    out = {"models": M, "gated_V1V2": gated12, "gated_V3": gated3, "H1": H1, "H1_hold": h1_hold, "H2": H2,
           "H2_labels": labs, "H3": H3, "H3_stats": h3}
    json.dump(out, open(a.out, "w"), indent=1)
    for t in tags:
        m = M[t]
        print(f"{t:14s} n {m['n_scored']:3d} V1 {m['V1_share']:.2f} V2 {m['V2_G_split']:.2f} V3 {m['V3_S_split']:.2f} | "
              f"H1 {m['H1_rho']:+.2f} {m['H1_ci']} | H2 {m['H2_partial']:+.2f} {m['H2_ci']} | K_off med {m['K_off_median']:.2e}")
    print("H1", H1, "| H2", H2, labs, "| H3", H3, h3)


if __name__ == "__main__":
    main()
