"""Score RLLEAK (notes/RLLEAK_PREREG.md) from results/rlleak_<tag>.json.

python sum_rlleak.py --tags olmo2-1b qwen3-0.6b qwen35-0.8b --out results/rlleak_summary.json
"""
import argparse, json, math
import numpy as np

FLOOR_ON = 0.01      # f_on below this is treated as 0.01 in the efficiency ratio
BAR_CLEAN, BAR_RAND, SHARE = 2.0, 1.5, 2 / 3


def eff(r, c):
    off0, on0 = r["rl"]["K_off"], r["rl"]["K_on"]
    f_off = (off0 - r[c]["K_off"]) / off0
    f_on = (on0 - r[c]["K_on"]) / on0
    return f_off / max(f_on, FLOOR_ON), f_off, f_on


def spear(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {"models": {}}
    verdict_models = []
    pooled = []
    for t in a.tags:
        d = json.load(open(f"results/rlleak_{t}.json"))
        assert not d["smoke"]
        runs = []
        for r in d["runs"]:
            row = {"run": r["run"], "Lambda_run": r["Lambda_run"],
                   "CR": r["rl"]["K_off"] / r["rl"]["K_on"], "K_off": r["rl"]["K_off"], "K_on": r["rl"]["K_on"],
                   "mlp_share_off": 1 - r["rev_all_mlp"]["K_off"] / r["rl"]["K_off"],
                   "mlp_share_on": 1 - r["rev_all_mlp"]["K_on"] / r["rl"]["K_on"]}
            for q in (0.2, 0.5):
                e_l, fo_l, fn_l = eff(r, f"q{q}_leaky"); e_c, fo_c, fn_c = eff(r, f"q{q}_clean")
                e_r = [eff(r, f"q{q}_rand{k}")[0] for k in range(3)]
                row[f"q{q}"] = {"eff_leaky": e_l, "eff_clean": e_c, "eff_rand_med": float(np.median(e_r)),
                                "f_off_leaky": fo_l, "f_on_leaky": fn_l, "f_off_clean": fo_c, "f_on_clean": fn_c,
                                "leak_vs_clean": e_l >= BAR_CLEAN * e_c if e_c > 0 else e_l > 0,
                                "leak_vs_rand": e_l >= BAR_RAND * float(np.median(e_r)) if np.median(e_r) > 0 else e_l > 0}
            runs.append(row)
            pooled.append((t, row["Lambda_run"], row["CR"]))
        n = len(runs)
        sup_c = sum(r["q0.2"]["leak_vs_clean"] for r in runs); sup_r = sum(r["q0.2"]["leak_vs_rand"] for r in runs)
        m = {"runs": runs, "n": n, "support_vs_clean": sup_c, "support_vs_rand": sup_r,
             "holds_vs_clean": sup_c >= math.ceil(SHARE * n), "holds_vs_rand": sup_r >= math.ceil(SHARE * n),
             "C1_rho_Lambda_CR": spear([r["Lambda_run"] for r in runs], [r["CR"] for r in runs])}
        out["models"][t] = m
        verdict_models.append(m["holds_vs_clean"])
    k = sum(verdict_models)
    out["C2"] = "LEAK-DRIVES" if k == len(verdict_models) else ("PARTIAL" if k > 0 else "NO")
    out["C2_secondary_vs_rand"] = [t for t in a.tags if out["models"][t]["holds_vs_rand"]]
    # pooled C1 with per-model centring of ranks (descriptive)
    zs_l, zs_c = [], []
    for t in a.tags:
        rows = [(L, C) for tt, L, C in pooled if tt == t]
        if len(rows) > 1:
            L = np.array([x[0] for x in rows]); C = np.log([x[1] for x in rows])
            zs_l += list((L - L.mean()) / (L.std() + 1e-12)); zs_c += list((C - C.mean()) / (C.std() + 1e-12))
    out["C1_pooled_within_model"] = spear(zs_l, zs_c)
    json.dump(out, open(a.out, "w"), indent=1)
    for t in a.tags:
        m = out["models"][t]
        print(f"{t:13s} runs {m['n']} | C2 leaky>=2x clean in {m['support_vs_clean']}/{m['n']} -> {m['holds_vs_clean']} | "
              f"vs rand {m['support_vs_rand']}/{m['n']} | C1 rho(Lambda, CR) {m['C1_rho_Lambda_CR']:+.2f}")
        for r in m["runs"]:
            q = r["q0.2"]
            print(f"   {r['run']:22s} CR {r['CR']:.2f} Lam {r['Lambda_run']:+.2f} mlp share off/on {r['mlp_share_off']:.2f}/{r['mlp_share_on']:.2f} | "
                  f"eff leaky {q['eff_leaky']:.2f} clean {q['eff_clean']:.2f} rand {q['eff_rand_med']:.2f} "
                  f"(f_off/f_on leaky {q['f_off_leaky']:.2f}/{q['f_on_leaky']:.2f}, clean {q['f_off_clean']:.2f}/{q['f_on_clean']:.2f})")
    print("C2", out["C2"], "| secondary vs random holds in", out["C2_secondary_vs_rand"], "| C1 pooled", round(out["C1_pooled_within_model"], 2))


if __name__ == "__main__":
    main()
