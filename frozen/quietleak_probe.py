"""QUIETLEAK probe (notes/QUIETLEAK_PREREG.md): on CLEANSPILL's units, (A) the same matched-size edit applied only at
the positions where the unit is on, and (B) each unit's leakage, its activation energy off its home kind over at home.

python quietleak_probe.py --model allenai/OLMo-2-0425-1B --tag olmo2-1b --units results/cleanspill_olmo2-1b.json \
       --out results/quietleak_olmo2-1b.json [--smoke]
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cleanspill_probe as cp   # unchanged CLEANSPILL code (sha256 ed8a7d67), used as a library

MIN_HOME_ON = 10   # units with fewer home-kind on positions in window C are not edited (too few to calibrate)
MAX_TRIES = 4      # first linear guess plus up to three secant corrections of the edit size


@torch.no_grad()
def leakage(model, P, win, bs, unit_sets, home_of):
    """Window A: per unit, mean s^2 on positions of other kinds / mean s^2 on home-kind positions (positions >= 1)."""
    ids, kind = win["A"]["ids"], win["A"]["kind"]
    K = len(cp.KINDS)
    acc = {l: torch.zeros(K, len(us), device="cuda") for l, us in unit_sets.items()}
    n = torch.zeros(K, device="cuda")
    cur = {}

    def mk(l, us):
        idx = torch.tensor(us, device="cuda")

        def pre(mod, args):
            s2 = args[0].float()[:, 1:, :][..., idx].pow(2).sum(1)      # [b, n_units]
            for j in range(s2.shape[0]):
                acc[l][cur["kind"][j]] += s2[j]
        return pre
    hs = [P.layers[l].mlp.down_proj.register_forward_pre_hook(mk(l, us)) for l, us in unit_sets.items()]
    try:
        for i in range(0, ids.shape[0], bs):
            cur["kind"] = kind[i:i + bs].tolist()
            for k in cur["kind"]:
                n[k] += cp.SEQ - 1
            model.model(input_ids=ids[i:i + bs].cuda(), use_cache=False)
    finally:
        for h in hs:
            h.remove()
    out = {}
    for l, us in unit_sets.items():
        mean = acc[l] / n[:, None]                                        # [K, n_units]
        for j, u in enumerate(us):
            h = home_of[(l, u)]
            other = torch.cat([mean[:h, j], mean[h + 1:, j]]).mean()
            out[(l, u)] = {"leak": float(other / mean[h, j].clamp_min(1e-30)),
                           "E_home": float(mean[h, j]), "E_other": float(other)}
    return out


@torch.no_grad()
def edit_on_only(model, P, win, bs, cl, l, j, i_unit, c, promo):
    """Scale unit i at floor l by (1+c) only where it is switched on (rule B, from the clean pass)."""
    ids = win["C"]["ids"]
    kls, dps, dw = [], [], []
    cur = {}

    def pre(mod, args):
        s = args[0].clone()
        m = cur["on"].to(s.device, s.dtype)
        s[..., i_unit] = s[..., i_unit] * (1.0 + c * m)
        return (s,) + tuple(args[1:])

    def wh(mod, args, out):
        cur["w"] = out.float()
    h1 = P.layers[l].mlp.down_proj.register_forward_pre_hook(pre)
    h2 = P.write_module(l).register_forward_hook(wh)
    try:
        for i in range(0, ids.shape[0], bs):
            on = cl["on"][l][i:i + bs, :, j].clone(); on[:, 0] = False
            cur["on"] = on
            lg = model(input_ids=ids[i:i + bs].cuda(), use_cache=False).logits.float()
            lp = F.log_softmax(lg, -1)
            lb = cl["logp"][i:i + bs].cuda().float()
            kls.append((lb.exp() * (lb - lp)).sum(-1)[:, :-1].cpu())
            dps.append((lp[..., promo] - lb[..., promo]).mean(-1)[:, :-1].cpu())
            dw.append((cur["w"] - cl["write"][l][i:i + bs].cuda()).norm(dim=-1).cpu())
    finally:
        h1.remove(); h2.remove()
    return torch.cat(kls), torch.cat(dps), torch.cat(dw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--units", required=True, help="CLEANSPILL result JSON for the same model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-dir", default=r"C:\Users\chaej\Desktop\works\evals\researches\data-curation\data")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--bs-logits", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(cp.SEED)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    log = lambda *x: print(f"[{a.tag} {time.time() - t0:7.1f}s]", *x, flush=True)
    src = json.load(open(a.units))
    assert src["script_sha256"].startswith("ed8a7d67") and not src["smoke"], "units must come from the frozen CLEANSPILL run"
    units = src["units"][:8] if a.smoke else src["units"]
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).cuda().eval()
    P = cp.Plumb(model)
    win, add_bos = cp.build_windows(tok, cp.kind_streams(a.data_dir))
    log("loaded", type(model).__name__, "units", len(units))
    floors = sorted({u["floor"] for u in units})
    unit_sets = {l: [u["unit"] for u in units if u["floor"] == l] for l in floors}
    home_of = {(u["floor"], u["unit"]): u["home"] for u in units}
    lk = leakage(model, P, win, a.bs, unit_sets, home_of)
    log("leakage done")
    cl = cp.clean_pass_C(model, P, win, a.bs_logits, floors, unit_sets)
    log("clean pass C done")
    kindC = win["C"]["kind"]
    rows = []
    for u in units:
        l, i = u["floor"], u["unit"]; j = unit_sets[l].index(i)
        r = {"floor": l, "unit": i, **lk[(l, i)]}
        home = (kindC == u["home"])[:, None].expand(-1, cp.SEQ).clone(); home[:, 0] = False
        on = cl["on"][l][..., j].clone(); on[:, 0] = False
        hon = home & on
        r["n_home_on"] = int(hon.sum())
        if r["n_home_on"] < MIN_HOME_ON:
            r["skipped"] = True; rows.append(r); continue
        s = cl["s"][l][..., j]
        T = cp.EPS * float(cl["xin"][l][home].median())
        lin = s.abs() * float(P.o_norm[l][i])
        if P.post[l] is not None:
            lin = lin / cl["rms"][l].clamp_min(1e-12)
        c = T / max(float(lin[hon].mean()), 1e-12)   # matched over the edited home positions (home & on)
        sgn = float(torch.sign(s[hon].mean()))
        promo = (P.WU @ (sgn * P.gf * P.o_eff[l][:, i])).topk(cp.N_PROMO).indices
        tries = []
        for attempt in range(MAX_TRIES):
            kl, dp, dw = edit_on_only(model, P, win, a.bs_logits, cl, l, j, i, c, promo)
            D = float(dw[hon].mean())
            tries.append({"c": c, "D": D})
            if abs(D / T - 1) <= cp.V1_TOL or attempt == MAX_TRIES - 1:
                break
            c = c * T / max(D, 1e-12)
        off = (kindC != u["home"])[:, None].expand(-1, cp.SEQ)[:, :-1] & ~on[:, :-1]
        off[:, 0] = False
        h_on = hon[:, :-1]
        K_off = float(kl[off].mean()) if off.any() else float("nan")
        E_on = float(dp[h_on].mean()) if h_on.any() else float("nan")
        r.update({"skipped": False, "T": T, "c": c, "D": D, "V1_ok": abs(D / T - 1) <= cp.V1_TOL, "tries": tries,
                  "K_off_onl": K_off, "E_on_onl": E_on, "K_all_onl": float(kl[:, 1:].mean()), "n_off": int(off.sum())})
        rows.append(r)
    log("on-only edits done; V1 ok", sum(r.get("V1_ok", False) for r in rows), "/", len(rows))
    out = {"tag": a.tag, "model": a.model, "smoke": a.smoke, "script_sha256": cp.sha_file(__file__),
           "library_sha256": cp.sha_file(cp.__file__), "units_from": os.path.basename(a.units),
           "units_from_sha256": cp.sha_file(a.units), "MIN_HOME_ON": MIN_HOME_ON, "MAX_TRIES": MAX_TRIES, "rows": rows,
           "seconds": round(time.time() - t0, 1)}
    json.dump(out, open(a.out, "w"), indent=1)
    log("wrote", a.out)


if __name__ == "__main__":
    main()
