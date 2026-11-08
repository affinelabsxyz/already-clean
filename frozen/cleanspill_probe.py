"""CLEANSPILL probe: per MLP unit, cleanliness (text-kind selectivity), downstream gain of its write (depth), and the
collateral of a matched-size edit. One code path for every model. See notes/CLEANSPILL_PREREG.md.

python cleanspill_probe.py --model allenai/OLMo-2-0425-1B --tag olmo2-1b --out results/cleanspill_olmo2-1b.json
       [--data-dir <curated domain pools>] [--smoke]
"""
import argparse, hashlib, json, math, os, random, sys, time
import numpy as np
import torch
import torch.nn.functional as F

KINDS = ["wiki", "math_easy", "math_hard", "code", "medical", "finance", "logic", "if"]
SEQ = 256            # tokens per sequence, BOS included when the tokenizer adds one
WIN = {"A": (0, 16), "B": (16, 18), "C": (18, 22)}   # chunk index ranges per kind
EPS = 0.02
TOP_FRAC = 0.02      # rule B switch-on: unit in its floor's top 2% by |s_i| * ||o_eff_i|| at that token
MIN_USE = 0.005      # eligible units: switched on at >= 0.5% of window-A tokens ...
MAX_USE = 0.20       # ... and at most 20% (keeps cleanliness from being a proxy for rarity)
N_BINS, PER_SIDE = 6, 20
N_RAND = 40
N_PROMO = 5
SEED = 0
V1_TOL = 0.05


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


# ---------------------------------------------------------------- text
def kind_streams(data_dir, n_items=4000):
    from datasets import load_dataset
    out = {}
    w = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    out["wiki"] = "\n".join(t for t in w["text"] if t.strip())
    for k in KINDS[1:]:
        rows = []
        with open(os.path.join(data_dir, k, "pool.jsonl"), encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= n_items:
                    break
                r = json.loads(line)
                a = r.get("answer")
                a = "" if a is None else (a if isinstance(a, str) else json.dumps(a, ensure_ascii=False))
                rows.append((r.get("question") or "") + ("\n" + a if a else ""))
        out[k] = "\n\n".join(rows)
    return out


def build_windows(tok, streams):
    bos = tok.bos_token_id
    add_bos = bos is not None and tok("a").input_ids[:1] == [bos]
    body = SEQ - (1 if add_bos else 0)
    need = max(b for _, b in WIN.values())
    win = {w: {"ids": [], "kind": [], "chunk": []} for w in WIN}
    for ki, k in enumerate(KINDS):
        ids = tok(streams[k], add_special_tokens=False).input_ids
        if len(ids) < need * body:
            raise SystemExit(f"kind {k}: only {len(ids)} tokens, need {need * body}")
        for c in range(need):
            seq = ([bos] if add_bos else []) + ids[c * body:(c + 1) * body]
            for w, (a, b) in WIN.items():
                if a <= c < b:
                    win[w]["ids"].append(seq); win[w]["kind"].append(ki); win[w]["chunk"].append(c)
    for w in win:
        win[w]["ids"] = torch.tensor(win[w]["ids"])
        win[w]["kind"] = torch.tensor(win[w]["kind"])
        win[w]["chunk"] = torch.tensor(win[w]["chunk"])
    return win, add_bos


# ---------------------------------------------------------------- model plumbing
class Plumb:
    def __init__(self, model):
        self.m = model
        base = model.model
        self.base = base
        self.layers = base.layers
        self.L = len(self.layers)
        self.final_norm = base.norm
        self.post = [getattr(l, "post_feedforward_layernorm", None) for l in self.layers]
        self.gemma = "Gemma" in type(model).__name__
        W = [l.mlp.down_proj.weight.detach().float() for l in self.layers]          # [d, ff]
        self.o_eff = []
        for l, Wl in enumerate(W):
            g = self._gain(self.post[l]) if self.post[l] is not None else None
            self.o_eff.append(Wl * g[:, None] if g is not None else Wl)
        self.o_norm = [o.norm(dim=0) for o in self.o_eff]                             # [ff]
        self.gf = self._gain(self.final_norm)
        self.WU = model.get_output_embeddings().weight.detach().float()             # [V, d]

    def _gain(self, norm):
        w = norm.weight.detach().float()
        return (1.0 + w) if self.gemma else w

    def write_module(self, l):
        return self.post[l] if self.post[l] is not None else self.layers[l].mlp


def hs_from(args, kwargs):
    return args[0] if args else kwargs["hidden_states"]


def out_tensor(o):
    return o[0] if isinstance(o, tuple) else o


def with_tensor(o, t):
    return (t,) + tuple(o[1:]) if isinstance(o, tuple) else t


@torch.no_grad()
def run(model, ids, bs, logits=True):
    outs = []
    for i in range(0, ids.shape[0], bs):
        x = ids[i:i + bs].cuda()
        if logits:
            outs.append(model(input_ids=x, use_cache=False).logits.float())
        else:
            model.model(input_ids=x, use_cache=False)
    return outs


# ---------------------------------------------------------------- stage 1: selectivity (window A)
@torch.no_grad()
def selectivity(model, P, win, bs):
    ids, kind, chunk = win["A"]["ids"], win["A"]["kind"], win["A"]["chunk"]
    K = len(KINDS)
    half = (chunk % 2).numpy()                          # split-half by chunk parity
    ff = [o.shape[0] for o in P.o_norm]
    cnt = [torch.zeros(2, K, n, device="cuda") for n in ff]
    tok_per = np.zeros((2, K))
    state = {}

    def mk(l):
        def pre(mod, args):
            s = args[0].float()                          # [b, T, ff]
            z = s.abs() * P.o_norm[l]
            k = max(1, int(round(TOP_FRAC * z.shape[-1])))
            thr = z.topk(k, dim=-1).values[..., -1:]
            on = (z >= thr)
            on[:, 0, :] = False                          # position 0 excluded
            for j in range(on.shape[0]):
                cnt[l][state["half"][j], state["kind"][j]] += on[j].sum(0).float()
        return pre
    hs = [P.layers[l].mlp.down_proj.register_forward_pre_hook(mk(l)) for l in range(P.L)]
    try:
        for i in range(0, ids.shape[0], bs):
            state["half"] = half[i:i + bs]; state["kind"] = kind[i:i + bs].numpy()
            for j in range(len(state["half"])):
                tok_per[state["half"][j], state["kind"][j]] += SEQ - 1
            model.model(input_ids=ids[i:i + bs].cuda(), use_cache=False)
    finally:
        for h in hs:
            h.remove()
    res = []
    tp = torch.tensor(tok_per, device="cuda", dtype=torch.float32)
    for l in range(P.L):
        c = cnt[l]                                       # [2, K, ff]
        use = c.sum((0, 1)) / tp.sum()
        def S_of(cc, tt):
            rate = cc / tt[:, None]                      # [K, ff]
            p = rate / rate.sum(0, keepdim=True).clamp_min(1e-12)
            H = -(p * p.clamp_min(1e-12).log()).sum(0)
            return 1 - H / math.log(K), p
        S, p = S_of(c.sum(0), tp.sum(0))
        S0, _ = S_of(c[0], tp[0]); S1, _ = S_of(c[1], tp[1])
        res.append({"use": use.cpu(), "S": S.cpu(), "S0": S0.cpu(), "S1": S1.cpu(), "p": p.cpu()})
    return res


def pick_units(P, sel, rng):
    units = []
    for b in range(N_BINS):
        f0, f1 = (b * P.L) // N_BINS, ((b + 1) * P.L) // N_BINS
        pool = []
        for l in range(f0, f1):
            ok = ((sel[l]["use"] >= MIN_USE) & (sel[l]["use"] <= MAX_USE)).nonzero().flatten().tolist()
            pool += [(float(sel[l]["S"][i]), rng.random(), l, i) for i in ok]
        pool.sort()
        lo, hi = pool[:PER_SIDE], pool[-PER_SIDE:] if len(pool) >= 2 * PER_SIDE else pool[PER_SIDE:]
        for side, grp in (("low", lo), ("high", hi)):
            for s, _, l, i in grp:
                units.append({"bin": b, "side": side, "floor": l, "unit": i})
    return units


# ---------------------------------------------------------------- stage 2: downstream gain of a push (window B)
@torch.no_grad()
def push_gain(model, P, win, bs, dirs_by_floor):
    """dirs_by_floor: {l: [unit_dir (d,) ...]} -> {l: [(G_dir, total, side, kappa, G_dir_h0, G_dir_h1), ...]}
    G_dir = median over tokens of |final change orthogonal to the clean final state| / |clean final state| / EPS:
    the change in what the final norm reads, per unit of relative push."""
    ids, chunk = win["B"]["ids"].cuda(), win["B"]["chunk"]
    hz = (chunk % 2).numpy()
    cap = {}

    def fin_pre(mod, args, kwargs):
        cap["xf"] = hs_from(args, kwargs).float()
    hf = P.final_norm.register_forward_pre_hook(fin_pre, with_kwargs=True)
    xl_out = {}

    def lay_out(l):
        def h(mod, args, kwargs, out):
            xl_out[l] = out_tensor(out).float().norm(dim=-1, keepdim=True)
        return h
    hl = [P.layers[l].register_forward_hook(lay_out(l), with_kwargs=True) for l in range(P.L)]
    model.model(input_ids=ids, use_cache=False)
    for h in hl:
        h.remove()
    x0 = cap["xf"].clone()
    x0h = x0[:, 1:] / x0[:, 1:].norm(dim=-1, keepdim=True)
    x0n = x0[:, 1:].norm(dim=-1)
    res = {}
    try:
        for l, dirs in dirs_by_floor.items():
            nrm = xl_out[l]                                   # [B, T, 1]
            out_l = []
            for u in dirs:
                uh = (u / u.norm()).to(torch.float32)
                delta = EPS * nrm * uh                        # [B, T, d]
                delta[:, 0, :] = 0

                def add(mod, args, kwargs, out):
                    t = out_tensor(out)
                    return with_tensor(out, t + delta.to(t.dtype))
                h = P.layers[l].register_forward_hook(add, with_kwargs=True)
                model.model(input_ids=ids, use_cache=False)
                h.remove()
                D = (cap["xf"] - x0)[:, 1:]                   # [B, T-1, d]
                dl = delta[:, 1:]
                dn = dl.norm(dim=-1)
                r = D - dl
                along = (r * dl).sum(-1) / dn.pow(2)
                side = (r - along[..., None] * dl).norm(dim=-1) / dn
                total = D.norm(dim=-1) / dn
                kap = -along
                Dp = D - (D * x0h).sum(-1, keepdim=True) * x0h
                gdir = Dp.norm(dim=-1) / x0n / EPS
                m0 = torch.tensor(hz == 0, device=total.device)
                out_l.append((gdir.median().item(), total.median().item(), side.median().item(), kap.median().item(),
                              gdir[m0].median().item(), gdir[~m0].median().item()))
            res[l] = out_l
    finally:
        hf.remove()
    return res


# ---------------------------------------------------------------- stage 3: matched-size edits (window C)
@torch.no_grad()
def clean_pass_C(model, P, win, bs, floors_needed, unit_sets):
    ids = win["C"]["ids"]
    st = {"logp": [], "write": {l: [] for l in floors_needed}, "rms": {l: [] for l in floors_needed},
          "xin": {l: [] for l in floors_needed}, "s": {l: [] for l in floors_needed}, "on": {l: [] for l in floors_needed}}
    hooks = []
    for l in floors_needed:
        us = torch.tensor(unit_sets[l], device="cuda")

        def mk_pre(l=l, us=us):
            def pre(mod, args):
                s = args[0].float()
                z = s.abs() * P.o_norm[l]
                k = max(1, int(round(TOP_FRAC * z.shape[-1])))
                thr = z.topk(k, dim=-1).values[..., -1:]
                st["s"][l].append(s[..., us].cpu()); st["on"][l].append((z[..., us] >= thr).cpu())
            return pre
        hooks.append(P.layers[l].mlp.down_proj.register_forward_pre_hook(mk_pre()))

        def mk_mlp(l=l):
            def h(mod, args, out):
                st["rms"][l].append(out.float().pow(2).mean(-1).sqrt().cpu())
            return h
        hooks.append(P.layers[l].mlp.register_forward_hook(mk_mlp()))

        def mk_w(l=l):
            def h(mod, args, out):
                st["write"][l].append(out.float().cpu())
            return h
        hooks.append(P.write_module(l).register_forward_hook(mk_w()))

        def mk_in(l=l):
            def pre(mod, args, kwargs):
                st["xin"][l].append(hs_from(args, kwargs).float().norm(dim=-1).cpu())
            return pre
        hooks.append(P.layers[l].register_forward_pre_hook(mk_in(), with_kwargs=True))
    try:
        for i in range(0, ids.shape[0], bs):
            lg = model(input_ids=ids[i:i + bs].cuda(), use_cache=False).logits.float()
            st["logp"].append(F.log_softmax(lg, -1).cpu())
    finally:
        for h in hooks:
            h.remove()
    st["logp"] = torch.cat(st["logp"])
    for key in ("write", "rms", "xin", "s", "on"):
        st[key] = {l: torch.cat(v) for l, v in st[key].items()}
    return st


@torch.no_grad()
def edit_unit(model, P, win, bs, cl, l, j_local, i_unit, c, promo):
    """Scale unit i at floor l by (1+c) everywhere; return per-position KL [N,T-1], dlogp of promoted [N,T-1],
    realized write-change norm [N,T]."""
    ids = win["C"]["ids"]
    dw, kls, dps = [], [], []
    cur = {}

    def pre(mod, args):
        s = args[0].clone()
        s[..., i_unit] = s[..., i_unit] * (1.0 + c)
        return (s,) + tuple(args[1:])

    def wh(mod, args, out):
        cur["w"] = out.float()
    h1 = P.layers[l].mlp.down_proj.register_forward_pre_hook(pre)
    h2 = P.write_module(l).register_forward_hook(wh)
    try:
        for i in range(0, ids.shape[0], bs):
            lg = model(input_ids=ids[i:i + bs].cuda(), use_cache=False).logits.float()
            lp = F.log_softmax(lg, -1)
            lb = cl["logp"][i:i + bs].cuda().float()
            kl = (lb.exp() * (lb - lp)).sum(-1)[:, :-1]     # position t predicts t+1; keep t = 0..T-2
            dp = (lp[..., promo] - lb[..., promo]).mean(-1)[:, :-1]
            kls.append(kl.cpu()); dps.append(dp.cpu())
            dw.append((cur["w"] - cl["write"][l][i:i + bs].cuda()).norm(dim=-1).cpu())
    finally:
        h1.remove(); h2.remove()
    return torch.cat(kls), torch.cat(dps), torch.cat(dw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-dir", default=r"C:\Users\chaej\Desktop\works\evals\researches\data-curation\data")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--bs-logits", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="tiny budget; output marked smoke, never scored")
    a = ap.parse_args()
    global PER_SIDE, N_RAND
    if a.smoke:
        PER_SIDE, N_RAND = 2, 2
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(SEED); rng = random.Random(SEED)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).cuda().eval()
    P = Plumb(model)
    win, add_bos = build_windows(tok, kind_streams(a.data_dir))
    log = lambda *x: print(f"[{a.tag} {time.time() - t0:7.1f}s]", *x, flush=True)
    log("loaded", type(model).__name__, "L", P.L, "add_bos", add_bos, {w: tuple(v["ids"].shape) for w, v in win.items()})

    sel = selectivity(model, P, win, a.bs)
    units = pick_units(P, sel, rng)
    log("selectivity done; units", len(units))

    # downstream gain: unit directions and random directions per floor
    dirs = {}
    for u in units:
        dirs.setdefault(u["floor"], []).append(P.o_eff[u["floor"]][:, u["unit"]])
    nunit = {l: len(v) for l, v in dirs.items()}
    d = P.o_eff[0].shape[0]
    for l in range(P.L):
        gen = torch.Generator(device="cpu").manual_seed(1000 + l)
        rnd = [torch.randn(d, generator=gen).cuda() for _ in range(N_RAND)]
        dirs.setdefault(l, []).extend(rnd)
    g = push_gain(model, P, win, a.bs, dirs)
    log("push gain done")
    rand_floor = {}
    for l in range(P.L):
        k = nunit.get(l, 0)
        rr = g[l][k:]
        rand_floor[l] = {"G_dir_med": float(np.median([x[0] for x in rr])), "total_med": float(np.median([x[1] for x in rr])),
                         "side_med": float(np.median([x[2] for x in rr])), "kappa_med": float(np.median([x[3] for x in rr])),
                         "G_dir_all": [x[0] for x in rr]}
    cnt = {}
    for u in units:
        l = u["floor"]; j = cnt.get(l, 0); cnt[l] = j + 1
        gd, tot, side, kap, g0_, g1_ = g[l][j]
        u.update({"G_dir": gd, "G_total": tot, "G_side": side, "G_kappa": kap, "G_dir_h0": g0_, "G_dir_h1": g1_,
                  "S": float(sel[l]["S"][u["unit"]]), "S_h0": float(sel[l]["S0"][u["unit"]]),
                  "S_h1": float(sel[l]["S1"][u["unit"]]), "use": float(sel[l]["use"][u["unit"]]),
                  "p": [float(x) for x in sel[l]["p"][:, u["unit"]]]})
        u["home"] = int(np.argmax(u["p"]))

    # edits
    floors = sorted({u["floor"] for u in units})
    unit_sets = {l: [u["unit"] for u in units if u["floor"] == l] for l in floors}
    cl = clean_pass_C(model, P, win, a.bs_logits, floors, unit_sets)
    log("clean pass C done")
    kindC = win["C"]["kind"]
    n_ok = 0
    for u in units:
        l, i = u["floor"], u["unit"]; j = unit_sets[l].index(i)
        home = (kindC == u["home"])[:, None].expand(-1, SEQ).clone(); home[:, 0] = False
        on = cl["on"][l][..., j].clone(); on[:, 0] = False
        s = cl["s"][l][..., j]
        T = EPS * float(cl["xin"][l][home].median())
        lin = s.abs() * float(P.o_norm[l][i])
        if P.post[l] is not None:
            lin = lin / cl["rms"][l].clamp_min(1e-12)
        c = T / max(float(lin[home].mean()), 1e-12)
        ho = (home & on)
        sgn = float(torch.sign(s[ho].mean())) if ho.any() else float(torch.sign(s[home].mean()))
        promo_dir = sgn * P.gf * P.o_eff[l][:, i]
        promo = (P.WU @ promo_dir).topk(N_PROMO).indices
        tries = []
        for attempt in range(2):
            kl, dp, dw = edit_unit(model, P, win, a.bs_logits, cl, l, j, i, c, promo)
            D = float(dw[home].mean())
            tries.append({"c": c, "D": D})
            if abs(D / T - 1) <= V1_TOL or attempt == 1:
                break
            c = c * T / max(D, 1e-12)
        ok = abs(D / T - 1) <= V1_TOL
        n_ok += ok
        off = (kindC != u["home"])[:, None].expand(-1, SEQ)[:, :-1] & ~on[:, :-1]
        off[:, 0] = False
        hon = (home & on)[:, :-1]
        K_off = float(kl[off].mean()) if off.any() else float("nan")
        E_on = float(dp[hon].mean()) if hon.any() else float("nan")
        u.update({"T": T, "c": c, "D": D, "V1_ok": bool(ok), "tries": tries, "K_off": K_off,
                  "K_all": float(kl[:, 1:].mean()), "E_on": E_on,
                  "R": (K_off / E_on) if (E_on == E_on and E_on > 0) else float("nan"),
                  "n_off": int(off.sum()), "n_home_on": int(hon.sum()), "s_sign": sgn,
                  "promo": [tok.decode([t]) for t in promo.tolist()]})
    log("edits done; V1 ok", n_ok, "/", len(units))
    out = {"tag": a.tag, "model": a.model, "smoke": a.smoke, "script_sha256": sha_file(__file__),
           "torch": torch.__version__, "dtype": "float32", "tf32": False, "L": P.L, "add_bos": add_bos,
           "params": {"KINDS": KINDS, "SEQ": SEQ, "WIN": WIN, "EPS": EPS, "TOP_FRAC": TOP_FRAC, "MIN_USE": MIN_USE,
                      "MAX_USE": MAX_USE, "N_BINS": N_BINS, "PER_SIDE": PER_SIDE, "N_RAND": N_RAND, "N_PROMO": N_PROMO, "SEED": SEED},
           "post_norm": P.post[0] is not None, "rand_floor": rand_floor, "units": units,
           "seconds": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    log("wrote", a.out)


if __name__ == "__main__":
    main()
