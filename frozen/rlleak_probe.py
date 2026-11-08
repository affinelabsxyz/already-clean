"""RLLEAK probe (notes/RLLEAK_PREREG.md): does an RL update's off-domain collateral come from the part of the update
that lands on leaky MLP units? For each existing GRPO-LoRA run (all trained on GSM8K-style maths), revert the MLP
update of matched-energy unit sets (most leaky / least leaky / random) and measure how much off-domain change and how
much on-domain change goes away.

python rlleak_probe.py --model Qwen/Qwen3.5-0.8B-Base --tag qwen35-0.8b --runs math_easy e3_floors0-7_s0 ... \
       --out results/rlleak_qwen35-0.8b.json [--smoke]
"""
import argparse, json, math, os, random, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cleanspill_probe as cp   # CLEANSPILL code (sha256 ed8a7d67) as a library: text windows, plumbing, constants

RUNS_DIR = r"C:\Users\chaej\Desktop\works\evals\researches\data-curation\runs_rl"
ON_KINDS = ["math_easy", "math_hard"]          # the training domain
QS = [0.2, 0.5]                                # share of the run's MLP update energy reverted per set (0.2 primary)
N_RAND = 3


@torch.no_grad()
def leakage_all(model, P, win, bs):
    """Window A, every MLP unit: mean s^2 over off-domain kinds (mean of kind means) / same over the maths kinds."""
    ids, kind = win["A"]["ids"], win["A"]["kind"]
    K = len(cp.KINDS)
    acc = [torch.zeros(K, P.o_norm[l].shape[0], device="cuda") for l in range(P.L)]
    n = torch.zeros(K, device="cuda")
    cur = {}

    def mk(l):
        def pre(mod, args):
            s2 = args[0].float()[:, 1:, :].pow(2).sum(1)
            for j in range(s2.shape[0]):
                acc[l][cur["kind"][j]] += s2[j]
        return pre
    hs = [P.layers[l].mlp.down_proj.register_forward_pre_hook(mk(l)) for l in range(P.L)]
    try:
        for i in range(0, ids.shape[0], bs):
            cur["kind"] = kind[i:i + bs].tolist()
            for k in cur["kind"]:
                n[k] += cp.SEQ - 1
            model.model(input_ids=ids[i:i + bs].cuda(), use_cache=False)
    finally:
        for h in hs:
            h.remove()
    on = [cp.KINDS.index(k) for k in ON_KINDS]
    off = [k for k in range(K) if k not in on]
    out = []
    for l in range(P.L):
        m = acc[l] / n[:, None]
        out.append((m[off].mean(0) / m[on].mean(0).clamp_min(1e-30)).cpu())
    return out


@torch.no_grad()
def kl_by_kind(model, win, logp, bs):
    """Mean next-token KL(base || model) over positions 1..T-2, per window-C sequence."""
    ids = win["C"]["ids"]
    per = []
    for i in range(0, ids.shape[0], bs):
        lp = F.log_softmax(model(input_ids=ids[i:i + bs].cuda(), use_cache=False).logits.float(), -1)
        lb = logp[i:i + bs].cuda()
        per.append((lb.exp() * (lb - lp)).sum(-1)[:, 1:-1].mean(-1).cpu())
    per = torch.cat(per)
    kind = win["C"]["kind"]
    onm = torch.tensor([cp.KINDS[k] in ON_KINDS for k in kind.tolist()])
    return float(per[~onm].mean()), float(per[onm].mean())


def mlp_mats(P, l):
    m = P.layers[l].mlp
    return m.gate_proj.weight, m.up_proj.weight, m.down_proj.weight


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-dir", default=r"C:\Users\chaej\Desktop\works\evals\researches\data-curation\data")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--bs-logits", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    t0 = time.time()
    log = lambda *x: print(f"[{a.tag} {time.time() - t0:7.1f}s]", *x, flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    load = lambda: AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).cuda().eval()
    model = load()
    P = cp.Plumb(model)
    win, _ = cp.build_windows(tok, cp.kind_streams(a.data_dir))
    if a.smoke:
        win["C"] = {k: v[::4] for k, v in win["C"].items()}
    lam = leakage_all(model, P, win, a.bs)
    log("leakage done")
    logp = []
    with torch.no_grad():
        for i in range(0, win["C"]["ids"].shape[0], a.bs_logits):
            logp.append(F.log_softmax(model(input_ids=win["C"]["ids"][i:i + a.bs_logits].cuda(), use_cache=False).logits.float(), -1).cpu())
    logp = torch.cat(logp)
    base_mlp = [[w.detach().cpu().clone() for w in mlp_mats(P, l)] for l in range(P.L)]
    del model, P
    torch.cuda.empty_cache()
    log("base cached")
    out_runs = []
    for run in a.runs:
        model = load()
        pm = PeftModel.from_pretrained(model, os.path.join(RUNS_DIR, run, "adapter"))
        model = pm.merge_and_unload().eval()
        P = cp.Plumb(model)
        rl_mlp = [[w.detach().cpu().clone() for w in mlp_mats(P, l)] for l in range(P.L)]
        # per-unit relative update energy
        m_all, lam_all, idx = [], [], []
        for l in range(P.L):
            (gb, ub, db), (gr, ur, dr) = base_mlp[l], rl_mlp[l]
            m = ((gr - gb).pow(2).sum(1) / gb.pow(2).sum(1).clamp_min(1e-30)
                 + (ur - ub).pow(2).sum(1) / ub.pow(2).sum(1).clamp_min(1e-30)
                 + (dr - db).pow(2).sum(0) / db.pow(2).sum(0).clamp_min(1e-30))
            m_all.append(m); lam_all.append(lam[l]); idx += [(l, i) for i in range(m.shape[0])]
        M = torch.cat(m_all).numpy(); LAM = torch.cat(lam_all).numpy()
        tot = float(M.sum())
        LOG = np.log(np.clip(LAM, 1e-30, None))
        upd = M > 0
        Lam_run = float((M[upd] * LOG[upd]).sum() / M[upd].sum()) if upd.any() else float("nan")

        def take(order, q):
            c = np.cumsum(M[order]); k = int(np.searchsorted(c, q * tot)) + 1
            return order[:k]

        order_leak = np.argsort(-LOG, kind="stable"); order_leak = order_leak[upd[order_leak]]
        order_clean = np.argsort(LOG, kind="stable"); order_clean = order_clean[upd[order_clean]]
        rng = np.random.default_rng(cp.SEED)

        def revert(sel):
            by = {}
            for g in sel:
                l, i = idx[g]; by.setdefault(l, []).append(i)
            for l, us in by.items():
                us_t = torch.tensor(us)
                g_, u_, d_ = mlp_mats(P, l)
                g_.data[us_t] = base_mlp[l][0][us_t].to(g_.device); u_.data[us_t] = base_mlp[l][1][us_t].to(u_.device)
                d_.data[:, us_t] = base_mlp[l][2][:, us_t].to(d_.device)
            return by

        def restore(by):
            for l, us in by.items():
                us_t = torch.tensor(us)
                g_, u_, d_ = mlp_mats(P, l)
                g_.data[us_t] = rl_mlp[l][0][us_t].to(g_.device); u_.data[us_t] = rl_mlp[l][1][us_t].to(u_.device)
                d_.data[:, us_t] = rl_mlp[l][2][:, us_t].to(d_.device)

        rec = {"run": run, "Lambda_run": Lam_run, "E_total": tot, "n_units": int(len(M)), "n_updated": int(upd.sum())}
        rec["rl"] = dict(zip(("K_off", "K_on"), kl_by_kind(model, win, logp, a.bs_logits)))
        # full MLP revert (attention updates kept): how much of the change is the MLP's at all
        by = revert(np.arange(len(M))[upd]); rec["rev_all_mlp"] = dict(zip(("K_off", "K_on"), kl_by_kind(model, win, logp, a.bs_logits))); restore(by)
        for q in QS:
            conds = {"leaky": take(order_leak, q), "clean": take(order_clean, q)}
            for r_ in range(N_RAND):
                perm = rng.permutation(np.arange(len(M))[upd]); conds[f"rand{r_}"] = take(perm, q)
            for name, sel in conds.items():
                by = revert(sel)
                k_off, k_on = kl_by_kind(model, win, logp, a.bs_logits)
                restore(by)
                rec[f"q{q}_{name}"] = {"K_off": k_off, "K_on": k_on, "n_units": int(len(sel)),
                                       "energy": float(M[sel].sum() / tot), "mean_log_leak": float(LOG[sel].mean())}
        out_runs.append(rec)
        log(run, "done", "units", rec["n_units"], "updated", rec["n_updated"], "sets", {k: v["n_units"] for k, v in rec.items() if k.startswith("q")})
        del model, P, pm
        torch.cuda.empty_cache()
    out = {"tag": a.tag, "model": a.model, "smoke": a.smoke, "script_sha256": cp.sha_file(__file__),
           "library_sha256": cp.sha_file(cp.__file__), "ON_KINDS": ON_KINDS, "QS": QS, "N_RAND": N_RAND,
           "runs": out_runs, "seconds": round(time.time() - t0, 1)}
    json.dump(out, open(a.out, "w"), indent=1)
    log("wrote", a.out)


if __name__ == "__main__":
    main()
