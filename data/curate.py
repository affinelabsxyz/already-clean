"""Build the shared RL pools and held-out eval sets for the 7 RLPRED domains.

Every record: {id, domain, split, source, question, answer, answer_type, meta}.
`question` is the raw problem text with no prompt template; templates live in the eval step so that every model
and machine uses the same ones. Pools are decontaminated against every eval set by 13-gram overlap.

Usage: .venv/Scripts/python.exe curate.py [--out data]
"""
import argparse
import ast
import hashlib
import json
import random
import re
import urllib.request
from pathlib import Path

import reasoning_gym as rg  # must load before `datasets`: the reverse order segfaults on Windows

from datasets import load_dataset

SEED = 0
NGRAM = 13
MEDMCQA_POOL = 20_000
GSM8K_EVAL = 300
MEDQA_EVAL = 500
RG_TASKS = ["knights_knaves", "zebra_puzzles", "mini_sudoku", "syllogism", "propositional_logic", "family_relationships"]
RG_POOL_PER_TASK = 1500
RG_EVAL_PER_TASK = 50
FINQA_URL = "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/{split}.json"


def rec(domain, split, source, question, answer, answer_type, meta=None, key=None):
    key = key or question
    rid = hashlib.sha1(f"{source}|{key}".encode()).hexdigest()[:16]
    return {"id": rid, "domain": domain, "split": split, "source": source, "question": question,
            "answer": answer, "answer_type": answer_type, "meta": meta or {}}


def parse(value):
    """Some HF columns hold Python-literal strings instead of lists."""
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def strip_user(prompt):
    return re.sub(r"^\s*user:\s*", "", prompt)


# ---------- domains ----------

def math_easy():
    ds = load_dataset("openai/gsm8k", "main")
    gold = lambda a: a.split("####")[-1].strip().replace(",", "")
    pool = [rec("math_easy", "pool", "gsm8k/train", r["question"], gold(r["answer"]), "number") for r in ds["train"]]
    test = [rec("math_easy", "eval", "gsm8k/test", r["question"], gold(r["answer"]), "number") for r in ds["test"]]
    return pool, random.Random(SEED).sample(test, GSM8K_EVAL)


def math_hard():
    pool = [rec("math_hard", "pool", "dolci-rl-zero-math", r["prompt"], r["ground_truth"], "math",
                {"origin": r["dataset"]}) for r in load_dataset("allenai/Dolci-RL-Zero-Math-7B", split="train")]
    ev = [rec("math_hard", "eval", "math-500", r["problem"], r["answer"], "math",
              {"subject": r["subject"], "level": int(r["level"]), "unique_id": r["unique_id"]})
          for r in load_dataset("HuggingFaceH4/MATH-500", split="test")]
    return pool, ev


def code():
    pool = []
    for r in load_dataset("allenai/Dolci-RL-Zero-Code-7B", split="train"):
        tests = parse(r["ground_truth"])
        if isinstance(tests, list) and tests and isinstance(tests[0], str) and tests[0].startswith("["):
            tests = parse(tests[0])
        kind = "code_stdio" if tests and isinstance(tests[0], dict) else "code_asserts"
        pool.append(rec("code", "pool", "dolci-rl-zero-code", strip_user(r["prompt"]), tests, kind,
                        {"reference": r["solution"]}, key=r["id"]))
    ev = []
    for r in load_dataset("evalplus/humanevalplus", split="test"):
        ev.append(rec("code", "eval", "humaneval+", r["prompt"], {"test": r["test"], "entry_point": r["entry_point"]},
                      "code_humaneval", {"task_id": r["task_id"], "reference": r["prompt"] + r["canonical_solution"]},
                      key=r["task_id"]))
    for r in load_dataset("evalplus/mbppplus", split="test"):
        ev.append(rec("code", "eval", "mbpp+", r["prompt"], {"test": r["test"], "test_list": parse(r["test_list"]),
                      "test_imports": parse(r["test_imports"])}, "code_mbpp", {"task_id": r["task_id"], "reference": r["code"]},
                      key=str(r["task_id"])))
    return pool, ev


def logic():
    def build(split, seed, n):
        out = []
        for task in RG_TASKS:
            ds = rg.create_dataset(task, size=n, seed=seed)
            for i in range(n):
                e = ds[i]
                out.append(rec("logic", split, f"reasoning-gym/{task}", e["question"], e["answer"], "reasoning_gym",
                               {"task": task, "seed": seed, "index": i, "size": n}, key=f"{task}|{seed}|{i}"))
        return out
    return build("pool", 1000, RG_POOL_PER_TASK), build("eval", 9, RG_EVAL_PER_TASK)


def instruction_following():
    pool = []
    for r in load_dataset("allenai/Dolci-RL-Zero-IF-7B", split="train"):
        gt = parse(r["ground_truth"])
        if isinstance(gt, list) and gt and isinstance(gt[0], str):
            gt = parse(gt[0])
        if isinstance(gt, list) and gt and isinstance(gt[0], dict):
            gt = gt[0]
        pool.append(rec("if", "pool", "dolci-rl-zero-if", strip_user(r["prompt"]), gt, "ifeval",
                        {"constraint": r["constraint"]}, key=r["key"]))
    ev = [rec("if", "eval", "ifeval", r["prompt"],
              {"instruction_id": list(parse(r["instruction_id_list"])), "kwargs": list(parse(r["kwargs"]))},
              "ifeval", {"key": r["key"]}, key=str(r["key"])) for r in load_dataset("google/IFEval", split="train")]
    return pool, ev


def mcq_text(question, options):
    return question.strip() + "\n" + "\n".join(f"{k}. {v}" for k, v in options.items())


def medical():
    rows = [r for r in load_dataset("openlifescienceai/medmcqa", split="train") if r["choice_type"] == "single"]
    rows = random.Random(SEED).sample(rows, MEDMCQA_POOL)
    pool = [rec("medical", "pool", "medmcqa/train",
                mcq_text(r["question"], {"A": r["opa"], "B": r["opb"], "C": r["opc"], "D": r["opd"]}),
                "ABCD"[int(r["cop"])], "mcq", {"subject": r["subject_name"]}, key=r["id"]) for r in rows]
    test = [rec("medical", "eval", "medqa-usmle/test", mcq_text(r["question"], parse(r["options"])),
                r["answer_idx"], "mcq", {"step": r["meta_info"]}) for r in
            load_dataset("GBaker/MedQA-USMLE-4-options", split="test")]
    return pool, random.Random(SEED).sample(test, MEDQA_EVAL)


def finance():
    def load(split):
        with urllib.request.urlopen(FINQA_URL.format(split=split)) as f:
            return json.load(f)
    def to_rec(r, split):
        table = "\n".join(" | ".join(map(str, row)) for row in r["table"])
        context = " ".join(r["pre_text"]) + "\n\n" + table + "\n\n" + " ".join(r["post_text"])
        question = context.strip() + "\n\nQuestion: " + r["qa"]["question"]
        return rec("finance", split, f"finqa/{'train' if split == 'pool' else 'test'}", question,
                   r["qa"]["exe_ans"], "number_tolerance", {"program": r["qa"].get("program")}, key=r["id"])
    return [to_rec(r, "pool") for r in load("train")], [to_rec(r, "eval") for r in load("test")]


DOMAINS = {"math_easy": math_easy, "math_hard": math_hard, "code": code, "logic": logic,
           "if": instruction_following, "medical": medical, "finance": finance}


# ---------- decontamination ----------

def ngrams(text, n=NGRAM):
    words = re.findall(r"\w+", text.lower())
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


# Pools are checked against these eval sets. Templated domains (procedural logic, IF constraint phrasing) share
# boilerplate across unrelated items, so 13-gram overlap would flag most of them; they use exact-question matching.
DECONTAM_AGAINST = {"math_easy": ["math_easy", "math_hard"], "math_hard": ["math_easy", "math_hard"],
                    "code": ["code"], "medical": ["medical"], "finance": ["finance"]}
EXACT_ONLY = {"logic", "if"}


def decontam_text(r):
    """FinQA train and test share filing text by design (same 10-K reports); compare only the question there."""
    if r["domain"] == "finance":
        return r["question"].rsplit("Question:", 1)[-1]
    return r["question"]


def decontaminate(pools, evals):
    removed = {}
    for name, pool in pools.items():
        if name in EXACT_ONLY:
            seen = {r["question"].strip() for r in evals.get(name, [])}
            kept = [r for r in pool if r["question"].strip() not in seen]
        else:
            banned = set()
            for ev_name in DECONTAM_AGAINST.get(name, [name]):
                for r in evals.get(ev_name, []):
                    banned |= ngrams(decontam_text(r))
            kept = [r for r in pool if not (ngrams(decontam_text(r)) & banned)]
        removed[name] = len(pool) - len(kept)
        pools[name] = kept
    return removed


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", errors="replace") as f:  # a few code rows hold lone surrogates
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--only", nargs="*", default=list(DOMAINS))
    args = ap.parse_args()
    out = Path(args.out)
    pools, evals = {}, {}
    for name in args.only:
        print(f"[{name}] loading", flush=True)
        pools[name], evals[name] = DOMAINS[name]()
    removed = decontaminate(pools, evals)
    mpath = out / "manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {"domains": {}}
    manifest.update({"seed": SEED, "ngram": NGRAM})
    for name in args.only:
        write_jsonl(out / name / "pool.jsonl", pools[name])
        write_jsonl(out / name / "eval.jsonl", evals[name])
        manifest["domains"][name] = {"pool": len(pools[name]), "eval": len(evals[name]),
                                     "pool_removed_by_decontam": removed[name],
                                     "pool_sources": sorted({r["source"] for r in pools[name]}),
                                     "eval_sources": sorted({r["source"] for r in evals[name]})}
        print(f"[{name}] pool {len(pools[name])} (-{removed[name]} decontam), eval {len(evals[name])}", flush=True)
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
