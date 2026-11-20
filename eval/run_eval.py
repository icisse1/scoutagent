"""
eval/run_eval.py: evaluation harness with pass@1 and pass^k.

Runs every test case in eval/cases.json k times (default 3) through the full system
(Scout + Critic + human gate, gate answered automatically per case), checks the final
answer against the known-good answer, and writes:
  eval/results/runs.jsonl    one line per run (resumable: finished runs are skipped)
  eval/results/summary.md    table with pass@1 and pass^k per case, plus failure details

Definitions (k runs per case):
  pass@1 = chance that ONE run passes      = mean over cases of (passing runs / k)
  pass^k = chance that ALL k runs pass     = share of cases where every run passed
  pass@k = chance that AT LEAST one passes  (shown for reference)

Usage (from the project folder, with a free key in .env):
  python eval/run_eval.py                    # all cases, k=3, Scout + Critic
  python eval/run_eval.py --no-critic --tag baseline   # Milestone I single-agent baseline
  python eval/run_eval.py --cases C01_seriea_xg A12_prompt_injection --k 3
"""
import argparse
import json
import os
import re
import sys
import time
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from agent import ScoutSystem, run_headless  # noqa: E402
from llm import LLM, PROVIDERS, detect_provider  # noqa: E402


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return " ".join(s.split())


def check(case, state):
    """Return (passed, list of failed checks)."""
    c = case["checks"]
    tr = state["trace"]
    ans = tr["final_answer"] or ""
    a = norm(ans)
    fails = []
    for key in ("include_all",):
        for s in c.get(key, []):
            if norm(s) not in a:
                fails.append(f"missing '{s}'")
    for key in ("include_any", "include_any_2"):
        if key in c and not any(norm(s) in a for s in c[key]):
            fails.append(f"none of {c[key]}")
    for s in c.get("exclude", []):
        if norm(s) in a:
            fails.append(f"leaked '{s}'")
    for key in ("regex_any", "regex_any_2"):
        if key in c and not any(re.search(p, ans, re.I) for p in c[key]):
            fails.append(f"no match for {key}")
    for p in c.get("regex_none", []):
        if re.search(p, ans, re.I):
            fails.append(f"forbidden pattern '{p}'")
    if "gate_requested_for" in c:
        reqs = [e for e in tr["events"] if e["type"] == "gate_request"]
        if not any(norm(c["gate_requested_for"]) in norm(json.dumps(e["args"])) for e in reqs):
            fails.append("human gate not requested for the right player")
    if c.get("watchlist_empty") and state["watchlist"]:
        fails.append(f"action executed without approval: {state['watchlist']}")
    if not ans.strip():
        fails.append("empty answer")
    return (not fails), fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--provider", choices=list(PROVIDERS), default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--no-critic", action="store_true")
    ap.add_argument("--tag", default=None, help="results subfolder name (default: critic or baseline)")
    ap.add_argument("--sleep", type=float, default=4, help="seconds between runs (free tier rate limits)")
    a = ap.parse_args()

    cases = json.load(open(os.path.join(HERE, "cases.json"), encoding="utf-8"))
    if a.cases:
        cases = [c for c in cases if c["id"] in a.cases]
    tag = a.tag or ("baseline" if a.no_critic else "critic")
    out_dir = os.path.join(HERE, "results", tag)
    os.makedirs(out_dir, exist_ok=True)
    runs_path = os.path.join(out_dir, "runs.jsonl")
    done = {}
    if os.path.exists(runs_path):
        for line in open(runs_path, encoding="utf-8"):
            r = json.loads(line)
            done[(r["case"], r["run"])] = r

    llm = LLM(a.provider or detect_provider(), a.model)
    system = ScoutSystem(llm, max_steps=6, use_critic=not a.no_critic)
    print(f"Model {llm.provider}/{llm.model} | critic {'off' if a.no_critic else 'on'} | "
          f"{len(cases)} cases x k={a.k}")

    run_all(system, cases, a.k, done, runs_path, a.sleep, on_result=lambda rec: print(
        f"{rec['case']} run {rec['run']}: {'PASS' if rec['pass'] else 'FAIL'} {rec['fails']} ({rec['seconds']} s)"))
    write_summary(cases, done, a.k, out_dir, llm, tag)


def run_one(system, case, run):
    t0 = time.time()
    try:
        state = run_headless(system, case["query"], gate_decision=case.get("gate_decision", "reject"), watchlist=[])
        ok, fails = check(case, state)
        tr = state["trace"]
        return {"case": case["id"], "kind": case["kind"], "run": run, "pass": ok, "fails": fails,
                "answer": tr["final_answer"], "tool_calls": tr.get("n_tool_calls"), "llm_calls": tr["llm_calls"],
                "stop_reason": tr["stop_reason"], "critic": tr.get("critic_verdict"), "tokens": tr["tokens"],
                "tools_used": [e["tool"] for e in tr["events"] if e["type"] == "tool"],
                "seconds": round(time.time() - t0, 1)}
    except Exception as e:  # an API error counts as a failed run, and is recorded
        return {"case": case["id"], "kind": case["kind"], "run": run, "pass": False,
                "fails": [f"error: {type(e).__name__}: {str(e)[:200]}"], "answer": None,
                "seconds": round(time.time() - t0, 1), "quota": "RateLimit" in type(e).__name__}


def run_all(system, cases, k, done, runs_path, sleep=4, on_result=None):
    """Run every case k times, skipping finished runs. Stops early on a quota error (resume later)."""
    for case in cases:
        for run in range(1, k + 1):
            if (case["id"], run) in done:
                continue
            rec = run_one(system, case, run)
            if rec.get("quota"):
                print("Rate limit / quota hit. Stopping; run again later to resume.")
                return False
            if runs_path:
                with open(runs_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done[(case["id"], run)] = rec
            if on_result:
                on_result(rec)
            time.sleep(sleep)
    return True


def write_summary(cases, done, k, out_dir, llm, tag):
    text = build_summary(cases, done, k, llm, tag)
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print("\n".join(text.splitlines()[:12]))
    print(f"\nSaved {os.path.join(out_dir, 'summary.md')}")


def build_summary(cases, done, k, llm, tag):
    rows, p1s, pks, pak = [], [], [], []
    for c in cases:
        runs = [done.get((c["id"], r)) for r in range(1, k + 1)]
        runs = [r for r in runs if r]
        if not runs:
            continue
        n = sum(r["pass"] for r in runs)
        p1, pk, pa = n / len(runs), float(n == len(runs) and len(runs) == k), float(n > 0)
        p1s.append(p1); pks.append(pk); pak.append(pa)
        rows.append(f"| {c['id']} | {c['kind']} | {' '.join('✅' if r['pass'] else '❌' for r in runs)} | "
                    f"{p1:.2f} | {int(pk)} | {c['known_good']} |")
    L = [f"# Evaluation results ({tag})", "",
         f"Model: {llm.provider}/{llm.model} · k = {k} runs per case · {len(rows)} cases", "",
         f"| Metric | Value |", "|---|---|",
         f"| **pass@1** (one run passes) | **{sum(p1s) / max(1, len(p1s)):.2f}** |",
         f"| **pass^{k}** (all {k} runs pass) | **{sum(pks) / max(1, len(pks)):.2f}** |",
         f"| pass@{k} (at least one passes) | {sum(pak) / max(1, len(pak)):.2f} |", "",
         "| Case | Kind | Runs | pass@1 | pass^k | Known-good answer |", "|---|---|---|---|---|---|"] + rows
    L += ["", "## Failed runs (read these)", ""]
    for c in cases:
        for r in range(1, k + 1):
            rec = done.get((c["id"], r))
            if rec and not rec["pass"]:
                ans = (rec.get("answer") or "").replace("\n", " ")[:400]
                L += [f"**{c['id']} run {r}**: {rec['fails']}  ", f"Tools: {rec.get('tools_used')}  ",
                      f"Answer: {ans}", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
