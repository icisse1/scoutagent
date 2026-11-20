"""
pages/1_Evaluation.py: run the evaluation harness from the browser (owner only).

Protected by EVAL_PASSWORD (set in Streamlit Secrets or .env). Same code as eval/run_eval.py:
14 cases x k runs, pass@1 and pass^k, failure details, downloadable results.
"""
import json
import os
import sys

import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

try:
    for k, v in st.secrets.items():
        if isinstance(v, str):
            os.environ.setdefault(k, v)
except Exception:
    pass

from llm import LLM, detect_provider, load_dotenv  # noqa: E402
from agent import ScoutSystem  # noqa: E402
from run_eval import build_summary, run_all  # noqa: E402

load_dotenv()
st.set_page_config(page_title="ScoutAgent evaluation", page_icon="📊", layout="wide")
st.title("📊 Evaluation harness")
st.write("Owner only. Runs every test case k times through the full system and reports pass@1 and pass^k.")

pw = os.environ.get("EVAL_PASSWORD")
if not pw:
    st.info("Evaluation is disabled on this deployment (no EVAL_PASSWORD set).")
    st.stop()
if st.text_input("Password", type="password") != pw:
    st.stop()

cases = json.load(open(os.path.join(ROOT, "eval", "cases.json"), encoding="utf-8"))
c1, c2, c3 = st.columns(3)
mode = c1.radio("System", ["Scout + Critic (Milestone II)", "Scout only (Milestone I baseline)"])
k = c2.number_input("Runs per case (k)", 1, 5, 3)
sleep = c3.number_input("Pause between runs (s)", 0, 30, int(os.environ.get("EVAL_SLEEP", 4)), help="Free tier allows about 10 requests per minute.")
tag = "critic" if mode.startswith("Scout +") else "baseline"

with st.expander(f"The {len(cases)} test cases"):
    st.dataframe([{"id": c["id"], "kind": c["kind"], "query": c["query"], "known good": c["known_good"]}
                  for c in cases], use_container_width=True, hide_index=True)

ss = st.session_state
ss.setdefault("eval_done", {})
done = ss.eval_done.setdefault(tag, {})
out_dir = os.path.join(ROOT, "eval", "results", tag)
os.makedirs(out_dir, exist_ok=True)
runs_path = os.path.join(out_dir, "runs.jsonl")
if not done and os.path.exists(runs_path):  # resume after a page refresh
    for line in open(runs_path, encoding="utf-8"):
        r = json.loads(line)
        done[(r["case"], r["run"])] = r

total = len(cases) * k
st.write(f"Finished runs: **{sum(1 for key in done if key[1] <= k)} / {total}** (finished runs are skipped, "
         "so you can press Run again to resume after a rate limit).")

if st.button("▶ Run evaluation", type="primary"):
    llm = LLM(detect_provider())
    system = ScoutSystem(llm, max_steps=6, use_critic=(tag == "critic"))
    bar = st.progress(0.0)
    log = st.container(height=300)

    def on_result(rec):
        n = sum(1 for key in done if key[1] <= k)
        bar.progress(min(1.0, n / total), text=f"{n}/{total} runs")
        log.write(f"{'✅' if rec['pass'] else '❌'} {rec['case']} run {rec['run']} · {rec['seconds']} s "
                  f"{'' if rec['pass'] else rec['fails']}")

    finished = run_all(system, cases, k, done, runs_path, sleep, on_result)
    if not finished:
        st.warning("The free-tier rate limit or daily quota was hit. Wait a few minutes (or until tomorrow) "
                   "and press Run again: finished runs are kept.")

if done:
    from types import SimpleNamespace
    from llm import PROVIDERS
    prov = detect_provider()
    label = SimpleNamespace(provider=prov, model=os.environ.get("SCOUT_MODEL") or PROVIDERS[prov][1])
    summary = build_summary(cases, done, k, label, tag)
    st.markdown(summary)
    runs_text = "\n".join(json.dumps(r, ensure_ascii=False) for r in done.values())
    d1, d2 = st.columns(2)
    d1.download_button("Download summary.md", summary, file_name=f"summary_{tag}.md")
    d2.download_button("Download runs.jsonl (all answers)", runs_text, file_name=f"runs_{tag}.jsonl")
