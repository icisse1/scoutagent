"""
run.py: run ScoutAgent (Scout + Critic) from the terminal and save the trace.

Examples:
  python run.py "We need a left back strong in progressive passing"
  python run.py --no-critic "Who had the highest non-penalty xG per 90 in Serie A?"
  python run.py --gate approve "Add Dimitri Payet to our watchlist"
"""
import argparse

from agent import ScoutSystem, run_headless
from llm import LLM, PROVIDERS, detect_provider

p = argparse.ArgumentParser()
p.add_argument("query")
p.add_argument("--provider", choices=list(PROVIDERS), default=None)
p.add_argument("--model", default=None)
p.add_argument("--max-steps", type=int, default=6)
p.add_argument("--no-critic", action="store_true", help="Milestone I single-agent baseline")
p.add_argument("--gate", choices=["approve", "reject", "ask"], default="ask",
               help="What to do if the agent asks for approval (ask = type y/n)")
a = p.parse_args()

system = ScoutSystem(LLM(a.provider or detect_provider(), a.model), max_steps=a.max_steps,
                     use_critic=not a.no_critic, save=True)
watchlist = []
if a.gate == "ask":
    state = system.start(a.query, watchlist=watchlist)
    while state["status"] == "awaiting_approval":
        call = state["pending"]["call"]
        print(f"\nAPPROVAL NEEDED: {call['name']}({ {k: v for k, v in call['args'].items() if k != 'reasoning'} })")
        state = system.resume(state, approved=input("Approve? [y/N] ").strip().lower() == "y")
else:
    state = run_headless(system, a.query, gate_decision=a.gate, watchlist=watchlist)

tr = state["trace"]
for e in tr["events"]:
    if e["type"] == "tool":
        print(f"\n[{e['n']}] SCOUT {e['tool']}  reasoning: {e['reasoning']}\n{e['observation'][:600]}")
    elif e["type"] == "critic":
        print(f"\n[{e['n']}] CRITIC {e['verdict'].upper()} {e['problems']}")
    elif e["type"] in ("gate_request", "gate_decision", "guard"):
        print(f"\n[{e['n']}] {e['type'].upper()} {e.get('args', '')} {e.get('decision', '')} {e.get('text', '')}")
print("\n=== FINAL ANSWER ===\n" + (tr["final_answer"] or ""))
print(f"\nWatchlist: {watchlist}")
print(f"Stop: {tr['stop_reason']} | tool calls {tr['n_tool_calls']} | LLM calls {tr['llm_calls']} | {tr['seconds']} s")
print(f"Trace saved: {tr['files']['markdown']}")
