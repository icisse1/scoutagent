"""
app.py: ScoutAgent public chat app (Streamlit).

Local:   streamlit run app.py          (key in .env)
Hosted:  Streamlit Community Cloud     (key in the app's Secrets, never in the UI or repo)
"""
import datetime as dt
import html
import json
import os
import re
import threading

import streamlit as st

# Secrets from Streamlit Cloud -> environment variables (keys never appear in the UI)
try:
    for k, v in st.secrets.items():
        if isinstance(v, str):
            os.environ.setdefault(k, v)
except Exception:
    pass

from agent import MAX_INPUT_CHARS, ScoutSystem, trace_to_markdown
from llm import LLM, detect_provider, load_dotenv
from tools import DF

load_dotenv()

# ---------- caps ----------
MAX_STEPS = int(os.environ.get("SCOUT_MAX_STEPS", 6))           # tool rounds per turn
MAX_TURNS = int(os.environ.get("SCOUT_MAX_TURNS", 8))           # requests per browser session
MAX_DAILY = int(os.environ.get("SCOUT_MAX_DAILY_RUNS", 150))    # requests per day for the whole app
MAX_TOKENS = int(os.environ.get("SCOUT_MAX_TOKENS", 2000))      # output tokens per LLM call
COUNTER = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".daily_counter.json")
_lock = threading.Lock()


def take_daily_slot():
    """Global daily cap shared by all visitors (protects the free API quota)."""
    today = dt.date.today().isoformat()
    with _lock:
        try:
            data = json.load(open(COUNTER))
        except Exception:
            data = {}
        n = data.get(today, 0)
        if n >= MAX_DAILY:
            return False
        json.dump({today: n + 1}, open(COUNTER, "w"))
        return True


# ---------- output guardrail ----------
# Model text is untrusted (a prompt injection can shape it). Never render it as HTML, and never let it
# load images or carry links: an <img> or ![](url) would make the visitor's browser send data to an
# outside server (exfiltration). Found in the Milestone III security review and fixed here.
_IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_REFDEF = re.compile(r"(?m)^\s*\[[^\]]+\]:\s*\S+.*$")
_AUTOLINK = re.compile(r"<(https?://[^>]+)>")


def esc(text):
    """For text placed inside our own HTML."""
    return html.escape(str(text or ""))


def safe_md(text):
    """For model markdown: keep tables and bold, drop images, links and raw URLs' clickability."""
    t = str(text or "")
    t = _IMG.sub("[image removed]", t)
    t = _LINK.sub(r"\1", t)
    t = _REFDEF.sub("", t)
    t = _AUTOLINK.sub(r"\1", t)
    t = re.sub(r"https?://", "hxxp://", t)  # visible but not a live link
    return t


st.set_page_config(page_title="ScoutAgent", page_icon="⚽", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1100px;}
.reason {border-left: 4px solid #1f7a4d; padding: .45rem .75rem; margin: .25rem 0; border-radius: 4px;
         background: rgba(31,122,77,.10);}
.critic {border-left: 4px solid #3b5bdb; padding: .45rem .75rem; margin: .25rem 0; border-radius: 4px;
         background: rgba(59,91,219,.10);}
.gate {border: 2px solid #d98200; padding: .8rem 1rem; border-radius: 8px; background: rgba(217,130,0,.10);}
.small {font-size: .85rem; opacity: .8;}
</style>
""", unsafe_allow_html=True)

ss = st.session_state
ss.setdefault("history", [])     # list of finished turns: {"query", "trace"}
ss.setdefault("watchlist", [])   # sandbox action target, lives only in this browser session
ss.setdefault("active", None)    # a run paused at the human gate
ss.setdefault("turns", 0)

provider = detect_provider()

# ---------- sidebar ----------
with st.sidebar:
    st.header("ScoutAgent")
    st.caption("Scout agent + Critic agent + human approval gate")
    st.markdown(f"**Model:** {provider} (free tier)")
    st.markdown(f"**Requests left this session:** {max(0, MAX_TURNS - ss.turns)} of {MAX_TURNS}")
    st.markdown(f"**Caps:** {MAX_STEPS} tool rounds per request, {MAX_TOKENS} tokens per answer, "
                f"{MAX_INPUT_CHARS} characters per message")
    st.divider()
    st.subheader("Watchlist (sandbox)")
    if ss.watchlist:
        for w in ss.watchlist:
            st.markdown(f"- **{esc(w['player'])}** ({esc(w['team'])}), {esc(w['priority'])}")
    else:
        st.caption("Empty. Ask the agent to add a player; you will be asked to approve.")
    st.divider()
    st.caption(f"Data: StatsBomb Open Data, 2015/16 Premier League, La Liga, Serie A, Ligue 1. "
               f"{len(DF):,} player seasons. No age, value or contract data.")
    st.caption("Nothing you type is stored after you close the tab. Do not enter personal information.")


# ---------- rendering ----------
def render_event(e):
    t = e["type"]
    if t == "tool":
        st.markdown(f"<div class='reason'>🔎 <b>Scout</b> · <code>{esc(e['tool'])}</code><br>{esc(e['reasoning'])}</div>",
                    unsafe_allow_html=True)
        with st.expander("Arguments and observation"):
            st.code(json.dumps(e["args"], indent=2, ensure_ascii=False), language="json")
            st.code(e["observation"], language="text")
    elif t == "gate_request":
        st.markdown(f"<div class='reason'>✋ <b>Scout</b> asks to run <code>{esc(e['tool'])}</code><br>"
                    f"{esc(e['reasoning'])}</div>", unsafe_allow_html=True)
    elif t == "gate_decision":
        icon = "✅" if e["decision"] == "approved" else "⛔"
        st.markdown(f"{icon} **Human decision: {e['decision']}** · {safe_md(e['observation'])}")
    elif t == "draft":
        with st.expander("Scout draft (before Critic review)"):
            st.markdown(safe_md(e["text"]))
    elif t == "critic":
        badge = "✅ PASS" if e["verdict"] == "pass" else "🔁 REVISE"
        body = "<br>".join(f"• {esc(p)}" for p in e["problems"]) or "Every claim is backed by a tool result."
        st.markdown(f"<div class='critic'>🧐 <b>Critic</b> · {badge}<br><span class='small'>{body}</span></div>",
                    unsafe_allow_html=True)
    elif t == "guard":
        st.warning(e["text"])


def render_turn(turn):
    with st.chat_message("user"):
        st.markdown(turn["query"])
    tr = turn["trace"]
    with st.chat_message("assistant", avatar="⚽"):
        with st.expander(f"Execution trace · {tr.get('n_tool_calls', 0)} tool calls · "
                         f"{tr['llm_calls']} LLM calls · {tr.get('seconds', '?')} s", expanded=False):
            for e in tr["events"]:
                render_event(e)
        st.markdown(safe_md(tr["final_answer"]) or "_No answer_")
        cv = tr.get("critic_verdict")
        if cv:
            st.caption(f"Critic: {cv['verdict'].upper()} after {cv['revisions']} revision(s) · "
                       f"stop reason: {tr['stop_reason']}")
        st.download_button("Download trace", trace_to_markdown(tr), file_name="scoutagent_trace.md",
                           key=f"dl_{tr['started']}_{id(turn)}")


st.title("⚽ ScoutAgent")
st.write("Describe the player your club needs. The **Scout** agent searches real match data, the **Critic** agent "
         "checks every number against the evidence, and any action waits for **your approval**.")

if not ss.history and not ss.active:
    st.markdown("**Try:** *We need a left back strong in progressive passing* · "
                "*Our striker Gonzalo Higuaín is leaving, find similar strikers* · "
                "*Add Dimitri Payet to our watchlist*")

for turn in ss.history:
    render_turn(turn)


def get_system():
    llm = LLM(provider, max_tokens=MAX_TOKENS)
    return ScoutSystem(llm, max_steps=MAX_STEPS, use_critic=True)


def live_callback(container):
    def cb(e):
        if e.get("type") != "final":
            with container:
                render_event(e)
    return cb


def finish_or_pause(state):
    if state["status"] == "done":
        ss.history.append({"query": state["query"], "trace": state["trace"]})
        ss.active = None
    else:
        ss.active = state
    st.rerun()


# ---------- human gate ----------
if ss.active and ss.active["status"] == "awaiting_approval":
    state = ss.active
    with st.chat_message("user"):
        st.markdown(state["query"])
    with st.chat_message("assistant", avatar="⚽"):
        with st.expander("Execution trace so far", expanded=True):
            for e in state["trace"]["events"]:
                render_event(e)
        call = state["pending"]["call"]
        args = {k: v for k, v in call["args"].items() if k != "reasoning"}
        gate = st.container(border=True)
        gate.markdown(f"### ✋ Approval needed\nThe Scout wants to run **`{call['name']}`** with exactly these "
                    f"arguments (sandbox: it only changes the watchlist in this browser tab):")
        gate.code(json.dumps(args, indent=2, ensure_ascii=False), language="json")
        c1, c2, _ = gate.columns([1, 1, 4])
        approve = c1.button("✅ Approve", type="primary")
        reject = c2.button("⛔ Reject")
        if approve or reject:
            box = st.container()
            with st.spinner("Continuing..."):
                try:
                    state = get_system().resume(state, approved=bool(approve), on_event=live_callback(box))
                except Exception as ex:
                    st.error(f"Run failed: {type(ex).__name__}: {str(ex)[:300]}")
                    ss.active = None
                    st.stop()
            finish_or_pause(state)
    st.stop()

# ---------- new request ----------
prompt = st.chat_input("e.g. Find a ball winning defensive midfielder outside the Premier League",
                       max_chars=MAX_INPUT_CHARS)
if prompt:
    if ss.turns >= MAX_TURNS:
        st.warning(f"Session limit reached ({MAX_TURNS} requests). Refresh the page to start a new session.")
        st.stop()
    if not take_daily_slot():
        st.warning("The daily request limit for this demo has been reached. Please try again tomorrow.")
        st.stop()
    ss.turns += 1
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar="⚽"):
        box = st.container()
        with st.spinner("Scout and Critic are working..."):
            try:
                state = get_system().start(prompt, watchlist=ss.watchlist, on_event=live_callback(box))
            except Exception as ex:
                st.error(f"Run failed: {type(ex).__name__}: {str(ex)[:300]}")
                st.stop()
    finish_or_pause(state)
