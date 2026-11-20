# ScoutAgent: Milestone II (published agent + evaluation)

An AI recruitment analyst for budget-conscious football clubs. A **Scout** agent searches real StatsBomb match data,
a **Critic** agent checks every claim against the evidence, and the one action (adding a player to a watchlist)
waits for **human approval**.

**Public link:** `https://<your-app-name>.streamlit.app`  (fill in after deploying, see below)

## What changed since Milestone I

| Milestone I | Milestone II |
|---|---|
| One Scout agent, 4 read-only data tools, max-steps guard | Same Scout agent (unchanged tools and guard) |
| | **+ Critic agent** (second agent, written reason below) |
| | **+ Human gate** on a sandbox action tool `add_to_watchlist` |
| Local Streamlit page | **Public chat app** on Streamlit Community Cloud, key in Secrets |
| | **+ Caps:** 600 characters per message, 6 tool rounds per request, 2,000 output tokens per LLM call, 1 revision, 8 requests per session, 150 requests per day for the whole app |
| | **+ Evaluation harness:** 14 cases × k runs, pass@1 and pass^k (`eval/`) |

### Workflow
```
user request
  -> Scout agent: reason -> tool -> observation (max 6 rounds)
       read-only tools: search_players, get_player_profile, find_similar_players, compare_players
       action tool:     add_to_watchlist  -> HUMAN GATE: run pauses, screen shows the exact JSON arguments,
                                             nothing happens until Approve or Reject
  -> Scout draft
  -> Critic agent: checks the draft against the raw tool outputs only (+ a script that flags numbers
                   not found in the evidence) -> "pass" or "revise" with a list of problems
  -> if "revise": Scout fixes the draft once (up to 2 extra tool rounds), Critic checks again
  -> final answer + Critic verdict + downloadable trace
```

### Why a second agent (Week 9 reasons: ownership + context isolation)
- **Ownership:** the Critic owns a single job, "is every claim backed by a tool result?". The Scout owns finding players.
  One agent writing and grading its own work is the failure we want to avoid (invented or misremembered numbers).
- **Context isolation:** the Critic never sees the Scout's prompt or reasoning. It gets only the request, the draft and
  the raw evidence, so it cannot be talked into agreeing by the Scout's own chain of thought.
- **Verifiable on its own:** `python eval/test_critic.py` tests the Critic's number check without any API key, and
  the eval can run with the Critic switched off (`--no-critic`, or "Scout only" on the Evaluation page) to compare
  against the Milestone I baseline.

## Publish it (free, about 15 minutes, no laptop needs to stay on)

1. **Gemini key:** https://aistudio.google.com/apikey → Create API key → copy it. (Free tier.)
2. **GitHub:** sign in at https://github.com → **New repository** → name `scoutagent` → **Public** → Create.
   On the new repo page click **uploading an existing file**, drag in everything inside this folder
   (`app.py`, `agent.py`, `tools.py`, `llm.py`, `run.py`, `build_dataset.py`, `requirements.txt`, `README.md`,
   `GOVERNANCE.md`, and the folders `data`, `eval`, `pages`) → **Commit changes**.
   Never upload a `.env` file.
3. **Streamlit Community Cloud:** https://share.streamlit.io → sign in with GitHub → **Create app** →
   **Deploy a public app from GitHub** → Repository `<you>/scoutagent`, Branch `main`, Main file `app.py`,
   choose an App URL (e.g. `scoutagent-yourname`) → **Advanced settings** → in **Secrets** paste:
   ```
   GEMINI_API_KEY = "your_key"
   EVAL_PASSWORD = "pick_a_password"
   ```
   → **Save** → **Deploy**. After 2 to 3 minutes the app is live at `https://<app-url>.streamlit.app`.
4. **Test it:** open the link on your phone with Wi-Fi off. It must load without a login.
5. Apps sleep after a long time without visits. Open the link before class; if it shows a "wake up" button, click it.

## Run the evaluation

**In the browser (easiest):** open your app → **Evaluation** in the left sidebar → enter `EVAL_PASSWORD` →
**Run evaluation** (14 cases × 3 runs, about 10 to 20 minutes on the free tier; if a rate limit stops it, press Run
again later, finished runs are kept) → download `summary.md` and `runs.jsonl`.
Optional: switch to "Scout only (Milestone I baseline)" and run again to compare.

**In a terminal:**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # paste your free key into .env
python eval/test_critic.py                 # Critic unit test, no key needed
python eval/run_eval.py                    # Scout + Critic, k=3
python eval/run_eval.py --no-critic        # Milestone I baseline
```
Results go to `eval/results/<critic|baseline>/summary.md` (pass@1, pass^k, every failed answer).

**Test set** (`eval/cases.json`, known-good answers computed from the data): 9 normal cases plus 5 edge or
adversarial cases (age and value request the data cannot answer, a season outside the data, a prompt injection
asking for the system prompt and key, a misspelled player name, and an attempt to skip the approval step).

## Run locally
```bash
pip install -r requirements.txt
cp .env.example .env     # add GEMINI_API_KEY (and EVAL_PASSWORD to use the Evaluation page)
streamlit run app.py
python run.py "We need a left back strong in progressive passing"
python run.py --gate approve "Add Dimitri Payet to our watchlist"
```

## Files
| File | Role |
|---|---|
| `app.py` | Public chat app: live trace, Critic badge, human gate, caps |
| `pages/1_Evaluation.py` | Password-protected evaluation page |
| `agent.py` | Scout loop, Critic agent, human gate (pause/resume), guards, traces |
| `tools.py` | 4 read-only data tools + the gated sandbox action |
| `llm.py` | Free-tier model wrapper (Gemini default, Groq backup), token cap, retry on rate limits |
| `eval/cases.json`, `eval/run_eval.py`, `eval/test_critic.py` | Evaluation harness |
| `build_dataset.py`, `data/players.csv` | StatsBomb Open Data (2015/16 Premier League, La Liga, Serie A, Ligue 1), 2,214 player seasons |
| `GOVERNANCE.md` | Data, human gate, what strangers can and cannot do, accountability |

Data: StatsBomb Open Data, https://github.com/statsbomb/open-data (free for non-commercial use).
