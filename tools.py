"""
tools.py: the tools the Scouting Agent can call.

Every tool reads the real StatsBomb based table in data/players.csv
(built by build_dataset.py) and returns a short text observation.
Every tool also takes a required "reasoning" argument so that the
agent's thinking is written into the trace before each action.
"""
import difflib
import json
import os
import unicodedata

import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "players.csv")
DF = pd.read_csv(DATA)

POSITIONS = {
    "GK": "Goalkeeper", "CB": "Center back", "LB": "Left back / left wing back",
    "RB": "Right back / right wing back", "DM": "Defensive midfielder", "CM": "Central midfielder",
    "AM": "Attacking midfielder", "LW": "Left winger / left midfielder",
    "RW": "Right winger / right midfielder", "ST": "Striker / center forward",
}
LEAGUES = sorted(DF["league"].unique().tolist())

METRICS = {
    "np_goals_p90": "Non penalty goals per 90",
    "np_xg_p90": "Non penalty expected goals (xG) per 90",
    "shots_p90": "Shots per 90",
    "shots_on_target_p90": "Shots on target per 90",
    "assists_p90": "Assists per 90",
    "xa_p90": "Expected assists (xG of shots created) per 90",
    "key_passes_p90": "Passes leading to a shot per 90",
    "passes_cmp_p90": "Completed passes per 90",
    "pass_completion_pct": "Pass completion %",
    "prog_passes_p90": "Progressive passes per 90 (completed, at least 10m toward goal)",
    "passes_final_third_p90": "Completed passes into the final third per 90",
    "passes_into_box_p90": "Completed passes into the penalty box per 90",
    "crosses_cmp_p90": "Completed crosses per 90",
    "prog_carries_p90": "Progressive carries per 90 (at least 10m toward goal)",
    "dribbles_cmp_p90": "Successful dribbles per 90",
    "dribble_success_pct": "Dribble success %",
    "tackles_p90": "Tackles per 90",
    "tackles_won_p90": "Tackles won per 90",
    "interceptions_p90": "Interceptions per 90",
    "pressures_p90": "Pressures applied per 90",
    "ball_recoveries_p90": "Ball recoveries per 90",
    "clearances_p90": "Clearances per 90",
    "blocks_p90": "Blocks per 90",
    "aerials_won_p90": "Aerial duels won per 90",
    "aerial_win_pct": "Aerial duel win %",
    "dribbled_past_p90": "Times dribbled past per 90 (lower is better)",
    "fouls_p90": "Fouls committed per 90 (lower is better)",
    "turnovers_p90": "Miscontrols + dispossessions per 90 (lower is better)",
}
LOWER_IS_BETTER = {"dribbled_past_p90", "fouls_p90", "turnovers_p90"}

# Which stats describe each kind of player (used for profiles and similarity)
PROFILE_METRICS = {
    "ST": ["np_goals_p90", "np_xg_p90", "shots_p90", "xa_p90", "key_passes_p90", "dribbles_cmp_p90",
           "aerials_won_p90", "pressures_p90", "prog_carries_p90"],
    "W": ["np_goals_p90", "np_xg_p90", "xa_p90", "key_passes_p90", "dribbles_cmp_p90", "prog_carries_p90",
          "passes_into_box_p90", "crosses_cmp_p90", "pressures_p90"],
    "MID": ["prog_passes_p90", "passes_cmp_p90", "pass_completion_pct", "key_passes_p90", "xa_p90",
            "prog_carries_p90", "tackles_p90", "interceptions_p90", "pressures_p90", "ball_recoveries_p90"],
    "FB": ["prog_passes_p90", "crosses_cmp_p90", "key_passes_p90", "xa_p90", "prog_carries_p90",
           "tackles_p90", "interceptions_p90", "pressures_p90", "aerials_won_p90", "dribbled_past_p90"],
    "CB": ["passes_cmp_p90", "pass_completion_pct", "prog_passes_p90", "tackles_p90", "interceptions_p90",
           "clearances_p90", "blocks_p90", "aerials_won_p90", "aerial_win_pct", "dribbled_past_p90"],
    "GK": ["passes_cmp_p90", "pass_completion_pct", "prog_passes_p90", "aerials_won_p90"],
}
FAMILY = {"ST": "ST", "LW": "W", "RW": "W", "AM": "W", "CM": "MID", "DM": "MID",
          "LB": "FB", "RB": "FB", "CB": "CB", "GK": "GK"}

SHOW_COLS = ["player", "team", "league", "position", "nationality", "minutes"]


# ---------- helpers ----------
def _norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return " ".join(s.replace("-", " ").replace(".", "").split())


DF["_n1"] = DF["player"].map(_norm)
DF["_n2"] = DF["full_name"].map(_norm)


def _find_player(name, league=None):
    """Return (row or None, message)."""
    q = _norm(name)
    pool = DF if not league else DF[DF["league"] == league]
    hits = pool[(pool["_n1"] == q) | (pool["_n2"] == q)]
    if hits.empty:
        hits = pool[pool["_n1"].str.contains(q, regex=False) | pool["_n2"].str.contains(q, regex=False)]
    if hits.empty:
        tokens = q.split()
        mask = pd.Series(True, index=pool.index)
        for t in tokens:
            mask &= pool["_n1"].str.contains(t, regex=False) | pool["_n2"].str.contains(t, regex=False)
        hits = pool[mask]
    if hits.empty:
        close = difflib.get_close_matches(q, pool["_n1"].tolist(), n=5, cutoff=0.6)
        sugg = pool[pool["_n1"].isin(close)]["player"].unique().tolist()
        vocab = sorted({w for n in pool["_n1"] for w in n.split()} | {w for n in pool["_n2"] for w in n.split()})
        for t in q.split():
            for w in difflib.get_close_matches(t, vocab, n=3, cutoff=0.75):
                m = pool[pool["_n1"].str.split().map(lambda ws: w in ws) | pool["_n2"].str.split().map(lambda ws: w in ws)]
                sugg += m.sort_values("minutes", ascending=False)["player"].head(3).tolist()
        sugg = list(dict.fromkeys(sugg))[:6]
        return None, f"No player matching '{name}' in the dataset. Did you mean: {sugg or 'no close names'}?"
    if hits["player_id"].nunique() > 1:
        opts = hits.sort_values("minutes", ascending=False).head(6)
        lst = "; ".join(f"{r.player} ({r.team}, {r.league}, {r.minutes} min)" for r in opts.itertuples())
        best = opts.iloc[0]
        return best, f"Note: '{name}' matched several players ({lst}). Using {best.player} (most minutes)."
    return hits.sort_values("minutes", ascending=False).iloc[0], ""


def _fmt(v):
    if pd.isna(v):
        return "n/a"
    if isinstance(v, (float, np.floating)):
        return f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}"
    return str(v)


def _table(df, cols):
    lines = [" | ".join(cols)]
    for _, r in df.iterrows():
        lines.append(" | ".join(_fmt(r[c]) for c in cols))
    return "\n".join(lines)


def _percentile(series, value, metric):
    s = series.dropna()
    if s.empty or pd.isna(value):
        return None
    pct = (s < value).mean() * 100 + (s == value).mean() * 50
    return round(100 - pct) if metric in LOWER_IS_BETTER else round(pct)


# ---------- tools ----------
def search_players(position, sort_by, leagues=None, min_minutes=900, nationality=None,
                   min_values=None, top_n=8, **_):
    if position not in POSITIONS:
        return f"Error: position must be one of {list(POSITIONS)}."
    if sort_by not in METRICS:
        return f"Error: sort_by must be one of {list(METRICS)}."
    top_n = max(1, min(int(top_n or 8), 15))
    pool = DF[(DF["position"] == position) & (DF["minutes"] >= (min_minutes or 0))]
    if leagues:
        bad = [l for l in leagues if l not in LEAGUES]
        if bad:
            return f"Error: unknown league(s) {bad}. Available: {LEAGUES}."
        pool = pool[pool["league"].isin(leagues)]
    if nationality:
        pool = pool[pool["nationality"].fillna("").str.lower().str.contains(nationality.lower())]
    applied = []
    for f in (min_values or []):
        m, v = f.get("metric"), f.get("min")
        if m not in METRICS:
            return f"Error: filter metric '{m}' not recognised."
        pool = pool[pool[m] >= v]
        applied.append(f"{m} >= {v}")
    n_total = len(pool)
    if n_total == 0:
        return ("0 players matched. Try lowering min_minutes, removing a filter, or adding leagues. "
                f"Filters used: position={position}, leagues={leagues or 'all'}, min_minutes={min_minutes}, "
                f"nationality={nationality}, {applied}.")
    asc = sort_by in LOWER_IS_BETTER
    res = pool.sort_values(sort_by, ascending=asc).head(top_n)
    extra = [m for m in PROFILE_METRICS[FAMILY[position]] if m != sort_by][:4]
    cols = SHOW_COLS + [sort_by] + extra
    return (f"{n_total} {POSITIONS[position].lower()}s matched (season 2015/16, min {min_minutes} minutes"
            f"{', ' + ', '.join(applied) if applied else ''}). Top {len(res)} by {sort_by}:\n"
            + _table(res, cols))


def get_player_profile(player_name, league=None, **_):
    row, note = _find_player(player_name, league)
    if row is None:
        return note
    peers = DF[(DF["position"] == row["position"]) & (DF["minutes"] >= 900)]
    lines = [note] if note else []
    lines.append(f"{row['player']} (full name: {row['full_name']}), {row['nationality']}, "
                 f"{POSITIONS[row['position']]} at {row['team']} ({row['league']} {row['season']}).")
    lines.append(f"Minutes {row['minutes']}, matches {row['matches']}, starts {row['starts']}, "
                 f"non penalty goals {row['np_goals']}, assists {row['assists']}, "
                 f"np xG {row['np_xg']}, xA {row['xa']}.")
    if row["minutes"] < 900:
        lines.append("Warning: fewer than 900 minutes, so per 90 numbers are unreliable (small sample).")
    lines.append(f"Per 90 stats with percentile vs {len(peers)} {row['position']}s in the 4 leagues (900+ min):")
    for m in PROFILE_METRICS[FAMILY[row["position"]]] + ["fouls_p90", "turnovers_p90"]:
        p = _percentile(peers[m], row[m], m)
        lines.append(f"- {METRICS[m]}: {_fmt(row[m])} (percentile {p if p is not None else 'n/a'})")
    return "\n".join(lines)


def find_similar_players(player_name, top_n=5, leagues=None, exclude_same_team=True, min_minutes=900, **_):
    row, note = _find_player(player_name)
    if row is None:
        return note
    fam = FAMILY[row["position"]]
    feats = PROFILE_METRICS[fam]
    same_family = [p for p, f in FAMILY.items() if f == fam]
    pool = DF[DF["position"].isin(same_family) & (DF["minutes"] >= (min_minutes or 0))].copy()
    if row["player_id"] not in pool["player_id"].values:
        pool = pd.concat([pool, DF[DF["player_id"] == row["player_id"]]])
    X = pool[feats].fillna(pool[feats].median())
    Z = (X - X.mean()) / X.std(ddof=0).replace(0, 1)
    target = Z.loc[pool["player_id"] == row["player_id"]].iloc[0].values
    norms = np.linalg.norm(Z.values, axis=1) * np.linalg.norm(target)
    pool["similarity"] = (Z.values @ target) / np.where(norms == 0, 1, norms)
    pool = pool[pool["player_id"] != row["player_id"]]
    if exclude_same_team:
        pool = pool[pool["team"] != row["team"]]
    if leagues:
        pool = pool[pool["league"].isin(leagues)]
    res = pool.sort_values("similarity", ascending=False).head(max(1, min(int(top_n or 5), 10)))
    res["similarity"] = res["similarity"].round(3)
    out = [note] if note else []
    out.append(f"Players most similar to {row['player']} ({row['position']}, {row['team']}), using cosine "
               f"similarity on standardised per 90 stats: {', '.join(feats)}.")
    out.append(f"Target values: " + ", ".join(f"{m}={_fmt(row[m])}" for m in feats[:6]))
    out.append(_table(res, SHOW_COLS + ["similarity"] + feats[:5]))
    return "\n".join(o for o in out if o)


def compare_players(player_names, metrics=None, **_):
    rows, notes = [], []
    for n in player_names[:6]:
        r, note = _find_player(n)
        if r is None:
            notes.append(note)
        else:
            rows.append(r)
            if note:
                notes.append(note)
    if not rows:
        return "\n".join(notes)
    metrics = [m for m in (metrics or []) if m in METRICS] or PROFILE_METRICS[FAMILY[rows[0]["position"]]]
    df = pd.DataFrame(rows)
    return "\n".join(notes + [_table(df, SHOW_COLS + metrics[:8])])


# ---------- schemas given to the LLM ----------
REASONING = {"type": "string",
             "description": "One or two sentences: what you are trying to find out with this call and why."}

TOOL_SCHEMAS = [
    {
        "name": "search_players",
        "description": "Search the player database (real StatsBomb event data, 2015/16 Premier League, La Liga, "
                       "Serie A, Ligue 1) for players in one position, filtered and ranked by a per 90 metric.",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": REASONING,
                "position": {"type": "string", "enum": list(POSITIONS),
                             "description": "; ".join(f"{k}={v}" for k, v in POSITIONS.items())},
                "sort_by": {"type": "string", "enum": list(METRICS), "description": "Metric to rank by."},
                "leagues": {"type": "array", "items": {"type": "string", "enum": LEAGUES},
                            "description": "Optional list of leagues. Omit for all four."},
                "min_minutes": {"type": "integer", "description": "Minimum minutes played (default 900)."},
                "nationality": {"type": "string", "description": "Optional nationality, e.g. 'Spain'."},
                "min_values": {"type": "array", "description": "Optional minimum thresholds.",
                               "items": {"type": "object", "properties": {
                                   "metric": {"type": "string", "enum": list(METRICS)},
                                   "min": {"type": "number"}}, "required": ["metric", "min"]}},
                "top_n": {"type": "integer", "description": "How many results (max 15, default 8)."},
            },
            "required": ["reasoning", "position", "sort_by"],
        },
    },
    {
        "name": "get_player_profile",
        "description": "Full season stat profile of one player, with percentile ranks against players in the "
                       "same position. Use it to check a candidate in depth before recommending him.",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": REASONING,
                "player_name": {"type": "string"},
                "league": {"type": "string", "enum": LEAGUES, "description": "Optional, to disambiguate."},
            },
            "required": ["reasoning", "player_name"],
        },
    },
    {
        "name": "find_similar_players",
        "description": "Find players whose statistical style is most similar to a given player (for example to "
                       "replace a player who is leaving).",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": REASONING,
                "player_name": {"type": "string"},
                "top_n": {"type": "integer", "description": "Default 5, max 10."},
                "leagues": {"type": "array", "items": {"type": "string", "enum": LEAGUES}},
                "exclude_same_team": {"type": "boolean", "description": "Default true."},
                "min_minutes": {"type": "integer", "description": "Default 900."},
            },
            "required": ["reasoning", "player_name"],
        },
    },
    {
        "name": "compare_players",
        "description": "Side by side table of 2 to 6 players on chosen metrics.",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": REASONING,
                "player_names": {"type": "array", "items": {"type": "string"}},
                "metrics": {"type": "array", "items": {"type": "string", "enum": list(METRICS)}},
            },
            "required": ["reasoning", "player_names"],
        },
    },
]

# ---------- gated action (human approval required) ----------
GATED_TOOLS = {"add_to_watchlist"}

WATCHLIST_SCHEMA = {
    "name": "add_to_watchlist",
    "description": "ACTION (needs human approval): add a player to the club's scouting watchlist with a priority "
                   "and a note. This is a SANDBOX: it only changes a list in the user's browser session, nothing is "
                   "sent or saved anywhere else. Only call it when the user explicitly asks to add a player.",
    "parameters": {
        "type": "object",
        "properties": {
            "reasoning": REASONING,
            "player_name": {"type": "string"},
            "priority": {"type": "string", "enum": ["high", "medium", "low"]},
            "note": {"type": "string", "description": "One sentence on why, based on tool evidence."},
        },
        "required": ["reasoning", "player_name", "priority", "note"],
    },
}
TOOL_SCHEMAS.append(WATCHLIST_SCHEMA)


def add_to_watchlist(watchlist, player_name, priority="medium", note="", **_):
    """Sandbox action. Only runs AFTER a human clicked Approve."""
    row, msg = _find_player(player_name)
    if row is None:
        return f"Not added. {msg}"
    entry = {"player": row["player"], "team": row["team"], "league": row["league"],
             "priority": priority, "note": str(note)[:200]}
    if any(w["player"] == entry["player"] for w in watchlist):
        return f"SANDBOX: {entry['player']} is already on the session watchlist."
    watchlist.append(entry)
    return (f"SANDBOX: added {entry['player']} ({entry['team']}) to the session watchlist with priority "
            f"{priority}. Nothing was sent or stored outside this browser session.")


TOOL_FUNCS = {
    "search_players": search_players,
    "get_player_profile": get_player_profile,
    "find_similar_players": find_similar_players,
    "compare_players": compare_players,
}


def run_tool(name, args):
    if name not in TOOL_FUNCS:
        return f"Error: unknown tool '{name}'. Available: {list(TOOL_FUNCS)}."
    try:
        return TOOL_FUNCS[name](**{k: v for k, v in args.items() if k != "reasoning"})
    except Exception as e:  # tool failure becomes an observation the agent can react to
        return f"Tool error ({type(e).__name__}): {e}"


if __name__ == "__main__":  # quick manual test
    print(search_players("LB", "prog_passes_p90", top_n=5))
    print(get_player_profile("Kante"))
    print(find_similar_players("Higuain", top_n=5))
    print(compare_players(["Marcelo", "Jordi Alba", "Alex Sandro"]))
    print(get_player_profile("Ronaldoo"))
    print(json.dumps(TOOL_SCHEMAS)[:200])
