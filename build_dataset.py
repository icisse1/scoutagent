"""
build_dataset.py  (the "Data Fetcher" step)

Downloads real event data from StatsBomb Open Data on GitHub and turns it into
one row per player per league season, with per 90 minute stats.

Run once:  python build_dataset.py
Output:    data/players.csv

Leagues: the four full 2015/16 seasons StatsBomb has released for free
(Premier League, La Liga, Serie A, Ligue 1), about 1,500 matches.
"""
import json
import os
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
LEAGUES = {  # competition_id/season_id : label
    "2/27": "Premier League",
    "11/27": "La Liga",
    "12/27": "Serie A",
    "7/27": "Ligue 1",
}
SEASON = "2015/16"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "players.csv")

# StatsBomb position name -> (position group, detailed role)
def position_group(pos: str) -> str:
    if pos == "Goalkeeper":
        return "GK"
    if "Center Back" in pos:
        return "CB"
    if pos in ("Left Back", "Left Wing Back"):
        return "LB"
    if pos in ("Right Back", "Right Wing Back"):
        return "RB"
    if "Defensive Midfield" in pos:
        return "DM"
    if "Center Midfield" in pos:
        return "CM"
    if "Attacking Midfield" in pos:
        return "AM"
    if pos in ("Left Wing", "Left Midfield"):
        return "LW"
    if pos in ("Right Wing", "Right Midfield"):
        return "RW"
    return "ST"  # Center Forward, Striker variants, Secondary Striker


def get_json(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception as e:  # network hiccup, retry
            if i == tries - 1:
                raise
    return None


def clock(s):  # "67:30" -> 67.5
    m, sec = s.split(":")
    return int(m) + int(sec) / 60


def process_match(match, league):
    mid = match["match_id"]
    events = get_json(f"{BASE}/events/{mid}.json")
    lineups = get_json(f"{BASE}/lineups/{mid}.json")
    end_clock = max(e["minute"] + e["second"] / 60 for e in events)

    rows = {}  # player_id -> dict
    for team in lineups:
        for p in team["lineup"]:
            mins_by_group = defaultdict(float)
            for pos in p["positions"]:
                start = clock(pos["from"])
                stop = clock(pos["to"]) if pos["to"] else end_clock
                mins_by_group[position_group(pos["position"])] += max(0.0, stop - start)
            minutes = sum(mins_by_group.values())
            if minutes <= 0:
                continue
            rows[p["player_id"]] = {
                "player_id": p["player_id"],
                "player": p["player_nickname"] or p["player_name"],
                "full_name": p["player_name"],
                "nationality": (p.get("country") or {}).get("name"),
                "team": team["team_name"],
                "league": league,
                "minutes": minutes,
                "starts": int(any(x["start_reason"] == "Starting XI" for x in p["positions"])),
                "pos_minutes": dict(mins_by_group),
                "stats": defaultdict(float),
            }

    shot_xg_by_keypass = {}
    for e in events:
        if e["type"]["name"] == "Shot" and e["shot"].get("key_pass_id"):
            shot_xg_by_keypass[e["shot"]["key_pass_id"]] = e["shot"].get("statsbomb_xg", 0) or 0

    for e in events:
        pl = e.get("player")
        if not pl or pl["id"] not in rows:
            continue
        s = rows[pl["id"]]["stats"]
        t = e["type"]["name"]
        loc = e.get("location") or [0, 0]
        if t == "Pass":
            p = e["pass"]
            ptype = (p.get("type") or {}).get("name", "")
            if ptype in ("Throw-in", "Kick Off"):
                continue
            completed = "outcome" not in p
            end = p.get("end_location") or loc
            s["passes_att"] += 1
            if completed:
                s["passes_cmp"] += 1
                if end[0] - loc[0] >= 10 and end[0] >= 48:
                    s["prog_passes"] += 1
                if loc[0] < 80 <= end[0]:
                    s["passes_final_third"] += 1
                if end[0] >= 102 and 18 <= end[1] <= 62 and not (loc[0] >= 102 and 18 <= loc[1] <= 62):
                    s["passes_into_box"] += 1
                if p.get("cross"):
                    s["crosses_cmp"] += 1
            if p.get("cross"):
                s["crosses_att"] += 1
            if p.get("shot_assist") or p.get("goal_assist"):
                s["key_passes"] += 1
                s["xa"] += shot_xg_by_keypass.get(e["id"], 0)
            if p.get("goal_assist"):
                s["assists"] += 1
        elif t == "Shot":
            sh = e["shot"]
            is_pen = (sh.get("type") or {}).get("name") == "Penalty"
            goal = sh["outcome"]["name"] == "Goal"
            if is_pen:
                s["pens_scored"] += goal
                continue
            s["shots"] += 1
            s["np_xg"] += sh.get("statsbomb_xg", 0) or 0
            s["np_goals"] += goal
            if sh["outcome"]["name"] in ("Goal", "Saved", "Saved To Post"):
                s["shots_on_target"] += 1
        elif t == "Carry":
            end = e["carry"].get("end_location") or loc
            if end[0] - loc[0] >= 10 and end[0] >= 48:
                s["prog_carries"] += 1
        elif t == "Dribble":
            s["dribbles_att"] += 1
            if e["dribble"]["outcome"]["name"] == "Complete":
                s["dribbles_cmp"] += 1
        elif t == "Duel":
            dt = e["duel"]["type"]["name"]
            if dt == "Tackle":
                s["tackles"] += 1
                if (e["duel"].get("outcome") or {}).get("name") in ("Won", "Success In Play", "Success Out"):
                    s["tackles_won"] += 1
            elif dt == "Aerial Lost":
                s["aerials_lost"] += 1
        elif t == "Interception":
            s["interceptions"] += 1
        elif t == "Pressure":
            s["pressures"] += 1
        elif t == "Ball Recovery":
            s["ball_recoveries"] += 1
        elif t == "Clearance":
            s["clearances"] += 1
            if (e.get("clearance") or {}).get("aerial_won"):
                s["aerials_won"] += 1
        elif t == "Block":
            s["blocks"] += 1
        elif t == "Dribbled Past":
            s["dribbled_past"] += 1
        elif t == "Foul Committed":
            s["fouls"] += 1
        elif t == "Dispossessed" or t == "Miscontrol":
            s["turnovers"] += 1
        # aerials won also recorded on passes/shots
        if t in ("Pass", "Shot") and (e.get(t.lower()) or {}).get("aerial_won"):
            s["aerials_won"] += 1
        if t == "Miscontrol" and (e.get("miscontrol") or {}).get("aerial_won"):
            s["aerials_won"] += 1
    return list(rows.values())


def main():
    season_rows = {}  # (player_id, league) -> aggregated
    for cs, league in LEAGUES.items():
        matches = get_json(f"{BASE}/matches/{cs}.json")
        print(f"{league}: {len(matches)} matches", flush=True)
        done = 0
        with ThreadPoolExecutor(max_workers=16) as pool:
            for result in pool.map(lambda m: process_match(m, league), matches):
                done += 1
                if done % 50 == 0:
                    print(f"  {done}/{len(matches)}", flush=True)
                for r in result:
                    key = (r["player_id"], league)
                    agg = season_rows.setdefault(key, {
                        **{k: r[k] for k in ("player_id", "player", "full_name", "nationality", "league")},
                        "teams": defaultdict(float), "minutes": 0.0, "matches": 0, "starts": 0,
                        "pos_minutes": defaultdict(float), "stats": defaultdict(float)})
                    agg["teams"][r["team"]] += r["minutes"]
                    agg["minutes"] += r["minutes"]
                    agg["matches"] += 1
                    agg["starts"] += r["starts"]
                    for g, m in r["pos_minutes"].items():
                        agg["pos_minutes"][g] += m
                    for k, v in r["stats"].items():
                        agg["stats"][k] += v

    out = []
    for agg in season_rows.values():
        mins = agg["minutes"]
        n90 = mins / 90
        st = agg["stats"]
        pos = max(agg["pos_minutes"], key=agg["pos_minutes"].get)
        row = {
            "player": agg["player"], "full_name": agg["full_name"], "player_id": agg["player_id"],
            "nationality": agg["nationality"], "league": agg["league"], "season": SEASON,
            "team": max(agg["teams"], key=agg["teams"].get),
            "position": pos, "minutes": round(mins), "matches": agg["matches"], "starts": agg["starts"],
            "np_goals": int(st["np_goals"]), "assists": int(st["assists"]),
            "np_xg": round(st["np_xg"], 2), "xa": round(st["xa"], 2),
            "pass_completion_pct": round(100 * st["passes_cmp"] / st["passes_att"], 1) if st["passes_att"] else None,
            "dribble_success_pct": round(100 * st["dribbles_cmp"] / st["dribbles_att"], 1) if st["dribbles_att"] else None,
            "aerial_win_pct": round(100 * st["aerials_won"] / (st["aerials_won"] + st["aerials_lost"]), 1)
            if (st["aerials_won"] + st["aerials_lost"]) else None,
        }
        for k in ["np_goals", "np_xg", "shots", "shots_on_target", "assists", "xa", "key_passes",
                  "passes_cmp", "prog_passes", "passes_final_third", "passes_into_box", "crosses_cmp",
                  "prog_carries", "dribbles_cmp", "tackles", "tackles_won", "interceptions", "pressures",
                  "ball_recoveries", "clearances", "blocks", "aerials_won", "dribbled_past", "fouls", "turnovers"]:
            row[f"{k}_p90"] = round(st[k] / n90, 3) if n90 > 0 else 0.0
        out.append(row)

    df = pd.DataFrame(out).sort_values(["league", "team", "minutes"], ascending=[True, True, False])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df)} player seasons to {OUT}")


if __name__ == "__main__":
    sys.exit(main())
