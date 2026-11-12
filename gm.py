"""The virtual GM: answers questions and reads league rules documents.

Every number in an answer comes from a Python function in this file (the
"tools"). The language model (Google Gemini) has two jobs only:

1. decide which tool fits the question, and
2. put the tool's result into plain English.

If no Gemini key is configured, or the model call fails, a keyword router
and plain templates answer instead, so the tool still works.
"""

from __future__ import annotations

import difflib
import io
import json
import re

import pandas as pd

from draft import POSITIONS, Draft

DEFAULT_MODEL = "gemini-4-hydrogen"

TEAM_NAMES = {
    "cardinals": "ARI", "falcons": "ATL", "ravens": "BAL", "bills": "BUF",
    "panthers": "CAR", "bears": "CHI", "bengals": "CIN", "browns": "CLE",
    "cowboys": "DAL", "broncos": "DEN", "lions": "DET", "packers": "GB",
    "texans": "HOU", "colts": "IND", "jaguars": "JAX", "chiefs": "KC",
    "rams": "LA", "chargers": "LAC", "raiders": "LV", "dolphins": "MIA",
    "vikings": "MIN", "patriots": "NE", "saints": "NO", "giants": "NYG",
    "jets": "NYJ", "eagles": "PHI", "steelers": "PIT", "seahawks": "SEA",
    "49ers": "SF", "niners": "SF", "buccaneers": "TB", "bucs": "TB",
    "titans": "TEN", "commanders": "WAS",
}

POSITION_WORDS = {
    "qb": "QB", "quarterback": "QB", "rb": "RB", "running back": "RB",
    "wr": "WR", "receiver": "WR", "wideout": "WR", "te": "TE",
    "tight end": "TE", "k": "K", "kicker": "K", "dst": "DST", "d/st": "DST",
    "defense": "DST",
}

TOOL_GUIDE = """
recommend            {}                              Best picks for the user's next turn.
best_available       {"position": "RB", "count": 5}  Top undrafted players; position is QB, RB, WR, TE, K, DST or null for all.
compare_players      {"player_a": "...", "player_b": "..."}  Two players side by side.
player_outlook       {"player": "..."}               One player's projection, history and adjustments.
coach_tendency       {"name": "..."}                 Run/pass tendency of a head coach, or of a team's coach (name or team).
offensive_line       {"team": "..."}                 A team's pass-protection and run-blocking numbers.
wait_on_position     {"position": "QB"}              What is likely left at that position at the user's next pick.
my_team              {}                              The user's roster so far and open slots.
general              {}                              Anything none of the above covers.
""".strip()


class GM:
    def __init__(self, draft: Draft, context: dict):
        self.draft = draft
        self.ctx = context
        pool = draft.pool
        self._names = {str(n).lower(): pid for pid, n in pool["name"].items()}

    # ----------------------------------------------------------- lookups

    def find_player(self, text: str):
        if not text:
            return None
        key = str(text).lower().strip()
        if key in self._names:
            return self._names[key]
        hits = [n for n in self._names if key in n]
        if len(hits) == 1:
            return self._names[hits[0]]
        close = difflib.get_close_matches(key, list(self._names), n=1, cutoff=0.72)
        if close:
            return self._names[close[0]]
        last = [n for n in self._names if n.split()[-1] == key.split()[-1]]
        if last:
            best = max(last, key=lambda n: self.draft.pool.at[self._names[n], "proj_points"])
            return self._names[best]
        return None

    def find_team(self, text: str):
        if not text:
            return None
        teams = set(self.ctx["teams"]["team"])
        raw = str(text).strip()
        if raw.upper() in teams:
            return raw.upper()
        low = raw.lower()
        for word, abbr in TEAM_NAMES.items():
            if word in low and abbr in teams:
                return abbr
        return None

    def _player_card(self, pid) -> dict:
        row = self.draft.pool.loc[pid]
        hist = self.ctx["history"]
        seasons = hist[hist["player_id"] == pid].sort_values("season")
        return {
            "name": row["name"], "position": row["pos"], "team": row["team"],
            "projected_points": round(float(row["proj_points"]), 1),
            "points_before_adjustments": round(float(row["base_points"]), 1),
            "projected_games": float(row["proj_games"]),
            "points_per_game": round(float(row["ppg"]), 2),
            "points_vs_replacement": round(float(row["vor"]), 1),
            "adjustments": row["notes"] or "none",
            "contract_year": bool(row["contract_year"]),
            "drafted": pid in self.draft.taken,
            "past_seasons": [
                {"season": int(r.season), "team": r.team, "games": int(r.games),
                 "fantasy_points": round(float(r.season_points), 1)}
                for r in seasons.itertuples()
            ],
        }

    # -------------------------------------------------------------- tools

    def recommend(self) -> dict:
        d = self.draft
        wait = d.picks_until(d.settings.my_team)
        return {
            "your_turn_now": d.on_clock() == d.settings.my_team,
            "picks_before_your_next_turn": wait,
            "recommendations": [
                {k: r[k] for k in ("name", "pos", "team", "proj_points", "vor",
                                   "drop_if_wait", "reason")}
                for r in d.recommend(n=3)
            ],
        }

    def best_available(self, position=None, count=5) -> dict:
        pos = _position(position)
        left = self.draft.available(pos).head(int(count or 5))
        return {"position": pos or "all", "players": [
            {"name": r["name"], "pos": r["pos"], "team": r["team"],
             "projected_points": float(r["proj_points"]),
             "points_vs_replacement": round(float(r["vor"]), 1)}
            for _, r in left.iterrows()
        ]}

    def compare_players(self, player_a=None, player_b=None) -> dict:
        a, b = self.find_player(player_a), self.find_player(player_b)
        missing = [n for n, p in ((player_a, a), (player_b, b)) if p is None]
        if missing:
            return {"error": f"Could not find: {', '.join(map(str, missing))}"}
        return {"players": [self._player_card(a), self._player_card(b)]}

    def player_outlook(self, player=None) -> dict:
        pid = self.find_player(player)
        if pid is None:
            return {"error": f"Could not find a player named {player!r}"}
        return self._player_card(pid)

    def coach_tendency(self, name=None) -> dict:
        teams, seasons = self.ctx["teams"], self.ctx["coach_seasons"]
        coach = None
        team = self.find_team(name)
        if team:
            match = teams[teams["team"] == team]
            coach = match["coach"].iloc[0] if not match.empty else None
        if not isinstance(coach, str) and name:
            options = [c for c in seasons["coach"].dropna().unique()]
            low = {c.lower(): c for c in options}
            key = str(name).lower()
            hits = [c for l, c in low.items() if key in l or l.split()[-1] in key]
            if not hits:
                close = difflib.get_close_matches(key, list(low), n=1, cutoff=0.6)
                hits = [low[close[0]]] if close else []
            coach = hits[0] if hits else None
        if not isinstance(coach, str):
            return {"error": f"Could not find a head coach or team matching {name!r}"}

        rows = seasons[seasons["coach"] == coach].sort_values("season")
        plays = rows["plays"].sum()
        career = float(rows["dropbacks"].sum() / max(plays, 1))
        league = float((rows["league_pass_rate"] * rows["plays"]).sum() / max(plays, 1))
        return {
            "coach": coach,
            "pass_rate": round(career, 3), "run_rate": round(1 - career, 3),
            "league_average_pass_rate": round(league, 3),
            "leans": ("run" if career < league - 0.01 else
                      "pass" if career > league + 0.01 else "balanced"),
            "games_in_sample": int(rows["games"].sum()),
            "by_season": [
                {"season": int(r.season), "team": r.team, "games": int(r.games),
                 "pass_rate": round(float(r.pass_rate), 3),
                 "league_pass_rate": round(float(r.league_pass_rate), 3)}
                for r in rows.itertuples()
            ],
            "note": ("Pass rate counts pass attempts and sacks against all plays. "
                     "It reflects the head coach's teams in the seasons loaded, "
                     "not the play-caller specifically, and cannot see the future."),
        }

    def offensive_line(self, team=None) -> dict:
        abbr = self.find_team(team)
        teams = self.ctx["teams"]
        match = teams[teams["team"] == abbr]
        if match.empty:
            return {"error": f"Could not find a team matching {team!r}"}
        r = match.iloc[0]
        return {
            "team": abbr, "season": int(r["season"]), "games": int(r["games"]),
            "sack_rate_allowed": round(float(r["sack_rate"]), 3),
            "pass_protection_rank": int(r["pass_block_rank"]),
            "yards_per_carry": round(float(r["yards_per_carry"]), 2),
            "run_blocking_rank": int(r["run_block_rank"]),
            "teams_ranked": int(len(teams)),
            "note": ("These are stand-ins built from team sack rate and yards per "
                     "carry, not a graded offensive line rating."),
        }

    def wait_on_position(self, position=None) -> dict:
        pos = _position(position)
        if pos is None:
            return {"error": "Tell me which position: QB, RB, WR, TE, K or DST."}
        d = self.draft
        me = d.settings.my_team
        wait = d.picks_until(me)
        now = d.available(pos).head(5)
        if wait is None:
            return {"error": "You have no later picks to wait for."}
        gone = d._expected_gone(wait, me)
        later = now[~now.index.isin(gone)]
        best_now = float(now["proj_points"].iloc[0]) if len(now) else 0.0
        best_later = float(later["proj_points"].iloc[0]) if len(later) else 0.0
        return {
            "position": pos, "picks_before_your_next_turn": wait,
            "best_now": now["name"].iloc[0] if len(now) else None,
            "best_now_points": best_now,
            "likely_best_at_next_turn": later["name"].iloc[0] if len(later) else None,
            "likely_best_points": best_later,
            "expected_cost_of_waiting": round(best_now - best_later, 1),
            "note": "Assumes other teams take the highest-value players available.",
        }

    def my_team(self) -> dict:
        d = self.draft
        me = d.settings.my_team
        lineup = [
            {"slot": slot,
             "player": d.pool.at[pid, "name"] if pid else None,
             "projected_points": float(d.pool.at[pid, "proj_points"]) if pid else None}
            for slot, pid in d.lineup(me)
        ]
        return {"lineup": lineup,
                "open_slots": {k: v for k, v in d.open_slots(me).items() if v},
                "picks_made": d.team_count(me),
                "picks_left": d.settings.roster_size - d.team_count(me)}

    TOOLS = ("recommend", "best_available", "compare_players", "player_outlook",
             "coach_tendency", "offensive_line", "wait_on_position", "my_team")

    def run(self, tool: str, args: dict | None) -> dict:
        if tool not in self.TOOLS:
            return {}
        try:
            return getattr(self, tool)(**(args or {}))
        except TypeError:
            return getattr(self, tool)()
        except Exception as exc:                       # keep the chat alive
            return {"error": f"That lookup failed: {exc}"}

    # ----------------------------------------------------------- routing

    def route_by_rules(self, question: str) -> tuple[str, dict]:
        q = question.lower()
        pos = _position_in(q)
        if "coach" in q or ("run" in q and "pass" in q):
            return "coach_tendency", {"name": question}
        if "line" in q and ("offensive" in q or "o-line" in q or "block" in q):
            return "offensive_line", {"team": question}
        if " vs " in q or "compare" in q or " or " in q:
            parts = re.split(r"\s+vs\.?\s+|\s+or\s+|\s+and\s+", re.sub(r"compare", "", q))
            if len(parts) >= 2:
                return "compare_players", {"player_a": _clean(parts[0]),
                                           "player_b": _clean(parts[1])}
        if "wait" in q and pos:
            return "wait_on_position", {"position": pos}
        if "best" in q or "available" in q or "top" in q:
            return "best_available", {"position": pos, "count": 5}
        if "my team" in q or "my roster" in q or "need" in q:
            return "my_team", {}
        if "who should" in q or "pick" in q or "take" in q or "recommend" in q:
            return "recommend", {}
        if self.find_player(_clean(q)):
            return "player_outlook", {"player": _clean(q)}
        return "general", {}

    def answer(self, question: str, history: list | None = None,
               api_key: str | None = None, model: str = DEFAULT_MODEL) -> str:
        if api_key:
            try:
                return self._answer_with_model(question, history or [], api_key, model)
            except Exception as exc:
                note = f"\n\n_(AI wording unavailable: {type(exc).__name__}. Showing the raw numbers.)_"
        else:
            note = ""
        tool, args = self.route_by_rules(question)
        if tool == "general":
            return HELP_TEXT + note
        return render(tool, self.run(tool, args)) + note

    def _answer_with_model(self, question, history, api_key, model) -> str:
        d = self.draft
        state = (f"{d.settings.teams}-team {d.settings.draft_format.lower()} draft, "
                 f"round {d.round_number}, pick {d.pick_number}. "
                 f"User picks from slot {d.settings.my_slot}.")
        recent = "\n".join(f"{role}: {text}" for role, text in history[-4:])
        route = ask_json(
            api_key, model,
            "You route questions for a fantasy football draft assistant.\n"
            f"Tools (name, example arguments, purpose):\n{TOOL_GUIDE}\n\n"
            f"Draft state: {state}\nRecent conversation:\n{recent or '(none)'}\n\n"
            f"Question: {question}\n\n"
            'Reply with JSON only: {"tool": "<name>", "args": {...}}',
        )
        tool = route.get("tool", "general")
        result = self.run(tool, route.get("args") or {})
        if tool == "general" or not result:
            guidance = ("No FantasyIntel data covers this question. Answer briefly from "
                        "general football knowledge, say clearly that it is not from "
                        "FantasyIntel's data, and do not state specific statistics.")
            data_block = "(none)"
        else:
            guidance = ("Answer using only the data below. Do not add statistics that "
                        "are not in it. If the data has an 'error' or 'note', pass that "
                        "on honestly.")
            data_block = json.dumps(result, default=str)
        return ask_text(
            api_key, model,
            "You are the FantasyIntel draft GM, advising one manager during a live "
            f"fantasy football draft. {guidance} Give a direct recommendation where "
            "one applies, in under 120 words, plain sentences.\n\n"
            f"Draft state: {state}\nQuestion: {question}\nData: {data_block}",
        )


HELP_TEXT = (
    "I can answer questions like:\n"
    "- Who should I pick?\n- Best available RB\n- Compare Player A vs Player B\n"
    "- Should I wait on a QB?\n- Will the Lions coach run more than pass?\n"
    "- How is the Eagles offensive line?\n- Show my team"
)


def render(tool: str, data: dict) -> str:
    """Plain-template answer, used when no language model is available."""
    if not data:
        return HELP_TEXT
    if "error" in data:
        return data["error"]
    if tool == "recommend":
        lines = [f"**{r['name']}** ({r['pos']}, {r['team']}): {r['reason']}"
                 for r in data["recommendations"]]
        return "\n\n".join(lines) or "No players left to recommend."
    if tool == "best_available":
        rows = [f"{i}. {p['name']} ({p['pos']}, {p['team']}): {p['projected_points']:.0f} pts"
                for i, p in enumerate(data["players"], 1)]
        return f"Best available ({data['position']}):\n" + "\n".join(rows)
    if tool == "compare_players":
        a, b = data["players"]
        better = a if a["projected_points"] >= b["projected_points"] else b
        return (f"{a['name']}: {a['projected_points']:.0f} pts "
                f"({a['points_per_game']} per game, adjustments: {a['adjustments']}).\n\n"
                f"{b['name']}: {b['projected_points']:.0f} pts "
                f"({b['points_per_game']} per game, adjustments: {b['adjustments']}).\n\n"
                f"The projection favors **{better['name']}**.")
    if tool == "player_outlook":
        past = ", ".join(f"{s['season']}: {s['fantasy_points']:.0f} in {s['games']} games"
                         for s in data["past_seasons"]) or "no history"
        return (f"**{data['name']}** ({data['position']}, {data['team']}) projects "
                f"{data['projected_points']:.0f} points over {data['projected_games']:.0f} "
                f"games. Past seasons: {past}. Adjustments: {data['adjustments']}.")
    if tool == "coach_tendency":
        return (f"**{data['coach']}** has called passes on {data['pass_rate']:.0%} of plays "
                f"and runs on {data['run_rate']:.0%}, against a league pass rate of "
                f"{data['league_average_pass_rate']:.0%} ({data['games_in_sample']} games). "
                f"He leans **{data['leans']}**. {data['note']}")
    if tool == "offensive_line":
        return (f"**{data['team']}** ({data['season']}): sack rate allowed "
                f"{data['sack_rate_allowed']:.1%} (rank {data['pass_protection_rank']} of "
                f"{data['teams_ranked']}), {data['yards_per_carry']} yards per carry "
                f"(rank {data['run_blocking_rank']}). {data['note']}")
    if tool == "wait_on_position":
        return (f"Best {data['position']} now: {data['best_now']} "
                f"({data['best_now_points']:.0f} pts). After {data['picks_before_your_next_turn']} "
                f"picks, likely best left: {data['likely_best_at_next_turn']} "
                f"({data['likely_best_points']:.0f} pts). Waiting costs about "
                f"{data['expected_cost_of_waiting']:.0f} points. {data['note']}")
    if tool == "my_team":
        rows = [f"{s['slot']}: {s['player'] or '(open)'}" for s in data["lineup"]]
        return "\n".join(rows)
    return HELP_TEXT


def _position(value):
    if value is None:
        return None
    key = str(value).lower().strip()
    if key.upper() in POSITIONS:
        return key.upper()
    return POSITION_WORDS.get(key.rstrip("s")) or POSITION_WORDS.get(key)


def _position_in(text: str):
    for word, pos in sorted(POSITION_WORDS.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(word)}s?\b", text):
            return pos
    return None


def _clean(text: str) -> str:
    text = re.sub(r"[?.!,]", " ", text)
    text = re.sub(r"\b(should i (draft|take|pick)|who is better|tell me about|"
                  r"what about|outlook for|how is|how about)\b", " ", text)
    return " ".join(text.split())


# --------------------------------------------------------------------------
# Gemini calls
# --------------------------------------------------------------------------

def _generate(api_key: str, model: str, contents, json_mode: bool):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        temperature=0.2,
        response_mime_type="application/json" if json_mode else "text/plain",
    )
    response = client.models.generate_content(model=model, contents=contents, config=config)
    return response.text or ""


def ask_text(api_key: str, model: str, prompt: str) -> str:
    return _generate(api_key, model, prompt, json_mode=False).strip()


def ask_json(api_key: str, model: str, contents) -> dict:
    text = _generate(api_key, model, contents, json_mode=True)
    match = re.search(r"\{.*\}", text, flags=re.S)
    parsed = json.loads(match.group(0) if match else text)
    return parsed if isinstance(parsed, dict) else {}


# --------------------------------------------------------------------------
# League rules reader
# --------------------------------------------------------------------------

RULES_PROMPT = """
Extract fantasy football league settings from the attached league rules.
Reply with JSON only, in exactly this shape. Use null for anything the
document does not state. Do not guess.

{
  "teams": <number of teams or null>,
  "draft_format": <"Snake" or "Auction" or null>,
  "league_type": <"Redraft" or "Keeper" or "Dynasty" or null>,
  "roster": {"QB": n, "RB": n, "WR": n, "TE": n, "FLEX": n, "K": n, "DST": n, "BENCH": n},
  "scoring": {
    "pass_yd_per_pt": <passing yards needed for 1 point>,
    "pass_td": n, "pass_int": <negative number>,
    "rush_yd_per_pt": <rushing yards needed for 1 point>, "rush_td": n,
    "rec": <points per reception>,
    "rec_yd_per_pt": <receiving yards needed for 1 point>, "rec_td": n,
    "fumble_lost": <negative number>, "two_pt": n,
    "fg": <points for a field goal under 50 yards>, "fg_50": <50 yards or longer>,
    "pat": n
  }
}

If scoring is written as points per yard (for example 0.04 per passing
yard), convert it to yards per point (25).
""".strip()

ROSTER_KEYS = ("QB", "RB", "WR", "TE", "FLEX", "K", "DST", "BENCH")
SCORING_LIMITS = {
    "pass_yd_per_pt": (1, 100), "pass_td": (0, 12), "pass_int": (-10, 0),
    "rush_yd_per_pt": (1, 100), "rush_td": (0, 12), "rec": (0, 3),
    "rec_yd_per_pt": (1, 100), "rec_td": (0, 12), "fumble_lost": (-10, 0),
    "two_pt": (0, 6), "fg": (0, 10), "fg_50": (0, 12), "pat": (0, 5),
}


def extract_text(file_name: str, data: bytes) -> str:
    name = file_name.lower()
    if name.endswith(".docx"):
        import docx
        document = docx.Document(io.BytesIO(data))
        parts = [p.text for p in document.paragraphs]
        for table in document.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text for cell in row.cells))
        return "\n".join(parts)
    return data.decode("utf-8", errors="replace")


def read_rules(file_name: str, data: bytes, api_key: str,
               model: str = DEFAULT_MODEL) -> dict:
    """Ask the language model to pull league settings out of a document."""
    if file_name.lower().endswith(".pdf"):
        from google.genai import types
        contents = [types.Part.from_bytes(data=data, mime_type="application/pdf"),
                    RULES_PROMPT]
    else:
        contents = RULES_PROMPT + "\n\nLEAGUE RULES:\n" + extract_text(file_name, data)[:60000]
    return clean_rules(ask_json(api_key, model, contents))


def clean_rules(raw: dict) -> dict:
    """Keep only values that are present and plausible."""
    out = {"roster": {}, "scoring": {}}
    teams = _number(raw.get("teams"))
    if teams in (8, 10, 12):
        out["teams"] = int(teams)
    elif teams is not None:
        out["teams_unsupported"] = int(teams)
    for key, options in (("draft_format", ("Snake", "Auction")),
                         ("league_type", ("Redraft", "Keeper", "Dynasty"))):
        value = str(raw.get(key) or "").strip().title()
        if value in options:
            out[key] = value
    for key in ROSTER_KEYS:
        value = _number((raw.get("roster") or {}).get(key))
        if value is not None and 0 <= value <= 15:
            out["roster"][key] = int(value)
    for key, (low, high) in SCORING_LIMITS.items():
        value = _number((raw.get("scoring") or {}).get(key))
        if value is not None and low <= value <= high:
            out["scoring"][key] = float(value)
    return out


def _number(value):
    try:
        if value is None or isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
