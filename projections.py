"""Turns season stats into projected fantasy points for the coming draft.

The method is a transparent formula, not a trained model:

1. Past performance: each player's per-game stat line is a weighted average
   of up to three seasons (most recent counts most).
2. Offensive line: a pass-protection index (sack rate) and a run-blocking
   index (yards per carry) for the player's team nudge the projection.
3. Coaching trend: how pass-heavy the team's head coach has been nudges
   passers and receivers one way and running backs the other.
4. Contract status: players entering the last year of a contract get a
   small bump.

Every adjustment is capped and recorded in the player's `notes`, so the GM
can explain where a number came from.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from data import STAT_COLS

SEASON_WEIGHTS = [0.55, 0.30, 0.15]      # most recent season first
FULL_SEASON = 17
OFFENSE = ("QB", "RB", "WR", "TE")
ADJ_PER_SD = 0.02                        # 2% per standard deviation
ADJ_CAP = 0.04                           # never more than 4% per factor
CONTRACT_YEAR_BUMP = 0.03

DEFAULT_SCORING = {
    "pass_yd_per_pt": 25.0, "pass_td": 4.0, "pass_int": -2.0,
    "rush_yd_per_pt": 10.0, "rush_td": 6.0,
    "rec": 1.0, "rec_yd_per_pt": 10.0, "rec_td": 6.0,
    "fumble_lost": -2.0, "two_pt": 2.0,
    "fg": 3.0, "fg_50": 5.0, "pat": 1.0,
    "dst_sack": 1.0, "dst_int": 2.0, "dst_fumble_rec": 2.0,
    "dst_td": 6.0, "dst_safety": 2.0,
}

# Points for a defense by points allowed in a game: (upper bound, points).
POINTS_ALLOWED_TIERS = [(0, 10), (6, 7), (13, 4), (20, 1), (27, 0), (34, -1), (999, -4)]


def offense_points(s: pd.DataFrame, sc: dict) -> pd.Series:
    """Fantasy points for offensive players and kickers from a stat table."""
    fumbles = s["sack_fumbles_lost"] + s["rushing_fumbles_lost"] + s["receiving_fumbles_lost"]
    two_pt = (s["passing_2pt_conversions"] + s["rushing_2pt_conversions"]
              + s["receiving_2pt_conversions"])
    long_fg = s["fg_made_50_59"] + s["fg_made_60_"]
    return (
        s["passing_yards"] / sc["pass_yd_per_pt"]
        + s["passing_tds"] * sc["pass_td"]
        + s["passing_interceptions"] * sc["pass_int"]
        + s["rushing_yards"] / sc["rush_yd_per_pt"]
        + s["rushing_tds"] * sc["rush_td"]
        + s["receptions"] * sc["rec"]
        + s["receiving_yards"] / sc["rec_yd_per_pt"]
        + s["receiving_tds"] * sc["rec_td"]
        + fumbles * sc["fumble_lost"]
        + two_pt * sc["two_pt"]
        + s["fg_made"] * sc["fg"]
        + long_fg * (sc["fg_50"] - sc["fg"])
        + s["pat_made"] * sc["pat"]
    )


def defense_points(s: pd.DataFrame, sc: dict) -> pd.Series:
    """Fantasy points for a team defense, excluding points allowed."""
    return (
        s["def_sacks"] * sc["dst_sack"]
        + s["def_interceptions"] * sc["dst_int"]
        + s["fumble_recovery_opp"] * sc["dst_fumble_rec"]
        + (s["def_tds"] + s["special_teams_tds"]) * sc["dst_td"]
        + s["def_safeties"] * sc["dst_safety"]
    )


def points_allowed_score(points: float) -> int:
    for upper, value in POINTS_ALLOWED_TIERS:
        if points <= upper:
            return value
    return POINTS_ALLOWED_TIERS[-1][1]


def _season_weights(seasons: list[int]) -> dict[int, float]:
    ordered = sorted(seasons, reverse=True)
    return {s: SEASON_WEIGHTS[i] for i, s in enumerate(ordered[:len(SEASON_WEIGHTS)])}


def _zscore(series: pd.Series) -> pd.Series:
    sd = series.std(ddof=0)
    if not sd or np.isnan(sd):
        return series * 0.0
    return (series - series.mean()) / sd


def _clip(series: pd.Series) -> pd.Series:
    return (series * ADJ_PER_SD).clip(-ADJ_CAP, ADJ_CAP)


def _norm_name(name) -> str:
    text = re.sub(r"[^a-z ]", "", str(name).lower())
    text = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", text)
    return " ".join(text.split())


# --------------------------------------------------------------------------
# Team context: offensive line and coaching
# --------------------------------------------------------------------------

def team_context(prep: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (teams, coach_seasons).

    teams: one row per team for the latest season, with line indexes, the
           current head coach and that coach's pass rate.
    coach_seasons: pass rate per coach, team and season, with the league
           average for that season.
    """
    tw = prep["team_weeks"].merge(
        prep["games"][["team", "season", "week", "coach"]],
        on=["team", "season", "week"], how="left",
    )
    tw["dropbacks"] = tw["attempts"] + tw["sacks_suffered"]
    tw["plays"] = tw["dropbacks"] + tw["carries"]
    latest = int(tw["season"].max())

    # Offensive line, latest season.
    now = tw[tw["season"] == latest].groupby("team").agg(
        dropbacks=("dropbacks", "sum"), sacks=("sacks_suffered", "sum"),
        carries=("carries", "sum"), rush_yards=("rushing_yards", "sum"),
        games=("week", "nunique"),
    )
    now["sack_rate"] = now["sacks"] / now["dropbacks"].clip(lower=1)
    now["yards_per_carry"] = now["rush_yards"] / now["carries"].clip(lower=1)
    now["pass_block_adj"] = _clip(-_zscore(now["sack_rate"]))
    now["run_block_adj"] = _clip(_zscore(now["yards_per_carry"]))
    now["pass_block_rank"] = now["sack_rate"].rank(method="min").astype(int)
    now["run_block_rank"] = now["yards_per_carry"].rank(ascending=False, method="min").astype(int)

    # Coaching.
    league = tw.groupby("season").agg(d=("dropbacks", "sum"), p=("plays", "sum"))
    league_rate = (league["d"] / league["p"].clip(lower=1)).rename("league_pass_rate")

    coached = tw.dropna(subset=["coach"])
    coach_seasons = coached.groupby(["coach", "team", "season"]).agg(
        dropbacks=("dropbacks", "sum"), plays=("plays", "sum"),
        games=("week", "nunique"),
    ).reset_index()
    coach_seasons["pass_rate"] = coach_seasons["dropbacks"] / coach_seasons["plays"].clip(lower=1)
    coach_seasons = coach_seasons.merge(league_rate, left_on="season", right_index=True)

    career = coached.groupby("coach").agg(d=("dropbacks", "sum"), p=("plays", "sum"),
                                         coach_games=("week", "count"))
    career["coach_pass_rate"] = career["d"] / career["p"].clip(lower=1)

    last_game = coached.sort_values(["season", "week"]).groupby("team").tail(1)
    current = last_game.set_index("team")["coach"].rename("coach")
    teams = now.join(current).join(career[["coach_pass_rate", "coach_games"]], on="coach")
    overall = float(league["d"].sum() / max(league["p"].sum(), 1))
    teams["coach_pass_rate"] = teams["coach_pass_rate"].fillna(overall)
    teams["league_pass_rate"] = overall
    teams["coach_pass_adj"] = _clip(_zscore(teams["coach_pass_rate"]))
    teams["season"] = latest
    return teams.reset_index(), coach_seasons


# --------------------------------------------------------------------------
# Contracts
# --------------------------------------------------------------------------

def contract_years(contracts: pd.DataFrame, upcoming_season: int) -> tuple[set, set]:
    """Players whose active contract ends after `upcoming_season`.

    Returns (player ids, normalized names) so we can match either way.
    """
    if contracts is None or contracts.empty:
        return set(), set()
    c = contracts.copy()
    if "is_active" in c.columns:
        c = c[c["is_active"].astype(str).str.lower().isin(["true", "1"])]
    c["year_signed"] = pd.to_numeric(c.get("year_signed"), errors="coerce")
    c["years"] = pd.to_numeric(c.get("years"), errors="coerce")
    c = c.dropna(subset=["year_signed", "years"])
    final_year = c["year_signed"] + c["years"] - 1
    expiring = c[(final_year == upcoming_season) & (c["years"] >= 2)]
    ids = set(expiring["gsis_id"].dropna()) if "gsis_id" in expiring.columns else set()
    names = {_norm_name(n) for n in expiring["player"].dropna()}
    return ids, names


# --------------------------------------------------------------------------
# The player pool
# --------------------------------------------------------------------------

def build_pool(prep: dict, scoring: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Project every draftable player.

    Returns (pool, context). `pool` has one row per player or team defense.
    `context` holds the team, coach and history tables the GM uses to
    answer questions.
    """
    sc = {**DEFAULT_SCORING, **(scoring or {})}
    ps = prep["player_seasons"].copy()
    latest = int(ps["season"].max())
    upcoming = latest + 1
    weights = _season_weights(ps["season"].unique().tolist())
    ps = ps[ps["season"].isin(weights)]
    ps["w"] = ps["season"].map(weights)

    # A season still in progress has fewer games; scale it to a full season.
    season_len = ps.groupby("season")["games"].transform("max").clip(lower=1)
    ps["games_scaled"] = (ps["games"] / season_len * FULL_SEASON).clip(upper=FULL_SEASON)
    ps["season_points"] = offense_points(ps, sc)

    teams, coach_seasons = team_context(prep)

    off = ps[ps["position"].isin(OFFENSE + ("K",))].copy()
    weighted = off[STAT_COLS].multiply(off["w"], axis=0)
    weighted["w_games"] = off["w"] * off["games"]
    weighted["w_avail"] = off["w"] * off["games_scaled"]
    weighted["w"] = off["w"]
    weighted["player_id"] = off["player_id"]
    agg = weighted.groupby("player_id").sum()

    per_game = agg[STAT_COLS].div(agg["w_games"].clip(lower=1e-9), axis=0)
    info = (off.sort_values("season").groupby("player_id")
            .agg(name=("name", "last"), pos=("position", "last"),
                 team=("team", "last"), last_season=("season", "max"),
                 total_games=("games", "sum")))
    pool = info.join(per_game)
    pool["ppg"] = offense_points(pool, sc)
    # Small samples are pulled down so a two-game hot streak is not a star.
    pool["ppg"] *= (pool["total_games"] / 6).clip(upper=1.0)
    typical_games = agg["w_avail"] / agg["w"].clip(lower=1e-9)
    pool["proj_games"] = (0.6 * typical_games + 0.4 * 15).clip(upper=FULL_SEASON).round(1)
    pool = pool[pool["last_season"] >= latest - 1]

    # Adjustments.
    ctx = teams.set_index("team")
    pool = pool.join(ctx[["pass_block_adj", "run_block_adj", "coach_pass_adj", "coach"]],
                     on="team")
    for col in ("pass_block_adj", "run_block_adj", "coach_pass_adj"):
        pool[col] = pool[col].fillna(0.0)

    is_rb = pool["pos"] == "RB"
    is_pass = pool["pos"].isin(["QB", "WR", "TE"])
    pool["line_adj"] = np.where(is_rb, pool["run_block_adj"],
                                np.where(is_pass, pool["pass_block_adj"], 0.0))
    pool["coach_adj"] = np.where(is_rb, -pool["coach_pass_adj"],
                                 np.where(is_pass, pool["coach_pass_adj"], 0.0))

    ids, names = contract_years(prep.get("contracts"), upcoming)
    in_contract_year = (pool.index.isin(ids)
                        | pool["name"].map(_norm_name).isin(names))
    pool["contract_year"] = in_contract_year & pool["pos"].isin(OFFENSE)
    pool["contract_adj"] = np.where(pool["contract_year"], CONTRACT_YEAR_BUMP, 0.0)

    multiplier = ((1 + pool["line_adj"]) * (1 + pool["coach_adj"])
                  * (1 + pool["contract_adj"]))
    pool["base_points"] = pool["ppg"] * pool["proj_games"]
    pool["proj_points"] = pool["base_points"] * multiplier
    pool["notes"] = [_notes(row) for row in pool.itertuples()]

    last = ps[ps["season"] == latest].set_index("player_id")
    pool["last_points"] = last["season_points"].reindex(pool.index)
    pool["last_games"] = last["games"].reindex(pool.index)

    keep = ["name", "pos", "team", "proj_points", "base_points", "ppg", "proj_games",
            "last_points", "last_games", "line_adj", "coach_adj", "contract_adj",
            "contract_year", "coach", "notes"]
    pool = pool[keep].reset_index()

    dst = _defenses(prep, sc, weights)
    pool = pd.concat([pool, dst], ignore_index=True)
    pool = pool[pool["proj_points"] > 0]
    pool["proj_points"] = pool["proj_points"].round(1)
    pool["ppg"] = pool["ppg"].round(2)
    pool = pool.sort_values("proj_points", ascending=False).reset_index(drop=True)

    history = ps[["player_id", "season", "team", "games", "season_points"]].copy()
    context = {
        "teams": teams, "coach_seasons": coach_seasons, "history": history,
        "latest_season": latest, "upcoming_season": upcoming, "scoring": sc,
    }
    return pool, context


def _notes(row) -> str:
    parts = []
    if abs(row.line_adj) >= 0.005:
        kind = "run blocking" if row.pos == "RB" else "pass protection"
        parts.append(f"{row.team} {kind} {row.line_adj:+.1%}")
    if abs(row.coach_adj) >= 0.005:
        coach = row.coach if isinstance(row.coach, str) else "coach"
        parts.append(f"{coach}'s play-calling {row.coach_adj:+.1%}")
    if row.contract_adj:
        parts.append(f"contract year {row.contract_adj:+.1%}")
    return "; ".join(parts)


def _defenses(prep: dict, sc: dict, weights: dict) -> pd.DataFrame:
    tw = prep["team_weeks"]
    tw = tw[tw["season"].isin(weights)].copy()
    tw["pts"] = defense_points(tw, sc)

    games = prep["games"]
    games = games[games["season"].isin(weights)].copy()
    games["pa_pts"] = games["points_allowed"].map(points_allowed_score)

    stat = tw.groupby(["team", "season"])["pts"].mean().rename("stat_ppg")
    allowed = games.groupby(["team", "season"])["pa_pts"].mean().rename("pa_ppg")
    both = pd.concat([stat, allowed], axis=1).fillna(0.0).reset_index()
    both["ppg"] = both["stat_ppg"] + both["pa_ppg"]
    both["w"] = both["season"].map(weights)
    both["wp"] = both["ppg"] * both["w"]
    team = both.groupby("team").agg(wp=("wp", "sum"), w=("w", "sum"))
    team["ppg"] = team["wp"] / team["w"]

    latest = max(weights)
    last = both[both["season"] == latest].set_index("team")["ppg"]
    out = pd.DataFrame({
        "player_id": ["DST_" + t for t in team.index],
        "name": [f"{t} D/ST" for t in team.index],
        "pos": "DST",
        "team": team.index,
        "ppg": team["ppg"].values,
        "proj_games": float(FULL_SEASON),
    })
    out["base_points"] = out["ppg"] * FULL_SEASON
    out["proj_points"] = out["base_points"]
    out["last_points"] = (last.reindex(team.index) * FULL_SEASON).values
    out["last_games"] = np.nan
    out["line_adj"] = 0.0
    out["coach_adj"] = 0.0
    out["contract_adj"] = 0.0
    out["contract_year"] = False
    out["coach"] = None
    out["notes"] = ""
    return out
