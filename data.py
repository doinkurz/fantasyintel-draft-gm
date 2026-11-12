"""Loads free NFL data (the nflverse project, through the nflreadpy package)
and reshapes it into the tables the projection engine needs.

Nothing in this file knows about fantasy scoring or drafts.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

ID_COLS = [
    "player_id", "player_display_name", "player_name", "position",
    "team", "recent_team", "season", "week", "season_type",
]

STAT_COLS = [
    "completions", "attempts", "passing_yards", "passing_tds",
    "passing_interceptions", "sacks_suffered", "sack_fumbles_lost",
    "passing_2pt_conversions",
    "carries", "rushing_yards", "rushing_tds", "rushing_fumbles_lost",
    "rushing_2pt_conversions",
    "receptions", "targets", "receiving_yards", "receiving_tds",
    "receiving_fumbles_lost", "receiving_2pt_conversions",
    "special_teams_tds",
    "def_sacks", "def_interceptions", "def_tds", "def_safeties",
    "fumble_recovery_opp",
    "fg_made", "fg_made_50_59", "fg_made_60_", "pat_made",
]

# Older nflverse files used these names.
ALIASES = {
    "recent_team": "team",
    "interceptions": "passing_interceptions",
    "sacks": "sacks_suffered",
}

SCHEDULE_COLS = [
    "season", "week", "game_type", "home_team", "away_team",
    "home_score", "away_score", "home_coach", "away_coach",
]

CONTRACT_COLS = [
    "player", "position", "team", "is_active", "year_signed", "years",
    "value", "apy", "gsis_id",
]


def current_season(today: dt.date | None = None) -> int:
    """The NFL season in progress or most recently finished."""
    today = today or dt.date.today()
    return today.year if today.month >= 9 else today.year - 1


def to_pandas(frame, cols: list[str]) -> pd.DataFrame:
    """Convert a polars or pandas table to pandas, keeping only `cols`."""
    if isinstance(frame, pd.DataFrame):
        return frame[[c for c in cols if c in frame.columns]].copy()
    keep = [c for c in cols if c in frame.columns]
    return pd.DataFrame(frame.select(keep).to_dict(as_series=False))


def load_raw(seasons: list[int] | None = None) -> dict[str, pd.DataFrame]:
    """Download weekly player stats, schedules and contracts."""
    import nflreadpy as nfl

    if seasons is None:
        latest = current_season()
        seasons = [latest - 2, latest - 1, latest]

    try:
        weekly = nfl.load_player_stats(seasons)
    except Exception:
        # The newest season's file may not exist yet (early September).
        seasons = seasons[:-1]
        weekly = nfl.load_player_stats(seasons)

    schedules = nfl.load_schedules(seasons)
    try:
        contracts = to_pandas(nfl.load_contracts(), CONTRACT_COLS)
    except Exception:
        contracts = pd.DataFrame(columns=CONTRACT_COLS)

    return {
        "weekly": to_pandas(weekly, ID_COLS + STAT_COLS + list(ALIASES)),
        "schedules": to_pandas(schedules, SCHEDULE_COLS),
        "contracts": contracts,
    }


def normalize_weekly(weekly: pd.DataFrame) -> pd.DataFrame:
    w = weekly.copy()
    rename = {old: new for old, new in ALIASES.items()
              if old in w.columns and new not in w.columns}
    w = w.rename(columns=rename)
    w = w.drop(columns=[c for c in ALIASES if c in w.columns])

    if "player_display_name" not in w.columns:
        w["player_display_name"] = w.get("player_name")
    elif "player_name" in w.columns:
        w["player_display_name"] = w["player_display_name"].fillna(w["player_name"])

    for col in STAT_COLS:
        if col not in w.columns:
            w[col] = 0.0
    w[STAT_COLS] = w[STAT_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if "season_type" in w.columns:
        is_regular = w["season_type"].astype(str).str.upper().str.startswith("REG")
        w = w[is_regular]

    w = w.dropna(subset=["player_id", "team", "season", "week"])
    w["season"] = w["season"].astype(int)
    w["week"] = w["week"].astype(int)
    return w.sort_values(["season", "week"]).reset_index(drop=True)


def prepare(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Turn raw downloads into season-level tables.

    Returns:
        player_seasons: one row per player per season (stat totals, games).
        team_weeks:     one row per team per game (stat totals).
        games:          one row per team per played game (points, coach).
        contracts:      contract records as downloaded.
    """
    w = normalize_weekly(raw["weekly"])

    grouped = w.groupby(["player_id", "season"], sort=False)
    totals = grouped[STAT_COLS].sum()
    meta = grouped.agg(
        name=("player_display_name", "last"),
        position=("position", "last"),
        team=("team", "last"),
        games=("week", "nunique"),
    )
    player_seasons = meta.join(totals).reset_index()

    team_weeks = (
        w.groupby(["team", "season", "week"], sort=False)[STAT_COLS]
        .sum()
        .reset_index()
    )

    return {
        "player_seasons": player_seasons,
        "team_weeks": team_weeks,
        "games": _long_games(raw["schedules"]),
        "contracts": raw.get("contracts", pd.DataFrame(columns=CONTRACT_COLS)),
    }


def _long_games(schedules: pd.DataFrame) -> pd.DataFrame:
    """One row per team per completed regular-season game."""
    s = schedules.copy()
    if "game_type" in s.columns:
        s = s[s["game_type"].astype(str).str.upper() == "REG"]
    s = s.dropna(subset=["home_score", "away_score"])
    for col in ("home_coach", "away_coach"):
        if col not in s.columns:
            s[col] = None

    home = pd.DataFrame({
        "season": s["season"], "week": s["week"],
        "team": s["home_team"], "opponent": s["away_team"],
        "points_for": s["home_score"], "points_allowed": s["away_score"],
        "coach": s["home_coach"],
    })
    away = pd.DataFrame({
        "season": s["season"], "week": s["week"],
        "team": s["away_team"], "opponent": s["home_team"],
        "points_for": s["away_score"], "points_allowed": s["home_score"],
        "coach": s["away_coach"],
    })
    games = pd.concat([home, away], ignore_index=True)
    games["season"] = games["season"].astype(int)
    games["week"] = games["week"].astype(int)
    return games.sort_values(["season", "week"]).reset_index(drop=True)
