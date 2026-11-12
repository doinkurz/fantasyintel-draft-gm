"""Draft state and the pick recommendation engine. Plain Python and pandas."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

POSITIONS = ["QB", "RB", "WR", "TE", "K", "DST"]
FLEX_ELIGIBLE = ("RB", "WR", "TE")
FLEX_SHARE = {"RB": 0.40, "WR": 0.45, "TE": 0.15}
# How many players beyond the starters a team will reasonably carry.
EXTRA_DEPTH = {"QB": 1, "RB": 3, "WR": 3, "TE": 1, "K": 0, "DST": 0}
BENCH_VALUE = {"QB": 0.5, "RB": 0.8, "WR": 0.8, "TE": 0.5, "K": 0.02, "DST": 0.02}
SCARCITY_WEIGHT = 0.5

DEFAULT_ROSTER = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1,
                  "K": 1, "DST": 1, "BENCH": 6}


@dataclass
class LeagueSettings:
    teams: int = 10
    my_slot: int = 1                      # 1-based draft position
    roster: dict = field(default_factory=lambda: dict(DEFAULT_ROSTER))
    team_names: list = field(default_factory=list)
    draft_format: str = "Snake"
    league_type: str = "Redraft"

    def __post_init__(self):
        names = list(self.team_names)[:self.teams]
        names += [""] * (self.teams - len(names))
        self.team_names = [n.strip() or f"Team {i + 1}" for i, n in enumerate(names)]
        self.roster = {**DEFAULT_ROSTER, **self.roster}

    @property
    def my_team(self) -> int:
        return self.my_slot - 1

    @property
    def roster_size(self) -> int:
        return int(sum(self.roster.values()))


class Draft:
    def __init__(self, pool: pd.DataFrame, settings: LeagueSettings):
        self.settings = settings
        self.pool = pool.set_index("player_id")
        self.picks: list[dict] = []       # {"team", "player_id", "keeper", "slot"}
        self.taken: set = set()
        n, rounds = settings.teams, settings.roster_size
        self.order = [
            team
            for rnd in range(rounds)
            for team in (range(n) if rnd % 2 == 0 else reversed(range(n)))
        ]
        self.ptr = 0
        self.replacement = self._replacement_levels()
        self.pool["vor"] = self.pool["proj_points"] - self.pool["pos"].map(self.replacement)

    # ---------------------------------------------------------------- state

    def _replacement_levels(self) -> dict:
        """Projected points of the best player left once starters are gone."""
        s = self.settings
        levels = {}
        for pos in POSITIONS:
            starters = s.roster.get(pos, 0) * s.teams
            starters += round(FLEX_SHARE.get(pos, 0) * s.roster.get("FLEX", 0) * s.teams)
            ranked = self.pool[self.pool["pos"] == pos]["proj_points"].sort_values(ascending=False)
            if ranked.empty:
                levels[pos] = 0.0
            else:
                levels[pos] = float(ranked.iloc[min(starters, len(ranked) - 1)])
        return levels

    def team_players(self, team: int) -> pd.DataFrame:
        ids = [p["player_id"] for p in self.picks if p["team"] == team]
        return self.pool.loc[ids]

    def team_count(self, team: int) -> int:
        return sum(1 for p in self.picks if p["team"] == team)

    def _advance(self):
        size = self.settings.roster_size
        while self.ptr < len(self.order) and self.team_count(self.order[self.ptr]) >= size:
            self.ptr += 1

    def on_clock(self) -> int | None:
        self._advance()
        return self.order[self.ptr] if self.ptr < len(self.order) else None

    @property
    def is_complete(self) -> bool:
        return self.on_clock() is None

    @property
    def round_number(self) -> int:
        return min(self.ptr // self.settings.teams + 1, self.settings.roster_size)

    @property
    def pick_number(self) -> int:
        return sum(1 for p in self.picks if not p["keeper"]) + 1

    def picks_until(self, team: int) -> int | None:
        """Picks by other teams before `team` is next on the clock.

        If `team` is on the clock now, counts to its following turn.
        Returns None when the team has no further picks.
        """
        self._advance()
        size = self.settings.roster_size
        counts = {t: self.team_count(t) for t in range(self.settings.teams)}
        waiting, seen_own = 0, False
        for i in range(self.ptr, len(self.order)):
            t = self.order[i]
            if counts[t] >= size:
                continue
            counts[t] += 1
            if t == team:
                if i == self.ptr and not seen_own:
                    seen_own = True
                    continue
                return waiting
            waiting += 1
        return None

    # -------------------------------------------------------------- actions

    def pick(self, player_id, team: int | None = None, keeper: bool = False) -> dict:
        if player_id not in self.pool.index:
            raise KeyError(f"Unknown player: {player_id}")
        if player_id in self.taken:
            raise ValueError(f"{self.pool.at[player_id, 'name']} is already drafted")
        if keeper:
            if team is None:
                raise ValueError("A keeper needs a team")
            if self.team_count(team) >= self.settings.roster_size:
                raise ValueError("That team's roster is full")
            slot = None
        else:
            team = self.on_clock()
            if team is None:
                raise ValueError("The draft is complete")
            slot = self.ptr
            self.ptr += 1
        record = {"team": team, "player_id": player_id, "keeper": keeper, "slot": slot,
                  "number": None if keeper else self.pick_number}
        self.picks.append(record)
        self.taken.add(player_id)
        return record

    def undo(self) -> dict | None:
        for i in range(len(self.picks) - 1, -1, -1):
            if not self.picks[i]["keeper"]:
                record = self.picks.pop(i)
                self.taken.discard(record["player_id"])
                self.ptr = record["slot"]
                return record
        return None

    def auto_pick(self) -> dict | None:
        team = self.on_clock()
        if team is None:
            return None
        best = self.recommend(team=team, n=1)
        return self.pick(best[0]["player_id"]) if best else None

    # -------------------------------------------------------------- queries

    def available(self, pos: str | None = None) -> pd.DataFrame:
        left = self.pool[~self.pool.index.isin(self.taken)]
        if pos:
            left = left[left["pos"] == pos]
        return left.sort_values("proj_points", ascending=False)

    def lineup(self, team: int) -> list[tuple[str, str | None]]:
        """Assign a team's players to roster slots, best players first."""
        roster = self.settings.roster
        slots = {pos: [] for pos in POSITIONS}
        flex, bench = [], []
        players = self.team_players(team).sort_values("proj_points", ascending=False)
        for pid, row in players.iterrows():
            pos = row["pos"]
            if len(slots[pos]) < roster.get(pos, 0):
                slots[pos].append(pid)
            elif pos in FLEX_ELIGIBLE and len(flex) < roster.get("FLEX", 0):
                flex.append(pid)
            else:
                bench.append(pid)
        out = []
        for pos in POSITIONS:
            for i in range(roster.get(pos, 0)):
                out.append((pos, slots[pos][i] if i < len(slots[pos]) else None))
        for i in range(roster.get("FLEX", 0)):
            out.append(("FLEX", flex[i] if i < len(flex) else None))
        for i in range(max(roster.get("BENCH", 0), len(bench))):
            out.append(("BENCH", bench[i] if i < len(bench) else None))
        return out

    def open_slots(self, team: int) -> dict:
        opened = {pos: 0 for pos in POSITIONS + ["FLEX", "BENCH"]}
        for slot, pid in self.lineup(team):
            if pid is None:
                opened[slot] += 1
        return opened

    def _expected_gone(self, k: int, exclude_team: int) -> set:
        """Players likely to be taken in the next k picks by other teams."""
        if k <= 0:
            return set()
        left = self.available()
        total = len(self.order)
        if self.ptr < 0.7 * total:        # kickers and defenses go late
            left = left[~left["pos"].isin(["K", "DST"])]
        return set(left.sort_values("vor", ascending=False).head(k).index)

    def recommend(self, team: int | None = None, n: int = 3) -> list[dict]:
        """Rank available players for `team` (default: your team)."""
        s = self.settings
        team = s.my_team if team is None else team
        left = self.available()
        if left.empty:
            return []

        opened = self.open_slots(team)
        counts = self.team_players(team)["pos"].value_counts().to_dict()
        remaining = s.roster_size - self.team_count(team)
        if remaining <= 0:
            return []
        must_fill = opened["K"] + opened["DST"]
        forced = remaining <= must_fill

        wait = self.picks_until(team)
        gone = self._expected_gone(wait or 0, team) if wait else set()
        later = left[~left.index.isin(gone)]
        next_best = later.groupby("pos")["proj_points"].max().to_dict()
        second = {}
        for pos, grp in later.groupby("pos"):
            vals = grp["proj_points"].nlargest(2).tolist()
            second[pos] = vals[1] if len(vals) > 1 else 0.0

        rows = []
        for pid, row in left.iterrows():
            pos = row["pos"]
            cap = (s.roster.get(pos, 0) + EXTRA_DEPTH[pos]
                   + (s.roster.get("FLEX", 0) if pos in FLEX_ELIGIBLE else 0))
            if pos in ("K", "DST"):
                if opened[pos] == 0:
                    need, role = 0.02, "depth"
                elif forced:
                    need, role = 10.0, f"your last open {pos} slot"
                else:
                    need, role = 0.15, f"open {pos} slot (can wait)"
            elif forced:
                need, role = 0.01, "depth"
            elif counts.get(pos, 0) >= cap:
                need, role = 0.05, "surplus"
            elif opened[pos] > 0:
                need, role = 1.0, f"open {pos} starting slot"
            elif pos in FLEX_ELIGIBLE and opened["FLEX"] > 0:
                need, role = 0.95, "open flex slot"
            else:
                need, role = BENCH_VALUE[pos], "bench depth"

            if wait is None:
                drop = 0.0
            else:
                fallback = next_best.get(pos, 0.0)
                if pid not in gone and fallback == row["proj_points"]:
                    fallback = second.get(pos, 0.0)
                drop = max(0.0, row["proj_points"] - fallback)

            value = row["vor"] + SCARCITY_WEIGHT * drop
            base = max(self.replacement.get(pos, 0.0), 1.0)
            score = need * max(value, 0.0) + 0.01 * need * row["proj_points"] / base
            rows.append((score, pid, role, drop))

        rows.sort(key=lambda r: r[0], reverse=True)
        out = []
        for score, pid, role, drop in rows[:n]:
            row = self.pool.loc[pid]
            reason = (f"Projects {row['proj_points']:.0f} points, "
                      f"{row['vor']:+.0f} versus a replacement-level {row['pos']}. "
                      f"Fills your {role}.")
            if wait is not None and drop >= 1:
                reason += (f" If you wait {wait} picks, the best {row['pos']} left "
                           f"projects about {drop:.0f} points lower.")
            if row["notes"]:
                reason += f" Adjustments: {row['notes']}."
            out.append({
                "player_id": pid, "name": row["name"], "pos": row["pos"],
                "team": row["team"], "proj_points": float(row["proj_points"]),
                "vor": float(row["vor"]), "drop_if_wait": float(drop),
                "role": role, "reason": reason, "score": float(score),
            })
        return out
