"""FantasyIntel Draft GM - the web app.

Run locally:   streamlit run app.py
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

import data
import gm as gm_module
import projections
from draft import DEFAULT_ROSTER, Draft, LeagueSettings

LOGO = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
ROSTER_LABELS = {"QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE",
                 "FLEX": "Flex (RB/WR/TE)", "K": "K", "DST": "D/ST", "BENCH": "Bench"}
SCORING_LABELS = {
    "pass_td": "Passing touchdown",
    "pass_yd_per_pt": "Passing yards per point",
    "pass_int": "Interception thrown",
    "rush_td": "Rushing touchdown",
    "rush_yd_per_pt": "Rushing yards per point",
    "rec": "Points per reception",
    "rec_td": "Receiving touchdown",
    "rec_yd_per_pt": "Receiving yards per point",
    "fumble_lost": "Fumble lost",
    "two_pt": "Two-point conversion",
    "fg": "Field goal (under 50)",
    "fg_50": "Field goal (50 or longer)",
    "pat": "Extra point",
}
PRESETS = {"PPR": 1.0, "Half PPR": 0.5, "Standard": 0.0}
BOARD_POSITIONS = ["All", "QB", "RB", "WR", "TE", "K", "DST"]

st.set_page_config(page_title="FantasyIntel Draft GM", page_icon="🏈", layout="wide")
st.markdown(
    "<style>h1, h2, h3, h1 span, h2 span, h3 span {color: #FF8A2A !important;} "
    "[data-testid='stMetricValue'] {color: #39E75F !important;}</style>",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Secrets and cached data
# --------------------------------------------------------------------------

def secret(name: str, default=None):
    try:
        return st.secrets.get(name, default)
    except Exception:            # no secrets file
        return default


API_KEY = secret("GEMINI_API_KEY")
MODEL = secret("GEMINI_MODEL", gm_module.DEFAULT_MODEL)


@st.cache_data(ttl=24 * 3600, show_spinner="Loading NFL data. The first load takes about a minute.")
def get_prepared() -> dict:
    return data.prepare(data.load_raw())


@st.cache_data(ttl=24 * 3600, show_spinner="Projecting players for your scoring.")
def get_pool(scoring_items: tuple):
    return projections.build_pool(get_prepared(), dict(scoring_items))


# --------------------------------------------------------------------------
# Session defaults
# --------------------------------------------------------------------------

def init_state():
    ss = st.session_state
    ss.setdefault("stage", "setup")
    ss.setdefault("w_format", "Snake")
    ss.setdefault("w_type", "Redraft")
    ss.setdefault("w_teams", 10)
    ss.setdefault("w_slot", 1)
    ss.setdefault("w_preset", "PPR")
    for key, value in DEFAULT_ROSTER.items():
        ss.setdefault(f"r_{key}", value)
    for key in SCORING_LABELS:
        ss.setdefault(f"s_{key}", float(projections.DEFAULT_SCORING[key]))
    ss.setdefault("chat", [])
    ss.setdefault("rules_msg", None)
    # Streamlit forgets a widget's value when the widget is off screen.
    # Re-saving the setup values keeps them while the draft room is showing.
    for key in list(ss.keys()):
        if key.startswith(("w_", "r_", "s_", "tn_")):
            ss[key] = ss[key]


def apply_preset():
    st.session_state["s_rec"] = PRESETS[st.session_state["w_preset"]]


def read_rules_clicked():
    ss = st.session_state
    upload = ss.get("rules_file")
    if upload is None:
        ss["rules_msg"] = ("warning", "Choose a file first.")
        return
    if not API_KEY:
        ss["rules_msg"] = ("error", "No Gemini API key is set, so the document reader "
                                    "is off. Add the key (see README) or fill in the "
                                    "fields by hand.")
        return
    try:
        found = gm_module.read_rules(upload.name, upload.getvalue(), API_KEY, MODEL)
    except Exception as exc:
        ss["rules_msg"] = ("error", f"Could not read that document: {exc}")
        return

    filled = []
    if "teams" in found:
        ss["w_teams"] = found["teams"]
        ss["w_slot"] = min(ss["w_slot"], found["teams"])
        filled.append("league size")
    if "draft_format" in found:
        ss["w_format"] = found["draft_format"]
        filled.append("draft format")
    if "league_type" in found:
        ss["w_type"] = found["league_type"]
        filled.append("league type")
    for key, value in found["roster"].items():
        ss[f"r_{key}"] = value
    if found["roster"]:
        filled.append(f"{len(found['roster'])} roster slots")
    for key, value in found["scoring"].items():
        ss[f"s_{key}"] = value
    if found["scoring"]:
        filled.append(f"{len(found['scoring'])} scoring values")

    note = ""
    if "teams_unsupported" in found:
        note = (f" The document says {found['teams_unsupported']} teams; this tool "
                "supports 8, 10 or 12, so league size was left alone.")
    if filled:
        ss["rules_msg"] = ("success", "Filled in: " + ", ".join(filled) + ". Check every "
                           "value below before you start; the reader can misread a "
                           "document." + note)
    else:
        ss["rules_msg"] = ("warning", "No settings were found in that document." + note)


# --------------------------------------------------------------------------
# Setup screen
# --------------------------------------------------------------------------

def header(subtitle: str):
    left, right = st.columns([1, 3], vertical_alignment="center")
    with left:
        if os.path.exists(LOGO):
            st.image(LOGO, width=260)
        else:
            st.title("FantasyIntel")
    with right:
        st.title("Draft GM")
        st.caption(subtitle)


def setup_screen():
    ss = st.session_state
    header("Step 1 of 2: League setup")
    st.write("Your GM uses these settings to project every player and plan each pick.")

    with st.container(border=True):
        st.subheader("Upload league rules")
        st.write("Add a PDF, Word or text file of your league's settings. Your GM reads "
                 "it and fills in the fields below for you to check. Optional.")
        st.file_uploader("League rules file", type=["pdf", "docx", "txt"], key="rules_file")
        st.button("Read rules and fill in settings", on_click=read_rules_clicked)
        if ss["rules_msg"]:
            kind, text = ss["rules_msg"]
            getattr(st, kind)(text)

    draft_col, league_col = st.columns(2)
    with draft_col, st.container(border=True):
        st.subheader("Draft")
        st.radio("Draft format", ["Snake", "Auction"], key="w_format", horizontal=True)
        if ss["w_format"] == "Auction":
            st.info("Auction support is planned. For now the draft room runs a snake draft.")
        st.radio("League type", ["Redraft", "Keeper", "Dynasty"], key="w_type", horizontal=True)
        if ss["w_type"] == "Dynasty":
            st.info("Dynasty values (age, future seasons) are planned. Projections "
                    "cover the coming season only.")

    with league_col, st.container(border=True):
        st.subheader("League")
        st.radio("Number of teams", [8, 10, 12], key="w_teams", horizontal=True)
        teams = ss["w_teams"]
        if ss["w_slot"] > teams:
            ss["w_slot"] = teams
        with st.expander("Team names (optional)"):
            st.caption("Name the teams in draft order, or leave blank to use "
                       "Team 1, Team 2 and so on.")
            cols = st.columns(2)
            for i in range(teams):
                cols[i % 2].text_input(f"Team {i + 1}", key=f"tn_{i}",
                                       placeholder=f"Team {i + 1}")
        st.selectbox("Your draft position", list(range(1, teams + 1)), key="w_slot",
                     format_func=lambda n: f"Pick {n}")
        st.caption("Snake order reverses every round.")

    with st.container(border=True):
        st.subheader("Roster slots")
        cols = st.columns(len(ROSTER_LABELS))
        for col, (key, label) in zip(cols, ROSTER_LABELS.items()):
            col.number_input(label, min_value=0, max_value=15, step=1, key=f"r_{key}")
        size = sum(int(ss[f"r_{k}"]) for k in ROSTER_LABELS)
        st.caption(f"{size} roster spots, {size * teams} total picks")

    with st.container(border=True):
        st.subheader("Scoring")
        st.radio("Start from a preset", list(PRESETS), key="w_preset", horizontal=True,
                 on_change=apply_preset)
        cols = st.columns(4)
        for i, (key, label) in enumerate(SCORING_LABELS.items()):
            cols[i % 4].number_input(label, step=0.5, format="%.2f", key=f"s_{key}")

    keepers = None
    if ss["w_type"] != "Redraft":
        with st.container(border=True):
            st.subheader("Players already on rosters")
            st.caption("Add one row per kept player. Each one uses a roster spot, so "
                       "that team makes one fewer pick at the end of the draft.")
            names = [ss.get(f"tn_{i}") or f"Team {i + 1}" for i in range(teams)]
            keepers = st.data_editor(
                pd.DataFrame({"Team": pd.Series(dtype="str"),
                              "Player": pd.Series(dtype="str")}),
                num_rows="dynamic", key="keepers",
                column_config={"Team": st.column_config.SelectboxColumn(options=names)},
            )

    problems = []
    if size == 0:
        problems.append("Add at least one roster slot.")
    for key in ("pass_yd_per_pt", "rush_yd_per_pt", "rec_yd_per_pt"):
        if ss[f"s_{key}"] <= 0:
            problems.append(f"{SCORING_LABELS[key]} must be above zero.")
    for problem in problems:
        st.error(problem)

    if st.button("Start draft", type="primary", disabled=bool(problems)):
        start_draft(keepers)


def start_draft(keepers):
    ss = st.session_state
    scoring = {key: float(ss[f"s_{key}"]) for key in SCORING_LABELS}
    try:
        pool, context = get_pool(tuple(sorted(scoring.items())))
    except Exception as exc:
        st.error(f"Could not load NFL data: {exc}")
        return

    teams = ss["w_teams"]
    settings = LeagueSettings(
        teams=teams, my_slot=ss["w_slot"],
        roster={key: int(ss[f"r_{key}"]) for key in ROSTER_LABELS},
        team_names=[ss.get(f"tn_{i}") or "" for i in range(teams)],
        draft_format=ss["w_format"], league_type=ss["w_type"],
    )
    draft = Draft(pool, settings)
    helper = gm_module.GM(draft, context)

    skipped = []
    if keepers is not None:
        for row in keepers.dropna(subset=["Player"]).itertuples():
            if not str(row.Player).strip():
                continue
            pid = helper.find_player(row.Player)
            team = (settings.team_names.index(row.Team)
                    if row.Team in settings.team_names else None)
            try:
                if pid is None or team is None:
                    raise ValueError("not found")
                draft.pick(pid, team=team, keeper=True)
            except Exception:
                skipped.append(str(row.Player))

    ss["draft"], ss["gm"], ss["chat"] = draft, helper, []
    ss["skipped_keepers"] = skipped
    ss["stage"] = "draft"
    st.rerun()


# --------------------------------------------------------------------------
# Draft room
# --------------------------------------------------------------------------

def draft_room():
    ss = st.session_state
    draft: Draft = ss["draft"]
    helper: gm_module.GM = ss["gm"]
    s = draft.settings
    me = s.my_team

    header("Step 2 of 2: Draft room")
    if ss.get("skipped_keepers"):
        st.warning("These kept players could not be matched and were skipped: "
                   + ", ".join(ss["skipped_keepers"]))

    clock = draft.on_clock()
    wait = draft.picks_until(me)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Round", draft.round_number if clock is not None else "Done")
    m2.metric("Overall pick", draft.pick_number if clock is not None else len(draft.picks))
    m3.metric("On the clock", "You" if clock == me else
              (s.team_names[clock] if clock is not None else "Complete"))
    m4.metric("Picks until your turn", "Now" if clock == me else
              (wait if wait is not None else "None left"))

    a1, a2, a3, a4 = st.columns(4)
    if a1.button("Auto-pick this pick", disabled=clock is None,
                 help="Let the GM choose for whichever team is on the clock."):
        draft.auto_pick()
        st.rerun()
    if a2.button("Simulate to my pick", disabled=clock is None or clock == me,
                 help="Auto-pick for the other teams until you are on the clock."):
        while draft.on_clock() not in (None, me):
            draft.auto_pick()
        st.rerun()
    if a3.button("Undo last pick", disabled=not any(not p["keeper"] for p in draft.picks)):
        draft.undo()
        st.rerun()
    if a4.button("New draft"):
        ss["stage"] = "setup"
        st.rerun()

    board_col, side_col = st.columns([2, 1])

    with board_col:
        if clock is None:
            st.success("The draft is complete.")
        else:
            st.subheader("GM recommendation for your team")
            target = "you" if clock == me else s.team_names[clock]
            recs = draft.recommend(n=3)
            for col, rec in zip(st.columns(3), recs):
                with col, st.container(border=True):
                    st.markdown(f"**{rec['name']}**  \n{rec['pos']}, {rec['team']} · "
                                f"{rec['proj_points']:.0f} pts")
                    st.caption(rec["reason"])
                    label = "Draft" if clock == me else f"Mark drafted by {target}"
                    if st.button(label, key=f"rec_{rec['player_id']}"):
                        draft.pick(rec["player_id"])
                        st.rerun()

        st.subheader("Player board")
        f1, f2 = st.columns([2, 1])
        pos = f1.radio("Position", BOARD_POSITIONS, horizontal=True,
                       format_func=lambda p: "D/ST" if p == "DST" else p)
        search = f2.text_input("Search by name")
        left = draft.available(None if pos == "All" else pos)
        if search:
            left = left[left["name"].str.contains(search, case=False, na=False, regex=False)]
        table = left.reset_index()[["name", "pos", "team", "proj_points", "vor",
                                    "ppg", "notes"]].head(300)
        table.columns = ["Player", "Pos", "Team", "Proj pts", "Vs replacement",
                         "Pts/game", "Adjustments"]
        event = st.dataframe(
            table, hide_index=True, height=420,
            on_select="rerun", selection_mode="single-row",
            key=f"board_{len(draft.picks)}_{pos}_{search}",
            column_config={"Vs replacement": st.column_config.NumberColumn(format="%.0f")},
        )
        chosen = event.selection.rows if event and event.selection else []
        if chosen and clock is not None:
            pid = left.index[chosen[0]]
            who = "your team" if clock == me else s.team_names[clock]
            if st.button(f"Draft {left.at[pid, 'name']} for {who}", type="primary"):
                draft.pick(pid)
                st.rerun()
        elif clock is not None:
            st.caption("Select a row to record the pick for the team on the clock.")

    with side_col:
        st.subheader("Your roster")
        rows = [{"Slot": "D/ST" if slot == "DST" else slot.title() if slot in ("FLEX", "BENCH") else slot,
                 "Player": draft.pool.at[pid, "name"] if pid else "",
                 "Proj": draft.pool.at[pid, "proj_points"] if pid else None}
                for slot, pid in draft.lineup(me)]
        st.dataframe(pd.DataFrame(rows), hide_index=True)

        with st.expander("Pick log"):
            log = [{"Pick": p["number"] or "Kept", "Team": s.team_names[p["team"]],
                    "Player": draft.pool.at[p["player_id"], "name"],
                    "Pos": draft.pool.at[p["player_id"], "pos"]}
                   for p in reversed(draft.picks)]
            st.dataframe(pd.DataFrame(log), hide_index=True)

        st.subheader("Ask the GM")
        if not API_KEY:
            st.caption("No Gemini key set: answers use built-in question patterns.")
        box = st.container(height=360)
        for role, text in ss["chat"]:
            box.chat_message(role).markdown(text)
        question = st.chat_input("Who should I pick?")
        if question:
            box.chat_message("user").markdown(question)
            with box.chat_message("assistant"):
                with st.spinner("Checking the numbers."):
                    reply = helper.answer(question, ss["chat"], API_KEY, MODEL)
                st.markdown(reply)
            ss["chat"] += [("user", question), ("assistant", reply)]


init_state()
if st.session_state["stage"] == "draft" and "draft" in st.session_state:
    draft_room()
else:
    setup_screen()
