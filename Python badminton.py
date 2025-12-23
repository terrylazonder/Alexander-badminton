import streamlit as st
import pandas as pd
from itertools import combinations
import random

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

NAME_COL = "Name"
RATING_COL = "Rating"
GENDER_COL = "Gender"  # "M" or "V"
FILE_PATH = "ratings.xlsx"

MAX_MATCHES = 6
MAX_PLAYERS_ON_COURT = MAX_MATCHES * 4  # 24 players

# Default tolerance (slider can override this)
BALANCE_TOLERANCE = 3


# -------------------------------------------------------
# SMALL HELPERS
# -------------------------------------------------------

def gender_icon(g: str) -> str:
    return "🚹" if g == "M" else "🚺"


# -------------------------------------------------------
# CONFLICT + BALANCE LOGIC
# -------------------------------------------------------

def compute_conflict_score(matches, history_pairs):
    score = 0
    for match in matches:
        players = match["team1"] + match["team2"]
        names = [p[NAME_COL] for p in players]
        for a, b in combinations(names, 2):
            if frozenset({a, b}) in history_pairs:
                score += 1
    return score


def compute_balance_score(matches, tolerance=BALANCE_TOLERANCE):
    """Penalty for unbalanced matches.

    Differences up to `tolerance` are treated as 'equal' (zero penalty).
    Only the excess beyond tolerance is penalized (squared).
    """
    total = 0
    for match in matches:
        sum1 = sum(p[RATING_COL] for p in match["team1"])
        sum2 = sum(p[RATING_COL] for p in match["team2"])
        diff = abs(sum1 - sum2)
        excess = max(0, diff - tolerance)
        total += excess ** 2
    return total


def select_players_for_courts(active_df, play_counts, max_players=MAX_PLAYERS_ON_COURT):
    """Pick up to max_players, rotating fairly so the same people are not always reserve."""
    if len(active_df) <= max_players:
        return active_df.copy()

    tmp = active_df.sample(frac=1).reset_index(drop=True)
    tmp["played"] = tmp[NAME_COL].map(lambda n: play_counts.get(n, 0))
    tmp = tmp.sort_values("played")
    return tmp.iloc[:max_players].drop(columns=["played"]).reset_index(drop=True)


# -------------------------------------------------------
# TEAM + MATCH HELPERS
# -------------------------------------------------------

def team_rating(team):
    return team[0][RATING_COL] + team[1][RATING_COL]


def make_matches_from_teams(teams):
    """Turn a list of 2-player teams into matches.

    Strategy:
    - Sort teams by team strength to get roughly even matchups.
    - Within small rating bands (chunks of 4 teams), shuffle to avoid
      getting the exact same matches every round.
    """
    if len(teams) < 2:
        return []
    if len(teams) % 2 == 1:
        teams = teams[:-1]

    teams = sorted(teams, key=team_rating)

    matches = []
    # Work in small chunks to add variety without ruining balance
    for i in range(0, len(teams), 4):
        chunk = teams[i:i + 4]
        if len(chunk) < 2:
            continue
        random.shuffle(chunk)
        for j in range(0, len(chunk), 2):
            if j + 1 < len(chunk):
                matches.append({"team1": chunk[j], "team2": chunk[j + 1]})
    return matches


# -------------------------------------------------------
# TEAM GENERATORS
# -------------------------------------------------------

def schedule_random(df):
    df = df.sample(frac=1).reset_index(drop=True)
    n = (len(df) // 4) * 4
    if n < 4:
        return []
    df = df.iloc[:n]

    teams = [[df.iloc[i], df.iloc[i + 1]] for i in range(0, n, 2)]
    return make_matches_from_teams(teams)


def schedule_gender(df):
    males = df[df[GENDER_COL] == "M"].sample(frac=1).reset_index(drop=True)
    females = df[df[GENDER_COL] == "V"].sample(frac=1).reset_index(drop=True)

    teams = []
    mixed = min(len(males), len(females))

    for i in range(mixed):
        teams.append([males.iloc[i], females.iloc[i]])

    leftover_m = males.iloc[mixed:]
    leftover_f = females.iloc[mixed:]

    for i in range(0, len(leftover_m) - 1, 2):
        teams.append([leftover_m.iloc[i], leftover_m.iloc[i + 1]])
    for i in range(0, len(leftover_f) - 1, 2):
        teams.append([leftover_f.iloc[i], leftover_f.iloc[i + 1]])

    if len(teams) < 2:
        return []
    if len(teams) % 2 == 1:
        teams = teams[:-1]

    return make_matches_from_teams(teams)


def schedule_competition(df, split):
    """
    Competition:
    - Prefer strong vs strong and weak vs weak based on split.
    - BUT if that would produce < 6 matches while there are enough players overall,
      fill the remaining matches using leftover players from BOTH pools.
    """
    strong = df[df[RATING_COL] < split].sample(frac=1).reset_index(drop=True)
    weak = df[df[RATING_COL] >= split].sample(frac=1).reset_index(drop=True)

    def build_matches_from_group(group, max_matches):
        max_players = max_matches * 4
        n = min(len(group), max_players)
        n = (n // 4) * 4
        if n < 4:
            return [], group.reset_index(drop=True)

        used = group.iloc[:n].reset_index(drop=True)
        left = group.iloc[n:].reset_index(drop=True)

        teams = [[used.iloc[i], used.iloc[i + 1]] for i in range(0, n, 2)]
        return make_matches_from_teams(teams), left

    # 1) Prefer in-pool matches
    strong_matches, strong_left = build_matches_from_group(strong, 3)
    weak_matches, weak_left = build_matches_from_group(weak, 3)

    matches = (strong_matches + weak_matches)[:MAX_MATCHES]

    # 2) If still short, fill using leftovers combined (so you get 6 matches if possible)
    if len(matches) < MAX_MATCHES:
        leftovers = pd.concat([strong_left, weak_left], ignore_index=True).sample(frac=1).reset_index(drop=True)

        remaining_needed = MAX_MATCHES - len(matches)
        fill_matches, _ = build_matches_from_group(leftovers, remaining_needed)

        matches += fill_matches

    return matches[:MAX_MATCHES]


# -------------------------------------------------------
# ROUND GENERATOR
# -------------------------------------------------------

def generate_round(df, mode, history_pairs, split_rating, balance_tolerance):
    best = None
    best_score = None

    for _ in range(200):
        if mode == "random":
            matches = schedule_random(df)
        elif mode == "gender":
            matches = schedule_gender(df)
        else:
            matches = schedule_competition(df, split_rating)

        matches = matches[:MAX_MATCHES]
        if not matches:
            continue

        conflict = compute_conflict_score(matches, history_pairs)
        balance = compute_balance_score(matches, tolerance=balance_tolerance)

        score = (conflict, balance)  # lexicographic: prefer fewer conflicts, then better balance

        if best is None or score < best_score:
            best = matches
            best_score = score
            if conflict == 0:
                break

    return best or []


def update_history(matches, history_pairs):
    for match in matches:
        players = match["team1"] + match["team2"]
        names = [p[NAME_COL] for p in players]
        for a, b in combinations(names, 2):
            history_pairs.add(frozenset({a, b}))


def update_play_counts(matches, play_counts):
    for match in matches:
        for p in match["team1"] + match["team2"]:
            play_counts[p[NAME_COL]] = play_counts.get(p[NAME_COL], 0) + 1


# -------------------------------------------------------
# DISPLAY
# -------------------------------------------------------

def display_matches(matches, active_df):
    used = set()

    for i, match in enumerate(matches, start=1):
        team1 = "<br>".join(
            f"{gender_icon(p[GENDER_COL])} {p[NAME_COL]}"
            for p in match["team1"]
        )
        team2 = "<br>".join(
            f"{gender_icon(p[GENDER_COL])} {p[NAME_COL]}"
            for p in match["team2"]
        )

        raw_html = f"""
        <div style="border:2px solid #F7D94C;
                    border-radius:10px;
                    padding:12px;
                    margin:12px 0;
                    width:100%;
                    box-sizing:border-box;">

          <div style="font-weight:bold;font-size:18px;margin-bottom:8px;">
            Match {i}
          </div>

          <div style="display:flex;
                      flex-wrap:nowrap;
                      gap:60px;
                      width:100%;
                      box-sizing:border-box;">

            <div style="min-width:200px;">
              <div style="font-weight:bold;font-size:16px;">Team 1</div>
              <div style="margin-top:4px;font-size:15px;">{team1}</div>
            </div>

            <div style="min-width:200px;">
              <div style="font-weight:bold;font-size:16px;">Team 2</div>
              <div style="margin-top:4px;font-size:15px;">{team2}</div>
            </div>

          </div>
        </div>
        """

        html = "\n".join(line.lstrip() for line in raw_html.splitlines())
        st.markdown(html, unsafe_allow_html=True)

        for p in match["team1"] + match["team2"]:
            used.add(p[NAME_COL])

    reserves = active_df[~active_df[NAME_COL].isin(used)]
    if not reserves.empty:
        st.subheader("Reserves")
        for _, r in reserves.iterrows():
            st.write(f"{gender_icon(r[GENDER_COL])} {r[NAME_COL]}")


# -------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------

def main():
    st.title("BMC Team Generator 🏸")

    if "history_pairs" not in st.session_state:
        st.session_state["history_pairs"] = set()
    if "play_counts" not in st.session_state:
        st.session_state["play_counts"] = {}

    history_pairs = st.session_state["history_pairs"]
    play_counts = st.session_state["play_counts"]

    try:
        df = pd.read_excel(FILE_PATH)
    except Exception as e:
        st.error(f"Cannot load {FILE_PATH}: {e}")
        return

    for col in (NAME_COL, RATING_COL, GENDER_COL):
        if col not in df.columns:
            st.error(f"Column '{col}' not found in Excel.")
            return

    all_names = df[NAME_COL].tolist()

    tab_info, tab_matches = st.tabs(["Info & Players", "Create Matches"])

    # ---------- TAB 1 ----------
    with tab_info:
        st.image("Badminton foto.jpeg", use_container_width=True)

        st.subheader("Players")
        st.dataframe(df[[NAME_COL, RATING_COL, GENDER_COL]])

        split_rating = st.number_input(
            "Split random",
            min_value=0,
            max_value=100,
            value=24,
            step=1,
        )

        balance_tolerance = st.slider(
            "Balance tolerance (rating points treated as equal)",
            min_value=0,
            max_value=10,
            value=BALANCE_TOLERANCE,
            step=1,
            help="Differences up to this value between team ratings add 0 balance penalty. Higher = more flexibility / variety.",
        )

    # ---------- TAB 2 ----------
    with tab_matches:
        st.subheader("Who is present this round?")
        selected_names = st.multiselect(
            "",
            options=all_names,
            default=all_names,
        )
        st.caption(f"{len(selected_names)} players selected as present.")

        active_df = df[df[NAME_COL].isin(selected_names)].reset_index(drop=True)
        if len(active_df) < 4:
            st.warning("Select at least 4 players.")
            return

        players_on_court = select_players_for_courts(active_df, play_counts)
        st.caption(
            f"{len(players_on_court)} players on court (max {MAX_PLAYERS_ON_COURT}); "
            "others are reserves this round."
        )

        col1, col2, col3 = st.columns(3)
        matches_placeholder = st.container()

        with col1:
            if st.button("Mixed"):
                matches = generate_round(players_on_court, "gender", history_pairs, split_rating, balance_tolerance)
                with matches_placeholder:
                    display_matches(matches, active_df)
                    update_history(matches, history_pairs)
                    update_play_counts(matches, play_counts)

        with col2:
            if st.button("Random"):
                matches = generate_round(players_on_court, "random", history_pairs, split_rating, balance_tolerance)
                with matches_placeholder:
                    display_matches(matches, active_df)
                    update_history(matches, history_pairs)
                    update_play_counts(matches, play_counts)

        with col3:
            if st.button("Competition"):
                matches = generate_round(players_on_court, "competition", history_pairs, split_rating, balance_tolerance)
                with matches_placeholder:
                    display_matches(matches, active_df)
                    update_history(matches, history_pairs)
                    update_play_counts(matches, play_counts)


if __name__ == "__main__":
    main()

