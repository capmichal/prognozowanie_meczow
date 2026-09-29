from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from experiment_utils import poisson_1x2_probs
from pipeline import BettingEngine


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


st.set_page_config(page_title="Final Demo | 1X2", page_icon="⚽", layout="wide")


def _ensure_final_predict_artifacts() -> None:
    required = [
        OUT / "final_summary_trackA.csv",
        OUT / "final_summary_trackB.csv",
        OUT / "final_significance.csv",
        OUT / "final_model_cards.csv",
        OUT / "final_decision_log.csv",
    ]
    if all(p.exists() for p in required):
        return
    subprocess.run(["python", str(ROOT / "run_final_pipeline.py"), "--mode", "both"], check=True, cwd=str(ROOT))


@st.cache_data
def load_final_tables() -> dict[str, pd.DataFrame]:
    _ensure_final_predict_artifacts()
    return {
        "track_a": pd.read_csv(OUT / "final_summary_trackA.csv"),
        "track_b": pd.read_csv(OUT / "final_summary_trackB.csv"),
        "significance": pd.read_csv(OUT / "final_significance.csv"),
        "model_cards": pd.read_csv(OUT / "final_model_cards.csv"),
        "decision_log": pd.read_csv(OUT / "final_decision_log.csv"),
    }


@st.cache_resource
def load_demo_engine() -> BettingEngine:
    engine = BettingEngine(
        path_results="datasets/premier_league_20-21.csv",
        path_shots="datasets/shots_premier_league_20-21.csv",
        window_size=5,
    )
    engine.prepare_data()
    return engine


def _team_list(engine: BettingEngine) -> list[str]:
    home_teams = engine.df["HomeTeam"].dropna().unique().tolist()
    away_teams = engine.df["AwayTeam"].dropna().unique().tolist()
    return sorted(set(home_teams + away_teams))


def _bookmaker_implied_probs(odd_h: float, odd_d: float, odd_a: float) -> tuple[np.ndarray, float]:
    q = np.array([1.0 / odd_h, 1.0 / odd_d, 1.0 / odd_a], dtype=float)
    overround = float(q.sum() - 1.0)
    p = q / q.sum()
    return p, overround


def _model_probs_from_xg(home_xg: float, away_xg: float) -> np.ndarray:
    p = poisson_1x2_probs(np.array([home_xg], dtype=float), np.array([away_xg], dtype=float), max_goals=10)
    return p[0]


def _fmt_pct(x: float) -> str:
    return f"{100.0 * float(x):.1f}%"


def render_predict_tab() -> None:
    st.subheader("Single Match Demo")
    st.caption("Interactive single-match demo uses xG proxy model for fast presentation. Final thesis evidence is in final tables.")

    engine = load_demo_engine()
    teams = _team_list(engine)

    c1, c2 = st.columns(2)
    with c1:
        home_team = st.selectbox("Home team", teams, index=0)
    with c2:
        away_team = st.selectbox("Away team", teams, index=1 if len(teams) > 1 else 0)

    st.markdown("Bookmaker odds (pre-match)")
    o1, o2, o3 = st.columns(3)
    with o1:
        odd_h = st.number_input("Home", min_value=1.01, value=2.10, step=0.01)
    with o2:
        odd_d = st.number_input("Draw", min_value=1.01, value=3.40, step=0.01)
    with o3:
        odd_a = st.number_input("Away", min_value=1.01, value=3.20, step=0.01)

    if st.button("Run match comparison", type="primary", use_container_width=True):
        if home_team == away_team:
            st.error("Choose two different teams.")
            return

        pred = engine.get_prediction(home_team, away_team)
        p_model = _model_probs_from_xg(float(pred["home_xg"]), float(pred["away_xg"]))
        p_book, overround = _bookmaker_implied_probs(float(odd_h), float(odd_d), float(odd_a))

        st.success(f"xG proxy: {home_team} {pred['home_xg']} - {pred['away_xg']} {away_team}")

        left, right = st.columns(2)
        with left:
            st.markdown("**Model probabilities (xG -> Poisson 1X2)**")
            st.write(pd.DataFrame({"Outcome": ["Home", "Draw", "Away"], "Probability": [_fmt_pct(x) for x in p_model]}))
        with right:
            st.markdown("**Bookmaker implied (de-vig normalized)**")
            st.write(pd.DataFrame({"Outcome": ["Home", "Draw", "Away"], "Probability": [_fmt_pct(x) for x in p_book]}))
            st.caption(f"Overround: {100.0 * overround:.2f}%")


def render_results_tab(tables: dict[str, pd.DataFrame]) -> None:
    st.subheader("Final Results (Track A / Track B)")
    st.markdown("**Track A**")
    st.dataframe(tables["track_a"], use_container_width=True, hide_index=True)

    st.markdown("**Track B**")
    st.dataframe(tables["track_b"], use_container_width=True, hide_index=True)

    st.markdown("**Final significance**")
    st.dataframe(tables["significance"], use_container_width=True, hide_index=True)


def render_method_tab(tables: dict[str, pd.DataFrame]) -> None:
    st.subheader("Method Notes")
    st.markdown(
        "- Track A: models without odds.\n"
        "- Track B: models with market features and mandatory bookmaker benchmark.\n"
        "- Final significance compares only finalists selected in Day 1.\n"
        "- Day 3 final pipeline consolidates tables for thesis reporting."
    )
    st.markdown("**Model cards**")
    st.dataframe(tables["model_cards"], use_container_width=True, hide_index=True)
    st.markdown("**Decision log**")
    st.dataframe(tables["decision_log"], use_container_width=True, hide_index=True)


def main() -> None:
    st.title("Final 1X2 Demo")
    st.caption("Master thesis demo app: final reporting + interactive single-match comparison")

    tables = load_final_tables()
    tab1, tab2, tab3 = st.tabs(["Prediction", "Final Tables", "Method"])
    with tab1:
        render_predict_tab()
    with tab2:
        render_results_tab(tables)
    with tab3:
        render_method_tab(tables)


if __name__ == "__main__":
    main()
