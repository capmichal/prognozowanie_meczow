from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd

Outcome = Literal["H", "D", "A"]


SEASONS_V1: tuple[str, ...] = (
    "14-15",
    "15-16",
    "16-17",
    "17-18",
    "18-19",
    "19-20",
    "20-21",
)

TRAIN_SEASONS_V1: set[str] = {"14-15", "15-16", "16-17", "17-18", "18-19"}
VALID_SEASONS_V1: set[str] = {"19-20"}
TEST_SEASONS_V1: set[str] = {"20-21"}


@dataclass(frozen=True)
class OddsColumns:
    home: str
    draw: str
    away: str


ODDS_B365 = OddsColumns("B365H", "B365D", "B365A")
ODDS_PS = OddsColumns("PSH", "PSD", "PSA")


# Understat team names are often longer than football-data.co.uk names.
# We map Understat -> football-data naming to make joins deterministic.
UNDERSTAT_TO_FOOTBALLDATA_TEAM: dict[str, str] = {
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Newcastle United": "Newcastle",
    "Wolverhampton Wanderers": "Wolves",
    "West Bromwich Albion": "West Brom",
    "Queens Park Rangers": "QPR",
    "Norwich City": "Norwich",
    "Stoke City": "Stoke",
    "Swansea City": "Swansea",
    "Hull City": "Hull",
    "Leicester City": "Leicester",
    "Cardiff City": "Cardiff",
    "Brighton": "Brighton",
    "Brighton and Hove Albion": "Brighton",
    "West Ham": "West Ham",
    "Tottenham": "Tottenham",
    "Tottenham Hotspur": "Tottenham",
    "Nottingham Forest": "Nott'm Forest",
    "Sheffield United": "Sheffield United",
    # La Liga
    "Atletico Madrid": "Ath Madrid",
    "Athletic Club": "Ath Bilbao",
    "Celta Vigo": "Celta",
    "Espanyol": "Espanol",
    "Deportivo La Coruna": "La Coruna",
    "Real Sociedad": "Sociedad",
    "Rayo Vallecano": "Vallecano",
    "Real Betis": "Betis",
    "Sporting Gijon": "Sp Gijon",
    "Real Valladolid": "Valladolid",
    "SD Huesca": "Huesca",
    "Las Palmas": "Las Palmas",
    # Bundesliga
    "Borussia Dortmund": "Dortmund",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "FC Cologne": "FC Koln",
    "Hamburger SV": "Hamburg",
    "Hannover 96": "Hannover",
    "Hertha Berlin": "Hertha",
    "Bayer Leverkusen": "Leverkusen",
    "Borussia M.Gladbach": "M'gladbach",
    "Mainz 05": "Mainz",
    "VfB Stuttgart": "Stuttgart",
    "RB Leipzig": "RB Leipzig",
    "RasenBallsport Leipzig": "RB Leipzig",
    "Fortuna Dusseldorf": "Fortuna Dusseldorf",
    "Fortuna Duesseldorf": "Fortuna Dusseldorf",
    "Nuernberg": "Nurnberg",
    "Nurnberg": "Nurnberg",
    "Arminia Bielefeld": "Bielefeld",
}


def normalize_understat_team_name(name: str) -> str:
    if not isinstance(name, str):
        return name
    name = name.strip()
    return UNDERSTAT_TO_FOOTBALLDATA_TEAM.get(name, name)


def load_results_season(csv_path: str | Path, *, season: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Some seasons have 'Time', some don't. We only require the columns below.
    required = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in results {csv_path}: {missing}")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce").dt.normalize()
    df = df.dropna(subset=["Date"]).copy()

    df["Season"] = season

    # Keep odds columns if present; we will decide later which to use.
    keep_cols = set(required) | {"Season"}
    for col in (ODDS_B365.home, ODDS_B365.draw, ODDS_B365.away, ODDS_PS.home, ODDS_PS.draw, ODDS_PS.away):
        if col in df.columns:
            keep_cols.add(col)

    return df.loc[:, [c for c in df.columns if c in keep_cols]].copy()


def load_shots_season(csv_path: str | Path, *, season: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = ["match_id", "date", "h_team", "a_team", "h_a", "xG"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in shots {csv_path}: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"]).copy()
    df["Season"] = season

    # Normalize team names to match football-data naming.
    df["h_team"] = df["h_team"].map(normalize_understat_team_name)
    df["a_team"] = df["a_team"].map(normalize_understat_team_name)

    return df


def build_matches_with_xg(results: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    """Merge results with xG aggregated from shot-level data.

    Returns match-level rows with Home_xG/Away_xG.
    """
    xg_grouped = shots.groupby(["Season", "match_id", "h_a"], as_index=False)["xG"].sum()
    xg_pivot = (
        xg_grouped.pivot(index=["Season", "match_id"], columns="h_a", values="xG")
        .reset_index()
        .rename(columns={"h": "Home_xG", "a": "Away_xG"})
    )
    xg_pivot["Home_xG"] = xg_pivot["Home_xG"].fillna(0.0)
    xg_pivot["Away_xG"] = xg_pivot["Away_xG"].fillna(0.0)

    match_info = (
        shots[["Season", "match_id", "date", "h_team", "a_team"]]
        .drop_duplicates(subset=["Season", "match_id"])
        .rename(columns={"date": "Date", "h_team": "HomeTeam", "a_team": "AwayTeam"})
    )

    merged = results.merge(
        match_info[["Season", "Date", "HomeTeam", "AwayTeam", "match_id"]],
        on=["Season", "Date", "HomeTeam", "AwayTeam"],
        how="left",
    )

    # Fallback for datasets where dates differ by a day between providers.
    # In league data, (Season, HomeTeam, AwayTeam) should be unique, so this
    # is a safe deterministic backup only for rows unmatched on exact date.
    unresolved_mask = merged["match_id"].isna().to_numpy()
    if unresolved_mask.any():
        match_info_by_teams = match_info[["Season", "HomeTeam", "AwayTeam", "match_id"]].copy()
        team_counts = (
            match_info_by_teams.groupby(["Season", "HomeTeam", "AwayTeam"], as_index=False)
            .size()
            .rename(columns={"size": "_n"})
        )
        unique_team_matches = team_counts[team_counts["_n"] == 1][["Season", "HomeTeam", "AwayTeam"]]
        match_info_by_teams = match_info_by_teams.merge(
            unique_team_matches,
            on=["Season", "HomeTeam", "AwayTeam"],
            how="inner",
        )

        fallback = merged.loc[unresolved_mask, ["Season", "HomeTeam", "AwayTeam"]].merge(
            match_info_by_teams,
            on=["Season", "HomeTeam", "AwayTeam"],
            how="left",
        )
        merged.loc[unresolved_mask, "match_id"] = fallback["match_id"].to_numpy()

    merged = merged.merge(
        xg_pivot[["Season", "match_id", "Home_xG", "Away_xG"]],
        on=["Season", "match_id"],
        how="left",
    )

    merged = merged.dropna(subset=["Home_xG", "Away_xG"]).copy()
    merged = merged.sort_values(["Date", "Season", "HomeTeam", "AwayTeam"]).reset_index(drop=True)

    # Unique index for stable ordering inside a date.
    merged["MatchIndex"] = np.arange(len(merged), dtype=int)
    return merged


def add_rolling_form_features(matches: pd.DataFrame, *, window_size: int) -> pd.DataFrame:
    """Compute per-team rolling means of goals and xG (shifted by 1 to avoid leakage).

    Rolling is computed continuously across seasons ordered by Date.
    """
    df = matches.copy()

    home = df[["Date", "MatchIndex", "HomeTeam", "FTHG", "FTAG", "Home_xG", "Away_xG"]].copy()
    home.columns = ["Date", "MatchIndex", "Team", "Goals_S", "Goals_C", "xG_S", "xG_C"]

    away = df[["Date", "MatchIndex", "AwayTeam", "FTAG", "FTHG", "Away_xG", "Home_xG"]].copy()
    away.columns = ["Date", "MatchIndex", "Team", "Goals_S", "Goals_C", "xG_S", "xG_C"]

    long_df = pd.concat([home, away], ignore_index=True)
    long_df = long_df.sort_values(["Team", "Date", "MatchIndex"]).reset_index(drop=True)

    metrics = ["Goals_S", "Goals_C", "xG_S", "xG_C"]

    def _rolling(group: pd.DataFrame) -> pd.DataFrame:
        rolled = (
            group[metrics]
            .rolling(window=window_size, min_periods=1)
            .mean()
            .shift(1)
        )
        rolled.columns = [f"Forma_{c}_{window_size}" for c in metrics]
        return rolled

    rolled = long_df.groupby("Team", group_keys=False).apply(_rolling)
    long_df = pd.concat([long_df, rolled], axis=1)

    form_cols = [f"Forma_{c}_{window_size}" for c in metrics]

    home_form = long_df.merge(
        df[["MatchIndex", "HomeTeam"]],
        left_on=["MatchIndex", "Team"],
        right_on=["MatchIndex", "HomeTeam"],
        how="inner",
    )[["MatchIndex"] + form_cols]
    home_form = home_form.rename(columns={c: f"{c}_H" for c in form_cols})

    away_form = long_df.merge(
        df[["MatchIndex", "AwayTeam"]],
        left_on=["MatchIndex", "Team"],
        right_on=["MatchIndex", "AwayTeam"],
        how="inner",
    )[["MatchIndex"] + form_cols]
    away_form = away_form.rename(columns={c: f"{c}_A" for c in form_cols})

    out = df.merge(home_form, on="MatchIndex", how="left").merge(away_form, on="MatchIndex", how="left")
    return out


def choose_odds_source(df: pd.DataFrame) -> pd.Series:
    """Choose which odds source to use per row: B365 if present, else PS, else NaN."""

    def _valid_triplet(cols: OddsColumns) -> pd.Series:
        ok = np.ones(len(df), dtype=bool)
        for c in (cols.home, cols.draw, cols.away):
            if c not in df.columns:
                return pd.Series(np.zeros(len(df), dtype=bool), index=df.index)
            ok &= df[c].notna().to_numpy()
            # Odds must be > 1.0 to make sense.
            ok &= (df[c].to_numpy() > 1.0)
        return pd.Series(ok, index=df.index)

    b365_ok = _valid_triplet(ODDS_B365)
    ps_ok = _valid_triplet(ODDS_PS)

    # Use object dtype (strings + missing)
    out = np.full(len(df), None, dtype=object)
    out[ps_ok.to_numpy()] = "PS"
    out[b365_ok.to_numpy()] = "B365"  # prefer B365 when both exist
    return pd.Series(out, index=df.index, dtype="object")


def implied_probs_from_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Compute implied probs from odds (with overround removed by normalization)."""
    source = choose_odds_source(df)

    p_home = np.full(len(df), np.nan, dtype=float)
    p_draw = np.full(len(df), np.nan, dtype=float)
    p_away = np.full(len(df), np.nan, dtype=float)

    def _fill(mask: np.ndarray, cols: OddsColumns) -> None:
        inv_h = 1.0 / df.loc[mask, cols.home].to_numpy(dtype=float)
        inv_d = 1.0 / df.loc[mask, cols.draw].to_numpy(dtype=float)
        inv_a = 1.0 / df.loc[mask, cols.away].to_numpy(dtype=float)
        s = inv_h + inv_d + inv_a
        inv_h /= s
        inv_d /= s
        inv_a /= s
        p_home[mask] = inv_h
        p_draw[mask] = inv_d
        p_away[mask] = inv_a

    b365_mask = source.eq("B365").to_numpy()
    ps_mask = source.eq("PS").to_numpy()

    if b365_mask.any():
        _fill(b365_mask, ODDS_B365)
    if ps_mask.any():
        _fill(ps_mask, ODDS_PS)

    return pd.DataFrame({"pH": p_home, "pD": p_draw, "pA": p_away}, index=df.index)


def outcome_from_score(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.where(
            df["FTHG"].to_numpy() > df["FTAG"].to_numpy(),
            "H",
            np.where(df["FTHG"].to_numpy() < df["FTAG"].to_numpy(), "A", "D"),
        ),
        index=df.index,
        dtype="object",
    )


def poisson_1x2_probs(lam_home: np.ndarray, lam_away: np.ndarray, *, max_goals: int = 10) -> np.ndarray:
    """Convert lambdas to 1X2 probabilities using independent Poisson scoreline.

    Returns array (n, 3) for [pH, pD, pA].
    """
    lam_home = np.asarray(lam_home, dtype=float)
    lam_away = np.asarray(lam_away, dtype=float)

    # Guard against invalid values
    lam_home = np.clip(lam_home, 1e-6, 20.0)
    lam_away = np.clip(lam_away, 1e-6, 20.0)

    goals = np.arange(max_goals + 1)

    # Compute Poisson PMFs without SciPy for simplicity and speed.
    # pmf(k; lam) = exp(-lam) * lam^k / k!
    fact = np.array([math.factorial(int(k)) for k in goals], dtype=float)

    pmf_h = np.exp(-lam_home[:, None]) * (lam_home[:, None] ** goals[None, :]) / fact[None, :]
    pmf_a = np.exp(-lam_away[:, None]) * (lam_away[:, None] ** goals[None, :]) / fact[None, :]

    # Score matrix probabilities: P(H=k, A=j) = pmf_h[k] * pmf_a[j]
    # For each match, build outer product.
    # Efficiently compute using broadcasting: (n, K, 1) * (n, 1, K) -> (n, K, K)
    score_prob = pmf_h[:, :, None] * pmf_a[:, None, :]

    # Home win: k > j, draw: k == j, away win: k < j
    k = goals[None, :, None]
    j = goals[None, None, :]

    pH = np.sum(score_prob * (k > j), axis=(1, 2))
    pD = np.sum(score_prob * (k == j), axis=(1, 2))
    pA = np.sum(score_prob * (k < j), axis=(1, 2))

    probs = np.stack([pH, pD, pA], axis=1)

    # Normalize to defend against truncation error due to max_goals.
    s = probs.sum(axis=1, keepdims=True)
    probs = probs / np.clip(s, 1e-12, None)
    return probs


def brier_score_multiclass(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    """Multiclass Brier score (mean squared error over classes)."""
    y_true = np.asarray(y_true)
    p_pred = np.asarray(p_pred, dtype=float)

    labels = np.array(["H", "D", "A"], dtype=object)
    y_onehot = (y_true[:, None] == labels[None, :]).astype(float)
    return float(np.mean(np.sum((p_pred - y_onehot) ** 2, axis=1)))


def log_loss_multiclass(y_true: np.ndarray, p_pred: np.ndarray, *, eps: float = 1e-12) -> float:
    """Multiclass log loss with fixed class order [H, D, A] for columns.

    Computes mean(-log(p_true)).
    """
    y_true = np.asarray(y_true)
    p_pred = clip_and_renormalize(p_pred, eps=eps)
    idx = {"H": 0, "D": 1, "A": 2}
    p_true = np.array([p_pred[i, idx[y_true[i]]] for i in range(len(y_true))], dtype=float)
    return float(np.mean(-np.log(np.clip(p_true, eps, 1.0))))


def clip_and_renormalize(probs: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    probs = np.clip(probs, eps, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs
