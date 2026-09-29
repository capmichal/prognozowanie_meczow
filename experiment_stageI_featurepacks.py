from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiment_utils import (
    ODDS_B365,
    ODDS_PS,
    add_rolling_form_features,
    brier_score_multiclass,
    build_matches_with_xg,
    choose_odds_source,
    clip_and_renormalize,
    implied_probs_from_odds,
    load_results_season,
    load_shots_season,
    log_loss_multiclass,
    outcome_from_score,
)


SEASONS: tuple[str, ...] = ("14-15", "15-16", "16-17", "17-18", "18-19", "19-20", "20-21")
LEAGUES: dict[str, str] = {"EPL": "premier_league", "LaLiga": "la_liga", "Bundesliga": "bundesliga"}
ROLLING_WINDOWS: tuple[int, ...] = (3, 5, 10)


def _season_paths(base: Path, league_prefix: str, season: str) -> tuple[Path, Path]:
    results = base / f"{league_prefix}_{season}.csv"
    shots = base / f"shots_{league_prefix}_{season}.csv"
    if not results.exists() or not shots.exists():
        raise FileNotFoundError(f"Missing dataset files for {league_prefix} {season}")
    return results, shots


def load_league_matches_with_shots(base: Path, league_prefix: str) -> pd.DataFrame:
    results_all: list[pd.DataFrame] = []
    shots_all: list[pd.DataFrame] = []
    for season in SEASONS:
        rp, sp = _season_paths(base, league_prefix, season)
        results_all.append(load_results_season(rp, season=season))
        shots_all.append(load_shots_season(sp, season=season))

    results = pd.concat(results_all, ignore_index=True)
    shots = pd.concat(shots_all, ignore_index=True)
    matches = build_matches_with_xg(results, shots)

    shot_cnt = (
        shots.groupby(["Season", "match_id", "h_a"], as_index=False)
        .size()
        .pivot(index=["Season", "match_id"], columns="h_a", values="size")
        .reset_index()
        .rename(columns={"h": "Home_Shots", "a": "Away_Shots"})
    )
    shot_cnt["Home_Shots"] = shot_cnt["Home_Shots"].fillna(0.0)
    shot_cnt["Away_Shots"] = shot_cnt["Away_Shots"].fillna(0.0)

    out = matches.merge(shot_cnt, on=["Season", "match_id"], how="left")
    out["Home_Shots"] = out["Home_Shots"].fillna(0.0)
    out["Away_Shots"] = out["Away_Shots"].fillna(0.0)
    return out


def make_time_folds(min_train_seasons: int = 3) -> list[dict[str, object]]:
    folds: list[dict[str, object]] = []
    for test_idx in range(min_train_seasons + 1, len(SEASONS)):
        valid_idx = test_idx - 1
        folds.append(
            {
                "fold_id": len(folds) + 1,
                "train_seasons": list(SEASONS[:valid_idx]),
                "valid_season": SEASONS[valid_idx],
                "test_season": SEASONS[test_idx],
            }
        )
    return folds


def compute_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    probs = clip_and_renormalize(probs)
    return {
        "log_loss": float(log_loss_multiclass(y_true, probs)),
        "brier": float(brier_score_multiclass(y_true, probs)),
    }


def add_rest_days_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    home = out[["Date", "MatchIndex", "HomeTeam"]].rename(columns={"HomeTeam": "Team"})
    away = out[["Date", "MatchIndex", "AwayTeam"]].rename(columns={"AwayTeam": "Team"})
    long_df = pd.concat([home, away], ignore_index=True).sort_values(["Team", "Date", "MatchIndex"]).reset_index(drop=True)
    long_df["RestDays"] = long_df.groupby("Team")["Date"].diff().dt.days.astype(float)

    home_rest = long_df.merge(
        out[["MatchIndex", "HomeTeam"]],
        left_on=["MatchIndex", "Team"],
        right_on=["MatchIndex", "HomeTeam"],
        how="inner",
    )[["MatchIndex", "RestDays"]].rename(columns={"RestDays": "RestDays_H"})
    away_rest = long_df.merge(
        out[["MatchIndex", "AwayTeam"]],
        left_on=["MatchIndex", "Team"],
        right_on=["MatchIndex", "AwayTeam"],
        how="inner",
    )[["MatchIndex", "RestDays"]].rename(columns={"RestDays": "RestDays_A"})

    out = out.merge(home_rest, on="MatchIndex", how="left").merge(away_rest, on="MatchIndex", how="left")
    out["RestDays_diff"] = out["RestDays_H"] - out["RestDays_A"]
    return out


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    p = implied_probs_from_odds(out)
    out["mkt_pH"] = p["pH"]
    out["mkt_pD"] = p["pD"]
    out["mkt_pA"] = p["pA"]

    src = choose_odds_source(out)
    overround = np.full(len(out), np.nan, dtype=float)

    def _calc_overround(mask: np.ndarray, cols: object) -> None:
        inv_h = 1.0 / out.loc[mask, cols.home].to_numpy(dtype=float)
        inv_d = 1.0 / out.loc[mask, cols.draw].to_numpy(dtype=float)
        inv_a = 1.0 / out.loc[mask, cols.away].to_numpy(dtype=float)
        overround[mask] = inv_h + inv_d + inv_a - 1.0

    b365_mask = src.eq("B365").to_numpy()
    ps_mask = src.eq("PS").to_numpy()
    if b365_mask.any():
        _calc_overround(b365_mask, ODDS_B365)
    if ps_mask.any():
        _calc_overround(ps_mask, ODDS_PS)

    out["mkt_overround"] = overround

    probs = out[["mkt_pH", "mkt_pD", "mkt_pA"]].to_numpy(dtype=float)
    eps = 1e-12
    probs_safe = np.clip(probs, eps, 1.0)
    probs_safe = probs_safe / probs_safe.sum(axis=1, keepdims=True)
    out["mkt_entropy"] = -np.sum(probs_safe * np.log(probs_safe), axis=1)
    sorted_probs = np.sort(probs_safe, axis=1)
    out["mkt_fav_gap"] = sorted_probs[:, -1] - sorted_probs[:, -2]
    return out


def add_rolling_shots_features(df: pd.DataFrame, *, window_size: int) -> pd.DataFrame:
    out = df.copy()
    home = out[["Date", "MatchIndex", "HomeTeam", "Home_Shots", "Away_Shots"]].copy()
    home.columns = ["Date", "MatchIndex", "Team", "Shots_S", "Shots_C"]
    away = out[["Date", "MatchIndex", "AwayTeam", "Away_Shots", "Home_Shots"]].copy()
    away.columns = ["Date", "MatchIndex", "Team", "Shots_S", "Shots_C"]

    long_df = pd.concat([home, away], ignore_index=True).sort_values(["Team", "Date", "MatchIndex"]).reset_index(drop=True)

    def _rolling(group: pd.DataFrame) -> pd.DataFrame:
        rolled = group[["Shots_S", "Shots_C"]].rolling(window=window_size, min_periods=1).mean().shift(1)
        rolled.columns = [f"Forma_{c}_{window_size}" for c in ["Shots_S", "Shots_C"]]
        return rolled

    rolled = long_df.groupby("Team", group_keys=False).apply(_rolling)
    long_df = pd.concat([long_df, rolled], axis=1)

    home_form = long_df.merge(
        out[["MatchIndex", "HomeTeam"]],
        left_on=["MatchIndex", "Team"],
        right_on=["MatchIndex", "HomeTeam"],
        how="inner",
    )[["MatchIndex", f"Forma_Shots_S_{window_size}", f"Forma_Shots_C_{window_size}"]]
    home_form = home_form.rename(
        columns={
            f"Forma_Shots_S_{window_size}": f"Forma_Shots_S_{window_size}_H",
            f"Forma_Shots_C_{window_size}": f"Forma_Shots_C_{window_size}_H",
        }
    )

    away_form = long_df.merge(
        out[["MatchIndex", "AwayTeam"]],
        left_on=["MatchIndex", "Team"],
        right_on=["MatchIndex", "AwayTeam"],
        how="inner",
    )[["MatchIndex", f"Forma_Shots_S_{window_size}", f"Forma_Shots_C_{window_size}"]]
    away_form = away_form.rename(
        columns={
            f"Forma_Shots_S_{window_size}": f"Forma_Shots_S_{window_size}_A",
            f"Forma_Shots_C_{window_size}": f"Forma_Shots_C_{window_size}_A",
        }
    )

    out = out.merge(home_form, on="MatchIndex", how="left").merge(away_form, on="MatchIndex", how="left")
    return out


def add_elo_features(df: pd.DataFrame, *, k_factor: float = 20.0, home_advantage: float = 80.0) -> pd.DataFrame:
    out = df.copy().sort_values(["Date", "MatchIndex"]).reset_index(drop=True)
    ratings: dict[str, float] = {}

    elo_h: list[float] = []
    elo_a: list[float] = []

    for _, row in out.iterrows():
        h = str(row["HomeTeam"])
        a = str(row["AwayTeam"])
        rh = ratings.get(h, 1500.0)
        ra = ratings.get(a, 1500.0)
        elo_h.append(rh)
        elo_a.append(ra)

        expected_h = 1.0 / (1.0 + 10.0 ** (-(rh + home_advantage - ra) / 400.0))
        if row["FTHG"] > row["FTAG"]:
            actual_h = 1.0
        elif row["FTHG"] < row["FTAG"]:
            actual_h = 0.0
        else:
            actual_h = 0.5

        new_rh = rh + k_factor * (actual_h - expected_h)
        new_ra = ra + k_factor * ((1.0 - actual_h) - (1.0 - expected_h))
        ratings[h] = new_rh
        ratings[a] = new_ra

    out["elo_home_pre"] = np.array(elo_h, dtype=float)
    out["elo_away_pre"] = np.array(elo_a, dtype=float)
    out["elo_diff_pre"] = out["elo_home_pre"] - out["elo_away_pre"]
    return out


def add_fold_dependent_features(df: pd.DataFrame, *, league_avg_home_xg: float, league_avg_away_xg: float) -> pd.DataFrame:
    out = df.copy()
    eps = 1e-6
    for w in ROLLING_WINDOWS:
        home_xg_s = out[f"Forma_xG_S_{w}_H"].to_numpy(dtype=float)
        away_xg_s = out[f"Forma_xG_S_{w}_A"].to_numpy(dtype=float)
        home_xg_c = out[f"Forma_xG_C_{w}_H"].to_numpy(dtype=float)
        away_xg_c = out[f"Forma_xG_C_{w}_A"].to_numpy(dtype=float)

        home_as = np.nan_to_num(home_xg_s / league_avg_home_xg, nan=1.0, posinf=1.0, neginf=1.0)
        away_as = np.nan_to_num(away_xg_s / league_avg_away_xg, nan=1.0, posinf=1.0, neginf=1.0)
        home_ds = np.nan_to_num(home_xg_c / league_avg_away_xg, nan=1.0, posinf=1.0, neginf=1.0)
        away_ds = np.nan_to_num(away_xg_c / league_avg_home_xg, nan=1.0, posinf=1.0, neginf=1.0)

        out[f"lam_home_xgform_{w}"] = home_as * away_ds * league_avg_home_xg
        out[f"lam_away_xgform_{w}"] = away_as * home_ds * league_avg_away_xg
        out[f"lam_diff_xgform_{w}"] = out[f"lam_home_xgform_{w}"] - out[f"lam_away_xgform_{w}"]

        out[f"xg_per_shot_H_{w}"] = out[f"Forma_xG_S_{w}_H"] / np.clip(out[f"Forma_Shots_S_{w}_H"], eps, None)
        out[f"xg_per_shot_A_{w}"] = out[f"Forma_xG_S_{w}_A"] / np.clip(out[f"Forma_Shots_S_{w}_A"], eps, None)
        out[f"xg_per_shot_diff_{w}"] = out[f"xg_per_shot_H_{w}"] - out[f"xg_per_shot_A_{w}"]

        out[f"shot_volume_diff_{w}"] = out[f"Forma_Shots_S_{w}_H"] - out[f"Forma_Shots_S_{w}_A"]
        out[f"shot_concede_diff_{w}"] = out[f"Forma_Shots_C_{w}_H"] - out[f"Forma_Shots_C_{w}_A"]

        out[f"oa_xg_home_{w}"] = out[f"Forma_xG_S_{w}_H"] / np.clip(out[f"Forma_xG_C_{w}_A"], eps, None)
        out[f"oa_xg_away_{w}"] = out[f"Forma_xG_S_{w}_A"] / np.clip(out[f"Forma_xG_C_{w}_H"], eps, None)
        out[f"oa_xg_ratio_diff_{w}"] = out[f"oa_xg_home_{w}"] - out[f"oa_xg_away_{w}"]

    # trend features (short minus long)
    out["trend_xg_home_3_10"] = out["Forma_xG_S_3_H"] - out["Forma_xG_S_10_H"]
    out["trend_xg_away_3_10"] = out["Forma_xG_S_3_A"] - out["Forma_xG_S_10_A"]
    out["trend_xg_diff_3_10"] = out["trend_xg_home_3_10"] - out["trend_xg_away_3_10"]
    out["trend_goals_home_3_10"] = out["Forma_Goals_S_3_H"] - out["Forma_Goals_S_10_H"]
    out["trend_goals_away_3_10"] = out["Forma_Goals_S_3_A"] - out["Forma_Goals_S_10_A"]
    out["trend_goals_diff_3_10"] = out["trend_goals_home_3_10"] - out["trend_goals_away_3_10"]
    return out


def build_feature_packs() -> dict[str, list[str]]:
    base: list[str] = []
    for w in (3, 5, 10):
        base += [
            f"Forma_Goals_S_{w}_H",
            f"Forma_Goals_C_{w}_H",
            f"Forma_xG_S_{w}_H",
            f"Forma_xG_C_{w}_H",
            f"Forma_Goals_S_{w}_A",
            f"Forma_Goals_C_{w}_A",
            f"Forma_xG_S_{w}_A",
            f"Forma_xG_C_{w}_A",
            f"lam_home_xgform_{w}",
            f"lam_away_xgform_{w}",
            f"lam_diff_xgform_{w}",
        ]
    base += ["RestDays_H", "RestDays_A", "RestDays_diff"]

    elo = ["elo_home_pre", "elo_away_pre", "elo_diff_pre"]

    shot_quality: list[str] = []
    for w in (3, 5, 10):
        shot_quality += [
            f"Forma_Shots_S_{w}_H",
            f"Forma_Shots_S_{w}_A",
            f"Forma_Shots_C_{w}_H",
            f"Forma_Shots_C_{w}_A",
            f"xg_per_shot_H_{w}",
            f"xg_per_shot_A_{w}",
            f"xg_per_shot_diff_{w}",
            f"shot_volume_diff_{w}",
            f"shot_concede_diff_{w}",
        ]

    opponent_adjusted = [f"oa_xg_home_{w}" for w in (3, 5, 10)] + [f"oa_xg_away_{w}" for w in (3, 5, 10)] + [
        f"oa_xg_ratio_diff_{w}" for w in (3, 5, 10)
    ]

    trend = [
        "trend_xg_home_3_10",
        "trend_xg_away_3_10",
        "trend_xg_diff_3_10",
        "trend_goals_home_3_10",
        "trend_goals_away_3_10",
        "trend_goals_diff_3_10",
    ]

    return {
        "fs_base": base,
        "fs_base_elo": base + elo,
        "fs_base_elo_shotq": base + elo + shot_quality,
        "fs_base_elo_shotq_oa": base + elo + shot_quality + opponent_adjusted,
        "fs_full": base + elo + shot_quality + opponent_adjusted + trend,
    }


def build_market_cols() -> list[str]:
    return ["mkt_pH", "mkt_pD", "mkt_pA", "mkt_overround", "mkt_entropy", "mkt_fav_gap"]


def run_fold(matches: pd.DataFrame, *, league_name: str, fold: dict[str, object]) -> list[dict[str, object]]:
    train_seasons = list(fold["train_seasons"])
    valid_season = str(fold["valid_season"])
    test_season = str(fold["test_season"])
    fold_id = int(fold["fold_id"])

    train = matches[matches["Season"].isin(train_seasons)].copy()
    valid = matches[matches["Season"] == valid_season].copy()
    test = matches[matches["Season"] == test_season].copy()
    train_valid = pd.concat([train, valid], ignore_index=True)

    league_avg_home_xg = float(train_valid["Home_xG"].mean())
    league_avg_away_xg = float(train_valid["Away_xG"].mean())

    y_map = {"H": 0, "D": 1, "A": 2}
    y_train = np.array([y_map[v] for v in train["y"].to_numpy()], dtype=int)
    y_test = test["y"].to_numpy()

    df_features = add_fold_dependent_features(matches, league_avg_home_xg=league_avg_home_xg, league_avg_away_xg=league_avg_away_xg)
    packs = build_feature_packs()
    market_cols = build_market_cols()

    fold_info = {
        "league": league_name,
        "fold_id": fold_id,
        "train": ",".join(train_seasons),
        "valid": valid_season,
        "test": test_season,
    }

    rows: list[dict[str, object]] = []

    # bookmaker on full test when available
    bm = implied_probs_from_odds(test)
    bm_mask = bm.notna().all(axis=1).to_numpy()
    if bm_mask.any():
        rows.append(
            {
                **fold_info,
                "track": "B",
                "feature_pack": "reference",
                "model": "bookmaker_implied",
                **compute_metrics(y_test[bm_mask], bm.loc[bm_mask, ["pH", "pD", "pA"]].to_numpy(dtype=float)),
                "n": int(bm_mask.sum()),
            }
        )

    for pack_name, sport_cols in packs.items():
        # Track A
        X_train_a = df_features.loc[train.index, sport_cols].to_numpy(dtype=float)
        X_test_a = df_features.loc[test.index, sport_cols].to_numpy(dtype=float)

        logreg_a = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=0.1, solver="lbfgs", max_iter=2000, random_state=42)),
            ]
        )
        logreg_a.fit(X_train_a, y_train)

        et_a = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", ExtraTreesClassifier(n_estimators=500, max_depth=8, min_samples_leaf=3, random_state=42, n_jobs=-1)),
            ]
        )
        et_a.fit(X_train_a, y_train)

        rows.append({**fold_info, "track": "A", "feature_pack": pack_name, "model": "ml_logreg", **compute_metrics(y_test, logreg_a.predict_proba(X_test_a)), "n": len(test)})
        rows.append({**fold_info, "track": "A", "feature_pack": pack_name, "model": "ml_extra_trees", **compute_metrics(y_test, et_a.predict_proba(X_test_a)), "n": len(test)})

        # Track B (sport + market) only for rows with market data
        train_m = df_features.loc[train.index, market_cols].notna().all(axis=1).to_numpy()
        test_m = df_features.loc[test.index, market_cols].notna().all(axis=1).to_numpy()
        if train_m.any() and test_m.any():
            all_cols_b = sport_cols + market_cols
            X_train_b = df_features.loc[train.index[train_m], all_cols_b].to_numpy(dtype=float)
            y_train_b = y_train[train_m]
            X_test_b = df_features.loc[test.index[test_m], all_cols_b].to_numpy(dtype=float)
            y_test_b = y_test[test_m]

            logreg_b = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(C=0.1, solver="lbfgs", max_iter=2000, random_state=42)),
                ]
            )
            logreg_b.fit(X_train_b, y_train_b)

            et_b = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clf", ExtraTreesClassifier(n_estimators=500, max_depth=8, min_samples_leaf=3, random_state=42, n_jobs=-1)),
                ]
            )
            et_b.fit(X_train_b, y_train_b)

            rows.append({**fold_info, "track": "B", "feature_pack": pack_name, "model": "ml_logreg", **compute_metrics(y_test_b, logreg_b.predict_proba(X_test_b)), "n": len(y_test_b)})
            rows.append({**fold_info, "track": "B", "feature_pack": pack_name, "model": "ml_extra_trees", **compute_metrics(y_test_b, et_b.predict_proba(X_test_b)), "n": len(y_test_b)})

    return rows


def summarize(df: pd.DataFrame, *, track: str) -> pd.DataFrame:
    d = df[df["track"] == track].copy()
    out = (
        d.groupby(["feature_pack", "model"], as_index=False)
        .agg(
            folds=("log_loss", "count"),
            log_loss_mean=("log_loss", "mean"),
            log_loss_std=("log_loss", "std"),
            brier_mean=("brier", "mean"),
            brier_std=("brier", "std"),
        )
        .sort_values(["feature_pack", "log_loss_mean"])
        .reset_index(drop=True)
    )
    out["log_loss_std"] = out["log_loss_std"].fillna(0.0)
    out["brier_std"] = out["brier_std"].fillna(0.0)
    return out


def summarize_delta_vs_base(df_summary: pd.DataFrame) -> pd.DataFrame:
    model_mask = df_summary["model"].isin(["ml_logreg", "ml_extra_trees"])
    d = df_summary[model_mask].copy()
    base = (
        d[d["feature_pack"] == "fs_base"][["model", "log_loss_mean"]]
        .rename(columns={"log_loss_mean": "base_log_loss_mean"})
        .reset_index(drop=True)
    )
    out = d.merge(base, on="model", how="left")
    out["delta_vs_base"] = out["log_loss_mean"] - out["base_log_loss_mean"]
    out["improvement_vs_base_pct"] = -100.0 * out["delta_vs_base"] / out["base_log_loss_mean"]
    return out.sort_values(["model", "log_loss_mean"]).reset_index(drop=True)


def main() -> None:
    base = Path("datasets")
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    folds = make_time_folds(min_train_seasons=3)
    all_rows: list[dict[str, object]] = []

    for league_name, league_prefix in LEAGUES.items():
        print(f"Loading {league_name} ({league_prefix})...")
        matches = load_league_matches_with_shots(base, league_prefix)
        matches["y"] = outcome_from_score(matches)
        for w in ROLLING_WINDOWS:
            matches = add_rolling_form_features(matches, window_size=w)
            matches = add_rolling_shots_features(matches, window_size=w)
        matches = add_rest_days_features(matches)
        matches = add_market_features(matches)
        matches = add_elo_features(matches)

        print(f"Running {league_name}: {len(folds)} rolling-origin folds")
        for fold in folds:
            print(
                f"  Fold {fold['fold_id']}: train={fold['train_seasons'][0]}..{fold['train_seasons'][-1]}, "
                f"valid={fold['valid_season']}, test={fold['test_season']}"
            )
            all_rows.extend(run_fold(matches, league_name=league_name, fold=fold))

    df_folds = pd.DataFrame(all_rows).sort_values(["track", "league", "fold_id", "feature_pack", "log_loss"]).reset_index(drop=True)
    summary_a = summarize(df_folds, track="A")
    summary_b = summarize(df_folds, track="B")
    delta_a = summarize_delta_vs_base(summary_a)
    delta_b = summarize_delta_vs_base(summary_b)

    out_folds = out_dir / "experiment_stageI_featurepacks_fold_metrics.csv"
    out_a = out_dir / "experiment_stageI_featurepacks_summary_trackA.csv"
    out_b = out_dir / "experiment_stageI_featurepacks_summary_trackB.csv"
    out_da = out_dir / "experiment_stageI_featurepacks_delta_trackA.csv"
    out_db = out_dir / "experiment_stageI_featurepacks_delta_trackB.csv"

    df_folds.to_csv(out_folds, index=False)
    summary_a.to_csv(out_a, index=False)
    summary_b.to_csv(out_b, index=False)
    delta_a.to_csv(out_da, index=False)
    delta_b.to_csv(out_db, index=False)

    print("\n=== Stage I summary Track A ===")
    print(summary_a.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print("\n=== Stage I summary Track B ===")
    print(summary_b.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print("\n=== Stage I delta vs base Track A ===")
    print(delta_a.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print("\n=== Stage I delta vs base Track B ===")
    print(delta_b.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    print(f"\nSaved fold metrics: {out_folds}")
    print(f"Saved Track A summary: {out_a}")
    print(f"Saved Track B summary: {out_b}")
    print(f"Saved Track A delta: {out_da}")
    print(f"Saved Track B delta: {out_db}")


if __name__ == "__main__":
    main()
