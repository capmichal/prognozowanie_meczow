from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

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


def load_league_matches(base: Path, league_prefix: str) -> pd.DataFrame:
    results_all: list[pd.DataFrame] = []
    shots_all: list[pd.DataFrame] = []
    for season in SEASONS:
        rp, sp = _season_paths(base, league_prefix, season)
        results_all.append(load_results_season(rp, season=season))
        shots_all.append(load_shots_season(sp, season=season))
    return build_matches_with_xg(pd.concat(results_all, ignore_index=True), pd.concat(shots_all, ignore_index=True))


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


def add_fold_dependent_features(df: pd.DataFrame, *, league_avg_home_xg: float, league_avg_away_xg: float) -> pd.DataFrame:
    out = df.copy()
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
    return out


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    p = implied_probs_from_odds(out)
    out["mkt_pH"] = p["pH"]
    out["mkt_pD"] = p["pD"]
    out["mkt_pA"] = p["pA"]

    src = choose_odds_source(out)
    overround = np.full(len(out), np.nan, dtype=float)

    def _calc_overround(mask: np.ndarray, cols: Any) -> None:
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
    out["mkt_logit_HA"] = np.log(probs_safe[:, 0] / probs_safe[:, 2])
    out["mkt_logit_D_notD"] = np.log(probs_safe[:, 1] / np.clip(probs_safe[:, 0] + probs_safe[:, 2], eps, None))
    return out


def build_feature_sets() -> tuple[list[str], list[str]]:
    sport: list[str] = []
    for w in (3, 5, 10):
        sport += [
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
    sport += ["RestDays_H", "RestDays_A", "RestDays_diff"]

    market = [
        "mkt_pH",
        "mkt_pD",
        "mkt_pA",
        "mkt_overround",
        "mkt_entropy",
        "mkt_fav_gap",
        "mkt_logit_HA",
        "mkt_logit_D_notD",
    ]
    return sport, market


def _fit_tuned(
    *,
    build_pipeline: Callable[[dict[str, Any]], Pipeline],
    param_grid: list[dict[str, Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid_labels: np.ndarray,
) -> tuple[Pipeline, dict[str, Any], float]:
    best_ll = float("inf")
    best_pipe: Pipeline | None = None
    best_params: dict[str, Any] | None = None
    for params in param_grid:
        pipe = build_pipeline(params)
        pipe.fit(X_train, y_train)
        ll = log_loss_multiclass(y_valid_labels, pipe.predict_proba(X_valid))
        if ll < best_ll:
            best_ll = ll
            best_pipe = pipe
            best_params = params
    if best_pipe is None or best_params is None:
        raise RuntimeError("Tuning failed")
    return best_pipe, best_params, float(best_ll)


def blend_probs_geometric(p_market: np.ndarray, p_sport: np.ndarray, beta: float) -> np.ndarray:
    p_market = clip_and_renormalize(p_market)
    p_sport = clip_and_renormalize(p_sport)
    eps = 1e-12
    log_mix = (1.0 - beta) * np.log(np.clip(p_market, eps, 1.0)) + beta * np.log(np.clip(p_sport, eps, 1.0))
    out = np.exp(log_mix)
    return clip_and_renormalize(out)


def tune_beta(y_valid: np.ndarray, p_market_valid: np.ndarray, p_sport_valid: np.ndarray) -> tuple[float, float]:
    grid = np.round(np.arange(0.0, 1.5001, 0.05), 2)
    best_beta = 0.0
    best_ll = float("inf")
    for beta in grid:
        p = blend_probs_geometric(p_market_valid, p_sport_valid, beta=float(beta))
        ll = log_loss_multiclass(y_valid, p)
        if ll < best_ll:
            best_ll = ll
            best_beta = float(beta)
    return best_beta, float(best_ll)


def run_fold(matches: pd.DataFrame, *, league_name: str, fold: dict[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
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
    y_valid = np.array([y_map[v] for v in valid["y"].to_numpy()], dtype=int)
    y_valid_labels = valid["y"].to_numpy()
    y_test_labels = test["y"].to_numpy()

    df_features = add_fold_dependent_features(matches, league_avg_home_xg=league_avg_home_xg, league_avg_away_xg=league_avg_away_xg)
    sport_cols, market_cols = build_feature_sets()

    X_train_s = df_features.loc[train.index, sport_cols].to_numpy(dtype=float)
    X_valid_s = df_features.loc[valid.index, sport_cols].to_numpy(dtype=float)
    X_test_s = df_features.loc[test.index, sport_cols].to_numpy(dtype=float)

    # Track A: sport-only models (full test)
    logreg_s_pipe, logreg_s_params, _ = _fit_tuned(
        build_pipeline=lambda p: Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=float(p["C"]), solver="lbfgs", max_iter=2000, random_state=42)),
            ]
        ),
        param_grid=[{"C": 0.1}, {"C": 1.0}, {"C": 10.0}],
        X_train=X_train_s,
        y_train=y_train,
        X_valid=X_valid_s,
        y_valid_labels=y_valid_labels,
    )
    et_s_pipe, et_s_params, _ = _fit_tuned(
        build_pipeline=lambda p: Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    ExtraTreesClassifier(
                        n_estimators=int(p["n_estimators"]),
                        max_depth=p["max_depth"],
                        min_samples_leaf=int(p["min_samples_leaf"]),
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        param_grid=[
            {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 1},
            {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 3},
            {"n_estimators": 500, "max_depth": 8, "min_samples_leaf": 1},
            {"n_estimators": 500, "max_depth": 8, "min_samples_leaf": 3},
        ],
        X_train=X_train_s,
        y_train=y_train,
        X_valid=X_valid_s,
        y_valid_labels=y_valid_labels,
    )

    p_logreg_s_test = logreg_s_pipe.predict_proba(X_test_s)
    p_et_s_test = et_s_pipe.predict_proba(X_test_s)

    fold_info = {
        "league": league_name,
        "fold_id": fold_id,
        "train": ",".join(train_seasons),
        "valid": valid_season,
        "test": test_season,
    }

    stage_g_rows: list[dict[str, object]] = []
    stage_h_rows: list[dict[str, object]] = []

    # Track A results
    stage_g_rows.append({**fold_info, "track": "A", "model": "ml_logreg_sport", "selected_params": json.dumps(logreg_s_params, sort_keys=True), **compute_metrics(y_test_labels, p_logreg_s_test), "n": len(test)})
    stage_g_rows.append({**fold_info, "track": "A", "model": "ml_extra_trees_sport", "selected_params": json.dumps(et_s_params, sort_keys=True), **compute_metrics(y_test_labels, p_et_s_test), "n": len(test)})

    # Track B requires market features present
    mkt_valid = df_features.loc[valid.index, market_cols]
    mkt_test = df_features.loc[test.index, market_cols]
    valid_mask = mkt_valid.notna().all(axis=1).to_numpy()
    test_mask = mkt_test.notna().all(axis=1).to_numpy()

    if valid_mask.any() and test_mask.any():
        X_train_b = df_features.loc[train.index, sport_cols + market_cols].to_numpy(dtype=float)
        X_valid_b = df_features.loc[valid.index[valid_mask], sport_cols + market_cols].to_numpy(dtype=float)
        y_valid_b = valid.loc[valid.index[valid_mask], "y"].to_numpy()
        X_test_b = df_features.loc[test.index[test_mask], sport_cols + market_cols].to_numpy(dtype=float)
        y_test_b = test.loc[test.index[test_mask], "y"].to_numpy()

        # for train set in track B, keep only rows with market features available
        train_mask = df_features.loc[train.index, market_cols].notna().all(axis=1).to_numpy()
        X_train_b = X_train_b[train_mask]
        y_train_b = y_train[train_mask]

        logreg_b_pipe, logreg_b_params, _ = _fit_tuned(
            build_pipeline=lambda p: Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(C=float(p["C"]), solver="lbfgs", max_iter=2000, random_state=42)),
                ]
            ),
            param_grid=[{"C": 0.1}, {"C": 1.0}, {"C": 10.0}],
            X_train=X_train_b,
            y_train=y_train_b,
            X_valid=X_valid_b,
            y_valid_labels=y_valid_b,
        )

        et_b_pipe, et_b_params, _ = _fit_tuned(
            build_pipeline=lambda p: Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        ExtraTreesClassifier(
                            n_estimators=int(p["n_estimators"]),
                            max_depth=p["max_depth"],
                            min_samples_leaf=int(p["min_samples_leaf"]),
                            random_state=42,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            param_grid=[
                {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 1},
                {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 3},
                {"n_estimators": 500, "max_depth": 8, "min_samples_leaf": 1},
                {"n_estimators": 500, "max_depth": 8, "min_samples_leaf": 3},
            ],
            X_train=X_train_b,
            y_train=y_train_b,
            X_valid=X_valid_b,
            y_valid_labels=y_valid_b,
        )

        p_logreg_b_test = logreg_b_pipe.predict_proba(X_test_b)
        p_et_b_test = et_b_pipe.predict_proba(X_test_b)

        # bookmaker baseline on same subset
        p_mkt_all = implied_probs_from_odds(test)
        p_mkt_test = p_mkt_all.loc[test.index[test_mask], ["pH", "pD", "pA"]].to_numpy(dtype=float)

        stage_g_rows.append({**fold_info, "track": "B", "model": "bookmaker_implied", "selected_params": "", **compute_metrics(y_test_b, p_mkt_test), "n": len(y_test_b)})
        stage_g_rows.append({**fold_info, "track": "B", "model": "ml_logreg_sport_market", "selected_params": json.dumps(logreg_b_params, sort_keys=True), **compute_metrics(y_test_b, p_logreg_b_test), "n": len(y_test_b)})
        stage_g_rows.append({**fold_info, "track": "B", "model": "ml_extra_trees_sport_market", "selected_params": json.dumps(et_b_params, sort_keys=True), **compute_metrics(y_test_b, p_et_b_test), "n": len(y_test_b)})

        # Stage H: market + residual blend (geometric pooling), tuned on valid subset
        p_mkt_valid = implied_probs_from_odds(valid).loc[valid.index[valid_mask], ["pH", "pD", "pA"]].to_numpy(dtype=float)
        p_mkt_test = implied_probs_from_odds(test).loc[test.index[test_mask], ["pH", "pD", "pA"]].to_numpy(dtype=float)

        # sport-only probs on same subsets for blending
        p_logreg_s_valid = logreg_s_pipe.predict_proba(df_features.loc[valid.index[valid_mask], sport_cols].to_numpy(dtype=float))
        p_logreg_s_test = logreg_s_pipe.predict_proba(df_features.loc[test.index[test_mask], sport_cols].to_numpy(dtype=float))
        p_et_s_valid = et_s_pipe.predict_proba(df_features.loc[valid.index[valid_mask], sport_cols].to_numpy(dtype=float))
        p_et_s_test = et_s_pipe.predict_proba(df_features.loc[test.index[test_mask], sport_cols].to_numpy(dtype=float))

        beta_logreg, valid_ll_logreg = tune_beta(y_valid_b, p_mkt_valid, p_logreg_s_valid)
        beta_et, valid_ll_et = tune_beta(y_valid_b, p_mkt_valid, p_et_s_valid)

        p_blend_logreg_test = blend_probs_geometric(p_mkt_test, p_logreg_s_test, beta_logreg)
        p_blend_et_test = blend_probs_geometric(p_mkt_test, p_et_s_test, beta_et)

        stage_h_rows.append(
            {
                **fold_info,
                "model": "market_plus_residual_logreg",
                "beta": beta_logreg,
                "valid_log_loss": valid_ll_logreg,
                **compute_metrics(y_test_b, p_blend_logreg_test),
                "n": len(y_test_b),
            }
        )
        stage_h_rows.append(
            {
                **fold_info,
                "model": "market_plus_residual_extra_trees",
                "beta": beta_et,
                "valid_log_loss": valid_ll_et,
                **compute_metrics(y_test_b, p_blend_et_test),
                "n": len(y_test_b),
            }
        )
        stage_h_rows.append(
            {
                **fold_info,
                "model": "bookmaker_implied",
                "beta": 0.0,
                "valid_log_loss": np.nan,
                **compute_metrics(y_test_b, p_mkt_test),
                "n": len(y_test_b),
            }
        )

    return stage_g_rows, stage_h_rows


def summarize_stage_g(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_a = (
        df[df["track"] == "A"]
        .groupby("model", as_index=False)
        .agg(
            folds=("log_loss", "count"),
            log_loss_mean=("log_loss", "mean"),
            log_loss_std=("log_loss", "std"),
            brier_mean=("brier", "mean"),
            brier_std=("brier", "std"),
        )
        .sort_values("log_loss_mean")
        .reset_index(drop=True)
    )
    summary_b = (
        df[df["track"] == "B"]
        .groupby("model", as_index=False)
        .agg(
            folds=("log_loss", "count"),
            log_loss_mean=("log_loss", "mean"),
            log_loss_std=("log_loss", "std"),
            brier_mean=("brier", "mean"),
            brier_std=("brier", "std"),
        )
        .sort_values("log_loss_mean")
        .reset_index(drop=True)
    )
    for s in (summary_a, summary_b):
        s["log_loss_std"] = s["log_loss_std"].fillna(0.0)
        s["brier_std"] = s["brier_std"].fillna(0.0)
    return summary_a, summary_b


def summarize_stage_h(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("model", as_index=False)
        .agg(
            folds=("log_loss", "count"),
            log_loss_mean=("log_loss", "mean"),
            log_loss_std=("log_loss", "std"),
            brier_mean=("brier", "mean"),
            brier_std=("brier", "std"),
            beta_mean=("beta", "mean"),
            beta_std=("beta", "std"),
        )
        .sort_values("log_loss_mean")
        .reset_index(drop=True)
    )
    for c in ("log_loss_std", "brier_std", "beta_std"):
        out[c] = out[c].fillna(0.0)
    return out


def main() -> None:
    base = Path("datasets")
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    folds = make_time_folds(min_train_seasons=3)
    stage_g_all: list[dict[str, object]] = []
    stage_h_all: list[dict[str, object]] = []

    for league_name, league_prefix in LEAGUES.items():
        print(f"Loading {league_name} ({league_prefix})...")
        matches = load_league_matches(base, league_prefix)
        matches["y"] = outcome_from_score(matches)
        for w in ROLLING_WINDOWS:
            matches = add_rolling_form_features(matches, window_size=w)
        matches = add_rest_days_features(matches)
        matches = add_market_features(matches)

        print(f"Running {league_name}: {len(folds)} rolling-origin folds")
        for fold in folds:
            print(
                f"  Fold {fold['fold_id']}: train={fold['train_seasons'][0]}..{fold['train_seasons'][-1]}, "
                f"valid={fold['valid_season']}, test={fold['test_season']}"
            )
            g_rows, h_rows = run_fold(matches, league_name=league_name, fold=fold)
            stage_g_all.extend(g_rows)
            stage_h_all.extend(h_rows)

    df_g = pd.DataFrame(stage_g_all).sort_values(["track", "league", "fold_id", "log_loss"]).reset_index(drop=True)
    df_h = pd.DataFrame(stage_h_all).sort_values(["league", "fold_id", "log_loss"]).reset_index(drop=True)

    summary_a, summary_b = summarize_stage_g(df_g)
    summary_h = summarize_stage_h(df_h)

    out_g_folds = out_dir / "experiment_stageG_market_features_fold_metrics.csv"
    out_g_a = out_dir / "experiment_stageG_market_features_summary_A.csv"
    out_g_b = out_dir / "experiment_stageG_market_features_summary_B.csv"
    out_h_folds = out_dir / "experiment_stageH_market_residual_fold_metrics.csv"
    out_h_summary = out_dir / "experiment_stageH_market_residual_summary.csv"

    df_g.to_csv(out_g_folds, index=False)
    summary_a.to_csv(out_g_a, index=False)
    summary_b.to_csv(out_g_b, index=False)
    df_h.to_csv(out_h_folds, index=False)
    summary_h.to_csv(out_h_summary, index=False)

    print("\n=== Stage G Track A summary ===")
    print(summary_a.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print("\n=== Stage G Track B summary ===")
    print(summary_b.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print("\n=== Stage H summary ===")
    print(summary_h.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    print(f"\nSaved Stage G folds: {out_g_folds}")
    print(f"Saved Stage G Track A summary: {out_g_a}")
    print(f"Saved Stage G Track B summary: {out_g_b}")
    print(f"Saved Stage H folds: {out_h_folds}")
    print(f"Saved Stage H summary: {out_h_summary}")


if __name__ == "__main__":
    main()
