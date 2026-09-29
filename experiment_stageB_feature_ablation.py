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
    add_rolling_form_features,
    brier_score_multiclass,
    build_matches_with_xg,
    clip_and_renormalize,
    implied_probs_from_odds,
    load_results_season,
    load_shots_season,
    log_loss_multiclass,
    outcome_from_score,
    poisson_1x2_probs,
)


SEASONS: tuple[str, ...] = (
    "14-15",
    "15-16",
    "16-17",
    "17-18",
    "18-19",
    "19-20",
    "20-21",
)

LEAGUES: dict[str, str] = {
    "EPL": "premier_league",
    "LaLiga": "la_liga",
    "Bundesliga": "bundesliga",
}

ROLLING_WINDOWS: tuple[int, ...] = (3, 5, 10)


def _season_paths(base: Path, league_prefix: str, season: str) -> tuple[Path, Path]:
    results = base / f"{league_prefix}_{season}.csv"
    shots = base / f"shots_{league_prefix}_{season}.csv"
    if not results.exists():
        raise FileNotFoundError(f"Missing results file: {results}")
    if not shots.exists():
        raise FileNotFoundError(f"Missing shots file: {shots}")
    return results, shots


def load_league_matches(base: Path, league_prefix: str, seasons: tuple[str, ...]) -> pd.DataFrame:
    results_all: list[pd.DataFrame] = []
    shots_all: list[pd.DataFrame] = []

    for season in seasons:
        results_path, shots_path = _season_paths(base, league_prefix, season)
        results_all.append(load_results_season(results_path, season=season))
        shots_all.append(load_shots_season(shots_path, season=season))

    results = pd.concat(results_all, ignore_index=True)
    shots = pd.concat(shots_all, ignore_index=True)
    return build_matches_with_xg(results, shots)


def make_time_folds(seasons: tuple[str, ...], *, min_train_seasons: int = 3) -> list[dict[str, object]]:
    folds: list[dict[str, object]] = []
    for test_idx in range(min_train_seasons + 1, len(seasons)):
        valid_idx = test_idx - 1
        train = list(seasons[:valid_idx])
        valid = seasons[valid_idx]
        test = seasons[test_idx]
        folds.append(
            {
                "fold_id": len(folds) + 1,
                "train_seasons": train,
                "valid_season": valid,
                "test_season": test,
            }
        )
    return folds


def compute_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    probs = clip_and_renormalize(probs)
    return {
        "log_loss": float(log_loss_multiclass(y_true, probs)),
        "brier": float(brier_score_multiclass(y_true, probs)),
    }


def build_xg_form_model_probs(
    df_all: pd.DataFrame,
    *,
    window_size: int,
    league_avg_home_xg: float,
    league_avg_away_xg: float,
    max_goals: int,
) -> np.ndarray:
    home_xg_s = df_all[f"Forma_xG_S_{window_size}_H"].to_numpy(dtype=float)
    away_xg_s = df_all[f"Forma_xG_S_{window_size}_A"].to_numpy(dtype=float)
    home_xg_c = df_all[f"Forma_xG_C_{window_size}_H"].to_numpy(dtype=float)
    away_xg_c = df_all[f"Forma_xG_C_{window_size}_A"].to_numpy(dtype=float)

    home_as = np.nan_to_num(home_xg_s / league_avg_home_xg, nan=1.0, posinf=1.0, neginf=1.0)
    away_as = np.nan_to_num(away_xg_s / league_avg_away_xg, nan=1.0, posinf=1.0, neginf=1.0)
    home_ds = np.nan_to_num(home_xg_c / league_avg_away_xg, nan=1.0, posinf=1.0, neginf=1.0)
    away_ds = np.nan_to_num(away_xg_c / league_avg_home_xg, nan=1.0, posinf=1.0, neginf=1.0)

    lam_home = home_as * away_ds * league_avg_home_xg
    lam_away = away_as * home_ds * league_avg_away_xg
    return poisson_1x2_probs(lam_home, lam_away, max_goals=max_goals)


def add_rest_days_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    home = out[["Date", "MatchIndex", "HomeTeam"]].rename(columns={"HomeTeam": "Team"})
    away = out[["Date", "MatchIndex", "AwayTeam"]].rename(columns={"AwayTeam": "Team"})
    long_df = pd.concat([home, away], ignore_index=True)
    long_df = long_df.sort_values(["Team", "Date", "MatchIndex"]).reset_index(drop=True)

    long_df["RestDays"] = (
        long_df.groupby("Team")["Date"].diff().dt.days.astype(float)
    )

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


def add_fold_dependent_features(
    df: pd.DataFrame,
    *,
    league_avg_home_xg: float,
    league_avg_away_xg: float,
) -> pd.DataFrame:
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

        out[f"edge_xg_home_{w}"] = out[f"Forma_xG_S_{w}_H"] - out[f"Forma_xG_C_{w}_A"]
        out[f"edge_xg_away_{w}"] = out[f"Forma_xG_S_{w}_A"] - out[f"Forma_xG_C_{w}_H"]
        out[f"edge_xg_diff_{w}"] = out[f"edge_xg_home_{w}"] - out[f"edge_xg_away_{w}"]

        out[f"edge_goals_home_{w}"] = out[f"Forma_Goals_S_{w}_H"] - out[f"Forma_Goals_C_{w}_A"]
        out[f"edge_goals_away_{w}"] = out[f"Forma_Goals_S_{w}_A"] - out[f"Forma_Goals_C_{w}_H"]
        out[f"edge_goals_diff_{w}"] = out[f"edge_goals_home_{w}"] - out[f"edge_goals_away_{w}"]

    return out


def build_feature_sets() -> dict[str, list[str]]:
    base_w5 = [
        "Forma_Goals_S_5_H",
        "Forma_Goals_C_5_H",
        "Forma_xG_S_5_H",
        "Forma_xG_C_5_H",
        "Forma_Goals_S_5_A",
        "Forma_Goals_C_5_A",
        "Forma_xG_S_5_A",
        "Forma_xG_C_5_A",
        "lam_home_xgform_5",
        "lam_away_xgform_5",
        "lam_diff_xgform_5",
    ]

    multiwindow = base_w5.copy()
    for w in (3, 10):
        multiwindow += [
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

    multiwindow_rest = multiwindow + ["RestDays_H", "RestDays_A", "RestDays_diff"]

    interactions = multiwindow_rest.copy()
    for w in ROLLING_WINDOWS:
        interactions += [
            f"edge_xg_home_{w}",
            f"edge_xg_away_{w}",
            f"edge_xg_diff_{w}",
            f"edge_goals_home_{w}",
            f"edge_goals_away_{w}",
            f"edge_goals_diff_{w}",
        ]

    return {
        "fs_base_w5": base_w5,
        "fs_multiwindow": multiwindow,
        "fs_multiwindow_rest": multiwindow_rest,
        "fs_interactions_full": interactions,
    }


def _fit_tuned(
    *,
    model_name: str,
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
        p_valid = pipe.predict_proba(X_valid)
        ll = log_loss_multiclass(y_valid_labels, p_valid)
        if ll < best_ll:
            best_ll = ll
            best_pipe = pipe
            best_params = params

    if best_pipe is None or best_params is None:
        raise RuntimeError(f"Failed to fit tuned model: {model_name}")

    return best_pipe, best_params, float(best_ll)


def run_fold(
    matches: pd.DataFrame,
    *,
    league_name: str,
    fold_id: int,
    train_seasons: list[str],
    valid_season: str,
    test_season: str,
    max_goals: int,
) -> list[dict[str, object]]:
    train = matches[matches["Season"].isin(train_seasons)].copy()
    valid = matches[matches["Season"] == valid_season].copy()
    test = matches[matches["Season"] == test_season].copy()

    if train.empty or valid.empty or test.empty:
        raise ValueError(f"{league_name} fold {fold_id} has an empty split.")

    train_valid = pd.concat([train, valid], ignore_index=True)
    league_avg_home_xg = float(train_valid["Home_xG"].mean())
    league_avg_away_xg = float(train_valid["Away_xG"].mean())

    y_test = test["y"].to_numpy()

    train_y = train["y"].to_numpy()
    freq = {
        "H": float(np.mean(train_y == "H")),
        "D": float(np.mean(train_y == "D")),
        "A": float(np.mean(train_y == "A")),
    }
    p_freq = np.tile([freq["H"], freq["D"], freq["A"]], (len(test), 1))

    bm_probs = implied_probs_from_odds(test)
    bm_mask = bm_probs.notna().all(axis=1).to_numpy()

    xg_probs_all = build_xg_form_model_probs(
        matches,
        window_size=5,
        league_avg_home_xg=league_avg_home_xg,
        league_avg_away_xg=league_avg_away_xg,
        max_goals=max_goals,
    )
    xg_probs_test = xg_probs_all[test.index.to_numpy(), :]

    df_features = add_fold_dependent_features(
        matches,
        league_avg_home_xg=league_avg_home_xg,
        league_avg_away_xg=league_avg_away_xg,
    )

    y_map = {"H": 0, "D": 1, "A": 2}
    y_train = np.array([y_map[v] for v in train["y"].to_numpy()], dtype=int)
    y_valid = np.array([y_map[v] for v in valid["y"].to_numpy()], dtype=int)

    feature_sets = build_feature_sets()

    fold_info = {
        "league": league_name,
        "fold_id": fold_id,
        "train": ",".join(train_seasons),
        "valid": valid_season,
        "test": test_season,
    }

    rows: list[dict[str, object]] = []

    def add_row(feature_set: str, model: str, probs: np.ndarray, params: dict[str, Any] | None = None, n: int | None = None) -> None:
        rows.append(
            {
                **fold_info,
                "feature_set": feature_set,
                "model": model,
                "selected_params": "" if params is None else json.dumps(params, sort_keys=True),
                **compute_metrics(y_test, probs),
                "n": len(test) if n is None else int(n),
            }
        )

    add_row("reference", "freq_baseline", p_freq)
    add_row("reference", "xg_form_poisson", xg_probs_test)
    if bm_mask.any():
        y_bm = y_test[bm_mask]
        p_bm = bm_probs.loc[bm_mask, ["pH", "pD", "pA"]].to_numpy(dtype=float)
        rows.append(
            {
                **fold_info,
                "feature_set": "reference",
                "model": "bookmaker_implied",
                "selected_params": "",
                **compute_metrics(y_bm, p_bm),
                "n": int(bm_mask.sum()),
            }
        )

    for feature_set, feature_cols in feature_sets.items():
        X_train = df_features.loc[train.index, feature_cols].to_numpy(dtype=float)
        X_valid = df_features.loc[valid.index, feature_cols].to_numpy(dtype=float)
        X_test = df_features.loc[test.index, feature_cols].to_numpy(dtype=float)

        logreg_pipe, logreg_params, _ = _fit_tuned(
            model_name="ml_logreg",
            build_pipeline=lambda p: Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            C=float(p["C"]),
                            solver="lbfgs",
                            max_iter=2000,
                            random_state=42,
                        ),
                    ),
                ]
            ),
            param_grid=[{"C": 0.1}, {"C": 1.0}, {"C": 10.0}],
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid_labels=valid["y"].to_numpy(),
        )
        add_row(feature_set, "ml_logreg", logreg_pipe.predict_proba(X_test), logreg_params)

        et_pipe, et_params, _ = _fit_tuned(
            model_name="ml_extra_trees",
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
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid_labels=valid["y"].to_numpy(),
        )
        add_row(feature_set, "ml_extra_trees", et_pipe.predict_proba(X_test), et_params)

    return rows


def summarize_by_league(df_folds: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df_folds.groupby(["league", "feature_set", "model"], as_index=False)
        .agg(
            folds=("log_loss", "count"),
            log_loss_mean=("log_loss", "mean"),
            log_loss_std=("log_loss", "std"),
            brier_mean=("brier", "mean"),
            brier_std=("brier", "std"),
        )
        .sort_values(["league", "feature_set", "log_loss_mean", "brier_mean"])
        .reset_index(drop=True)
    )
    summary["log_loss_std"] = summary["log_loss_std"].fillna(0.0)
    summary["brier_std"] = summary["brier_std"].fillna(0.0)
    return summary


def summarize_global(df_folds: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df_folds.groupby(["feature_set", "model"], as_index=False)
        .agg(
            leagues=("league", "nunique"),
            folds=("log_loss", "count"),
            log_loss_mean=("log_loss", "mean"),
            log_loss_std=("log_loss", "std"),
            brier_mean=("brier", "mean"),
            brier_std=("brier", "std"),
        )
        .sort_values(["feature_set", "log_loss_mean", "brier_mean"])
        .reset_index(drop=True)
    )
    summary["log_loss_std"] = summary["log_loss_std"].fillna(0.0)
    summary["brier_std"] = summary["brier_std"].fillna(0.0)
    return summary


def summarize_ablation(df_folds: pd.DataFrame) -> pd.DataFrame:
    ml_only = df_folds[df_folds["model"].isin(["ml_logreg", "ml_extra_trees"])].copy()
    base = (
        ml_only[ml_only["feature_set"] == "fs_base_w5"]
        .groupby("model", as_index=False)["log_loss"]
        .mean()
        .rename(columns={"log_loss": "base_log_loss_mean"})
    )
    summary = (
        ml_only.groupby(["feature_set", "model"], as_index=False)
        .agg(log_loss_mean=("log_loss", "mean"), brier_mean=("brier", "mean"))
        .merge(base, on="model", how="left")
    )
    summary["delta_vs_base"] = summary["log_loss_mean"] - summary["base_log_loss_mean"]
    summary["improvement_vs_base_pct"] = -100.0 * summary["delta_vs_base"] / summary["base_log_loss_mean"]
    return summary.sort_values(["model", "log_loss_mean"]).reset_index(drop=True)


def main() -> None:
    base = Path("datasets")
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    folds = make_time_folds(SEASONS, min_train_seasons=3)
    max_goals = 10

    all_rows: list[dict[str, object]] = []

    for league_name, league_prefix in LEAGUES.items():
        print(f"Loading {league_name} ({league_prefix})...")
        matches = load_league_matches(base, league_prefix, SEASONS)
        matches["y"] = outcome_from_score(matches)

        for w in ROLLING_WINDOWS:
            matches = add_rolling_form_features(matches, window_size=w)
        matches = add_rest_days_features(matches)

        print(f"Running {league_name}: {len(folds)} rolling-origin folds")
        for fold in folds:
            fold_id = int(fold["fold_id"])
            train_seasons = list(fold["train_seasons"])
            valid_season = str(fold["valid_season"])
            test_season = str(fold["test_season"])
            print(
                f"  Fold {fold_id}: train={train_seasons[0]}..{train_seasons[-1]}, "
                f"valid={valid_season}, test={test_season}"
            )
            all_rows.extend(
                run_fold(
                    matches,
                    league_name=league_name,
                    fold_id=fold_id,
                    train_seasons=train_seasons,
                    valid_season=valid_season,
                    test_season=test_season,
                    max_goals=max_goals,
                )
            )

    df_folds = pd.DataFrame(all_rows).sort_values(["league", "fold_id", "feature_set", "log_loss"]).reset_index(drop=True)
    df_summary_by_league = summarize_by_league(df_folds)
    df_summary_global = summarize_global(df_folds)
    df_ablation = summarize_ablation(df_folds)

    out_folds = out_dir / "experiment_stageB_feature_ablation_fold_metrics.csv"
    out_summary_league = out_dir / "experiment_stageB_feature_ablation_summary_by_league.csv"
    out_summary_global = out_dir / "experiment_stageB_feature_ablation_summary_global.csv"
    out_ablation = out_dir / "experiment_stageB_feature_ablation_delta_vs_base.csv"

    df_folds.to_csv(out_folds, index=False)
    df_summary_by_league.to_csv(out_summary_league, index=False)
    df_summary_global.to_csv(out_summary_global, index=False)
    df_ablation.to_csv(out_ablation, index=False)

    print("\n=== Stage B global summary ===")
    print(df_summary_global.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    print("\n=== Stage B ablation delta vs fs_base_w5 ===")
    print(df_ablation.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    print(f"\nSaved fold metrics: {out_folds}")
    print(f"Saved summary by league: {out_summary_league}")
    print(f"Saved global summary: {out_summary_global}")
    print(f"Saved ablation delta: {out_ablation}")


if __name__ == "__main__":
    main()
