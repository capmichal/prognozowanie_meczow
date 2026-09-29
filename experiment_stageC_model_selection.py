from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

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


def build_feature_sets() -> dict[str, list[str]]:
    multiwindow: list[str] = []
    for w in (3, 5, 10):
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
    return {
        "fs_multiwindow": multiwindow,
        "fs_multiwindow_rest": multiwindow + ["RestDays_H", "RestDays_A", "RestDays_diff"],
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
) -> tuple[Pipeline, dict[str, Any]]:
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
        raise RuntimeError(f"Tuning failed for {model_name}")
    return best_pipe, best_params


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
    y_valid = np.array([y_map[v] for v in valid["y"].to_numpy()], dtype=int)
    y_test_labels = test["y"].to_numpy()

    df_features = add_fold_dependent_features(matches, league_avg_home_xg=league_avg_home_xg, league_avg_away_xg=league_avg_away_xg)
    feature_sets = build_feature_sets()

    bm_probs = implied_probs_from_odds(test)
    bm_mask = bm_probs.notna().all(axis=1).to_numpy()

    rows: list[dict[str, object]] = []
    info = {
        "league": league_name,
        "fold_id": fold_id,
        "train": ",".join(train_seasons),
        "valid": valid_season,
        "test": test_season,
    }

    if bm_mask.any():
        y_bm = y_test_labels[bm_mask]
        p_bm = bm_probs.loc[bm_mask, ["pH", "pD", "pA"]].to_numpy(dtype=float)
        rows.append({**info, "feature_set": "reference", "model": "bookmaker_implied", "selected_params": "", **compute_metrics(y_bm, p_bm), "n": int(bm_mask.sum())})

    for fs_name, fs_cols in feature_sets.items():
        X_train = df_features.loc[train.index, fs_cols].to_numpy(dtype=float)
        X_valid = df_features.loc[valid.index, fs_cols].to_numpy(dtype=float)
        X_test = df_features.loc[test.index, fs_cols].to_numpy(dtype=float)

        candidates: list[tuple[str, Pipeline, dict[str, Any]]] = []

        logreg_pipe, logreg_params = _fit_tuned(
            model_name="ml_logreg",
            build_pipeline=lambda p: Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(C=float(p["C"]), solver="lbfgs", max_iter=2000, random_state=42)),
                ]
            ),
            param_grid=[{"C": 0.1}, {"C": 1.0}, {"C": 10.0}],
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid_labels=valid["y"].to_numpy(),
        )
        candidates.append(("ml_logreg", logreg_pipe, logreg_params))

        et_pipe, et_params = _fit_tuned(
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
        candidates.append(("ml_extra_trees", et_pipe, et_params))

        xgb_pipe, xgb_params = _fit_tuned(
            model_name="ml_xgboost",
            build_pipeline=lambda p: Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        XGBClassifier(
                            objective="multi:softprob",
                            num_class=3,
                            n_estimators=int(p["n_estimators"]),
                            max_depth=int(p["max_depth"]),
                            learning_rate=float(p["learning_rate"]),
                            subsample=float(p["subsample"]),
                            colsample_bytree=float(p["colsample_bytree"]),
                            eval_metric="mlogloss",
                            random_state=42,
                            n_jobs=-1,
                            tree_method="hist",
                        ),
                    ),
                ]
            ),
            param_grid=[
                {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9},
                {"n_estimators": 500, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.9, "colsample_bytree": 0.9},
            ],
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid_labels=valid["y"].to_numpy(),
        )
        candidates.append(("ml_xgboost", xgb_pipe, xgb_params))

        cat_pipe, cat_params = _fit_tuned(
            model_name="ml_catboost",
            build_pipeline=lambda p: Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        CatBoostClassifier(
                            loss_function="MultiClass",
                            iterations=int(p["iterations"]),
                            depth=int(p["depth"]),
                            learning_rate=float(p["learning_rate"]),
                            l2_leaf_reg=float(p["l2_leaf_reg"]),
                            random_seed=42,
                            verbose=False,
                            allow_writing_files=False,
                        ),
                    ),
                ]
            ),
            param_grid=[
                {"iterations": 400, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 3.0},
                {"iterations": 600, "depth": 5, "learning_rate": 0.03, "l2_leaf_reg": 5.0},
            ],
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid_labels=valid["y"].to_numpy(),
        )
        candidates.append(("ml_catboost", cat_pipe, cat_params))

        for model_name, pipe, params in candidates:
            probs = pipe.predict_proba(X_test)
            rows.append(
                {
                    **info,
                    "feature_set": fs_name,
                    "model": model_name,
                    "selected_params": json.dumps(params, sort_keys=True),
                    **compute_metrics(y_test_labels, probs),
                    "n": len(test),
                }
            )

    return rows


def summarize_by_league(df_folds: pd.DataFrame) -> pd.DataFrame:
    out = (
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
    out["log_loss_std"] = out["log_loss_std"].fillna(0.0)
    out["brier_std"] = out["brier_std"].fillna(0.0)
    return out


def summarize_global(df_folds: pd.DataFrame) -> pd.DataFrame:
    out = (
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
    out["log_loss_std"] = out["log_loss_std"].fillna(0.0)
    out["brier_std"] = out["brier_std"].fillna(0.0)
    return out


def main() -> None:
    base = Path("datasets")
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    folds = make_time_folds(min_train_seasons=3)
    all_rows: list[dict[str, object]] = []

    for league_name, league_prefix in LEAGUES.items():
        print(f"Loading {league_name} ({league_prefix})...")
        matches = load_league_matches(base, league_prefix)
        matches["y"] = outcome_from_score(matches)
        for w in ROLLING_WINDOWS:
            matches = add_rolling_form_features(matches, window_size=w)
        matches = add_rest_days_features(matches)

        print(f"Running {league_name}: {len(folds)} rolling-origin folds")
        for fold in folds:
            print(
                f"  Fold {fold['fold_id']}: train={fold['train_seasons'][0]}..{fold['train_seasons'][-1]}, "
                f"valid={fold['valid_season']}, test={fold['test_season']}"
            )
            all_rows.extend(run_fold(matches, league_name=league_name, fold=fold))

    df_folds = pd.DataFrame(all_rows).sort_values(["league", "fold_id", "feature_set", "log_loss"]).reset_index(drop=True)
    df_by_league = summarize_by_league(df_folds)
    df_global = summarize_global(df_folds)

    out_folds = out_dir / "experiment_stageC_model_selection_fold_metrics.csv"
    out_by_league = out_dir / "experiment_stageC_model_selection_summary_by_league.csv"
    out_global = out_dir / "experiment_stageC_model_selection_summary_global.csv"

    df_folds.to_csv(out_folds, index=False)
    df_by_league.to_csv(out_by_league, index=False)
    df_global.to_csv(out_global, index=False)

    print("\n=== Stage C global summary ===")
    print(df_global.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print(f"\nSaved fold metrics: {out_folds}")
    print(f"Saved summary by league: {out_by_league}")
    print(f"Saved global summary: {out_global}")


if __name__ == "__main__":
    main()
