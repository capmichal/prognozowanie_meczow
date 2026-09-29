from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import PredefinedSplit
from sklearn.neural_network import MLPClassifier
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
    expected_col = f"Forma_xG_S_{window_size}_H"
    df = df_all if expected_col in df_all.columns else add_rolling_form_features(df_all, window_size=window_size)

    home_xg_s = df[f"Forma_xG_S_{window_size}_H"].to_numpy(dtype=float)
    away_xg_s = df[f"Forma_xG_S_{window_size}_A"].to_numpy(dtype=float)
    home_xg_c = df[f"Forma_xG_C_{window_size}_H"].to_numpy(dtype=float)
    away_xg_c = df[f"Forma_xG_C_{window_size}_A"].to_numpy(dtype=float)

    home_as = np.nan_to_num(home_xg_s / league_avg_home_xg, nan=1.0, posinf=1.0, neginf=1.0)
    away_as = np.nan_to_num(away_xg_s / league_avg_away_xg, nan=1.0, posinf=1.0, neginf=1.0)
    home_ds = np.nan_to_num(home_xg_c / league_avg_away_xg, nan=1.0, posinf=1.0, neginf=1.0)
    away_ds = np.nan_to_num(away_xg_c / league_avg_home_xg, nan=1.0, posinf=1.0, neginf=1.0)

    lam_home = home_as * away_ds * league_avg_home_xg
    lam_away = away_as * home_ds * league_avg_away_xg
    return poisson_1x2_probs(lam_home, lam_away, max_goals=max_goals)


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
    window_size: int,
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

    all_xg_probs = build_xg_form_model_probs(
        matches,
        window_size=window_size,
        league_avg_home_xg=league_avg_home_xg,
        league_avg_away_xg=league_avg_away_xg,
        max_goals=max_goals,
    )
    xg_probs_test = all_xg_probs[test.index.to_numpy(), :]

    feature_cols = [
        f"Forma_Goals_S_{window_size}_H",
        f"Forma_Goals_C_{window_size}_H",
        f"Forma_xG_S_{window_size}_H",
        f"Forma_xG_C_{window_size}_H",
        f"Forma_Goals_S_{window_size}_A",
        f"Forma_Goals_C_{window_size}_A",
        f"Forma_xG_S_{window_size}_A",
        f"Forma_xG_C_{window_size}_A",
    ]

    df_features = matches.copy()
    home_xg_s = df_features[f"Forma_xG_S_{window_size}_H"].to_numpy(dtype=float)
    away_xg_s = df_features[f"Forma_xG_S_{window_size}_A"].to_numpy(dtype=float)
    home_xg_c = df_features[f"Forma_xG_C_{window_size}_H"].to_numpy(dtype=float)
    away_xg_c = df_features[f"Forma_xG_C_{window_size}_A"].to_numpy(dtype=float)

    home_as = np.nan_to_num(home_xg_s / league_avg_home_xg, nan=1.0, posinf=1.0, neginf=1.0)
    away_as = np.nan_to_num(away_xg_s / league_avg_away_xg, nan=1.0, posinf=1.0, neginf=1.0)
    home_ds = np.nan_to_num(home_xg_c / league_avg_away_xg, nan=1.0, posinf=1.0, neginf=1.0)
    away_ds = np.nan_to_num(away_xg_c / league_avg_home_xg, nan=1.0, posinf=1.0, neginf=1.0)

    df_features["lam_home_xgform"] = home_as * away_ds * league_avg_home_xg
    df_features["lam_away_xgform"] = away_as * home_ds * league_avg_away_xg
    df_features["lam_diff_xgform"] = df_features["lam_home_xgform"] - df_features["lam_away_xgform"]

    feature_cols += ["lam_home_xgform", "lam_away_xgform", "lam_diff_xgform"]

    y_map = {"H": 0, "D": 1, "A": 2}
    X_train = df_features.loc[train.index, feature_cols].to_numpy(dtype=float)
    y_train = np.array([y_map[v] for v in train["y"].to_numpy()], dtype=int)
    X_valid = df_features.loc[valid.index, feature_cols].to_numpy(dtype=float)
    y_valid = np.array([y_map[v] for v in valid["y"].to_numpy()], dtype=int)
    X_test = df_features.loc[test.index, feature_cols].to_numpy(dtype=float)

    # LogReg tuning
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
    p_logreg_test = logreg_pipe.predict_proba(X_test)

    # LogReg isotonic calibration on validation fold
    logreg_for_cal = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=float(logreg_params["C"]),
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )
    X_cal = np.vstack([X_train, X_valid])
    y_cal = np.concatenate([y_train, y_valid])
    test_fold = np.concatenate([np.full(len(X_train), -1, dtype=int), np.zeros(len(X_valid), dtype=int)])
    cal_split = PredefinedSplit(test_fold=test_fold)
    cal_logreg = CalibratedClassifierCV(logreg_for_cal, method="isotonic", cv=cal_split)
    cal_logreg.fit(X_cal, y_cal)
    p_logreg_cal_test = cal_logreg.predict_proba(X_test)

    # HGB fixed setup + sigmoid calibration
    hgb_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                HistGradientBoostingClassifier(
                    loss="log_loss",
                    learning_rate=0.05,
                    max_depth=3,
                    max_iter=300,
                    random_state=42,
                ),
            ),
        ]
    )
    hgb_pipe.fit(X_train, y_train)
    p_hgb_test = hgb_pipe.predict_proba(X_test)

    hgb_for_cal = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                HistGradientBoostingClassifier(
                    loss="log_loss",
                    learning_rate=0.05,
                    max_depth=3,
                    max_iter=300,
                    random_state=42,
                ),
            ),
        ]
    )
    cal_hgb = CalibratedClassifierCV(hgb_for_cal, method="sigmoid", cv=cal_split)
    cal_hgb.fit(X_cal, y_cal)
    p_hgb_cal_test = cal_hgb.predict_proba(X_test)

    # NEW: Random Forest tuning
    rf_pipe, rf_params, _ = _fit_tuned(
        model_name="ml_random_forest",
        build_pipeline=lambda p: Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
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
            {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 1},
            {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 3},
            {"n_estimators": 400, "max_depth": 8, "min_samples_leaf": 1},
            {"n_estimators": 400, "max_depth": 8, "min_samples_leaf": 3},
        ],
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid_labels=valid["y"].to_numpy(),
    )
    p_rf_test = rf_pipe.predict_proba(X_test)

    # NEW: Extra Trees tuning
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
    p_et_test = et_pipe.predict_proba(X_test)

    # NEW: MLP tuning
    mlp_pipe, mlp_params, _ = _fit_tuned(
        model_name="ml_mlp",
        build_pipeline=lambda p: Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=tuple(p["hidden_layer_sizes"]),
                        alpha=float(p["alpha"]),
                        learning_rate_init=float(p["learning_rate_init"]),
                        max_iter=1200,
                        random_state=42,
                    ),
                ),
            ]
        ),
        param_grid=[
            {"hidden_layer_sizes": [32], "alpha": 1e-4, "learning_rate_init": 1e-3},
            {"hidden_layer_sizes": [64], "alpha": 1e-4, "learning_rate_init": 1e-3},
            {"hidden_layer_sizes": [64, 32], "alpha": 1e-4, "learning_rate_init": 1e-3},
            {"hidden_layer_sizes": [64, 32], "alpha": 1e-3, "learning_rate_init": 5e-4},
        ],
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid_labels=valid["y"].to_numpy(),
    )
    p_mlp_test = mlp_pipe.predict_proba(X_test)

    # NEW: XGBoost tuning
    xgb_pipe, xgb_params, _ = _fit_tuned(
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
                        reg_lambda=1.0,
                        random_state=42,
                        n_jobs=-1,
                        tree_method="hist",
                        eval_metric="mlogloss",
                    ),
                ),
            ]
        ),
        param_grid=[
            {
                "n_estimators": 300,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
            },
            {
                "n_estimators": 500,
                "max_depth": 3,
                "learning_rate": 0.03,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
            },
        ],
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid_labels=valid["y"].to_numpy(),
    )
    p_xgb_test = xgb_pipe.predict_proba(X_test)

    # NEW: LightGBM tuning
    lgbm_pipe, lgbm_params, _ = _fit_tuned(
        model_name="ml_lightgbm",
        build_pipeline=lambda p: Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    LGBMClassifier(
                        objective="multiclass",
                        num_class=3,
                        n_estimators=int(p["n_estimators"]),
                        learning_rate=float(p["learning_rate"]),
                        num_leaves=int(p["num_leaves"]),
                        min_child_samples=int(p["min_child_samples"]),
                        subsample=float(p["subsample"]),
                        colsample_bytree=float(p["colsample_bytree"]),
                        random_state=42,
                        n_jobs=-1,
                        verbosity=-1,
                    ),
                ),
            ]
        ),
        param_grid=[
            {
                "n_estimators": 300,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "min_child_samples": 20,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
            },
            {
                "n_estimators": 500,
                "learning_rate": 0.03,
                "num_leaves": 31,
                "min_child_samples": 30,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
            },
        ],
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid_labels=valid["y"].to_numpy(),
    )
    p_lgbm_test = lgbm_pipe.predict_proba(X_test)

    # NEW: CatBoost tuning
    cat_pipe, cat_params, _ = _fit_tuned(
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
    p_cat_test = cat_pipe.predict_proba(X_test)

    fold_info = {
        "league": league_name,
        "fold_id": fold_id,
        "train": ",".join(train_seasons),
        "valid": valid_season,
        "test": test_season,
    }

    rows: list[dict[str, object]] = []

    def add_row(model: str, probs: np.ndarray, params: dict[str, Any] | None = None, n: int | None = None) -> None:
        rows.append(
            {
                **fold_info,
                "model": model,
                "selected_params": "" if params is None else json.dumps(params, sort_keys=True),
                **compute_metrics(y_test, probs),
                "n": len(test) if n is None else int(n),
            }
        )

    add_row("freq_baseline", p_freq)
    add_row("xg_form_poisson", xg_probs_test)
    add_row("ml_logreg", p_logreg_test, logreg_params)
    add_row("ml_logreg_isotonic", p_logreg_cal_test, {"base_C": logreg_params["C"], "calibration": "isotonic"})
    add_row("ml_hgb", p_hgb_test, {"learning_rate": 0.05, "max_depth": 3, "max_iter": 300})
    add_row("ml_hgb_sigmoid", p_hgb_cal_test, {"calibration": "sigmoid"})
    add_row("ml_random_forest", p_rf_test, rf_params)
    add_row("ml_extra_trees", p_et_test, et_params)
    add_row("ml_mlp", p_mlp_test, mlp_params)
    add_row("ml_xgboost", p_xgb_test, xgb_params)
    add_row("ml_lightgbm", p_lgbm_test, lgbm_params)
    add_row("ml_catboost", p_cat_test, cat_params)

    if bm_mask.any():
        y_bm = y_test[bm_mask]
        p_bm = bm_probs.loc[bm_mask, ["pH", "pD", "pA"]].to_numpy(dtype=float)
        rows.append(
            {
                **fold_info,
                "model": "bookmaker_implied",
                "selected_params": "",
                **compute_metrics(y_bm, p_bm),
                "n": int(bm_mask.sum()),
            }
        )
    else:
        rows.append(
            {
                **fold_info,
                "model": "bookmaker_implied",
                "selected_params": "",
                "log_loss": np.nan,
                "brier": np.nan,
                "n": 0,
            }
        )

    return rows


def summarize_by_league(df_folds: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df_folds.groupby(["league", "model"], as_index=False)
        .agg(
            folds=("log_loss", "count"),
            log_loss_mean=("log_loss", "mean"),
            log_loss_std=("log_loss", "std"),
            brier_mean=("brier", "mean"),
            brier_std=("brier", "std"),
        )
        .sort_values(["league", "log_loss_mean", "brier_mean"])
        .reset_index(drop=True)
    )
    summary["log_loss_std"] = summary["log_loss_std"].fillna(0.0)
    summary["brier_std"] = summary["brier_std"].fillna(0.0)
    return summary


def summarize_global(df_folds: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df_folds.groupby("model", as_index=False)
        .agg(
            leagues=("league", "nunique"),
            folds=("log_loss", "count"),
            log_loss_mean=("log_loss", "mean"),
            log_loss_std=("log_loss", "std"),
            brier_mean=("brier", "mean"),
            brier_std=("brier", "std"),
        )
        .sort_values(["log_loss_mean", "brier_mean"])
        .reset_index(drop=True)
    )
    summary["log_loss_std"] = summary["log_loss_std"].fillna(0.0)
    summary["brier_std"] = summary["brier_std"].fillna(0.0)
    return summary


def main() -> None:
    base = Path("datasets")
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    folds = make_time_folds(SEASONS, min_train_seasons=3)
    window_size = 5
    max_goals = 10

    all_rows: list[dict[str, object]] = []

    for league_name, league_prefix in LEAGUES.items():
        print(f"Loading {league_name} ({league_prefix})...")
        matches = load_league_matches(base, league_prefix, SEASONS)
        matches["y"] = outcome_from_score(matches)
        matches = add_rolling_form_features(matches, window_size=window_size)

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
                    window_size=window_size,
                    max_goals=max_goals,
                )
            )

    df_folds = pd.DataFrame(all_rows).sort_values(["league", "fold_id", "log_loss"]).reset_index(drop=True)
    df_summary_by_league = summarize_by_league(df_folds)
    df_summary_global = summarize_global(df_folds)

    out_folds = out_dir / "experiment_stageA_boosting_fold_metrics.csv"
    out_summary_league = out_dir / "experiment_stageA_boosting_summary_by_league.csv"
    out_summary_global = out_dir / "experiment_stageA_boosting_summary_global.csv"

    df_folds.to_csv(out_folds, index=False)
    df_summary_by_league.to_csv(out_summary_league, index=False)
    df_summary_global.to_csv(out_summary_global, index=False)

    print("\n=== Stage A summary by league (mean across folds) ===")
    print(df_summary_by_league.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    print("\n=== Stage A global summary (all leagues + folds) ===")
    print(df_summary_global.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    print(f"\nSaved fold metrics: {out_folds}")
    print(f"Saved summary by league: {out_summary_league}")
    print(f"Saved global summary: {out_summary_global}")


if __name__ == "__main__":
    main()
