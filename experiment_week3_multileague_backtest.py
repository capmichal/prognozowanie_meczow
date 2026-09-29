from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import PredefinedSplit
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

    best_c: float | None = None
    best_valid_ll = float("inf")
    best_logreg_pipe: Pipeline | None = None

    for c in (0.1, 1.0, 10.0):
        pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=c,
                        solver="lbfgs",
                        max_iter=2000,
                        random_state=42,
                    ),
                ),
            ]
        )
        pipe.fit(X_train, y_train)
        p_valid = pipe.predict_proba(X_valid)
        valid_ll = log_loss_multiclass(valid["y"].to_numpy(), p_valid)
        if valid_ll < best_valid_ll:
            best_valid_ll = valid_ll
            best_c = float(c)
            best_logreg_pipe = pipe

    if best_logreg_pipe is None or best_c is None:
        raise RuntimeError(f"{league_name} fold {fold_id}: failed to fit logistic regression")

    p_logreg_test = best_logreg_pipe.predict_proba(X_test)

    logreg_for_cal = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=best_c,
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

    fold_info = {
        "league": league_name,
        "fold_id": fold_id,
        "train": ",".join(train_seasons),
        "valid": valid_season,
        "test": test_season,
    }

    rows: list[dict[str, object]] = []
    rows.append(
        {
            **fold_info,
            "model": "freq_baseline",
            "logreg_c": np.nan,
            **compute_metrics(y_test, p_freq),
            "n": len(test),
        }
    )
    rows.append(
        {
            **fold_info,
            "model": "xg_form_poisson",
            "logreg_c": np.nan,
            **compute_metrics(y_test, xg_probs_test),
            "n": len(test),
        }
    )
    rows.append(
        {
            **fold_info,
            "model": "ml_logreg",
            "logreg_c": best_c,
            **compute_metrics(y_test, p_logreg_test),
            "n": len(test),
        }
    )
    rows.append(
        {
            **fold_info,
            "model": "ml_logreg_isotonic",
            "logreg_c": best_c,
            **compute_metrics(y_test, p_logreg_cal_test),
            "n": len(test),
        }
    )
    rows.append(
        {
            **fold_info,
            "model": "ml_hgb",
            "logreg_c": np.nan,
            **compute_metrics(y_test, p_hgb_test),
            "n": len(test),
        }
    )
    rows.append(
        {
            **fold_info,
            "model": "ml_hgb_sigmoid",
            "logreg_c": np.nan,
            **compute_metrics(y_test, p_hgb_cal_test),
            "n": len(test),
        }
    )

    if bm_mask.any():
        y_bm = y_test[bm_mask]
        p_bm = bm_probs.loc[bm_mask, ["pH", "pD", "pA"]].to_numpy(dtype=float)
        rows.append(
            {
                **fold_info,
                "model": "bookmaker_implied",
                "logreg_c": np.nan,
                **compute_metrics(y_bm, p_bm),
                "n": int(bm_mask.sum()),
            }
        )
    else:
        rows.append(
            {
                **fold_info,
                "model": "bookmaker_implied",
                "logreg_c": np.nan,
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

    out_folds = out_dir / "experiment_week3_multileague_fold_metrics.csv"
    out_summary_league = out_dir / "experiment_week3_multileague_summary_by_league.csv"
    out_summary_global = out_dir / "experiment_week3_multileague_summary_global.csv"

    df_folds.to_csv(out_folds, index=False)
    df_summary_by_league.to_csv(out_summary_league, index=False)
    df_summary_global.to_csv(out_summary_global, index=False)

    print("\n=== Summary by league (mean across folds) ===")
    print(df_summary_by_league.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    print("\n=== Global summary (all leagues + folds) ===")
    print(df_summary_global.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    print(f"\nSaved fold metrics: {out_folds}")
    print(f"Saved summary by league: {out_summary_league}")
    print(f"Saved global summary: {out_summary_global}")


if __name__ == "__main__":
    main()
