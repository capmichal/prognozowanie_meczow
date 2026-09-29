from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import PredefinedSplit

from experiment_utils import (
    SEASONS_V1,
    TEST_SEASONS_V1,
    TRAIN_SEASONS_V1,
    VALID_SEASONS_V1,
    add_rolling_form_features,
    brier_score_multiclass,
    build_matches_with_xg,
    clip_and_renormalize,
    implied_probs_from_odds,
    log_loss_multiclass,
    load_results_season,
    load_shots_season,
    outcome_from_score,
    poisson_1x2_probs,
)


def _season_paths(base: Path, season: str) -> tuple[Path, Path]:
    results = base / f"premier_league_{season}.csv"
    shots = base / f"shots_premier_league_{season}.csv"
    if not results.exists():
        raise FileNotFoundError(f"Missing results file: {results}")
    if not shots.exists():
        raise FileNotFoundError(f"Missing shots file: {shots}")
    return results, shots


def load_epl_2014_2021(base: Path) -> pd.DataFrame:
    results_all: list[pd.DataFrame] = []
    shots_all: list[pd.DataFrame] = []

    for season in SEASONS_V1:
        results_path, shots_path = _season_paths(base, season)
        results_all.append(load_results_season(results_path, season=season))
        shots_all.append(load_shots_season(shots_path, season=season))

    results = pd.concat(results_all, ignore_index=True)
    shots = pd.concat(shots_all, ignore_index=True)

    matches = build_matches_with_xg(results, shots)
    return matches


def split_sets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["Season"].isin(TRAIN_SEASONS_V1)].copy()
    valid = df[df["Season"].isin(VALID_SEASONS_V1)].copy()
    test = df[df["Season"].isin(TEST_SEASONS_V1)].copy()

    if train.empty or valid.empty or test.empty:
        raise ValueError("One of splits is empty; check Season labels.")

    return train, valid, test


def build_xg_form_model_probs(
    df_all: pd.DataFrame,
    *,
    window_size: int,
    league_avg_home_xg: float,
    league_avg_away_xg: float,
    max_goals: int,
) -> np.ndarray:
    """Your baseline: rolling-xG AS/DS -> lambdas -> Poisson scoreline -> P(1X2)."""

    # Avoid recomputing rolling features if they are already present.
    expected_col = f"Forma_xG_S_{window_size}_H"
    df = df_all if expected_col in df_all.columns else add_rolling_form_features(df_all, window_size=window_size)

    # Rolling xG scored/conceded (home and away)
    home_xg_s = df[f"Forma_xG_S_{window_size}_H"].to_numpy(dtype=float)
    away_xg_s = df[f"Forma_xG_S_{window_size}_A"].to_numpy(dtype=float)

    home_xg_c = df[f"Forma_xG_C_{window_size}_H"].to_numpy(dtype=float)
    away_xg_c = df[f"Forma_xG_C_{window_size}_A"].to_numpy(dtype=float)

    # Attack/Defense strengths vs league average.
    # Fill NaNs with 1.0 to avoid dropping early-season matches.
    home_as = np.nan_to_num(home_xg_s / league_avg_home_xg, nan=1.0, posinf=1.0, neginf=1.0)
    away_as = np.nan_to_num(away_xg_s / league_avg_away_xg, nan=1.0, posinf=1.0, neginf=1.0)

    home_ds = np.nan_to_num(home_xg_c / league_avg_away_xg, nan=1.0, posinf=1.0, neginf=1.0)
    away_ds = np.nan_to_num(away_xg_c / league_avg_home_xg, nan=1.0, posinf=1.0, neginf=1.0)

    lam_home = home_as * away_ds * league_avg_home_xg
    lam_away = away_as * home_ds * league_avg_away_xg

    probs = poisson_1x2_probs(lam_home, lam_away, max_goals=max_goals)
    return probs


def compute_metrics(
    *,
    y_true: np.ndarray,
    probs: np.ndarray,
) -> dict[str, float]:
    probs = clip_and_renormalize(probs)
    return {
        "log_loss": float(log_loss_multiclass(y_true, probs)),
        "brier": float(brier_score_multiclass(y_true, probs)),
    }


def plot_cumulative_log_loss(
    dates: pd.Series,
    y_true: np.ndarray,
    probs_by_name: dict[str, np.ndarray],
    *,
    out_path: Path,
) -> None:
    df_plot = pd.DataFrame({"Date": pd.to_datetime(dates).dt.normalize()})

    for name, probs in probs_by_name.items():
        probs = clip_and_renormalize(probs)
        idx = {"H": 0, "D": 1, "A": 2}
        p_true = np.array([probs[i, idx[y_true[i]]] for i in range(len(y_true))], dtype=float)
        neglog = -np.log(np.clip(p_true, 1e-12, 1.0))
        df_plot[name] = neglog

    df_plot = df_plot.sort_values("Date").reset_index(drop=True)

    plt.figure(figsize=(10, 5))
    for name in probs_by_name.keys():
        cum = df_plot[name].expanding().mean()
        plt.plot(df_plot["Date"], cum, label=name)

    plt.title("Cumulative Log Loss (Test Season 20/21)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative mean -log(p_true)")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    base = Path("datasets")
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    print("Loading EPL seasons 2014/15–2020/21...")
    matches = load_epl_2014_2021(base)

    print(f"Loaded matches with xG: {len(matches)}")

    # Create target labels
    matches["y"] = outcome_from_score(matches)

    window_size = 5
    max_goals = 10

    print("Computing rolling form features...")
    matches = add_rolling_form_features(matches, window_size=window_size)

    # Split
    train, valid, test = split_sets(matches)

    # League averages computed ONLY on train+valid to avoid leakage.
    train_valid = pd.concat([train, valid], ignore_index=True)
    league_avg_home_xg = float(train_valid["Home_xG"].mean())
    league_avg_away_xg = float(train_valid["Away_xG"].mean())

    # Baseline 1: frequency baseline based on train outcomes
    train_y = train["y"].to_numpy()
    freq = {
        "H": float(np.mean(train_y == "H")),
        "D": float(np.mean(train_y == "D")),
        "A": float(np.mean(train_y == "A")),
    }
    p_freq = np.tile([freq["H"], freq["D"], freq["A"]], (len(test), 1))

    # Baseline 2: bookmaker implied probs
    bm_probs = implied_probs_from_odds(test)
    bm_mask = bm_probs.notna().all(axis=1).to_numpy()

    # Model: xG-form -> Poisson scoreline
    print("Computing xG-form model probabilities...")
    all_xg_probs = build_xg_form_model_probs(
        matches,
        window_size=window_size,
        league_avg_home_xg=league_avg_home_xg,
        league_avg_away_xg=league_avg_away_xg,
        max_goals=max_goals,
    )

    # Extract only test rows (same order as test)
    test_idx = test.index.to_numpy()
    xg_probs_test = all_xg_probs[test_idx, :]

    # Metrics
    y_test = test["y"].to_numpy()

    # ML features
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

    # Add two derived features from xG-form AS/DS model (helps ML learn non-linearities cheaply).
    df_for_features = matches.copy()
    home_xg_s = df_for_features[f"Forma_xG_S_{window_size}_H"].to_numpy(dtype=float)
    away_xg_s = df_for_features[f"Forma_xG_S_{window_size}_A"].to_numpy(dtype=float)
    home_xg_c = df_for_features[f"Forma_xG_C_{window_size}_H"].to_numpy(dtype=float)
    away_xg_c = df_for_features[f"Forma_xG_C_{window_size}_A"].to_numpy(dtype=float)

    home_as = np.nan_to_num(home_xg_s / league_avg_home_xg, nan=1.0, posinf=1.0, neginf=1.0)
    away_as = np.nan_to_num(away_xg_s / league_avg_away_xg, nan=1.0, posinf=1.0, neginf=1.0)
    home_ds = np.nan_to_num(home_xg_c / league_avg_away_xg, nan=1.0, posinf=1.0, neginf=1.0)
    away_ds = np.nan_to_num(away_xg_c / league_avg_home_xg, nan=1.0, posinf=1.0, neginf=1.0)

    df_for_features["lam_home_xgform"] = home_as * away_ds * league_avg_home_xg
    df_for_features["lam_away_xgform"] = away_as * home_ds * league_avg_away_xg
    df_for_features["lam_diff_xgform"] = df_for_features["lam_home_xgform"] - df_for_features["lam_away_xgform"]

    feature_cols += ["lam_home_xgform", "lam_away_xgform", "lam_diff_xgform"]

    # Build split matrices (no leakage: calibration uses valid only)
    y_map = {"H": 0, "D": 1, "A": 2}
    X_train = df_for_features.loc[train.index, feature_cols].to_numpy(dtype=float)
    y_train = np.array([y_map[v] for v in train["y"].to_numpy()], dtype=int)
    X_valid = df_for_features.loc[valid.index, feature_cols].to_numpy(dtype=float)
    y_valid = np.array([y_map[v] for v in valid["y"].to_numpy()], dtype=int)
    X_test = df_for_features.loc[test.index, feature_cols].to_numpy(dtype=float)

    # Model: multinomial logistic regression (tuned on validation)
    print("Training ML model: multinomial logistic regression...")
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
        raise RuntimeError("Failed to fit logistic regression model")

    p_logreg_test = best_logreg_pipe.predict_proba(X_test)

    # Calibrated logreg (isotonic) using validation season only.
    # scikit-learn>=1.8 removed cv='prefit', so we calibrate via PredefinedSplit:
    # fit base model on train fold, calibrate on valid fold.
    print("Calibrating logistic regression (isotonic) on validation season...")
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

    # Model: gradient boosting (HistGradientBoosting) + calibration
    print("Training ML model: HistGradientBoostingClassifier...")
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

    print("Calibrating HGB (sigmoid) on validation season...")
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

    rows: list[dict[str, object]] = []

    rows.append({"model": "freq_baseline", **compute_metrics(y_true=y_test, probs=p_freq), "n": len(test)})

    # xG model
    rows.append({"model": "xg_form_poisson", **compute_metrics(y_true=y_test, probs=xg_probs_test), "n": len(test)})

    # ML models
    rows.append(
        {
            "model": f"ml_logreg_c{best_c:g}",
            **compute_metrics(y_true=y_test, probs=p_logreg_test),
            "n": len(test),
        }
    )
    rows.append(
        {
            "model": f"ml_logreg_isotonic_c{best_c:g}",
            **compute_metrics(y_true=y_test, probs=p_logreg_cal_test),
            "n": len(test),
        }
    )
    rows.append({"model": "ml_hgb", **compute_metrics(y_true=y_test, probs=p_hgb_test), "n": len(test)})
    rows.append({"model": "ml_hgb_sigmoid", **compute_metrics(y_true=y_test, probs=p_hgb_cal_test), "n": len(test)})

    # bookmaker only where odds exist
    if bm_mask.any():
        y_bm = y_test[bm_mask]
        p_bm = bm_probs.loc[bm_mask, ["pH", "pD", "pA"]].to_numpy(dtype=float)
        rows.append({"model": "bookmaker_implied", **compute_metrics(y_true=y_bm, probs=p_bm), "n": int(bm_mask.sum())})
    else:
        rows.append({"model": "bookmaker_implied", "log_loss": np.nan, "brier": np.nan, "n": 0})

    metrics_df = pd.DataFrame(rows).sort_values("log_loss")

    out_csv = out_dir / "experiment_v1_metrics.csv"
    metrics_df.to_csv(out_csv, index=False)

    print("\n=== Experiment v1 (EPL) ===")
    print("Split: train 14/15–18/19 | valid 19/20 | test 20/21")
    print(f"League avg xG (train+valid): home={league_avg_home_xg:.3f}, away={league_avg_away_xg:.3f}")
    print(metrics_df.to_string(index=False))
    print(f"\nSaved metrics: {out_csv}")

    # Plot cumulative log loss on test (only for models that cover all test matches)
    probs_for_plot = {
        "freq_baseline": p_freq,
        "xg_form_poisson": xg_probs_test,
        f"ml_logreg_c{best_c:g}": p_logreg_test,
        f"ml_logreg_isotonic_c{best_c:g}": p_logreg_cal_test,
        "ml_hgb": p_hgb_test,
        "ml_hgb_sigmoid": p_hgb_cal_test,
    }

    out_png = out_dir / "experiment_v1_cum_logloss_20-21.png"
    plot_cumulative_log_loss(test["Date"], y_test, probs_for_plot, out_path=out_png)
    print(f"Saved plot: {out_png}")


if __name__ == "__main__":
    main()
