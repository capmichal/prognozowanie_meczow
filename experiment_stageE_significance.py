from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiment_utils import (
    add_rolling_form_features,
    build_matches_with_xg,
    clip_and_renormalize,
    load_results_season,
    load_shots_season,
    outcome_from_score,
)


SEASONS: tuple[str, ...] = ("14-15", "15-16", "16-17", "17-18", "18-19", "19-20", "20-21")
LEAGUES: dict[str, str] = {"EPL": "premier_league", "LaLiga": "la_liga", "Bundesliga": "bundesliga"}
ROLLING_WINDOWS: tuple[int, ...] = (3, 5, 10)

CONFIGS: dict[str, dict[str, str]] = {
    "et_rest_sigmoid": {
        "feature_set": "fs_multiwindow_rest",
        "model": "ml_extra_trees",
        "variant": "sigmoid",
    },
    "et_multi_sigmoid": {
        "feature_set": "fs_multiwindow",
        "model": "ml_extra_trees",
        "variant": "sigmoid",
    },
    "logreg_multi_sigmoid": {
        "feature_set": "fs_multiwindow",
        "model": "ml_logreg",
        "variant": "sigmoid",
    },
}

PAIRWISE: tuple[tuple[str, str], ...] = (
    ("et_rest_sigmoid", "et_multi_sigmoid"),
    ("et_rest_sigmoid", "logreg_multi_sigmoid"),
    ("et_multi_sigmoid", "logreg_multi_sigmoid"),
)


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


def fit_calibrated(
    *,
    base_pipe: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    method: str,
) -> CalibratedClassifierCV:
    X_cal = np.vstack([X_train, X_valid])
    y_cal = np.concatenate([y_train, y_valid])
    test_fold = np.concatenate([np.full(len(X_train), -1, dtype=int), np.zeros(len(X_valid), dtype=int)])
    cal_split = PredefinedSplit(test_fold=test_fold)
    cal = CalibratedClassifierCV(base_pipe, method=method, cv=cal_split)
    cal.fit(X_cal, y_cal)
    return cal


def per_match_log_loss(y_true: np.ndarray, probs: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    p = clip_and_renormalize(probs, eps=eps)
    idx = {"H": 0, "D": 1, "A": 2}
    y_idx = np.array([idx[v] for v in y_true], dtype=int)
    return -np.log(np.clip(p[np.arange(len(y_true)), y_idx], eps, 1.0))


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
    y_test = test["y"].to_numpy()

    df_features = add_fold_dependent_features(matches, league_avg_home_xg=league_avg_home_xg, league_avg_away_xg=league_avg_away_xg)
    feature_sets = build_feature_sets()

    rows: list[dict[str, object]] = []
    base_info = {
        "league": league_name,
        "fold_id": fold_id,
        "train": ",".join(train_seasons),
        "valid": valid_season,
        "test": test_season,
    }

    for fs_name, fs_cols in feature_sets.items():
        X_train = df_features.loc[train.index, fs_cols].to_numpy(dtype=float)
        X_valid = df_features.loc[valid.index, fs_cols].to_numpy(dtype=float)
        X_test = df_features.loc[test.index, fs_cols].to_numpy(dtype=float)

        logreg_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=0.1, solver="lbfgs", max_iter=2000, random_state=42)),
            ]
        )
        logreg_pipe.fit(X_train, y_train)
        logreg_sig = fit_calibrated(
            base_pipe=logreg_pipe,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            method="sigmoid",
        )

        et_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", ExtraTreesClassifier(n_estimators=500, max_depth=8, min_samples_leaf=3, random_state=42, n_jobs=-1)),
            ]
        )
        et_pipe.fit(X_train, y_train)
        et_sig = fit_calibrated(
            base_pipe=et_pipe,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            method="sigmoid",
        )

        model_outputs: list[tuple[str, str, np.ndarray, dict[str, Any]]] = [
            ("ml_logreg", "sigmoid", logreg_sig.predict_proba(X_test), {"C": 0.1, "calibration": "sigmoid"}),
            (
                "ml_extra_trees",
                "sigmoid",
                et_sig.predict_proba(X_test),
                {"n_estimators": 500, "max_depth": 8, "min_samples_leaf": 3, "calibration": "sigmoid"},
            ),
        ]

        for model_name, variant, probs, params in model_outputs:
            losses = per_match_log_loss(y_test, probs)
            for i in range(len(test)):
                rows.append(
                    {
                        **base_info,
                        "match_local_idx": i,
                        "Date": str(test.iloc[i]["Date"].date()),
                        "HomeTeam": test.iloc[i]["HomeTeam"],
                        "AwayTeam": test.iloc[i]["AwayTeam"],
                        "y": y_test[i],
                        "feature_set": fs_name,
                        "model": model_name,
                        "variant": variant,
                        "config_id": f"{model_name}:{fs_name}:{variant}",
                        "selected_params": json.dumps(params, sort_keys=True),
                        "sample_log_loss": float(losses[i]),
                    }
                )

    return rows


def bootstrap_stratified_pairwise(
    df_long: pd.DataFrame,
    *,
    config_a: str,
    config_b: str,
    n_boot: int,
    rng_seed: int,
) -> dict[str, float]:
    # Build wide table with one row per match and two columns with sample losses
    key_cols = ["league", "fold_id", "match_local_idx"]
    wide = (
        df_long[df_long["config_id"].isin([config_a, config_b])]
        .pivot_table(index=key_cols, columns="config_id", values="sample_log_loss", aggfunc="first")
        .dropna()
        .reset_index()
    )

    if wide.empty:
        raise ValueError(f"No overlapping samples for pair {config_a} vs {config_b}")

    diff = wide[config_a].to_numpy(dtype=float) - wide[config_b].to_numpy(dtype=float)
    point_estimate = float(np.mean(diff))

    # Stratified bootstrap by (league, fold_id)
    group_ids = wide["league"].astype(str) + "::" + wide["fold_id"].astype(str)
    unique_groups = group_ids.unique().tolist()
    groups: list[np.ndarray] = [np.where(group_ids.to_numpy() == g)[0] for g in unique_groups]
    group_weights = np.array([len(g) / len(wide) for g in groups], dtype=float)

    rng = np.random.default_rng(rng_seed)
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        total = 0.0
        for w, idx in zip(group_weights, groups, strict=True):
            sampled = idx[rng.integers(0, len(idx), size=len(idx))]
            total += float(w * np.mean(diff[sampled]))
        boot[b] = total

    ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
    p_left = float(np.mean(boot <= 0.0))
    p_right = float(np.mean(boot >= 0.0))
    p_two = float(min(1.0, 2.0 * min(p_left, p_right)))

    return {
        "delta_mean": point_estimate,
        "ci_2_5": float(ci_low),
        "ci_97_5": float(ci_high),
        "p_two_sided": p_two,
        "p_left": p_left,
        "p_right": p_right,
    }


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

    df_match_losses = pd.DataFrame(all_rows)

    # Filter to the three configurations requested by Stage E
    df_match_losses = df_match_losses[df_match_losses["config_id"].isin([
        "ml_extra_trees:fs_multiwindow_rest:sigmoid",
        "ml_extra_trees:fs_multiwindow:sigmoid",
        "ml_logreg:fs_multiwindow:sigmoid",
    ])].copy()

    summary_cfg = (
        df_match_losses.groupby("config_id", as_index=False)
        .agg(
            n=("sample_log_loss", "count"),
            log_loss_mean=("sample_log_loss", "mean"),
            log_loss_std=("sample_log_loss", "std"),
        )
        .sort_values("log_loss_mean")
        .reset_index(drop=True)
    )

    pair_rows: list[dict[str, object]] = []
    pair_map = {
        "et_rest_sigmoid": "ml_extra_trees:fs_multiwindow_rest:sigmoid",
        "et_multi_sigmoid": "ml_extra_trees:fs_multiwindow:sigmoid",
        "logreg_multi_sigmoid": "ml_logreg:fs_multiwindow:sigmoid",
    }
    for a_key, b_key in PAIRWISE:
        cfg_a = pair_map[a_key]
        cfg_b = pair_map[b_key]
        res = bootstrap_stratified_pairwise(
            df_match_losses,
            config_a=cfg_a,
            config_b=cfg_b,
            n_boot=10000,
            rng_seed=42,
        )
        pair_rows.append(
            {
                "config_a": cfg_a,
                "config_b": cfg_b,
                **res,
                "interpretation": "a_better" if res["delta_mean"] < 0 else "b_better",
            }
        )

    df_pairs = pd.DataFrame(pair_rows).sort_values("delta_mean").reset_index(drop=True)

    out_losses = out_dir / "experiment_stageE_significance_match_logloss.csv"
    out_cfg = out_dir / "experiment_stageE_significance_config_summary.csv"
    out_pairs = out_dir / "experiment_stageE_significance_pairwise.csv"

    df_match_losses.to_csv(out_losses, index=False)
    summary_cfg.to_csv(out_cfg, index=False)
    df_pairs.to_csv(out_pairs, index=False)

    print("\n=== Stage E config summary ===")
    print(summary_cfg.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print("\n=== Stage E pairwise bootstrap ===")
    print(df_pairs.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print(f"\nSaved match log-loss rows: {out_losses}")
    print(f"Saved config summary: {out_cfg}")
    print(f"Saved pairwise tests: {out_pairs}")


if __name__ == "__main__":
    main()
