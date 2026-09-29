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
        "ece": float(multiclass_ece_toplabel(y_true, probs, n_bins=10)),
    }


def multiclass_ece_toplabel(y_true: np.ndarray, probs: np.ndarray, *, n_bins: int = 10) -> float:
    probs = clip_and_renormalize(probs)
    idx = {"H": 0, "D": 1, "A": 2}
    y_idx = np.array([idx[v] for v in y_true], dtype=int)
    pred_idx = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)
    correct = (pred_idx == y_idx).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not mask.any():
            continue
        bin_acc = float(correct[mask].mean())
        bin_conf = float(conf[mask].mean())
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def reliability_bins(y_true: np.ndarray, probs: np.ndarray, *, n_bins: int = 10) -> pd.DataFrame:
    probs = clip_and_renormalize(probs)
    idx = {"H": 0, "D": 1, "A": 2}
    y_idx = np.array([idx[v] for v in y_true], dtype=int)
    pred_idx = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)
    correct = (pred_idx == y_idx).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, Any]] = []
    n = len(conf)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not mask.any():
            rows.append({"bin": i, "bin_left": lo, "bin_right": hi, "count": 0, "weight": 0.0, "avg_conf": np.nan, "emp_acc": np.nan})
            continue
        rows.append(
            {
                "bin": i,
                "bin_left": lo,
                "bin_right": hi,
                "count": int(mask.sum()),
                "weight": float(mask.sum() / n),
                "avg_conf": float(conf[mask].mean()),
                "emp_acc": float(correct[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


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


def run_fold(matches: pd.DataFrame, *, league_name: str, fold: dict[str, object]) -> tuple[list[dict[str, object]], list[pd.DataFrame]]:
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

    rows: list[dict[str, object]] = []
    rel_rows: list[pd.DataFrame] = []
    info = {
        "league": league_name,
        "fold_id": fold_id,
        "train": ",".join(train_seasons),
        "valid": valid_season,
        "test": test_season,
    }

    bm_probs = implied_probs_from_odds(test)
    bm_mask = bm_probs.notna().all(axis=1).to_numpy()
    if bm_mask.any():
        y_bm = y_test_labels[bm_mask]
        p_bm = bm_probs.loc[bm_mask, ["pH", "pD", "pA"]].to_numpy(dtype=float)
        rows.append({**info, "feature_set": "reference", "model": "bookmaker_implied", "variant": "raw", "selected_params": "", **compute_metrics(y_bm, p_bm), "n": int(bm_mask.sum())})
        rel = reliability_bins(y_bm, p_bm, n_bins=10)
        rel["league"] = league_name
        rel["fold_id"] = fold_id
        rel["feature_set"] = "reference"
        rel["model"] = "bookmaker_implied"
        rel["variant"] = "raw"
        rel_rows.append(rel)

    for fs_name, fs_cols in feature_sets.items():
        X_train = df_features.loc[train.index, fs_cols].to_numpy(dtype=float)
        X_valid = df_features.loc[valid.index, fs_cols].to_numpy(dtype=float)
        X_test = df_features.loc[test.index, fs_cols].to_numpy(dtype=float)

        # finalist 1: logreg (best setting from prior stages: C=0.1)
        logreg_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=0.1, solver="lbfgs", max_iter=2000, random_state=42)),
            ]
        )
        logreg_pipe.fit(X_train, y_train)
        p_raw = logreg_pipe.predict_proba(X_test)
        rows.append({**info, "feature_set": fs_name, "model": "ml_logreg", "variant": "raw", "selected_params": json.dumps({"C": 0.1}, sort_keys=True), **compute_metrics(y_test_labels, p_raw), "n": len(test)})
        rel = reliability_bins(y_test_labels, p_raw, n_bins=10)
        rel["league"] = league_name
        rel["fold_id"] = fold_id
        rel["feature_set"] = fs_name
        rel["model"] = "ml_logreg"
        rel["variant"] = "raw"
        rel_rows.append(rel)

        for method in ("sigmoid", "isotonic"):
            cal = fit_calibrated(base_pipe=logreg_pipe, X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid, method=method)
            p_cal = cal.predict_proba(X_test)
            rows.append({**info, "feature_set": fs_name, "model": "ml_logreg", "variant": method, "selected_params": json.dumps({"C": 0.1, "calibration": method}, sort_keys=True), **compute_metrics(y_test_labels, p_cal), "n": len(test)})
            rel = reliability_bins(y_test_labels, p_cal, n_bins=10)
            rel["league"] = league_name
            rel["fold_id"] = fold_id
            rel["feature_set"] = fs_name
            rel["model"] = "ml_logreg"
            rel["variant"] = method
            rel_rows.append(rel)

        # finalist 2: extra trees (best setting from prior stages)
        et_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", ExtraTreesClassifier(n_estimators=500, max_depth=8, min_samples_leaf=3, random_state=42, n_jobs=-1)),
            ]
        )
        et_pipe.fit(X_train, y_train)
        p_raw = et_pipe.predict_proba(X_test)
        rows.append({**info, "feature_set": fs_name, "model": "ml_extra_trees", "variant": "raw", "selected_params": json.dumps({"n_estimators": 500, "max_depth": 8, "min_samples_leaf": 3}, sort_keys=True), **compute_metrics(y_test_labels, p_raw), "n": len(test)})
        rel = reliability_bins(y_test_labels, p_raw, n_bins=10)
        rel["league"] = league_name
        rel["fold_id"] = fold_id
        rel["feature_set"] = fs_name
        rel["model"] = "ml_extra_trees"
        rel["variant"] = "raw"
        rel_rows.append(rel)

        for method in ("sigmoid", "isotonic"):
            cal = fit_calibrated(base_pipe=et_pipe, X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid, method=method)
            p_cal = cal.predict_proba(X_test)
            rows.append({**info, "feature_set": fs_name, "model": "ml_extra_trees", "variant": method, "selected_params": json.dumps({"n_estimators": 500, "max_depth": 8, "min_samples_leaf": 3, "calibration": method}, sort_keys=True), **compute_metrics(y_test_labels, p_cal), "n": len(test)})
            rel = reliability_bins(y_test_labels, p_cal, n_bins=10)
            rel["league"] = league_name
            rel["fold_id"] = fold_id
            rel["feature_set"] = fs_name
            rel["model"] = "ml_extra_trees"
            rel["variant"] = method
            rel_rows.append(rel)

    return rows, rel_rows


def summarize(df_folds: pd.DataFrame, by_league: bool) -> pd.DataFrame:
    group_cols = ["feature_set", "model", "variant"]
    if by_league:
        group_cols = ["league"] + group_cols
    out = (
        df_folds.groupby(group_cols, as_index=False)
        .agg(
            folds=("log_loss", "count"),
            log_loss_mean=("log_loss", "mean"),
            log_loss_std=("log_loss", "std"),
            brier_mean=("brier", "mean"),
            brier_std=("brier", "std"),
            ece_mean=("ece", "mean"),
            ece_std=("ece", "std"),
        )
        .sort_values(group_cols + ["log_loss_mean"]) 
        .reset_index(drop=True)
    )
    for c in ("log_loss_std", "brier_std", "ece_std"):
        out[c] = out[c].fillna(0.0)
    return out


def summarize_reliability(df_rel: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df_rel.groupby(["feature_set", "model", "variant", "bin"], as_index=False)
        .agg(
            count=("count", "sum"),
            weight=("weight", "sum"),
            avg_conf=("avg_conf", "mean"),
            emp_acc=("emp_acc", "mean"),
        )
        .sort_values(["feature_set", "model", "variant", "bin"])
        .reset_index(drop=True)
    )
    return agg


def main() -> None:
    base = Path("datasets")
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    folds = make_time_folds(min_train_seasons=3)
    all_rows: list[dict[str, object]] = []
    all_rel: list[pd.DataFrame] = []

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
            rows, rel = run_fold(matches, league_name=league_name, fold=fold)
            all_rows.extend(rows)
            all_rel.extend(rel)

    df_folds = pd.DataFrame(all_rows).sort_values(["league", "fold_id", "feature_set", "model", "variant", "log_loss"]).reset_index(drop=True)
    df_rel = pd.concat(all_rel, ignore_index=True)

    df_summary_global = summarize(df_folds, by_league=False)
    df_summary_by_league = summarize(df_folds, by_league=True)
    df_rel_global = summarize_reliability(df_rel)

    out_folds = out_dir / "experiment_stageD_calibration_fold_metrics.csv"
    out_summary_global = out_dir / "experiment_stageD_calibration_summary_global.csv"
    out_summary_by_league = out_dir / "experiment_stageD_calibration_summary_by_league.csv"
    out_rel = out_dir / "experiment_stageD_calibration_reliability_bins.csv"

    df_folds.to_csv(out_folds, index=False)
    df_summary_global.to_csv(out_summary_global, index=False)
    df_summary_by_league.to_csv(out_summary_by_league, index=False)
    df_rel_global.to_csv(out_rel, index=False)

    print("\n=== Stage D global summary ===")
    print(df_summary_global.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print(f"\nSaved fold metrics: {out_folds}")
    print(f"Saved global summary: {out_summary_global}")
    print(f"Saved by-league summary: {out_summary_by_league}")
    print(f"Saved reliability bins: {out_rel}")


if __name__ == "__main__":
    main()
