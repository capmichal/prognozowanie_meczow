from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

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


SEASONS: tuple[str, ...] = ("14-15", "15-16", "16-17", "17-18", "18-19", "19-20", "20-21")
LEAGUES: dict[str, str] = {"EPL": "premier_league", "LaLiga": "la_liga", "Bundesliga": "bundesliga"}


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


def dixon_coles_tau(i: int, j: int, lam_home: float, lam_away: float, rho: float) -> float:
    if i == 0 and j == 0:
        return 1.0 - (lam_home * lam_away * rho)
    if i == 0 and j == 1:
        return 1.0 + (lam_home * rho)
    if i == 1 and j == 0:
        return 1.0 + (lam_away * rho)
    if i == 1 and j == 1:
        return 1.0 - rho
    return 1.0


def dixon_coles_1x2_probs(lam_home: np.ndarray, lam_away: np.ndarray, rho: float, *, max_goals: int = 10) -> np.ndarray:
    lam_home = np.asarray(lam_home, dtype=float)
    lam_away = np.asarray(lam_away, dtype=float)
    lam_home = np.clip(lam_home, 1e-6, 20.0)
    lam_away = np.clip(lam_away, 1e-6, 20.0)

    goals = np.arange(max_goals + 1)
    fact = np.array([math.factorial(int(k)) for k in goals], dtype=float)

    pmf_h = np.exp(-lam_home[:, None]) * (lam_home[:, None] ** goals[None, :]) / fact[None, :]
    pmf_a = np.exp(-lam_away[:, None]) * (lam_away[:, None] ** goals[None, :]) / fact[None, :]

    n = len(lam_home)
    probs = np.zeros((n, 3), dtype=float)

    for m in range(n):
        score_prob = np.outer(pmf_h[m], pmf_a[m])
        for i in (0, 1):
            for j in (0, 1):
                score_prob[i, j] *= dixon_coles_tau(i, j, float(lam_home[m]), float(lam_away[m]), float(rho))

        score_prob = np.clip(score_prob, 0.0, None)
        s = float(score_prob.sum())
        if s <= 0.0:
            score_prob[:] = 1.0 / score_prob.size
        else:
            score_prob /= s

        pH = float(np.sum(np.tril(score_prob, k=-1)))
        pD = float(np.sum(np.diag(score_prob)))
        pA = float(np.sum(np.triu(score_prob, k=1)))
        probs[m] = [pH, pD, pA]

    return clip_and_renormalize(probs)


def build_lambdas(df_all: pd.DataFrame, *, window_size: int, league_avg_home_xg: float, league_avg_away_xg: float) -> tuple[np.ndarray, np.ndarray]:
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
    return lam_home, lam_away


def find_best_rho(y_valid: np.ndarray, lam_home_valid: np.ndarray, lam_away_valid: np.ndarray, *, max_goals: int) -> tuple[float, float]:
    grid = np.round(np.arange(-0.25, 0.2501, 0.01), 2)
    best_rho = 0.0
    best_ll = float("inf")
    for rho in grid:
        p = dixon_coles_1x2_probs(lam_home_valid, lam_away_valid, rho=float(rho), max_goals=max_goals)
        ll = log_loss_multiclass(y_valid, p)
        if ll < best_ll:
            best_ll = ll
            best_rho = float(rho)
    return best_rho, float(best_ll)


def run_fold(matches: pd.DataFrame, *, league_name: str, fold: dict[str, object], window_size: int, max_goals: int) -> list[dict[str, object]]:
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

    lam_home_all, lam_away_all = build_lambdas(
        matches,
        window_size=window_size,
        league_avg_home_xg=league_avg_home_xg,
        league_avg_away_xg=league_avg_away_xg,
    )

    idx_train = train.index.to_numpy()
    idx_valid = valid.index.to_numpy()
    idx_test = test.index.to_numpy()

    y_test = test["y"].to_numpy()
    y_valid = valid["y"].to_numpy()

    # Frequency baseline (from train)
    train_y = train["y"].to_numpy()
    freq = {
        "H": float(np.mean(train_y == "H")),
        "D": float(np.mean(train_y == "D")),
        "A": float(np.mean(train_y == "A")),
    }
    p_freq_test = np.tile([freq["H"], freq["D"], freq["A"]], (len(test), 1))

    # Poisson independent
    p_pois_test = poisson_1x2_probs(lam_home_all[idx_test], lam_away_all[idx_test], max_goals=max_goals)

    # Dixon-Coles rho tuned on validation
    rho, valid_ll = find_best_rho(
        y_valid,
        lam_home_all[idx_valid],
        lam_away_all[idx_valid],
        max_goals=max_goals,
    )
    p_dc_test = dixon_coles_1x2_probs(lam_home_all[idx_test], lam_away_all[idx_test], rho=rho, max_goals=max_goals)

    # Bookmaker benchmark when available
    bm_probs = implied_probs_from_odds(test)
    bm_mask = bm_probs.notna().all(axis=1).to_numpy()

    fold_info = {
        "league": league_name,
        "fold_id": fold_id,
        "train": ",".join(train_seasons),
        "valid": valid_season,
        "test": test_season,
    }

    rows: list[dict[str, object]] = []
    rows.append({**fold_info, "model": "freq_baseline", "rho": np.nan, **compute_metrics(y_test, p_freq_test), "n": len(test)})
    rows.append({**fold_info, "model": "poisson_independent", "rho": np.nan, **compute_metrics(y_test, p_pois_test), "n": len(test)})
    rows.append(
        {
            **fold_info,
            "model": "poisson_dixon_coles",
            "rho": rho,
            "valid_log_loss": valid_ll,
            **compute_metrics(y_test, p_dc_test),
            "n": len(test),
        }
    )

    if bm_mask.any():
        y_bm = y_test[bm_mask]
        p_bm = bm_probs.loc[bm_mask, ["pH", "pD", "pA"]].to_numpy(dtype=float)
        rows.append({**fold_info, "model": "bookmaker_implied", "rho": np.nan, **compute_metrics(y_bm, p_bm), "n": int(bm_mask.sum())})

    return rows


def summarize(df_folds: pd.DataFrame, by_league: bool) -> pd.DataFrame:
    group_cols = ["model"]
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
            rho_mean=("rho", "mean"),
            rho_std=("rho", "std"),
        )
        .sort_values(group_cols + ["log_loss_mean"]) 
        .reset_index(drop=True)
    )
    for c in ("log_loss_std", "brier_std", "rho_std"):
        out[c] = out[c].fillna(0.0)
    return out


def main() -> None:
    base = Path("datasets")
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    folds = make_time_folds(min_train_seasons=3)
    window_size = 5
    max_goals = 10

    all_rows: list[dict[str, object]] = []

    for league_name, league_prefix in LEAGUES.items():
        print(f"Loading {league_name} ({league_prefix})...")
        matches = load_league_matches(base, league_prefix)
        matches["y"] = outcome_from_score(matches)
        matches = add_rolling_form_features(matches, window_size=window_size)

        print(f"Running {league_name}: {len(folds)} rolling-origin folds")
        for fold in folds:
            print(
                f"  Fold {fold['fold_id']}: train={fold['train_seasons'][0]}..{fold['train_seasons'][-1]}, "
                f"valid={fold['valid_season']}, test={fold['test_season']}"
            )
            all_rows.extend(
                run_fold(
                    matches,
                    league_name=league_name,
                    fold=fold,
                    window_size=window_size,
                    max_goals=max_goals,
                )
            )

    df_folds = pd.DataFrame(all_rows).sort_values(["league", "fold_id", "log_loss"]).reset_index(drop=True)
    df_summary_global = summarize(df_folds, by_league=False)
    df_summary_by_league = summarize(df_folds, by_league=True)

    out_folds = out_dir / "experiment_stageF_poisson_dc_fold_metrics.csv"
    out_global = out_dir / "experiment_stageF_poisson_dc_summary_global.csv"
    out_by_league = out_dir / "experiment_stageF_poisson_dc_summary_by_league.csv"

    df_folds.to_csv(out_folds, index=False)
    df_summary_global.to_csv(out_global, index=False)
    df_summary_by_league.to_csv(out_by_league, index=False)

    print("\n=== Stage F global summary ===")
    print(df_summary_global.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print(f"\nSaved fold metrics: {out_folds}")
    print(f"Saved global summary: {out_global}")
    print(f"Saved by-league summary: {out_by_league}")


if __name__ == "__main__":
    main()
