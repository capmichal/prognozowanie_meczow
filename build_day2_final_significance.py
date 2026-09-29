from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(".")
OUT = ROOT / "outputs"


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def _candidate_series(
    *,
    candidate_row: pd.Series,
    track: str,
    df_i5: pd.DataFrame,
    df_f2: pd.DataFrame,
) -> pd.DataFrame:
    source = str(candidate_row["source"])
    model = str(candidate_row["model"])
    feature_pack = str(candidate_row["feature_pack"])
    variant = str(candidate_row.get("variant", "") or "")

    keys = ["league", "fold_id", "test"]

    if source == "stageI5":
        d = df_i5[
            (df_i5["track"] == track)
            & (df_i5["feature_pack"] == feature_pack)
            & (df_i5["model"] == model)
        ].copy()
        out = d[keys + ["log_loss", "brier"]]
        return out.drop_duplicates(subset=keys).reset_index(drop=True)

    if source == "stageF2":
        d = df_f2[
            (df_f2["track"] == track)
            & (df_f2["feature_pack"] == feature_pack)
            & (df_f2["model"] == model)
            & (df_f2["variant"] == variant)
        ].copy()
        out = d[keys + ["log_loss", "brier"]]
        return out.drop_duplicates(subset=keys).reset_index(drop=True)

    raise ValueError(f"Unknown source: {source}")


def _paired_bootstrap_ci(diff: np.ndarray, *, n_boot: int = 20000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(diff)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diff[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def _sign_flip_p_value(diff: np.ndarray) -> float:
    # Exact randomization test under H0 (symmetry around 0)
    n = len(diff)
    obs = float(np.mean(diff))
    if n == 0:
        return float("nan")

    means = []
    for signs in product((-1.0, 1.0), repeat=n):
        s = np.array(signs, dtype=float)
        means.append(float(np.mean(s * diff)))
    means_arr = np.array(means, dtype=float)
    p = float(np.mean(np.abs(means_arr) >= abs(obs)))
    return p


def _compare_pair(
    *,
    track: str,
    left_id: str,
    right_id: str,
    scoreboard: pd.DataFrame,
    df_i5: pd.DataFrame,
    df_f2: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    left = scoreboard[scoreboard["candidate_id"] == left_id].iloc[0]
    right = scoreboard[scoreboard["candidate_id"] == right_id].iloc[0]

    s_left = _candidate_series(candidate_row=left, track=track, df_i5=df_i5, df_f2=df_f2)
    s_right = _candidate_series(candidate_row=right, track=track, df_i5=df_i5, df_f2=df_f2)

    merged = s_left.merge(s_right, on=["league", "fold_id", "test"], suffixes=("_left", "_right"))
    merged["candidate_left"] = left_id
    merged["candidate_right"] = right_id
    merged["track"] = track
    merged["delta_log_loss_left_minus_right"] = merged["log_loss_left"] - merged["log_loss_right"]
    merged["delta_brier_left_minus_right"] = merged["brier_left"] - merged["brier_right"]

    d_ll = merged["delta_log_loss_left_minus_right"].to_numpy(dtype=float)
    d_br = merged["delta_brier_left_minus_right"].to_numpy(dtype=float)

    ll_ci_lo, ll_ci_hi = _paired_bootstrap_ci(d_ll)
    br_ci_lo, br_ci_hi = _paired_bootstrap_ci(d_br)

    summary = {
        "track": track,
        "left": left_id,
        "right": right_id,
        "n_folds": int(len(merged)),
        "mean_log_loss_left": float(merged["log_loss_left"].mean()),
        "mean_log_loss_right": float(merged["log_loss_right"].mean()),
        "delta_log_loss_left_minus_right": float(d_ll.mean()),
        "delta_log_loss_ci95_lo": ll_ci_lo,
        "delta_log_loss_ci95_hi": ll_ci_hi,
        "p_value_sign_flip_log_loss": _sign_flip_p_value(d_ll),
        "left_better_logloss_folds": int(np.sum(d_ll < 0.0)),
        "right_better_logloss_folds": int(np.sum(d_ll > 0.0)),
        "ties_logloss_folds": int(np.sum(np.isclose(d_ll, 0.0))),
        "mean_brier_left": float(merged["brier_left"].mean()),
        "mean_brier_right": float(merged["brier_right"].mean()),
        "delta_brier_left_minus_right": float(d_br.mean()),
        "delta_brier_ci95_lo": br_ci_lo,
        "delta_brier_ci95_hi": br_ci_hi,
        "p_value_sign_flip_brier": _sign_flip_p_value(d_br),
    }

    return merged, summary


def main() -> None:
    day1_a = _read(OUT / "day1_final_scoreboard_trackA.csv")
    day1_b = _read(OUT / "day1_final_scoreboard_trackB.csv")
    i5 = _read(OUT / "experiment_stageI5_modelzoo_fold_metrics.csv")
    f2 = _read(OUT / "experiment_stageF2_poisson_dc_covariates_fold_metrics.csv")

    pairs = [
        ("A", "A1", "A2"),
        ("B", "B1", "B0"),
        ("B", "B2", "B0"),
        ("B", "B1", "B2"),
    ]

    fold_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for track, left, right in pairs:
        board = day1_a if track == "A" else day1_b
        folds, summary = _compare_pair(
            track=track,
            left_id=left,
            right_id=right,
            scoreboard=board,
            df_i5=i5,
            df_f2=f2,
        )
        fold_rows.append(folds)
        summary_rows.append(summary)

    df_folds = pd.concat(fold_rows, ignore_index=True)
    df_summary = pd.DataFrame(summary_rows)

    out_folds = OUT / "day2_final_significance_fold_deltas.csv"
    out_summary = OUT / "day2_final_significance_summary.csv"
    df_folds.to_csv(out_folds, index=False)
    df_summary.to_csv(out_summary, index=False)

    print("Saved:")
    print(f"- {out_folds}")
    print(f"- {out_summary}")
    print("\n=== Day 2 significance summary ===")
    print(df_summary.to_string(index=False, float_format=lambda v: f"{v:.6f}"))


if __name__ == "__main__":
    main()
