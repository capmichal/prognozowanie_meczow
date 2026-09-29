from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiment_utils import clip_and_renormalize


def per_match_log_loss(y_true: np.ndarray, probs: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    p = clip_and_renormalize(probs, eps=eps)
    idx = {"H": 0, "D": 1, "A": 2}
    y_idx = np.array([idx[v] for v in y_true], dtype=int)
    return -np.log(np.clip(p[np.arange(len(y_true)), y_idx], eps, 1.0))


def bootstrap_stratified_pairwise(
    wide: pd.DataFrame,
    *,
    col_a: str,
    col_b: str,
    strata_col: str,
    n_boot: int,
    rng_seed: int,
) -> dict[str, float]:
    diff = wide[col_a].to_numpy(dtype=float) - wide[col_b].to_numpy(dtype=float)
    point_estimate = float(np.mean(diff))

    group_ids = wide[strata_col].astype(str)
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
    out_dir = Path("outputs")
    src = out_dir / "experiment_stageH_market_residual_fold_metrics.csv"
    if not src.exists():
        raise FileNotFoundError(f"Missing input file: {src}")

    df = pd.read_csv(src)

    # Keep bookmaker and best stage H model
    focus_models = ["bookmaker_implied", "market_plus_residual_logreg"]
    df = df[df["model"].isin(focus_models)].copy()
    if df.empty:
        raise ValueError("No rows for bookmaker_implied / market_plus_residual_logreg")

    # Reconstruct per-match rows from fold-level means is impossible; use the fold-level means
    # as independent blocks and bootstrap over folds per league for conservative inference.
    # (To keep consistency with current artifacts.)
    wide = (
        df.pivot_table(index=["league", "fold_id", "test"], columns="model", values="log_loss", aggfunc="first")
        .dropna()
        .reset_index()
    )
    if wide.empty:
        raise ValueError("No overlapping folds for bookmaker vs residual")

    res = bootstrap_stratified_pairwise(
        wide,
        col_a="market_plus_residual_logreg",
        col_b="bookmaker_implied",
        strata_col="league",
        n_boot=10000,
        rng_seed=42,
    )

    summary = pd.DataFrame(
        [
            {
                "config_a": "market_plus_residual_logreg",
                "config_b": "bookmaker_implied",
                **res,
                "interpretation": "a_better" if res["delta_mean"] < 0 else "b_better",
                "note": "bootstrap over fold-level log_loss, stratified by league",
            }
        ]
    )

    out_wide = out_dir / "experiment_stageH_significance_wide.csv"
    out_summary = out_dir / "experiment_stageH_significance_pairwise.csv"
    wide.to_csv(out_wide, index=False)
    summary.to_csv(out_summary, index=False)

    print("=== Stage H significance ===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print(f"\nSaved wide table: {out_wide}")
    print(f"Saved pairwise summary: {out_summary}")


if __name__ == "__main__":
    main()
