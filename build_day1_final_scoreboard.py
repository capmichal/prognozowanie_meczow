from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(".")
OUT = ROOT / "outputs"


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def main() -> None:
    stage_i5_a = _read(OUT / "experiment_stageI5_modelzoo_summary_trackA.csv")
    stage_i5_b = _read(OUT / "experiment_stageI5_modelzoo_summary_trackB.csv")
    stage_f2_a = _read(OUT / "experiment_stageF2_poisson_dc_covariates_summary_trackA.csv")
    stage_f2_b = _read(OUT / "experiment_stageF2_poisson_dc_covariates_summary_trackB.csv")

    # Track A candidates: best from I.5 and best from F2
    i5_a_best = stage_i5_a.sort_values("log_loss_mean").iloc[0]
    f2_a_best = stage_f2_a.sort_values("log_loss_mean").iloc[0]
    track_a = pd.DataFrame(
        [
            {
                "source": "stageI5",
                "candidate_id": "A1",
                "model": str(i5_a_best["model"]),
                "feature_pack": str(i5_a_best["feature_pack"]),
                "variant": "",
                "log_loss_mean": float(i5_a_best["log_loss_mean"]),
                "brier_mean": float(i5_a_best["brier_mean"]),
                "role": "primary_ml_candidate",
            },
            {
                "source": "stageF2",
                "candidate_id": "A2",
                "model": str(f2_a_best["model"]),
                "feature_pack": str(f2_a_best["feature_pack"]),
                "variant": str(f2_a_best["variant"]),
                "log_loss_mean": float(f2_a_best["log_loss_mean"]),
                "brier_mean": float(f2_a_best["brier_mean"]),
                "role": "statistical_control_candidate",
            },
        ]
    )

    # Track B candidates: bookmaker baseline + best ML from I.5 + best Poisson/DC from F2
    i5_b_ml = stage_i5_b[stage_i5_b["model"] != "bookmaker_implied"].sort_values("log_loss_mean").iloc[0]
    f2_b_nonbm = stage_f2_b[stage_f2_b["model"] != "bookmaker_implied"].sort_values("log_loss_mean").iloc[0]
    bm = stage_i5_b[stage_i5_b["model"] == "bookmaker_implied"].iloc[0]
    track_b = pd.DataFrame(
        [
            {
                "source": "stageI5",
                "candidate_id": "B0",
                "model": "bookmaker_implied",
                "feature_pack": str(bm["feature_pack"]),
                "variant": "",
                "log_loss_mean": float(bm["log_loss_mean"]),
                "brier_mean": float(bm["brier_mean"]),
                "role": "market_benchmark",
            },
            {
                "source": "stageI5",
                "candidate_id": "B1",
                "model": str(i5_b_ml["model"]),
                "feature_pack": str(i5_b_ml["feature_pack"]),
                "variant": "",
                "log_loss_mean": float(i5_b_ml["log_loss_mean"]),
                "brier_mean": float(i5_b_ml["brier_mean"]),
                "role": "primary_ml_candidate",
            },
            {
                "source": "stageF2",
                "candidate_id": "B2",
                "model": str(f2_b_nonbm["model"]),
                "feature_pack": str(f2_b_nonbm["feature_pack"]),
                "variant": str(f2_b_nonbm["variant"]),
                "log_loss_mean": float(f2_b_nonbm["log_loss_mean"]),
                "brier_mean": float(f2_b_nonbm["brier_mean"]),
                "role": "statistical_control_candidate",
            },
        ]
    ).sort_values("log_loss_mean").reset_index(drop=True)

    out_a = OUT / "day1_final_scoreboard_trackA.csv"
    out_b = OUT / "day1_final_scoreboard_trackB.csv"
    out_scope = OUT / "day1_final_scope_decisions.csv"

    track_a.to_csv(out_a, index=False)
    track_b.to_csv(out_b, index=False)

    scope = pd.DataFrame(
        [
            {
                "track": "A",
                "primary_finalist": "A1",
                "secondary_finalist": "A2",
                "decision_note": "Primary = najlepszy model ML (I.5); Secondary = najlepszy Poisson/DC (F2) jako kontrola statystyczna.",
            },
            {
                "track": "B",
                "primary_finalist": "B1",
                "secondary_finalist": "B2",
                "decision_note": "Benchmark obowiazkowy: B0 bookmaker_implied; porownanie modeli B1 i B2 vs B0.",
            },
        ]
    )
    scope.to_csv(out_scope, index=False)

    print("Saved:")
    print(f"- {out_a}")
    print(f"- {out_b}")
    print(f"- {out_scope}")


if __name__ == "__main__":
    main()
