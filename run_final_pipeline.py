from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def _run_python(script_name: str) -> None:
    script_path = ROOT / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")
    subprocess.run(["python", str(script_path)], check=True, cwd=str(ROOT))


def _ensure_day1_day2() -> None:
    req = [
        OUT / "day1_final_scoreboard_trackA.csv",
        OUT / "day1_final_scoreboard_trackB.csv",
        OUT / "day2_final_significance_summary.csv",
    ]
    missing = [p for p in req if not p.exists()]
    if not missing:
        return

    if not (OUT / "day1_final_scoreboard_trackA.csv").exists() or not (OUT / "day1_final_scoreboard_trackB.csv").exists():
        _run_python("build_day1_final_scoreboard.py")

    if not (OUT / "day2_final_significance_summary.csv").exists():
        _run_python("build_day2_final_significance.py")


def _annotate_track_a(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["track"] = "A"
    out["rank_by_log_loss"] = out["log_loss_mean"].rank(method="dense", ascending=True).astype(int)
    out["is_final_primary"] = out["candidate_id"].eq("A1")
    out["is_final_secondary"] = out["candidate_id"].eq("A2")
    out["narrative_label"] = np.where(
        out["candidate_id"].eq("A1"),
        "Primary ML finalist",
        np.where(out["candidate_id"].eq("A2"), "Statistical control finalist", "Other"),
    )
    return out.sort_values(["rank_by_log_loss", "candidate_id"]).reset_index(drop=True)


def _annotate_track_b(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["track"] = "B"
    out["rank_by_log_loss"] = out["log_loss_mean"].rank(method="dense", ascending=True).astype(int)
    out["is_market_benchmark"] = out["candidate_id"].eq("B0")
    out["is_final_primary"] = out["candidate_id"].eq("B1")
    out["is_final_secondary"] = out["candidate_id"].eq("B2")
    out["narrative_label"] = np.select(
        [out["candidate_id"].eq("B0"), out["candidate_id"].eq("B1"), out["candidate_id"].eq("B2")],
        ["Market benchmark", "Primary ML finalist", "Statistical control finalist"],
        default="Other",
    )
    return out.sort_values(["rank_by_log_loss", "candidate_id"]).reset_index(drop=True)


def _annotate_significance(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["significant_logloss_p05"] = out["p_value_sign_flip_log_loss"] < 0.05
    out["ci_excludes_zero_logloss"] = (out["delta_log_loss_ci95_lo"] > 0.0) | (out["delta_log_loss_ci95_hi"] < 0.0)
    out["left_better_logloss"] = out["delta_log_loss_left_minus_right"] < 0.0

    def _decision(row: pd.Series) -> str:
        if bool(row["significant_logloss_p05"]) and bool(row["left_better_logloss"]):
            return "left_significantly_better"
        if bool(row["significant_logloss_p05"]) and not bool(row["left_better_logloss"]):
            return "right_significantly_better"
        return "no_significant_difference"

    out["decision_logloss"] = out.apply(_decision, axis=1)
    return out


def run_report_mode() -> None:
    _ensure_day1_day2()

    day1_a = pd.read_csv(OUT / "day1_final_scoreboard_trackA.csv")
    day1_b = pd.read_csv(OUT / "day1_final_scoreboard_trackB.csv")
    day2_s = pd.read_csv(OUT / "day2_final_significance_summary.csv")

    final_a = _annotate_track_a(day1_a)
    final_b = _annotate_track_b(day1_b)
    final_sig = _annotate_significance(day2_s)

    out_a = OUT / "final_summary_trackA.csv"
    out_b = OUT / "final_summary_trackB.csv"
    out_sig = OUT / "final_significance.csv"
    out_manifest = OUT / "final_report_manifest.csv"

    final_a.to_csv(out_a, index=False)
    final_b.to_csv(out_b, index=False)
    final_sig.to_csv(out_sig, index=False)

    manifest = pd.DataFrame(
        [
            {
                "artifact": "final_summary_trackA",
                "path": str(out_a.relative_to(ROOT)),
                "source": "day1_final_scoreboard_trackA",
                "purpose": "Final Track A scoreboard for thesis tables",
            },
            {
                "artifact": "final_summary_trackB",
                "path": str(out_b.relative_to(ROOT)),
                "source": "day1_final_scoreboard_trackB",
                "purpose": "Final Track B scoreboard for thesis tables",
            },
            {
                "artifact": "final_significance",
                "path": str(out_sig.relative_to(ROOT)),
                "source": "day2_final_significance_summary",
                "purpose": "Final paired significance comparisons",
            },
        ]
    )
    manifest.to_csv(out_manifest, index=False)

    print("Saved final report artifacts:")
    print(f"- {out_a}")
    print(f"- {out_b}")
    print(f"- {out_sig}")
    print(f"- {out_manifest}")


def run_predict_mode() -> None:
    _ensure_day1_day2()

    final_a_path = OUT / "final_summary_trackA.csv"
    final_b_path = OUT / "final_summary_trackB.csv"
    final_sig_path = OUT / "final_significance.csv"
    if not final_a_path.exists() or not final_b_path.exists() or not final_sig_path.exists():
        run_report_mode()

    final_a = pd.read_csv(final_a_path)
    final_b = pd.read_csv(final_b_path)
    final_sig = pd.read_csv(final_sig_path)

    cards = pd.concat([final_a, final_b], ignore_index=True)
    cards["is_recommended_for_demo"] = cards["candidate_id"].isin(["A1", "B1", "B0"])

    decisions = final_sig[["track", "left", "right", "decision_logloss", "p_value_sign_flip_log_loss"]].copy()
    decisions = decisions.rename(columns={"p_value_sign_flip_log_loss": "p_value"})

    demo_setup = pd.DataFrame(
        [
            {
                "item": "primary_trackA_candidate",
                "value": "A1",
                "description": "Primary ML finalist for no-odds setting",
            },
            {
                "item": "primary_trackB_candidate",
                "value": "B1",
                "description": "Primary ML finalist for with-odds setting",
            },
            {
                "item": "trackB_benchmark",
                "value": "B0",
                "description": "Market benchmark (bookmaker implied)",
            },
            {
                "item": "demo_predictor_note",
                "value": "xg_proxy",
                "description": "Interactive app uses xG proxy for single-match demo and final tables for thesis evidence",
            },
        ]
    )

    out_cards = OUT / "final_model_cards.csv"
    out_decisions = OUT / "final_decision_log.csv"
    out_setup = OUT / "final_predict_setup.csv"
    cards.to_csv(out_cards, index=False)
    decisions.to_csv(out_decisions, index=False)
    demo_setup.to_csv(out_setup, index=False)

    print("Saved final predict artifacts:")
    print(f"- {out_cards}")
    print(f"- {out_decisions}")
    print(f"- {out_setup}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final project pipeline entrypoint")
    parser.add_argument(
        "--mode",
        choices=["report", "predict", "both"],
        default="report",
        help="Pipeline mode. Day 3 supports report fully.",
    )
    return parser.parse_args()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    args = parse_args()

    if args.mode in ("report", "both"):
        run_report_mode()
    if args.mode in ("predict", "both"):
        run_predict_mode()


if __name__ == "__main__":
    main()
