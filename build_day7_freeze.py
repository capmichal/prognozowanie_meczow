from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


REQUIRED_FINAL_FILES = [
    "outputs/final_summary_trackA.csv",
    "outputs/final_summary_trackB.csv",
    "outputs/final_significance.csv",
    "outputs/final_report_manifest.csv",
    "outputs/final_model_cards.csv",
    "outputs/final_decision_log.csv",
    "outputs/final_predict_setup.csv",
]

REQUIRED_DOC_FILES = [
    "FINAL_README.md",
    "day0_szkielet_pracy_i_mapa_drogi.md",
    "day1_final_scoreboard_i_scope.md",
    "day2_final_significance.md",
    "day3_final_pipeline_report.md",
    "day4_predict_mode_i_app_final.md",
    "day5_final_docs.md",
    "day6_slajdy_i_narracja.md",
    "day6_notatka_dzienna.md",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_final_both() -> None:
    subprocess.run(["python", str(ROOT / "run_final_pipeline.py"), "--mode", "both"], check=True, cwd=str(ROOT))


def _build_manifest(paths: list[str], category: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    now = datetime.now(timezone.utc).isoformat()
    for rel in paths:
        p = ROOT / rel
        exists = p.exists()
        rows.append(
            {
                "category": category,
                "path": rel,
                "exists": bool(exists),
                "size_bytes": int(p.stat().st_size) if exists else None,
                "sha256": _sha256(p) if exists else None,
                "checked_at_utc": now,
            }
        )
    return rows


def _sanity_checks() -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    a = pd.read_csv(OUT / "final_summary_trackA.csv")
    b = pd.read_csv(OUT / "final_summary_trackB.csv")
    s = pd.read_csv(OUT / "final_significance.csv")

    add("trackA_non_empty", len(a) > 0, f"rows={len(a)}")
    add("trackB_non_empty", len(b) > 0, f"rows={len(b)}")
    add("significance_non_empty", len(s) > 0, f"rows={len(s)}")
    add("trackA_has_A1", bool((a["candidate_id"] == "A1").any()), "candidate A1 present")
    add("trackB_has_B0", bool((b["candidate_id"] == "B0").any()), "candidate B0 present")
    add(
        "trackB_benchmark_rank1",
        bool(b.loc[b["candidate_id"] == "B0", "rank_by_log_loss"].iloc[0] == 1),
        "B0 rank_by_log_loss == 1",
    )
    add(
        "significance_B1_vs_B0",
        bool(
            (
                (s["track"] == "B")
                & (s["left"] == "B1")
                & (s["right"] == "B0")
                & (s["decision_logloss"] == "right_significantly_better")
            ).any()
        ),
        "B1 vs B0 decision as expected",
    )

    return checks


def main() -> None:
    OUT.mkdir(exist_ok=True)

    _run_final_both()

    manifest_rows = _build_manifest(REQUIRED_FINAL_FILES, "final_artifact")
    manifest_rows += _build_manifest(REQUIRED_DOC_FILES, "documentation")
    manifest = pd.DataFrame(manifest_rows)

    checks = pd.DataFrame(_sanity_checks())

    out_manifest = OUT / "day7_freeze_manifest.csv"
    out_checks = OUT / "day7_freeze_checks.csv"
    out_report = OUT / "day7_freeze_report.csv"

    manifest.to_csv(out_manifest, index=False)
    checks.to_csv(out_checks, index=False)

    report = pd.DataFrame(
        [
            {
                "status": "PASS" if bool(manifest["exists"].all() and checks["ok"].all()) else "FAIL",
                "artifacts_checked": int((manifest["category"] == "final_artifact").sum()),
                "docs_checked": int((manifest["category"] == "documentation").sum()),
                "all_files_exist": bool(manifest["exists"].all()),
                "all_checks_ok": bool(checks["ok"].all()),
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        ]
    )
    report.to_csv(out_report, index=False)

    print("Saved freeze outputs:")
    print(f"- {out_manifest}")
    print(f"- {out_checks}")
    print(f"- {out_report}")
    print("\nFreeze status:")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
