"""Recover immutable protocol inputs from the accepted analysis package.

Only artifacts that were frozen before outcome evaluation, plus compile-time
identity manifests, are recovered here.  Scientific outcome tables are not
used as analysis inputs.  All path fields are normalized to project-relative
form or cleared when the historical location no longer exists.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd


PATH_COLUMNS = {
    "compiled_dir",
    "idf_path",
    "epw_path",
    "schedule_path",
    "event_schedule_path",
    "weather_path",
    "source_result_dir",
}


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def normalize_manifest(source: Path, destination: Path) -> None:
    frame = pd.read_csv(source).fillna("")
    for column in PATH_COLUMNS & set(frame.columns):
        frame[column] = ""
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, encoding="utf-8-sig")


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-root", type=Path, default=root / "result/accepted")
    parser.add_argument("--output-dir", type=Path, default=root / "intermediate/protocol")
    args = parser.parse_args(argv)
    accepted = args.accepted_root.resolve()
    output = args.output_dir.resolve()

    lockbox = output / "lockbox"
    for name in (
        "lockbox_predictions_frozen.csv",
        "lockbox_cohorts_frozen.csv",
        "lockbox_baselines_frozen.csv",
    ):
        link_or_copy(accepted / "04_validation" / name, lockbox / name)
    link_or_copy(
        root / "docs/audit/production_protocol/lockbox_prediction_rules.json",
        lockbox / "lockbox_prediction_rules.json",
    )

    mechanism_input = accepted / "02_mechanisms/manifests"
    mechanism_output = output / "mechanisms"
    for source in mechanism_input.iterdir():
        if not source.is_file():
            continue
        destination = mechanism_output / source.name
        if source.suffix.lower() == ".csv":
            normalize_manifest(source, destination)
        else:
            link_or_copy(source, destination)

    approved = json.loads(
        (root / "docs/audit/production_protocol/original_run_configs/completion_approval.user.json").read_text(
            encoding="utf-8"
        )
    )
    portable_approval = {
        "approval_status": approved["approval_status"],
        "approved_at": approved["approved_at"],
        "approval_scope": approved["approval_scope"],
        "completion_code_lock_sha256": approved["completion_code_lock_sha256"],
        "future_building_results_decision": approved["future_building_results_decision"],
        "lockbox_release": approved["lockbox_release"],
        "mechanism_levels_status": approved["mechanism_levels_status"],
        "physical_inputs_changed": False,
        "solver_calls_authorized_for_reanalysis": 0,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis_approval.json").write_text(
        json.dumps(portable_approval, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    compatibility = pd.read_csv(
        accepted / "05_evidence/validation_evidence_compatibility.csv"
    )
    base_columns = [
        "validation_layer",
        "dataset",
        "role",
        "status",
        "primary_metric",
        "value",
        "claim_boundary",
    ]
    evidence = compatibility[base_columns].copy()
    evidence_dir = output / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence.to_csv(
        evidence_dir / "validation_evidence_matrix.csv",
        index=False,
        encoding="utf-8-sig",
    )
    registry = pd.read_csv(root / "config/external_evidence_registry.csv").fillna("")
    for row in registry.to_dict("records"):
        if not row["source_id"]:
            continue
        match = evidence[evidence["dataset"].astype(str).eq(str(row["dataset_match"]))]
        snapshot = match.copy()
        snapshot.insert(0, "source_id", row["source_id"])
        snapshot.to_csv(
            evidence_dir / f"{row['source_id']}.csv",
            index=False,
            encoding="utf-8-sig",
        )
    provenance = {
        "status": "protocol_inputs_recovered",
        "lockbox_identity": "byte-identical pre-first-look files",
        "mechanism_identity": "compile-time maps with historical absolute path fields removed",
        "evidence_identity": "frozen claim ledger; provider inputs remain under data/",
        "scientific_outcomes_used_as_inputs": False,
    }
    (output / "RECOVERY_PROVENANCE.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(provenance, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
