"""Run the complete solver-free analysis from preserved simulation evidence.

This entry point never compiles or launches EnergyPlus.  It consumes the
verified Open6, lockbox and mechanism run directories already present under
``intermediate/simulation_outputs`` and writes every derived analysis table to
a new, initially empty result directory.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import pandas as pd

from utils.e8_completion_analysis import mechanism_effects
from utils.e8_completion_common import atomic_csv, atomic_json, load_config, sha256
from utils.e8_completion_execution import (
    _collect_group_tables,
    _complete_analysis,
    _copy_open6,
    _freeze_open6_stock,
    _protocol,
    summarize,
)
from utils.e8_completion_release import write_dependency_matrices


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def require_empty(directory: Path) -> None:
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError(f"analysis output directory is not empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)


def stage_lockbox_protocol(config: dict, approval: dict) -> None:
    root = Path(config["results_root"])
    protocol = Path(config["protocol_input_root"])
    validation = root / "04_validation"
    validation.mkdir(parents=True, exist_ok=True)
    names = (
        "lockbox_predictions_frozen.csv",
        "lockbox_cohorts_frozen.csv",
        "lockbox_baselines_frozen.csv",
    )
    receipt_path = root / "00_protocol/lockbox_prediction_receipt.json"
    rules = root / "00_protocol/lockbox_prediction_rules.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        expected = {
            "prediction": validation / names[0],
            "cohorts": validation / names[1],
            "baselines": validation / names[2],
            "rules": rules,
            "upstream_receipt": root / "00_protocol/upstream_receipt.json",
        }
        if receipt.get("status") != "FROZEN_BEFORE_LOCKBOX":
            raise RuntimeError("existing lockbox prediction receipt has an invalid status")
        for label, path in expected.items():
            if not path.is_file() or sha256(path) != receipt.get(f"{label}_sha256"):
                raise RuntimeError(f"existing frozen lockbox {label} changed")
        if receipt.get("analysis_contract_sha256") != sha256(
            Path(config["workspace_root"]) / "config/completion_analysis_contract.json"
        ):
            raise RuntimeError("analysis contract changed after the lockbox freeze")
        return
    for name in names:
        link_or_copy(protocol / "lockbox" / name, validation / name)
    link_or_copy(protocol / "lockbox/lockbox_prediction_rules.json", rules)
    receipt = {
        "status": "FROZEN_BEFORE_LOCKBOX",
        "event_ids": ["HKO_LI_0054", "HKO_LI_0066"],
        "created_at": time.time(),
        "prediction_path": str(validation / names[0]),
        "prediction_sha256": sha256(validation / names[0]),
        "cohorts_path": str(validation / names[1]),
        "cohorts_sha256": sha256(validation / names[1]),
        "baselines_path": str(validation / names[2]),
        "baselines_sha256": sha256(validation / names[2]),
        "rules_path": str(rules),
        "rules_sha256": sha256(rules),
        "upstream_receipt_path": str(root / "00_protocol/upstream_receipt.json"),
        "upstream_receipt_sha256": sha256(root / "00_protocol/upstream_receipt.json"),
        "analysis_contract_sha256": sha256(
            Path(config["workspace_root"]) / "config/completion_analysis_contract.json"
        ),
        "approval_contract_sha256": approval["completion_code_lock_sha256"],
        "first_event_cannot_update_second": True,
        "not_historical_forward_prediction": True,
        "recovery_identity": (
            "byte-identical frozen pre-lockbox protocol input recovered from the accepted package"
        ),
    }
    atomic_json(receipt_path, receipt)


def stage_lockbox_manifest(config: dict) -> None:
    root = Path(config["results_root"])
    project_root = Path(config["workspace_root"])
    source = Path(config["lockbox_runtime_root"]) / "run_manifest.csv"
    destination = root / "00_protocol/lockbox_manifests/lockbox_run_manifest.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(source).fillna("")
    if len(frame) != 2_970:
        raise RuntimeError(f"lockbox manifest has {len(frame):,} rows, expected 2,970")
    compiled_root = project_root / "intermediate/compiled_simulation_inputs/lockbox"
    compiled_names = frame["compiled_dir"].map(lambda value: Path(str(value)).name)
    compiled_dirs = [compiled_root / event / name for event, name in zip(frame.event_id, compiled_names)]
    missing = [path for path in compiled_dirs if not path.is_dir()]
    if missing:
        raise RuntimeError(f"relocated compiled lockbox input is missing: {missing[0]}")
    relative_dirs = [os.path.relpath(path, project_root).replace("\\", "/") for path in compiled_dirs]
    frame["compiled_dir"] = relative_dirs
    for column, filename in (
        ("idf_path", "model.idf"),
        ("epw_path", "weather.epw"),
        ("schedule_path", "schedules.csv"),
        ("event_schedule_path", "event_schedules.parquet"),
        ("weather_path", "weather.parquet"),
    ):
        frame[column] = [f"{directory}/{filename}" for directory in relative_dirs]
    frame["source_result_dir"] = frame["run_id"].map(
        lambda run_id: os.path.relpath(
            Path(config["lockbox_runtime_root"]) / "02_runs/g3_results" / str(run_id),
            project_root,
        ).replace("\\", "/")
    )
    atomic_csv(destination, frame)


def materialize_existing_mechanisms(config: dict) -> dict:
    root = Path(config["results_root"])
    protocol = Path(config["protocol_input_root"]) / "mechanisms"
    manifests = root / "02_mechanisms/manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    for source in protocol.iterdir():
        if source.is_file():
            link_or_copy(source, manifests / source.name)

    tables = _collect_group_tables(
        [Path(config["mechanism_runtime_root"]) / "groups"],
        root / "02_mechanisms/group_tables",
    )
    merged = tables["energy"].merge(
        tables["thermal"],
        on=["event_id", "base_case_id", "prototype_id", "floor_position"],
        suffixes=("", "_thermal"),
        validate="one_to_one",
    )
    logical = pd.read_csv(manifests / "mechanism_logical_map.csv").fillna("")
    effects = mechanism_effects(merged, logical)
    atomic_csv(root / "02_mechanisms/controlled_pair_results.csv", effects)
    summary = effects.groupby(
        ["factor", "prototype_id", "level"], as_index=False
    ).median(numeric_only=True)
    atomic_csv(root / "02_mechanisms/controlled_effect_summary.csv", summary)
    bounds = [
        column
        for column in effects
        if column.startswith("effect_")
        and (column.endswith("_lower_h") or column.endswith("_upper_h"))
    ]
    paired = effects.groupby(
        ["event_id", "prototype_id", "factor", "level"], as_index=False
    ).agg(actual_pair_count=("mechanism_id", "size"))
    for column in bounds:
        values = (
            effects.groupby(["event_id", "prototype_id", "factor", "level"])[column]
            .median()
            .reset_index(name=column)
        )
        paired = paired.merge(
            values,
            on=["event_id", "prototype_id", "factor", "level"],
            validate="one_to_one",
        )
    atomic_csv(
        root / "02_mechanisms/residential_paired_event_prototype_effects.csv", paired
    )
    return {"groups": len(merged), "effects": len(effects)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config/analysis.json",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted solver-free run after validating completed stage outputs.",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.output_dir:
        output = args.output_dir
        if not output.is_absolute():
            output = Path(config["workspace_root"]) / output
        config["results_root"] = str(output.resolve())
    root = Path(config["results_root"])
    if args.resume:
        if not root.is_dir() or not any(root.iterdir()):
            raise RuntimeError("--resume requires a non-empty interrupted analysis directory")
    else:
        require_empty(root)

    executable = Path(config["energyplus"])
    solver_before = {
        "path": executable.relative_to(Path(config["workspace_root"])).as_posix(),
        "size": executable.stat().st_size,
        "mtime_ns": executable.stat().st_mtime_ns,
        "sha256": sha256(executable),
    }
    started = time.time()
    _protocol(config)
    approval = json.loads(
        (Path(config["protocol_input_root"]) / "analysis_approval.json").read_text(
            encoding="utf-8"
        )
    )
    receipt = {
        "status": "qualified_preserved_simulation_evidence",
        "simulation_outputs_root": os.path.relpath(
            config["simulation_outputs_root"], config["workspace_root"]
        ).replace("\\", "/"),
        "energyplus_calls": 0,
    }
    open6_required = [
        root / f"01_core_e8/open6_{name}.csv"
        for name in ("energy", "thermal", "scores", "source_index")
    ]
    if not (args.resume and all(path.is_file() for path in open6_required)):
        _copy_open6(config, receipt)
    stock_required = [
        root / "03_stock/open6_prediction_targets.parquet",
        *[
            root / f"03_stock/open6_prediction_events/{event}.parquet"
            for event in config["upstream_open_events"]
        ],
        *[
            root / f"03_stock/open6_recovery_trajectories/{event}_prototype_floor_recovery.parquet"
            for event in config["upstream_open_events"]
        ],
    ]
    if not (args.resume and all(path.is_file() for path in stock_required)):
        _freeze_open6_stock(config)
    stage_lockbox_protocol(config, approval)
    stage_lockbox_manifest(config)
    summarize(config)
    mechanism = materialize_existing_mechanisms(config)
    write_dependency_matrices(
        root,
        Path(config["workspace_root"]),
        Path(config["paper_panel_registry"]),
    )
    _complete_analysis(config, approval)

    solver_after = {
        "path": solver_before["path"],
        "size": executable.stat().st_size,
        "mtime_ns": executable.stat().st_mtime_ns,
        "sha256": sha256(executable),
    }
    if solver_before != solver_after:
        raise RuntimeError("EnergyPlus executable changed during solver-free analysis")
    audit = {
        "status": "complete",
        "started_at_epoch": started,
        "finished_at_epoch": time.time(),
        "energyplus_calls": 0,
        "resumed_after_interruption": bool(args.resume),
        "solver_before": solver_before,
        "solver_after": solver_after,
        "mechanism": mechanism,
        "output_root": os.path.relpath(root, config["workspace_root"]).replace("\\", "/"),
    }
    atomic_json(root / "00_protocol/solver_free_analysis_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
