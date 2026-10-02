"""Audit and narrowly replace C70 runs that exceed their declared total-cooling cap.

The frozen logical manifest is not rewritten.  A corrected physical input may
supersede one logical run only after a real EnergyPlus run, exact parsing, and
capacity QC.  Unaffected scenarios and unaffected C70 runs are never solved.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from utils import e8_g3_postprocess as science
from utils import g4a_local_execution as core
from utils.e8_g3_closure import physical_content_hash
from utils.e8_g3_runner import _c70_total_capacity_ems, load_production


DEFAULT_CONFIG = ROOT / "intermediate/simulation_outputs/upstream/E8_open6/event_configs/HKO_LI_0049.json"
RESOLUTION = ROOT / "intermediate/simulation_outputs/upstream/E8_open6/c70_capacity_resolution"
SUPERSEDED_ARCHIVE = ROOT / "intermediate/simulation_outputs/archive/c70_superseded"
RATE_TOLERANCE_W = 1e-6


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def input_hashes(row: dict, compiled: Path, engine: Path) -> dict[str, str]:
    return {
        "idf": core.digest(compiled / "model.idf"),
        "epw": core.digest(compiled / "weather.epw"),
        "schedule": core.digest(compiled / "schedules.csv"),
        "event_schedule": core.digest(compiled / "event_schedules.parquet"),
        "engine": core.digest(engine),
    }


def capacity_audit(c: dict, state: dict) -> pd.DataFrame:
    """Audit every completed/parsed C70 result; missing evidence is not a pass."""
    rows: list[dict[str, object]] = []
    assignment = pd.read_csv(Path(c["cache_root"]) / "01_inputs/E8_event_assignment.csv").set_index("parent_event_id")
    for run_id, item in state["rows"].items():
        row = core.effective_row(item)
        if row["scenario_id"] != "HW_C70" or item["status"] not in core.GOOD:
            continue
        root = Path(item["result_dir"])
        parsed = root / "parsed/native_timestep_zones.parquet"
        if not parsed.is_file():
            rows.append({"run_id": run_id, "zone_id": "", "audit_status": "missing_parsed"})
            continue
        zones = pd.read_parquet(parsed)
        zones["timestamp_hkt"] = pd.to_datetime(zones["timestamp_hkt"])
        event = assignment.loc[row["event_id"]]
        event_start = pd.Timestamp(event.event_start, tz="Asia/Hong_Kong")
        event_end = pd.Timestamp(event.event_end, tz="Asia/Hong_Kong")
        zones = zones.loc[science.formal_event_mask(zones, event_start, event_end)].copy()
        idf = (root / "inputs/model.idf").read_text(encoding="utf-8", errors="replace")
        caps = science.c70_idf_total_capacities(idf)
        expected = set(json.loads(row["conditioned_zones_json"]))
        if set(caps) != expected:
            rows.append({"run_id": run_id, "zone_id": "", "audit_status": "idf_zone_set_mismatch",
                         "expected_zones": ";".join(sorted(expected)), "idf_zones": ";".join(sorted(caps))})
            continue
        zones = zones.loc[zones["zone_id"].isin(expected)].copy()
        for zone_id in sorted(expected):
            part = zones.loc[zones["zone_id"].eq(zone_id)]
            if part.empty or not np.isfinite(part["cooling_total_j"].to_numpy(dtype=float)).all():
                rows.append({"run_id": run_id, "zone_id": zone_id, "audit_status": "missing_or_nonfinite"})
                continue
            rates = part["cooling_total_j"].to_numpy(dtype=float) / 900.0
            cap = float(caps[zone_id])
            peak_index = int(np.argmax(rates))
            rows.append({
                "run_id": run_id,
                "event_id": row["event_id"],
                "base_case_id": row["base_case_id"],
                "prototype_id": row["prototype_id"],
                "floor_position": row["floor_position"],
                "zone_id": zone_id,
                "declared_total_capacity_w": cap,
                "maximum_reported_total_cooling_rate_w": float(rates[peak_index]),
                "maximum_reported_sensible_cooling_rate_w": float(
                    (part["cooling_sensible_j"].to_numpy(dtype=float) / 900.0).max()
                ),
                "maximum_total_capacity_utilization": float(rates.max() / cap),
                "overrun_rows": int((rates > cap + RATE_TOLERANCE_W).sum()),
                "peak_timestamp_hkt": str(part.iloc[peak_index]["timestamp_hkt"]),
                "audit_status": "affected" if bool((rates > cap + RATE_TOLERANCE_W).any()) else "pass",
                "frozen_physics_hash": item["row"]["physics_hash"],
                "effective_physics_hash": row["physics_hash"],
            })
    return pd.DataFrame(rows)


def build_replacement(c: dict, item: dict, affected_zones: set[str]) -> dict:
    row = core.effective_row(item)
    source = Path(row["compiled_dir"])
    target = RESOLUTION / "corrected_inputs" / row["run_id"]
    target.mkdir(parents=True, exist_ok=True)
    for name in ("weather.epw", "schedules.csv", "event_schedules.parquet", "weather.parquet"):
        dest = target / name
        if not dest.exists():
            shutil.copy2(source / name, dest)
        elif core.digest(dest) != core.digest(source / name):
            raise core.Blocked(f"Existing correction input differs: {dest}")
    original_idf = (source / "model.idf").read_text(encoding="utf-8", errors="replace")
    production = load_production(Path(c["project_root"]))
    capacity = float(
        production.prototype_parameter_table().set_index("prototype_id")
        .loc[row["prototype_id"], "cooling_capacity_w_m2"]
    )
    corrected_idf = _c70_total_capacity_ems(
        original_idf,
        affected_zones,
        json.loads(row["zone_areas_json"]),
        capacity,
    )
    idf_path = target / "model.idf"
    if idf_path.exists() and idf_path.read_text(encoding="utf-8") != corrected_idf:
        raise core.Blocked(f"Existing corrected IDF differs: {idf_path}")
    if not idf_path.exists():
        temp = idf_path.with_suffix(".idf.writing")
        temp.write_text(corrected_idf, encoding="utf-8")
        os.replace(temp, idf_path)
    hashes = input_hashes(row, target, Path(c["energyplus"]))
    replacement_row = dict(row)
    replacement_row.update(
        compiled_dir=str(target),
        idf_path=str(idf_path),
        epw_path=str(target / "weather.epw"),
        schedule_path=str(target / "schedules.csv"),
        event_schedule_path=str(target / "event_schedules.parquet"),
        weather_path=str(target / "weather.parquet"),
        file_byte_sha256=hashes,
        capacity_control_implementation=(
            "native_Maximum_Total_Cooling_Capacity_plus_targeted_EMS_supply_mass_flow_"
            "feedback_for_observed_overrun_zones"
        ),
    )
    replacement_row["physics_hash"] = physical_content_hash(pd.Series(replacement_row))
    receipt = {
        "run_id": row["run_id"],
        "supersedes_physics_hash": item["row"]["physics_hash"],
        "replacement_physics_hash": replacement_row["physics_hash"],
        "affected_zones": sorted(affected_zones),
        "row_overrides": {
            key: replacement_row[key]
            for key in (
                "compiled_dir", "idf_path", "epw_path", "schedule_path",
                "event_schedule_path", "weather_path", "file_byte_sha256",
                "physics_hash", "capacity_control_implementation",
            )
        },
        "reason": "observed EnergyPlus 26.1 IdealLoads total-cooling output exceeded declared C70 total cap",
        "method": "inside-HVAC-iteration EMS mass-flow feedback only for observed-overrun zones",
        "created_at": time.time(),
    }
    core.atomic_json(target / "input_manifest.json", {**replacement_row, "physical_replacement": receipt})
    core.atomic_json(RESOLUTION / "replacement_plan.json", receipt)
    return receipt


def solve_and_parse(c: dict, item: dict, receipt: dict) -> tuple[Path, dict, dict]:
    row = dict(item["row"])
    row.update(receipt["row_overrides"])
    attempt_id = f"c70fix_{int(time.time())}"
    attempt = RESOLUTION / "production_attempts" / row["run_id"] / attempt_id
    inputs, raw = attempt / "inputs", attempt / "raw"
    raw.mkdir(parents=True, exist_ok=False)
    core.stage_inputs(c, row, inputs)
    stdout = (raw / "stdout.txt").open("wb")
    stderr = (raw / "stderr.txt").open("wb")
    command = [c["energyplus"], "-w", str(inputs / "weather.epw"), "-d", str(raw), str(inputs / "model.idf")]
    started = time.time()
    try:
        proc = subprocess.Popen(
            command, cwd=str(attempt), stdout=stdout, stderr=stderr,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        code = proc.wait(timeout=float(c["run_timeout_seconds"]))
    finally:
        stdout.close()
        stderr.close()
    launch = {
        "attempt_id": attempt_id, "run_id": row["run_id"], "is_retry": True,
        "physical_correction": True, "launched": True, "pid": proc.pid,
        "created": None, "host": core.socket.gethostname(), "started": started,
        "finished": time.time(), "returncode": code, "command": command,
        "raw_dir": str(raw), "reservation": "finished",
        "identity_status": "exited_owned_handle",
        "event_id": row["event_id"], "base_case_id": row["base_case_id"],
        "prototype_id": row["prototype_id"], "floor_position": row["floor_position"],
        "scenario_id": row["scenario_id"], "physics_hash": row["physics_hash"],
        "raw_output_schema_hash": row["raw_output_schema_hash"],
        "analysis_version": c["analysis_version"],
    }
    core.atomic_json(attempt / "launch_receipt.json", launch)
    if code != 0 or not core.raw_complete(raw):
        raise core.Blocked(f"Corrected EnergyPlus run failed: returncode={code}; raw_complete={core.raw_complete(raw)}")
    temp_output = RESOLUTION / "replacement_parse" / attempt_id
    temp_c = dict(c, output_dir=str(temp_output), hardlink_inputs=False, compact_attempt_after_qc=False)
    temp_item = {"status": "raw_complete", "raw_dir": str(raw), "raw_hashes": core.file_hashes(raw)}
    core.parse_run(temp_c, row, temp_item, load_production(Path(c["project_root"])))
    return core.science.result_dir(temp_output, row["run_id"]), temp_item, launch


def corrected_qc(c: dict, row: dict, result: Path, affected_zones: set[str]) -> pd.DataFrame:
    zones = pd.read_parquet(result / "parsed/native_timestep_zones.parquet")
    zones["timestamp_hkt"] = pd.to_datetime(zones["timestamp_hkt"])
    assignment = pd.read_csv(Path(c["cache_root"]) / "01_inputs/E8_event_assignment.csv").set_index("parent_event_id")
    event = assignment.loc[row["event_id"]]
    zones = zones.loc[science.formal_event_mask(
        zones,
        pd.Timestamp(event.event_start, tz="Asia/Hong_Kong"),
        pd.Timestamp(event.event_end, tz="Asia/Hong_Kong"),
    )].copy()
    caps = science.c70_idf_total_capacities((result / "inputs/model.idf").read_text(encoding="utf-8"))
    records = []
    for zone_id, cap in sorted(caps.items()):
        part = zones.loc[zones["zone_id"].eq(zone_id)]
        rates = part["cooling_total_j"].to_numpy(dtype=float) / 900.0
        records.append({
            "run_id": row["run_id"], "zone_id": zone_id,
            "targeted_for_correction": zone_id in affected_zones,
            "declared_total_capacity_w": cap,
            "maximum_reported_total_cooling_rate_w": float(rates.max()),
            "maximum_total_capacity_utilization": float(rates.max() / cap),
            "overrun_rows": int((rates > cap + RATE_TOLERANCE_W).sum()),
            "pass": bool((rates <= cap + RATE_TOLERANCE_W).all()),
        })
    result = pd.DataFrame(records)
    if not result["pass"].all():
        raise core.Blocked(f"Corrected C70 still exceeds declared total capacity: {result.loc[~result['pass']].to_dict('records')}")
    return result


def install_replacement(c: dict, state: dict, run_id: str, receipt: dict,
                        replacement_result: Path, parsed_item: dict, launch: dict) -> None:
    item = state["rows"][run_id]
    canonical = science.result_dir(Path(c["output_dir"]), run_id)
    archive = SUPERSEDED_ARCHIVE / run_id / item["row"]["physics_hash"]
    if archive.exists():
        raise core.Blocked(f"Superseded archive already exists; inspect before replacing: {archive}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    # Both locations are on the same volume.  The old evidence is renamed, not deleted.
    os.replace(canonical, archive)
    try:
        os.replace(replacement_result, canonical)
    except BaseException:
        os.replace(archive, canonical)
        raise
    launch["archived_original_result"] = str(archive)
    launch["installed_result"] = str(canonical)
    state["attempts"].append(launch)
    item["attempts"].append(launch["attempt_id"])
    item["physical_replacement"] = receipt
    item.update(
        status="qc_passed", error="", raw_dir=str(canonical / "raw"),
        raw_hashes=core.file_hashes(canonical / "raw"),
        parsed_hashes=core.file_hashes(canonical / "parsed"),
        parser_version=core.VERSION, qc="exact_parser_finite_fields_and_targeted_total_capacity_passed",
        result_dir=str(canonical),
    )
    state["groups"].pop(core.pair_key(item["row"]), None)
    state["stop_reason"] = "targeted_c70_physical_replacement_installed"
    state.setdefault("physical_replacement_receipts", []).append(receipt)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--execute", action="store_true",
                        help="Run only the C70 inputs proven affected by the completed-output audit.")
    args = parser.parse_args(argv)
    c = core.load_config(args.config)
    state = core.load_state(c)
    if not state:
        raise core.Blocked("Event state is missing")
    if state.get("active_session"):
        raise core.Blocked("Production session is active; correction cannot run concurrently")
    audit = capacity_audit(c, state)
    RESOLUTION.mkdir(parents=True, exist_ok=True)
    core.csv_write(RESOLUTION / "completed_c70_capacity_audit.csv", audit)
    incomplete = audit.loc[~audit["audit_status"].isin(["pass", "affected"])]
    if not incomplete.empty:
        raise core.Blocked(f"C70 audit has incomplete evidence: {incomplete.to_dict('records')}")
    affected = audit.loc[audit["audit_status"].eq("affected")]
    summary = {
        "completed_c70_runs_audited": int(audit["run_id"].nunique()),
        "zone_records_audited": int(len(audit)),
        "affected_runs": int(affected["run_id"].nunique()),
        "affected_zones": int(len(affected)),
        "execute_requested": bool(args.execute),
    }
    core.atomic_json(RESOLUTION / "audit_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not args.execute:
        return 0
    if affected.empty:
        print("No affected completed C70 input; no solver call made.")
        return 0
    production = load_production(Path(c["project_root"]))
    with core.exclusive_lock(Path(c["output_dir"]) / "execution.lock"):
        state = core.load_state(c)
        if not state or state.get("active_session"):
            raise core.Blocked("Production state changed or became active after the audit")
        version = core.verify_versions(c)
        core.reconcile_execution_identity(c, state, version)
        core.validate_state(state, pd.read_csv(c["manifest"]).fillna(""))
        for run_id, rows in affected.groupby("run_id"):
            item = state["rows"][run_id]
            if item.get("physical_replacement"):
                raise core.Blocked(f"Run already has a physical replacement: {run_id}")
            zones = set(rows["zone_id"])
            receipt = build_replacement(c, item, zones)
            replacement_result, parsed_item, launch = solve_and_parse(c, item, receipt)
            effective = dict(item["row"])
            effective.update(receipt["row_overrides"])
            qc = corrected_qc(c, effective, replacement_result, zones)
            core.csv_write(RESOLUTION / f"{run_id}_corrected_capacity_qc.csv", qc)
            install_replacement(c, state, run_id, receipt, replacement_result, parsed_item, launch)
        core.completed_groups(c, state, production)
        state["stop_reason"] = "targeted_c70_correction_complete_resume_ready"
        core.atomic_json(Path(c["output_dir"]) / "state.json", state)
        core.reports(c, state)
        core.atomic_json(Path(c["output_dir"]) / "state.json", state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
