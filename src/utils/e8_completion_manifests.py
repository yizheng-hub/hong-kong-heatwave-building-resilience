"""Compile approved lockbox and residential-mechanism manifests.

Compilation is separated from execution.  Lockbox compilation is only callable
after an immutable prediction receipt and a bound approval have been verified.
All physics identities use final IDF/EPW/schedule/engine bytes.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import time
from typing import Any

import pandas as pd

from utils.e8_completion_common import (
    ALL_EVENTS, LOCKBOX_EVENTS, SCENARIOS, Blocked, atomic_csv, atomic_json, sha256,
)
from utils.e8_g3_closure import physical_content_hash
from utils.e8_g3_runner import _compile_case, load_production


REFERENCE_IDS = {
    "res_public_rental": "B0001",
    "res_subsidised_sale": "B0002",
    "res_private_highrise": "B0003",
    "res_old_lowrise_tenement": "B0004",
    "res_village_house": "B0005",
}

SOURCE_INPUTS = ("model.idf", "weather.epw", "schedules.csv", "event_schedules.parquet",
                 "weather.parquet", "input_manifest.json")
SOURCE_PARSED = ("native_timestep_building.parquet", "native_timestep_zones.parquet", "parser_audit.json")


def _portable_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _resolve_source_directory(upstream: Path, source: dict[str, Any]) -> Path | None:
    declared = str(source.get("source_result_dir", ""))
    if declared:
        candidate = Path(declared)
        if not candidate.is_absolute():
            candidate = upstream / candidate
        if candidate.is_dir():
            return candidate
    event_id = str(source.get("event_id", ""))
    run_ids = []
    for key in ("resolved_run_id", "source_run_id", "run_id"):
        value = str(source.get(key, ""))
        if value and value not in run_ids:
            run_ids.append(value)
    roots = [
        upstream / "intermediate/simulation_outputs/upstream/E8_open6/events" / event_id / "02_runs/g3_results",
        upstream / "intermediate/simulation_outputs/upstream/G4a_local/02_runs/g3_results",
        upstream / "intermediate/simulation_outputs/upstream/02_runs/g3_results",
        upstream / "intermediate/simulation_outputs/completion/lockbox/02_runs/g3_results",
        upstream / "intermediate/simulation_outputs/completion/mechanisms/02_runs/g3_results",
    ]
    # Identifiers are ordered from the current resolved run to historical cache
    # identities.  Resolve one identifier at a time so a valid current run is not
    # made ambiguous merely because the same physical input also exists in G3.
    for run_id in run_ids:
        unique = []
        for root in roots:
            match = root / run_id
            if match.is_dir() and match.resolve() not in unique:
                unique.append(match.resolve())
        if len(unique) > 1:
            raise Blocked(f"multiple current source directories match {run_id}: {unique}")
        if unique:
            return unique[0]
    return None


def hydrate_source_index(
    frame: pd.DataFrame,
    *,
    result_root: Path | None = None,
    project_root: Path | None = None,
) -> pd.DataFrame:
    """Resolve a lightweight logical index to complete, byte-verified read-only evidence."""
    required = {"run_id", "event_id", "base_case_id", "prototype_id", "floor_position", "scenario_id", "physics_hash"}
    if missing := sorted(required - set(frame.columns)):
        raise Blocked(f"source index missing identity columns: {missing}")
    rows = []
    for source in frame.fillna("").to_dict("records"):
        directory = Path(str(source.get("source_result_dir", ""))) if source.get("source_result_dir") else None
        if directory is not None and not directory.is_absolute() and project_root is not None:
            directory = Path(project_root) / directory
        if (directory is None or not directory.exists()) and result_root is not None:
            resolved = str(source.get("resolved_run_id") or source.get("source_run_id") or source["run_id"])
            directory = Path(result_root) / "02_runs/g3_results" / resolved
        if (directory is None or not directory.exists()) and project_root is not None:
            directory = _resolve_source_directory(Path(project_root), source)
        if directory is None or not directory.exists():
            raise Blocked(f"source result directory missing for {source['run_id']}")
        missing_files = [str(directory / part / name) for part, names in
                         (("inputs", SOURCE_INPUTS), ("parsed", SOURCE_PARSED)) for name in names
                         if not (directory / part / name).is_file()]
        if missing_files:
            raise Blocked(f"source evidence incomplete for {source['run_id']}: {missing_files}")
        manifest_path = directory / "inputs/input_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for field in ("event_id", "prototype_id", "floor_position", "scenario_id"):
            if str(manifest.get(field)) != str(source[field]):
                raise Blocked(f"source identity mismatch for {source['run_id']}: {field}")
        declared = manifest.get("file_byte_sha256", {})
        for key, name in (("idf", "model.idf"), ("epw", "weather.epw"), ("schedule", "schedules.csv"),
                          ("event_schedule", "event_schedules.parquet")):
            if declared.get(key) != sha256(directory / "inputs" / name):
                raise Blocked(f"source byte hash mismatch for {source['run_id']}: {name}")
        rows.append({**source, "source_result_dir": str(directory.resolve()),
                     "source_input_manifest": str(manifest_path.resolve()),
                     "source_input_manifest_sha256": sha256(manifest_path),
                     "outage_start_hkt": manifest.get("outage_start_hkt"),
                     "outage_end_hkt": manifest.get("outage_end_hkt"),
                     "zone_areas_json": manifest.get("zone_areas_json"),
                     "conditioned_zones_json": manifest.get("conditioned_zones_json"),
                     "prototype_overrides": json.dumps(manifest.get("prototype_overrides", {}), sort_keys=True),
                     "inputs_path": str((directory / "inputs").resolve()),
                     "parsed_path": str((directory / "parsed").resolve())})
    result = pd.DataFrame(rows)
    if result.duplicated(["event_id", "base_case_id", "prototype_id", "floor_position", "scenario_id"]).any():
        raise Blocked("hydrated source index has duplicate logical identities")
    return result


def effective_open6_source_index(upstream: Path) -> pd.DataFrame:
    """Return the current Open6 logical index with every physical source resolved.

    The combined Open6 index intentionally leaves newly solved result directories
    blank and retains the original hash for a run later superseded by an audited
    C70 repair. Completion consumers need an immutable, explicit index instead
    of either convention.
    """
    path = upstream / "intermediate/simulation_outputs/upstream/E8_open6/combined/open6_logical_physical_source_index.csv"
    if not path.exists():
        raise Blocked("Open6 source index is not available")
    frame = pd.read_csv(path).fillna("")
    if "source_result_dir" not in frame:
        frame["source_result_dir"] = ""
    for index, row in frame.iterrows():
        directory = _resolve_source_directory(upstream, row.to_dict())
        if directory is None:
            raise Blocked(f"Open6 source result directory missing for {row.run_id}")
        frame.at[index, "source_result_dir"] = _portable_path(directory, upstream)
    for replacement in _discover_c70_supersessions(upstream, frame):
        mask = frame.run_id.astype(str).eq(replacement["run_id"])
        if int(mask.sum()) != 1:
            raise Blocked("C70 supersession receipt does not identify exactly one Open6 logical run")
        if str(frame.loc[mask, "physics_hash"].iloc[0]) != replacement["supersedes_physics_hash"]:
            raise Blocked("C70 supersession receipt does not match the frozen logical source hash")
        frame.loc[mask, "physics_hash"] = replacement["replacement_physics_hash"]
        frame.loc[mask, "source_result_dir"] = replacement["source_result_dir"]
        frame.loc[mask, "disposition"] = "physical_replacement_receipt"
    return frame


def _upstream_source_map(upstream: Path) -> dict[str, dict[str, Any]]:
    frame = hydrate_source_index(effective_open6_source_index(upstream), project_root=upstream)
    result: dict[str, dict[str, Any]] = {}
    for row in frame.to_dict("records"):
        result.setdefault(str(row["physics_hash"]), row)
    return result


def _discover_c70_supersessions(
    upstream: Path,
    source_index: pd.DataFrame | None = None,
) -> list[dict[str, str]]:
    """Resolve every archived C70 run to its current byte-verified replacement."""
    if source_index is None:
        index_path = upstream / "intermediate/simulation_outputs/upstream/E8_open6/combined/open6_logical_physical_source_index.csv"
        if not index_path.is_file():
            raise Blocked("Open6 source index is not available")
        source_index = pd.read_csv(index_path).fillna("")
    archive = upstream / "intermediate/simulation_outputs/archive/c70_superseded"
    if not archive.is_dir():
        return []
    replacements: list[dict[str, str]] = []
    for run_dir in sorted(path for path in archive.iterdir() if path.is_dir()):
        old_versions = sorted(path for path in run_dir.iterdir() if path.is_dir())
        if len(old_versions) != 1:
            raise Blocked(f"C70 supersession archive must contain exactly one old version: {run_dir.name}")
        matches = source_index[source_index.run_id.astype(str).eq(run_dir.name)]
        if len(matches) != 1:
            raise Blocked(f"C70 supersession is not unique in the Open6 source index: {run_dir.name}")
        row = matches.iloc[0]
        old_hash = old_versions[0].name
        if str(row.physics_hash) != old_hash:
            raise Blocked(f"C70 archived hash does not match frozen source index: {run_dir.name}")
        current = _resolve_source_directory(upstream, row.to_dict())
        if current is None:
            raise Blocked(f"C70 current result directory is missing: {run_dir.name}")
        manifest_path = current / "inputs/input_manifest.json"
        if not manifest_path.is_file():
            raise Blocked(f"C70 current input manifest is missing: {run_dir.name}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if str(row.scenario_id) != "HW_C70" or str(manifest.get("scenario_id")) != "HW_C70":
            raise Blocked(f"C70 supersession is not an HW_C70 run: {run_dir.name}")
        new_hash = str(manifest.get("physics_hash", ""))
        byte_identity = physical_content_hash(
            pd.Series({"file_byte_sha256": manifest.get("file_byte_sha256", {})})
        )
        if not new_hash or byte_identity != new_hash:
            raise Blocked(f"C70 replacement byte-level physics hash mismatch: {run_dir.name}")
        implementation = str(manifest.get("capacity_control_implementation", ""))
        if "targeted_EMS_supply_mass_flow_feedback" not in implementation:
            raise Blocked(f"C70 replacement does not declare the reviewed in-solver control: {run_dir.name}")
        replacements.append({
            "run_id": run_dir.name,
            "event_id": str(row.event_id),
            "base_case_id": str(row.base_case_id),
            "prototype_id": str(row.prototype_id),
            "floor_position": str(row.floor_position),
            "scenario_id": "HW_C70",
            "supersedes_physics_hash": old_hash,
            "replacement_physics_hash": new_hash,
            "superseded_result_dir": _portable_path(old_versions[0], upstream),
            "source_result_dir": _portable_path(current, upstream),
            "capacity_control_implementation": implementation,
        })
    return replacements


def audit_v13_migration(upstream: Path, output_dir: Path, candidate_lock_sha256: str) -> dict[str, Any]:
    """Audit the v1.2 -> v1.3 supersession without accepting it for the user."""
    version = upstream / "config/open_event_version_lock.json"
    repair = upstream / "intermediate/simulation_outputs/upstream/E8_open6/c70_capacity_resolution"
    plan_path = repair / "replacement_plan.json"; report_path = repair / "C70_TOTAL_CAPACITY_REPAIR_REPORT.md"
    issues = []
    if not version.is_file() or sha256(version) != candidate_lock_sha256:
        issues.append("candidate v1.3 version-lock hash is absent or different")
    if not plan_path.is_file() or not report_path.is_file():
        issues.append("C70 replacement plan or repair report is missing"); plan = {}
    else:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    rows = []
    if plan:
        run_id = str(plan["run_id"]); old_hash = str(plan["supersedes_physics_hash"]); new_hash = str(plan["replacement_physics_hash"])
        current = upstream / "intermediate/simulation_outputs/upstream/E8_open6/events/HKO_LI_0049/02_runs/g3_results" / run_id
        old = upstream / "intermediate/simulation_outputs/archive/c70_superseded" / run_id / old_hash
        for version_id, source_root, expected in (("v1.2_superseded", old, old_hash), ("v1.3_effective", current, new_hash)):
            manifest = source_root / "inputs/input_manifest.json"
            manifest_data = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else {}
            byte_identity = None
            if manifest_data.get("file_byte_sha256"):
                byte_identity = physical_content_hash(pd.Series({"file_byte_sha256": manifest_data["file_byte_sha256"]}))
            if byte_identity != expected:
                issues.append(f"{version_id} byte-level physics hash mismatch")
            if version_id == "v1.3_effective" and manifest_data.get("physics_hash") != expected:
                issues.append("v1.3 effective manifest does not declare the replacement physics hash")
            for relative in ("inputs/model.idf", "inputs/weather.epw", "inputs/schedules.csv", "inputs/event_schedules.parquet",
                             "raw/eplusout.csv", "raw/eplusout.err", "parsed/native_timestep_building.parquet",
                             "parsed/native_timestep_zones.parquet", "parsed/parser_audit.json"):
                evidence = source_root / relative
                rows.append({"run_id": run_id, "evidence_version": version_id, "physics_hash": expected,
                             "relative_path": relative, "path": str(evidence), "exists": evidence.is_file(),
                             "bytes": evidence.stat().st_size if evidence.is_file() else None,
                             "sha256": sha256(evidence) if evidence.is_file() else ""})
                if not evidence.is_file(): issues.append(f"{version_id} evidence missing: {relative}")
        event_root = upstream / "intermediate/simulation_outputs/upstream/E8_open6/events/HKO_LI_0049"
        group_root = event_root / "groups/HKO_LI_0049/B0127"
        state_path = event_root / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
        group_id = "HKO_LI_0049|B0127|com_office|top"
        receipt = state.get("groups", {}).get(group_id, {})
        replacement_member = receipt.get("dependencies", {}).get("members", {}).get(run_id, {})
        if receipt.get("status") != "metrics_complete": issues.append("affected B0127 group is not metrics_complete")
        if replacement_member.get("physics_hash") != new_hash:
            issues.append("affected B0127 dependency receipt does not use replacement physics")
        for name in ("energy.csv", "thermal.csv", "scores.csv", "qc.csv"):
            matches = [group_root / name] if (group_root / name).is_file() else []
            rows.append({"run_id": run_id, "evidence_version": "v1.3_group_dependency", "physics_hash": new_hash,
                         "relative_path": name, "path": str(matches[0]) if len(matches) == 1 else "",
                         "exists": len(matches) == 1, "bytes": matches[0].stat().st_size if len(matches) == 1 else None,
                         "sha256": sha256(matches[0]) if len(matches) == 1 else ""})
            if len(matches) != 1:
                issues.append(f"affected B0127 group evidence is missing: {name}")
            elif receipt.get("hashes", {}).get(str(matches[0])) != sha256(matches[0]):
                issues.append(f"affected B0127 group receipt hash mismatch: {name}")
        rows.append({"run_id": run_id, "evidence_version": "v1.3_group_dependency", "physics_hash": new_hash,
                     "relative_path": "state.json#groups/" + group_id, "path": str(state_path),
                     "exists": bool(receipt), "bytes": state_path.stat().st_size if state_path.is_file() else None,
                     "sha256": sha256(state_path) if state_path.is_file() else ""})
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(output_dir / "v1_3_physical_evidence_crosswalk.csv", pd.DataFrame(rows))
    result = {"migration": "v1.2_to_v1.3_targeted_C70_total_capacity", "evidence_complete": not issues,
              "scientific_acceptance": "pending_user_confirmation", "candidate_version_lock_sha256": candidate_lock_sha256,
              "replacement_plan_sha256": sha256(plan_path) if plan_path.is_file() else None,
              "repair_report_sha256": sha256(report_path) if report_path.is_file() else None,
              "issues": issues, "rows": len(rows), "does_not_modify_upstream": True}
    atomic_json(output_dir / "v1_3_migration_audit.json", result)
    return result


def audit_targeted_c70_migration(
    upstream: Path,
    output_dir: Path,
    candidate_lock_sha256: str,
    expected_replacements: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Audit every reviewed C70 physical supersession and its current group dependency."""
    issues: list[str] = []
    version = upstream / "config/open_event_version_lock.json"
    if not version.is_file() or sha256(version) != candidate_lock_sha256:
        issues.append("candidate Open6 version-lock hash is absent or different")
    try:
        replacements = _discover_c70_supersessions(upstream)
    except Blocked as exc:
        replacements = []
        issues.append(str(exc))
    expected = expected_replacements or []
    identity_fields = (
        "run_id", "event_id", "base_case_id",
        "supersedes_physics_hash", "replacement_physics_hash",
    )
    if expected:
        actual_identity = {tuple(item[field] for field in identity_fields) for item in replacements}
        expected_identity = {tuple(str(item[field]) for field in identity_fields) for item in expected}
        if actual_identity != expected_identity:
            issues.append("discovered C70 supersessions differ from the frozen expected replacement set")
    elif replacements:
        issues.append("C70 supersessions were discovered but no frozen expected replacement set was supplied")

    rows: list[dict[str, Any]] = []
    for item in replacements:
        run_id = item["run_id"]
        old_hash = item["supersedes_physics_hash"]
        new_hash = item["replacement_physics_hash"]
        current = Path(item["source_result_dir"])
        old = Path(item["superseded_result_dir"])
        for evidence_version, source_root, expected_hash in (
            ("superseded", old, old_hash),
            ("effective", current, new_hash),
        ):
            manifest_path = source_root / "inputs/input_manifest.json"
            manifest = (
                json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.is_file() else {}
            )
            byte_identity = None
            if manifest.get("file_byte_sha256"):
                byte_identity = physical_content_hash(
                    pd.Series({"file_byte_sha256": manifest["file_byte_sha256"]})
                )
            if byte_identity != expected_hash:
                issues.append(f"{evidence_version} byte-level physics hash mismatch: {run_id}")
            if evidence_version == "effective" and manifest.get("physics_hash") != expected_hash:
                issues.append(f"effective manifest does not declare replacement physics hash: {run_id}")
            for relative in (
                "inputs/model.idf", "inputs/weather.epw", "inputs/schedules.csv",
                "inputs/event_schedules.parquet", "raw/eplusout.csv", "raw/eplusout.err",
                "parsed/native_timestep_building.parquet",
                "parsed/native_timestep_zones.parquet", "parsed/parser_audit.json",
            ):
                evidence = source_root / relative
                rows.append({
                    "run_id": run_id,
                    "event_id": item["event_id"],
                    "base_case_id": item["base_case_id"],
                    "evidence_version": evidence_version,
                    "physics_hash": expected_hash,
                    "relative_path": relative,
                    "path": str(evidence),
                    "exists": evidence.is_file(),
                    "bytes": evidence.stat().st_size if evidence.is_file() else None,
                    "sha256": sha256(evidence) if evidence.is_file() else "",
                })
                if not evidence.is_file():
                    issues.append(f"{evidence_version} evidence missing for {run_id}: {relative}")

        event_id = item["event_id"]
        base_case_id = item["base_case_id"]
        group_id = "|".join((
            event_id, base_case_id, item["prototype_id"], item["floor_position"],
        ))
        event_root = upstream / "intermediate/simulation_outputs/upstream/E8_open6/events" / event_id
        group_root = event_root / "groups" / event_id / base_case_id
        state_path = event_root / "state.json"
        state = (
            json.loads(state_path.read_text(encoding="utf-8"))
            if state_path.is_file() else {}
        )
        receipt = state.get("groups", {}).get(group_id, {})
        member = receipt.get("dependencies", {}).get("members", {}).get(run_id, {})
        if receipt.get("status") != "metrics_complete":
            issues.append(f"affected group is not metrics_complete: {group_id}")
        if member.get("physics_hash") != new_hash:
            issues.append(f"affected group dependency does not use replacement physics: {group_id}")
        for name in ("energy.csv", "thermal.csv", "scores.csv", "qc.csv"):
            evidence = group_root / name
            exists = evidence.is_file()
            digest = sha256(evidence) if exists else ""
            rows.append({
                "run_id": run_id,
                "event_id": event_id,
                "base_case_id": base_case_id,
                "evidence_version": "group_dependency",
                "physics_hash": new_hash,
                "relative_path": name,
                "path": str(evidence),
                "exists": exists,
                "bytes": evidence.stat().st_size if exists else None,
                "sha256": digest,
            })
            if not exists:
                issues.append(f"affected group evidence is missing for {group_id}: {name}")
            elif receipt.get("hashes", {}).get(str(evidence)) != digest:
                issues.append(f"affected group receipt hash mismatch for {group_id}: {name}")
        qc_path = group_root / "qc.csv"
        if qc_path.is_file():
            qc = pd.read_csv(qc_path)
            for audit_name in (
                "C70_declared_capacity", "C70_zone_capacity",
                "C70_idf_total_capacity_input",
            ):
                audit_rows = qc[qc.audit.astype(str).eq(audit_name)]
                passed = (
                    len(audit_rows) == 1
                    and str(audit_rows.iloc[0]["pass"]).strip().lower() == "true"
                )
                if not passed:
                    issues.append(f"replacement group has not passed {audit_name}: {group_id}")

    combined_qc = upstream / "intermediate/simulation_outputs/upstream/E8_open6/combined/open6_physics_qc.csv"
    if not combined_qc.is_file():
        issues.append("final Open6 combined physics QC table is missing")
    else:
        qc = pd.read_csv(combined_qc)
        for audit_name in ("C70_declared_capacity", "C70_zone_capacity"):
            audit_rows = qc[qc.audit.astype(str).eq(audit_name)]
            pass_values = audit_rows["pass"].astype(str).str.strip().str.lower().eq("true")
            if len(audit_rows) != 1782 or not pass_values.all():
                issues.append(f"final Open6 {audit_name} coverage/pass check failed")

    output_dir.mkdir(parents=True, exist_ok=True)
    crosswalk = output_dir / "targeted_c70_physical_evidence_crosswalk.csv"
    registry = output_dir / "targeted_c70_supersession_registry.csv"
    atomic_csv(crosswalk, pd.DataFrame(rows))
    atomic_csv(registry, pd.DataFrame(replacements))
    result = {
        "migration": "reviewed_targeted_C70_total_capacity_supersessions",
        "evidence_complete": not issues,
        "scientific_acceptance": "pending_user_confirmation",
        "candidate_version_lock_sha256": candidate_lock_sha256,
        "replacement_count": len(replacements),
        "replacements": replacements,
        "crosswalk_sha256": sha256(crosswalk),
        "replacement_registry_sha256": sha256(registry),
        "issues": issues,
        "rows": len(rows),
        "does_not_modify_upstream": True,
    }
    atomic_json(output_dir / "targeted_c70_migration_audit.json", result)
    return result


def _frozen_production(workspace: Path, upstream: Path):
    module = load_production(upstream, code_root=workspace)
    loaded = Path(module.__file__).resolve()
    if workspace not in loaded.parents:
        raise Blocked(f"Production model was not imported from isolated workspace: {loaded}")
    return module


def _decorate(row: dict[str, Any], workspace: Path, upstream: Path, role: str) -> dict[str, Any]:
    row = dict(row)
    content = physical_content_hash(pd.Series(row))
    row.update(
        legacy_physics_hash=row["physics_hash"],
        physics_hash=content,
        physics_hash_definition="sha256 final IDF EPW schedule engine bytes",
        run_id="e8completion_" + content[:20],
        role=role,
        raw_output_schema_hash=json.loads((upstream / "intermediate/simulation_outputs/upstream/00_protocol/output_schema.json").read_text(encoding="utf-8"))["output_schema_hash"],
        output_extraction_version="closure-support-v1",
        output_extraction_hash=sha256(workspace / "src/utils/e8_g3_runner.py"),
        analysis_hash=sha256(workspace / "src/utils/e8_g3_postprocess.py"),
        approved_to_run=True,
    )
    return row


def _assert_release(prediction_receipt: Path, approval: dict[str, Any]) -> None:
    if not prediction_receipt.exists():
        raise Blocked("Lockbox prediction receipt must exist before compilation")
    receipt = json.loads(prediction_receipt.read_text(encoding="utf-8"))
    if receipt.get("status") != "FROZEN_BEFORE_LOCKBOX" or set(receipt.get("event_ids", [])) != set(LOCKBOX_EVENTS):
        raise Blocked("Lockbox prediction receipt is not valid")
    if approval.get("lockbox_release") != "released_after_prediction_freeze":
        raise Blocked("Approval has not released both lockboxes after prediction freeze")
    for label in ("prediction", "cohorts", "baselines", "rules", "upstream_receipt"):
        path = Path(receipt.get(label + "_path", ""))
        if not path.is_file() or sha256(path) != receipt.get(label + "_sha256"):
            raise Blocked(f"lockbox release receipt does not bind intact {label}")
    receipt_code_lock = receipt.get("approval_contract_sha256")
    approved_code_lock = approval.get("completion_code_lock_sha256")
    if receipt_code_lock != approved_code_lock:
        amendment = approval.get("resource_only_amendment", {})
        approved_handoff = (
            amendment.get("status") == "approved"
            and amendment.get("scope") == "resource_limits_only"
            and amendment.get("parent_handoff_code_change_approved") is True
            and amendment.get("prior_completion_code_lock_sha256") == receipt_code_lock
            and amendment.get("new_completion_code_lock_sha256") == approved_code_lock
            and amendment.get("scientific_contract_changed") is False
            and amendment.get("lockbox_prediction_receipt_changed") is False
        )
        if not approved_handoff:
            raise Blocked("lockbox receipt is not bound to the approved completion code")


def compile_lockboxes(c: dict[str, Any], approval: dict[str, Any], prediction_receipt: Path) -> pd.DataFrame:
    _assert_release(prediction_receipt, approval)
    workspace = Path(c["workspace_root"])
    upstream = Path(c["upstream_project_root"])
    out = Path(c["results_root"]) / "00_protocol/compiled_inputs/lockbox"
    delivery = Path(c["results_root"]) / "00_protocol/lockbox_manifests"
    production = _frozen_production(workspace, upstream)
    design = pd.read_csv(upstream / "intermediate/foundation_pipeline/02_building_response/thermal_experiment_design.csv")
    source_map = _upstream_source_map(upstream)
    rows = []
    for event_id in LOCKBOX_EVENTS:
        for design_row in design.to_dict("records"):
            series = pd.Series(design_row)
            for scenario in SCENARIOS:
                compiled = _compile_case(
                    upstream, upstream / "intermediate/simulation_outputs/upstream", production, series, event_id, scenario, 14,
                    "lockbox_core", compile_output_root=out / event_id,
                    manifest_extras={"lockbox": True, "score_contract": "e8-completion-analysis-v1"},
                )
                row = _decorate(compiled, workspace, upstream, "core_group_" + str(series.base_case_id))
                cached = source_map.get(row["physics_hash"])
                row.update(
                    disposition="reuse_complete" if cached else "new",
                    source_batch=cached.get("source_batch", "") if cached else "",
                    source_run_id=cached.get("run_id", "") if cached else "",
                    source_result_dir=cached.get("source_result_dir", "") if cached else "",
                    lockbox=True,
                )
                rows.append(row)
    frame = pd.DataFrame(rows)
    if len(frame) != 2970 or frame.groupby(["event_id", "base_case_id"]).ngroups != 594:
        raise Blocked("Lockbox manifest cardinality is not 2,970 runs / 594 groups")
    duplicate = frame.duplicated("physics_hash", keep=False) & ~frame.disposition.eq("reuse_complete")
    if duplicate.any():
        raise Blocked("Unexpected within-lockbox physical duplicate requires explicit aliasing")
    delivery.mkdir(parents=True, exist_ok=True)
    atomic_csv(delivery / "lockbox_run_manifest.csv", frame)
    for event_id in LOCKBOX_EVENTS:
        atomic_csv(delivery / f"{event_id}.csv", frame.loc[frame.event_id.eq(event_id)])
    summary = {"status": "COMPILED_NOT_EXECUTED", "events": list(LOCKBOX_EVENTS), "runs": len(frame),
               "groups": 594, "new": int(frame.disposition.eq("new").sum()),
               "reuse": int(frame.disposition.eq("reuse_complete").sum()),
               "manifest_sha256": sha256(delivery / "lockbox_run_manifest.csv"),
               "prediction_receipt_sha256": sha256(prediction_receipt), "energyplus_calls": 0}
    atomic_json(delivery / "lockbox_manifest_summary.json", summary)
    return frame


def _mechanism_variants(levels: dict[str, Any], prototypes: pd.DataFrame) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    indexed = prototypes.set_index("prototype_id")
    for prototype_id, reference_id in REFERENCE_IDS.items():
        p = indexed.loc[prototype_id]
        variants.extend([
            {"prototype_id": prototype_id, "reference_design_id": reference_id,
             "factor": "roof_top_exposure", "level": floor, "reference": floor == "middle",
             "design_updates": {"floor_position": floor}, "prototype_overrides": {}}
            for floor in levels["factors"]["roof_top_exposure"]["levels"]
        ])
        for factor, field in (("thermal_mass", "thermal_capacitance_kj_m2k"),
                              ("nominal_cooling_capacity", "cooling_capacity_w_m2")):
            reference = float(p[field])
            for multiplier in levels["factors"][factor]["multipliers_of_prototype_reference"]:
                variants.append({"prototype_id": prototype_id, "reference_design_id": reference_id,
                                 "factor": factor, "level": f"x{float(multiplier):.3f}",
                                 "reference": float(multiplier) == 1.0, "design_updates": {},
                                 "prototype_overrides": {field: reference * float(multiplier)}})
        for offset in levels["factors"]["AC_operating_time"]["levels_h"]:
            variants.append({"prototype_id": prototype_id, "reference_design_id": reference_id,
                             "factor": "AC_operating_time", "level": f"{int(offset):+d}h",
                             "reference": int(offset) == 0,
                             "design_updates": {"start_offset_h": int(offset)}, "prototype_overrides": {}})
    return variants


_CAPACITY_LIMITED_SCENARIOS = frozenset({"HW_RATED", "HW_C70", "HW_O24_SYNC"})


def _validate_mechanism_physical_difference(factor: str, scenario: str, changed: list[str]) -> bool:
    """Validate the one-factor byte difference and return scenario applicability.

    Nominal cooling capacity is intentionally absent from the two unlimited
    controls. Those logical comparisons therefore alias identical physical
    inputs; the capacity treatment must change the IDF in every capacity-
    limited branch.
    """
    changed_set = set(changed)
    if factor == "nominal_cooling_capacity":
        applicable = scenario in _CAPACITY_LIMITED_SCENARIOS
        if applicable and "idf" not in changed_set:
            raise Blocked(f"{factor} did not change final IDF bytes for {scenario}")
        if not applicable and changed_set:
            raise Blocked(f"{factor} unexpectedly changed unlimited-control bytes for {scenario}")
        return applicable
    if factor in {"thermal_mass", "roof_top_exposure"}:
        if "idf" not in changed_set:
            raise Blocked(f"{factor} did not change final IDF bytes")
        return True
    if factor == "AC_operating_time":
        if not {"schedule", "event_schedule"}.intersection(changed_set):
            raise Blocked("AC operating-time factor did not change schedule bytes")
        return True
    raise Blocked(f"Unknown mechanism factor: {factor}")


def compile_mechanisms(c: dict[str, Any], approval: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if approval.get("mechanism_levels_status") != "approved":
        raise Blocked("Mechanism levels are not approved")
    workspace = Path(c["workspace_root"])
    upstream = Path(c["upstream_project_root"])
    levels_path = workspace / "config/mechanism_levels_recommended.json"
    levels = json.loads(levels_path.read_text(encoding="utf-8"))
    if sha256(levels_path) != approval.get("mechanism_levels_sha256"):
        raise Blocked("Approved mechanism level hash differs")
    production = _frozen_production(workspace, upstream)
    prototypes = production.prototype_parameter_table()
    design = pd.read_csv(upstream / "intermediate/foundation_pipeline/02_building_response/thermal_experiment_design.csv")
    source_map = _upstream_source_map(upstream)
    out = Path(c["results_root"]) / "00_protocol/compiled_inputs/mechanisms"
    delivery = Path(c["results_root"]) / "02_mechanisms/manifests"
    logical_rows = []
    for variant in _mechanism_variants(levels, prototypes):
        base = design.loc[design.base_case_id.eq(variant["reference_design_id"])].iloc[0].copy()
        for field, value in variant["design_updates"].items():
            base[field] = value
        mechanism_id = f"{variant['prototype_id']}__{variant['factor']}__{variant['level']}"
        for event_id in levels["events"]:
            for scenario in SCENARIOS:
                compiled = _compile_case(
                    upstream, upstream / "intermediate/simulation_outputs/upstream", production, base, event_id, scenario, 14,
                    "mechanism_physical", compile_output_root=out / event_id,
                    prototype_overrides=variant["prototype_overrides"],
                    manifest_extras={"mechanism_id": mechanism_id, "factor": variant["factor"],
                                     "level": variant["level"], "reference": variant["reference"]},
                )
                row = _decorate(compiled, workspace, upstream, "mechanism_physical")
                row.update(mechanism_id=mechanism_id, base_case_id=mechanism_id,
                           factor=variant["factor"], level=variant["level"],
                           reference=variant["reference"], logical_base_case_id=mechanism_id,
                           reference_design_id=variant["reference_design_id"], lockbox=False)
                logical_rows.append(row)
    logical = pd.DataFrame(logical_rows)
    if len(logical) != 825 or logical.groupby(["mechanism_id", "event_id"]).ngroups != 165:
        raise Blocked("Mechanism logical cardinality is not 825 / 165 groups")
    difference_rows = []
    for (prototype, factor, event, scenario), group in logical.groupby(
            ["prototype_id", "factor", "event_id", "scenario_id"], sort=False):
        reference = group[group.reference]
        if len(reference) != 1:
            raise Blocked(f"mechanism reference is not unique: {prototype} {factor} {event} {scenario}")
        ref = reference.iloc[0]
        ref_hashes = ref.file_byte_sha256
        if isinstance(ref_hashes, str):
            ref_hashes = json.loads(ref_hashes.replace("'", '"'))
        for _, treatment in group[~group.reference].iterrows():
            hashes = treatment.file_byte_sha256
            if isinstance(hashes, str):
                hashes = json.loads(hashes.replace("'", '"'))
            changed = sorted(key for key in ("idf", "epw", "schedule", "event_schedule")
                             if hashes.get(key) != ref_hashes.get(key))
            if hashes.get("epw") != ref_hashes.get("epw"):
                raise Blocked("one-factor mechanism comparison changed weather bytes")
            applicable = _validate_mechanism_physical_difference(factor, scenario, changed)
            difference_rows.append({"prototype_id": prototype, "factor": factor, "event_id": event,
                                    "scenario_id": scenario, "reference_level": ref.level,
                                    "treatment_level": treatment.level, "changed_physical_objects": "|".join(changed),
                                    "factor_applicable_to_scenario": applicable,
                                    "expected_physical_change": "idf_or_schedule" if applicable else "none_physical_alias",
                                    "reference_physics_hash": ref.physics_hash,
                                    "treatment_physics_hash": treatment.physics_hash})
    physical_rows, resolved = [], []
    seen: dict[str, dict[str, Any]] = {}
    for row in logical.to_dict("records"):
        content = str(row["physics_hash"])
        cached = source_map.get(content)
        if cached:
            source = {"resolved_run_id": cached["run_id"], "source_batch": cached.get("source_batch", "Open6"),
                      "source_result_dir": cached["source_result_dir"], "disposition": "reuse_complete"}
        elif content in seen:
            primary = seen[content]
            source = {"resolved_run_id": primary["run_id"], "source_batch": "E8_completion_mechanisms",
                      "source_result_dir": "", "disposition": "alias_pending_primary"}
        else:
            primary = dict(row)
            primary.update(disposition="new", source_batch="", source_run_id="", source_result_dir="")
            seen[content] = primary
            physical_rows.append(primary)
            source = {"resolved_run_id": primary["run_id"], "source_batch": "E8_completion_mechanisms",
                      "source_result_dir": "", "disposition": "new"}
        resolved.append({**row, **source})
    physical = pd.DataFrame(physical_rows)
    logical = pd.DataFrame(resolved)
    delivery.mkdir(parents=True, exist_ok=True)
    atomic_csv(delivery / "mechanism_logical_map.csv", logical)
    atomic_csv(delivery / "mechanism_physical_run_manifest.csv", physical)
    diff = logical[["mechanism_id", "prototype_id", "factor", "level", "reference",
                    "event_id", "scenario_id", "reference_design_id", "physics_hash",
                    "resolved_run_id", "disposition"]].copy()
    diff["single_changed_object"] = diff.factor.map({
        "roof_top_exposure": "floor/roof boundary role",
        "thermal_mass": "InternalMass thickness; envelope U fixed",
        "nominal_cooling_capacity": "IdealLoads design capacity in RATED/C70/O24; unlimited controls are physical aliases; COP fixed",
        "AC_operating_time": "AC availability schedule offset only",
    })
    atomic_csv(delivery / "mechanism_pair_difference_contract.csv", diff)
    atomic_csv(delivery / "mechanism_actual_input_difference_audit.csv", pd.DataFrame(difference_rows))
    summary = {"status": "COMPILED_NOT_EXECUTED", "logical_rows": len(logical),
               "logical_groups": int(logical.groupby(["mechanism_id", "event_id"]).ngroups),
               "unique_physical": int(logical.physics_hash.nunique()), "new_physical": len(physical),
               "reused_or_aliased": len(logical) - len(physical),
               "physical_manifest_sha256": sha256(delivery / "mechanism_physical_run_manifest.csv"),
               "logical_map_sha256": sha256(delivery / "mechanism_logical_map.csv"),
               "energyplus_calls": 0, "generated_at": time.time()}
    atomic_json(delivery / "mechanism_manifest_summary.json", summary)
    return logical, physical
