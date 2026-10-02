"""Compile and freeze the six non-lockbox E8 work-order manifests.

This module never launches EnergyPlus.  It reuses the production input compiler,
records exact cache provenance, and content-deduplicates only newly compiled files.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import pandas as pd

from utils.e8_g3_closure import physical_content_hash
from utils.e8_g3_runner import CORE_SCENARIO_IDS, _compile_case, load_production
from utils import e8_g3_postprocess as science


OPEN_EVENTS = (
    "HKO_LI_0049", "HKO_LI_0052", "HKO_LI_0058",
    "HKO_LI_0068", "HKO_LI_0085", "HKO_LI_0090",
)
LOCKBOX_EVENTS = frozenset({"HKO_LI_0054", "HKO_LI_0066"})
EXPECTED_DATES = {
    "HKO_LI_0049": ("2005-07-18", "2005-07-20"),
    "HKO_LI_0052": ("2009-08-30", "2009-09-08"),
    "HKO_LI_0054": ("2012-05-02", "2012-05-04"),
    "HKO_LI_0058": ("2014-09-28", "2014-09-30"),
    "HKO_LI_0066": ("2017-08-18", "2017-08-22"),
    "HKO_LI_0068": ("2018-05-14", "2018-06-02"),
    "HKO_LI_0085": ("2022-07-19", "2022-08-02"),
    "HKO_LI_0090": ("2023-07-22", "2023-07-28"),
}
PREHISTORY_DAYS = 14
ANALYSIS_VERSION = "G3_metrics_v2_closure-g4a-group-adapter-v1"
OUTPUT_EXTRACTION_VERSION = "closure-support-v1"
ALL_EVENTS = ("HKO_LI_0049", "HKO_LI_0052", "HKO_LI_0054", "HKO_LI_0058",
              "HKO_LI_0066", "HKO_LI_0068", "HKO_LI_0085", "HKO_LI_0090")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".writing")
    frame.to_csv(temp, index=False, encoding="utf-8-sig")
    os.replace(temp, path)


def write_json(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".writing")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(temp, path)


def _date(value) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def validate_scope(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignment = pd.read_csv(root / "config/events/E8_event_assignment.csv")
    indexed = assignment.set_index("parent_event_id")
    if not set(OPEN_EVENTS) <= set(indexed.index) or not LOCKBOX_EVENTS <= set(indexed.index):
        raise ValueError("Frozen E8 assignment lacks an approved open or lockbox identity")
    for event_id, (start, end) in EXPECTED_DATES.items():
        row = indexed.loc[event_id]
        if (_date(row.event_start), _date(row.event_end)) != (start, end):
            raise ValueError(f"Frozen event date conflict for {event_id}: {row.event_start}--{row.event_end}")
        if not bool(row.hourly_qualified):
            raise ValueError(f"Hourly weather is not qualified: {event_id}")
    design = pd.read_csv(root / "intermediate/foundation_pipeline/02_building_response/thermal_experiment_design.csv")
    if len(design) != 297 or design.base_case_id.nunique() != 297:
        raise ValueError("Frozen design is not 297 unique base cases")
    if design.groupby(["prototype_id", "floor_position"]).ngroups != 56:
        raise ValueError("Frozen design no longer covers 56 response units")
    return assignment, design


def _validate_event_release(events: tuple[str, ...], release_record: Path | None) -> None:
    selected_lockboxes = set(events) & LOCKBOX_EVENTS
    if not selected_lockboxes:
        return
    if tuple(events) != ALL_EVENTS or release_record is None or not release_record.exists():
        raise ValueError("Lockbox inputs require the full frozen E8 order and an explicit release record")
    release = json.loads(release_record.read_text(encoding="utf-8"))
    if release.get("status") != "released" or set(release.get("event_ids", [])) != LOCKBOX_EVENTS:
        raise ValueError("Lockbox release record is not valid for 0054/0066")


def logical_target(root: Path, events: tuple[str, ...] = OPEN_EVENTS,
                   release_record: Path | None = None) -> pd.DataFrame:
    _validate_event_release(tuple(events), release_record)
    assignment, design = validate_scope(root)
    rows = []
    for event_id in events:
        event = assignment.loc[assignment.parent_event_id.eq(event_id)].iloc[0]
        for d in design.to_dict("records"):
            for scenario in CORE_SCENARIO_IDS:
                rows.append({
                    "event_id": event_id, "event_start": event.event_start, "event_end": event.event_end,
                    "base_case_id": d["base_case_id"], "prototype_id": d["prototype_id"],
                    "floor_position": d["floor_position"], "scenario_id": scenario,
                    "prehistory_days": PREHISTORY_DAYS, "lockbox": False,
                })
    frame = pd.DataFrame(rows)
    if len(frame) != len(events) * 1485 or frame.groupby(["event_id", "base_case_id"]).ngroups != len(events) * 297:
        raise AssertionError("E8 logical target cardinality changed")
    if not (set(events) & LOCKBOX_EVENTS) and set(frame.event_id) & LOCKBOX_EVENTS:
        raise AssertionError("Lockbox leaked into open target")
    return frame


def _deduplicate_file(path: Path, objects: Path, known_hash: str | None = None) -> str:
    value = known_hash or sha256(path)
    target = objects / value[:2] / value
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256(target) != value:
            raise ValueError(f"Content store corruption: {target}")
        path.unlink()
        try:
            os.link(target, path)
        except OSError:
            shutil.copy2(target, path)
    else:
        try:
            os.link(path, target)
        except OSError:
            shutil.copy2(path, target)
    if sha256(path) != value:
        raise ValueError(f"Deduplicated input changed: {path}")
    return value


def _g4a_source(root: Path) -> tuple[dict[tuple[str, str, str], dict], dict]:
    manifest = pd.read_csv(root / "intermediate/simulation_outputs/upstream/G4a_local/run_manifest.csv").fillna("")
    state = json.loads((root / "intermediate/simulation_outputs/upstream/G4a_local/state.json").read_text(encoding="utf-8"))
    if len(manifest) != 280 or any(item["status"] != "metrics_complete" for item in state["rows"].values()):
        raise ValueError("G4-a is not a complete 280-run cache")
    verification = state.get("last_verification", {})
    if not verification.get("complete"):
        raise ValueError("G4-a current evidence verification is not complete")
    return {(r.event_id, r.base_case_id, r.scenario_id): r._asdict() for r in manifest.itertuples(index=False)}, state


def _g3_content_cache(root: Path) -> dict[str, dict]:
    path = root / "intermediate/simulation_outputs/upstream/02_runs/g3_manifest_revised.csv"
    if not path.exists():
        return {}
    cache = {}
    for row in pd.read_csv(path).fillna("").to_dict("records"):
        if int(row.get("prehistory_days", -1)) != PREHISTORY_DAYS or not str(row.get("role", "")).startswith("core_group"):
            continue
        result = science.result_dir(root / "intermediate/simulation_outputs/upstream", row["run_id"])
        result_json = result / "run_result.json"
        raw = result / "raw/eplusout.csv"
        if not result_json.exists() or not raw.exists():
            continue
        if json.loads(result_json.read_text(encoding="utf-8")).get("status") != "passed":
            continue
        cache.setdefault(physical_content_hash(pd.Series(row)), row)
    return cache


def compile_open6(root: Path, delivery: Path, *, progress_every: int = 50,
                  events: tuple[str, ...] = OPEN_EVENTS,
                  release_record: Path | None = None) -> pd.DataFrame:
    root, delivery = root.resolve(), delivery.resolve()
    delivery.mkdir(parents=True, exist_ok=True)
    events = tuple(events)
    target = logical_target(root, events, release_record)
    write_csv(target, delivery / "open6_logical_target.csv")
    g4a, _ = _g4a_source(root)
    g3_cache = _g3_content_cache(root)
    production = load_production(root)
    schema = json.loads((root / "intermediate/simulation_outputs/upstream/00_protocol/output_schema.json").read_text(encoding="utf-8"))["output_schema_hash"]
    compile_root = delivery / "compiled_inputs"
    objects = delivery / "compiled_objects"
    rows = []
    started = time.perf_counter()
    design = pd.read_csv(root / "intermediate/foundation_pipeline/02_building_response/thermal_experiment_design.csv")
    total_runs = len(events) * 1485
    for event_id in events:
        for _, design_row in design.iterrows():
            for scenario in CORE_SCENARIO_IDS:
                key = (event_id, str(design_row.base_case_id), scenario)
                if key in g4a:
                    row = dict(g4a[key])
                    row.update(disposition="reuse_complete", reason="verified_G4a_complete_exact_logical_and_physical_identity",
                               source_batch="G4a_0049_local_v1", source_run_id=row["run_id"],
                               source_result_dir=str(science.result_dir(root / "intermediate/simulation_outputs/upstream/G4a_local", row["run_id"])),
                               approved_to_run=False)
                else:
                    before = time.perf_counter()
                    row = _compile_case(root, root / "intermediate/simulation_outputs/upstream", production, design_row, event_id,
                                        scenario, PREHISTORY_DAYS, "open6_core",
                                        compile_output_root=compile_root / event_id)
                    hashes = row["file_byte_sha256"] if isinstance(row["file_byte_sha256"], dict) else ast.literal_eval(row["file_byte_sha256"])
                    for field, filename in (("idf", "model.idf"), ("epw", "weather.epw"),
                                            ("schedule", "schedules.csv"), ("event_schedule", "event_schedules.parquet")):
                        _deduplicate_file(Path(row["compiled_dir"]) / filename, objects, hashes[field])
                    for filename in ("weather.parquet", "input_manifest.json"):
                        _deduplicate_file(Path(row["compiled_dir"]) / filename, objects)
                    content = physical_content_hash(pd.Series(row))
                    cached = g3_cache.get(content)
                    logical = hashlib.sha256(f"{event_id}|{design_row.base_case_id}|{scenario}|{content}".encode()).hexdigest()
                    row.update(legacy_g3_physics_hash=row["physics_hash"], physics_hash=content,
                               physics_hash_definition="sha256 canonical JSON of final IDF EPW schedule engine byte hashes",
                               run_id="e8open6_" + logical[:20], cached_g3_run_id=cached["run_id"] if cached else "",
                               disposition="reextract" if cached else "new",
                               reason="exact_physical_bytes_and_complete_retained_G3_raw" if cached else "no_qualified_content_identical_cache",
                               source_batch="G3_metrics_v2_closure" if cached else "", source_run_id=cached["run_id"] if cached else "",
                               source_result_dir=str(science.result_dir(root / "intermediate/simulation_outputs/upstream", cached["run_id"])) if cached else "",
                               compile_seconds=time.perf_counter() - before, approved_to_run=False,
                               counterfactual_reference="fixed_recent_reference_benchmark" if scenario == "CF_UNL" else "not_applicable",
                               analysis_version=ANALYSIS_VERSION)
                row.update(raw_output_schema_hash=schema, output_extraction_version=OUTPUT_EXTRACTION_VERSION,
                           output_extraction_hash=sha256(root / "src/utils/e8_g3_runner.py"),
                           analysis_hash=sha256(root / "src/utils/e8_g3_postprocess.py"),
                           weather_parser_sha256=sha256(Path(row["weather_path"])),
                           input_manifest_sha256=sha256(Path(row["compiled_dir"]) / "input_manifest.json"),
                           lockbox=False)
                rows.append(row)
                if len(rows) % progress_every == 0:
                    print(f"E8 compiled/mapped {len(rows)}/{total_runs}; EnergyPlus calls=0", flush=True)
    result = pd.DataFrame(rows)
    if len(result) != total_runs or result.run_id.nunique() != total_runs:
        raise ValueError(f"Compiled E8 manifest does not have {total_runs} unique logical run IDs")
    result["content_duplicate_of"] = result.groupby("physics_hash")["run_id"].transform("first")
    result["physical_duplicate"] = result["content_duplicate_of"].ne(result["run_id"])
    if result.physical_duplicate.any():
        raise ValueError("Physical duplicates require an explicit alias dependency; refusing duplicate solver dispatch")
    write_csv(result, delivery / "open6_target_manifest.csv")
    write_csv(result.loc[~result.disposition.eq("reuse_complete")], delivery / "open6_incremental_manifest.csv")
    registry = result[["event_id", "base_case_id", "prototype_id", "floor_position", "scenario_id", "run_id",
                       "physics_hash", "raw_output_schema_hash", "analysis_hash", "disposition", "source_batch",
                       "source_run_id", "source_result_dir"]].copy()
    write_csv(registry, delivery / "open6_cache_registry.csv")
    event_manifests = {}
    for event_id in events:
        path = delivery / "event_manifests" / f"{event_id}.csv"
        write_csv(result.loc[result.event_id.eq(event_id)].copy(), path)
        event_manifests[event_id] = {"path": str(path), "sha256": sha256(path),
                                     "rows": 1485, "groups": 297,
                                     "dispositions": result.loc[result.event_id.eq(event_id), "disposition"].value_counts().to_dict()}
    summary = {
        "status": "CODE_INPUTS_COMPILED_NOT_EXECUTED", "energyplus_calls": 0,
        "events": list(events), "lockboxes_excluded": sorted(LOCKBOX_EVENTS - set(events)),
        "logical_runs": len(result), "logical_groups": len(events) * 297,
        "unique_physics": int(result.physics_hash.nunique()),
        "dispositions": result.disposition.value_counts().to_dict(),
        "g4a_reused": int(result.disposition.eq("reuse_complete").sum()),
        "compile_seconds": time.perf_counter() - started,
        "manifest_sha256": sha256(delivery / "open6_target_manifest.csv"),
        "event_manifests": event_manifests,
    }
    write_json(summary, delivery / "open6_dry_run_summary.json")
    return result


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    frame = compile_open6(root, root / "intermediate/simulation_outputs/upstream/open6_delivery")
    print(json.dumps({"rows": len(frame), "dispositions": frame.disposition.value_counts().to_dict(),
                      "energyplus_calls": 0}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
