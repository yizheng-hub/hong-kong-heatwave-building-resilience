"""Zero-solver closure of G3 support boundaries and compilation-only G4-a request."""
from __future__ import annotations

import argparse
import __future__
import ast
import difflib
import hashlib
import io
import json
import math
import re
import shutil
import sys
import time
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from utils.e8_g3_postprocess import (
    building_crossing_from_zones, building_recovery_from_zones, capacity_and_phase_audit,
    load_monitoring_support, outage_audit, outage_score_bounds, recovery_diagnostic,
    recovery_score_bounds, result_dir, write_csv, write_json,
    threshold_crossing_diagnostic, zone_timestep_service_accounting,
)
from utils.simulation_metrics import as_hkt, outage_metrics_v2, sha256_file, raw_hash_reconciliation
from utils.e8_g3_runner import (
    CORE_SCENARIO_IDS, _compile_case, load_production, parse_native_output,
)
from utils.e8_preflight import stable_sha256

VERSION = "G3_metrics_v2_closure"


def snapshot_evidence(work):
    paths = list((work / "02_runs" / "g3_results").rglob("*")) + list((work / "G3_metrics_v2").rglob("*"))
    paths.append(work / "02_runs" / "g3_call_ledger.csv")
    return {str(p.relative_to(work)): sha256_file(p) for p in paths if p.is_file()}


def cache_diagnostics(manifest, work, out, production):
    begin = time.perf_counter()
    audits, metrics = outage_audit(manifest, work)
    ca, cooling = capacity_and_phase_audit(manifest, work, production)
    write_csv(pd.concat([audits, ca]), out / "production_physics_audit.csv")
    write_csv(cooling, out / "cooling_service_diagnostics.csv")
    v2, zone_diagnostics, hottest = outage_metrics_v2(manifest, work)
    write_csv(v2, out / "corrected_v2_summary.csv")
    write_csv(zone_diagnostics, out / "zone_same_mask_diagnostics.csv")
    write_csv(hottest, out / "hottest_zone_identity.csv")
    support_rows, crossing_rows, recovery_rows, score_rows, compare_rows, simultaneous_rows = [], [], [], [], [], []
    for letter in "ABCD":
        role = f"core_group_{letter}"
        group = manifest.loc[manifest.role.eq(role)].set_index("scenario_id")
        rated, outage = group.loc["HW_RATED"], group.loc["HW_O24_SYNC"]
        start, restore = as_hkt(outage.outage_start_hkt), as_hkt(outage.outage_end_hkt)
        frames = {}
        for scenario, item in [("HW_RATED", rated), ("HW_O24_SYNC", outage)]:
            frame = load_monitoring_support(work, item, rated)
            frames[scenario] = frame
            window = frame.loc[frame.timestamp_hkt.gt(start) & frame.timestamp_hkt.le(restore)]
            for zone_id, z in window.groupby("zone_id"):
                support_rows.append(dict(role=role, scenario_id=scenario, zone_id=zone_id,
                    expected_rows=len(z), required_rows=int(z.monitoring_required.sum()),
                    required_valid_rows=int(z.support_status.eq("required_valid").sum()),
                    required_missing_rows=int(z.support_status.eq("required_missing").sum()),
                    not_required_rows=int(z.support_status.eq("not_required").sum()),
                    monitoring_zone_hours=.25 * int(z.support_status.eq("required_valid").sum()),
                    output_missing_rows=int((~z.output_present).sum())))
            cross, zone_cross = building_crossing_from_zones(frame, start)
            zone_cross.insert(0, "scenario_id", scenario); zone_cross.insert(0, "role", role)
            crossing_rows.extend(zone_cross.to_dict("records"))
            fixed_frame = frame.copy(); fixed_frame["occupied_for_screen"] = True
            fixed, _ = building_crossing_from_zones(fixed_frame, start)
            crossing_rows.append({**fixed, "role": role, "scenario_id": scenario, "zone_id": "BUILDING_FIXED_CONDITIONED"})
            rec, simultaneous = building_recovery_from_zones(frame, restore)
            simultaneous.insert(0, "scenario_id", scenario); simultaneous.insert(0, "role", role)
            simultaneous_rows.append(simultaneous)
            rw = frame.loc[frame.timestamp_hkt.gt(restore) & frame.timestamp_hkt.le(restore + pd.Timedelta(hours=24))]
            recovery_rows.append(dict(role=role, scenario_id=scenario, zone_id="ALL_CONDITIONED", **rec,
                planned_cooling_zone_hours=.25 * int((rw.structural_conditioning_present & rw.planned_service_required).sum()),
                restore_hkt=restore, end_hkt=restore + pd.Timedelta(hours=24)))
            for zone, z in frame.loc[frame.structural_conditioning_present].groupby("zone_id"):
                r = recovery_diagnostic(z.timestamp_hkt, z.temperature_c, restore)
                wz = rw.loc[rw.zone_id.eq(zone)]
                recovery_rows.append(dict(role=role, scenario_id=scenario, zone_id=zone, **r,
                    planned_cooling_zone_hours=.25 * int(wz.planned_service_required.sum()),
                    restore_hkt=restore, end_hkt=restore + pd.Timedelta(hours=24)))
            if scenario == "HW_O24_SYNC":
                for component, result, mapper in [("O", cross, outage_score_bounds), ("R", rec, recovery_score_bounds)]:
                    mapped = mapper(result)
                    mapped["time_lower_closed"] = bool(result.get("lower_h") == result.get("upper_h") or result.get("lower_bound_source") in {"eligibility_interval_start", "window_start_without_state"})
                    mapped["time_upper_closed"] = bool(np.isfinite(result.get("upper_h", math.nan)))
                    mapped["bound_semantics"] = "endpoint flags and observation_status govern; interval means/interpolation do not assert instantaneous temperatures"
                    score_rows.append(dict(role=role, event_id=outage.event_id, component=component,
                        effective_monitoring_hours=cross.get("effective_monitoring_hours") if component == "O" else .25 * max(0, rec["valid_required_samples"] - 1),
                        effective_monitoring_zone_hours=cross.get("effective_monitoring_zone_hours") if component == "O" else .25 * max(0, rec["valid_required_samples"] - 1) * len(json.loads(outage.conditioned_zones_json)),
                        **mapped))
        a, b = frames["HW_RATED"], frames["HW_O24_SYNC"]
        keys = ["timestamp_hkt", "zone_id"]
        comparison = a[keys + ["temperature_c"]].merge(b, on=keys, suffixes=("_rated", "_outage"), validate="one_to_one")
        comparison["temperature_difference_c"] = comparison.temperature_c_outage - comparison.temperature_c_rated
        comparison["role"] = role
        rr = comparison.loc[comparison.timestamp_hkt.between(restore, restore + pd.Timedelta(hours=24))]
        write_csv(rr[keys + ["role", "temperature_c_rated", "temperature_c_outage", "temperature_difference_c", "structural_conditioning_present", "planned_service_required"]], out / f"{letter}_paired_recovery_trace.csv")
        if letter == "A":
            trace = comparison.loc[comparison.timestamp_hkt.between(start, start + pd.Timedelta(hours=4))].copy()
            trace["relative_h"] = (trace.timestamp_hkt - start).dt.total_seconds() / 3600
            trace["interval_start_hkt"] = trace.timestamp_hkt - pd.Timedelta(minutes=15)
            trace["interval_end_hkt"] = trace.timestamp_hkt
            trace["temperature_semantics"] = "TimeStep interval mean; adjacent-mean interpolation is an explicit diagnostic model"
            trace["qualification_semantics"] = "reference hourly schedule applies to preceding 15-minute interval"
            write_csv(trace, out / "A_support_trace.csv")
    write_csv(pd.DataFrame(support_rows), out / "expected_monitoring_support_audit.csv")
    write_csv(pd.DataFrame(crossing_rows), out / "zone_crossing_support.csv")
    write_csv(pd.DataFrame(recovery_rows), out / "rated_vs_outage_recovery_diagnostic.csv")
    write_csv(pd.concat(simultaneous_rows), out / "all_required_zone_recovery_support.csv")
    write_csv(pd.DataFrame(score_rows), out / "BERI_score_bounds.csv")
    write_csv(metrics, out / "corrected_censoring_and_pairing_metrics.csv")
    old = pd.read_csv(work / "G3_metrics_v2" / "01_metrics" / "g3_outage_recovery_metrics_v2.csv").set_index("role")
    for row in v2.to_dict("records"):
        for field in ["first_crossing_32c_h", "first_crossing_lower_h", "first_crossing_upper_h", "first_crossing_status", "t_recovery_24h", "main_absolute_dh32", "main_same_zone_positive_extra_dh26"]:
            prior = old.loc[row["role"], field]
            now = row[field]
            equal = (pd.isna(prior) and pd.isna(now)) or prior == now
            compare_rows.append(dict(role=row["role"], field=field, v2_value=prior, closure_value=now,
                change="value_retained_support_verified" if equal else "censoring_or_support_correction"))
    write_csv(pd.DataFrame(compare_rows), out / "metrics_before_after.csv")
    return {"four_pair_metrics_seconds": time.perf_counter() - begin}


def edge_case_evidence(work, out):
    old_dir = work / "G3_metrics_v2/05_publication/E8_G3_metrics_v2_review/code"
    old = {"pd": pd, "np": np, "math": math, "Path": Path}
    for name in ["utils__e8_g3_postprocess.py", "src__utils__simulation_metrics.py"]:
        tree = ast.parse((old_dir / name).read_text(encoding="utf-8"))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(old_dir / name), "exec", flags=__future__.annotations.compiler_flag), old)
    start = pd.Timestamp("2020-07-01", tz="Asia/Hong_Kong")
    times = pd.Series(pd.date_range(start, periods=97, freq="15min"))
    def zone_frame(a, b):
        return pd.concat([pd.DataFrame({"timestamp_hkt": times, "zone_id": zone, "temperature_c": values, "occupied_for_screen": True, "structural_conditioning_present": True}) for zone, values in [("A", a), ("B", b)]], ignore_index=True)
    missing = pd.Series(31., index=times.index); missing.iloc[20] = np.nan
    early = [31.] + [np.nan] * 7 + [33.] * 89
    late = [31.] * 48 + [33.] * 49
    keys = pd.DataFrame({"timestamp_hkt": [start, start], "zone_id": ["A", "B"], "zone_area_m2": [1., 1.], "cooling_total_j": [10., 0.]})
    cases = [
        ("01_t0_interpolation", "threshold_crossing_diagnostic", lambda fn: fn(times.iloc[:2], pd.Series([31., 33.]), start, 32.)),
        ("02_six_hour_gap", "threshold_crossing_diagnostic", lambda fn: fn(times.iloc[:25], pd.Series([31.] + [np.nan] * 23 + [33.]), start, 32.)),
        ("03_midwindow_missing", "threshold_crossing_diagnostic", lambda fn: fn(times, missing, start, 32.)),
        ("04_earlier_interval_vs_later_exact", "building_crossing_from_zones", lambda fn: fn(zone_frame(early, late), start)[0]),
        ("05_safe_zone_and_missing_zone", "building_crossing_from_zones", lambda fn: fn(zone_frame([31.] * 97, [np.nan] * 97), start)[0]),
        ("06_thirty_hour_recovery", "recovery_diagnostic", lambda fn: fn(pd.Series(pd.date_range(start, periods=121, freq="15min")), pd.Series([27.] * 120 + [25.]), start)),
        ("07_recovery_gap", "recovery_diagnostic", lambda fn: fn(times, missing - 4, start)),
        ("08_missing_zone_all_recovered", "_recovery_all_zones", lambda fn: fn(zone_frame([25.] * 97, [np.nan] * 97), start, 26., 24.)),
        ("09_missing_cooling_reference_key", "zone_timestep_service_accounting", lambda fn: fn(keys, keys.iloc[1:])[1]),
    ]
    current = dict(globals())
    current["_recovery_all_zones"] = lambda frame, at, threshold, window: building_recovery_from_zones(frame, at, threshold, window)[0]
    rows = []
    for case_id, function, call in cases:
        row = {"case_id": case_id, "function": function, "fixture": "synthetic_only", "old_source": str(old_dir)}
        for label, namespace in [("old", old), ("new", current)]:
            try:
                value = call(namespace[function])
                row[label + "_result"] = json.dumps(value, default=str, ensure_ascii=False)
            except Exception as exc:
                row[label + "_result"] = type(exc).__name__ + ": " + str(exc)
        rows.append(row)
    write_csv(pd.DataFrame(rows), out / "nine_edge_cases_old_new.csv")


def warnings_reconciled(manifest, work):
    rows = []
    for item in manifest.itertuples(index=False):
        path = result_dir(work, item.run_id) / "raw" / "eplusout.err"
        text = path.read_text(encoding="utf-8", errors="replace")
        matches = list(re.finditer(r"^.*\*\* Warning \*\*.*$", text, re.M))
        footer = re.search(r"Completed Successfully--\s*(\d+) Warning", text)
        footer_count = int(footer.group(1)) if footer else None
        for idx, match in enumerate(matches):
            block = text[match.start():matches[idx + 1].start() if idx + 1 < len(matches) else len(text)]
            first_line = block.splitlines()[0].strip()
            recurring = "*************" in first_line
            total = re.search(r"This error occurred\s+(\d+) total times", block)
            warmup = re.search(r"during Warmup\s+(\d+) times", block)
            sizing = re.search(r"during Sizing\s+(\d+) times", block)
            obj = re.search(r'"([^"]+)"', first_line)
            phase = "warmup" if (warmup and int(warmup.group(1)) > 0) or "During Warmup, Environment" in block else "input_check" if "GroundTemperature" in first_line else "unspecified"
            rows.append(dict(run_id=item.run_id, role=item.role, scenario_id=item.scenario_id,
                record_kind="recurring_summary" if recurring else "initial_message",
                object_id=obj.group(1) if obj else "Site:GroundTemperature:BuildingSurface",
                category="humidity_saturation" if "saturation limit" in first_line else "ground_temperature" if "GroundTemperature" in first_line else "other",
                message=first_line, stage=phase, text_entries=1,
                recurring_occurrences=int(total.group(1)) if total else 0,
                recurring_warmup_occurrences=int(warmup.group(1)) if warmup else 0,
                recurring_sizing_occurrences=int(sizing.group(1)) if sizing else 0,
                footer_warning_count=footer_count, err_sha256=sha256_file(path)))
    return pd.DataFrame(rows)


def physical_content_hash(row):
    hashes = row.file_byte_sha256 if isinstance(row.file_byte_sha256, dict) else ast.literal_eval(row.file_byte_sha256)
    return stable_sha256({key: hashes[key] for key in ("idf", "epw", "schedule", "engine")})


def compile_g4a(root, work, out, manifest, production):
    design = pd.read_csv(root / "intermediate/foundation_pipeline/02_building_response/thermal_experiment_design.csv")
    selected = design.sort_values("base_case_id").drop_duplicates(["prototype_id", "floor_position"])
    if len(selected) != 56:
        raise ValueError(f"frozen response units differ: {len(selected)}")
    old_by_content = {physical_content_hash(row): row for row in manifest.itertuples(index=False) if row.role.startswith("core_group") and row.prehistory_days == 14}
    rows = []
    compile_root = out / "G4a_compiled_inputs"
    for _, design_row in selected.iterrows():
        for scenario in CORE_SCENARIO_IDS:
            old = manifest.loc[manifest.event_id.eq("HKO_LI_0049") & manifest.base_case_id.eq(design_row.base_case_id) & manifest.scenario_id.eq(scenario) & manifest.prehistory_days.eq(14)]
            role = old.iloc[0].role if len(old) else "G4a_engineering_coverage"
            start = time.perf_counter()
            compiled = _compile_case(root, work, production, design_row, "HKO_LI_0049", scenario, 14, role, compile_output_root=compile_root)
            elapsed = time.perf_counter() - start
            content = physical_content_hash(pd.Series(compiled))
            cached = old_by_content.get(content)
            reusable = cached is not None
            if reusable:
                raw = result_dir(work, cached.run_id) / "raw/eplusout.csv"
                payload = json.loads((result_dir(work, cached.run_id) / "run_result.json").read_text(encoding="utf-8"))
                reusable = raw.exists() and payload.get("status") == "passed"
            rows.append({**compiled, "legacy_g3_physics_hash": compiled["physics_hash"],
                "physics_hash": content, "physics_hash_definition": "sha256 canonical JSON of final IDF EPW schedule engine byte hashes",
                "run_id": "e8g4a_" + content[:20], "cached_g3_run_id": cached.run_id if reusable else "",
                "disposition": "reextract" if reusable else "new", "reason": "exact_physical_bytes_and_complete_retained_raw" if reusable else "no_qualified_content_identical_G3_cache",
                "compile_seconds": elapsed, "approved_to_run": False,
                "counterfactual_reference": "fixed_recent_reference_benchmark" if scenario == "CF_UNL" else "not_applicable",
                "analysis_version": VERSION})
            if len(rows) % 20 == 0:
                print(f"compiled {len(rows)}/280 inputs; solver calls=0", flush=True)
    result = pd.DataFrame(rows)
    result["content_duplicate_of"] = result.groupby("physics_hash")["run_id"].transform("first")
    write_csv(result, out / "G4a_run_manifest.csv")
    write_csv(selected, out / "G4a_selected_frozen_designs.csv")
    return result


def parser_benchmark(manifest, work, production):
    rows = []
    params = production.prototype_parameter_table().set_index("prototype_id")
    for role in ["core_group_A", "core_group_C", "core_group_D"]:
        item = manifest.loc[manifest.role.eq(role) & manifest.scenario_id.eq("HW_RATED")].iloc[0]
        folder = result_dir(work, item.run_id)
        weather = pd.read_parquet(folder / "inputs/weather.parquet")
        weather["timestamp_hkt"] = pd.to_datetime(weather.timestamp_hkt, utc=True).dt.tz_convert("Asia/Hong_Kong")
        # Read the same COP used by the persisted parser from the production table.
        cop_col = "baseline_ac_cop"
        start = time.perf_counter()
        parsed = parse_native_output(folder / "raw/eplusout.csv", weather, folder / "inputs/event_schedules.parquet", json.loads(item.zone_areas_json), set(json.loads(item.conditioned_zones_json)), float(params.loc[item.prototype_id, cop_col]))
        elapsed = time.perf_counter() - start
        start = time.perf_counter()
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for raw in sorted((folder / "raw").glob("*")):
                if raw.is_file(): archive.write(raw, raw.name)
        rows.append(dict(role=role, hours=len(weather), parse_seconds=elapsed,
            archive_compression_in_memory_seconds=time.perf_counter() - start,
            archive_bytes=len(buffer.getvalue()), returned_frames=len(parsed), writes_to_original=False))
    return pd.DataFrame(rows)


def make_budget(root, work, out, g4a, timings):
    g4a["raw_output_schema_hash"] = json.loads((work / "00_protocol/output_schema.json").read_text(encoding="utf-8"))["output_schema_hash"]
    g4a["output_extraction_version"] = "closure-support-v1"
    g4a["output_extraction_hash"] = sha256_file(root / "src/utils/e8_g3_runner.py")
    g4a["analysis_hash"] = sha256_file(root / "src/utils/e8_g3_postprocess.py")
    write_csv(g4a, out / "G4a_run_manifest.csv")
    inventory = pd.read_csv(work / "02_runs/g3_run_inventory.csv")
    same_event = inventory.loc[inventory.event_id.eq("HKO_LI_0049")]
    measured = json.loads((work / "02_runs/g3_resource_summary.json").read_text(encoding="utf-8"))
    parsed = pd.read_csv(out / "parser_timing.csv")
    unique = g4a.drop_duplicates("physics_hash")
    counts = unique.disposition.value_counts().to_dict()
    new = int(counts.get("new", 0))
    raw_bytes = float(same_event.run_storage_bytes.quantile(.9)) * new
    disk = shutil.disk_usage(root)
    compiled_bytes = sum(p.stat().st_size for p in (out / "G4a_compiled_inputs").rglob("*") if p.is_file())
    budget = {
        "authorization": "pending_user_G4a_resource_approval", "event": "HKO_LI_0049", "prehistory_days": 14,
        "design_units": int(g4a[["prototype_id", "floor_position"]].drop_duplicates().shape[0]),
        "required_design_scenarios": len(g4a), "required_unique_physical_inputs": len(unique),
        "reused_ready": 0, "reextract": int(counts.get("reextract", 0)), "new_solver_inputs": new,
        "blocked_inputs": int(counts.get("blocked", 0)), "exclusive_counts_sum": len(unique),
        "g4a_is_subset_of_1485": True, "proposed_concurrency": 4, "proposed_retry_calls": 15,
        "proposed_total_solver_call_ceiling": new + 15, "proposed_wall_limit_minutes": 60,
        "proposed_new_run_storage_ceiling_gib": 5, "proposed_stop_free_disk_gib": 10,
        "disk_free_gib_at_request": disk.free / 1024**3,
        "measured_solver_p50_seconds": measured["solver_elapsed_seconds_p50"],
        "measured_solver_p90_seconds": measured["solver_elapsed_seconds_p90"],
        "measured_peak_solver_rss_mib": measured["measured_peak_rss_bytes_max"] / 1024**2,
        "measured_compile_p50_seconds": float(g4a.compile_seconds.median()),
        "measured_compile_p90_seconds": float(g4a.compile_seconds.quantile(.9)),
        "measured_parser_p50_seconds": float(parsed.parse_seconds.median()),
        "measured_parser_p90_seconds": float(parsed.parse_seconds.quantile(.9)),
        "measured_archive_cpu_p90_seconds": float(parsed.archive_compression_in_memory_seconds.quantile(.9)),
        "measured_four_pair_postprocessing_seconds": timings["four_pair_metrics_seconds"],
        "estimated_serial_solver_plus_parser_seconds": new * (measured["solver_elapsed_seconds_p90"] + float(parsed.parse_seconds.quantile(.9))),
        "estimated_remaining_serial_pipeline_seconds": new * (measured["solver_elapsed_seconds_p90"] + float(parsed.parse_seconds.quantile(.9)) + float(parsed.archive_compression_in_memory_seconds.quantile(.9))) + len(g4a[["prototype_id", "floor_position"]].drop_duplicates()) / 4 * timings["four_pair_metrics_seconds"],
        "estimate_method": "new runs*(measured solver p90+parser p90+in-memory archive p90)+56/4*measured four-case metric extraction; compilation already complete; durable archive I/O and retries unmeasured, so this is a planning estimate only",
        "same_event_run_storage_p90_bytes": float(same_event.run_storage_bytes.quantile(.9)),
        "estimated_new_retained_run_gib": raw_bytes / 1024**3,
        "compiled_inputs_already_created_gib": compiled_bytes / 1024**3,
        "temporary_storage_reserve_gib": 1.0,
        "uncertainty": "Projection uses current short event and observed prototypes; compile/parse/archive/metrics and retries are additional. P90-scaled estimate is not an upper bound. Proposed wall/storage limits are stop limits, not completion promises.",
        "manifest_sha256": sha256_file(out / "G4a_run_manifest.csv"),
        "physics_hash_version": "content-v1-IDF-EPW-schedule-engine", "energyplus_calls_this_closure": 0,
    }
    write_json(budget, out / "G4a_resource_budget.json")
    storage = []
    for row in same_event.itertuples(index=False):
        folder = result_dir(work, row.run_id)
        storage.append({"run_id": row.run_id, **{name + "_bytes": sum(p.stat().st_size for p in (folder / name).rglob("*") if p.is_file()) for name in ["raw", "parsed", "inputs"]}})
    write_csv(pd.DataFrame(storage), out / "G4a_storage_measurements.csv")
    return budget


def write_contracts(out):
    schema = json.loads((ROOT / "intermediate/simulation_outputs/upstream/G3_metrics_v2/00_protocol/output_schema_v2.json").read_text(encoding="utf-8"))
    schema["version"] = "closure-support-v1"
    for var in schema["variables"]:
        if var.get("field_id") in {"zone_air_temperature", "zone_relative_humidity"}:
            var["temporal_semantics"] = "reported interval average at interval-end label; native RDD Average; adjacent-mean interpolation is a stated diagnostic model"
    schema["semantics_evidence"] = "cached EnergyPlus26.1 RDD Zone Average / HVAC Average; https://energyplus.readthedocs.io/en/latest/guides/input-output-reference/1.10-group-thermal-zone-descriptiongeometry.html"
    write_json(schema, out / "output_schema_closure.json")
    write_json({
        "analysis_version": VERSION, "time_unit": "h", "score_unit": "fraction",
        "monitoring": "structurally conditioned AND reference occupancy fraction>0.15; threshold excludes modeled background occupancy and is not a validated human occupancy or health threshold",
        "native_temperature_semantics": "15-minute interval mean; interpolation between contiguous same-zone eligible means is explicitly model-based and does not establish instantaneous threshold time",
        "censoring": {"interval": "preserve lower/upper and boundary provenance; no midpoint", "right_outage": "complete required support for 24h: truncated S_O=1", "right_recovery": "complete required support with no recovery by24h: S_R=0", "unknown": "bound interval with unknown status; no point score", "not_applicable": "NA"},
        "qualification_boundary": "Lower boundary can be inclusive when eligibility starts with an already-hot space; (L,U] notation alone does not establish safety before eligibility.",
        "event_aggregation": "Require exactly every expected event x N,C,O,R. Median of lower/upper separately; even median is arithmetic mean of middle two. No dropna. Geometric mean of four lower/upper component medians, multiplied by100; true zeros retained.",
        "ranking": "Interval scores are not exact ties; lower/upper rankings only as labeled diagnostics. No precise queue from arbitrary interval points.",
        "heat_rule": "unchanged v2 same-zone/time positive excess DH26, maximum over eligible zones per step then integrate; distinct from difference of maxima",
        "endpoint_precision": "unrounded native times retained; exact points under declared interpolation have closed endpoints; censor bounds preserve status and source",
    }, out / "BERI_censoring_contract.json")
    (out / "HKUST_time_source_resolution.md").write_text("""# HKUST source-time resolution

NASA source header explicitly says LST at longitude 114.2634 E. NASA defines LST using 15-degree swaths; this longitude lies in the 112.5–127.5 E swath centred on120E, corresponding to UTC+8. This source-based inference supports no clock-offset shift relative to HKT (UTC+8). NASA supplies hourly average values. Interval start/end convention is a separate unresolved issue in the retained CSV metadata.

Sources: [NASA hourly time standards](https://power.larc.nasa.gov/docs/services/api/temporal/hourly/), [HKO local time](https://www.hko.gov.hk/en/gts/time/basicterms-localtime.htm).

HKO Co-WIN station6136 retained CSVs contain naive obstime and QC flags; the stored metadata does not establish whether temperature/RH represent an hour mean, endpoint sample, or another reporting support. Public general HKO time definitions establish HKT, but do not determine the support of these specific fields. This remains unresolved.

HKUST processing uses the T60 cumulative reading at t minus the preceding reading, so derived energy_kwh(t) is computationally the interval ending at t. Original resampling and timezone conventions are not established by the local naive timestamps. Existing model hourly sums use interval-start labels. Consequently same-label observed/model pairing has a potential one-hour support mismatch. No shift is applied without confirming source semantics, and no error-minimizing alignment is used. A definitive re-pair may be possible from cached model hours after provenance is confirmed; a solver call is not automatically required.

The 336h calibration/test windows and all four observed objects remain unchanged. The original hourly results are retained as retrospective comparisons with unresolved external interval alignment; calibration hour-of-week outperforms the generic physical prototype for all four. These results do not establish strong hourly predictive validation. Raw EnergyPlus date-label agreement concerns the simulation output axis only.

The [source paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11599614/) describes T60 resampling and cumulative meters; it does not specify timezone or resampling label support. The two real-prehistory candidates remain unexecuted. Exact interval metadata remains unresolved. No tuning, new window selection, error-driven time shift, or EnergyPlus call was made.
""", encoding="utf-8")


def write_reports(root, work, out):
    status = []
    for filename in ["regression_tests.xml", "full_tests.xml"]:
        path = out / filename
        if path.exists():
            suites = ET.parse(path).getroot()
            for suite in suites.iter("testsuite"):
                status.append(dict(file=filename, **suite.attrib))
    write_csv(pd.DataFrame(status), out / "test_execution_summary.csv")
    budget = json.loads((out / "G4a_resource_budget.json").read_text(encoding="utf-8"))
    warnings = pd.read_csv(out / "warning_count_reconciliation.csv")
    support = pd.read_csv(out / "expected_monitoring_support_audit.csv")
    metrics = pd.read_csv(out / "corrected_censoring_and_pairing_metrics.csv")
    roles = {row.role: row for row in metrics.itertuples(index=False)}
    test_text = "; ".join(f"{x['file']}: {x.get('tests')} collected, {x.get('failures')} failures, {x.get('errors')} errors, {x.get('skipped')} skipped" for x in status)
    (out / "closure_report.md").write_text(f"""# G3观测支持与删失边界关闭报告

本轮EnergyPlus调用为0。26个G3运行的原始/解析文件及v2证据保持原样；814个冻结文件的SHA256在处理前后相同，调用账本仍为27条。0054/0066没有运行或读取建筑性能。当前输出目录为G3_metrics_v2_closure。

## 已关闭的实现问题

- 独立区域字典、输入天气时间轴、参考日程生成expected zone×time网格；缺行、非有限值、重复、区域属性不一致均有检查。已知不监测与要求监测却缺输出分别标记。
- 同区越阈支持保留t0；6小时缺口的下界保留真实先前支持；较早缺口不会被后来的安全记录抹去。建筑首次事件由各区界限取最早可能/最早确定上界，较晚精确点和任一区右删失均不能覆盖未知区。
- 恢复观察固定[restore,restore+24h]；全空调区逐时完整后才合成最大温度。窗口内缺口可能掩盖首次恢复，保留该不确定性。持续恢复仍仅为完整连续区间诊断。
- 冷量按预定区域×时间格核对两侧，再逐区匹配。相同缺行在两侧同时出现也会报错。空监测集合的主度时为NA；真实零值单独保留。
- BERI提供原始时间、删失状态、分数界限及端点来源。事件分项必须完整，偶数中位数采用中间两个数的均值，界限依单调性通过中位数和几何平均传播。

代码位置见code_location_index.csv。九组反例从冻结v2源码AST和当前函数分别执行，详见nine_edge_cases_old_new.csv；合成fixture不进入论文源表。用户提到的check_v2_edge_cases.py未在已查附件/桌面/项目中定位；此处依据用户逐项提供的反例重建，并以实际生产outage_audit注入测试覆盖。

## A案例的观测支持

15:00的Living记录为35.975944°C，参考占用分数0.03，主监测不成立。18:00标签对应17:45–18:00，仍为背景占用；18:15标签对应18:00–18:15，参考占用为1。主首次合格过热的界限来自18:00的资格开始和18:15的首个合格记录，保留3.00–3.25h的区间支持；资格开始时已过热的可能性保留。该区间不表示Living此前三小时低于32°C。固定空调空间诊断保留t0已过热。

TimeStep的温度/RH是带区间末端标签的平均输出：当前引擎RDD写明Zone Average/HVAC Average。v2合同的state/end-of-zone-timestep描述已在新output_schema_closure.json更正；原合同保留。相邻均值的线性插值仍按既定诊断模型计算，不能被写成观测到了精确瞬时越阈。无需改变物理输入。

## 同规则额定对照

A、D的RATED与O24都在24h观察末端右删失：各空间可分别降到26°C，但该运行安排下未同时满足全空调空间目标。恢复分数0描述这个绝对服务条件。停电残余后果由逐区配对温差和附加度时另外呈现。B的额定对照在恢复时刻已达标、停电支路在1.5h达标；C对应0h和0.5h。

主有效监测覆盖（时间并集）A/B/C/D为14/11/24/24h。区域小时另列，避免把多个区域相加当24h持续观测。所有当前缓存要求监测的记录缺失数为{int(support.required_missing_rows.sum())}。

## 证据边界与尚未关闭事项

- HKUST：NASA经度对应15°LST带与HKT的名义时差一致；Co-WIN各变量及HKUST原始重采样标签的区间支持尚未完全核实。代码对累计电表差分得到的是结束于t的能量，而模型聚合标签位于区间开始，存在需要源元数据确认的对齐问题。未按误差平移、未调参、未改变336h窗口。4/4的hour-of-week基线更好仍保留，当前不支持强小时负荷预测验证。
- CF沿用King's Park2016–2025 fixed_recent_reference_benchmark；正式事件温湿替换、其他天气和前史保持。它可产生负需求增量；S_N只对正相对增量惩罚。包含部分测试年份，T1–T3仍为固定参考下回顾性前推。新标签和分析合同不改变相容CF字节。
- 告警按文本、对象、重复及末尾计数分别保留：{int(warnings.text_entries.sum())}条Warning文本；末尾合计{int(warnings.footer_warning_count.sum())}次Warning；{int(warnings.recurring_occurrences.sum())}次在重复摘要中登记。{int(warnings.footer_reconciled.sum())}/26运行可按初始消息+重复次数与末尾对齐。C_O24的5个对象各5次重复均标warmup，末尾31 Warning。无Severe不能写成无Warning。地温输入未改。
- DOCX原始LaTeX失败保留，本轮不改手稿。资料年份的未知/推断身份沿用既有审计，固定状态接口覆盖不等于历史存量重建。

测试实际记录：{test_text}。全仓报告中预期保留的文档失败单列；初次发现的旧测试缺t0、新增辅助模块枚举未更新，修正依据和原失败XML均在包中。

## G4-a申请

已编译一个非锁箱事件HKO_LI_0049，14日真实前史，56个冻结原型—楼层设计×5工况。选择规则为各合法原型—楼层组中base_case_id最小者，不依结果选择。280个最终物理输入均唯一，其中{budget['reextract']}个精确字节命中G3缓存、{budget['new_solver_inputs']}个需要求解。物理内容哈希只依IDF、EPW、外部日程、引擎字节；G3旧哈希另存，角色、评分或图件不进入新物理哈希。

建议仅批准G4-a工程覆盖：并发上限4，265个新输入，最多15次技术重试，总调用上限280；墙钟停止上限60分钟，新增运行文件5GiB，磁盘剩余10GiB停止调度。具体实测与估算见G4a_resource_budget.json。这些是资源停止上限，非完成时长保证。G4-a是后续1485清单的子集，已完成输入不重复求解。

本轮停止在申请关口。G4整事件、G5及住宅机制批量均未获本轮授权。
""", encoding="utf-8")
    (out / "G4a_GO_NO_GO.md").write_text(f"""# G4-a工程覆盖建议（待资源授权）

| 层面 | 判断 | 依据 |
|---|---|---|
| 数值接口/缓存完整性 | GO，仅限工程覆盖 | 边界及生产路径回归；A–D缓存重算；原证据哈希保持 |
| 参数与评分定义 | GO，按当前固定合同 | 14日真实前史；同区匹配；删失界限；0.15为背景排除规则；均值插值为诊断模型 |
| 观测验证/论文证据 | NO-GO，尚未定稿 | HKUST源区间语义待证实；4/4简单时段基线更好；不能写作成功的强预测验证 |
| 资源授权 | NO-GO，等待用户 | 最多280个唯一核心输入；15重提取+265新求解；建议4并发、15次重试、60min/5GiB停止上限 |

清单SHA256：{budget['manifest_sha256']}

仅建议下一步G4-a覆盖关。不会据此启动G4-a，也不授权297设计整事件、其他E8事件、G5、锁箱或住宅机制批量。DOCX失败与工程覆盖分列。若覆盖中发现公共物理输入错误，停止受影响分支并记录最小变更。
""", encoding="utf-8")
    locations = []
    for relative in ["src/utils/e8_g3_postprocess.py", "src/utils/simulation_metrics.py", "src/utils/e8_g3_closure.py"]:
        tree = ast.parse((root / relative).read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef): locations.append(dict(file=relative, function=node.name, line=node.lineno))
    write_csv(pd.DataFrame(locations), out / "code_location_index.csv")
    source_time = pd.read_csv(work / "G3_metrics_v2/02_validation/hkust_time_source_audit.csv")
    source_time.loc[source_time.source.eq("NASA POWER Sai Kung"), "status"] = "longitude swath implies UTC+8; no offset shift indicated; hourly support label still unresolved"
    source_time.loc[source_time.source.eq("HKUST processed meter objects"), "status"] = "diff(cumulative) ends at t; source resampling support/timezone unresolved; model aggregation starts at t; possible support mismatch"
    write_csv(source_time, out / "hkust_time_source_resolution.csv")
    for name in ["hkust_baselines_v2.csv", "hkust_metrics_v2.csv", "hkust_boundaries_v2.csv"]:
        shutil.copy2(work / "G3_metrics_v2/02_validation" / name, out / name)
    for name in ["counterfactual_daily_reference_sample_audit.csv", "counterfactual_leave_event_year_out.csv", "E8_event_and_reference_daily_targets.csv"]:
        shutil.copy2(work / "G3_metrics_v2/03_counterfactual" / name, out / name)
    (out / "counterfactual_reference_status.md").write_text("User's current instruction retains King's Park2016–2025 fixed_recent_reference_benchmark. Cached CF bytes remain integration-origin evidence, linked here to the current fixed-reference definition; no fabricated dated approval record. Formal event T/RH only, negative net increments retained, no strict historical-information claim. No alternate reference was selected using outcomes. Physical hashes unchanged.\n", encoding="utf-8")


def package(root, work, out):
    write_reports(root, work, out)
    code_dir = out / "review_code"; code_dir.mkdir(exist_ok=True)
    old_code = work / "G3_metrics_v2/05_publication/E8_G3_metrics_v2_review/code"
    diff = []
    for relative in ["src/utils/e8_g3_postprocess.py", "src/utils/simulation_metrics.py", "src/utils/e8_g3_runner.py", "src/utils/e8_g3_closure.py"]:
        path = root / relative
        shutil.copy2(path, code_dir / relative.replace("/", "__"))
        prior = old_code / relative.replace("/", "__")
        diff.extend(difflib.unified_diff(prior.read_text(encoding="utf-8").splitlines(True) if prior.exists() else [], path.read_text(encoding="utf-8").splitlines(True), fromfile="v2/" + relative, tofile="closure/" + relative))
    (out / "code.diff").write_text("".join(diff), encoding="utf-8")
    excluded = {"G4a_compiled_inputs"}
    files = [p for p in out.rglob("*") if p.is_file() and not any(x in excluded for x in p.relative_to(out).parts) and p.suffix not in {".zip", ".sha256"} and p.name != "PACKAGE_MANIFEST_SHA256.csv"]
    write_csv(pd.DataFrame([dict(path=p.relative_to(out).as_posix(), bytes=p.stat().st_size, sha256=sha256_file(p)) for p in sorted(files)]), out / "PACKAGE_MANIFEST_SHA256.csv")
    files.append(out / "PACKAGE_MANIFEST_SHA256.csv")
    archive = out / "E8_G3_closure_G4a_review.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):
            z.write(p, p.relative_to(out).as_posix())
    archive.with_suffix(".zip.sha256").write_text(sha256_file(archive) + "  " + archive.name + "\n", encoding="ascii")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-g4a", action="store_true")
    parser.add_argument("--package-only", action="store_true")
    args = parser.parse_args()
    work = ROOT / "intermediate/simulation_outputs/upstream"; out = work / VERSION; out.mkdir(exist_ok=True)
    if args.package_only:
        package(ROOT, work, out); return
    before = snapshot_evidence(work)
    manifest = pd.read_csv(work / "02_runs/g3_manifest_revised.csv")
    production = load_production(ROOT)
    timings = cache_diagnostics(manifest, work, out, production)
    edge_case_evidence(work, out)
    write_json(timings, out / "postprocessing_timing.json")
    warnings = warnings_reconciled(manifest, work)
    write_csv(warnings, out / "warnings_reconciled.csv")
    summary = warnings.groupby("run_id").agg(text_entries=("text_entries", "sum"), initial_messages=("record_kind", lambda x: int(x.eq("initial_message").sum())), recurring_objects=("record_kind", lambda x: int(x.eq("recurring_summary").sum())), recurring_occurrences=("recurring_occurrences", "sum"), recurring_warmup_occurrences=("recurring_warmup_occurrences", "sum"), footer_warning_count=("footer_warning_count", "first")).reset_index()
    summary["reconstructed_warning_count"] = summary.initial_messages + summary.recurring_occurrences
    summary["footer_reconciled"] = summary.reconstructed_warning_count.eq(summary.footer_warning_count)
    write_csv(summary, out / "warning_count_reconciliation.csv")
    write_csv(raw_hash_reconciliation(manifest, work), out / "raw_hash_reconciliation.csv")
    write_contracts(out)
    write_csv(parser_benchmark(manifest, work, production), out / "parser_timing.csv")
    if args.compile_g4a:
        g4a = compile_g4a(ROOT, work, out, manifest, production)
        make_budget(ROOT, work, out, g4a, timings)
    elif (out / "G4a_run_manifest.csv").exists():
        make_budget(ROOT, work, out, pd.read_csv(out / "G4a_run_manifest.csv"), timings)
    for filename in ["representative_A_HW_RATED_eplusout.err", "representative_C_HW_O24_SYNC_eplusout.err", "representative_D_HW_O24_SYNC_eplusout.err"]:
        shutil.copy2(work / "G3_metrics_v2/05_publication/E8_G3_metrics_v2_review" / filename, out / filename)
    for source in [work / "01_inputs/E8_event_assignment.csv", work / "01_inputs/lockbox_manifest.json", work / "G3_metrics_v2/00_protocol/thermal_monitoring_contract.csv", work / "G3_metrics_v2/00_protocol/estimand_and_field_contract_v2.csv"]:
        if source.exists(): shutil.copy2(source, out / source.name)
    after = snapshot_evidence(work)
    if before != after:
        raise RuntimeError("Frozen G3/v2 evidence changed during closure")
    write_json({"energyplus_calls": 0, "ledger_rows": len(pd.read_csv(work / "02_runs/g3_call_ledger.csv")), "frozen_evidence_files_verified": len(before), "all_hashes_unchanged": True, "lockbox_performance_read": False, "G4a_executed": False, "snapshot_sha256": stable_sha256(before)}, out / "zero_solver_attestation.json")
    print(json.dumps({"output": str(out), "solver_calls": 0, "frozen_files_verified": len(before)}))


if __name__ == "__main__":
    main()
