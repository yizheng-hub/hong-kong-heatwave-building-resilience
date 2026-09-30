"""Evidence inventory, figure Source Data contracts and final release gate."""
from __future__ import annotations

import json
from pathlib import Path
import time
import zipfile
import pandas as pd
import pyarrow.parquet as pq

from utils.e8_completion_common import ALL_EVENTS, FINAL_LABEL, Blocked, atomic_csv, atomic_json, sha256


FIGURES = [
    ("F1", "a", "05_evidence/climate_catalog_and_event_roles.csv", "event_id|E8_role", "event", "65-event catalog and E8 role"),
    ("F1", "b", "05_evidence/climate_catalog_and_event_roles.csv", "event_id|E8_role", "event", "event-role background"),
    ("F1", "c", "05_evidence/climate_catalog_and_event_roles.csv", "event_id|E8_role", "event", "event-role background"),
    ("F2", "a", "05_evidence/validation_evidence_compatibility.csv", "validation_layer|dataset|primary_metric|value|claim_boundary|compatibility_decision", "reported", "meteorological validation"),
    ("F2", "b", "05_evidence/validation_evidence_compatibility.csv", "validation_layer|dataset|primary_metric|value|claim_boundary|compatibility_decision", "reported", "population validation"),
    ("F2", "c", "05_evidence/validation_evidence_compatibility.csv", "validation_layer|dataset|primary_metric|value|claim_boundary|compatibility_decision", "reported", "building response evidence"),
    ("F2", "d", "05_evidence/validation_evidence_compatibility.csv", "validation_layer|dataset|primary_metric|value|claim_boundary|compatibility_decision", "reported", "economic and future evidence"),
    ("F3", "a", "03_stock/building_reference_beri.parquet", "BUILDINGSTRUCTUREID|BERI_lower|BERI_upper|main_class", "score", "building type"),
    ("F3", "b", "03_stock/building_reference_beri.parquet", "BUILDINGSTRUCTUREID|BERI_lower|BERI_upper|prototype_floor_signature", "score", "component-floor composition"),
    ("F3", "c", "03_stock/building_reference_beri.parquet", "BUILDINGSTRUCTUREID|BERI_lower|BERI_upper|construction_year_value", "score", "construction era"),
    ("F3", "d", "03_stock/building_reference_beri.parquet", "BUILDINGSTRUCTUREID|BERI_lower|BERI_upper|NA_reason", "score", "BERI bounds and missingness"),
    ("F4", "a", "03_stock/city_event_energy_peak.csv", "event_id|E_HW_kwh|E_CF_kwh|net_delta_kwh", "kWh", "all eight events"),
    ("F4", "b", "03_stock/city_event_energy_peak.csv", "event_id|simultaneous_HW_peak_kw|simultaneous_CF_peak_kw|HW_peak_time_hkt", "kW", "simultaneous city peak"),
    ("F5", "a", "02_mechanisms/controlled_pair_results.csv", "mechanism_id|factor|prototype_id|event_id|level|reference_net_delta_kwh|treatment_net_delta_kwh|effect_net_delta_kwh|effect_net_delta_kwh_lower|effect_net_delta_kwh_upper|effect_net_delta_kwh_unit", "kWh", "strict physical pairs"),
    ("F5", "b", "02_mechanisms/controlled_pair_results.csv", "mechanism_id|factor|prototype_id|event_id|level|reference_net_delta_kwh|treatment_net_delta_kwh|effect_net_delta_kwh|effect_net_delta_kwh_lower|effect_net_delta_kwh_upper|effect_net_delta_kwh_unit", "kWh", "strict physical pairs"),
    ("F5", "c", "02_mechanisms/controlled_pair_results.csv", "mechanism_id|factor|prototype_id|event_id|level|reference_net_delta_kwh|treatment_net_delta_kwh|effect_net_delta_kwh|effect_net_delta_kwh_lower|effect_net_delta_kwh_upper|effect_net_delta_kwh_unit", "kWh", "strict physical pairs"),
    ("F5", "d", "02_mechanisms/controlled_pair_results.csv", "mechanism_id|factor|prototype_id|event_id|level|reference_net_delta_kwh|treatment_net_delta_kwh|effect_net_delta_kwh|effect_net_delta_kwh_lower|effect_net_delta_kwh_upper|effect_net_delta_kwh_unit", "kWh", "strict physical pairs"),
    ("F6", "a", "04_validation/lockbox_first_look_rank.csv", "fold_id|test_event|method|outcome|spearman|kendall_tau_b|first_look", "dimensionless", "immutable lockbox first look"),
    ("F6", "b", "04_validation/lockbox_first_look_capture.csv", "fold_id|test_event|method|outcome|fraction|expected_burden_capture|first_look", "fraction", "immutable lockbox first look"),
    ("F6", "c-f", "04_validation/event_holdout_rank_metrics.csv", "fold_id|test_event|method|outcome|spearman|kendall_tau_b", "dimensionless", "LOEO8 and T1-T3"),
    ("S1", "all", "04_validation/event_holdout_screening_capture.csv", "fold_id|test_event|fraction|method|outcome|expected_burden_capture", "fraction", "5/10/20 percent screening"),
    ("S2", "all", "03_stock/stock_event_coverage.csv", "event_id|rows|evaluable|not_evaluable", "count", "coverage and censoring"),
    ("S3", "all", "05_evidence/future_compatibility.csv", "result|compatibility|final_decision", "status", "future-result compatibility"),
]


def write_dependency_matrices(root: Path, upstream: Path, registry_path: Path | None = None) -> None:
    rows = []
    if registry_path is not None:
        registry = pd.read_csv(registry_path).fillna("")
        required = {"figure_id", "panel_id", "source_table", "columns", "units", "event_domain", "event_column",
                    "primary_key", "time_window", "aggregation_order", "uncertainty_identity", "scope",
                    "disposition", "required_for_final"}
        if missing := sorted(required - set(registry)):
            raise Blocked(f"paper panel registry missing columns: {missing}")
        if registry.duplicated(["figure_id", "panel_id"]).any():
            raise Blocked("paper panel registry has duplicate figure-panel identities")
        for item in registry.to_dict("records"):
            rows.append({**item, "filters": "frozen in analysis contract",
                         "source_hash": "filled_at_freeze", "status": str(item["disposition"])})
    else:
      for figure, panels, table, columns, units, scope in FIGURES:
        if table == "05_evidence/climate_catalog_and_event_roles.csv":
            event_domain, event_column, primary_key = "catalog65", "event_id", "event_id"
        elif table == "02_mechanisms/controlled_pair_results.csv":
            event_domain, event_column, primary_key = "mechanism3", "event_id", "mechanism_id"
        elif "lockbox_first_look" in table:
            event_domain, event_column = "lockbox2", "test_event"
            primary_key = "fold_id|test_event|method|outcome|fraction" if "capture" in table else "fold_id|test_event|method|outcome"
        elif table in {"03_stock/city_event_energy_peak.csv", "03_stock/stock_event_coverage.csv"}:
            event_domain, event_column, primary_key = "core8", "event_id", "event_id"
        elif "event_holdout" in table:
            event_domain, event_column = "core8", "test_event"
            suffix = "|fraction" if "capture" in table else ""
            primary_key = "fold_id|test_event|method|outcome|score_endpoint_scenario" + suffix
        elif table == "03_stock/building_reference_beri.parquet":
            event_domain, event_column, primary_key = "none", "", "BUILDINGSTRUCTUREID"
        else:
            event_domain, event_column, primary_key = "none", "", ""
        rows.append({"figure_id": figure, "panel_id": panels, "source_table": table, "columns": columns,
                     "units": units, "filters": "frozen in analysis contract", "aggregation_order": "source table metadata",
                     "time_window": "complete mother event unless explicitly display window", "uncertainty_identity": "event/design/censoring separated",
                     "scope": scope, "event_domain": event_domain, "event_column": event_column,
                     "primary_key": primary_key, "source_hash": "filled_at_freeze", "status": "recompute",
                     "required_for_final": True, "disposition": "recompute"})
    contract_path = root / "06_source_data/figure_source_data_contract.csv"
    proposed = pd.DataFrame(rows)
    if contract_path.exists():
        existing = pd.read_csv(contract_path).fillna("")
        identity = [c for c in proposed.columns if c not in {"source_hash", "status"}]
        same_identity = (set(identity) <= set(existing) and len(existing) == len(proposed) and
                         existing[identity].astype(str).equals(proposed[identity].astype(str)))
        if not same_identity and existing.get("status", pd.Series(dtype=str)).eq("verified_source_table").any():
            raise Blocked("paper panel registry changed after Source Data were verified")
        if same_identity:
            proposed["source_hash"] = existing["source_hash"].values
            proposed["status"] = existing["status"].values
    atomic_csv(contract_path, proposed)
    evidence = [
        ("65-event catalog and trends", "verified_reuse", "unchanged climate catalog; verify hashes"),
        ("942-grid hazard background", "verified_reuse", "not per-building hourly forcing"),
        ("Open6/lockbox/mechanism building physics", "approved_new_simulation", "current E8 contracts"),
        ("PolyU", "verified_reuse", "non-residential observed load response only"),
        ("HKUST", "verified_reuse", "limited retrospective hourly prediction; time semantics limitation retained"),
        ("CU-BEMS", "verified_reuse", "commercial load/indoor dynamics within actual coverage"),
        ("NIST", "verified_reuse", "directional residential HVAC/thermal-inertia checks; not local HK validation"),
        ("future five-model event energy", "blocked_with_reason", "compatibility with new physics/score contracts must be established"),
        ("representative-model future thermal response", "blocked_with_reason", "cannot be called five-model thermal conclusion"),
    ]
    evidence_path = root / "05_evidence/paper_result_dependency_matrix.csv"
    if not evidence_path.exists():
        atomic_csv(evidence_path, pd.DataFrame(evidence, columns=["paper_result", "status", "reason_or_action"]))
    future = pd.DataFrame([
        {"result": "five_model_event_energy", "compatibility": "pending_hash_audit", "default_decision": "retain_as_background_only_if_compatible",
         "alternative": "exclude_from_main_text", "requires_new_solver": "only if explicitly approved"},
        {"result": "representative_model_thermal", "compatibility": "pending_hash_audit", "default_decision": "supplement_only_if_compatible",
         "alternative": "exclude", "requires_new_solver": "only if explicitly approved"},
    ])
    future_path = root / "05_evidence/future_compatibility.csv"
    if not future_path.exists():
        atomic_csv(future_path, future)


def finalize_source_contract(root: Path) -> pd.DataFrame:
    path = root / "06_source_data/figure_source_data_contract.csv"
    contract = pd.read_csv(path)
    issues = []
    for index, row in contract.iterrows():
        source = root / str(row.source_table)
        if not source.exists() or source.stat().st_size == 0:
            issues.append(str(row.source_table)); continue
        table = pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)
        required = [value for value in str(row["columns"]).split("|") if value]
        missing = sorted(set(required) - set(table.columns))
        if table.empty or missing:
            issues.append(f"{row.source_table}:empty_or_missing_columns={missing}"); continue
        event_domain = str(getattr(row, "event_domain", "none"))
        event_column = str(getattr(row, "event_column", ""))
        expected_domains = {
            "core8": set(ALL_EVENTS),
            "lockbox2": {"HKO_LI_0054", "HKO_LI_0066"},
            "mechanism3": {"HKO_LI_0068", "HKO_LI_0085", "HKO_LI_0090"},
        }
        if event_domain == "catalog65":
            if event_column not in table or table[event_column].nunique(dropna=True) != 65:
                issues.append(f"{row.source_table}:catalog_event_domain_not_65"); continue
        elif event_domain in expected_domains:
            if event_column not in table or set(table[event_column].dropna().astype(str)) != expected_domains[event_domain]:
                issues.append(f"{row.source_table}:event_domain_mismatch:{event_domain}"); continue
        primary_key = [value for value in str(getattr(row, "primary_key", "")).split("|") if value and value != "nan"]
        if primary_key:
            missing_key = sorted(set(primary_key) - set(table.columns))
            if missing_key or table.duplicated(primary_key).any():
                issues.append(f"{row.source_table}:invalid_primary_key={primary_key}:missing={missing_key}"); continue
        contract.at[index, "source_hash"] = sha256(source)
        contract.at[index, "status"] = "verified_source_table"
    atomic_csv(path, contract)
    if issues: raise Blocked(f"figure Source Data missing: {issues}")
    return contract


def verify_final(root: Path, task_matrix: Path) -> dict:
    profile_path = root / "00_protocol/execution_profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8")) if profile_path.exists() else {}
    is_acceptance = profile.get("profile") == "acceptance_fixture"
    expected_counts = profile.get("expected_counts", {}) if is_acceptance else {}
    group_count = int(expected_counts.get("core_groups", 2376))
    score_count = int(expected_counts.get("stage_scores", group_count * 4))
    logical_count = int(expected_counts.get("logical_runs", group_count * 5))
    building_count = int(expected_counts.get("buildings", 330761))
    source_path = root / "06_source_data/figure_source_data_contract.csv"
    registered_rows = len(pd.read_csv(source_path)) if source_path.exists() else len(FIGURES)
    source_contract_count = int(expected_counts.get("source_contract_rows", registered_rows))
    final_label = "ACCEPTANCE_PROFILE_COMPLETE" if is_acceptance else FINAL_LABEL
    required = pd.read_csv(task_matrix)
    state_path = root / "00_protocol/task_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"tasks": {}}
    issues = []
    decisions_path = root / "00_protocol/decisions_required.csv"
    if decisions_path.exists():
        decisions = pd.read_csv(decisions_path)
        accepted = {"complete", "approved", "approved_and_computed", "excluded_by_scope_approval",
                    "not_required_by_approved_scope"}
        if {"required", "status", "item"}.issubset(decisions.columns):
            required_mask = decisions.required.astype(str).str.lower().isin(["true", "1", "yes"])
            unresolved = decisions[required_mask & ~decisions.status.isin(accepted)]
            issues.extend(f"required_decision:{item}:{status}" for item, status in
                          unresolved[["item", "status"]].itertuples(index=False, name=None))
        else:
            issues.append("required_decisions:invalid_schema")
    for row in required[required.required.astype(str).str.lower().isin(["true", "1", "yes"])].itertuples(index=False):
        if row.task_id == "T15_release":
            continue
        status = state.get("tasks", {}).get(row.task_id, {}).get("status")
        if status != "complete":
            issues.append(f"{row.task_id}:{status or 'missing'}")
    expected = {
        "core_group_rows": (root / "01_core_e8/e8_design_event_energy_service.csv", group_count),
        "stage_scores": (root / "01_core_e8/e8_stage_score_bounds.csv", score_count),
        "logical_source_index": (root / "01_core_e8/e8_logical_physical_source_index.csv", logical_count),
        "mechanism_effects": (root / "02_mechanisms/controlled_pair_results.csv", 1),
        "stock_coverage": (root / "03_stock/stock_event_coverage.csv", 8),
        "reference_rating": (root / "03_stock/building_reference_beri.parquet", building_count),
        "lockbox_first_look": (root / "04_validation/lockbox_first_look_rank.csv", 1),
        "rank_folds": (root / "04_validation/event_holdout_rank_metrics.csv", 1),
        "source_contract": (root / "06_source_data/figure_source_data_contract.csv", source_contract_count),
    }
    checks = {}
    for name, (path, minimum) in expected.items():
        ok = path.exists() and path.stat().st_size > 0
        rows = len(pd.read_csv(path)) if ok and path.suffix == ".csv" else (
            pq.ParquetFile(path).metadata.num_rows if ok and path.suffix == ".parquet" else None)
        if not ok or (rows is not None and rows < minimum):
            issues.append(f"{name}:missing_or_incomplete")
        checks[name] = {"path": str(path), "exists": ok, "rows": rows, "sha256": sha256(path) if ok else None}
    coverage_path = root / "03_stock/stock_event_coverage.csv"
    if coverage_path.exists():
        coverage = pd.read_csv(coverage_path)
        if set(coverage.event_id) != set(ALL_EVENTS) or not coverage.rows.eq(building_count).all():
            issues.append(f"stock_coverage:not_{building_count}_rows_for_each_of_8_events")
    rank_path = root / "04_validation/event_holdout_rank_metrics.csv"
    if rank_path.exists():
        folds = set(pd.read_csv(rank_path).fold_id.unique())
        expected_folds = {f"LOEO_{event}" for event in ALL_EVENTS} | {"T1", "T2", "T3"}
        if folds != expected_folds: issues.append("rank_folds:not_exactly_LOEO8_plus_T1_T2_T3")
    first_path = root / "04_validation/lockbox_first_look_rank.csv"
    if first_path.exists() and set(pd.read_csv(first_path).test_event.unique()) != {"HKO_LI_0054", "HKO_LI_0066"}:
        issues.append("lockbox_first_look:both_events_not_present")
    exact_specs = [
        ("energy_keys", root / "01_core_e8/e8_design_event_energy_service.csv",
         ["event_id", "base_case_id", "prototype_id", "floor_position"], group_count),
        ("thermal_keys", root / "01_core_e8/e8_design_event_thermal_recovery.csv",
         ["event_id", "base_case_id", "prototype_id", "floor_position"], group_count),
        ("logical_keys", root / "01_core_e8/e8_logical_physical_source_index.csv",
         ["event_id", "base_case_id", "prototype_id", "floor_position", "scenario_id"], logical_count),
    ]
    for label, path, keys, exact in exact_specs:
        if path.exists():
            frame = pd.read_csv(path)
            if len(frame) != exact or frame.duplicated(keys).any() or set(frame.event_id) != set(ALL_EVENTS):
                issues.append(f"{label}:cardinality_identity_or_primary_key_failure")
    score_path = root / "01_core_e8/e8_stage_score_bounds.csv"
    if score_path.exists():
        score = pd.read_csv(score_path)
        score_keys = ["event_id", "base_case_id", "prototype_id", "floor_position", "component"]
        support = score.groupby(score_keys[:-1]).component.agg(lambda x: set(x.astype(str)))
        if len(score) != score_count or score.duplicated(score_keys).any() or not support.map(lambda x: x == set("NCOR")).all():
            issues.append("stage_scores:exact_NCOR_support_or_primary_key_failure")
    receipt_path = root / "00_protocol/lockbox_prediction_receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        for label in ("prediction", "cohorts", "baselines", "rules", "upstream_receipt"):
            path = Path(receipt.get(label + "_path", ""))
            if not path.is_file() or sha256(path) != receipt.get(label + "_sha256"):
                issues.append(f"lockbox_prediction_receipt:{label}_hash_failure")
    contract_path = root / "06_source_data/figure_source_data_contract.csv"
    if contract_path.exists():
        contract = pd.read_csv(contract_path)
        for row in contract.itertuples(index=False):
            source = root / str(row.source_table)
            if not source.is_file() or sha256(source) != row.source_hash:
                issues.append(f"source_contract:stale_hash:{row.figure_id}:{row.panel_id}")
    complete = not issues
    if is_acceptance and profile.get("energyplus_calls") != 0:
        issues.append("acceptance_profile:energyplus_calls_not_zero")
        complete = False
    return {"checked_at": time.time(), "complete": complete,
            "status": final_label if complete else "PARTIAL_OR_BLOCKED", "profile": profile.get("profile", "production"),
            "production_ready": bool(complete and not is_acceptance), "issues": issues, "checks": checks}


def freeze_release(root: Path, task_matrix: Path) -> dict:
    finalize_source_contract(root)
    verification = verify_final(root, task_matrix)
    release = root / "07_release"
    release.mkdir(parents=True, exist_ok=True)
    atomic_json(release / "final_verification.json", verification)
    if not verification["complete"]:
        return verification
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and "runtime" not in path.parts and path.name != "numerical_catalog.csv":
            files.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    atomic_csv(release / "numerical_catalog.csv", pd.DataFrame(files))
    # Findings are generated from frozen tables; they are not pre-populated claims.
    energy = pd.read_csv(root / "01_core_e8/e8_design_event_energy_service.csv")
    city = pd.read_csv(root / "03_stock/city_event_energy_peak.csv")
    rating = pd.read_parquet(root / "03_stock/building_reference_beri.parquet")
    peak = city.loc[pd.to_numeric(city.simultaneous_HW_peak_kw, errors="coerce").idxmax()]
    findings = {"status": FINAL_LABEL, "events": list(ALL_EVENTS), "generated_from": "frozen numerical tables",
                "design_event_groups": len(energy), "building_ratings": len(rating),
                "beri_evaluable": int(rating.BERI_lower.notna().sum()),
                "largest_simultaneous_HW_peak_event": str(peak.event_id),
                "largest_simultaneous_HW_peak_kw": float(peak.simultaneous_HW_peak_kw),
                "source_hashes": {"energy": sha256(root / "01_core_e8/e8_design_event_energy_service.csv"),
                                  "city": sha256(root / "03_stock/city_event_energy_peak.csv"),
                                  "rating": sha256(root / "03_stock/building_reference_beri.parquet")},
                "claims_require_manuscript_interpretation": True}
    atomic_json(release / "key_findings.json", findings)
    state_path = root / "00_protocol/task_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["completion_label"] = verification["status"]
    state["last_error"] = None
    state["updated_at"] = time.time()
    atomic_json(state_path, state)
    marker = release / verification["status"]
    writing = release / f"{verification['status']}.writing"
    writing.write_text(verification["status"] + "\n", encoding="utf-8")
    writing.replace(marker)
    return verification


def package_review(root: Path, task_matrix: Path) -> Path:
    verification = verify_final(root, task_matrix)
    release = root / "07_release"; release.mkdir(parents=True, exist_ok=True)
    atomic_json(release / "final_verification.json", verification)
    archive = release / "E8_completion_review.zip"
    allowed = ("00_protocol", "01_core_e8", "02_mechanisms", "04_validation", "05_evidence", "06_source_data", "07_release")
    profile_path = root / "00_protocol/execution_profile.json"
    package_profile = json.loads(profile_path.read_text(encoding="utf-8")) if profile_path.exists() else {}
    workspace = Path(package_profile.get("code_root", Path(task_matrix).resolve().parent.parent)).resolve()
    def included_entries():
        selected = []
        for folder in allowed:
            base = root / folder
            if not base.exists():
                continue
            for path in base.rglob("*"):
                if (path.is_file() and path.suffix.lower() not in {".parquet", ".sql", ".eso", ".mtr", ".zip", ".writing", ".sha256"}
                        and ".writing." not in path.name and path != archive):
                    selected.append((path, str(path.relative_to(root)).replace("\\", "/")))
        code_patterns = ("src/utils/e8_completion_*.py", "src/workflows/run_e8*.py",
                         "config/completion_*", "config/e8_completion.json",
                         "reports/REVIEW_E8_COMPLETION_V7_IMPLEMENTATION.md", "reports/V1_3_MIGRATION_PLAN.md",
                         "TASK_PLAN.md", "PROGRESS.md", "DECISIONS_REQUIRED_FINAL.md")
        for pattern in code_patterns:
            for path in workspace.glob(pattern):
                if path.is_file() and ".writing." not in path.name:
                    selected.append((path, "review_source/" + str(path.relative_to(workspace)).replace("\\", "/")))
        return sorted(set(selected), key=lambda item: item[1])
    manifest_path = release / "review_archive_manifest.csv"
    before = [(path, archive_name) for path, archive_name in included_entries() if path != manifest_path]
    atomic_csv(manifest_path, pd.DataFrame([{"path": archive_name, "sha256": sha256(path),
                                             "bytes": path.stat().st_size} for path, archive_name in before]))
    selected = included_entries()
    with zipfile.ZipFile(archive.with_suffix(".zip.writing"), "w", zipfile.ZIP_DEFLATED) as z:
        for path, archive_name in selected:
            z.write(path, archive_name)
    archive.with_suffix(".zip.writing").replace(archive)
    with zipfile.ZipFile(archive) as z:
        names = set(z.namelist())
        expected_names = {archive_name for _, archive_name in selected}
        if names != expected_names or any(name.lower().endswith((".zip", ".writing")) for name in names):
            raise Blocked("review archive content does not match its frozen include set")
    (archive.with_suffix(".zip.sha256")).write_text(sha256(archive) + "  " + archive.name + "\n", encoding="utf-8")
    return archive
