"""Memory-bounded propagation from design/event metrics to the fixed stock.

Energy-like quantities are additive by component area.  Thermal screening
quantities are composed from eligible component consequences; unavailable
simultaneous trajectories are never replaced by maxima of independent recovery
times.  Those rows remain explicitly non-evaluable until the trajectory stage
has supplied a simultaneous result.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from utils.e8_completion_common import ALL_EVENTS, Blocked, atomic_csv, atomic_parquet
from utils import e8_g3_postprocess as science


POPULATION_FIELD = "allocated_population__ssug_landuse_residential_gfa"
REGION_BOUNDS = {
    "Ho Man Tin": (836126.18034143, 818910.80931283, 837345.03745461, 820422.94111362),
    "Mong Kok": (835132.28, 819514.11, 836016.10, 820302.95),
    "Tsim Sha Tsui": (835001.12, 817109.65, 836317.21, 818280.89),
}


def floor_weights(floors: float) -> dict[str, float]:
    n = max(1, int(round(float(floors))))
    if n == 1:
        return {"single": 1.0}
    if n == 2:
        return {"bottom": 0.5, "top": 0.5}
    return {"bottom": 1 / n, "middle": (n - 2) / n, "top": 1 / n}


def stock_components(component_map: pd.DataFrame, attributes: pd.DataFrame, expected_count: int = 330761) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = component_map[["BUILDINGSTRUCTUREID"]].drop_duplicates()
    if len(ids) != expected_count:
        raise Blocked(f"fixed-stock ID coverage is {len(ids)}, expected {expected_count}")
    attrs = attributes.drop_duplicates("BUILDINGSTRUCTUREID")
    keep = ["BUILDINGSTRUCTUREID", "construction_year_value", POPULATION_FIELD, "ZONE_LABEL", "geometry"]
    buildings = component_map.merge(attrs[[c for c in keep if c in attrs]], on="BUILDINGSTRUCTUREID", how="left", validate="one_to_one")
    buildings["population_reference_2021"] = pd.to_numeric(buildings.get(POPULATION_FIELD), errors="coerce")
    main_class = buildings["main_class"].astype(str)
    known_class = main_class.isin(["Residential", "Commercial", "Mixed-use"])
    residential_required = main_class.isin(["Residential", "Mixed-use"])
    commercial_required = main_class.isin(["Commercial", "Mixed-use"])
    residential_required = residential_required.where(
        known_class, buildings["residential_prototype_id"].notna())
    commercial_required = commercial_required.where(
        known_class, buildings["commercial_prototype_id"].notna())
    residential_area = pd.to_numeric(buildings["residential_gfa_m2"], errors="coerce").fillna(0.0)
    commercial_area = pd.to_numeric(buildings["commercial_gfa_m2"], errors="coerce").fillna(0.0)
    buildings["modeled_component_area_m2"] = (
        residential_area.where(residential_required, 0.0)
        + commercial_area.where(commercial_required, 0.0)
    )
    buildings["unmapped_gfa_m2"] = (
        pd.to_numeric(buildings["gfa_value_m2"], errors="coerce")
        - buildings["modeled_component_area_m2"]
    ).clip(lower=0.0)
    buildings["mapped_gfa_fraction"] = (
        buildings["modeled_component_area_m2"]
        / pd.to_numeric(buildings["gfa_value_m2"], errors="coerce").replace(0, np.nan)
    )
    buildings["area_mapping_status"] = np.where(
        buildings["unmapped_gfa_m2"].le(
            pd.to_numeric(buildings["gfa_value_m2"], errors="coerce").clip(lower=1) * 1e-6),
        "full_target_component_coverage",
        "partial_target_component_coverage",
    )
    rows = []
    for r in buildings.itertuples(index=False):
        for sector, area_name, prototype_name in (("residential", "residential_gfa_m2", "residential_prototype_id"),
                                                   ("commercial", "commercial_gfa_m2", "commercial_prototype_id")):
            area = float(getattr(r, area_name) or 0)
            prototype = getattr(r, prototype_name)
            if area <= 0 or pd.isna(prototype):
                continue
            for role, weight in floor_weights(getattr(r, "floors_value")).items():
                rows.append({"BUILDINGSTRUCTUREID": r.BUILDINGSTRUCTUREID, "component": sector,
                             "prototype_id": prototype, "floor_position": role,
                             "component_area_m2": area * weight, "component_fraction_within_sector": weight,
                             "population_reference_2021": (getattr(r, "population_reference_2021", np.nan) * weight
                                                            if sector == "residential" else 0.0)})
    components = pd.DataFrame(rows)
    area = components.groupby("BUILDINGSTRUCTUREID").component_area_m2.sum()
    expected = buildings.set_index("BUILDINGSTRUCTUREID").modeled_component_area_m2
    mismatch = ((area - expected).abs() / expected.clip(lower=1)).fillna(1)
    if (mismatch > 1e-6).any():
        raise Blocked(f"component-area conservation failed for {(mismatch > 1e-6).sum()} buildings")
    return buildings, components


def aggregate_design_metrics(energy: pd.DataFrame, thermal: pd.DataFrame, scores: pd.DataFrame,
                             prototype_areas: pd.DataFrame) -> pd.DataFrame:
    e = energy.copy(); t = thermal.copy()
    keys = ["event_id", "base_case_id", "prototype_id", "floor_position"]
    for label, frame in (("energy", e), ("thermal", t)):
        missing = [column for column in keys if column not in frame]
        if missing or frame.duplicated(keys).any():
            raise Blocked(f"{label} design table has missing keys or duplicates: {missing}")
    required_score = keys + ["component", "score_lower", "score_upper", "NA_reason"]
    if missing := [column for column in required_score if column not in scores]:
        raise Blocked(f"score table missing required columns: {missing}")
    if scores.duplicated(keys + ["component"]).any():
        raise Blocked("score table has duplicate design-event components")
    support = scores.groupby(keys, dropna=False).component.agg(lambda x: tuple(sorted(x.astype(str)))).reset_index()
    if not support.component.map(lambda value: value == tuple(sorted("NCOR"))).all():
        raise Blocked("each design-event requires exactly N,C,O,R score rows")
    for bound in ("lower", "upper"):
        scores["score_" + bound] = pd.to_numeric(scores["score_" + bound], errors="coerce")
    partial = scores.score_lower.isna() ^ scores.score_upper.isna()
    invalid = scores.score_lower.notna() & ((scores.score_lower < 0) | (scores.score_upper > 1) |
                                            (scores.score_lower > scores.score_upper))
    unexplained = scores.score_lower.isna() & scores.NA_reason.fillna("").astype(str).str.strip().eq("")
    if partial.any() or invalid.any() or unexplained.any():
        raise Blocked("score intervals are partial, invalid, or missing an NA reason")
    score = scores.set_index(keys + ["component"])[["score_lower", "score_upper"]].unstack("component")
    score.columns = [f"score_{component}_{bound.split('_')[-1]}" for bound, component in score.columns]
    score = score.reset_index()
    frame = e.merge(t, on=keys, suffixes=("", "_thermal"), validate="one_to_one").merge(
        score, on=keys, how="left", validate="one_to_one", indicator=True)
    if not frame._merge.eq("both").all() or len(frame) != len(e) or len(frame) != len(t):
        raise Blocked("energy, thermal and score design support differ")
    frame = frame.drop(columns="_merge")
    frame = frame.merge(prototype_areas[["prototype_id", "floor_area_m2"]].rename(columns={"floor_area_m2": "model_floor_area_m2"}),
                        on="prototype_id", how="left", validate="many_to_one")
    additive = [c for c in ("E_HW_kwh", "E_CF_kwh", "net_delta_kwh", "sum_positive_timestep_delta_kwh",
                            "required_ref_kwh_thermal", "delivered_raw_kwh_thermal", "shortfall_ref_kwh_thermal") if c in frame]
    # Design cases represent the same response unit.  Aggregate by median only after normalising by model floor area.
    if frame.model_floor_area_m2.isna().any() or frame.model_floor_area_m2.le(0).any():
        raise Blocked("prototype floor area is missing or non-positive")
    for col in additive:
        frame[col + "_per_m2"] = pd.to_numeric(frame[col], errors="coerce") / frame.model_floor_area_m2
    group = ["event_id", "prototype_id", "floor_position"]
    numeric = [c for c in frame.columns if c.endswith("_per_m2") or c.startswith("score_") or c in (
        "absolute_occupied_degree_hours_above_32c", "absolute_occupied_degree_hours_above_26c",
        "positive_extra_occupied_degree_hours_above_26c_vs_rated", "first_crossing_lower_h", "first_crossing_upper_h",
        "recovery_lower_h", "recovery_upper_h", "maximum_occupied_outage_temperature_c", "effective_monitoring_hours")]
    def complete_median(values):
        numeric_values = pd.to_numeric(values, errors="coerce")
        return float(numeric_values.median()) if numeric_values.notna().all() else np.nan
    result = frame.groupby(group, as_index=False)[numeric].agg(complete_median)
    counts = frame.groupby(group, as_index=False).base_case_id.agg(design_case_count="nunique")
    result = result.merge(counts, on=group, validate="one_to_one")
    result["complete_design_support"] = frame.groupby(group)[numeric].apply(
        lambda x: bool(x.notna().all().all()), include_groups=False).to_numpy()
    return result


def propagate_event(buildings: pd.DataFrame, components: pd.DataFrame, response: pd.DataFrame, event_id: str) -> pd.DataFrame:
    r = response[response.event_id.eq(event_id)]
    joined = components.merge(r, on=["prototype_id", "floor_position"], how="left", validate="many_to_one", indicator=True)
    additive = [c for c in r.columns if c.endswith("_per_m2")]
    for col in additive:
        joined[col.replace("_per_m2", "")] = joined[col] * joined.component_area_m2
    rows = []
    for building_id, g in joined.groupby("BUILDINGSTRUCTUREID", sort=False):
        out = {"BUILDINGSTRUCTUREID": building_id, "event_id": event_id,
               "component_rows": len(g), "mapping_complete": bool(g._merge.eq("both").all()),
               "prototype_floor_signature": "|".join(sorted(set(g.prototype_id + "@" + g.floor_position)))}
        if not out["mapping_complete"]:
            out["evaluation_status"] = "not_evaluable_missing_response_unit"
            out["NA_reason"] = "one_or_more_prototype_floor_response_units_missing"
        else:
            out["evaluation_status"] = "evaluable_summary"
            out["NA_reason"] = ""
        for col in [c.replace("_per_m2", "") for c in additive]:
            out[col] = g[col].sum(min_count=len(g))
        for col in [c for c in r if c.startswith("score_")]:
            # Worst component is retained for service screening; not an area mean.
            out[col] = g[col].min(skipna=False)
        out["cooling_shortfall_kwh"] = out.get("shortfall_ref_kwh_thermal", np.nan)
        total_area = g.component_area_m2.sum(min_count=len(g))
        for source, target in (("absolute_occupied_degree_hours_above_32c", "absolute_dh32"),
                               ("positive_extra_occupied_degree_hours_above_26c_vs_rated", "extra_dh26")):
            out[target] = ((g[source] * g.component_area_m2).sum(min_count=len(g)) / total_area
                           if source in g and pd.notna(total_area) and total_area > 0 else np.nan)
            out["worst_component_" + target] = g[source].max(skipna=False) if source in g else np.nan
        out["crossing_lower_h"] = g.first_crossing_lower_h.min(skipna=False) if "first_crossing_lower_h" in g else np.nan
        out["crossing_upper_h"] = g.first_crossing_upper_h.min(skipna=False) if "first_crossing_upper_h" in g else np.nan
        out["effective_monitoring_hours"] = g.effective_monitoring_hours.min(skipna=False) if "effective_monitoring_hours" in g else np.nan
        # Simultaneous recovery cannot be inferred from independent component first-recovery times.
        out["recovery_hours"] = np.nan
        out["recovery_status"] = "requires_simultaneous_component_trajectory_composition"
        residential = g[g.component.eq("residential")]
        out["population_degree_hours"] = ((residential.absolute_occupied_degree_hours_above_32c *
                                             residential.population_reference_2021).sum(min_count=len(residential))
                                            if len(residential) and "absolute_occupied_degree_hours_above_32c" in residential else 0.0)
        rows.append(out)
    result = pd.DataFrame(rows)
    return buildings[["BUILDINGSTRUCTUREID", "main_class", "sub_class", "gfa_value_m2", "construction_year_value",
                      "population_reference_2021", "wrf_south_north", "wrf_west_east"]].merge(
                          result, on="BUILDINGSTRUCTUREID", how="left", validate="one_to_one")


def propagate_stock(
    upstream: Path,
    response: pd.DataFrame,
    output_dir: Path,
    prepared_stock_root: Path | None = None,
) -> dict:
    source = Path(prepared_stock_root) if prepared_stock_root else upstream / "intermediate/prepared_stock"
    component_map = pd.read_parquet(source / "building_hourly_component_map.parquet")
    attributes = pd.read_parquet(source / "all_sector_building_model_inputs.parquet")
    buildings, components = stock_components(component_map, attributes)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_parquet(output_dir / "fixed_stock_buildings.parquet", buildings)
    atomic_parquet(output_dir / "building_component_map.parquet", components)
    try:
        from shapely import from_wkb, get_x, get_y
        geometry = from_wkb(buildings.geometry.to_numpy())
        centroid = np.array([g.centroid if g is not None else None for g in geometry], dtype=object)
        links = buildings[["BUILDINGSTRUCTUREID", "ZONE_LABEL"]].copy()
        links["centroid_x"] = [g.x if g is not None else np.nan for g in centroid]
        links["centroid_y"] = [g.y if g is not None else np.nan for g in centroid]
        links["study_region"] = "Hong Kong total"
        region_rows = [links]
        for name, (xmin, ymin, xmax, ymax) in REGION_BOUNDS.items():
            subset = links[links.centroid_x.between(xmin, xmax) & links.centroid_y.between(ymin, ymax)].copy()
            subset["study_region"] = name; region_rows.append(subset)
        atomic_parquet(output_dir / "building_spatial_links.parquet", pd.concat(region_rows, ignore_index=True))
    except ImportError:
        raise Blocked("shapely is required to create real building-to-region spatial links")
    counts = []
    for event in ALL_EVENTS:
        frame = propagate_event(buildings, components, response, event)
        path = output_dir / "events" / f"{event}.parquet"
        atomic_parquet(path, frame)
        counts.append({"event_id": event, "rows": len(frame), "evaluable": int(frame.mapping_complete.fillna(False).sum()),
                       "not_evaluable": int((~frame.mapping_complete.fillna(False)).sum())})
    atomic_csv(output_dir / "stock_event_coverage.csv", pd.DataFrame(counts))
    links = pd.read_parquet(output_dir / "building_spatial_links.parquet")
    region_rows = []
    for event in ALL_EVENTS:
        frame = pd.read_parquet(output_dir / "events" / f"{event}.parquet")
        joined = links.merge(frame, on="BUILDINGSTRUCTUREID", how="left", validate="many_to_one")
        for region, group in joined.groupby("study_region"):
            region_rows.append({"event_id": event, "study_region": region, "buildings": group.BUILDINGSTRUCTUREID.nunique(),
                                "E_HW_kwh": group.get("E_HW_kwh", pd.Series(dtype=float)).sum(min_count=1),
                                "net_delta_kwh": group.get("net_delta_kwh", pd.Series(dtype=float)).sum(min_count=1),
                                "population_degree_hours": group.population_degree_hours.sum(min_count=1),
                                "not_evaluable_buildings": int(group.evaluation_status.ne("evaluable_summary").sum())})
    atomic_csv(output_dir / "region_event_summary.csv", pd.DataFrame(region_rows))
    return {"buildings": len(buildings), "component_rows": len(components), "events": len(counts),
            "all_event_rows": int(sum(x["rows"] for x in counts))}


def city_event_energy_trajectories(run_index: pd.DataFrame, components: pd.DataFrame,
                                   prototype_areas: pd.DataFrame, event_dates: pd.DataFrame,
                                   output_dir: Path) -> pd.DataFrame:
    """Aggregate simultaneous city power from median design trajectories.

    This never sums individual-building maxima.  The fixed-stock component area
    is applied to prototype/floor power intensity at every native timestamp.
    """
    def hkt(value):
        stamp = pd.Timestamp(value)
        return stamp.tz_localize("Asia/Hong_Kong") if stamp.tzinfo is None else stamp.tz_convert("Asia/Hong_Kong")

    area = prototype_areas.set_index("prototype_id").floor_area_m2.to_dict()
    stock_area = components.groupby(["prototype_id", "floor_position"], as_index=False).component_area_m2.sum()
    summaries = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for event_id in ALL_EVENTS:
        dates = event_dates[event_dates.parent_event_id.eq(event_id)].iloc[0]
        start_value = dates.mother_start_hkt if "mother_start_hkt" in dates else dates.event_start
        end_value = dates.mother_end_hkt if "mother_end_hkt" in dates else dates.event_end
        start = hkt(start_value)
        end = hkt(end_value) + pd.Timedelta(days=1)
        scenario_frames = {}
        for scenario in ("CF_UNL", "HW_UNL"):
            pieces = []
            subset = run_index[run_index.event_id.eq(event_id) & run_index.scenario_id.eq(scenario)]
            for row in subset.itertuples(index=False):
                path = Path(row.source_result_dir) / "parsed/native_timestep_building.parquet"
                if not path.exists(): raise Blocked(f"parsed trajectory missing: {path}")
                d = pd.read_parquet(path, columns=["timestamp_hkt", "total_modeled_electricity_j"])
                d.timestamp_hkt = d.timestamp_hkt.map(hkt)
                d = d[(d.timestamp_hkt > start) & (d.timestamp_hkt <= end)].copy()
                d["power_intensity_w_m2"] = d.total_modeled_electricity_j / 900.0 / float(area[row.prototype_id])
                d["prototype_id"] = row.prototype_id; d["floor_position"] = row.floor_position
                pieces.append(d[["timestamp_hkt", "prototype_id", "floor_position", "power_intensity_w_m2"]])
            design = pd.concat(pieces, ignore_index=True)
            median = design.groupby(["timestamp_hkt", "prototype_id", "floor_position"], as_index=False).power_intensity_w_m2.median()
            weighted = median.merge(stock_area, on=["prototype_id", "floor_position"], how="left", validate="many_to_one")
            weighted["power_kw"] = weighted.power_intensity_w_m2 * weighted.component_area_m2 / 1000.0
            city = weighted.groupby("timestamp_hkt", as_index=False).power_kw.sum()
            city["scenario_id"] = scenario; scenario_frames[scenario] = city
        paired = scenario_frames["HW_UNL"].merge(scenario_frames["CF_UNL"], on="timestamp_hkt", suffixes=("_hw", "_cf"), validate="one_to_one")
        paired["event_id"] = event_id; paired["delta_kw"] = paired.power_kw_hw - paired.power_kw_cf
        atomic_parquet(output_dir / f"{event_id}_city_15min.parquet", paired)
        dt_h = .25
        peak_idx = paired.power_kw_hw.idxmax()
        summaries.append({"event_id": event_id, "E_HW_kwh": paired.power_kw_hw.sum() * dt_h,
                          "E_CF_kwh": paired.power_kw_cf.sum() * dt_h,
                          "net_delta_kwh": paired.delta_kw.sum() * dt_h,
                          "positive_timestep_delta_kwh": paired.delta_kw.clip(lower=0).sum() * dt_h,
                          "negative_timestep_delta_kwh": paired.delta_kw.clip(upper=0).sum() * dt_h,
                          "simultaneous_HW_peak_kw": paired.power_kw_hw.max(),
                          "simultaneous_CF_peak_kw": paired.power_kw_cf.max(),
                          "HW_peak_time_hkt": paired.loc[peak_idx, "timestamp_hkt"],
                          "paired_difference_at_HW_peak_kw": paired.loc[peak_idx, "delta_kw"]})
    result = pd.DataFrame(summaries)
    atomic_csv(output_dir.parent / "city_event_energy_peak.csv", result)
    return result


def apply_simultaneous_recovery(run_index: pd.DataFrame, components: pd.DataFrame,
                                stock_event_dir: Path, trajectory_dir: Path,
                                event_ids=ALL_EVENTS) -> pd.DataFrame:
    """Compose all required component trajectories before measuring recovery."""
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    signatures = (components.groupby("BUILDINGSTRUCTUREID")
                  .apply(lambda g: "|".join(sorted(set(g.prototype_id + "@" + g.floor_position))), include_groups=False)
                  .rename("signature").reset_index())
    members = {sig: [tuple(x.split("@", 1)) for x in sig.split("|")] for sig in signatures.signature.unique()}
    for event in event_ids:
        curves = []; subset = run_index[run_index.event_id.eq(event) & run_index.scenario_id.eq("HW_O24_SYNC")]
        restores = set()
        for item in subset.itertuples(index=False):
            zones = pd.read_parquet(Path(item.source_result_dir) / "parsed/native_timestep_zones.parquet")
            zones["timestamp_hkt"] = zones.timestamp_hkt.map(
                lambda value: (pd.Timestamp(value).tz_localize("Asia/Hong_Kong")
                               if pd.Timestamp(value).tzinfo is None
                               else pd.Timestamp(value).tz_convert("Asia/Hong_Kong")))
            restore = pd.Timestamp(item.outage_end_hkt)
            if restore.tzinfo is None: restore = restore.tz_localize("Asia/Hong_Kong")
            restores.add(restore.isoformat())
            z = zones[zones.structural_conditioning_present & zones.timestamp_hkt.between(restore, restore + pd.Timedelta(hours=24))]
            curve = z.groupby("timestamp_hkt", as_index=False).temperature_c.max()
            curve["prototype_id"] = item.prototype_id; curve["floor_position"] = item.floor_position
            curves.append(curve)
        if len(restores) != 1: raise Blocked(f"recovery clocks differ within {event}: {restores}")
        restore = pd.Timestamp(next(iter(restores)))
        median = pd.concat(curves).groupby(["timestamp_hkt", "prototype_id", "floor_position"], as_index=False).temperature_c.median()
        atomic_parquet(trajectory_dir / f"{event}_prototype_floor_recovery.parquet", median)
        signature_results = []
        for signature, units in members.items():
            parts = []
            for prototype, floor in units:
                part = median[median.prototype_id.eq(prototype) & median.floor_position.eq(floor)][["timestamp_hkt", "temperature_c"]]
                if part.empty: parts = []; break
                parts.append(part.rename(columns={"temperature_c": f"t{len(parts)}"}))
            if not parts:
                signature_results.append({"signature": signature, "recovery_hours": np.nan,
                                          "recovery_status": "missing_component_trajectory", "recovery_lower_h": np.nan,
                                          "recovery_upper_h": np.nan, "score_R_lower": np.nan, "score_R_upper": np.nan})
                continue
            combined = parts[0]
            for part in parts[1:]: combined = combined.merge(part, on="timestamp_hkt", how="outer", validate="one_to_one")
            values = combined.drop(columns="timestamp_hkt").max(axis=1, skipna=False)
            result = science.recovery_diagnostic(combined.timestamp_hkt, values, restore, threshold=26.0, window_hours=24.0)
            score = science.recovery_score_bounds(result)
            signature_results.append({"signature": signature, "recovery_hours": result.get("hours"),
                                      "recovery_status": result.get("status"), "recovery_lower_h": result.get("lower_h"),
                                      "recovery_upper_h": result.get("upper_h"), "score_R_lower": score.get("score_lower"),
                                      "score_R_upper": score.get("score_upper")})
        mapping = signatures.merge(pd.DataFrame(signature_results), on="signature", validate="many_to_one")
        path = stock_event_dir / f"{event}.parquet"; frame = pd.read_parquet(path)
        for col in ("recovery_hours", "recovery_status", "recovery_lower_h", "recovery_upper_h", "score_R_lower", "score_R_upper"):
            if col in frame: frame = frame.drop(columns=col)
        frame = frame.merge(mapping.drop(columns="signature"), on="BUILDINGSTRUCTUREID", how="left", validate="one_to_one")
        atomic_parquet(path, frame)
        summaries.append({"event_id": event, "buildings": len(frame),
                          "evaluable_recovery": int(frame.recovery_status.ne("missing_component_trajectory").sum()),
                          "right_censored": int(frame.recovery_status.eq("right_censored").sum())})
    summary = pd.DataFrame(summaries); atomic_csv(stock_event_dir.parent / "simultaneous_recovery_coverage.csv", summary)
    return summary


def region_snapshot_input_from_runs(run_index: pd.DataFrame, components: pd.DataFrame,
                                    spatial_links: pd.DataFrame, prototype_areas: pd.DataFrame,
                                    event_dates: pd.DataFrame, city_peaks: pd.DataFrame,
                                    snapshot_decision: dict) -> pd.DataFrame:
    """Build only approved static building snapshots from native trajectories.

    This is intentionally not a frame renderer. It reads the exact native
    interval selected by each approved role, aggregates design responses by
    prototype/floor, maps that response to the fixed stock, and returns a
    compact building-role table consumed by the regional analysis function.
    """
    def hkt(value):
        stamp = pd.Timestamp(value)
        return stamp.tz_localize("Asia/Hong_Kong") if stamp.tzinfo is None else stamp.tz_convert("Asia/Hong_Kong")

    required = {"event_id", "prototype_id", "floor_position", "scenario_id", "source_result_dir",
                "outage_start_hkt", "outage_end_hkt", "base_case_id", "zone_areas_json",
                "conditioned_zones_json"}
    if missing := sorted(required - set(run_index)):
        raise Blocked(f"run index cannot support regional snapshots: {missing}")
    area = prototype_areas.set_index("prototype_id").floor_area_m2.to_dict()
    if set(components.prototype_id) - set(area): raise Blocked("prototype floor area missing for regional snapshots")
    if spatial_links.duplicated(["BUILDINGSTRUCTUREID", "study_region"]).any():
        raise Blocked("spatial links contain duplicate building-region records")
    event_start_field = "mother_start_hkt" if "mother_start_hkt" in event_dates else "event_start"
    result = []
    for event in ALL_EVENTS:
        event_rows = run_index[run_index.event_id.eq(event)]
        reference_identity = ["event_id", "base_case_id", "prototype_id", "floor_position"]
        rated_rows = event_rows[event_rows.scenario_id.eq("HW_RATED")]
        if rated_rows.duplicated(reference_identity).any():
            raise Blocked(f"snapshot HW_RATED references are not unique: {event}")
        rated_lookup = rated_rows.set_index(reference_identity, drop=False)
        outage_rows = event_rows[event_rows.scenario_id.eq("HW_O24_SYNC")]
        starts = {hkt(x) for x in outage_rows.outage_start_hkt.dropna()}; restores = {hkt(x) for x in outage_rows.outage_end_hkt.dropna()}
        if len(starts) != 1 or len(restores) != 1: raise Blocked(f"snapshot clocks differ within {event}")
        outage, restore = next(iter(starts)), next(iter(restores))
        date_row = event_dates[event_dates.parent_event_id.eq(event)]
        if len(date_row) != 1: raise Blocked(f"event-date identity is not unique: {event}")
        event_start = hkt(date_row.iloc[0][event_start_field]) + pd.Timedelta(minutes=15)
        peak_row = city_peaks[city_peaks.event_id.eq(event)]
        if len(peak_row) != 1: raise Blocked(f"city peak identity is not unique: {event}")
        targets = {"event_start": event_start, "hw_peak": hkt(peak_row.iloc[0].HW_peak_time_hkt),
                   # Native timestep values are interval means labelled at the interval end.
                   # These roles therefore select the first complete interval after the transition.
                   "outage_start": outage + pd.Timedelta(minutes=15),
                   "restore": restore + pd.Timedelta(minutes=15),
                   "restore_plus_6h": restore + pd.Timedelta(hours=6),
                   "restore_plus_24h": restore + pd.Timedelta(hours=24)}
        targets = {role: targets[role] for role in snapshot_decision["snapshot_roles"]}
        response_rows = []
        for role, target in targets.items():
            scenario = "HW_UNL" if role in {"event_start", "hw_peak"} else "HW_O24_SYNC"
            pieces = []
            for run in event_rows[event_rows.scenario_id.eq(scenario)].itertuples(index=False):
                directory = Path(run.source_result_dir) / "parsed"
                building_path = directory / "native_timestep_building.parquet"
                zone_path = directory / "native_timestep_zones.parquet"
                if not building_path.is_file() or not zone_path.is_file():
                    raise Blocked(f"snapshot source trajectory missing: {run.source_result_dir}")
                building = pd.read_parquet(building_path)
                building["timestamp_hkt"] = pd.to_datetime(building.timestamp_hkt).map(hkt)
                b = building[building.timestamp_hkt.eq(target)]
                power = (pd.to_numeric(b.total_modeled_electricity_j, errors="coerce").iloc[0] / 900 / 1000
                         if len(b) == 1 and "total_modeled_electricity_j" in b else np.nan)
                zones = pd.read_parquet(zone_path); zones["timestamp_hkt"] = pd.to_datetime(zones.timestamp_hkt).map(hkt)
                z = zones[zones.timestamp_hkt.eq(target)].copy()
                reference_key = (run.event_id, run.base_case_id, run.prototype_id, run.floor_position)
                if reference_key not in rated_lookup.index:
                    raise Blocked(f"snapshot HW_RATED reference schedule missing: {reference_key}")
                reference = rated_lookup.loc[reference_key]
                if isinstance(reference, pd.DataFrame):
                    raise Blocked(f"snapshot HW_RATED reference schedule is not unique: {reference_key}")
                try:
                    weather = pd.read_parquet(Path(run.source_result_dir) / "inputs/weather.parquet")
                    expected = science.expected_native_timestamps(weather)
                    positions = np.flatnonzero(expected == target)
                    if len(positions) != 1:
                        raise ValueError(f"target timestamp has {len(positions)} matches in native grid")
                    hour_index = int(positions[0] // 4)
                    support = science.build_expected_monitoring_support(
                        z,
                        pd.read_parquet(Path(run.source_result_dir) / "inputs/event_schedules.parquet").iloc[[hour_index]].reset_index(drop=True),
                        pd.read_parquet(Path(reference.source_result_dir) / "inputs/event_schedules.parquet").iloc[[hour_index]].reset_index(drop=True),
                        science.expected_native_timestamps(weather.iloc[[hour_index]].reset_index(drop=True)),
                        pd.DataFrame([
                            {"zone_id": str(zone), "zone_area_m2": float(zone_area),
                             "structural_conditioning_present": str(zone) in set(json.loads(run.conditioned_zones_json))}
                            for zone, zone_area in json.loads(run.zone_areas_json).items()
                        ]),
                    )
                    z = support[support.timestamp_hkt.eq(target)].copy()
                except (KeyError, TypeError, ValueError) as exc:
                    raise Blocked(f"snapshot monitoring support reconstruction failed for {run.run_id}: {exc}") from exc
                eligible = z.monitoring_required.fillna(False).astype(bool)
                temperature = pd.to_numeric(z.loc[eligible, "temperature_c"], errors="coerce").max()
                pieces.append({"prototype_id": run.prototype_id, "floor_position": run.floor_position,
                               "electric_power_kw": power, "temperature_c": temperature,
                               "thermal_evaluable": bool(eligible.any() and pd.notna(temperature))})
            design = pd.DataFrame(pieces)
            response = design.groupby(["prototype_id", "floor_position"], as_index=False).agg(
                electric_power_kw=("electric_power_kw", "median"), temperature_c=("temperature_c", "median"),
                thermal_evaluable=("thermal_evaluable", "all"))
            response["power_intensity_kw_m2"] = response.apply(
                lambda row: row.electric_power_kw / float(area[row.prototype_id]), axis=1)
            mapped = components.merge(response, on=["prototype_id", "floor_position"], how="left", validate="many_to_one", indicator=True)
            mapped["component_power_kw"] = mapped.power_intensity_kw_m2 * mapped.component_area_m2
            building = mapped.groupby("BUILDINGSTRUCTUREID", as_index=False).agg(
                electric_power_kw=("component_power_kw", lambda x: x.sum(min_count=len(x))),
                temperature_c=("temperature_c", lambda x: x.max(skipna=False)),
                thermal_evaluable=("thermal_evaluable", "all"),
                population_reference_2021=("population_reference_2021", "sum"),
                mapping_complete=("_merge", lambda x: bool(x.eq("both").all())))
            building.loc[~building.mapping_complete, ["electric_power_kw", "temperature_c"]] = np.nan
            building["event_id"] = event; building["snapshot_role"] = role; building["timestamp_hkt"] = target.isoformat()
            response_rows.append(building)
        event_building = pd.concat(response_rows, ignore_index=True)
        if event_building.duplicated(["BUILDINGSTRUCTUREID", "snapshot_role"]).any():
            raise Blocked(f"snapshot building-role records are not unique: {event}")
        result.append(spatial_links[["BUILDINGSTRUCTUREID", "study_region"]].merge(
            event_building, on="BUILDINGSTRUCTUREID", how="left", validate="many_to_many"))
    return pd.concat(result, ignore_index=True)
