"""Limited, auditable EnergyPlus runner for the E8 G3 integration gate.

The module reuses the production prototype, schedule, IDF and EPW builders but
never calls the legacy aggregate parser.  It is intentionally limited to the
approved G3 manifest and records every solver invocation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import psutil
except ImportError:  # Resource evidence remains explicitly unavailable.
    psutil = None

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from utils.e8_preflight import (
    E8_PROTOCOL_VERSION,
    SCORE_DEFINITION_VERSION,
    WEATHER_PROCESSING_VERSION,
    _bounded_humidity_shift,
    _monotone_daily_temperature,
    first_threshold_crossing_hours,
    sha256_file,
    stable_sha256,
    sustained_recovery_hours,
)


SCENARIO_MAP = {
    "CF_UNL": ("counterfactual", "normal", False),
    "HW_UNL": ("heatwave", "normal", False),
    "HW_RATED": ("heatwave", "capacity_100pct", True),
    "HW_C70": ("heatwave", "capacity_70pct", True),
    "HW_O24_SYNC": ("heatwave", "outage_common_24h", True),
    "VALIDATION_NORMAL": ("observed_weather", "normal", False),
}
CORE_SCENARIO_IDS = ("CF_UNL", "HW_UNL", "HW_RATED", "HW_C70", "HW_O24_SYNC")
G3_GROUPS = (
    ("A", "HKO_LI_0049", "B0016", "res_public_rental", "top"),
    ("B", "HKO_LI_0049", "B0006", "com_office", "middle"),
    ("C", "HKO_LI_0049", "B0012", "com_data_centre", "middle"),
    ("D", "HKO_LI_0068", "B0080", "res_old_lowrise_tenement", "top"),
)
HKUST_CALIBRATION = (pd.Timestamp("2023-07-10 00:00"), pd.Timestamp("2023-07-23 23:00"))
HKUST_TEST = (pd.Timestamp("2023-07-24 00:00"), pd.Timestamp("2023-08-06 23:00"))
MAX_G3_CALLS = 30
MAX_G3_WALL_SECONDS = 2 * 60 * 60
MAX_G3_BYTES = 5 * 1024**3
PARSER_VERSION = "e8-zone-native-parser-v4-time-asserted"


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _write_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def load_production(project_root: Path, code_root: Path | None = None):
    """Load frozen production code while keeping runtime data roots explicit."""
    path = (
        (code_root or project_root)
        / "src"
        / "workflows"
        / "stages"
        / "build_building_response_and_exposure.py"
    )
    spec = importlib.util.spec_from_file_location("e8_production_result2", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ENGINE = (
        project_root
        / "software"
        / "energyplus"
        / "EnergyPlus-26.1.0-6f2e40d102-Windows-x86_64"
        / "energyplus.exe"
    )
    return module


def _event_assignment(work_root: Path) -> pd.DataFrame:
    assignment = pd.read_csv(work_root / "01_inputs" / "E8_event_assignment.csv")
    assignment["event_start"] = pd.to_datetime(assignment["event_start"], format="mixed")
    assignment["event_end"] = pd.to_datetime(assignment["event_end"], format="mixed")
    return assignment.set_index("parent_event_id")


def load_event_weather(
    project_root: Path, work_root: Path, event_id: str, prehistory_days: int
) -> pd.DataFrame:
    path = (
        work_root / "01_inputs" / "hourly_weather"
        / f"{event_id}_{WEATHER_PROCESSING_VERSION}.parquet"
    )
    frame = pd.read_parquet(path)
    frame["timestamp_hkt"] = pd.to_datetime(frame["timestamp_hkt"], utc=True).dt.tz_convert(
        "Asia/Hong_Kong"
    )
    assignment = _event_assignment(work_root).loc[event_id]
    start = pd.Timestamp(assignment["event_start"]).tz_localize("Asia/Hong_Kong")
    end = pd.Timestamp(assignment["event_end"]).tz_localize("Asia/Hong_Kong")
    keep_start = start - pd.Timedelta(days=prehistory_days)
    keep_end = end + pd.Timedelta(days=2, hours=23)
    frame = frame.loc[frame["timestamp_hkt"].between(keep_start, keep_end)].copy()
    expected = pd.date_range(keep_start, keep_end, freq="h")
    if len(frame) != len(expected) or not frame["timestamp_hkt"].reset_index(drop=True).equals(
        pd.Series(expected, name="timestamp_hkt")
    ):
        raise ValueError(f"{event_id} lacks a continuous {prehistory_days}-day G3 window")
    frame["event_id"] = event_id
    local_date = frame["timestamp_hkt"].dt.tz_localize(None).dt.normalize()
    frame["event_phase"] = np.select(
        [local_date.between(start.tz_localize(None), end.tz_localize(None)), local_date.lt(start.tz_localize(None))],
        ["heatwave", "prehistory"],
        default="recovery",
    )
    frame["ALLSKY_SFC_SW_DWN"] = pd.to_numeric(frame["ALLSKY_SFC_SW_DWN"], errors="raise")
    frame["WS10M"] = pd.to_numeric(frame["WS10M"], errors="raise")
    frame["PRECTOTCORR"] = pd.to_numeric(frame["PRECTOTCORR"], errors="raise")
    frame["atmospheric_pressure_pa"] = pd.to_numeric(frame["PS"], errors="raise") * 1000.0
    return frame


def add_integration_counterfactual(project_root: Path, weather: pd.DataFrame) -> pd.DataFrame:
    """Add the legacy 2016--2025 calendar-day target, explicitly integration-only."""
    daily_path = project_root / "intermediate" / "source_derived" / "weather" / "hko_kp_daily_2016_2025.csv"
    daily = pd.read_csv(daily_path, parse_dates=["date"])
    daily["month_day"] = daily["date"].dt.strftime("%m-%d")
    typical = (
        daily.groupby("month_day", as_index=False)[["tmin_c", "tmean_c", "tmax_c", "rh_mean_pct"]]
        .median()
        .rename(
            columns={
                "tmin_c": "cf_tmin_c",
                "tmean_c": "cf_tmean_c",
                "tmax_c": "cf_tmax_c",
                "rh_mean_pct": "cf_rh_mean_pct",
            }
        )
    )
    frame = weather.copy()
    frame["local_date"] = frame["timestamp_hkt"].dt.tz_localize(None).dt.normalize()
    frame["month_day"] = frame["local_date"].dt.strftime("%m-%d")
    frame = frame.merge(typical, on="month_day", how="left", validate="many_to_one")
    frame["counterfactual_temperature_c"] = frame["temperature_c"]
    frame["counterfactual_relative_humidity_pct"] = frame["relative_humidity_pct"]
    for _, indices in frame.loc[frame["event_phase"].eq("heatwave")].groupby("local_date").groups.items():
        idx = list(indices)
        target = frame.loc[idx[0]]
        if target[["cf_tmin_c", "cf_tmean_c", "cf_tmax_c", "cf_rh_mean_pct"]].isna().any():
            raise ValueError(f"counterfactual target missing for {target['month_day']}")
        frame.loc[idx, "counterfactual_temperature_c"] = _monotone_daily_temperature(
            frame.loc[idx, "T2M"], target["cf_tmin_c"], target["cf_tmean_c"], target["cf_tmax_c"]
        ).to_numpy()
        frame.loc[idx, "counterfactual_relative_humidity_pct"] = _bounded_humidity_shift(
            frame.loc[idx, "RH2M"], target["cf_rh_mean_pct"]
        ).to_numpy()
    return frame


def _e8_idf(base_idf: str) -> str:
    lines = []
    for line in base_idf.splitlines():
        if line.strip().startswith("Output:Variable,"):
            continue
        lines.append(line)
    requests = [
        "Output:Variable, *, Zone Mean Air Temperature, Timestep;",
        "Output:Variable, *, Zone Air Relative Humidity, Timestep;",
        "Output:Variable, *, Zone People Occupant Count, Timestep;",
        "Output:Variable, *, Zone Ideal Loads Supply Air Total Cooling Energy, Timestep;",
        "Output:Variable, *, Zone Ideal Loads Supply Air Sensible Cooling Energy, Timestep;",
        "Output:Variable, *, Zone Ideal Loads Supply Air Total Cooling Rate, Timestep;",
        "Output:Variable, *, Zone Cooling Setpoint Not Met Time, Timestep;",
        "Output:Variable, *, Lights Electricity Energy, Timestep;",
        "Output:Variable, *, Lights Electricity Rate, Timestep;",
        "Output:Variable, *, Electric Equipment Electricity Energy, Timestep;",
        "Output:Variable, *, Electric Equipment Electricity Rate, Timestep;",
        "Output:Variable, *, Electric Equipment Convective Heating Energy, Timestep;",
        "Output:Meter, Electricity:Facility, Timestep;",
        "Output:VariableDictionary, IDF;",
        "Output:EnergyManagementSystem, Verbose, Verbose, ErrorsOnly;",
        "Output:SQLite, SimpleAndTabular;",
    ]
    return "\n".join(lines + requests + [""])


def _event_only_c70_idf(
    idf: str,
    conditioned_zones: set[str],
    zone_areas: dict[str, float],
    capacity_w_m2: float,
) -> str:
    """Switch between rated and 70% IdealLoads systems without resetting state."""
    result = idf
    for zone in sorted(conditioned_zones):
        rated_capacity = capacity_w_m2 * float(zone_areas[zone])
        c70_capacity = 0.70 * rated_capacity
        ideal_pattern = re.compile(
            rf"ZoneHVAC:IdealLoadsAirSystem,\s*{re.escape(zone)}IdealLoads,[^;]+;"
        )
        equipment_pattern = re.compile(
            rf"ZoneHVAC:EquipmentList,\s*{re.escape(zone)}Equipment,[^;]+;"
        )
        connection_pattern = re.compile(
            rf"ZoneHVAC:EquipmentConnections,\s*{re.escape(zone)},[^;]+;"
        )
        ideal_replacement = "\n".join(
            [
                f"ZoneHVAC:IdealLoadsAirSystem, {zone}RatedIdealLoads, {zone}_rated_capacity_ac, {zone}RatedSupplyNode, , , 50, 13, 0.0156, 0.0077, NoLimit, , , LimitCapacity, , {rated_capacity:.3f}, , {zone}_rated_capacity_ac, Humidistat, , None;",
                f"ZoneHVAC:IdealLoadsAirSystem, {zone}C70IdealLoads, {zone}_c70_capacity_ac, {zone}C70SupplyNode, , , 50, 13, 0.0156, 0.0077, NoLimit, , , LimitCapacity, , {c70_capacity:.3f}, , {zone}_c70_capacity_ac, Humidistat, , None;",
            ]
        )
        equipment_replacement = (
            f"ZoneHVAC:EquipmentList, {zone}Equipment, SequentialLoad, "
            f"ZoneHVAC:IdealLoadsAirSystem, {zone}RatedIdealLoads, 1, 1, , , "
            f"ZoneHVAC:IdealLoadsAirSystem, {zone}C70IdealLoads, 2, 2, , ;"
        )
        connection_replacement = "\n".join(
            [
                f"NodeList, {zone}SupplyNodes, {zone}RatedSupplyNode, {zone}C70SupplyNode;",
                f"ZoneHVAC:EquipmentConnections, {zone}, {zone}Equipment, {zone}SupplyNodes, , {zone}AirNode, {zone}ReturnNode;",
            ]
        )
        result, ideal_count = ideal_pattern.subn(ideal_replacement, result, count=1)
        result, equipment_count = equipment_pattern.subn(equipment_replacement, result, count=1)
        result, connection_count = connection_pattern.subn(connection_replacement, result, count=1)
        if (ideal_count, equipment_count, connection_count) != (1, 1, 1):
            raise ValueError(
                f"failed to transform C70 HVAC objects for {zone}: "
                f"{ideal_count}/{equipment_count}/{connection_count}"
            )
    return result


def _c70_total_capacity_ems(idf: str, conditioned_zones: set[str], zone_areas: dict[str, float],
                            capacity_w_m2: float) -> str:
    """Apply the reviewed v1.3 in-solver correction only to proven overrun zones."""
    objects, programs = [], []
    for zone in sorted(conditioned_zones):
        token = re.sub(r"[^A-Za-z0-9_]", "", zone)
        if not token: raise ValueError(f"C70 EMS cannot form a safe identifier for zone {zone!r}")
        capacity = .70 * capacity_w_m2 * float(zone_areas[zone])
        total_sensor = f"{token}C70TotalCoolingRate"; mass_sensor = f"{token}C70SupplyMassFlow"
        actuator = f"{token}C70MassFlowOverride"; program = f"{token}C70TotalCapacityController"
        programs.append(program)
        objects.extend([
            f"EnergyManagementSystem:Sensor, {total_sensor}, {zone}C70IdealLoads, Zone Ideal Loads Supply Air Total Cooling Rate;",
            f"EnergyManagementSystem:Sensor, {mass_sensor}, {zone}C70IdealLoads, Zone Ideal Loads Supply Air Mass Flow Rate;",
            f"EnergyManagementSystem:Actuator, {actuator}, {zone}C70IdealLoads, Ideal Loads Air System, Air Mass Flow Rate;",
            (f"EnergyManagementSystem:Program, {program}, IF {total_sensor} > {capacity:.6f}, "
             f"SET {actuator} = {mass_sensor} * {capacity:.6f} / {total_sensor}, ELSE, SET {actuator} = Null, ENDIF;"),
        ])
    if not programs: raise ValueError("C70 EMS requires at least one proven overrun zone")
    objects.append("EnergyManagementSystem:ProgramCallingManager, C70TotalCapacityManager, "
                   "InsideHVACSystemIterationLoop, " + ", ".join(programs) + ";")
    return idf.rstrip() + "\n" + "\n".join(objects) + "\n"


def _file_hashes(paths: dict[str, Path]) -> dict[str, str]:
    return {name: sha256_file(path) for name, path in paths.items()}


def _compile_case(
    project_root: Path,
    work_root: Path,
    production,
    design_row: pd.Series,
    event_id: str,
    scenario_id: str,
    prehistory_days: int,
    role: str,
    weather_override: pd.DataFrame | None = None,
    compile_output_root: Path | None = None,
    prototype_overrides: dict[str, float] | None = None,
    manifest_extras: dict[str, object] | None = None,
) -> dict[str, object]:
    climate_case, supply_stage, limited = SCENARIO_MAP[scenario_id]
    if weather_override is None:
        weather = load_event_weather(project_root, work_root, event_id, prehistory_days)
        weather = add_integration_counterfactual(project_root, weather)
    else:
        weather = weather_override.copy()
    if climate_case == "counterfactual":
        if "counterfactual_temperature_c" not in weather:
            raise ValueError("counterfactual weather is not available for this case")
        weather["temperature_c"] = weather["counterfactual_temperature_c"]
        weather["relative_humidity_pct"] = weather["counterfactual_relative_humidity_pct"]
    prototype_id = str(design_row["prototype_id"])
    prototype = production.prototype_parameter_table().set_index("prototype_id").loc[
        prototype_id
    ].copy()
    prototype["prototype_id"] = prototype_id
    for field, value in (prototype_overrides or {}).items():
        if field not in prototype.index:
            raise KeyError(f"Unknown prototype override: {field}")
        prototype[field] = value
    seed = int(production.RANDOM_SEED + sum(map(ord, f"{event_id}{design_row['base_case_id']}")))
    schedules, outage_start, outage_end, capacity_fraction = production.behaviour_schedules(
        weather,
        prototype,
        str(design_row["behaviour"]),
        int(design_row["start_offset_h"]),
        float(design_row["ac_use_probability"]),
        supply_stage,
        seed,
    )
    if scenario_id == "HW_C70":
        event_mask = weather["event_phase"].eq("heatwave").to_numpy(dtype=bool)
        for zone in production.zone_layout(prototype):
            name = str(zone["name"])
            if not bool(zone["conditioned"]):
                continue
            base = schedules[f"{name}_ac"].to_numpy(dtype=float)
            schedules[f"{name}_rated_capacity_ac"] = base * (~event_mask)
            schedules[f"{name}_c70_capacity_ac"] = base * event_mask
    annual = production.annualize_schedules(schedules, weather)
    idf_capacity_fraction = 1.0 if scenario_id == "HW_C70" else capacity_fraction
    idf, zone_areas, conditioned = production.build_idf(
        prototype,
        int(design_row["orientation_deg"]),
        str(design_row["floor_position"]),
        float(design_row["setpoint_c"]),
        annual,
        free_running=False,
        capacity_fraction=idf_capacity_fraction,
        cooling_capacity_limited=limited,
    )
    if scenario_id == "HW_C70":
        idf = _event_only_c70_idf(
            idf,
            conditioned,
            zone_areas,
            float(prototype["cooling_capacity_w_m2"]),
        )
        ems_zones = set((manifest_extras or {}).get("c70_total_capacity_ems_zones", []))
        if ems_zones:
            if not ems_zones <= conditioned:
                raise ValueError(f"C70 EMS zones are not conditioned zones: {sorted(ems_zones-conditioned)}")
            idf = _c70_total_capacity_ems(idf, ems_zones, zone_areas,
                                          float(prototype["cooling_capacity_w_m2"]))
    idf = _e8_idf(production.add_run_period(idf, weather))
    compile_key = stable_sha256(
        {
            "event_id": event_id,
            "base_case_id": design_row["base_case_id"],
            "scenario_id": scenario_id,
            "prehistory_days": prehistory_days,
            "weather_version": WEATHER_PROCESSING_VERSION,
            "score_version": SCORE_DEFINITION_VERSION,
            "prototype_overrides": prototype_overrides or {},
            "manifest_extras": manifest_extras or {},
        }
    )[:20]
    compile_dir = (compile_output_root or work_root / "02_runs" / "g3_compiled") / f"g3_{compile_key}"
    compile_dir.mkdir(parents=True, exist_ok=True)
    idf_path = compile_dir / "model.idf"
    epw_path = compile_dir / "weather.epw"
    schedule_path = compile_dir / "schedules.csv"
    event_schedule_path = compile_dir / "event_schedules.parquet"
    weather_path = compile_dir / "weather.parquet"
    idf_path.write_text(idf, encoding="ascii")
    production.write_epw(weather, epw_path)
    annual.to_csv(schedule_path, index=False, header=False)
    schedules.to_parquet(event_schedule_path, index=False)
    weather.to_parquet(weather_path, index=False)
    engine = production.ENGINE.resolve()
    byte_hashes = _file_hashes(
        {
            "idf": idf_path,
            "epw": epw_path,
            "schedule": schedule_path,
            "event_schedule": event_schedule_path,
            "engine": engine,
        }
    )
    normalized = {
        "protocol": E8_PROTOCOL_VERSION,
        "role": role,
        "event_id": event_id,
        "base_case_id": design_row["base_case_id"],
        "prototype_id": design_row["prototype_id"],
        "floor_position": design_row["floor_position"],
        "orientation_deg": int(design_row["orientation_deg"]),
        "behaviour": design_row["behaviour"],
        "setpoint_c": float(design_row["setpoint_c"]),
        "ac_use_probability": float(design_row["ac_use_probability"]),
        "start_offset_h": int(design_row["start_offset_h"]),
        "scenario_id": scenario_id,
        "climate_case": climate_case,
        "supply_stage": supply_stage,
        "prehistory_days": prehistory_days,
        "behavior_seed": seed,
        "capacity_fraction": float(capacity_fraction),
        "cooling_capacity_limited": bool(limited),
        "outage_start_hkt": outage_start,
        "outage_end_hkt": outage_end,
        "counterfactual_status": "integration_only_2016_2025" if scenario_id == "CF_UNL" else "not_applicable",
        "capacity_control_implementation": (
            ("native_Maximum_Total_Cooling_Capacity_plus_targeted_EMS_supply_mass_flow_feedback_for_observed_overrun_zones"
             if (manifest_extras or {}).get("c70_total_capacity_ems_zones") else
             "dual_ideal_loads_schedule_switch_rated_prehistory_70pct_event_rated_recovery")
            if scenario_id == "HW_C70" else "matches_declared_interval"
        ),
        "prototype_overrides": prototype_overrides or {},
        "effective_cooling_capacity_w_m2": float(prototype["cooling_capacity_w_m2"]),
        "effective_thermal_capacitance_kj_m2k": float(prototype["thermal_capacitance_kj_m2k"]),
        "file_byte_sha256": byte_hashes,
    }
    normalized.update(manifest_extras or {})
    normalized_hash = stable_sha256(normalized)
    physics_hash = stable_sha256({"normalized_manifest_hash": normalized_hash, **byte_hashes})
    manifest = {
        **normalized,
        "run_id": f"e8g3_{physics_hash[:20]}",
        "normalized_manifest_hash": normalized_hash,
        "physics_hash": physics_hash,
        "compiled_dir": str(compile_dir),
        "idf_path": str(idf_path),
        "epw_path": str(epw_path),
        "schedule_path": str(schedule_path),
        "event_schedule_path": str(event_schedule_path),
        "weather_path": str(weather_path),
        "zone_areas_json": json.dumps(zone_areas, sort_keys=True),
        "conditioned_zones_json": json.dumps(sorted(conditioned)),
        "status": "compiled_not_run",
    }
    _write_json(manifest, compile_dir / "input_manifest.json")
    return manifest


def load_hkust_validation_weather(project_root: Path) -> pd.DataFrame:
    """Load one complete local-HKO weather window near HKUST (station 6136)."""
    root = project_root / "data" / "weather" / "cowin_hourly" / "2023"
    sources = {
        "temp.csv": ("temperature_c", 1.0),
        "rh.csv": ("relative_humidity_pct", 1.0),
        "slp.csv": ("atmospheric_pressure_pa", 100.0),
        "ws.csv": ("WS10M", 1.0 / 3.6),
        "wd.csv": ("WD10M", 1.0),
        "60rf.csv": ("PRECTOTCORR", 1.0),
    }
    parts = []
    for filename, (field, factor) in sources.items():
        path = root / filename
        frame = pd.read_csv(path)
        value_column = next(
            column for column in frame.columns
            if column not in {"obstime", "station_id", "qcscore"}
        )
        quality = frame["qcscore"].astype(str).isin(["G", "M"]) if filename == "wd.csv" else frame["qcscore"].astype(str).eq("G")
        frame = frame.loc[frame["station_id"].astype(str).eq("6136") & quality].copy()
        frame["timestamp_hkt"] = pd.to_datetime(frame["obstime"], format="mixed").dt.tz_localize(
            "Asia/Hong_Kong"
        )
        frame[field] = pd.to_numeric(frame[value_column], errors="coerce") * factor
        parts.append(frame[["timestamp_hkt", field]])
    nasa_path = (
        project_root / "data" / "external_validation_raw" / "hkust"
        / "nasa_power_sai_kung_20220101_20240527.csv"
    )
    header = next(
        index for index, line in enumerate(nasa_path.read_text(encoding="utf-8").splitlines())
        if line.startswith("YEAR,")
    )
    nasa = pd.read_csv(nasa_path, skiprows=header).replace(-999.0, np.nan)
    nasa["timestamp_hkt"] = pd.to_datetime(
        dict(year=nasa["YEAR"], month=nasa["MO"], day=nasa["DY"], hour=nasa["HR"])
    ).dt.tz_localize("Asia/Hong_Kong")
    nasa["ALLSKY_SFC_SW_DWN"] = pd.to_numeric(
        nasa["ALLSKY_SFC_SW_DWN"], errors="coerce"
    )
    parts.append(nasa[["timestamp_hkt", "ALLSKY_SFC_SW_DWN"]])
    weather = parts[0]
    for part in parts[1:]:
        weather = weather.merge(part, on="timestamp_hkt", how="inner", validate="one_to_one")
    start = HKUST_CALIBRATION[0].tz_localize("Asia/Hong_Kong")
    end = HKUST_TEST[1].tz_localize("Asia/Hong_Kong")
    weather = weather.loc[weather["timestamp_hkt"].between(start, end)].sort_values(
        "timestamp_hkt"
    ).reset_index(drop=True)
    expected = pd.date_range(start, end, freq="h")
    if len(weather) != len(expected) or not weather["timestamp_hkt"].reset_index(drop=True).equals(
        pd.Series(expected, name="timestamp_hkt")
    ):
        raise ValueError("HKUST validation weather is not a complete 672-hour window")
    if weather.drop(columns="timestamp_hkt").isna().any().any():
        raise ValueError("HKUST validation weather contains missing values")
    weather["event_id"] = "HKUST_G3_2023"
    weather["event_phase"] = "heatwave"
    weather["weather_source"] = "HKO Co-WIN station 6136 (Tseung Kwan O Government Secondary School)"
    return weather


def _compile_validation_cases(
    project_root: Path, work_root: Path, production
) -> list[dict[str, object]]:
    weather = load_hkust_validation_weather(project_root)
    designs = (
        {
            "base_case_id": "HKUST_COM_EDU_G3",
            "prototype_id": "com_education",
            "floor_position": "middle",
            "orientation_deg": 0,
            "behaviour": "standard",
            "setpoint_c": 25.0,
            "ac_use_probability": 1.0,
            "start_offset_h": 0,
        },
        {
            "base_case_id": "HKUST_RES_PROXY_G3",
            "prototype_id": "res_private_highrise",
            "floor_position": "middle",
            "orientation_deg": 0,
            "behaviour": "working",
            "setpoint_c": 25.0,
            "ac_use_probability": 1.0,
            "start_offset_h": 0,
        },
    )
    return [
        _compile_case(
            project_root,
            work_root,
            production,
            pd.Series(design),
            "HKUST_G3_2023",
            "VALIDATION_NORMAL",
            0,
            "hkust_same_chain_validation",
            weather_override=weather,
        )
        for design in designs
    ]


def compile_g3(project_root: Path, work_root: Path) -> pd.DataFrame:
    production = load_production(project_root)
    design = pd.read_csv(
        project_root / "intermediate" / "foundation_pipeline" / "02_building_response"
        / "thermal_experiment_design.csv"
    ).set_index("base_case_id")
    rows = []
    for group_id, event_id, base_case_id, prototype_id, floor in G3_GROUPS:
        if base_case_id not in design.index:
            raise KeyError(base_case_id)
        row = design.loc[base_case_id].copy()
        row["base_case_id"] = base_case_id
        if row["prototype_id"] != prototype_id or row["floor_position"] != floor:
            raise ValueError(f"G3 design mismatch for {base_case_id}: {row.to_dict()}")
        for scenario_id in CORE_SCENARIO_IDS:
            rows.append(
                _compile_case(
                    project_root, work_root, production, row, event_id,
                    scenario_id, 14, f"core_group_{group_id}",
                )
            )
        if group_id == "D":
            for days in (2, 7):
                for scenario_id in ("HW_RATED", "HW_O24_SYNC"):
                    rows.append(
                        _compile_case(
                            project_root, work_root, production, row, event_id,
                            scenario_id, days, f"prehistory_{days}d_group_D",
                        )
                    )
    rows.extend(_compile_validation_cases(project_root, work_root, production))
    manifest = pd.DataFrame(rows)
    schema_path = work_root / "00_protocol" / "output_schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(schema_path)
    manifest["output_schema_hash"] = json.loads(
        schema_path.read_text(encoding="utf-8")
    )["output_schema_hash"]
    first = (
        manifest["role"].eq("core_group_A")
        & manifest["scenario_id"].eq("HW_RATED")
    )
    manifest["call_order"] = np.arange(2, len(manifest) + 2)
    manifest.loc[first, "call_order"] = 1
    manifest = manifest.sort_values("call_order").reset_index(drop=True)
    manifest["call_order"] = np.arange(1, len(manifest) + 1)
    manifest_hash = stable_sha256(
        manifest[["run_id", "physics_hash", "role", "call_order"]].to_dict(orient="records")
    )
    manifest["g3_manifest_hash"] = manifest_hash
    _write_csv(manifest, work_root / "02_runs" / "g3_manifest_revised.csv")
    coverage = (
        manifest.groupby(["role", "parent_event_id" if "parent_event_id" in manifest else "event_id", "prototype_id", "floor_position"], as_index=False)
        .agg(runs=("run_id", "size"), scenarios=("scenario_id", lambda x: "|".join(sorted(set(x)))))
    )
    _write_csv(coverage, work_root / "02_runs" / "g3_coverage_report.csv")
    _write_json(
        {"runs": len(manifest), "manifest_hash": manifest_hash, "lockbox_runs": 0},
        work_root / "02_runs" / "g3_manifest_revised.json",
    )
    return manifest


def _output_columns(raw: pd.DataFrame, variable: str, unit: str) -> list[str]:
    pattern = re.compile(
        rf"^[^:]+:{re.escape(variable)}\s*\[{re.escape(unit)}\]\(TimeStep\)$",
        re.IGNORECASE,
    )
    return [str(column) for column in raw.columns if pattern.match(str(column).strip())]


def _require_columns(raw: pd.DataFrame, variable: str, unit: str) -> list[str]:
    columns = _output_columns(raw, variable, unit)
    if not columns:
        raise KeyError(f"required EnergyPlus output missing: {variable} [{unit}](TimeStep)")
    return columns


def parse_energyplus_datetime_labels(
    labels: pd.Series, first_timestamp: pd.Timestamp
) -> pd.DatetimeIndex:
    """Parse strict EnergyPlus interval-end labels, including ``24:00``."""
    first = pd.Timestamp(first_timestamp)
    timezone = first.tz
    year = first.year
    previous_month_day: tuple[int, int] | None = None
    parsed: list[pd.Timestamp] = []
    pattern = re.compile(
        r"^\s*(\d{1,2})/(\d{1,2})\s+(\d{1,2}):(\d{2}):(\d{2})\s*$"
    )
    for raw_label in labels.astype(str):
        match = pattern.match(raw_label)
        if match is None:
            raise ValueError(f"unrecognised EnergyPlus Date/Time label: {raw_label!r}")
        month, day, hour, minute, second = map(int, match.groups())
        month_day = (month, day)
        if previous_month_day is not None and month_day < previous_month_day:
            year += 1
        if hour == 24:
            if minute != 0 or second != 0:
                raise ValueError(f"invalid EnergyPlus 24-hour label: {raw_label!r}")
            value = pd.Timestamp(year=year, month=month, day=day) + pd.Timedelta(days=1)
        elif 0 <= hour <= 23:
            value = pd.Timestamp(
                year=year, month=month, day=day, hour=hour,
                minute=minute, second=second,
            )
        else:
            raise ValueError(f"invalid EnergyPlus hour in label: {raw_label!r}")
        if timezone is not None:
            value = value.tz_localize(timezone)
        parsed.append(value)
        previous_month_day = month_day
    result = pd.DatetimeIndex(parsed)
    if result.has_duplicates:
        raise ValueError("EnergyPlus Date/Time labels contain duplicates")
    if not result.is_monotonic_increasing:
        raise ValueError("EnergyPlus Date/Time labels are not monotonic")
    return result


def parse_native_output(
    raw_path: Path,
    weather: pd.DataFrame,
    event_schedule_path: Path,
    zone_areas: dict[str, float],
    conditioned_zones: set[str],
    cop: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    raw = pd.read_csv(raw_path)
    expected_rows = len(weather) * 4
    if len(raw) != expected_rows:
        raise ValueError(f"native rows {len(raw)} != expected {expected_rows}")
    interval_h = 0.25
    timestamps = pd.date_range(
        weather["timestamp_hkt"].iloc[0] + pd.Timedelta(minutes=15),
        periods=expected_rows,
        freq="15min",
    )
    if "Date/Time" not in raw.columns:
        raise KeyError("required EnergyPlus Date/Time column missing")
    labelled_timestamps = parse_energyplus_datetime_labels(
        raw["Date/Time"], timestamps[0]
    )
    if not labelled_timestamps.equals(timestamps):
        mismatch = np.flatnonzero(labelled_timestamps != timestamps)
        first_bad = int(mismatch[0]) if len(mismatch) else -1
        raise ValueError(
            "EnergyPlus Date/Time labels do not match the weather-derived "
            f"15-minute interval-end axis; first mismatch row={first_bad}"
        )
    temp_cols = _require_columns(raw, "Zone Mean Air Temperature", "C")
    rh_cols = _require_columns(raw, "Zone Air Relative Humidity", "%")
    occ_cols = _require_columns(raw, "Zone People Occupant Count", "")
    total_cols = _require_columns(raw, "Zone Ideal Loads Supply Air Total Cooling Energy", "J")
    sensible_cols = _require_columns(raw, "Zone Ideal Loads Supply Air Sensible Cooling Energy", "J")
    rate_cols = _require_columns(raw, "Zone Ideal Loads Supply Air Total Cooling Rate", "W")
    unmet_cols = _require_columns(raw, "Zone Cooling Setpoint Not Met Time", "hr")
    lights_cols = _require_columns(raw, "Lights Electricity Energy", "J")
    equipment_cols = _require_columns(raw, "Electric Equipment Electricity Energy", "J")
    gain_cols = _require_columns(raw, "Electric Equipment Convective Heating Energy", "J")
    meter_matches = [
        str(column) for column in raw.columns
        if re.match(r"^Electricity:Facility\s*\[J\]\(TimeStep\)$", str(column).strip(), re.I)
    ]
    if len(meter_matches) != 1:
        raise KeyError(f"required facility meter missing or duplicated: {meter_matches}")
    identities = {
        "temperature": temp_cols,
        "rh": rh_cols,
        "occupants": occ_cols,
        "total_cooling": total_cols,
        "sensible_cooling": sensible_cols,
        "cooling_rate": rate_cols,
        "unmet": unmet_cols,
        "lights": lights_cols,
        "equipment": equipment_cols,
        "equipment_convective_gain": gain_cols,
        "facility_meter": meter_matches,
    }
    def zone_key(column: str, ideal_loads: bool = False) -> str:
        key = column.split(":", 1)[0].strip()
        if ideal_loads:
            key = re.sub(r"(?:Rated|C70)?IdealLoads$", "", key, flags=re.IGNORECASE)
        return key.title().replace(" ", "")
    zone_sets = {
        name: {
            zone_key(
                column,
                ideal_loads=name in {
                    "total_cooling", "sensible_cooling", "cooling_rate"
                },
            )
            for column in columns
        }
        for name, columns in identities.items()
        if name in {"temperature", "rh", "occupants", "total_cooling", "sensible_cooling", "cooling_rate", "unmet"}
    }
    expected_zones = {str(zone).title().replace(" ", "") for zone in zone_areas}
    expected_conditioned = {
        str(zone).title().replace(" ", "") for zone in conditioned_zones
    }
    for name, zones in zone_sets.items():
        expected = (
            expected_conditioned
            if name in {"total_cooling", "sensible_cooling", "cooling_rate"}
            else expected_zones
        )
        if zones != expected:
            raise ValueError(f"{name} zone keys {sorted(zones)} != {sorted(expected)}")
    native = pd.DataFrame({"timestamp_hkt": timestamps})
    total_area = float(sum(zone_areas.values()))
    for zone in zone_areas:
        canonical = str(zone).title().replace(" ", "")
        for prefix, columns in (
            ("temperature_c", temp_cols), ("rh_pct", rh_cols),
            ("occupants", occ_cols), ("unmet_hours", unmet_cols),
        ):
            column = next(item for item in columns if zone_key(item) == canonical)
            native[f"{zone}__{prefix}"] = pd.to_numeric(raw[column], errors="raise")
        for prefix, columns in (
            ("cooling_total_j", total_cols),
            ("cooling_sensible_j", sensible_cols),
            ("cooling_rate_w", rate_cols),
        ):
            selected = [
                item for item in columns if zone_key(item, ideal_loads=True) == canonical
            ]
            if zone in conditioned_zones:
                if not selected:
                    raise KeyError(f"conditioned zone {zone} lacks {prefix}")
                native[f"{zone}__{prefix}"] = raw[selected].apply(
                    pd.to_numeric, errors="raise"
                ).sum(axis=1)
            else:
                native[f"{zone}__{prefix}"] = 0.0
    native["lights_j"] = raw[lights_cols].apply(pd.to_numeric, errors="raise").sum(axis=1)
    native["equipment_j"] = raw[equipment_cols].apply(pd.to_numeric, errors="raise").sum(axis=1)
    native["equipment_convective_gain_j"] = raw[gain_cols].apply(pd.to_numeric, errors="raise").sum(axis=1)
    native["facility_electricity_j"] = pd.to_numeric(raw[meter_matches[0]], errors="raise")
    native["cooling_total_j"] = raw[total_cols].apply(pd.to_numeric, errors="raise").sum(axis=1)
    native["cooling_sensible_j"] = raw[sensible_cols].apply(pd.to_numeric, errors="raise").sum(axis=1)
    native["cooling_rate_w"] = raw[rate_cols].apply(pd.to_numeric, errors="raise").sum(axis=1)
    native["cooling_electricity_j"] = native["cooling_total_j"] / float(cop)
    native["cooling_latent_j"] = (native["cooling_total_j"] - native["cooling_sensible_j"]).clip(lower=0)
    native["noncooling_modeled_end_use_j"] = native["lights_j"] + native["equipment_j"]
    native["noncooling_electric_balance_residual_j"] = (
        native["facility_electricity_j"] - native["noncooling_modeled_end_use_j"]
    )
    native["total_modeled_electricity_j"] = (
        native["facility_electricity_j"] + native["cooling_electricity_j"]
    )
    event_schedule = pd.read_parquet(event_schedule_path)
    if len(event_schedule) != len(weather):
        raise ValueError("event schedule and weather have different row counts")
    repeated = event_schedule.loc[event_schedule.index.repeat(4)].reset_index(drop=True)
    for suffix in ("_occupancy", "_ac", "_vent", "_mechvent", "_lights", "_plugs"):
        columns = [column for column in repeated if column.endswith(suffix)]
        if not columns:
            raise KeyError(f"required schedule family missing: {suffix}")
        native[f"schedule{suffix}_sum"] = repeated[columns].sum(axis=1)
    monitored_zones = list(zone_areas)
    long_rows = []
    for zone in monitored_zones:
        long_rows.append(
            pd.DataFrame(
                {
                    "timestamp_hkt": timestamps,
                    "zone_id": zone,
                    "zone_area_m2": float(zone_areas[zone]),
                    "temperature_c": native[f"{zone}__temperature_c"],
                    "relative_humidity_pct": native[f"{zone}__rh_pct"],
                    "occupants": native[f"{zone}__occupants"],
                    "cooling_total_j": native[f"{zone}__cooling_total_j"],
                    "cooling_sensible_j": native[f"{zone}__cooling_sensible_j"],
                    "cooling_rate_w": native[f"{zone}__cooling_rate_w"],
                    "unmet_hours": native[f"{zone}__unmet_hours"],
                    "structural_conditioning_present": zone in conditioned_zones,
                }
            )
        )
    zone_native = pd.concat(long_rows, ignore_index=True)
    audit = {
        "native_rows": len(native),
        "zone_native_rows": len(zone_native),
        "interval_hours": interval_h,
        "timestamp_source": "EnergyPlus Date/Time asserted equal to weather-derived interval-end axis",
        "timestamp_semantics": "TimeStep state variables are reported at interval end; energy variables accumulate over the ending interval",
        "exact_fields": identities,
        "total_cooling_boundary": "Zone Ideal Loads Supply Air; no zone-air-system cooling field added",
        "cop_conversion_count": 1,
        "facility_meter_boundary": (
            "Electricity:Facility excludes IdealLoads district cooling; modeled cooling electricity "
            "is added exactly once from supply-air total cooling divided by COP"
        ),
        "maximum_absolute_noncooling_electric_balance_residual_j": float(
            native["noncooling_electric_balance_residual_j"].abs().max()
        ),
        "sum_noncooling_electric_balance_residual_j": float(
            native["noncooling_electric_balance_residual_j"].sum()
        ),
    }
    return native, zone_native, audit


def _append_call_ledger(path: Path, row: dict[str, object]) -> None:
    frame = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        frame = pd.concat([old, frame], ignore_index=True)
    _write_csv(frame, path)


def _tree_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def _budget_state(work_root: Path) -> tuple[Path, dict[str, object]]:
    path = work_root / "02_runs" / "g3_budget.json"
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
    else:
        state = {
            "max_calls": MAX_G3_CALLS,
            "max_wall_seconds": MAX_G3_WALL_SECONDS,
            "max_bytes": MAX_G3_BYTES,
            "first_solver_started_utc": None,
        }
    return path, state


def _check_budget_before_call(work_root: Path, prior_calls: int) -> dict[str, object]:
    path, state = _budget_state(work_root)
    if prior_calls >= int(state["max_calls"]):
        raise RuntimeError(f"G3 EnergyPlus call ceiling reached: {prior_calls}/{state['max_calls']}")
    if state.get("first_solver_started_utc"):
        started = pd.Timestamp(state["first_solver_started_utc"])
        elapsed = (pd.Timestamp.now(tz="UTC") - started).total_seconds()
        if elapsed >= float(state["max_wall_seconds"]):
            raise RuntimeError(f"G3 wall-clock ceiling reached: {elapsed:.1f}s")
    used = _tree_size_bytes(work_root / "02_runs")
    if used >= int(state["max_bytes"]):
        raise RuntimeError(f"G3 storage ceiling reached: {used} bytes")
    if not state.get("first_solver_started_utc"):
        state["first_solver_started_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
        _write_json(state, path)
    return state


def _safe_remove_staging(path: Path, work_root: Path) -> None:
    resolved = path.resolve()
    allowed = (work_root / "02_runs" / "g3_staging").resolve()
    if allowed not in resolved.parents:
        raise ValueError(f"refuse to remove staging outside {allowed}: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _run_energyplus_with_peak_memory(command: list[str], cwd: Path) -> tuple[subprocess.CompletedProcess[str], int | None]:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    peak_rss_bytes: int | None = None
    monitored = psutil.Process(process.pid) if psutil is not None else None
    while process.poll() is None:
        if monitored is not None:
            try:
                rss = monitored.memory_info().rss
                rss += sum(child.memory_info().rss for child in monitored.children(recursive=True))
                peak_rss_bytes = max(peak_rss_bytes or 0, int(rss))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        time.sleep(0.01)
    stdout, stderr = process.communicate()
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr), peak_rss_bytes


def run_manifest_rows(
    project_root: Path,
    work_root: Path,
    only_run_ids: set[str] | None = None,
    max_calls: int = 30,
) -> pd.DataFrame:
    manifest = pd.read_csv(work_root / "02_runs" / "g3_manifest_revised.csv").sort_values("call_order")
    if only_run_ids is not None:
        manifest = manifest.loc[manifest["run_id"].isin(only_run_ids)].copy()
    ledger_path = work_root / "02_runs" / "g3_call_ledger.csv"
    prior_calls = len(pd.read_csv(ledger_path)) if ledger_path.exists() else 0
    production = load_production(project_root)
    results = []
    for row in manifest.to_dict(orient="records"):
        final_dir = work_root / "02_runs" / "g3_results" / row["run_id"]
        done = final_dir / "run_result.json"
        if done.exists():
            payload = json.loads(done.read_text(encoding="utf-8"))
            if (
                payload.get("physics_hash") == row["physics_hash"]
                and payload.get("status") == "passed"
                and payload.get("parser_version") == PARSER_VERSION
            ):
                results.append(payload)
                continue
            if (
                payload.get("physics_hash") == row["physics_hash"]
                and payload.get("status") in {"failed_parser", "passed"}
                and (final_dir / "raw" / "eplusout.csv").exists()
            ):
                input_dir = final_dir / "inputs"
                raw_dir = final_dir / "raw"
                parsed_dir = final_dir / "parsed"
                parsed_dir.mkdir(parents=True, exist_ok=True)
                weather = pd.read_parquet(input_dir / "weather.parquet")
                weather["timestamp_hkt"] = pd.to_datetime(
                    weather["timestamp_hkt"], utc=True
                ).dt.tz_convert("Asia/Hong_Kong")
                native, zone_native, parser_audit = parse_native_output(
                    raw_dir / "eplusout.csv",
                    weather,
                    input_dir / "event_schedules.parquet",
                    json.loads(row["zone_areas_json"]),
                    set(json.loads(row["conditioned_zones_json"])),
                    float(
                        production.prototype_parameter_table().set_index("prototype_id").loc[
                            row["prototype_id"], "baseline_ac_cop"
                        ]
                    ),
                )
                native.to_parquet(parsed_dir / "native_timestep_building.parquet", index=False)
                zone_native.to_parquet(parsed_dir / "native_timestep_zones.parquet", index=False)
                _write_json(parser_audit, parsed_dir / "parser_audit.json")
                payload.update(
                    {
                        "status": "passed",
                        "parser_version": PARSER_VERSION,
                        "parser_audit": parser_audit,
                        "reparsed_without_solver_call": True,
                        "parser_error_previous": payload.get("parser_error"),
                    }
                )
                payload.pop("parser_error", None)
                _write_json(payload, done)
                results.append(payload)
                continue
        if max_calls != MAX_G3_CALLS:
            raise ValueError(f"approved G3 call ceiling is fixed at {MAX_G3_CALLS}")
        _check_budget_before_call(work_root, prior_calls)
        staging = work_root / "02_runs" / "g3_staging" / row["run_id"]
        _safe_remove_staging(staging, work_root)
        input_dir = staging / "inputs"
        raw_dir = staging / "raw"
        parsed_dir = staging / "parsed"
        input_dir.mkdir(parents=True)
        raw_dir.mkdir(parents=True)
        parsed_dir.mkdir(parents=True)
        for name in (
            "model.idf", "weather.epw", "schedules.csv", "event_schedules.parquet",
            "weather.parquet", "input_manifest.json",
        ):
            shutil.copy2(Path(row["compiled_dir"]) / name, input_dir / name)
        started = time.perf_counter()
        process, peak_rss_bytes = _run_energyplus_with_peak_memory(
            [str(production.ENGINE), "-w", str(input_dir / "weather.epw"), "-d", str(raw_dir), str(input_dir / "model.idf")],
            cwd=staging,
        )
        elapsed = time.perf_counter() - started
        prior_calls += 1
        err_path = raw_dir / "eplusout.err"
        err_text = err_path.read_text(encoding="utf-8", errors="replace") if err_path.exists() else ""
        call_row = {
            "call_number": prior_calls,
            "run_id": row["run_id"],
            "physics_hash": row["physics_hash"],
            "started_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "elapsed_seconds": elapsed,
            "returncode": process.returncode,
            "severe_errors": err_text.count("** Severe  **"),
            "fatal_errors": len(re.findall(r"\*\*\s+Fatal\s+\*\*", err_text)),
            "peak_rss_bytes": peak_rss_bytes,
            "run_storage_bytes": _tree_size_bytes(staging),
        }
        _append_call_ledger(ledger_path, call_row)
        (raw_dir / "solver_stdout.txt").write_text(process.stdout, encoding="utf-8")
        (raw_dir / "solver_stderr.txt").write_text(process.stderr, encoding="utf-8")
        if process.returncode != 0 or call_row["fatal_errors"]:
            failure = {**row, **call_row, "status": "failed_solver"}
            _write_json(failure, staging / "run_result.json")
            if final_dir.exists():
                raise FileExistsError(final_dir)
            staging.replace(final_dir)
            results.append(failure)
            break
        weather = pd.read_parquet(input_dir / "weather.parquet")
        weather["timestamp_hkt"] = pd.to_datetime(weather["timestamp_hkt"], utc=True).dt.tz_convert("Asia/Hong_Kong")
        try:
            native, zone_native, parser_audit = parse_native_output(
                raw_dir / "eplusout.csv",
                weather,
                input_dir / "event_schedules.parquet",
                json.loads(row["zone_areas_json"]),
                set(json.loads(row["conditioned_zones_json"])),
                float(
                    production.prototype_parameter_table().set_index("prototype_id").loc[
                        row["prototype_id"], "baseline_ac_cop"
                    ]
                ),
            )
            native.to_parquet(parsed_dir / "native_timestep_building.parquet", index=False)
            zone_native.to_parquet(parsed_dir / "native_timestep_zones.parquet", index=False)
            _write_json(parser_audit, parsed_dir / "parser_audit.json")
            payload = {
                **row,
                **call_row,
                "status": "passed",
                "parser_version": PARSER_VERSION,
                "parser_audit": parser_audit,
                "raw_output_sha256": sha256_file(raw_dir / "eplusout.csv"),
            }
        except Exception as exc:
            payload = {
                **row,
                **call_row,
                "status": "failed_parser",
                "parser_error": f"{type(exc).__name__}: {exc}",
            }
        _write_json(payload, staging / "run_result.json")
        if final_dir.exists():
            raise FileExistsError(final_dir)
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(final_dir)
        results.append(payload)
        if payload["status"] != "passed":
            break
    result = pd.DataFrame(results)
    _write_csv(result, work_root / "02_runs" / "g3_latest_run_status.csv")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--run-first-gate", action="store_true")
    parser.add_argument("--run-core", action="store_true")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    work_root = (args.work_root or project_root / "intermediate/simulation_outputs/upstream").resolve()
    manifest = compile_g3(project_root, work_root) if args.compile else pd.read_csv(
        work_root / "02_runs" / "g3_manifest_revised.csv"
    )
    if args.run_id:
        if args.run_id not in set(manifest["run_id"].astype(str)):
            raise KeyError(f"run_id is not in approved G3 manifest: {args.run_id}")
        run_manifest_rows(project_root, work_root, {args.run_id}, max_calls=30)
    elif args.run_first_gate:
        first = manifest.sort_values("call_order").iloc[0]
        run_manifest_rows(project_root, work_root, {str(first["run_id"])}, max_calls=30)
    elif args.run_core:
        run_manifest_rows(project_root, work_root, max_calls=30)


if __name__ == "__main__":
    main()
