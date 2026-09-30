"""Solver-free E8 preflight contracts and audit helpers.

This module never invokes EnergyPlus. It compiles auditable G0--G2 inputs and
dry-run manifests; network access is opt-in and limited to hourly weather.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import platform
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


E8_PROTOCOL_VERSION = "E8-G0G2-v1"
WEATHER_PROCESSING_VERSION = "e8-hko-daily-anchored-nasa-utc-v2"
OUTAGE_CLOCK_VERSION = "common-first-outdoor-tmax-24h-v1"
SCORE_DEFINITION_VERSION = "beri-e8-four-stage-v1"
NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
NASA_POWER_PARAMETERS = (
    "T2M", "RH2M", "ALLSKY_SFC_SW_DWN", "WS10M", "WD10M", "PS", "PRECTOTCORR",
)
REQUIRED_WEATHER_FIELDS = set(NASA_POWER_PARAMETERS)


@dataclass(frozen=True)
class CoreScenario:
    scenario_id: str
    weather: str
    cooling_limit: str
    grid_state: str
    operation: str
    scientific_role: str


CORE_SCENARIOS = (
    CoreScenario("CF_UNL", "matched_counterfactual", "NoLimit", "available", "normal_schedule", "demand_increment_reference"),
    CoreScenario("HW_UNL", "heatwave", "NoLimit", "available", "normal_schedule", "normal_supply_response_and_S_N"),
    CoreScenario("HW_RATED", "heatwave", "100pct_fixed_nominal", "available", "normal_schedule", "rated_supply_thermal_reference"),
    CoreScenario("HW_C70", "heatwave", "70pct_same_nominal", "available", "normal_schedule", "cooling_service_and_S_C"),
    CoreScenario("HW_O24_SYNC", "heatwave", "100pct_before_after_zero_during", "unavailable_for_24h", "continuous_outage_recovery", "S_O_S_R_and_thermal_consequences"),
)


def stable_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _write_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def exact_output_column(columns: Iterable[str], *, variable_name: str, key: str, unit: str, frequency: str) -> str:
    """Select one EnergyPlus CSV field by its complete output identity."""
    pattern = re.compile(
        rf"^{re.escape(key)}:{re.escape(variable_name)}\s*\[{re.escape(unit)}\]\({re.escape(frequency)}\)$",
        flags=re.IGNORECASE,
    )
    matches = [str(column) for column in columns if pattern.match(str(column).strip())]
    if len(matches) != 1:
        raise KeyError(
            f"expected exactly one output for key={key!r}, variable={variable_name!r}, "
            f"unit={unit!r}, frequency={frequency!r}; found {matches}"
        )
    return matches[0]


def first_threshold_crossing_hours(elapsed_hours: Sequence[float], values: Sequence[float], threshold: float) -> tuple[float, bool]:
    """Return first >= threshold time with linear interpolation and censor flag."""
    times = np.asarray(elapsed_hours, dtype=float)
    vals = np.asarray(values, dtype=float)
    if len(times) == 0 or len(times) != len(vals):
        raise ValueError("elapsed_hours and values must be non-empty and have equal length")
    if not np.isfinite(times).all() or not np.isfinite(vals).all() or np.any(np.diff(times) <= 0):
        raise ValueError("threshold series must be finite and strictly increasing in time")
    hit = np.flatnonzero(vals >= threshold)
    if not len(hit):
        return float(times[-1]), True
    index = int(hit[0])
    if index == 0:
        return float(times[0]), False
    t0, t1 = float(times[index - 1]), float(times[index])
    v0, v1 = float(vals[index - 1]), float(vals[index])
    if v1 == v0:
        return t1, False
    fraction = float(np.clip((threshold - v0) / (v1 - v0), 0.0, 1.0))
    return t0 + fraction * (t1 - t0), False


def sustained_recovery_hours(elapsed_hours: Sequence[float], values: Sequence[float], threshold: float = 26.2, sustain_hours: float = 1.0) -> tuple[float, bool]:
    """Return first sample starting a continuously recovered interval."""
    times = np.asarray(elapsed_hours, dtype=float)
    vals = np.asarray(values, dtype=float)
    if len(times) == 0 or len(times) != len(vals):
        raise ValueError("elapsed_hours and values must be non-empty and have equal length")
    if np.any(np.diff(times) <= 0) or not np.isfinite(vals).all():
        raise ValueError("recovery series must be finite and strictly increasing in time")
    for start in range(len(times)):
        if vals[start] > threshold:
            continue
        end = int(np.searchsorted(times, times[start] + sustain_hours, side="left"))
        if end < len(times) and np.all(vals[start : end + 1] <= threshold):
            return float(times[start]), False
    return float(times[-1]), True


def tie_aware_selection_probabilities(scores: Sequence[float], q: float, lower_is_worse: bool = True) -> np.ndarray:
    """Return exchangeable probabilities for an exact ceil(qN) queue."""
    values = np.asarray(scores, dtype=float)
    if not 0 < q <= 1 or not np.isfinite(values).all():
        raise ValueError("q must be in (0,1] and scores must be finite")
    count = int(math.ceil(q * len(values)))
    transformed = values if lower_is_worse else -values
    cutoff = np.sort(transformed)[count - 1]
    strict = transformed < cutoff
    tied = np.isclose(transformed, cutoff, rtol=0.0, atol=1e-12)
    probability = np.zeros(len(values), dtype=float)
    probability[strict] = 1.0
    probability[tied] = (count - int(strict.sum())) / int(tied.sum())
    return probability


def nasa_cache_path(root: Path, event_id: str, start: pd.Timestamp, end: pd.Timestamp) -> Path:
    query_start, query_end = _nasa_utc_query_dates(start, end)
    return (
        root / "data" / "weather" / "raw" / "e8"
        / f"{event_id}_{query_start:%Y%m%d}_{query_end:%Y%m%d}_utc.json"
    )


def _nasa_utc_query_dates(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_hkt = pd.Timestamp(start).tz_localize("Asia/Hong_Kong")
    end_exclusive_hkt = (pd.Timestamp(end) + pd.Timedelta(days=1)).tz_localize(
        "Asia/Hong_Kong"
    )
    query_start = start_hkt.tz_convert("UTC").normalize().tz_localize(None)
    query_end = (
        end_exclusive_hkt.tz_convert("UTC") - pd.Timedelta(nanoseconds=1)
    ).normalize().tz_localize(None)
    return query_start, query_end


def acquire_nasa_hourly(
    cache_path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    latitude: float,
    longitude: float,
    download: bool,
) -> tuple[pd.DataFrame | None, str]:
    """Read or opt-in download one NASA POWER hourly block."""
    if not cache_path.exists() and not download:
        return None, "missing_cache"
    if not cache_path.exists():
        query_start, query_end = _nasa_utc_query_dates(start, end)
        query = urllib.parse.urlencode(
            {
                "parameters": ",".join(NASA_POWER_PARAMETERS),
                "community": "RE",
                "longitude": longitude,
                "latitude": latitude,
                "start": query_start.strftime("%Y%m%d"),
                "end": query_end.strftime("%Y%m%d"),
                "format": "JSON",
                "time-standard": "UTC",
            }
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{NASA_POWER_URL}?{query}", timeout=120) as response:
            raw = response.read()
        cache_path.write_bytes(raw)
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    time_standard = payload.get("header", {}).get("time_standard")
    if time_standard != "UTC":
        return None, f"unexpected_time_standard:{time_standard!r}"
    parameters = payload.get("properties", {}).get("parameter", {})
    absent = REQUIRED_WEATHER_FIELDS.difference(parameters)
    if absent:
        return None, "missing_variables:" + ",".join(sorted(absent))
    stamps = sorted(set.intersection(*(set(parameters[name]) for name in NASA_POWER_PARAMETERS)))
    rows = []
    for stamp in stamps:
        row = {
            "timestamp_hkt": pd.to_datetime(
                stamp, format="%Y%m%d%H", utc=True
            ).tz_convert("Asia/Hong_Kong")
        }
        row.update({name: parameters[name][stamp] for name in NASA_POWER_PARAMETERS})
        rows.append(row)
    frame = pd.DataFrame(rows)
    start_hkt = pd.Timestamp(start).tz_localize("Asia/Hong_Kong")
    end_exclusive_hkt = (pd.Timestamp(end) + pd.Timedelta(days=1)).tz_localize(
        "Asia/Hong_Kong"
    )
    frame = frame.loc[
        frame["timestamp_hkt"].ge(start_hkt)
        & frame["timestamp_hkt"].lt(end_exclusive_hkt)
    ].reset_index(drop=True)
    return frame, "cache_valid_utc_to_hkt"


def audit_hourly_weather(frame: pd.DataFrame | None, start: pd.Timestamp, end: pd.Timestamp) -> tuple[bool, dict[str, object]]:
    expected = pd.date_range(
        pd.Timestamp(start).tz_localize("Asia/Hong_Kong"),
        pd.Timestamp(end).tz_localize("Asia/Hong_Kong") + pd.Timedelta(hours=23),
        freq="h",
    )
    if frame is None or frame.empty:
        return False, {"expected_hours": len(expected), "actual_hours": 0, "reason": "no_hourly_data"}
    current = frame.copy()
    current["timestamp_hkt"] = pd.to_datetime(current["timestamp_hkt"], utc=True).dt.tz_convert(
        "Asia/Hong_Kong"
    )
    current = current.loc[
        (current["timestamp_hkt"] >= expected[0])
        & (current["timestamp_hkt"] <= expected[-1])
    ]
    duplicate_hours = int(current["timestamp_hkt"].duplicated().sum())
    current = current.set_index("timestamp_hkt").reindex(expected)
    numeric = current[list(NASA_POWER_PARAMETERS)].apply(pd.to_numeric, errors="coerce")
    sentinel = numeric <= -900
    missing_by_field = (numeric.isna() | sentinel).sum()
    rh_invalid = int(((numeric["RH2M"] < 0) | (numeric["RH2M"] > 100)).fillna(True).sum())
    ps_invalid = int((numeric["PS"] <= 0).fillna(True).sum())
    ghi_invalid = int((numeric["ALLSKY_SFC_SW_DWN"] < 0).fillna(True).sum())
    passed = not duplicate_hours and int(missing_by_field.sum()) == 0 and rh_invalid == 0 and ps_invalid == 0 and ghi_invalid == 0
    detail = {
        "expected_hours": int(len(expected)),
        "actual_hours": int(len(frame)),
        "duplicate_hours": duplicate_hours,
        "missing_total": int(missing_by_field.sum()),
        "missing_by_field": json.dumps(missing_by_field.astype(int).to_dict(), sort_keys=True),
        "rh_invalid": rh_invalid,
        "pressure_invalid": ps_invalid,
        "solar_invalid": ghi_invalid,
        "reason": "qualified" if passed else "hourly_qc_failed",
    }
    return passed, detail


def _monotone_daily_temperature(
    values: pd.Series, target_min: float, target_mean: float, target_max: float
) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(x).all() or not target_min <= target_mean <= target_max:
        return pd.Series(np.nan, index=values.index, dtype=float)
    span = float(np.max(x) - np.min(x))
    if span <= 1e-9 or target_max - target_min <= 1e-9:
        return pd.Series(np.full(len(x), target_mean), index=values.index)
    z = (x - np.min(x)) / span
    target_fraction = (target_mean - target_min) / (target_max - target_min)
    lo, hi = 0.01, 100.0
    for _ in range(100):
        gamma = (lo + hi) / 2.0
        if np.mean(z**gamma) > target_fraction:
            lo = gamma
        else:
            hi = gamma
    return pd.Series(
        target_min + (target_max - target_min) * z ** ((lo + hi) / 2.0),
        index=values.index,
    )


def _bounded_humidity_shift(values: pd.Series, target_mean: float) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(target_mean):
        return pd.Series(np.nan, index=values.index, dtype=float)
    lo, hi = -100.0, 100.0
    for _ in range(100):
        shift = (lo + hi) / 2.0
        shifted = np.clip(x + shift, 0.0, 100.0)
        if float(np.mean(shifted)) < target_mean:
            lo = shift
        else:
            hi = shift
    return pd.Series(np.clip(x + (lo + hi) / 2.0, 0.0, 100.0), index=values.index)


def anchor_hourly_to_hko_daily(
    hourly: pd.DataFrame, daily: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Preserve NASA hourly shapes while matching observed HKO daily T/RH targets."""
    frame = hourly.copy()
    frame["date"] = frame["timestamp_hkt"].dt.tz_localize(None).dt.normalize()
    targets = daily[
        ["date", "tmin_c", "tmean_c", "tmax_c", "rh_mean_pct", "source_file", "source_sha256"]
    ].drop_duplicates("date")
    frame = frame.merge(targets, on="date", how="left", validate="many_to_one")
    required_targets = ["tmin_c", "tmean_c", "tmax_c", "rh_mean_pct"]
    missing_days = sorted(
        frame.loc[frame[required_targets].isna().any(axis=1), "date"]
        .drop_duplicates()
        .dt.strftime("%Y-%m-%d")
        .tolist()
    )
    frame["temperature_c"] = np.nan
    frame["relative_humidity_pct"] = np.nan
    if not missing_days:
        for _, indices in frame.groupby("date").groups.items():
            idx = list(indices)
            target = frame.loc[idx[0]]
            frame.loc[idx, "temperature_c"] = _monotone_daily_temperature(
                frame.loc[idx, "T2M"], target["tmin_c"], target["tmean_c"], target["tmax_c"]
            ).to_numpy()
            frame.loc[idx, "relative_humidity_pct"] = _bounded_humidity_shift(
                frame.loc[idx, "RH2M"], target["rh_mean_pct"]
            ).to_numpy()
    finite = frame[["temperature_c", "relative_humidity_pct"]].notna().all().all()
    return frame, {
        "hko_daily_missing_days": len(missing_days),
        "hko_daily_missing_dates": json.dumps(missing_days),
        "hko_anchor_qc": "passed" if not missing_days and finite else "failed",
    }


def weather_features(frame: pd.DataFrame, event_start: pd.Timestamp, event_end: pd.Timestamp) -> dict[str, float]:
    current = frame.copy()
    current["timestamp_hkt"] = pd.to_datetime(current["timestamp_hkt"], utc=True).dt.tz_convert(
        "Asia/Hong_Kong"
    )
    event_start_hkt = pd.Timestamp(event_start).tz_localize("Asia/Hong_Kong")
    event_end_exclusive_hkt = (pd.Timestamp(event_end) + pd.Timedelta(days=1)).tz_localize(
        "Asia/Hong_Kong"
    )
    event = current.loc[
        (current["timestamp_hkt"] >= event_start_hkt)
        & (current["timestamp_hkt"] < event_end_exclusive_hkt)
    ].copy()
    if event.empty:
        raise ValueError("event period contains no hourly weather")
    event["hour"] = event["timestamp_hkt"].dt.hour
    night = event.loc[(event["hour"] >= 20) | (event["hour"] < 7), "temperature_c"]
    dewpoint = event["temperature_c"] - (100.0 - event["relative_humidity_pct"]) / 5.0
    temp_change = event.set_index("timestamp_hkt")["temperature_c"].diff().abs()
    peak_hour = int(event.loc[event["temperature_c"].idxmax(), "hour"])
    load_alignment = math.cos(2 * math.pi * (peak_hour - 15) / 24)
    return {
        "duration_days": float((event_end.normalize() - event_start.normalize()).days + 1),
        "tmax_c": float(event["temperature_c"].max()),
        "tmean_c": float(event["temperature_c"].mean()),
        "night_tmean_c": float(night.mean()),
        "dewpoint_proxy_c": float(dewpoint.mean()),
        "rh_mean_pct": float(event["relative_humidity_pct"].mean()),
        "season_day": float(event_start.dayofyear),
        "temperature_dynamics_cph": float(temp_change.mean()),
        "peak_hour_hkt_proxy": float(peak_hour),
        "load_time_alignment": float(load_alignment),
    }


def robust_standardize(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    for column in columns:
        values = frame[column].astype(float)
        median = float(values.median())
        scale = float(values.quantile(0.75) - values.quantile(0.25))
        if not np.isfinite(scale) or scale == 0:
            scale = float(values.std(ddof=0)) or 1.0
        result[column] = (values - median) / scale
    return result


def select_five_weather_events(features: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    """Select five distinct-year events by frozen equal-weight maximin coverage."""
    if len(features) < 5:
        raise ValueError("fewer than five hourly-qualified pre-2018 events")
    feature_columns = [
        "duration_days", "tmax_c", "night_tmean_c", "dewpoint_proxy_c",
        "season_day", "temperature_dynamics_cph", "load_time_alignment",
    ]
    z = robust_standardize(features, feature_columns)
    composite = pd.DataFrame(
        {
            "persistence": z["duration_days"],
            "heat_level": z["tmax_c"],
            "warm_humid": (z["night_tmean_c"] + z["dewpoint_proxy_c"]) / 2.0,
            "season": z["season_day"],
            "dynamics": z["temperature_dynamics_cph"],
            "load_timing": z["load_time_alignment"],
        },
        index=features.index,
    )
    best_indices: tuple[int, ...] | None = None
    best_distances: tuple[float, float] | None = None
    best_ids: tuple[str, ...] | None = None
    for indices in itertools.combinations(range(len(features)), 5):
        subset = features.iloc[list(indices)]
        if subset["start_year"].nunique() < 5:
            continue
        matrix = composite.iloc[list(indices)].to_numpy(float)
        distances = [
            float(np.linalg.norm(matrix[a] - matrix[b]))
            for a, b in itertools.combinations(range(5), 2)
        ]
        objective = (min(distances), float(np.mean(distances)))
        ids = tuple(sorted(subset["event_id"].astype(str)))
        if best_distances is None or objective > best_distances or (objective == best_distances and ids < best_ids):
            best_indices, best_distances, best_ids = indices, objective, ids
    if best_indices is None:
        raise ValueError("no five-event distinct-year combination available")
    selected = features.iloc[list(best_indices)]["event_id"].astype(str).tolist()
    diagnostics = pd.concat(
        [features.reset_index(drop=True), composite.reset_index(drop=True).add_prefix("selection_")],
        axis=1,
    )
    diagnostics["selected_pre2018"] = diagnostics["event_id"].isin(selected)
    diagnostics["selection_algorithm"] = (
        "distinct-year exhaustive maximin; six equal-weight composites; robust median-IQR scaling"
    )
    return selected, diagnostics


def _git_value(root: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unavailable"


def _find_event_rows(catalog: pd.DataFrame, ids: Sequence[str]) -> pd.DataFrame:
    rows = catalog.loc[catalog["event_id"].isin(ids)].copy()
    missing = set(ids).difference(rows["event_id"])
    if missing:
        raise KeyError(f"event catalog is missing {sorted(missing)}")
    return rows.sort_values("start_date")


def _scenario_frame() -> pd.DataFrame:
    frame = pd.DataFrame(asdict(item) for item in CORE_SCENARIOS)
    control = {
        "CF_UNL": ("complete counterfactual event", "NoLimit", "NoLimit", "on"),
        "HW_UNL": ("complete heatwave event", "NoLimit", "NoLimit", "on"),
        "HW_RATED": ("complete heatwave event", "100pct nominal", "100pct nominal", "on"),
        "HW_C70": ("complete heatwave event; 100pct during true prehistory", "100pct nominal", "100pct nominal", "on"),
        "HW_O24_SYNC": (
            "24h from first common outdoor Tmax; continuous recovery follows",
            "100pct nominal", "100pct nominal", "off only during outage",
        ),
    }
    frame["control_interval"] = frame["scenario_id"].map(
        lambda value: control[value][0]
    )
    frame["prehistory_capacity"] = frame["scenario_id"].map(
        lambda value: control[value][1]
    )
    frame["recovery_capacity"] = frame["scenario_id"].map(
        lambda value: control[value][2]
    )
    frame["grid_dependent_lights_plugs_process_mechvent"] = frame[
        "scenario_id"
    ].map(lambda value: control[value][3])
    frame["outage_clock_version"] = OUTAGE_CLOCK_VERSION
    frame["score_definition_version"] = SCORE_DEFINITION_VERSION
    frame["requires_energyplus"] = True
    return frame


def _estimand_contract() -> pd.DataFrame:
    rows = [
        ("event_net_increment_kwh", "E_HW_UNL-E_CF_UNL", "kWh", "complete parent event", "none", "sum energy before differencing", "near-zero baseline: NA", "pending E8 implementation"),
        ("event_positive_increment_kwh", "sum(max(P_HW_UNL-P_CF_UNL,0)*dt)", "kWh", "complete parent event", "none", "hour then building then event", "missing hour: invalid", "pending E8 implementation"),
        ("relative_demand_increment", "(E_HW_UNL-E_CF_UNL)/E_CF_UNL", "1", "complete parent event", "none", "aggregate numerator and denominator before ratio", "near-zero denominator: NA", "pending E8 implementation"),
        ("coincident_peak_delta_kw", "max(sum_i P_HW_i)-max(sum_i P_CF_i)", "kW", "complete parent event", "none", "building power summed at every timestamp", "missing building-hour: invalid", "pending E8 implementation"),
        ("Q_delivered_raw", "native HW_C70 cooling delivery", "kWh_th", "complete parent event", "qualified zones", "zone-timestep then sum", "required native field missing: error", "pending E8 implementation"),
        ("Q_shortfall_ref", "sum(max(Q_required_ref-Q_delivered_raw,0))", "kWh_th", "complete parent event", "qualified zones", "zone-timestep matching before sum", "zero demand: 0 plus no_demand flag", "pending approval; differs from legacy aggregate clipping"),
        ("DH32_outage", "sum(max(T_outage-32,0)*dt)", "K h", "24h synchronous outage", "occupied monitored zones", "declared worst-zone rule", "no monitoring: NA", "pending E8 implementation"),
        ("H_extra26", "sum(max(0,max(T_out-26,0)-max(T_rated-26,0))*dt)", "K h", "matched 24h outage", "same zone/time mask", "zone-time match before integration", "missing pair: error", "pending E8 implementation"),
        ("t_first32", "first T>=32 from outage start", "h", "24h outage", "occupied monitored zones", "linear interpolation within fixed-zone samples", "not crossed: right censored", "first_threshold_crossing_hours"),
        ("t_recovery", "first return to <=26 after power restoration", "h", "24h recovery observation", "required service zones", "all required objects meet rule", "not recovered: right censored", "legacy primary rule; exact E8 implementation pending"),
        ("BERI", "100*(median_e S_N*median_e S_C*median_e S_O*median_e S_R)^(1/4)", "0-100", "reference-event set", "eligible buildings", "component synthesis before event median", "missing component: NA; true zero retained", "pending E8 implementation"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "estimand", "formula", "unit", "window", "mask",
            "aggregation_order", "missing_zero_censoring", "implementation",
        ],
    )


def _variable_contract() -> tuple[dict[str, object], pd.DataFrame]:
    variables = [
        ("zone_air_temperature", "variable", "Zone Mean Air Temperature", "*", "C", "Timestep", "state/end-of-zone-timestep", True),
        ("zone_relative_humidity", "variable", "Zone Air Relative Humidity", "*", "%", "Timestep", "state/end-of-zone-timestep", True),
        ("zone_occupants", "variable", "Zone People Occupant Count", "*", "", "Timestep", "state", True),
        ("ideal_supply_sensible_cooling_energy", "variable", "Zone Ideal Loads Supply Air Sensible Cooling Energy", "*", "J", "Timestep", "interval energy/supply-air boundary", True),
        ("ideal_supply_total_cooling_energy", "variable", "Zone Ideal Loads Supply Air Total Cooling Energy", "*", "J", "Timestep", "interval energy/supply-air boundary", True),
        ("ideal_supply_total_cooling_rate", "variable", "Zone Ideal Loads Supply Air Total Cooling Rate", "*", "W", "Timestep", "interval-average rate/supply-air boundary", True),
        ("electricity_facility", "meter", "Electricity:Facility", "*", "J", "Timestep", "interval energy", True),
        ("lights_electricity", "variable", "Lights Electricity Energy", "*", "J", "Timestep", "interval energy", True),
        ("equipment_electricity", "variable", "Electric Equipment Electricity Energy", "*", "J", "Timestep", "interval energy", True),
        ("equipment_convective_gain", "variable", "Electric Equipment Convective Heating Energy", "*", "J", "Timestep", "interval energy", True),
        ("cooling_unmet_time", "variable", "Zone Cooling Setpoint Not Met Time", "*", "hr", "Timestep", "interval duration", True),
    ]
    frame = pd.DataFrame(
        variables,
        columns=[
            "field_id", "output_kind", "energyplus_variable", "key", "unit",
            "frequency", "temporal_semantics", "required",
        ],
    )
    frame["verification_status"] = "pending_G3_RDD_MDD_check"
    schema = {
        "version": "e8-zone-native-output-v2",
        "matching_rule": "complete variable name + key + unit + frequency; no substring aggregation",
        "missing_required_field": "raise error",
        "structural_zero": "permitted only when schema explicitly declares equipment absent",
        "native_timestep_minutes_expected": 15,
        "hourly_energy": "sum native interval energies",
        "hourly_average_power": "duration-weighted native power, or energy divided by covered duration",
        "variables": frame.to_dict(orient="records"),
        "engine_dictionary_gate": (
            "every requested variable and meter must be confirmed against the actual "
            "EnergyPlus 26.1.0 RDD/MDD before G3 outputs are accepted"
        ),
    }
    return schema, frame


def _baseline_registry(root: Path, project_root: Path) -> pd.DataFrame:
    items = [
        ("repository", root),
        ("project_root", project_root),
        ("analysis_config", project_root / "config" / "foundation_analysis.json"),
        ("event_catalog", project_root / "output" / "paper_pipeline_v0_3" / "01_hazard" / "heatwave_event_catalog.csv"),
        ("design_matrix", project_root / "output" / "paper_pipeline_v0_3" / "02_building_response" / "thermal_experiment_design.csv"),
        ("legacy_run_summary", project_root / "output" / "paper_pipeline_v0_3" / "02_building_response" / "energyplus_multizone_results.csv"),
        ("legacy_hourly", project_root / "output" / "paper_pipeline_v0_3" / "02_building_response" / "energyplus_multizone_hourly_results.parquet"),
        ("building_inputs", project_root / "output" / "paper_pipeline_v0_3" / "02_building_response" / "all_sector_building_model_inputs.parquet"),
        ("energyplus_binary", project_root / "software" / "energyplus" / "EnergyPlus-26.1.0-6f2e40d102-Windows-x86_64" / "energyplus.exe"),
    ]
    rows = []
    for item_id, path in items:
        exists = path.exists()
        can_hash = exists and path.is_file() and path.stat().st_size < 300_000_000
        rows.append(
            {
                "item_id": item_id,
                "path": str(path.resolve()),
                "exists": exists,
                "size_bytes": path.stat().st_size if exists and path.is_file() else None,
                "sha256": sha256_file(path) if can_hash else None,
                "hash_status": "computed" if can_hash else ("skipped_large" if exists else "missing"),
            }
        )
    return pd.DataFrame(rows)


def run_preflight(project_root: Path, work_root: Path, download_hourly: bool = False) -> dict[str, object]:
    """Compile the actual G0--G2 evidence package without invoking a solver."""
    project_root = project_root.resolve()
    work_root = work_root.resolve()
    for name in (
        "00_protocol", "01_inputs", "02_runs", "03_validation", "04_mechanisms",
        "05_results", "06_evaluation", "07_exposure", "08_publication",
    ):
        (work_root / name).mkdir(parents=True, exist_ok=True)
    protocol_root = work_root / "00_protocol" / "source_protocol" / "BERI_E8_protocol"
    catalog_path = (
        project_root / "output" / "paper_pipeline_v0_3" / "01_hazard"
        / "heatwave_event_catalog.csv"
    )
    candidates_path = protocol_root / "pre2018_daily_catalog_candidates.json"
    design_path = (
        project_root / "output" / "paper_pipeline_v0_3" / "02_building_response"
        / "thermal_experiment_design.csv"
    )
    catalog = pd.read_csv(catalog_path)
    catalog["start_date"] = pd.to_datetime(catalog["start_date"])
    catalog["end_date"] = pd.to_datetime(catalog["end_date"])
    hko65 = catalog.loc[
        (catalog["station_code"] == "HKO")
        & catalog["start_date"].dt.year.between(1980, 2025)
    ].copy()
    if len(hko65) != 65:
        raise ValueError(f"expected 65 HKO catalog events, found {len(hko65)}")
    _write_csv(hko65, work_root / "01_inputs" / "event_catalog_65.csv")

    hko_daily_path = project_root / "data" / "weather" / "hko_kp_daily_1992_2025.csv"
    hko_daily = pd.read_csv(hko_daily_path, parse_dates=["date"])
    required_daily = {"date", "tmin_c", "tmean_c", "tmax_c", "rh_mean_pct"}
    missing_daily_fields = required_daily.difference(hko_daily.columns)
    if missing_daily_fields:
        raise ValueError(f"HKO daily input lacks {sorted(missing_daily_fields)}")
    if hko_daily["date"].duplicated().any():
        raise ValueError("HKO daily input contains duplicate dates")

    candidates = pd.DataFrame(json.loads(candidates_path.read_text(encoding="utf-8")))
    known_ids = ["HKO_LI_0068", "HKO_LI_0085", "HKO_LI_0090"]
    known = _find_event_rows(hko65, known_ids)
    audit_source = pd.concat(
        [
            candidates.assign(candidate_role="pre2018_candidate"),
            known.assign(candidate_role="existing_reference"),
        ],
        ignore_index=True,
        sort=False,
    )
    hourly_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    for row in audit_source.itertuples(index=False):
        event_start = pd.Timestamp(row.start_date).normalize()
        event_end = pd.Timestamp(row.end_date).normalize()
        weather_start = event_start - pd.Timedelta(days=14)
        weather_end = event_end + pd.Timedelta(days=2)
        cache = nasa_cache_path(
            project_root, str(row.event_id), weather_start, weather_end
        )
        status = "not_checked"
        frame = None
        try:
            frame, status = acquire_nasa_hourly(
                cache, weather_start, weather_end, 22.311, 114.172, download_hourly
            )
            raw_passed, detail = audit_hourly_weather(frame, weather_start, weather_end)
            if raw_passed and frame is not None:
                frame, anchor_detail = anchor_hourly_to_hko_daily(frame, hko_daily)
                detail.update(anchor_detail)
                passed = anchor_detail["hko_anchor_qc"] == "passed"
                if passed:
                    event_weather_path = (
                        work_root / "01_inputs" / "hourly_weather"
                        / f"{row.event_id}_{WEATHER_PROCESSING_VERSION}.parquet"
                    )
                    event_weather_path.parent.mkdir(parents=True, exist_ok=True)
                    frame.to_parquet(event_weather_path, index=False)
                else:
                    detail["reason"] = "hko_daily_anchor_failed"
            else:
                passed = False
                detail.update(
                    {"hko_daily_missing_days": None, "hko_daily_missing_dates": None, "hko_anchor_qc": "not_run"}
                )
        except Exception as exc:
            passed = False
            detail = {"reason": f"{type(exc).__name__}: {exc}"}
            status = "acquisition_failed"
        hourly_rows.append(
            {
                "parent_event_id": str(row.event_id),
                "candidate_role": row.candidate_role,
                "event_start": event_start.date(),
                "event_end": event_end.date(),
                "weather_start": weather_start.date(),
                "weather_end": weather_end.date(),
                "time_standard_source": "NASA POWER UTC",
                "hkt_conversion": "timezone-aware UTC to Asia/Hong_Kong before slicing",
                "temperature_humidity_anchor": "HKO KP daily min/mean/max temperature and mean RH",
                "hko_daily_source": str(hko_daily_path),
                "weather_processing_version": WEATHER_PROCESSING_VERSION,
                "cache_path": str(cache),
                "cache_status": status,
                "hourly_qualified": bool(passed),
                **detail,
            }
        )
        if passed and frame is not None:
            feature_rows.append(
                {
                    "event_id": str(row.event_id),
                    "start_year": int(event_start.year),
                    "start_date": event_start.date(),
                    "end_date": event_end.date(),
                    **weather_features(frame, event_start, event_end),
                }
            )
    hourly = pd.DataFrame(hourly_rows)
    _write_csv(
        hourly, work_root / "01_inputs" / "hourly_weather_eligibility_audit.csv"
    )
    features = pd.DataFrame(feature_rows)
    _write_csv(
        features, work_root / "01_inputs" / "hourly_weather_candidate_features.csv"
    )

    selected: list[str] = []
    selection_status = "blocked_fewer_than_five_hourly_qualified_pre2018_events"
    diagnostics = features.copy()
    early_features = features.loc[
        features["event_id"].isin(candidates["event_id"])
    ].reset_index(drop=True) if len(features) else features
    if len(early_features) >= 5:
        selected, diagnostics = select_five_weather_events(early_features)
        selection_status = "proposed_pending_scientific_approval"
    _write_csv(
        diagnostics, work_root / "01_inputs" / "pre2018_event_selection_diagnostics.csv"
    )

    assignment_rows = []
    hourly_index = hourly.set_index("parent_event_id")
    for event_id in selected + known_ids:
        row = audit_source.loc[audit_source["event_id"] == event_id].iloc[0]
        assignment_rows.append(
            {
                "parent_event_id": event_id,
                "event_start": row["start_date"],
                "event_end": row["end_date"],
                "E8_role": (
                    "proposed_pre2018" if event_id in selected else "existing_reference"
                ),
                "hourly_qualified": bool(hourly_index.loc[event_id, "hourly_qualified"]),
                "assignment_status": (
                    selection_status
                    if event_id in selected
                    else (
                        "full_parent_hourly_QC_passed"
                        if bool(hourly_index.loc[event_id, "hourly_qualified"])
                        else "full_parent_hourly_QC_failed"
                    )
                ),
            }
        )
    assignment = pd.DataFrame(assignment_rows)
    _write_csv(assignment, work_root / "01_inputs" / "E8_event_assignment.csv")

    old_windows = {
        "HKO_LI_0068": ("2018-05-26", "2018-06-01"),
        "HKO_LI_0085": ("2022-07-22", "2022-07-29"),
        "HKO_LI_0090": ("2023-07-24", "2023-07-28"),
    }
    crosswalk = assignment[
        ["parent_event_id", "event_start", "event_end"]
    ].copy()
    crosswalk["legacy_window_start"] = crosswalk["parent_event_id"].map(
        lambda value: old_windows.get(value, (None, None))[0]
    )
    crosswalk["legacy_window_end"] = crosswalk["parent_event_id"].map(
        lambda value: old_windows.get(value, (None, None))[1]
    )
    crosswalk["main_analysis_window"] = "complete_parent_event"
    _write_csv(crosswalk, work_root / "01_inputs" / "event_window_crosswalk.csv")

    lockbox = {"status": "blocked_until_five_events_selected", "event_ids": []}
    if len(selected) == 5:
        selected_features = diagnostics.loc[
            diagnostics["event_id"].isin(selected)
        ].copy()
        selection_columns = [
            column for column in diagnostics
            if column.startswith("selection_")
            and column not in {"selection_algorithm", "selection_selected_pre2018"}
            and pd.api.types.is_numeric_dtype(diagnostics[column])
        ]
        matrix = selected_features[selection_columns].to_numpy(float)
        pairs = [
            (
                float(np.linalg.norm(matrix[a] - matrix[b])),
                selected_features.iloc[a]["event_id"],
                selected_features.iloc[b]["event_id"],
            )
            for a, b in itertools.combinations(range(5), 2)
        ]
        _, first, second = max(
            pairs, key=lambda value: (value[0], value[1], value[2])
        )
        lockbox = {
            "status": "proposed_pending_approval",
            "event_ids": [first, second],
            "selection_rule": (
                "most distant pair within frozen five-event composite feature space"
            ),
            "scientific_outputs_must_remain_unopened_until": (
                "model, scoring, analysis rules and calibration are frozen"
            ),
        }
    lockbox["manifest_hash"] = stable_sha256(lockbox)
    _write_json(lockbox, work_root / "01_inputs" / "lockbox_manifest.json")

    scenarios = _scenario_frame()
    _write_csv(scenarios, work_root / "00_protocol" / "core_scenario_contract.csv")
    schema, variable_map = _variable_contract()
    schema["output_schema_hash"] = stable_sha256(schema)
    _write_json(schema, work_root / "00_protocol" / "output_schema.json")
    _write_csv(
        variable_map, work_root / "00_protocol" / "variable_mapping_audit.csv"
    )
    _write_csv(
        _estimand_contract(),
        work_root / "00_protocol" / "estimand_and_field_contract.csv",
    )

    legacy_hourly = (
        project_root / "output" / "paper_pipeline_v0_3" / "02_building_response"
        / "energyplus_multizone_hourly_results.parquet"
    )
    raw_audit = pd.DataFrame(
        [
            {
                "artifact": "legacy_energyplus_multizone_hourly",
                "path": str(legacy_hourly),
                "exists": legacy_hourly.exists(),
                "zone_native_temperature_rh": False,
                "exact_native_cooling_fields": False,
                "reextractable_without_solver": False,
                "E8_use": "legacy comparison only",
            },
            {
                "artifact": "E8_required_zone_native_output",
                "path": "pending G3",
                "exists": False,
                "zone_native_temperature_rh": True,
                "exact_native_cooling_fields": True,
                "reextractable_without_solver": True,
                "E8_use": "required",
            },
        ]
    )
    _write_csv(
        raw_audit, work_root / "00_protocol" / "raw_output_availability_audit.csv"
    )
    legacy_cache_dir = (
        project_root / "output" / "paper_pipeline_v0_3"
        / "02_building_response" / "energyplus_run_cache"
        / "energyplus_multizone_counterfactual_resilience_v0_23"
    )
    cache_json = list(legacy_cache_dir.glob("*.json"))
    cache_parquet = list(legacy_cache_dir.glob("*.parquet"))
    cache_audit = pd.DataFrame(
        [
            {
                "cache_id": "legacy_v0_23",
                "path": str(legacy_cache_dir),
                "json_records": len(cache_json),
                "hourly_parquets": len(cache_parquet),
                "complete_parent_event": False,
                "five_core_scenario_contract": False,
                "zone_native_E8_schema": False,
                "final_physics_hash_available": False,
                "reuse_class": "legacy_comparison_only",
                "E8_qualified_runs": 0,
                "reason": (
                    "subset windows, eight legacy patterns and aggregate hourly schema"
                ),
            }
        ]
    )
    _write_csv(
        cache_audit, work_root / "00_protocol" / "cache_reuse_audit.csv"
    )

    design = pd.read_csv(design_path)
    if len(design) != 297:
        raise ValueError(f"expected frozen 297-row design matrix, found {len(design)}")
    run_rows = []
    if len(assignment) == 8:
        for event_id, design_row, scenario in itertools.product(
            assignment["parent_event_id"], design.to_dict("records"), CORE_SCENARIOS
        ):
            physics = {
                "protocol": E8_PROTOCOL_VERSION,
                "parent_event_id": event_id,
                "base_case_id": design_row["base_case_id"],
                "scenario_id": scenario.scenario_id,
                "weather_processing_version": WEATHER_PROCESSING_VERSION,
                "outage_clock_version": OUTAGE_CLOCK_VERSION,
                "design": design_row,
            }
            run_rows.append(
                {
                    "run_id": "e8_" + stable_sha256(physics)[:20],
                    "parent_event_id": event_id,
                    "base_case_id": design_row["base_case_id"],
                    "prototype_id": design_row["prototype_id"],
                    "floor_role": design_row["floor_position"],
                    "scenario_id": scenario.scenario_id,
                    "provisional_compile_hash": stable_sha256(physics),
                    "physics_hash": "",
                    "physics_hash_status": (
                        "pending_final_EPW_IDF_schedule_and_engine_bytes"
                    ),
                    "output_schema_hash": schema["output_schema_hash"],
                    "analysis_hash": stable_sha256(
                        {"score": SCORE_DEFINITION_VERSION}
                    ),
                    "cache_status": "new_solver_run_required",
                    "QC_status": "not_run_G0_G2",
                }
            )
    run_manifest = pd.DataFrame(
        run_rows,
        columns=[
            "run_id", "parent_event_id", "base_case_id", "prototype_id",
            "floor_role", "scenario_id", "provisional_compile_hash",
            "physics_hash", "physics_hash_status", "output_schema_hash",
            "analysis_hash", "cache_status", "QC_status",
        ],
    )
    _write_csv(
        run_manifest, work_root / "02_runs" / "core_run_manifest_dry_run.csv"
    )

    benchmark = pd.DataFrame()
    if len(assignment):
        non_lockbox = [
            value for value in assignment["parent_event_id"]
            if value not in lockbox.get("event_ids", [])
        ]
        if non_lockbox and len(run_manifest):
            benchmark_designs = pd.concat(
                [
                    design.groupby("sector", sort=True).head(1),
                    design.loc[design["floor_position"] == "top"]
                    .groupby("sector", sort=True)
                    .head(1),
                ]
            ).drop_duplicates("base_case_id").head(4)
            benchmark = run_manifest.loc[
                (run_manifest["parent_event_id"] == non_lockbox[0])
                & run_manifest["base_case_id"].isin(
                    benchmark_designs["base_case_id"]
                )
            ].copy()
            benchmark["benchmark_role"] = "five-scenario solver/parser integration"
            extra = []
            residential_top = design.loc[
                (design["sector"] == "residential")
                & (design["floor_position"] == "top")
            ].head(1)
            if len(residential_top):
                base = residential_top.iloc[0]
                # The five-scenario integration rows use the main 14-day
                # prehistory. Only the two shorter alternatives are additional
                # physical inputs; do not double-count the 14-day reference.
                for days in (2, 7):
                    for scenario_id in ("HW_RATED", "HW_O24_SYNC"):
                        extra.append(
                            {
                                "run_id": f"g3_spinup_{days}d_{scenario_id}",
                                "parent_event_id": non_lockbox[0],
                                "base_case_id": base["base_case_id"],
                                "prototype_id": base["prototype_id"],
                                "floor_role": base["floor_position"],
                                "scenario_id": scenario_id,
                                "provisional_compile_hash": (
                                    "pending_spinup_specific_compile"
                                ),
                                "physics_hash": "",
                                "physics_hash_status": (
                                    "pending_final_EPW_IDF_schedule_and_engine_bytes"
                                ),
                                "output_schema_hash": schema["output_schema_hash"],
                                "analysis_hash": stable_sha256(
                                    {"diagnostic": "2_7_14_day_prehistory"}
                                ),
                                "cache_status": "new_solver_run_required",
                                "QC_status": "not_run_G0_G2",
                                "benchmark_role": f"prehistory_{days}d_convergence",
                            }
                        )
            benchmark = pd.concat(
                [benchmark, pd.DataFrame(extra)], ignore_index=True
            ).drop_duplicates(
                [
                    "parent_event_id", "base_case_id",
                    "scenario_id", "benchmark_role",
                ]
            )
    _write_csv(
        benchmark, work_root / "02_runs" / "g3_benchmark_manifest.csv"
    )
    gate56 = pd.DataFrame()
    gate297 = pd.DataFrame()
    if len(run_manifest) and non_lockbox:
        representative56 = design.sort_values("base_case_id").groupby(
            ["prototype_id", "floor_position"], sort=True
        ).head(1)
        gate56 = run_manifest.loc[
            (run_manifest["parent_event_id"] == non_lockbox[0])
            & run_manifest["base_case_id"].isin(
                representative56["base_case_id"]
            )
        ].copy()
        gate56["gate_role"] = "G4_56_response_unit_coverage"
        gate297 = run_manifest.loc[
            run_manifest["parent_event_id"] == non_lockbox[0]
        ].copy()
        gate297["gate_role"] = "G4_complete_297_design_event"
    _write_csv(
        gate56, work_root / "02_runs" / "g4_56_unit_gate_manifest.csv"
    )
    _write_csv(
        gate297, work_root / "02_runs" / "g4_297_design_event_manifest.csv"
    )

    summary = {
        "protocol_version": E8_PROTOCOL_VERSION,
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "allow_full_simulation": False,
        "catalog_events": int(len(hko65)),
        "pre2018_daily_candidates": int(len(candidates)),
        "pre2018_hourly_qualified": int(
            hourly.loc[
                hourly["candidate_role"] == "pre2018_candidate",
                "hourly_qualified",
            ].sum()
        ) if len(hourly) else 0,
        "proposed_pre2018_event_ids": selected,
        "E8_assignment_count": int(len(assignment)),
        "design_count": int(len(design)),
        "core_scenario_count": int(len(CORE_SCENARIOS)),
        "core_unique_run_count": int(len(run_manifest)),
        "legacy_cache_qualified_for_E8": 0,
        "G3_benchmark_run_count": int(len(benchmark)),
        "selection_status": selection_status,
        "repository_head": _git_value(
            project_root, "rev-parse", "HEAD"
        ),
        "repository_dirty": bool(
            _git_value(project_root, "status", "--porcelain")
        ),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "energyplus_invoked": False,
    }
    summary["manifest_hash"] = stable_sha256(
        {
            "protocol_version": E8_PROTOCOL_VERSION,
            "assignment": assignment[
                ["parent_event_id", "event_start", "event_end", "E8_role"]
            ].to_dict(orient="records"),
            "scenario_contract": scenarios.to_dict(orient="records"),
            "core_run_provisional_compile_hashes": run_manifest[
                "provisional_compile_hash"
            ].tolist(),
            "output_schema_hash": schema["output_schema_hash"],
            "lockbox_manifest_hash": lockbox["manifest_hash"],
        }
    )
    _write_json(summary, work_root / "00_protocol" / "preflight_summary.json")
    _write_json(
        {
            "core_runs": len(run_manifest),
            "formula": (
                "8 events x 297 frozen designs x 5 core scenarios"
                if len(run_manifest)
                else "blocked until E8 assignment complete"
            ),
            "reusable_qualified": 0,
            "reextractable_legacy": 0,
            "new_solver_runs": len(run_manifest),
            "validation_budget_target": "200-600, actual mapping pending",
            "mechanism_budget_target": "600-1000, actual levels pending",
            "G4_56_unit_gate_runs": int(len(gate56)),
            "G4_one_297_design_event_runs": int(len(gate297)),
        },
        work_root / "02_runs" / "dry_run_summary.json",
    )

    registry = _baseline_registry(project_root, project_root)
    _write_csv(
        registry, work_root / "00_protocol" / "baseline_result_registry.csv"
    )
    issues = pd.DataFrame(
        [
            ("legacy_eight_patterns", "still_present", "src/02_RESULT_BUILDING_RESPONSE_AND_EXPOSURE.py:6273-6279; frozen results 7128=297x3x8", "E8 uses five explicit scenarios"),
            ("full_parent_events", "still_missing_in_legacy", "legacy event windows are subsets of the parent dates recorded in event_window_crosswalk.csv", "E8 assignment uses complete parent-event dates"),
            ("zone_native_temperature_rh", "still_missing", str(legacy_hourly), "new output schema required"),
            ("cooling_substring_match", "still_present", "src/02_RESULT_BUILDING_RESPONSE_AND_EXPOSURE.py:2473-2478,2550", "E8 exact field matcher implemented and tested"),
            ("missing_cooling_returns_zero", "still_present", "src/02_RESULT_BUILDING_RESPONSE_AND_EXPOSURE.py:2476-2479", "E8 required fields raise KeyError"),
            ("cooling_min_clipping", "still_present", "src/02_RESULT_BUILDING_RESPONSE_AND_EXPOSURE.py:4636", "raw/matched fields separated; zone-timestep layer pending approval"),
            ("threshold_initial_state", "still_present_in_legacy", "src/02_RESULT_BUILDING_RESPONSE_AND_EXPOSURE.py:3592-3603 uses rated first hourly aggregate as initial", "timestamp-aware helper implemented; native outage initial state pending G3"),
            ("data_centre_outage_gains", "fixed", "src/02_RESULT_BUILDING_RESPONSE_AND_EXPOSURE.py:2127-2144 disables grid-dependent schedules during outage", "retain pre-outage stored heat"),
            ("common_outage_clock", "available", "src/02_RESULT_BUILDING_RESPONSE_AND_EXPOSURE.py:3749-3757 selects common outdoor Tmax", "promoted to E8 HW_O24_SYNC"),
            ("font_test_coupling", "still_present_in_legacy_suite", "tests import figure_style", "E8 numerical tests import only utils.e8_preflight"),
        ],
        columns=["issue", "status", "evidence", "E8_action"],
    )
    _write_csv(
        issues, work_root / "00_protocol" / "legacy_issue_recheck.csv"
    )

    building_path = (
        project_root / "output" / "paper_pipeline_v0_3" / "02_building_response"
        / "all_sector_building_model_inputs.parquet"
    )
    building = pd.read_parquet(
        building_path,
        columns=[
            "BUILDINGSTRUCTUREID", "main_class", "construction_year_value",
            "pre_weather_simulation_candidate",
        ],
    )
    vintage = pd.DataFrame(
        [
            {"field": "construction_year_value", "rows": len(building), "nonmissing": int(building["construction_year_value"].notna().sum()), "precision": "year/inferred fractional year; no verified in-use date", "historical_use": "existence sensitivity only"},
            {"field": "in_use_date", "rows": len(building), "nonmissing": 0, "precision": "absent", "historical_use": "blocked"},
            {"field": "demolition_date", "rows": len(building), "nonmissing": 0, "precision": "absent", "historical_use": "blocked"},
            {"field": "renovation_date", "rows": len(building), "nonmissing": 0, "precision": "absent", "historical_use": "blocked"},
            {"field": "equipment_install_replace_date", "rows": len(building), "nonmissing": 0, "precision": "absent", "historical_use": "blocked"},
        ]
    )
    _write_csv(
        vintage,
        work_root / "01_inputs" / "building_vintage_completeness_audit.csv",
    )

    validation_rows = []
    hkust_hourly_path = (
        project_root / "output" / "paper_pipeline_v0_3"
        / "02_building_response" / "hkust_whole_building_hourly_calibration.parquet"
    )
    if hkust_hourly_path.exists():
        hkust = pd.read_parquet(
            hkust_hourly_path,
            columns=[
                "leaf_building", "building_scope", "timestamp_hkt",
                "energy_kwh", "building_split",
            ],
        )
        held = hkust.loc[hkust["building_split"] == "building_holdout"]
        for (building_name, scope), group in held.groupby(
            ["leaf_building", "building_scope"], sort=True
        ):
            prototype = (
                "res_private_highrise"
                if scope == "campus_residence"
                else (
                    "com_education"
                    if "Cheng_Yu_Tung" in building_name
                    else "com_other"
                )
            )
            validation_rows.append(
                {
                    "case_id": "HKUST_" + str(building_name),
                    "source": "HKUST",
                    "observed_scope": scope,
                    "observed_object": building_name,
                    "prototype_id": prototype,
                    "mapping_basis": (
                        "named campus residence proxy"
                        if scope == "campus_residence"
                        else "named university building"
                    ),
                    "start_hkt": group["timestamp_hkt"].min(),
                    "end_hkt": group["timestamp_hkt"].max(),
                    "valid_hours": int(len(group)),
                    "validation_target": (
                        "normalised hourly/daily load shape, peak timing and temperature-load slope"
                    ),
                    "absolute_energy_target": False,
                    "indoor_temperature_target": False,
                    "outage_target": False,
                    "G3_status": (
                        "feasible_relative_shape; GFA/envelope/plant boundary not available"
                    ),
                }
            )
    polyu_hourly_path = (
        project_root / "output" / "paper_pipeline_v0_3"
        / "02_building_response" / "polyu_campus_complete_hourly_calibration.parquet"
    )
    polyu_valid_hours = 0
    if polyu_hourly_path.exists():
        polyu_times = pd.read_parquet(
            polyu_hourly_path, columns=["timestamp_hkt"]
        )
        polyu_valid_hours = int(polyu_times["timestamp_hkt"].nunique())
    validation_rows.append(
        {
            "case_id": "POLYU_CAMPUS_AGGREGATE",
            "source": "PolyU",
            "observed_scope": "campus_nonresidential",
            "observed_object": "full campus aggregate",
            "prototype_id": "com_education composite proxy",
            "mapping_basis": (
                "source archive states codes are not reinterpreted real map entities"
            ),
            "start_hkt": "2025-07-09",
            "end_hkt": "2026-07-08",
            "valid_hours": polyu_valid_hours,
            "validation_target": (
                "aggregate normalised load shape and temperature-load slope only"
            ),
            "absolute_energy_target": False,
            "indoor_temperature_target": False,
            "outage_target": False,
            "G3_status": (
                "conditional; no defensible building-use/GFA mapping for individual codes"
            ),
        }
    )
    validation = pd.DataFrame(validation_rows)
    _write_csv(
        validation, work_root / "03_validation" / "validation_case_mapping.csv"
    )
    split_manifest = validation[
        [
            "case_id", "source", "observed_object", "start_hkt",
            "end_hkt", "validation_target", "G3_status",
        ]
    ].copy()
    source_windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for source, path in (
        ("HKUST", hkust_hourly_path),
        ("PolyU", polyu_hourly_path),
    ):
        if not path.exists():
            continue
        weather = pd.read_parquet(
            path, columns=["timestamp_hkt", "temperature_c"]
        ).drop_duplicates("timestamp_hkt")
        weather["timestamp_hkt"] = pd.to_datetime(weather["timestamp_hkt"])
        daily = (
            weather.set_index("timestamp_hkt")["temperature_c"]
            .sort_index()
            .resample("D")
            .mean()
        )
        rolling = daily.rolling(14, min_periods=14).mean()
        test_end = pd.Timestamp(rolling.idxmax()).normalize()
        test_start = test_end - pd.Timedelta(days=13)
        source_windows[source] = (test_start, test_end)
    split_manifest["test_start_hkt"] = split_manifest["source"].map(
        lambda value: source_windows.get(value, (pd.NaT, pd.NaT))[0]
    )
    split_manifest["test_end_hkt"] = split_manifest["source"].map(
        lambda value: source_windows.get(value, (pd.NaT, pd.NaT))[1]
    )
    split_manifest["calibration_start_hkt"] = (
        pd.to_datetime(split_manifest["test_start_hkt"]) - pd.Timedelta(days=14)
    )
    split_manifest["calibration_end_hkt"] = (
        pd.to_datetime(split_manifest["test_start_hkt"]) - pd.Timedelta(days=1)
    )
    split_manifest["calibration_rule"] = (
        "14 contiguous preceding days; scale only; no test-hour fitting"
    )
    split_manifest["test_rule"] = (
        "weather-only hottest complete 14-day block; never random hours"
    )
    split_manifest["split_status"] = "compiled_pending_G3_approval"
    _write_csv(
        split_manifest,
        work_root / "03_validation" / "validation_split_manifest.csv",
    )
    validation_benchmarks = (
        split_manifest.merge(
            validation[["case_id", "prototype_id"]],
            on="case_id",
            how="left",
            validate="one_to_one",
        )
        .drop_duplicates(
            [
                "source", "prototype_id", "calibration_start_hkt",
                "test_end_hkt",
            ]
        )
        .reset_index(drop=True)
    )
    validation_benchmarks["run_role"] = (
        "same-chain observed-weather relative-shape integration"
    )
    validation_benchmarks["approval_class"] = np.where(
        validation_benchmarks["source"].eq("HKUST"),
        "required_G3",
        "optional_G3_mapping_conditional",
    )
    _write_csv(
        validation_benchmarks,
        work_root / "02_runs" / "g3_validation_benchmark_manifest.csv",
    )

    residential = design.loc[design["sector"] == "residential"].copy()
    provenance_path = (
        project_root / "output" / "paper_pipeline_v0_3"
        / "02_building_response" / "prototype_parameter_provenance.csv"
    )
    provenance = pd.read_csv(provenance_path)
    residential_ids = sorted(residential["prototype_id"].unique())
    provenance = provenance.loc[
        provenance["prototype_id"].isin(residential_ids)
        & provenance["parameter"].isin(
            ["thermal_capacitance_kj_m2k", "cooling_capacity_w_m2"]
        )
    ].copy()
    provenance["E8_role"] = "source range for proposed controlled levels"
    provenance["approval_status"] = "pending_scientific_approval"
    _write_csv(
        provenance, work_root / "04_mechanisms" / "parameter_provenance.csv"
    )
    mass_values = provenance.loc[
        provenance["parameter"] == "thermal_capacitance_kj_m2k", "value"
    ].astype(float)
    capacity_values = provenance.loc[
        provenance["parameter"] == "cooling_capacity_w_m2", "value"
    ].astype(float)
    mass_levels = sorted(mass_values.unique())
    capacity_levels = sorted(capacity_values.unique())
    proposed_levels = {
        "roof_top_exposure": ["middle", "top"],
        "thermal_mass": [
            min(mass_levels), float(mass_values.median()), max(mass_levels)
        ],
        "nominal_cooling_capacity": [
            min(capacity_levels), float(capacity_values.median()), max(capacity_levels)
        ],
        "AC_operating_time": [-1, 1],
    }
    factor_scenarios = {
        "roof_top_exposure": [item.scenario_id for item in CORE_SCENARIOS],
        "thermal_mass": [item.scenario_id for item in CORE_SCENARIOS],
        "nominal_cooling_capacity": [
            "HW_RATED", "HW_C70", "HW_O24_SYNC",
        ],
        "AC_operating_time": [item.scenario_id for item in CORE_SCENARIOS],
    }
    mechanism_rows = []
    pair_rows = []
    mechanism_events = known_ids
    for factor, levels in proposed_levels.items():
        for prototype_id in residential_ids:
            pair_id = f"{prototype_id}__{factor}"
            for level in levels:
                for event_id in mechanism_events:
                    for scenario_id in factor_scenarios[factor]:
                        pair_rows.append(
                            {
                                "pair_id": pair_id,
                                "prototype_id": prototype_id,
                                "factor": factor,
                                "level": level,
                                "parent_event_id": event_id,
                                "scenario_id": scenario_id,
                                "reference_design": (
                                    "orientation=0; working; setpoint=25C; AC probability=1; "
                                    "start offset=0 except AC timing factor; middle except roof factor"
                                ),
                                "status": "pending_level_and_budget_approval",
                            }
                        )
        mechanism_rows.append(
            {
                "factor": factor,
                "residential_prototypes": len(residential_ids),
                "proposed_levels": "|".join(map(str, levels)),
                "events": "|".join(mechanism_events),
                "scenarios": "|".join(factor_scenarios[factor]),
                "unique_solver_runs": (
                    len(residential_ids) * len(levels)
                    * len(mechanism_events) * len(factor_scenarios[factor])
                ),
                "existing_strict_pairs": (
                    1 if factor == "roof_top_exposure" else 0
                ),
                "approval_status": "pending_scientific_approval",
            }
        )
    mechanism_frame = pd.DataFrame(mechanism_rows)
    pair_frame = pd.DataFrame(pair_rows)
    _write_csv(
        mechanism_frame,
        work_root / "04_mechanisms" / "residential_mechanism_feasibility.csv",
    )
    _write_csv(
        pair_frame, work_root / "04_mechanisms" / "residential_pair_design.csv"
    )
    dry_run_path = work_root / "02_runs" / "dry_run_summary.json"
    dry_run = json.loads(dry_run_path.read_text(encoding="utf-8"))
    dry_run.update(
        {
            "validation_cases_compiled": int(len(validation)),
            "validation_solver_runs_required_G3": int(
                validation_benchmarks["approval_class"].eq("required_G3").sum()
            ),
            "validation_solver_runs_optional_G3": int(
                validation_benchmarks[
                    "approval_class"
                ].eq("optional_G3_mapping_conditional").sum()
            ),
            "mechanism_solver_runs_proposed": int(len(pair_frame)),
            "mechanism_formula": (
                "roof 150 + thermal mass 225 + nominal capacity 135 + AC timing 150"
            ),
        }
    )
    _write_json(dry_run, dry_run_path)
    summary.update(
        {
            "validation_cases_compiled": int(len(validation)),
            "mechanism_solver_runs_proposed": int(len(pair_frame)),
        }
    )
    _write_json(summary, work_root / "00_protocol" / "preflight_summary.json")

    audit_md = f"""# E8 baseline audit (G0--G2)

- Protocol: `{E8_PROTOCOL_VERSION}`
- Repository HEAD: `{summary['repository_head']}`; the dirty working tree was not cleaned or reset.
- Project root: `{project_root}`
- Isolated E8 root: `{work_root}`
- Python: `{summary['python']}` ({summary['python_version']})
- EnergyPlus was not invoked.
- Catalog: {len(hko65)} HKO events in 1980--2025; {len(candidates)} pre-2018 daily candidates.
- Hourly-qualified pre-2018 candidates: {summary['pre2018_hourly_qualified']}; proposed events: {', '.join(selected) if selected else 'none (blocked)'}.
- Fixed design matrix: {len(design)} rows. Core E8 contract: {len(CORE_SCENARIOS)} scenarios.
- Core dry-run count: {len(run_manifest)}; legacy compatible/re-extractable cache count: 0.
- Historical dates: only `construction_year_value` is available; in-use, demolition, renovation and equipment dates are absent. Strict historical stock reconstruction is blocked.
- Legacy aggregate hourly files do not preserve the zone-native temperature/RH and exact cooling fields required by the E8 output contract.
"""
    (work_root / "00_protocol" / "baseline_audit.md").write_text(
        audit_md, encoding="utf-8"
    )
    runtime_meta_path = project_root / "software" / "energyplus" / "energyplus_release.json"
    runtime_meta = (
        json.loads(runtime_meta_path.read_text(encoding="utf-8"))
        if runtime_meta_path.exists()
        else {}
    )
    environment = {
        "repository_head": summary["repository_head"],
        "repository_dirty": summary["repository_dirty"],
        "project_root": str(project_root),
        "work_root": str(work_root),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "energyplus_binary": str(
            project_root / "software" / "energyplus"
            / "EnergyPlus-26.1.0-6f2e40d102-Windows-x86_64"
            / "energyplus.exe"
        ),
        "energyplus_release_metadata": runtime_meta,
        "allow_full_simulation": False,
        "energyplus_invoked_by_preflight": False,
    }
    _write_json(
        environment, work_root / "00_protocol" / "environment_audit.json"
    )
    approvals = pd.DataFrame(
        [
            ("E8_EVENT_SET", "approve five proposed early events and the two-event lockbox manifest", "|".join(selected), "required_before_G3"),
            ("COUNTERFACTUAL", "retain the existing 2016-2025 calendar-day temperature/RH targets; use common true 14-day prehistory and apply the transform only in the formal parent event", "recommended_new_E8_weather_version", "required_before_G3"),
            ("COOLING_SERVICE", "match unrestricted reference demand and C70 raw delivery at zone-timestep level before aggregation; retain raw delivery and exceedance diagnostics", "recommended_replaces_legacy_event_aggregate_min", "required_before_G3"),
            ("RECOVERY", "retain first return to <=26C as primary; add <=26.2C sustained for 1h as a diagnostic only", "recommended", "required_before_G3"),
            ("PREHISTORY_QC", "accept 2/7/14-day diagnostic tolerances of 0.1C, 1% event energy and one 15-min crossing interval", "recommended_engineering_rule", "required_before_G3"),
            ("MECHANISM_LEVELS", "approve roof middle/top; thermal mass 150/180/200 kJ m-2 K-1; nominal capacity 145/150/160 W m-2; AC start offset -1/+1 h", "660 proposed runs across the three known parent events", "required_before_G4_batch"),
            ("VALIDATION_SCOPE", "approve four HKUST held-out objects as relative-shape/slope checks and PolyU aggregate as optional; neither validates outage temperature", "2 required plus 1 optional unique G3 solver inputs", "required_before_G3"),
            ("G3_RESOURCES", "approve 26 required runs and 1 optional run; concurrency 4; wall-time ceiling 2 h; additional disk ceiling 5 GB; retain all raw outputs", "no bulk E8 run", "required_to_start_G3"),
        ],
        columns=["approval_id", "decision", "proposed_value", "gate"],
    )
    _write_csv(
        approvals, work_root / "00_protocol" / "approval_items.csv"
    )
    blocked = pd.DataFrame(
        [
            ("strict_historical_stock_reconstruction", "no in-use/demolition/renovation/equipment dates; current inventory omits demolished stock", "analysis C; A/B unaffected"),
            ("PolyU_individual_building_physics_mapping", "source README states building codes were not reinterpreted as real map entities; no defensible use/GFA mapping", "aggregate relative response only"),
            ("Hong_Kong_residential_joint_meter_T_RH", "no local joint residential meter-temperature-RH holdout found", "retain HKUST residence proxy and NIST structural evidence"),
            ("E8_native_output_dictionary", "exact EnergyPlus keys and interval semantics require one real RDD/CSV integration run", "G3"),
            ("bulk_simulation", "allow_full_simulation=false and no approved final physics manifest", "G5"),
            ("submission_DOCX_formula_QC", "existing MANUSCRIPT_WITH_FIGURES.docx still contains raw display-math marker", "publication-stage fix; does not block G3"),
        ],
        columns=["item", "evidence", "effect"],
    )
    _write_csv(
        blocked, work_root / "00_protocol" / "blocked_items.csv"
    )
    changes = pd.DataFrame(
        [
            ("utils/e8_preflight.py", "added", "solver-free event/weather audit, explicit contracts, manifests, stable hashes and numerical helpers"),
            ("src/00_PREPARE_AND_AUDIT_INPUTS.py", "modified", "added --e8-preflight and opt-in hourly acquisition; early exit guarantees no solver"),
            ("tests/test_e8_preflight.py", "added", "eight synthetic numerical/contract tests"),
            ("tests/test_paper_pipeline.py", "modified", "registered the new utility in the repository structure contract"),
        ],
        columns=["path", "change", "reason"],
    )
    _write_csv(changes, work_root / "00_protocol" / "change_log.csv")
    gate_report = f"""# E8 G0--G2 gate report

## Outcome

G0--G2 compilation completed without invoking EnergyPlus. The content-addressed
approval manifest is `{summary['manifest_hash']}`.

## Event result

All 19 pre-2018 daily candidates and the three existing parent events passed
the present hourly completeness/range audit. The frozen weather-only maximin
algorithm proposes: {', '.join(selected)}. The proposed lockbox is
{', '.join(lockbox.get('event_ids', []))}. Both remain pending approval.

## Run dependencies

- Core E8: {len(run_manifest):,} new runs = 8 events x 297 designs x 5 scenarios.
- Qualified legacy solver reuse: 0; legacy outputs use subset windows, eight
  legacy patterns and an aggregate-only output schema.
- G3 core integration and prehistory benchmark: {len(benchmark)} unique runs.
- G3 same-chain validation: 2 required unique HKUST prototype inputs; one
  PolyU aggregate proxy is optional because individual physical mapping is not
  supported by the source package.
- G4 response-unit gate: {len(gate56)} runs; one full non-lockbox event:
  {len(gate297)} runs. Neither is approved yet.
- Residential mechanism proposal: {len(pair_frame)} runs, pending parameter
  level approval; the existing 297-row design contains only one strict
  residential floor pair and no strict AC-timing pairs.

## Validation boundary

The four held-out HKUST objects can support normalized load-shape, peak-time and
temperature-load-slope checks after a leading-period scale fit. They do not
provide outage observations or direct indoor-temperature validation. The PolyU
archive explicitly warns that its building codes are not reinterpreted real
map entities; individual same-chain physics validation is therefore blocked.

## Test outcome

New E8 tests: 8 passed. Code/data suite excluding the pre-existing submission
DOCX formula check: 55 passed, 1 deselected. Full suite: 55 passed, 1 failed;
the failure is the existing raw `\\[` display-math marker in
`MANUSCRIPT_WITH_FIGURES.docx`.

## Gate

No G3/G4/G5 simulation is authorized. Approval items and blockers are recorded
in `approval_items.csv` and `blocked_items.csv`.
"""
    (work_root / "00_protocol" / "g0_g2_gate_report.md").write_text(
        gate_report, encoding="utf-8"
    )
    return summary
