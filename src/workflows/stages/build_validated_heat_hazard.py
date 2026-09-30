"""Build long-term, multi-definition and spatially validated Hong Kong heat hazard.

The script reads the frozen HKO inputs prepared by 00, implements and audits
the published Li et al. (2025) textual definition for the temporal trend, and models official very-hot-day and
hot-night counts as separate spatial outcomes. A 100 m output grid is released
only when land-cover/morphology models predict unseen stations better than both
nearest-station and geographic-only baselines. This script never downloads data.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
from scipy.ndimage import distance_transform_edt, uniform_filter
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
CONFIG = json.loads((ROOT / "config" / "foundation_analysis.json").read_text(encoding="utf-8"))
HAZARD_CONFIG = CONFIG["hazard"]
RANDOM_SEED = int(CONFIG["random_seed"])
OUT = ROOT / CONFIG["output_root"] / "01_hazard"
CACHE = ROOT / "data" / "weather" / "hko_open_data_daily"
CIS_CACHE = ROOT / "data" / "weather" / "hko_cis_daily_yearly"
BUILDINGS = ROOT / "output" / "heatwave_stock_foundation_v0_3" / "building_stock_repaired.geoparquet"
COORD_JS = ROOT / "data" / "weather" / "raw" / "hko_multistation" / "hko_station_coordinates.js"
LEGACY_WEATHER = ROOT / "intermediate" / "source_derived" / "weather" / "hko_multistation_daily_2016_2025.csv"
CIS_YEAR_API = "https://www.hko.gov.hk/cis/individual_day/daily_{year}.xml"
HKO_OPEN_DATA_API = "https://data.weather.gov.hk/weatherAPI/opendata/opendata.php"
WORLD_COVER_DIR = ROOT / "data" / "spatial" / "esa_worldcover_2021_v200"
WORLD_COVER_TILES = [
    WORLD_COVER_DIR / "ESA_WorldCover_10m_2021_v200_N21E111_Map.tif",
    WORLD_COVER_DIR / "ESA_WorldCover_10m_2021_v200_N21E114_Map.tif",
]
CSDI_ROAD_DIR = ROOT / "data" / "spatial" / "csdi_hk_transport_road_network_2026_08_31"
WRF_HEATWAVE_ROOT = ROOT / "data" / "weather" / "cmip6_wrf_prd" / "data" / "daily_max_data"
CMIP6_POINT_ROOT = ROOT / "data" / "climate" / "cmip6_hong_kong_point_raw"
CMIP6_COMPOUND_ROOT = ROOT / "data" / "climate" / "cmip6_hong_kong_compound_amon_raw"

# Shared source scope is configured once and used by 00 preparation and 01 analysis.
DAILY_STATIONS = [str(code) for code in HAZARD_CONFIG["hko_daily_stations"]]
CIS_YEARS = range(int(HAZARD_CONFIG["hko_cis_start_year"]), int(HAZARD_CONFIG["hko_cis_end_year"]) + 1)
OFFSHORE_EXCLUSIONS = {"WGL"}
ALIASES: dict[str, str] = {}
OFFICIAL_COORDINATE_OVERRIDES = {
    # HKO Summary of Meteorological Observations: Tai Po (Conservation
    # Studies Centre), 22°26'46"N 114°10'44"E.  The current JavaScript map
    # omits a temperature coordinate because this historical site is closed.
    "TPO": (22 + 26 / 60 + 46 / 3600, 114 + 10 / 60 + 44 / 3600),
}
WARM_MONTHS = set(HAZARD_CONFIG["warm_months"])
HKO_PUBLISHED_ANNUAL_COUNTS = [
    {"year": 2023, "hko_very_hot_days_published": 54, "hko_hot_nights_published": 56, "hko_extremely_hot_days_published": 4, "source": "https://www.hko.gov.hk/en/wxinfo/pastwx/2023/ywx2023.htm"},
    {"year": 2024, "hko_very_hot_days_published": 52, "hko_hot_nights_published": 50, "hko_extremely_hot_days_published": 2, "source": "https://www.hko.gov.hk/en/wxinfo/pastwx/2024/ywx2024.htm"},
    {"year": 2025, "hko_very_hot_days_published": 53, "hko_hot_nights_published": 54, "hko_extremely_hot_days_published": 4, "source": "https://www.hko.gov.hk/en/wxinfo/pastwx/2025/ywx2025.htm"},
]


def number(value: object) -> float:
    if value is None:
        return math.nan
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group()) if match else math.nan


def parse_hko_payload(payload: dict, station: str, data_type: str) -> pd.DataFrame:
    fields = [str(x).strip().lower() for x in payload.get("fields", [])]
    data = payload.get("data", [])
    if not fields or not data:
        return pd.DataFrame(columns=["date", "station_code", data_type])

    def locate(term: str, fallback: int) -> int:
        for index, field in enumerate(fields):
            if term in field:
                return index
        return fallback

    iy, im, iday = locate("year", 0), locate("month", 1), locate("day", 2)
    candidates = [i for i, field in enumerate(fields) if "value" in field or "temperature" in field]
    ivalue = candidates[-1] if candidates else 3
    rows = []
    for row in data:
        try:
            date = pd.Timestamp(int(number(row[iy])), int(number(row[im])), int(number(row[iday])))
        except (ValueError, TypeError, IndexError):
            continue
        rows.append({"date": date, "station_code": station, data_type: number(row[ivalue])})
    return pd.DataFrame(rows)


def parse_hko_cis_year(payload: dict, year: int) -> pd.DataFrame:
    elements = payload.get("stn", {}).get("data", [])
    maximum = next((element for element in elements if element.get("code") == "MAX_TEMP"), None)
    if maximum is None:
        raise ValueError(f"MAX_TEMP is absent from HKO CIS daily file for {year}")
    rows = []
    for day_row in maximum.get("dayData", []):
        day = int(number(day_row[0]))
        for month, raw in enumerate(day_row[1:13], start=1):
            try:
                date = pd.Timestamp(year, month, day)
            except ValueError:
                continue
            rows.append({"date": date, "station_code": "HKO", "tmax_c": number(raw)})
    return pd.DataFrame(rows)


def load_hko_cis_tmax(cache: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the frozen HKO CIS series prepared by 00; this analysis never downloads data."""
    expected = [cache / f"daily_{year}.json" for year in CIS_YEARS]
    missing = [path for path in expected if not path.is_file()]
    if missing:
        preview = ", ".join(path.name for path in missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise FileNotFoundError(
            f"Missing {len(missing)} HKO CIS input files ({preview}{suffix}). "
            "Run src/workflows/stages/prepare_and_audit_inputs.py first."
        )
    frames, audit = [], []
    for year, path in zip(CIS_YEARS, expected):
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        frame = parse_hko_cis_year(payload, year)
        expected_rows = 366 if pd.Timestamp(year, 12, 31).is_leap_year else 365
        if len(frame) != expected_rows or int(frame["tmax_c"].notna().sum()) != expected_rows:
            raise ValueError(
                f"Invalid HKO CIS cache {path}: expected {expected_rows} complete days, "
                f"found {len(frame)} rows and {int(frame['tmax_c'].notna().sum())} numeric Tmax values. "
                "Rerun 00 to prepare and validate the input."
            )
        frames.append(frame)
        audit.append(
            {
                "year": year,
                "source_url": CIS_YEAR_API.format(year=year),
                "cache_file": str(path),
                "rows": len(frame),
                "nonmissing_tmax_rows": int(frame["tmax_c"].notna().sum()),
                "transport": "validated_cache",
            }
        )
    weather = pd.concat(frames, ignore_index=True).sort_values("date")
    return weather, pd.DataFrame(audit)


def load_hko_open_data_cache(station: str, data_type: str, cache: Path) -> pd.DataFrame:
    """Read one frozen HKO Open Data file prepared and validated by 00."""
    path = cache / f"{data_type}_{station}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing HKO Open Data input: {path}. "
            "Run src/workflows/stages/prepare_and_audit_inputs.py first."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid HKO Open Data cache {path}: {exc}. Rerun 00.") from exc
    frame = parse_hko_payload(payload, station, data_type)
    nonmissing = int(frame[data_type].notna().sum()) if data_type in frame else 0
    if frame.empty or nonmissing == 0:
        raise ValueError(
            f"Invalid HKO Open Data cache {path}: rows={len(frame)}, nonmissing={nonmissing}. Rerun 00."
        )
    frame.attrs["transport"] = "validated_cache"
    return frame


def audit_available_cis_against_open_api(api_tmax: pd.DataFrame, cis_cache: Path) -> pd.DataFrame:
    """Compare every complete local CIS year with the canonical HKO all-year series."""
    api = api_tmax.loc[:, ["date", "tmax_c"]].copy()
    api["date"] = pd.to_datetime(api["date"])
    rows = []
    for year in CIS_YEARS:
        path = cis_cache / f"daily_{year}.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        cis = parse_hko_cis_year(payload, year).rename(columns={"tmax_c": "cis_tmax_c"})
        expected_rows = 366 if pd.Timestamp(year, 12, 31).is_leap_year else 365
        if len(cis) != expected_rows or int(cis["cis_tmax_c"].notna().sum()) != expected_rows:
            raise ValueError(f"Invalid complete CIS file during source audit: {path}")
        overlap = cis.merge(
            api.rename(columns={"tmax_c": "open_api_tmax_c"}),
            on="date",
            how="inner",
            validate="one_to_one",
        )
        difference = (overlap["cis_tmax_c"] - overlap["open_api_tmax_c"]).abs()
        rows.append(
            {
                "year": year,
                "expected_calendar_days": expected_rows,
                "overlap_days": len(overlap),
                "numeric_comparison_days": int(difference.notna().sum()),
                "maximum_absolute_difference_c": float(difference.max()),
                "differing_days": int(difference.gt(0).sum()),
                "exact_match": bool(
                    len(overlap) == expected_rows
                    and int(difference.notna().sum()) == expected_rows
                    and difference.eq(0).all()
                ),
                "cis_cache_file": str(path),
            }
        )
    return pd.DataFrame(rows)


def load_weather(cache: Path, allow_legacy: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    tasks = [
        (station, api_type, column)
        for station in DAILY_STATIONS
        for api_type, column in [("CLMMAXT", "tmax_c"), ("CLMMINT", "tmin_c")]
    ]
    missing = [cache / f"{api_type}_{station}.json" for station, api_type, _ in tasks if not (cache / f"{api_type}_{station}.json").is_file()]
    if missing:
        if allow_legacy and len(missing) == len(tasks):
            weather = pd.read_csv(LEGACY_WEATHER, usecols=["date", "station_code", "tmax_c", "tmin_c"])
            weather["date"] = pd.to_datetime(weather["date"])
            audit = pd.DataFrame(
                [{
                    "station_code": "legacy_14_station_file",
                    "variable": "tmax_c+tmin_c",
                    "status": "fallback",
                    "rows": len(weather),
                    "nonmissing_rows": int(weather[["tmax_c", "tmin_c"]].notna().all(axis=1).sum()),
                    "transport": "frozen_legacy_input",
                }]
            )
            return weather, audit
        preview = ", ".join(path.name for path in missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise FileNotFoundError(
            f"Missing {len(missing)} HKO Open Data input files ({preview}{suffix}). "
            "Run src/workflows/stages/prepare_and_audit_inputs.py first."
        )

    parts: list[pd.DataFrame] = []
    audit: list[dict] = []
    for station in DAILY_STATIONS:
        station_variables: dict[str, pd.DataFrame] = {}
        for api_type, column in [("CLMMAXT", "tmax_c"), ("CLMMINT", "tmin_c")]:
            frame = load_hko_open_data_cache(station, api_type, cache).rename(columns={api_type: column})
            station_variables[column] = frame
            audit.append(
                {
                    "station_code": station,
                    "variable": column,
                    "status": "available",
                    "rows": len(frame),
                    "nonmissing_rows": int(frame[column].notna().sum()),
                    "transport": "validated_cache",
                }
            )
        parts.append(
            station_variables["tmax_c"].merge(
                station_variables["tmin_c"], on=["date", "station_code"], how="outer"
            )
        )
    weather = pd.concat(parts, ignore_index=True)
    weather["date"] = pd.to_datetime(weather["date"])
    weather = weather.drop_duplicates(["station_code", "date"], keep="last").sort_values(["station_code", "date"])
    weather = weather.loc[weather["date"].dt.year.le(2025)].copy()
    return weather, pd.DataFrame(audit)


def parse_station_coordinates(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(r'stationConfigAWS\["([^\"]+)"\]\s*=\s*\{(.*?)(?=\n\};)', re.DOTALL)
    rows = []
    for key, block in pattern.findall(text):
        match = re.search(
            r'temp:\s*\{\s*code:\s*"([^\"]*)"\s*,\s*latitude:\s*"([^\"]*)"\s*,\s*longitude:\s*"([^\"]*)"',
            block,
        )
        if not match:
            continue
        temp_code, lat, lon = match.groups()
        if not lat or not lon:
            continue
        for code in {key.upper(), temp_code.upper()}:
            if code:
                rows.append({"lookup_code": code, "latitude": float(lat), "longitude": float(lon)})
    coords = pd.DataFrame(rows).drop_duplicates("lookup_code")
    requested = pd.DataFrame({"station_code": DAILY_STATIONS})
    requested["lookup_code"] = requested["station_code"].replace(ALIASES)
    result = requested.merge(coords, on="lookup_code", how="left").drop(columns="lookup_code")
    for station_code, (latitude, longitude) in OFFICIAL_COORDINATE_OVERRIDES.items():
        mask = result["station_code"].eq(station_code) & result["latitude"].isna()
        result.loc[mask, ["latitude", "longitude"]] = [latitude, longitude]
    return result


def warm_day_index(dates: pd.Series, policy: str = "noleap_calendar") -> np.ndarray:
    if policy == "native_dayofyear":
        return dates.dt.dayofyear.to_numpy(int)
    if policy != "noleap_calendar":
        raise ValueError(f"Unknown calendar-day policy: {policy}")
    # A fixed non-leap reference maps 29 February to 28 February and keeps all
    # May--September calendar days aligned between leap and non-leap years.
    return np.array([pd.Timestamp(2001, int(x.month), min(int(x.day), 28 if int(x.month) == 2 else int(x.day))).dayofyear for x in dates])


def moving_percentile_threshold(
    frame: pd.DataFrame,
    start_year: int,
    end_year: int,
    interpolation: str = "linear",
    sample_scope: str = "full_year",
    day_index_policy: str = "noleap_calendar",
) -> pd.Series:
    baseline = frame.loc[frame["date"].dt.year.between(start_year, end_year) & frame["tmax_c"].notna()].copy()
    if sample_scope == "warm_months":
        baseline = baseline.loc[baseline["date"].dt.month.isin(WARM_MONTHS)].copy()
    elif sample_scope != "full_year":
        raise ValueError(f"Unknown threshold sample scope: {sample_scope}")
    if len(baseline) < 10 * 153:
        return pd.Series(np.nan, index=frame.index)
    base_doy = warm_day_index(baseline["date"], day_index_policy)
    target_doy = warm_day_index(frame["date"], day_index_policy)
    thresholds: dict[int, float] = {}
    for doy in np.unique(target_doy):
        sample = baseline.loc[np.abs(base_doy - doy) <= 7, "tmax_c"]
        if len(sample) < 100:
            thresholds[int(doy)] = math.nan
        elif interpolation.startswith("numpy_"):
            thresholds[int(doy)] = float(np.quantile(sample.to_numpy(float), float(HAZARD_CONFIG["li_percentile"]) / 100.0, method=interpolation.removeprefix("numpy_")))
        else:
            thresholds[int(doy)] = float(sample.quantile(float(HAZARD_CONFIG["li_percentile"]) / 100.0, interpolation=interpolation))
    return pd.Series([thresholds.get(int(day), math.nan) for day in target_doy], index=frame.index)


def consecutive_intervals(
    dates: pd.Series,
    exceed: pd.Series,
    minimum_days: int = int(HAZARD_CONFIG["li_minimum_consecutive_days"]),
    merge_gap_days: int = int(HAZARD_CONFIG["li_merge_gap_days"]),
    assembly: str = "qualify_then_merge",
) -> list[tuple[pd.Timestamp, pd.Timestamp, set[pd.Timestamp]]]:
    hot_dates = [pd.Timestamp(x) for x in dates.loc[exceed.fillna(False)].sort_values()]
    runs: list[list[pd.Timestamp]] = []
    for date in hot_dates:
        if not runs or (date - runs[-1][-1]).days != 1:
            runs.append([date])
        else:
            runs[-1].append(date)
    if assembly == "qualify_then_merge":
        candidate_runs = [run for run in runs if len(run) >= minimum_days]
    elif assembly == "bridge_then_qualify":
        candidate_runs = []
        for run in runs:
            if candidate_runs and (run[0] - candidate_runs[-1][-1]).days - 1 <= merge_gap_days:
                candidate_runs[-1].extend(run)
            else:
                candidate_runs.append(list(run))
        candidate_runs = [run for run in candidate_runs if len(run) >= minimum_days]
    else:
        raise ValueError(f"Unknown event assembly: {assembly}")
    merged: list[tuple[pd.Timestamp, pd.Timestamp, set[pd.Timestamp]]] = []
    for run in candidate_runs:
        start, end, members = run[0], run[-1], set(run)
        if merged and (start - merged[-1][1]).days - 1 <= merge_gap_days:
            old_start, _, old_members = merged[-1]
            merged[-1] = (old_start, end, old_members | members)
        else:
            merged.append((start, end, members))
    return merged


def climpact_period_completeness(frame: pd.DataFrame, column: str) -> dict[str, int | bool]:
    """Apply the WMO Climpact rule: <=3 missing days/month and <=15/period."""
    missing_by_month = frame[column].isna().groupby(frame["date"].dt.to_period("M")).sum()
    missing = int(frame[column].isna().sum())
    max_monthly_missing = int(missing_by_month.max()) if len(missing_by_month) else 0
    return {
        "expected": int(len(frame)),
        "observed": int(frame[column].notna().sum()),
        "missing": missing,
        "max_monthly_missing": max_monthly_missing,
        "complete": bool(missing <= 15 and max_monthly_missing <= 3),
    }


def definition_metrics(weather: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    flags, annual, coverage, events = [], [], [], []
    for station, raw in weather.groupby("station_code", sort=True):
        raw = raw.sort_values("date").drop_duplicates("date", keep="last").copy()
        if raw.empty or raw["tmax_c"].notna().sum() == 0:
            continue
        first_observed_year = int(raw["date"].dt.year.min())
        last_observed_year = int(raw["date"].dt.year.max())
        full_index = pd.date_range(f"{first_observed_year}-01-01", f"{last_observed_year}-12-31", freq="D")
        full_station_frame = raw.set_index("date").reindex(full_index).rename_axis("date").reset_index()
        full_station_frame["station_code"] = station
        full_station_frame["li_analysis_window"] = full_station_frame["date"].dt.month.isin(WARM_MONTHS)
        first_year = max(1980, int(full_station_frame.loc[full_station_frame["tmax_c"].notna(), "date"].dt.year.min()))
        # Exact HKO replication uses Li et al.'s full 1980--2023 sample to
        # construct the calendar-day thresholds.  Other stations use their
        # complete available record and are explicitly a transfer analysis.
        threshold_end_year = 2023 if station == "HKO" else 2025
        full_threshold = moving_percentile_threshold(full_station_frame, first_year, threshold_end_year)
        full_station_frame["li_threshold_c"] = full_threshold.where(full_station_frame["li_analysis_window"])
        full_station_frame["li_hot_member"] = (
            full_station_frame["li_analysis_window"]
            & full_station_frame["tmax_c"].gt(full_station_frame["li_threshold_c"])
        )
        full_station_frame["li_event_span"] = False
        full_station_frame["li_event_id"] = pd.NA
        warm_frame = full_station_frame.loc[full_station_frame["li_analysis_window"]]
        intervals = consecutive_intervals(warm_frame["date"], warm_frame["li_hot_member"])
        for event_number, (start, end, _) in enumerate(intervals, start=1):
            mask = full_station_frame["date"].between(start, end)
            event_id = f"{station}_LI_{event_number:04d}"
            full_station_frame.loc[mask, "li_event_span"] = True
            full_station_frame.loc[mask, "li_event_id"] = event_id
        full_station_frame["hko_vhd"] = full_station_frame["tmax_c"].ge(float(HAZARD_CONFIG["hko_very_hot_day_c"]))
        full_station_frame["hko_ehd"] = full_station_frame["tmax_c"].ge(float(HAZARD_CONFIG["hko_extremely_hot_day_c"]))
        full_station_frame["hko_hot_night"] = full_station_frame["tmin_c"].ge(float(HAZARD_CONFIG["hko_hot_night_c"]))
        full_station_frame["li_excess_c"] = (full_station_frame["tmax_c"] - full_station_frame["li_threshold_c"]).clip(lower=0)

        completeness_by_year = {}
        for year, year_frame in full_station_frame.groupby(full_station_frame["date"].dt.year):
            warm_year = year_frame.loc[year_frame["li_analysis_window"]]
            full_tmax = climpact_period_completeness(year_frame, "tmax_c")
            full_tmin = climpact_period_completeness(year_frame, "tmin_c")
            warm_tmax = climpact_period_completeness(warm_year, "tmax_c")
            warm_tmin = climpact_period_completeness(warm_year, "tmin_c")
            completeness_by_year[int(year)] = {
                "full_tmax": full_tmax,
                "full_tmin": full_tmin,
                "warm_tmax": warm_tmax,
                "warm_tmin": warm_tmin,
            }
            coverage.append(
                {
                    "station_code": station,
                    "year": int(year),
                    "days_expected": full_tmax["expected"],
                    "tmax_days_observed": full_tmax["observed"],
                    "tmin_days_observed": full_tmin["observed"],
                    "tmax_days_missing": full_tmax["missing"],
                    "tmin_days_missing": full_tmin["missing"],
                    "max_monthly_tmax_days_missing": full_tmax["max_monthly_missing"],
                    "max_monthly_tmin_days_missing": full_tmin["max_monthly_missing"],
                    "warm_days_expected": warm_tmax["expected"],
                    "warm_tmax_days_observed": warm_tmax["observed"],
                    "warm_tmin_days_observed": warm_tmin["observed"],
                    "complete_tmax": full_tmax["complete"],
                    "complete_tmin": full_tmin["complete"],
                    "complete_li_tmax": warm_tmax["complete"],
                    "complete_warm_tmin": warm_tmin["complete"],
                    "completeness_rule": "WMO Climpact: <=3 missing days/month and <=15 missing days/period",
                    "threshold_sample_start_year": first_year,
                    "threshold_sample_end_year": threshold_end_year,
                }
            )

        for event_id, event_frame in full_station_frame.loc[full_station_frame["li_event_id"].notna()].groupby("li_event_id", sort=False):
            duration_days = int((event_frame["date"].max() - event_frame["date"].min()).days + 1)
            cumulative_intensity = float(event_frame["li_excess_c"].sum())
            event_year = int(event_frame["date"].min().year)
            events.append({
                "station_code": station,
                "event_id": event_id,
                "start_date": event_frame["date"].min(),
                "end_date": event_frame["date"].max(),
                "start_year": event_year,
                "duration_days": duration_days,
                "qualifying_hot_days": int(event_frame["li_hot_member"].sum()),
                "mean_intensity_c": cumulative_intensity / duration_days,
                "cumulative_intensity_c_days": cumulative_intensity,
                "input_year_complete": completeness_by_year[event_year]["warm_tmax"]["complete"],
            })
        flags.append(full_station_frame)
        for year, group in full_station_frame.groupby(full_station_frame["date"].dt.year):
            status = completeness_by_year[int(year)]
            warm_group = group.loc[group["li_analysis_window"]]
            li_complete = bool(status["warm_tmax"]["complete"])
            event_ids = warm_group.loc[warm_group["li_event_span"], "li_event_id"].dropna().unique()
            durations = [int((full_station_frame.loc[full_station_frame["li_event_id"].eq(eid), "date"].max() - full_station_frame.loc[full_station_frame["li_event_id"].eq(eid), "date"].min()).days + 1) for eid in event_ids]
            event_cumulative_intensities = [
                float(
                    full_station_frame.loc[
                        full_station_frame["li_event_id"].eq(eid),
                        "li_excess_c",
                    ].sum()
                )
                for eid in event_ids
            ]
            annual.append(
                {
                    "station_code": station,
                    "year": int(year),
                    "li_event_span_days": int(warm_group["li_event_span"].sum()) if li_complete else math.nan,
                    "li_qualifying_hot_days": int((warm_group["li_hot_member"] & warm_group["li_event_id"].notna()).sum()) if li_complete else math.nan,
                    "li_event_count": len(event_ids) if li_complete else math.nan,
                    "li_mean_event_duration_days": (float(np.mean(durations)) if durations else 0.0) if li_complete else math.nan,
                    "li_cumulative_intensity_c_days": float(warm_group.loc[warm_group["li_event_span"], "li_excess_c"].sum()) if li_complete else math.nan,
                    "li_mean_event_cumulative_intensity_c_days": (
                        float(np.mean(event_cumulative_intensities)) if event_cumulative_intensities else 0.0
                    ) if li_complete else math.nan,
                    "hko_very_hot_days": int(group["hko_vhd"].sum()) if status["full_tmax"]["complete"] else math.nan,
                    "hko_extremely_hot_days": int(group["hko_ehd"].sum()) if status["full_tmax"]["complete"] else math.nan,
                    "hko_hot_nights": int(group["hko_hot_night"].sum()) if status["full_tmin"]["complete"] else math.nan,
                    "hko_very_hot_days_may_sep": int(warm_group["hko_vhd"].sum()) if status["warm_tmax"]["complete"] else math.nan,
                    "hko_extremely_hot_days_may_sep": int(warm_group["hko_ehd"].sum()) if status["warm_tmax"]["complete"] else math.nan,
                    "hko_hot_nights_may_sep": int(warm_group["hko_hot_night"].sum()) if status["warm_tmin"]["complete"] else math.nan,
                }
            )
    return pd.concat(flags, ignore_index=True), pd.DataFrame(annual), pd.DataFrame(coverage), pd.DataFrame(events)


def fixed_effect_slope(frame: pd.DataFrame, value: str) -> tuple[float, float]:
    data = frame[["station_code", "year", value]].dropna().copy()
    if data["station_code"].nunique() < 1 or len(data) < 8:
        return math.nan, math.nan
    y = data[value].to_numpy(float)
    station_dummies = pd.get_dummies(data["station_code"], drop_first=True, dtype=float).to_numpy()
    x = np.column_stack([np.ones(len(data)), data["year"].to_numpy(float) - data["year"].mean(), station_dummies])
    beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ beta
    dof = max(len(y) - x.shape[1], 1)
    variance = float(residual @ residual / dof)
    covariance = variance * np.linalg.pinv(x.T @ x)
    return float(beta[1]), float(math.sqrt(max(covariance[1, 1], 0.0)))


def li_implementation_sensitivity(weather: pd.DataFrame) -> pd.DataFrame:
    """Audit plausible choices left unspecified in Li et al. (2025).

    The documented primary implementation is not selected by goodness of fit
    to the published event count.  This table exists to locate discrepancies
    caused by percentile interpolation, leap-day indexing, comparison
    semantics, or the order of event qualification and gap bridging.
    """
    hko = weather.loc[
        weather["station_code"].eq("HKO")
        & weather["date"].dt.year.between(1980, 2023)
        & weather["tmax_c"].notna(),
        ["date", "station_code", "tmax_c"],
    ].sort_values("date").copy()
    if hko.empty:
        raise RuntimeError("HKO Tmax observations for 1980--2023 are required for Li replication.")
    warm = hko.loc[hko["date"].dt.month.isin(WARM_MONTHS)].copy()
    records = []
    for sample_scope in ["full_year", "warm_months"]:
        for day_index_policy in ["noleap_calendar", "native_dayofyear"]:
            for interpolation in [
                "linear", "lower", "higher", "midpoint", "nearest",
                "numpy_inverted_cdf", "numpy_averaged_inverted_cdf", "numpy_closest_observation",
                "numpy_interpolated_inverted_cdf", "numpy_hazen", "numpy_weibull",
                "numpy_median_unbiased", "numpy_normal_unbiased",
            ]:
                full_threshold = moving_percentile_threshold(
                    hko,
                    1980,
                    2023,
                    interpolation=interpolation,
                    sample_scope=sample_scope,
                    day_index_policy=day_index_policy,
                )
                threshold = full_threshold.loc[warm.index]
                for comparison in ["strict_gt", "inclusive_ge"]:
                    hot = warm["tmax_c"].gt(threshold) if comparison == "strict_gt" else warm["tmax_c"].ge(threshold)
                    for assembly in ["qualify_then_merge", "bridge_then_qualify"]:
                        intervals = consecutive_intervals(warm["date"], hot, assembly=assembly)
                        event_rows = []
                        annual_rows = []
                        for start, end, members in intervals:
                            mask = warm["date"].between(start, end)
                            excess = (warm.loc[mask, "tmax_c"] - threshold.loc[mask]).clip(lower=0)
                            duration = int((end - start).days + 1)
                            event_rows.append(
                                {
                                    "start": start,
                                    "year": int(start.year),
                                    "duration": duration,
                                    "hot_days": len(members),
                                    "cumulative": float(excess.sum()),
                                    "mean_span_intensity": float(excess.sum() / duration),
                                    "span_dates": set(pd.date_range(start, end, freq="D")),
                                }
                            )
                        for year in range(1980, 2024):
                            year_events = [event for event in event_rows if event["year"] == year]
                            span_dates = set().union(*(event["span_dates"] for event in year_events)) if year_events else set()
                            annual_rows.append(
                                {
                                    "station_code": "HKO",
                                    "year": year,
                                    "event_count": len(year_events),
                                    "mean_duration": float(np.mean([event["duration"] for event in year_events])) if year_events else 0.0,
                                    "total_cumulative_intensity": float(sum(event["cumulative"] for event in year_events)),
                                    "mean_event_cumulative_intensity": float(
                                        np.mean([event["cumulative"] for event in year_events])
                                    ) if year_events else 0.0,
                                    "event_span_days": len(span_dates),
                                }
                            )
                        annual_frame = pd.DataFrame(annual_rows)
                        trend_metrics = [
                            "event_count",
                            "mean_duration",
                            "total_cumulative_intensity",
                            "mean_event_cumulative_intensity",
                            "event_span_days",
                        ]
                        slope_statistics = {
                            metric: fixed_effect_slope(annual_frame, metric)
                            for metric in trend_metrics
                        }
                        post_2010_inclusive = [event for event in event_rows if event["year"] >= 2010]
                        post_2010_exclusive = [event for event in event_rows if event["year"] > 2010]
                        records.append(
                            {
                                "sample_scope": sample_scope,
                                "day_index_policy": day_index_policy,
                                "quantile_interpolation": interpolation,
                                "comparison": comparison,
                                "event_assembly": assembly,
                                "documented_primary": bool(
                                    sample_scope == "full_year"
                                    and day_index_policy == "noleap_calendar"
                                    and interpolation == "linear"
                                    and comparison == "strict_gt"
                                    and assembly == "qualify_then_merge"
                                ),
                                "event_count_1980_2023": len(event_rows),
                                "minimum_duration_days": min((event["duration"] for event in event_rows), default=math.nan),
                                "maximum_duration_days": max((event["duration"] for event in event_rows), default=math.nan),
                                "post_2010_inclusive_event_count": len(post_2010_inclusive),
                                "post_2010_exclusive_event_count": len(post_2010_exclusive),
                                "post_2010_inclusive_mean_intensity_c": float(np.mean([event["mean_span_intensity"] for event in post_2010_inclusive])) if post_2010_inclusive else math.nan,
                                "event_frequency_slope_per_year": slope_statistics["event_count"][0],
                                "event_frequency_slope_standard_error": slope_statistics["event_count"][1],
                                "mean_duration_slope_days_per_year": slope_statistics["mean_duration"][0],
                                "mean_duration_slope_standard_error": slope_statistics["mean_duration"][1],
                                "annual_total_cumulative_intensity_slope_c_days_per_year": slope_statistics["total_cumulative_intensity"][0],
                                "annual_total_cumulative_intensity_slope_standard_error": slope_statistics["total_cumulative_intensity"][1],
                                "annual_mean_event_cumulative_intensity_slope_c_days_per_year": slope_statistics["mean_event_cumulative_intensity"][0],
                                "annual_mean_event_cumulative_intensity_slope_standard_error": slope_statistics["mean_event_cumulative_intensity"][1],
                                "event_span_days_slope_per_year": slope_statistics["event_span_days"][0],
                                "event_span_days_slope_standard_error": slope_statistics["event_span_days"][1],
                            }
                        )
    result = pd.DataFrame(records)
    result["published_event_summary_matched"] = (
        result["event_count_1980_2023"].eq(57)
        & result["minimum_duration_days"].eq(3)
        & result["maximum_duration_days"].eq(20)
        & result["post_2010_inclusive_event_count"].eq(41)
        & result["post_2010_inclusive_mean_intensity_c"].sub(0.90).abs().le(0.05)
    )
    result["published_trends_matched"] = (
        result["event_frequency_slope_per_year"].sub(0.08).abs().le(0.02)
        & result["mean_duration_slope_days_per_year"].sub(0.09).abs().le(0.04)
        & result["annual_mean_event_cumulative_intensity_slope_c_days_per_year"].sub(0.13).abs().le(0.04)
        & result["event_span_days_slope_per_year"].sub(0.49).abs().le(0.09)
    )
    result["published_key_metrics_matched"] = (
        result["published_event_summary_matched"]
        & result["published_trends_matched"]
    )
    return result.sort_values(
        ["published_key_metrics_matched", "documented_primary", "event_count_1980_2023"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def trend_table(annual: pd.DataFrame) -> pd.DataFrame:
    metrics = [c for c in annual.columns if c not in {"station_code", "year"}]
    rows = []
    scopes = [
        ("Li_exact_HKO_1980_2023", annual["station_code"].eq("HKO") & annual["year"].between(1980, 2023)),
        ("HKO_extended_1980_2025", annual["station_code"].eq("HKO") & annual["year"].between(1980, 2025)),
        ("KP_1993_2025", annual["station_code"].eq("KP") & annual["year"].between(1993, 2025)),
        ("multistation_2000_2025", annual["year"].between(2000, 2025)),
    ]
    for scope, mask in scopes:
        subset = annual.loc[mask]
        for metric in metrics:
            slope, se = fixed_effect_slope(subset, metric)
            rows.append({"scope": scope, "metric": metric, "slope_per_year": slope, "standard_error": se, "n_station_years": len(subset), "n_stations": subset["station_code"].nunique()})
    return pd.DataFrame(rows)


def contemporary_station_target(
    annual: pd.DataFrame,
    value: str,
    start_year: int = 2011,
    end_year: int = 2025,
) -> pd.DataFrame:
    """Create a common-period spatial target without temporal extrapolation."""
    period = annual.loc[annual["year"].between(start_year, end_year), ["station_code", "year", value]].dropna().copy()
    period_years = end_year - start_year + 1
    minimum_years = max(10, int(math.ceil(0.70 * period_years)))
    summary = period.groupby("station_code").agg(
        target=(value, "mean"),
        within_station_sd=(value, "std"),
        station_years=(value, "count"),
        first_year=("year", "min"),
        last_year=("year", "max"),
    ).reset_index()
    summary = summary.loc[summary["station_years"].ge(minimum_years)].copy()
    summary["target_period_start_year"] = start_year
    summary["target_period_end_year"] = end_year
    summary["minimum_valid_years"] = minimum_years
    summary["temporal_extrapolation_applied"] = False
    return summary


def gdal_program(name: str) -> Path:
    executable = f"{name}.exe" if os.name == "nt" else name
    candidates = [
        Path(sys.executable).resolve().parent / "Library" / "bin" / executable,
        Path(sys.executable).resolve().parent / executable,
    ]
    discovered = shutil.which(executable) or shutil.which(name)
    if discovered:
        candidates.append(Path(discovered))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"GDAL program is missing from the fixed environment: {name}")


def run_checked(command: list[str]) -> None:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(command)}\n{completed.stderr.strip()}")


def worldcover_grid_features(
    all_cells: pd.DataFrame,
    xmin: int,
    xmax: int,
    ymin: int,
    ymax: int,
    cell_m: int,
    derived_dir: Path,
) -> pd.DataFrame:
    missing = [str(path) for path in WORLD_COVER_TILES if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Both ESA WorldCover tiles are required for Hong Kong: {missing}")
    derived_dir.mkdir(parents=True, exist_ok=True)
    vrt = derived_dir / "worldcover_hk_tiles.vrt"
    local_10m = derived_dir / "worldcover_hk_10m_epsg2326.tif"
    left, right = xmin * cell_m, (xmax + 1) * cell_m
    bottom, top = ymin * cell_m, (ymax + 1) * cell_m
    run_checked([str(gdal_program("gdalbuildvrt")), "-q", "-overwrite", str(vrt), *map(str, WORLD_COVER_TILES)])
    run_checked(
        [
            str(gdal_program("gdalwarp")), "-q", "-overwrite", "-t_srs", "EPSG:2326",
            "-te", str(left), str(bottom), str(right), str(top), "-tr", "10", "10", "-tap", "-r", "near",
            "-srcnodata", "0", "-dstnodata", "0", "-co", "COMPRESS=DEFLATE", "-co", "TILED=YES",
            str(vrt), str(local_10m),
        ]
    )
    class_expressions = {
        "worldcover_tree_fraction": "A==10",
        "worldcover_vegetation_fraction": "((A==10)+(A==20)+(A==30)+(A==40)+(A==90)+(A==95)+(A==100))>0",
        "worldcover_built_fraction": "A==50",
        "worldcover_water_fraction": "A==80",
    }
    features = all_cells[["gx", "gy"]].copy()
    for feature, expression in class_expressions.items():
        mask = derived_dir / f"{feature}_10m.tif"
        fraction = derived_dir / f"{feature}_{cell_m}m.tif"
        xyz = derived_dir / f"{feature}_{cell_m}m.xyz"
        run_checked(
            [
                str(gdal_program("gdal")), "raster", "calc", "-q", "-i", f"A={local_10m}", "-o", str(mask),
                "--calc", expression, "--ot", "UInt8", "--nodata", "none", "--co", "COMPRESS=DEFLATE", "--overwrite",
            ]
        )
        run_checked(
            [
                str(gdal_program("gdalwarp")), "-q", "-overwrite", "-te", str(left), str(bottom), str(right), str(top),
                "-tr", str(cell_m), str(cell_m), "-tap", "-r", "average", "-ot", "Float32", "-dstnodata", "-9999",
                "-co", "COMPRESS=DEFLATE", str(mask), str(fraction),
            ]
        )
        run_checked([str(gdal_program("gdal_translate")), "-q", "-of", "XYZ", str(fraction), str(xyz)])
        table = pd.read_csv(xyz, sep=r"\s+", names=["x_m", "y_m", feature])
        table = table.loc[table[feature].gt(-9000)].copy()
        table["gx"] = np.floor(table["x_m"] / cell_m).astype(int)
        table["gy"] = np.floor(table["y_m"] / cell_m).astype(int)
        features = features.merge(table[["gx", "gy", feature]], on=["gx", "gy"], how="left", validate="one_to_one")

    ny, nx = ymax - ymin + 1, xmax - xmin + 1
    ix = features["gx"].to_numpy(int) - xmin
    iy = features["gy"].to_numpy(int) - ymin
    for feature in class_expressions:
        values = pd.to_numeric(features[feature], errors="coerce").fillna(0).clip(0, 1).to_numpy(float)
        array = np.zeros((ny, nx), dtype=float)
        array[iy, ix] = values
        for radius in HAZARD_CONFIG["morphology_buffer_radii_m"]:
            radius_cells = max(1, int(round(radius / cell_m)))
            size = 2 * radius_cells + 1
            smoothed = uniform_filter(array, size=size, mode="constant")
            features[f"{feature}_{radius}m"] = smoothed[iy, ix]
    water = np.zeros((ny, nx), dtype=float)
    water[iy, ix] = features["worldcover_water_fraction"].fillna(0).to_numpy(float)
    water_distance = distance_transform_edt(water <= 0.01) * cell_m
    features["distance_to_mapped_water_m"] = water_distance[iy, ix]
    return features


def road_grid_features(
    all_cells: pd.DataFrame,
    xmin: int,
    xmax: int,
    ymin: int,
    ymax: int,
    cell_m: int,
    derived_dir: Path,
) -> pd.DataFrame:
    road_files = sorted(CSDI_ROAD_DIR.glob("centerline_*_simplified5m.geojson"))
    if len(road_files) != 13:
        raise FileNotFoundError(f"Expected 13 validated CSDI road chunks, found {len(road_files)} in {CSDI_ROAD_DIR}")
    derived_dir.mkdir(parents=True, exist_ok=True)
    left, right = xmin * cell_m, (xmax + 1) * cell_m
    bottom, top = ymin * cell_m, (ymax + 1) * cell_m
    features = all_cells[["gx", "gy"]].copy()
    ny, nx = ymax - ymin + 1, xmax - xmin + 1
    ix = features["gx"].to_numpy(int) - xmin
    iy = features["gy"].to_numpy(int) - ymin

    for road_class, where in [("all", None), ("major", "ROUTE_NUM IS NOT NULL")]:
        raster_10m = derived_dir / f"road_{road_class}_10m.tif"
        raster_10m.unlink(missing_ok=True)
        for index, road_file in enumerate(road_files):
            command = [str(gdal_program("gdal_rasterize")), "-q", "-burn", "1", "-at"]
            if where:
                command.extend(["-where", where])
            if index == 0:
                command.extend([
                    "-ot", "Byte", "-a_nodata", "0", "-init", "0", "-te", str(left), str(bottom), str(right), str(top),
                    "-tr", "10", "10", "-tap", "-co", "COMPRESS=DEFLATE", "-co", "TILED=YES",
                ])
            command.extend([str(road_file), str(raster_10m)])
            run_checked(command)

        fraction = derived_dir / f"road_{road_class}_fraction_{cell_m}m.tif"
        xyz = derived_dir / f"road_{road_class}_fraction_{cell_m}m.xyz"
        run_checked([
            str(gdal_program("gdalwarp")), "-q", "-overwrite", "-te", str(left), str(bottom), str(right), str(top),
            "-tr", str(cell_m), str(cell_m), "-tap", "-r", "average", "-ot", "Float32", "-dstnodata", "-9999",
            "-co", "COMPRESS=DEFLATE", str(raster_10m), str(fraction),
        ])
        run_checked([str(gdal_program("gdal_translate")), "-q", "-of", "XYZ", str(fraction), str(xyz)])
        feature = f"road_{road_class}_pixel_fraction"
        table = pd.read_csv(xyz, sep=r"\s+", names=["x_m", "y_m", feature])
        table = table.loc[table[feature].gt(-9000)].copy()
        table["gx"] = np.floor(table["x_m"] / cell_m).astype(int)
        table["gy"] = np.floor(table["y_m"] / cell_m).astype(int)
        features = features.merge(table[["gx", "gy", feature]], on=["gx", "gy"], how="left", validate="one_to_one")
        values = pd.to_numeric(features[feature], errors="coerce").fillna(0).clip(0, 1).to_numpy(float)
        array = np.zeros((ny, nx), dtype=float)
        array[iy, ix] = values
        for radius in HAZARD_CONFIG["morphology_buffer_radii_m"]:
            radius_cells = max(1, int(round(radius / cell_m)))
            smoothed = uniform_filter(array, size=2 * radius_cells + 1, mode="constant")
            features[f"road_{road_class}_length_density_proxy_{radius}m"] = smoothed[iy, ix] / 10.0
        if road_class == "major":
            distance = distance_transform_edt(array <= 0.0) * cell_m
            features["distance_to_major_route_m"] = distance[iy, ix]
    return features


def grid_predictors(buildings: gpd.GeoDataFrame, coordinates: pd.DataFrame, cell_m: int, derived_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stock = buildings.to_crs(2326).copy()
    reps = stock.geometry.representative_point()
    stock["x"] = reps.x
    stock["y"] = reps.y
    stock["gx"] = np.floor(stock["x"] / cell_m).astype(int)
    stock["gy"] = np.floor(stock["y"] / cell_m).astype(int)
    stock["footprint"] = pd.to_numeric(stock["footprint_area_m2"], errors="coerce").clip(lower=0).fillna(0)
    height = pd.to_numeric(stock["height_value_m"], errors="coerce").clip(lower=0).fillna(0)
    stock["volume"] = stock["footprint"] * height
    stock["facade"] = pd.to_numeric(stock["perimeter_m"], errors="coerce").clip(lower=0).fillna(0) * height
    stock["elevation"] = pd.to_numeric(stock["dtm_elevation_mpd"], errors="coerce")
    use_text = stock["hkbfetd_sub_class"].fillna("").astype(str)
    use_groups = {
        "residential": r"Private Housing|Public Housing|Other Housing|HA Subsidized",
        "commercial": r"Commercial|Office|Retail|Restaurant|Accommodation|Data Centre",
        "education": r"Education",
        "health": r"Human Health",
        "industrial": r"Industrial|Manufacturing|Textile|Food & Beverage",
    }
    aggregation = {
        "footprint": ("footprint", "sum"),
        "volume": ("volume", "sum"),
        "facade": ("facade", "sum"),
        "building_count": ("BUILDINGSTRUCTUREID", "size"),
        "elevation": ("elevation", "median"),
    }
    for label, pattern in use_groups.items():
        column = f"footprint_{label}"
        stock[column] = stock["footprint"].where(use_text.str.contains(pattern, regex=True, na=False), 0.0)
        aggregation[column] = (column, "sum")
    grouped = stock.groupby(["gx", "gy"], as_index=False).agg(**aggregation)
    xmin, xmax = int(stock["gx"].min()) - 4, int(stock["gx"].max()) + 4
    ymin, ymax = int(stock["gy"].min()) - 4, int(stock["gy"].max()) + 4
    nx, ny = xmax - xmin + 1, ymax - ymin + 1
    arrays = {name: np.zeros((ny, nx), dtype=float) for name in ["footprint", "volume", "facade", "building_count", *[f"footprint_{label}" for label in use_groups]]}
    elevation = np.full((ny, nx), np.nan, dtype=float)
    ix = grouped["gx"].to_numpy(int) - xmin
    iy = grouped["gy"].to_numpy(int) - ymin
    for name in arrays:
        arrays[name][iy, ix] = grouped[name].to_numpy(float)
    elevation[iy, ix] = grouped["elevation"].to_numpy(float)
    known = np.isfinite(elevation)
    if known.any():
        nearest = distance_transform_edt(~known, return_distances=False, return_indices=True)
        elevation = elevation[tuple(nearest)]
    else:
        elevation[:] = 0.0
    feature_arrays = {"terrain_elevation_proxy_m": elevation}
    for radius in HAZARD_CONFIG["morphology_buffer_radii_m"]:
        cells = max(1, int(round(radius / cell_m)))
        size = 2 * cells + 1
        area = (size * cell_m) ** 2
        for raw_name, output_name in [("footprint", "building_coverage"), ("volume", "building_volume_density"), ("facade", "facade_area_density"), ("building_count", "building_count_density")]:
            total = uniform_filter(arrays[raw_name], size=size, mode="constant") * size * size
            feature_arrays[f"{output_name}_{radius}m"] = total / area
        for label in use_groups:
            total = uniform_filter(arrays[f"footprint_{label}"], size=size, mode="constant") * size * size
            feature_arrays[f"building_use_{label}_coverage_{radius}m"] = total / area
    yy, xx = np.indices((ny, nx))
    all_cells = pd.DataFrame({"gx": (xx.ravel() + xmin), "gy": (yy.ravel() + ymin), "x_m": (xx.ravel() + xmin + 0.5) * cell_m, "y_m": (yy.ravel() + ymin + 0.5) * cell_m})
    for name, values in feature_arrays.items():
        all_cells[name] = values.ravel()
    worldcover = worldcover_grid_features(all_cells, xmin, xmax, ymin, ymax, cell_m, derived_dir)
    all_cells = all_cells.merge(worldcover, on=["gx", "gy"], how="left", validate="one_to_one")
    roads = road_grid_features(all_cells, xmin, xmax, ymin, ymax, cell_m, derived_dir)
    all_cells = all_cells.merge(roads, on=["gx", "gy"], how="left", validate="one_to_one")
    all_cells["x_km"] = all_cells["x_m"] / 1000.0
    all_cells["y_km"] = all_cells["y_m"] / 1000.0
    transformer = Transformer.from_crs(4326, 2326, always_xy=True)
    station = coordinates.dropna(subset=["longitude", "latitude"]).copy()
    station["x_m"], station["y_m"] = transformer.transform(station["longitude"].to_numpy(), station["latitude"].to_numpy())
    station["gx"] = np.floor(station["x_m"] / cell_m).astype(int)
    station["gy"] = np.floor(station["y_m"] / cell_m).astype(int)
    station = station.merge(all_cells.drop(columns=["x_m", "y_m"]), on=["gx", "gy"], how="left")
    station["x_km"] = station["x_m"] / 1000.0
    station["y_km"] = station["y_m"] / 1000.0
    building_index = stock[["BUILDINGSTRUCTUREID", "gx", "gy"]].copy()
    # The surrounding rectangle is only an internal convolution surface.
    # Offshore and empty cells must not change building-hazard percentiles.
    cells = grouped[["gx", "gy"]].merge(all_cells, on=["gx", "gy"], how="left", validate="one_to_one")
    return cells, station, building_index


def model_factories(feature_names: list[str]) -> dict[str, object]:
    ridge = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    trees = make_pipeline(SimpleImputer(strategy="median"), ExtraTreesRegressor(n_estimators=int(HAZARD_CONFIG["extra_trees_estimators"]), min_samples_leaf=2, max_features=0.8, random_state=RANDOM_SEED, n_jobs=-1))
    stacking = StackingRegressor(
        estimators=[("ridge", ridge), ("trees", trees)],
        final_estimator=RidgeCV(alphas=np.logspace(-3, 3, 13)),
        cv=5,
        passthrough=False,
        n_jobs=None,
    )
    geo = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    return {"geographic_ridge": geo, "lur_ridge": ridge, "extra_trees": trees, "stacking": stacking}


def spatial_cv(stations: pd.DataFrame, geo_features: list[str], all_features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    xy = stations[["x_m", "y_m"]].to_numpy(float)
    y = stations["target"].to_numpy(float)
    factories = model_factories(all_features)
    for test in range(len(stations)):
        train = np.arange(len(stations)) != test
        distances = np.linalg.norm(xy[train] - xy[test], axis=1)
        nearest_index = np.flatnonzero(train)[int(np.argmin(distances))]
        safe_distance = np.maximum(distances, 100.0)
        records.extend([
            {"station_code": stations.iloc[test]["station_code"], "model": "nearest_station", "observed": y[test], "predicted": y[nearest_index]},
            {"station_code": stations.iloc[test]["station_code"], "model": "idw_p2", "observed": y[test], "predicted": float(np.sum(y[train] / safe_distance**2) / np.sum(1 / safe_distance**2))},
        ])
        for name, estimator in factories.items():
            features = geo_features if name == "geographic_ridge" else all_features
            estimator.fit(stations.loc[train, features], y[train])
            prediction = float(estimator.predict(stations.loc[[test], features])[0])
            records.append({"station_code": stations.iloc[test]["station_code"], "model": name, "observed": y[test], "predicted": prediction})
    predictions = pd.DataFrame(records)
    metrics = []
    for model, group in predictions.groupby("model"):
        rho = spearmanr(group["observed"], group["predicted"]).statistic
        metrics.append({"model": model, "n_stations": len(group), "mae": mean_absolute_error(group["observed"], group["predicted"]), "rmse": math.sqrt(mean_squared_error(group["observed"], group["predicted"])), "spearman": float(rho)})
    return predictions, pd.DataFrame(metrics).sort_values("mae")


def paired_error_superiority(predictions: pd.DataFrame, candidate: str, baseline: str, seed: int = RANDOM_SEED, resamples: int = int(HAZARD_CONFIG["paired_bootstrap_resamples"])) -> dict[str, float | str | bool]:
    table = predictions.loc[predictions["model"].isin([candidate, baseline])].pivot(index="station_code", columns="model", values=["observed", "predicted"])
    observed = table[("observed", candidate)].to_numpy(float)
    candidate_error = table[("predicted", candidate)].to_numpy(float) - observed
    baseline_error = table[("predicted", baseline)].to_numpy(float) - observed
    rng = np.random.default_rng(seed + sum(map(ord, candidate + baseline)))
    samples = rng.integers(0, len(table), size=(resamples, len(table)))
    mae_difference = np.mean(np.abs(candidate_error[samples]), axis=1) - np.mean(np.abs(baseline_error[samples]), axis=1)
    candidate_rmse = np.sqrt(np.mean(candidate_error[samples] ** 2, axis=1))
    baseline_rmse = np.sqrt(np.mean(baseline_error[samples] ** 2, axis=1))
    rmse_difference = candidate_rmse - baseline_rmse
    candidate_prediction = table[("predicted", candidate)].to_numpy(float)
    baseline_prediction = table[("predicted", baseline)].to_numpy(float)
    rank_differences = []
    for sample in samples:
        candidate_rho = spearmanr(observed[sample], candidate_prediction[sample]).statistic
        baseline_rho = spearmanr(observed[sample], baseline_prediction[sample]).statistic
        if math.isfinite(candidate_rho) and math.isfinite(baseline_rho):
            rank_differences.append(candidate_rho - baseline_rho)
    observed_candidate_rho = float(spearmanr(observed, candidate_prediction).statistic)
    observed_baseline_rho = float(spearmanr(observed, baseline_prediction).statistic)
    rank_lower95 = float(np.quantile(rank_differences, 0.05)) if rank_differences else math.nan
    return {
        "candidate": candidate,
        "baseline": baseline,
        "n_stations": len(table),
        "mean_mae_difference_days": float(np.mean(np.abs(candidate_error)) - np.mean(np.abs(baseline_error))),
        "mae_difference_bootstrap_upper95_days": float(np.quantile(mae_difference, 0.95)),
        "rmse_difference_days": float(np.sqrt(np.mean(candidate_error**2)) - np.sqrt(np.mean(baseline_error**2))),
        "rmse_difference_bootstrap_upper95_days": float(np.quantile(rmse_difference, 0.95)),
        "candidate_spearman": observed_candidate_rho,
        "baseline_spearman": observed_baseline_rho,
        "spearman_difference": observed_candidate_rho - observed_baseline_rho,
        "spearman_difference_bootstrap_lower95": rank_lower95,
        "superior_at_one_sided_95pct": bool(
            np.quantile(mae_difference, 0.95) < 0
            and np.quantile(rmse_difference, 0.95) < 0
            and observed_candidate_rho > 0
            and math.isfinite(rank_lower95)
            and rank_lower95 > 0
        ),
    }


def select_model(metrics: pd.DataFrame, predictions: pd.DataFrame) -> tuple[str, str, bool, pd.DataFrame]:
    indexed = metrics.set_index("model")
    tests = []
    for candidate in ["idw_p2", "geographic_ridge", "lur_ridge", "extra_trees", "stacking"]:
        if candidate not in indexed.index:
            continue
        for baseline in ["nearest_station", "geographic_ridge"]:
            if candidate != baseline and baseline in indexed.index:
                tests.append(paired_error_superiority(predictions, candidate, baseline))
    test_frame = pd.DataFrame(tests)

    morphology_passed = []
    for candidate in ["lur_ridge", "extra_trees", "stacking"]:
        comparisons = test_frame.loc[(test_frame["candidate"] == candidate) & test_frame["baseline"].isin(["nearest_station", "geographic_ridge"])]
        if len(comparisons) == 2 and comparisons["superior_at_one_sided_95pct"].all():
            morphology_passed.append(candidate)
    if morphology_passed:
        selected = str(indexed.loc[morphology_passed].sort_values("mae").index[0])
        return selected, "validated_morphology_surface", True, test_frame

    interpolation_passed = []
    for candidate in ["idw_p2", "geographic_ridge"]:
        comparison = test_frame.loc[(test_frame["candidate"] == candidate) & (test_frame["baseline"] == "nearest_station")]
        if len(comparison) == 1 and bool(comparison.iloc[0]["superior_at_one_sided_95pct"]):
            interpolation_passed.append(candidate)
    if interpolation_passed:
        selected = str(indexed.loc[interpolation_passed].sort_values("mae").index[0])
        return selected, "validated_interpolation_not_microclimate", False, test_frame
    return "nearest_station", "fine_surface_withheld", False, test_frame


def idw_predict(train_xy: np.ndarray, train_y: np.ndarray, query_xy: np.ndarray, chunk: int = 5000) -> np.ndarray:
    output = np.empty(len(query_xy), dtype=float)
    for start in range(0, len(query_xy), chunk):
        query = query_xy[start : start + chunk]
        distance = np.linalg.norm(query[:, None, :] - train_xy[None, :, :], axis=2)
        weight = 1 / np.maximum(distance, 100.0) ** 2
        output[start : start + chunk] = (weight @ train_y) / weight.sum(axis=1)
    return output


def nearest_predict(train_xy: np.ndarray, train_y: np.ndarray, query_xy: np.ndarray, chunk: int = 5000) -> np.ndarray:
    output = np.empty(len(query_xy), dtype=float)
    for start in range(0, len(query_xy), chunk):
        query = query_xy[start : start + chunk]
        distance = np.linalg.norm(query[:, None, :] - train_xy[None, :, :], axis=2)
        output[start : start + chunk] = train_y[np.argmin(distance, axis=1)]
    return output


def wrf_projection_cases() -> list[tuple[str, str]]:
    return [
        ("ssp245", "1120"),
        ("ssp126", "4049"),
        ("ssp126", "9099"),
        ("ssp245", "4049"),
        ("ssp245", "9099"),
        ("ssp585", "4049"),
        ("ssp585", "9099"),
    ]


def wrf_temperature_path(scenario: str, period_code: str, statistic: str) -> Path:
    folder = "daily_max" if statistic == "dailymax" else "daily_min"
    return (
        WRF_HEATWAVE_ROOT
        / scenario.upper()
        / folder
        / f"{scenario}_t2_{period_code}_era5_corrected_{statistic}.nc"
    )


def grid_run_metrics(flags: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    events = np.zeros(flags.shape[1], dtype=float)
    event_days = np.zeros(flags.shape[1], dtype=float)
    maximum_duration = np.zeros(flags.shape[1], dtype=float)
    for column in range(flags.shape[1]):
        padded = np.r_[False, flags[:, column], False].astype(np.int8)
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        lengths = ends - starts
        qualifying = lengths[lengths >= 3]
        if len(qualifying):
            events[column] = len(qualifying)
            event_days[column] = qualifying.sum()
            maximum_duration[column] = qualifying.max()
    return events, event_days, maximum_duration


def wrf_grid_climatology(values: np.ndarray, dates: pd.DatetimeIndex, threshold_c: float) -> dict[str, np.ndarray]:
    annual_threshold_days = []
    annual_events = []
    annual_event_days = []
    annual_maximum_duration = []
    for year in sorted(np.unique(dates.year)):
        year_values = values[dates.year == year]
        flags = year_values >= threshold_c
        events, event_days, maximum_duration = grid_run_metrics(flags)
        annual_threshold_days.append(flags.sum(axis=0))
        annual_events.append(events)
        annual_event_days.append(event_days)
        annual_maximum_duration.append(maximum_duration)
    return {
        "mean_c": np.mean(values, axis=0),
        "p95_c": np.quantile(values, 0.95, axis=0),
        "threshold_days_per_summer": np.mean(annual_threshold_days, axis=0),
        "events_per_summer": np.mean(annual_events, axis=0),
        "event_days_per_summer": np.mean(annual_event_days, axis=0),
        "maximum_duration_days": np.mean(annual_maximum_duration, axis=0),
    }


def wrf_hko_daily_diagnostic(
    weather: pd.DataFrame,
    coordinates: pd.DataFrame,
    latitude: np.ndarray,
    longitude: np.ndarray,
    land_mask: np.ndarray,
    dates: pd.DatetimeIndex,
    tmax: np.ndarray,
    tmin: np.ndarray,
    output: Path,
) -> pd.DataFrame:
    stations = coordinates.drop_duplicates("station_code").reset_index(drop=True)
    latitude0 = float(stations["latitude"].mean())
    grid_xy = np.c_[
        longitude.ravel() * np.cos(np.deg2rad(latitude0)) * 111.32,
        latitude.ravel() * 110.57,
    ]
    valid_flat = np.flatnonzero(land_mask.ravel())
    station_xy = np.c_[
        stations["longitude"].to_numpy(float) * np.cos(np.deg2rad(latitude0)) * 111.32,
        stations["latitude"].to_numpy(float) * 110.57,
    ]
    distance, position = cKDTree(grid_xy[valid_flat]).query(station_xy)
    flat_index = valid_flat[position]
    row_index, column_index = np.unravel_index(flat_index, latitude.shape)
    frames = []
    for index, station in stations.iterrows():
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "station_code": station["station_code"],
                    "wrf_land_grid_distance_km": distance[index],
                    "wrf_tmax_c": tmax[:, row_index[index], column_index[index]],
                    "wrf_tmin_c": tmin[:, row_index[index], column_index[index]],
                }
            )
        )
    paired = pd.concat(frames, ignore_index=True).merge(
        weather[["date", "station_code", "tmax_c", "tmin_c"]],
        on=["date", "station_code"],
        how="left",
        validate="one_to_one",
    )
    paired.to_parquet(output / "wrf_hko_daily_paired_diagnostic.parquet", index=False)
    splits = [
        ("all_2011_2020", pd.Timestamp("2011-01-01"), pd.Timestamp("2020-12-31")),
        ("pre_correction_temporal_holdout_2011_2015", pd.Timestamp("2011-01-01"), pd.Timestamp("2015-12-31")),
        ("source_overlap_2016_2020", pd.Timestamp("2016-01-01"), pd.Timestamp("2020-12-31")),
    ]
    rows = []
    for evaluation_period, start, end in splits:
        period = paired.loc[paired["date"].between(start, end)]
        station_groups = [("ALL", period)]
        if evaluation_period == "all_2011_2020":
            station_groups.extend((str(code), group) for code, group in period.groupby("station_code"))
        for station_code, group in station_groups:
            for observed_column, predicted_column, threshold in [
                ("tmax_c", "wrf_tmax_c", float(HAZARD_CONFIG["hko_very_hot_day_c"])),
                ("tmin_c", "wrf_tmin_c", float(HAZARD_CONFIG["hko_hot_night_c"])),
            ]:
                valid = group.dropna(subset=[observed_column, predicted_column])
                if len(valid) == 0:
                    continue
                observed = valid[observed_column].to_numpy(float)
                predicted = valid[predicted_column].to_numpy(float)
                error = predicted - observed
                observed_hot = observed >= threshold
                predicted_hot = predicted >= threshold
                true_positive = int(np.sum(observed_hot & predicted_hot))
                false_positive = int(np.sum(~observed_hot & predicted_hot))
                false_negative = int(np.sum(observed_hot & ~predicted_hot))
                precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else math.nan
                recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else math.nan
                rho = spearmanr(observed, predicted).statistic
                rows.append(
                    {
                        "evaluation_period": evaluation_period,
                        "station_code": station_code,
                        "variable": observed_column.removesuffix("_c"),
                        "pairs": len(valid),
                        "stations": valid["station_code"].nunique(),
                        "mae_c": float(np.mean(np.abs(error))),
                        "rmse_c": float(np.sqrt(np.mean(error**2))),
                        "bias_c": float(np.mean(error)),
                        "pearson_r": float(np.corrcoef(observed, predicted)[0, 1]),
                        "spearman_r": float(rho),
                        "threshold_c": threshold,
                        "observed_threshold_days": int(observed_hot.sum()),
                        "predicted_threshold_days": int(predicted_hot.sum()),
                        "threshold_precision": precision,
                        "threshold_recall": recall,
                        "threshold_f1": 2 * precision * recall / (precision + recall) if precision + recall else math.nan,
                        "maximum_land_grid_distance_km": float(valid["wrf_land_grid_distance_km"].max()),
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(output / "wrf_hko_daily_diagnostic_metrics.csv", index=False, encoding="utf-8-sig")
    return result


WRF_BIAS_QUANTILES = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])


def idw_matrix_predict(
    train_xy: np.ndarray,
    train_values: np.ndarray,
    query_xy: np.ndarray,
    power: float,
    neighbours: int,
) -> np.ndarray:
    """Predict several bias-curve ordinates with one spatial weighting."""
    distance = np.linalg.norm(query_xy[:, None, :] - train_xy[None, :, :], axis=2)
    k = min(int(neighbours), len(train_xy))
    nearest = np.argpartition(distance, k - 1, axis=1)[:, :k]
    selected_distance = np.take_along_axis(distance, nearest, axis=1)
    selected_values = train_values[nearest]
    exact = selected_distance < 1e-9
    weights = 1.0 / np.maximum(selected_distance, 1e-6) ** float(power)
    weights = weights / weights.sum(axis=1, keepdims=True)
    predicted = np.einsum("qk,qkf->qf", weights, selected_values)
    for row in np.flatnonzero(exact.any(axis=1)):
        predicted[row] = selected_values[row, np.flatnonzero(exact[row])[0]]
    return predicted


def station_distributional_bias(
    paired: pd.DataFrame,
    observed_column: str,
    modeled_column: str,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    period = paired.loc[paired["date"].dt.year.between(start_year, end_year)].dropna(
        subset=[observed_column, modeled_column]
    ).copy()
    period["month"] = period["date"].dt.month
    rows = []
    for (station_code, month), group in period.groupby(["station_code", "month"]):
        if len(group) < 100:
            continue
        observed = np.quantile(group[observed_column].to_numpy(float), WRF_BIAS_QUANTILES)
        modeled = np.quantile(group[modeled_column].to_numpy(float), WRF_BIAS_QUANTILES)
        for quantile, observed_value, modeled_value in zip(
            WRF_BIAS_QUANTILES, observed, modeled
        ):
            rows.append(
                {
                    "station_code": station_code,
                    "month": int(month),
                    "quantile": float(quantile),
                    "observed_quantile_c": float(observed_value),
                    "modeled_quantile_c": float(modeled_value),
                    "bias_c": float(observed_value - modeled_value),
                }
            )
    return pd.DataFrame(rows)


def select_distributional_idw(
    calibration_bias: pd.DataFrame,
    coordinates: pd.DataFrame,
) -> tuple[float, int, pd.DataFrame]:
    coordinate_lookup = coordinates.set_index("station_code")[["longitude", "latitude"]]
    latitude0 = float(coordinates["latitude"].mean())
    rows = []
    for power in [0.5, 1.0, 1.5, 2.0, 3.0]:
        for neighbours in [4, 6, 8, 12, 20]:
            errors = []
            for (_, _), group in calibration_bias.groupby(["month", "quantile"]):
                available = group.loc[group["station_code"].isin(coordinate_lookup.index)].copy()
                if len(available) < 8:
                    continue
                xy = coordinate_lookup.loc[available["station_code"]].to_numpy(float)
                xy[:, 0] *= np.cos(np.deg2rad(latitude0)) * 111.32
                xy[:, 1] *= 110.57
                values = available["bias_c"].to_numpy(float)[:, None]
                for test in range(len(available)):
                    train = np.arange(len(available)) != test
                    predicted = idw_matrix_predict(
                        xy[train], values[train], xy[[test]], power, neighbours
                    )[0, 0]
                    errors.append(predicted - values[test, 0])
            errors = np.asarray(errors, dtype=float)
            rows.append(
                {
                    "power": power,
                    "neighbours": neighbours,
                    "bias_curve_points": int(len(errors)),
                    "mae_c": float(np.mean(np.abs(errors))),
                    "rmse_c": float(np.sqrt(np.mean(errors**2))),
                }
            )
    audit = pd.DataFrame(rows).sort_values(
        ["mae_c", "rmse_c", "power", "neighbours"], kind="mergesort"
    )
    best = audit.iloc[0]
    audit["selected"] = (
        audit["power"].eq(best["power"])
        & audit["neighbours"].eq(best["neighbours"])
    )
    return float(best["power"]), int(best["neighbours"]), audit


def predict_bias_curves(
    bias: pd.DataFrame,
    coordinates: pd.DataFrame,
    query_longitude: np.ndarray,
    query_latitude: np.ndarray,
    power: float,
    neighbours: int,
) -> dict[int, np.ndarray]:
    coordinate_lookup = coordinates.set_index("station_code")[["longitude", "latitude"]]
    latitude0 = float(coordinates["latitude"].mean())
    query_xy = np.c_[
        query_longitude * np.cos(np.deg2rad(latitude0)) * 111.32,
        query_latitude * 110.57,
    ]
    result = {}
    for month, group in bias.groupby("month"):
        pivot = group.pivot(index="station_code", columns="quantile", values="bias_c")
        pivot = pivot.reindex(columns=WRF_BIAS_QUANTILES).dropna()
        pivot = pivot.loc[pivot.index.intersection(coordinate_lookup.index)]
        train_xy = coordinate_lookup.loc[pivot.index].to_numpy(float)
        train_xy[:, 0] *= np.cos(np.deg2rad(latitude0)) * 111.32
        train_xy[:, 1] *= 110.57
        result[int(month)] = idw_matrix_predict(
            train_xy, pivot.to_numpy(float), query_xy, power, neighbours
        )
    return result


def apply_distributional_bias(
    values: np.ndarray,
    dates: pd.DatetimeIndex,
    bias_curves: dict[int, np.ndarray],
) -> np.ndarray:
    corrected = np.asarray(values, dtype=float).copy()
    for month, curves in bias_curves.items():
        mask = dates.month == month
        block = corrected[mask]
        if block.shape[1] != curves.shape[0]:
            raise ValueError("WRF bias curve and grid dimensions differ")
        for column in range(block.shape[1]):
            quantiles = empirical_quantile(block[:, column])
            block[:, column] += np.interp(
                quantiles, WRF_BIAS_QUANTILES, curves[column]
            )
        corrected[mask] = block
    return corrected


def validate_and_apply_wrf_spatial_bias(
    weather: pd.DataFrame,
    coordinates: pd.DataFrame,
    dates: pd.DatetimeIndex,
    baseline_tmax: np.ndarray,
    baseline_tmin: np.ndarray,
    grid: pd.DataFrame,
    building_mapping: pd.DataFrame,
    output: Path,
) -> dict:
    """Calibrate distributions, not non-synchronous GCM and observed days."""
    paired = pd.read_parquet(output / "wrf_hko_daily_paired_diagnostic.parquet")
    paired["date"] = pd.to_datetime(paired["date"])
    coordinates = coordinates.drop_duplicates("station_code").dropna(
        subset=["longitude", "latitude"]
    )
    validation_rows = []
    corrected_arrays = {}
    selected_models = {}
    for variable, observed_column, modeled_column, values, threshold in [
        ("tmax", "tmax_c", "wrf_tmax_c", baseline_tmax, float(HAZARD_CONFIG["hko_very_hot_day_c"])),
        ("tmin", "tmin_c", "wrf_tmin_c", baseline_tmin, float(HAZARD_CONFIG["hko_hot_night_c"])),
    ]:
        calibration_bias = station_distributional_bias(
            paired, observed_column, modeled_column, 2011, 2015
        )
        power, neighbours, selection = select_distributional_idw(
            calibration_bias, coordinates
        )
        selection.insert(0, "variable", variable)
        selection.to_csv(
            output / f"wrf_{variable}_distributional_idw_selection.csv",
            index=False,
            encoding="utf-8-sig",
        )
        selected_models[variable] = {"power": power, "neighbours": neighbours}

        station_codes = sorted(set(calibration_bias["station_code"]).intersection(coordinates["station_code"]))
        holdout_rows = []
        for held_station in station_codes:
            train_bias = calibration_bias.loc[
                ~calibration_bias["station_code"].eq(held_station)
            ]
            held_coordinate = coordinates.loc[
                coordinates["station_code"].eq(held_station)
            ].iloc[0]
            curves = predict_bias_curves(
                train_bias,
                coordinates,
                np.array([held_coordinate["longitude"]], dtype=float),
                np.array([held_coordinate["latitude"]], dtype=float),
                power,
                neighbours,
            )
            evaluation = paired.loc[
                paired["station_code"].eq(held_station)
                & paired["date"].dt.year.between(2016, 2020)
            ].dropna(subset=[observed_column, modeled_column]).copy()
            for month, group in evaluation.groupby(evaluation["date"].dt.month):
                if month not in curves or len(group) < 100:
                    continue
                raw = group[modeled_column].to_numpy(float)
                observed = group[observed_column].to_numpy(float)
                raw_quantile = empirical_quantile(raw)
                corrected = raw + np.interp(
                    raw_quantile, WRF_BIAS_QUANTILES, curves[int(month)][0]
                )
                observed_q = np.quantile(observed, WRF_BIAS_QUANTILES)
                raw_q = np.quantile(raw, WRF_BIAS_QUANTILES)
                corrected_q = np.quantile(corrected, WRF_BIAS_QUANTILES)
                holdout_rows.append(
                    {
                        "station_code": held_station,
                        "month": int(month),
                        "pairs": int(len(group)),
                        "raw_quantile_mae_c": float(np.mean(np.abs(raw_q - observed_q))),
                        "corrected_quantile_mae_c": float(np.mean(np.abs(corrected_q - observed_q))),
                        "raw_p95_error_c": float(raw_q[-1] - observed_q[-1]),
                        "corrected_p95_error_c": float(corrected_q[-1] - observed_q[-1]),
                        "observed_threshold_fraction": float(np.mean(observed >= threshold)),
                        "raw_threshold_fraction": float(np.mean(raw >= threshold)),
                        "corrected_threshold_fraction": float(np.mean(corrected >= threshold)),
                    }
                )
        holdout = pd.DataFrame(holdout_rows)
        holdout.to_csv(
            output / f"wrf_{variable}_distributional_spatiotemporal_holdout.csv",
            index=False,
            encoding="utf-8-sig",
        )
        raw_quantile_mae = float(holdout["raw_quantile_mae_c"].mean())
        corrected_quantile_mae = float(holdout["corrected_quantile_mae_c"].mean())
        raw_p95_mae = float(holdout["raw_p95_error_c"].abs().mean())
        corrected_p95_mae = float(holdout["corrected_p95_error_c"].abs().mean())
        raw_threshold_mae = float(
            (holdout["raw_threshold_fraction"] - holdout["observed_threshold_fraction"]).abs().mean()
        )
        corrected_threshold_mae = float(
            (holdout["corrected_threshold_fraction"] - holdout["observed_threshold_fraction"]).abs().mean()
        )
        passed = bool(
            corrected_quantile_mae < raw_quantile_mae
            and corrected_p95_mae < raw_p95_mae
            and corrected_threshold_mae < raw_threshold_mae
        )
        validation_rows.append(
            {
                "variable": variable,
                "calibration_period": "2011-2015",
                "held_out_period": "2016-2020",
                "held_out_stations": int(holdout["station_code"].nunique()),
                "idw_power": power,
                "idw_neighbours": neighbours,
                "raw_quantile_mae_c": raw_quantile_mae,
                "corrected_quantile_mae_c": corrected_quantile_mae,
                "raw_p95_mae_c": raw_p95_mae,
                "corrected_p95_mae_c": corrected_p95_mae,
                "raw_threshold_fraction_mae": raw_threshold_mae,
                "corrected_threshold_fraction_mae": corrected_threshold_mae,
                "all_three_distribution_metrics_improved": passed,
            }
        )

        full_bias = station_distributional_bias(
            paired, observed_column, modeled_column, 2011, 2020
        )
        grid_curves = predict_bias_curves(
            full_bias,
            coordinates,
            grid["wrf_grid_longitude"].to_numpy(float),
            grid["wrf_grid_latitude"].to_numpy(float),
            power,
            neighbours,
        )
        corrected_arrays[variable] = apply_distributional_bias(values, dates, grid_curves)

    validation = pd.DataFrame(validation_rows)
    validation.to_csv(
        output / "wrf_distributional_spatial_bias_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    canonical = bool(validation["all_three_distribution_metrics_improved"].all())
    corrected_day = wrf_grid_climatology(
        corrected_arrays["tmax"], dates, float(HAZARD_CONFIG["hko_very_hot_day_c"])
    )
    corrected_night = wrf_grid_climatology(
        corrected_arrays["tmin"], dates, float(HAZARD_CONFIG["hko_hot_night_c"])
    )
    corrected_grid = grid.copy()
    for prefix, metrics in [("tmax", corrected_day), ("tmin", corrected_night)]:
        for metric, metric_values in metrics.items():
            corrected_grid[f"bias_corrected_{prefix}_{metric}"] = metric_values
    corrected_grid["hazard_spatial_resolution_m"] = 1000
    corrected_grid["hazard_spatial_method"] = "WRF_1km_distributional_QM_with_HKO_spatiotemporal_LOSO_IDW"
    corrected_grid["hazard_validation_status"] = np.where(
        canonical, "validated_distributional_spatial_surface", "sensitivity_only"
    )
    corrected_grid.to_parquet(
        output / "wrf_hko_bias_corrected_historical_grid_metrics.parquet", index=False
    )
    building_hazard = building_mapping.merge(
        corrected_grid.drop(
            columns=[
                "building_count",
                "mean_building_distance_km",
                "maximum_building_distance_km",
                "buildings_within_2km",
            ]
        ),
        on=["wrf_south_north", "wrf_west_east", "wrf_grid_latitude", "wrf_grid_longitude"],
        how="left",
        validate="many_to_one",
    )
    building_hazard.to_parquet(
        output / "building_heat_hazard_wrf1km.parquet", index=False
    )
    result = {
        "method": "monthly quantile-bias curves; IDW hyperparameters selected in 2011-2015 station LOSO; independent 2016-2020 station-and-time holdout",
        "daily_pairing_used_as_validation": False,
        "validation_target": "warm-season temperature distributions, p95 and threshold frequency",
        "selected_models": selected_models,
        "canonical_high_resolution_hazard": canonical,
        "grid_resolution_m": 1000,
        "grid_cells_used_by_buildings": int(len(corrected_grid)),
        "buildings_mapped": int(len(building_hazard)),
        "validation": validation.to_dict(orient="records"),
    }
    (output / "wrf_distributional_spatial_bias_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def empirical_quantile(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = (np.arange(len(values)) + 0.5) / len(values)
    return ranks


def quantile_map(values: np.ndarray, model_reference: np.ndarray, observed_reference: np.ndarray) -> np.ndarray:
    model_reference = np.sort(np.asarray(model_reference, dtype=float))
    observed_reference = np.sort(np.asarray(observed_reference, dtype=float))
    quantiles = np.searchsorted(model_reference, values, side="right") / len(model_reference)
    quantiles = np.clip(quantiles, 1e-4, 1 - 1e-4)
    observed_probability = (np.arange(len(observed_reference)) + 0.5) / len(observed_reference)
    return np.interp(quantiles, observed_probability, observed_reference)


def quantile_delta_map(future: np.ndarray, model_historical: np.ndarray, observed_historical: np.ndarray) -> np.ndarray:
    quantiles = empirical_quantile(np.asarray(future, dtype=float))
    model_historical = np.sort(np.asarray(model_historical, dtype=float))
    observed_historical = np.sort(np.asarray(observed_historical, dtype=float))
    model_probability = (np.arange(len(model_historical)) + 0.5) / len(model_historical)
    observed_probability = (np.arange(len(observed_historical)) + 0.5) / len(observed_historical)
    model_at_quantile = np.interp(quantiles, model_probability, model_historical)
    observed_at_quantile = np.interp(quantiles, observed_probability, observed_historical)
    return observed_at_quantile + (future - model_at_quantile)


def one_dimensional_heatwave_metrics(values: np.ndarray, threshold_c: float) -> tuple[int, int, int]:
    padded = np.r_[False, np.asarray(values) >= threshold_c, False].astype(np.int8)
    changes = np.diff(padded)
    lengths = np.flatnonzero(changes == -1) - np.flatnonzero(changes == 1)
    qualifying = lengths[lengths >= 3]
    if len(qualifying) == 0:
        return 0, 0, 0
    return int(len(qualifying)), int(qualifying.sum()), int(qualifying.max())


def weighted_warm_season_mean(frame: pd.DataFrame, variable: str) -> float:
    warm = frame.loc[frame["month"].between(6, 9)]
    weights = warm["days_in_month"].to_numpy(float)
    return float(np.average(warm[variable].to_numpy(float), weights=weights))


def analyse_cmip6_multimodel_tmax(
    weather: pd.DataFrame,
    wrf_projections: pd.DataFrame,
    output: Path,
) -> dict:
    models = [str(value) for value in HAZARD_CONFIG["future_projection"]["multimodel_point_models"]]
    scenarios = [str(value) for value in HAZARD_CONFIG["future_projection"]["multimodel_scenarios"]]
    parts = []
    for model in models:
        for experiment in ["historical", *scenarios]:
            path = CMIP6_POINT_ROOT / f"{model}_{experiment}_r1i1p1f1_tasmax.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing CMIP6 point cache: {path}")
            parts.append(pd.read_csv(path))
    raw = pd.concat(parts, ignore_index=True)
    observed = weather.loc[
        weather["station_code"].eq("KP") & weather["date"].dt.year.between(1992, 2014),
        ["date", "tmax_c"],
    ].dropna().copy()
    observed["year"] = observed["date"].dt.year
    observed["month"] = observed["date"].dt.month

    validation_rows = []
    historical = raw.loc[raw["experiment_id"].eq("historical")]
    for model, frame in historical.groupby("source_id"):
        calibration_model = frame.loc[frame["year"].le(2005)]
        validation_model = frame.loc[frame["year"].ge(2006)].copy()
        calibration_observed = observed.loc[observed["year"].le(2005)]
        validation_observed = observed.loc[observed["year"].ge(2006)]
        for month in range(1, 13):
            model_month = validation_model.loc[validation_model["month"].eq(month), "tasmax_raw_c"].to_numpy(float)
            mapped = quantile_map(
                model_month,
                calibration_model.loc[calibration_model["month"].eq(month), "tasmax_raw_c"].to_numpy(float),
                calibration_observed.loc[calibration_observed["month"].eq(month), "tmax_c"].to_numpy(float),
            )
            observed_month = validation_observed.loc[validation_observed["month"].eq(month), "tmax_c"].to_numpy(float)
            validation_rows.append(
                {
                    "source_id": model,
                    "month": month,
                    "model_corrected_median_c": float(np.median(mapped)),
                    "observed_median_c": float(np.median(observed_month)),
                    "median_bias_c": float(np.median(mapped) - np.median(observed_month)),
                    "model_corrected_p95_c": float(np.quantile(mapped, 0.95)),
                    "observed_p95_c": float(np.quantile(observed_month, 0.95)),
                    "p95_bias_c": float(np.quantile(mapped, 0.95) - np.quantile(observed_month, 0.95)),
                }
            )
    historical_validation = pd.DataFrame(validation_rows)
    historical_validation.to_csv(
        output / "cmip6_tmax_historical_holdout_validation.csv", index=False, encoding="utf-8-sig"
    )

    periods = {
        "mid_century_2041_2060": (2041, 2060),
        "late_century_2081_2100": (2081, 2100),
    }
    corrected_parts = []
    for (model, scenario), frame in raw.loc[raw["experiment_id"].isin(scenarios)].groupby(
        ["source_id", "experiment_id"]
    ):
        model_historical = historical.loc[historical["source_id"].eq(model)]
        for period, (start_year, end_year) in periods.items():
            period_frame = frame.loc[frame["year"].between(start_year, end_year)].copy()
            for month, month_frame in period_frame.groupby("month"):
                month_frame = month_frame.copy()
                month_frame["tasmax_qdm_c"] = quantile_delta_map(
                    month_frame["tasmax_raw_c"].to_numpy(float),
                    model_historical.loc[model_historical["month"].eq(month), "tasmax_raw_c"].to_numpy(float),
                    observed.loc[observed["month"].eq(month), "tmax_c"].to_numpy(float),
                )
                month_frame["future_period"] = period
                corrected_parts.append(month_frame)
    corrected = pd.concat(corrected_parts, ignore_index=True).sort_values(
        ["source_id", "experiment_id", "future_period", "year", "sequence_index"]
    )
    corrected.to_parquet(output / "cmip6_tmax_future_daily_qdm.parquet", index=False)

    observed_warm = observed.loc[observed["month"].between(6, 9), "tmax_c"].to_numpy(float)
    observed_mean = float(np.mean(observed_warm))
    observed_p95 = float(np.quantile(observed_warm, 0.95))
    model_rows = []
    for (model, scenario, period), frame in corrected.groupby(
        ["source_id", "experiment_id", "future_period"]
    ):
        warm = frame.loc[frame["month"].between(6, 9)].copy()
        annual = []
        for year, year_frame in warm.groupby("year"):
            events, event_days, maximum_duration = one_dimensional_heatwave_metrics(
                year_frame.sort_values("sequence_index")["tasmax_qdm_c"].to_numpy(float),
                float(HAZARD_CONFIG["hko_very_hot_day_c"]),
            )
            annual.append(
                {
                    "threshold_days": int((year_frame["tasmax_qdm_c"] >= float(HAZARD_CONFIG["hko_very_hot_day_c"])).sum()),
                    "events": events,
                    "event_days": event_days,
                    "maximum_duration": maximum_duration,
                }
            )
        annual_frame = pd.DataFrame(annual)
        mean_tmax = float(warm["tasmax_qdm_c"].mean())
        p95_tmax = float(warm["tasmax_qdm_c"].quantile(0.95))
        model_rows.append(
            {
                "source_id": model,
                "experiment_id": scenario,
                "future_period": period,
                "years": int(warm["year"].nunique()),
                "mean_tmax_c": mean_tmax,
                "p95_tmax_c": p95_tmax,
                "delta_mean_tmax_c": mean_tmax - observed_mean,
                "delta_p95_tmax_c": p95_tmax - observed_p95,
                "very_hot_days_per_summer": float(annual_frame["threshold_days"].mean()),
                "heatwave_events_per_summer": float(annual_frame["events"].mean()),
                "heatwave_event_days_per_summer": float(annual_frame["event_days"].mean()),
                "maximum_heatwave_duration_days": float(annual_frame["maximum_duration"].mean()),
            }
        )
    model_summary = pd.DataFrame(model_rows)
    model_summary.to_csv(output / "cmip6_tmax_future_model_summary.csv", index=False, encoding="utf-8-sig")
    metric_columns = [
        "mean_tmax_c", "p95_tmax_c", "delta_mean_tmax_c", "delta_p95_tmax_c",
        "very_hot_days_per_summer", "heatwave_events_per_summer",
        "heatwave_event_days_per_summer", "maximum_heatwave_duration_days",
    ]
    ensemble_rows = []
    for (scenario, period), frame in model_summary.groupby(["experiment_id", "future_period"]):
        for metric in metric_columns:
            values = frame[metric]
            ensemble_rows.append(
                {
                    "experiment_id": scenario,
                    "future_period": period,
                    "metric": metric,
                    "models": int(len(values)),
                    "ensemble_p10": float(values.quantile(0.10)),
                    "ensemble_p50": float(values.quantile(0.50)),
                    "ensemble_p90": float(values.quantile(0.90)),
                    "ensemble_minimum": float(values.min()),
                    "ensemble_maximum": float(values.max()),
                }
            )
    pd.DataFrame(ensemble_rows).to_csv(
        output / "cmip6_tmax_future_ensemble_summary.csv", index=False, encoding="utf-8-sig"
    )

    blended_parts = []
    period_to_wrf = {
        "mid_century_2041_2060": "4049",
        "late_century_2081_2100": "9099",
    }
    for row in model_summary.itertuples(index=False):
        wrf = wrf_projections.loc[
            wrf_projections["scenario"].eq(row.experiment_id)
            & wrf_projections["period_code"].eq(period_to_wrf[row.future_period])
        ].copy()
        weights = wrf["building_count"].to_numpy(float)
        wrf_mean_center = float(np.average(wrf["delta_tmax_mean_c"], weights=weights))
        wrf_p95_center = float(np.average(wrf["delta_tmax_p95_c"], weights=weights))
        blended = wrf[
            ["wrf_south_north", "wrf_west_east", "wrf_grid_latitude", "wrf_grid_longitude", "building_count", "maximum_building_distance_km"]
        ].copy()
        blended["source_id"] = row.source_id
        blended["experiment_id"] = row.experiment_id
        blended["gcm_future_period"] = row.future_period
        blended["wrf_spatial_period_code"] = period_to_wrf[row.future_period]
        blended["delta_mean_tmax_c"] = row.delta_mean_tmax_c + wrf["delta_tmax_mean_c"] - wrf_mean_center
        blended["delta_p95_tmax_c"] = row.delta_p95_tmax_c + wrf["delta_tmax_p95_c"] - wrf_p95_center
        blended["spatial_pattern_source"] = "1km_MPI-ESM1-2-HR_WRF_change_anomaly"
        blended["city_magnitude_source"] = "five_model_HKO_KP_anchored_QDM"
        blended_parts.append(blended)
    blended_field = pd.concat(blended_parts, ignore_index=True)
    blended_field.to_parquet(output / "cmip6_wrf_multimodel_tmax_change_field.parquet", index=False)

    factor_rows = []
    for model in models:
        for scenario in scenarios:
            baseline_variables = {}
            future_variables = {}
            for variable in ["huss", "psl", "rsds", "sfcWind"]:
                historical_path = CMIP6_COMPOUND_ROOT / f"{model}_historical_r1i1p1f1_Amon_{variable}.csv"
                future_path = CMIP6_COMPOUND_ROOT / f"{model}_{scenario}_r1i1p1f1_Amon_{variable}.csv"
                if not historical_path.exists() or not future_path.exists():
                    raise FileNotFoundError(f"Missing compound CMIP6 cache for {model} {scenario} {variable}")
                historical_variable = pd.read_csv(historical_path)
                future_variable = pd.read_csv(future_path)
                baseline_variables[variable] = weighted_warm_season_mean(historical_variable, variable)
                future_variables[variable] = future_variable
            for period, (start_year, end_year) in periods.items():
                factor = {
                    "source_id": model,
                    "experiment_id": scenario,
                    "future_period": period,
                    "maximum_temperature_delta_c": float(
                        model_summary.loc[
                            model_summary["source_id"].eq(model)
                            & model_summary["experiment_id"].eq(scenario)
                            & model_summary["future_period"].eq(period),
                            "delta_mean_tmax_c",
                        ].iloc[0]
                    ),
                }
                for variable in ["huss", "psl", "rsds", "sfcWind"]:
                    future_value = weighted_warm_season_mean(
                        future_variables[variable].loc[future_variables[variable]["year"].between(start_year, end_year)],
                        variable,
                    )
                    factor[f"historical_{variable}"] = baseline_variables[variable]
                    factor[f"future_{variable}"] = future_value
                    factor[f"{variable}_ratio"] = future_value / baseline_variables[variable]
                factor_rows.append(factor)
    factors = pd.DataFrame(factor_rows)
    factors.to_csv(output / "cmip6_compound_change_factors.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "models": models,
        "scenarios": scenarios,
        "temperature_frequency": "daily tasmax",
        "compound_frequency": "monthly huss psl rsds sfcWind",
        "historical_reference": "HKO King's Park 1992-2014",
        "future_periods": periods,
        "maximum_holdout_monthly_median_bias_c": float(historical_validation["median_bias_c"].abs().max()),
        "maximum_holdout_monthly_p95_bias_c": float(historical_validation["p95_bias_c"].abs().max()),
        "multimodel_tmax_change_field_rows": int(len(blended_field)),
        "tmax_spatial_method": "five-model HKO-anchored QDM magnitude plus centered 1 km WRF spatial change anomaly",
        "tmin_multimodel_daily_projection_available": False,
        "compound_factors_are_hourly_weather": False,
        "compound_factor_use": "apply to HKO-constrained observed hourly event analogues; do not interpret as a daily forecast",
    }
    (output / "cmip6_multimodel_validation.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def analyse_cmip6_multimodel_tmin(
    weather: pd.DataFrame,
    wrf_projections: pd.DataFrame,
    output: Path,
) -> dict:
    models = [str(value) for value in HAZARD_CONFIG["future_projection"]["multimodel_point_models"]]
    scenarios = [str(value) for value in HAZARD_CONFIG["future_projection"]["multimodel_scenarios"]]
    parts = []
    for model in models:
        for experiment in ["historical", *scenarios]:
            path = CMIP6_POINT_ROOT / f"{model}_{experiment}_r1i1p1f1_tasmin.csv"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing CMIP6 daily minimum-temperature cache: {path}. Run 00 first."
                )
            parts.append(pd.read_csv(path))
    raw = pd.concat(parts, ignore_index=True)
    observed = weather.loc[
        weather["station_code"].eq("KP") & weather["date"].dt.year.between(1992, 2014),
        ["date", "tmin_c"],
    ].dropna().copy()
    observed["year"] = observed["date"].dt.year
    observed["month"] = observed["date"].dt.month

    validation_rows = []
    historical = raw.loc[raw["experiment_id"].eq("historical")]
    for model, frame in historical.groupby("source_id"):
        calibration_model = frame.loc[frame["year"].le(2005)]
        validation_model = frame.loc[frame["year"].ge(2006)]
        calibration_observed = observed.loc[observed["year"].le(2005)]
        validation_observed = observed.loc[observed["year"].ge(2006)]
        for month in range(1, 13):
            mapped = quantile_map(
                validation_model.loc[
                    validation_model["month"].eq(month), "tasmin_raw_c"
                ].to_numpy(float),
                calibration_model.loc[
                    calibration_model["month"].eq(month), "tasmin_raw_c"
                ].to_numpy(float),
                calibration_observed.loc[
                    calibration_observed["month"].eq(month), "tmin_c"
                ].to_numpy(float),
            )
            observed_month = validation_observed.loc[
                validation_observed["month"].eq(month), "tmin_c"
            ].to_numpy(float)
            validation_rows.append(
                {
                    "source_id": model,
                    "month": month,
                    "model_corrected_median_c": float(np.median(mapped)),
                    "observed_median_c": float(np.median(observed_month)),
                    "median_bias_c": float(np.median(mapped) - np.median(observed_month)),
                    "model_corrected_p95_c": float(np.quantile(mapped, 0.95)),
                    "observed_p95_c": float(np.quantile(observed_month, 0.95)),
                    "p95_bias_c": float(
                        np.quantile(mapped, 0.95) - np.quantile(observed_month, 0.95)
                    ),
                }
            )
    historical_validation = pd.DataFrame(validation_rows)
    historical_validation.to_csv(
        output / "cmip6_tmin_historical_holdout_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    periods = {
        "mid_century_2041_2060": (2041, 2060),
        "late_century_2081_2100": (2081, 2100),
    }
    corrected_parts = []
    for (model, scenario), frame in raw.loc[raw["experiment_id"].isin(scenarios)].groupby(
        ["source_id", "experiment_id"]
    ):
        model_historical = historical.loc[historical["source_id"].eq(model)]
        for period, (start_year, end_year) in periods.items():
            period_frame = frame.loc[frame["year"].between(start_year, end_year)]
            for month, month_frame in period_frame.groupby("month"):
                month_frame = month_frame.copy()
                month_frame["tasmin_qdm_c"] = quantile_delta_map(
                    month_frame["tasmin_raw_c"].to_numpy(float),
                    model_historical.loc[
                        model_historical["month"].eq(month), "tasmin_raw_c"
                    ].to_numpy(float),
                    observed.loc[observed["month"].eq(month), "tmin_c"].to_numpy(float),
                )
                month_frame["future_period"] = period
                corrected_parts.append(month_frame)
    corrected = pd.concat(corrected_parts, ignore_index=True).sort_values(
        ["source_id", "experiment_id", "future_period", "year", "sequence_index"]
    )
    corrected.to_parquet(output / "cmip6_tmin_future_daily_qdm.parquet", index=False)

    observed_warm = observed.loc[observed["month"].between(6, 9), "tmin_c"].to_numpy(float)
    observed_mean = float(np.mean(observed_warm))
    observed_p95 = float(np.quantile(observed_warm, 0.95))
    model_rows = []
    threshold = float(HAZARD_CONFIG["hko_hot_night_c"])
    for (model, scenario, period), frame in corrected.groupby(
        ["source_id", "experiment_id", "future_period"]
    ):
        warm = frame.loc[frame["month"].between(6, 9)]
        annual = []
        for _, year_frame in warm.groupby("year"):
            events, event_nights, maximum_duration = one_dimensional_heatwave_metrics(
                year_frame.sort_values("sequence_index")["tasmin_qdm_c"].to_numpy(float),
                threshold,
            )
            annual.append(
                {
                    "threshold_nights": int((year_frame["tasmin_qdm_c"] >= threshold).sum()),
                    "events": events,
                    "event_nights": event_nights,
                    "maximum_duration": maximum_duration,
                }
            )
        annual_frame = pd.DataFrame(annual)
        mean_tmin = float(warm["tasmin_qdm_c"].mean())
        p95_tmin = float(warm["tasmin_qdm_c"].quantile(0.95))
        model_rows.append(
            {
                "source_id": model,
                "experiment_id": scenario,
                "future_period": period,
                "years": int(warm["year"].nunique()),
                "mean_tmin_c": mean_tmin,
                "p95_tmin_c": p95_tmin,
                "delta_mean_tmin_c": mean_tmin - observed_mean,
                "delta_p95_tmin_c": p95_tmin - observed_p95,
                "hot_nights_per_summer": float(annual_frame["threshold_nights"].mean()),
                "nighttime_heatwave_events_per_summer": float(annual_frame["events"].mean()),
                "nighttime_heatwave_event_nights_per_summer": float(
                    annual_frame["event_nights"].mean()
                ),
                "maximum_nighttime_heatwave_duration_nights": float(
                    annual_frame["maximum_duration"].mean()
                ),
            }
        )
    model_summary = pd.DataFrame(model_rows)
    model_summary.to_csv(
        output / "cmip6_tmin_future_model_summary.csv", index=False, encoding="utf-8-sig"
    )
    metric_columns = [
        "mean_tmin_c",
        "p95_tmin_c",
        "delta_mean_tmin_c",
        "delta_p95_tmin_c",
        "hot_nights_per_summer",
        "nighttime_heatwave_events_per_summer",
        "nighttime_heatwave_event_nights_per_summer",
        "maximum_nighttime_heatwave_duration_nights",
    ]
    ensemble_rows = []
    for (scenario, period), frame in model_summary.groupby(["experiment_id", "future_period"]):
        for metric in metric_columns:
            values = frame[metric]
            ensemble_rows.append(
                {
                    "experiment_id": scenario,
                    "future_period": period,
                    "metric": metric,
                    "models": int(len(values)),
                    "ensemble_p10": float(values.quantile(0.10)),
                    "ensemble_p50": float(values.quantile(0.50)),
                    "ensemble_p90": float(values.quantile(0.90)),
                    "ensemble_minimum": float(values.min()),
                    "ensemble_maximum": float(values.max()),
                }
            )
    pd.DataFrame(ensemble_rows).to_csv(
        output / "cmip6_tmin_future_ensemble_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    period_to_wrf = {
        "mid_century_2041_2060": "4049",
        "late_century_2081_2100": "9099",
    }
    blended_parts = []
    for row in model_summary.itertuples(index=False):
        wrf = wrf_projections.loc[
            wrf_projections["scenario"].eq(row.experiment_id)
            & wrf_projections["period_code"].eq(period_to_wrf[row.future_period])
        ].copy()
        weights = wrf["building_count"].to_numpy(float)
        wrf_mean_center = float(np.average(wrf["delta_tmin_mean_c"], weights=weights))
        wrf_p95_center = float(np.average(wrf["delta_tmin_p95_c"], weights=weights))
        blended = wrf[
            [
                "wrf_south_north",
                "wrf_west_east",
                "wrf_grid_latitude",
                "wrf_grid_longitude",
                "building_count",
                "maximum_building_distance_km",
            ]
        ].copy()
        blended["source_id"] = row.source_id
        blended["experiment_id"] = row.experiment_id
        blended["gcm_future_period"] = row.future_period
        blended["wrf_spatial_period_code"] = period_to_wrf[row.future_period]
        blended["delta_mean_tmin_c"] = (
            row.delta_mean_tmin_c + wrf["delta_tmin_mean_c"] - wrf_mean_center
        )
        blended["delta_p95_tmin_c"] = (
            row.delta_p95_tmin_c + wrf["delta_tmin_p95_c"] - wrf_p95_center
        )
        blended["spatial_pattern_source"] = "1km_MPI-ESM1-2-HR_WRF_change_anomaly"
        blended["city_magnitude_source"] = "five_model_HKO_KP_anchored_QDM"
        blended_parts.append(blended)
    blended_field = pd.concat(blended_parts, ignore_index=True)
    blended_field.to_parquet(
        output / "cmip6_wrf_multimodel_tmin_change_field.parquet", index=False
    )

    factor_path = output / "cmip6_compound_change_factors.csv"
    factors = pd.read_csv(factor_path)
    factors = factors.merge(
        model_summary[
            [
                "source_id",
                "experiment_id",
                "future_period",
                "delta_mean_tmin_c",
            ]
        ].rename(columns={"delta_mean_tmin_c": "minimum_temperature_delta_c"}),
        on=["source_id", "experiment_id", "future_period"],
        how="left",
        validate="one_to_one",
    )
    factors.to_csv(factor_path, index=False, encoding="utf-8-sig")
    return {
        "tmin_temperature_frequency": "daily tasmin",
        "tmin_maximum_holdout_monthly_median_bias_c": float(
            historical_validation["median_bias_c"].abs().max()
        ),
        "tmin_maximum_holdout_monthly_p95_bias_c": float(
            historical_validation["p95_bias_c"].abs().max()
        ),
        "multimodel_tmin_change_field_rows": int(len(blended_field)),
        "tmin_spatial_method": "five-model HKO-anchored QDM magnitude plus centered 1 km WRF spatial change anomaly",
        "tmin_multimodel_daily_projection_available": True,
    }


def analyse_wrf_future_projection(
    stock: gpd.GeoDataFrame,
    weather: pd.DataFrame,
    coordinates: pd.DataFrame,
    output: Path,
) -> dict:
    baseline_max_path = wrf_temperature_path("ssp245", "1120", "dailymax")
    baseline_min_path = wrf_temperature_path("ssp245", "1120", "dailymin")
    with xr.open_dataset(baseline_max_path, engine="netcdf4") as maximum_dataset, xr.open_dataset(
        baseline_min_path, engine="netcdf4"
    ) as minimum_dataset:
        latitude = maximum_dataset["XLAT"].values
        longitude = maximum_dataset["XLONG"].values
        land_use = maximum_dataset["LU_INDEX"].values
        dates = pd.DatetimeIndex(maximum_dataset["Time"].values)
        maximum_variable = next(iter(maximum_dataset.data_vars))
        minimum_variable = next(iter(minimum_dataset.data_vars))
        baseline_tmax_full = maximum_dataset[maximum_variable].values
        baseline_tmin_full = minimum_dataset[minimum_variable].values

    building_centroids = gpd.GeoSeries(stock.geometry.centroid, crs=stock.crs).to_crs(4326)
    latitude0 = float(building_centroids.y.mean())
    grid_xy = np.c_[
        longitude.ravel() * np.cos(np.deg2rad(latitude0)) * 111.32,
        latitude.ravel() * 110.57,
    ]
    building_xy = np.c_[
        building_centroids.x.to_numpy(float) * np.cos(np.deg2rad(latitude0)) * 111.32,
        building_centroids.y.to_numpy(float) * 110.57,
    ]
    land_mask = np.isfinite(land_use) & ~np.isclose(land_use, 16.0)
    land_flat = np.flatnonzero(land_mask.ravel())
    distance, land_position = cKDTree(grid_xy[land_flat]).query(building_xy)
    flat_index = land_flat[land_position]
    row_index, column_index = np.unravel_index(flat_index, latitude.shape)
    building_mapping = pd.DataFrame(
        {
            "BUILDINGSTRUCTUREID": stock["BUILDINGSTRUCTUREID"].astype(str).to_numpy(),
            "wrf_south_north": row_index,
            "wrf_west_east": column_index,
            "wrf_grid_latitude": latitude[row_index, column_index],
            "wrf_grid_longitude": longitude[row_index, column_index],
            "wrf_land_grid_distance_km": distance,
            "wrf_land_grid_within_2km": distance <= 2.0,
        }
    )
    building_mapping.to_parquet(output / "building_wrf_land_grid_mapping.parquet", index=False)
    grid = (
        building_mapping.groupby(["wrf_south_north", "wrf_west_east", "wrf_grid_latitude", "wrf_grid_longitude"], as_index=False)
        .agg(
            building_count=("BUILDINGSTRUCTUREID", "size"),
            mean_building_distance_km=("wrf_land_grid_distance_km", "mean"),
            maximum_building_distance_km=("wrf_land_grid_distance_km", "max"),
            buildings_within_2km=("wrf_land_grid_within_2km", "sum"),
        )
        .sort_values(["wrf_south_north", "wrf_west_east"])
        .reset_index(drop=True)
    )
    selected_rows = grid["wrf_south_north"].to_numpy(int)
    selected_columns = grid["wrf_west_east"].to_numpy(int)
    baseline_tmax = baseline_tmax_full[:, selected_rows, selected_columns]
    baseline_tmin = baseline_tmin_full[:, selected_rows, selected_columns]
    baseline_day = wrf_grid_climatology(baseline_tmax, dates, float(HAZARD_CONFIG["hko_very_hot_day_c"]))
    baseline_night = wrf_grid_climatology(baseline_tmin, dates, float(HAZARD_CONFIG["hko_hot_night_c"]))

    projection_parts = []
    for scenario, period_code in wrf_projection_cases():
        if scenario == "ssp245" and period_code == "1120":
            case_dates = dates
            tmax = baseline_tmax
            tmin = baseline_tmin
        else:
            with xr.open_dataset(wrf_temperature_path(scenario, period_code, "dailymax"), engine="netcdf4") as maximum_dataset, xr.open_dataset(
                wrf_temperature_path(scenario, period_code, "dailymin"), engine="netcdf4"
            ) as minimum_dataset:
                case_dates = pd.DatetimeIndex(maximum_dataset["Time"].values)
                tmax = maximum_dataset[next(iter(maximum_dataset.data_vars))].values[:, selected_rows, selected_columns]
                tmin = minimum_dataset[next(iter(minimum_dataset.data_vars))].values[:, selected_rows, selected_columns]
        if np.any(tmax < tmin):
            raise ValueError(f"WRF Tmax is below Tmin for {scenario} {period_code}")
        day = wrf_grid_climatology(tmax, case_dates, float(HAZARD_CONFIG["hko_very_hot_day_c"]))
        night = wrf_grid_climatology(tmin, case_dates, float(HAZARD_CONFIG["hko_hot_night_c"]))
        part = grid.copy()
        part["scenario"] = scenario
        part["period_code"] = period_code
        part["first_year"] = int(case_dates.year.min())
        part["last_year"] = int(case_dates.year.max())
        for prefix, metrics, baseline_metrics in [("tmax", day, baseline_day), ("tmin", night, baseline_night)]:
            for metric, values in metrics.items():
                part[f"{prefix}_{metric}"] = values
                part[f"delta_{prefix}_{metric}"] = values - baseline_metrics[metric]
        projection_parts.append(part)
    projections = pd.concat(projection_parts, ignore_index=True)
    projections["absolute_temperature_role"] = "published_product_diagnostic_only"
    projections["downstream_weather_role"] = "spatial_change_signal_anchored_to_HKO"
    projections.to_parquet(output / "wrf_hong_kong_grid_projection_metrics.parquet", index=False)

    multimodel = analyse_cmip6_multimodel_tmax(weather, projections, output)
    multimodel.update(analyse_cmip6_multimodel_tmin(weather, projections, output))
    (output / "cmip6_multimodel_validation.json").write_text(
        json.dumps(multimodel, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    diagnostics = wrf_hko_daily_diagnostic(
        weather,
        coordinates,
        latitude,
        longitude,
        land_mask,
        dates,
        baseline_tmax_full,
        baseline_tmin_full,
        output,
    )
    spatial_bias = validate_and_apply_wrf_spatial_bias(
        weather,
        coordinates,
        dates,
        baseline_tmax,
        baseline_tmin,
        grid,
        building_mapping,
        output,
    )
    holdout = diagnostics.loc[
        diagnostics["evaluation_period"].eq("pre_correction_temporal_holdout_2011_2015")
        & diagnostics["station_code"].eq("ALL")
    ].set_index("variable")
    summary = {
        "source_doi": "10.5281/zenodo.7206739",
        "article_doi": "10.1175/JAMC-D-22-0201.1",
        "driver_model": "MPI-ESM1-2-HR",
        "wrf_grid_shape": list(latitude.shape),
        "wrf_land_grids_used_by_buildings": int(len(grid)),
        "buildings_mapped": int(len(building_mapping)),
        "median_building_to_land_grid_distance_km": float(np.median(distance)),
        "p99_building_to_land_grid_distance_km": float(np.quantile(distance, 0.99)),
        "maximum_building_to_land_grid_distance_km": float(np.max(distance)),
        "buildings_farther_than_2km_from_wrf_land_grid": int(np.sum(distance > 2.0)),
        "pre_correction_temporal_holdout_tmax_mae_c": float(holdout.loc["tmax", "mae_c"]),
        "pre_correction_temporal_holdout_tmin_mae_c": float(holdout.loc["tmin", "mae_c"]),
        "unadjusted_absolute_wrf_temperature_used_downstream": False,
        "downstream_primary_fields": [
            "delta_tmax_mean_c", "delta_tmax_p95_c", "delta_tmin_mean_c", "delta_tmin_p95_c"
        ],
        "ssp245_ssp585_uncertainty": "combine these WRF spatial change patterns with the five-model CMIP6 point ensemble",
        "ssp126_uncertainty": "single-model sensitivity only",
        "multimodel": multimodel,
        "distributional_spatial_bias_correction": spatial_bias,
    }
    (output / "wrf_future_projection_validation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    parser.add_argument("--cis-cache-dir", type=Path, default=CIS_CACHE)
    parser.add_argument("--allow-legacy-weather", action="store_true")
    parser.add_argument("--cell-m", type=int, default=int(HAZARD_CONFIG["output_grid_m"]))
    parser.add_argument("--allow-unvalidated-map", action="store_true", help="Write a sensitivity-only map when the morphology gate fails.")
    parser.add_argument("--li-replication-only", action="store_true", help="Audit reproducibility of the Li et al. textual method using HKO Tmax only, then stop.")
    parser.add_argument(
        "--future-projection-only",
        action="store_true",
        help="Build and diagnose the downloaded 1 km CMIP6-WRF change fields, then stop before the historical morphology model.",
    )
    parser.add_argument(
        "--li-replication-source",
        choices=["open-api", "cited-cis"],
        default="open-api",
        help="Use the canonical HKO all-year API by default; cited-cis requires the optional 44-file archive.",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    if args.li_replication_only:
        if args.li_replication_source == "cited-cis":
            cited_tmax, cited_audit = load_hko_cis_tmax(args.cis_cache_dir.resolve())
            cited_audit.to_csv(output / "li_cited_hko_cis_source_audit.csv", index=False, encoding="utf-8-sig")
        else:
            cited_tmax = load_hko_open_data_cache("HKO", "CLMMAXT", args.cache_dir.resolve()).rename(columns={"CLMMAXT": "tmax_c"})
            cited_audit = pd.DataFrame(
                [
                    {
                        "station_code": "HKO",
                        "source_url": f"{HKO_OPEN_DATA_API}?dataType=CLMMAXT&station=HKO&rformat=json",
                        "cache_file": str(args.cache_dir.resolve() / "CLMMAXT_HKO.json"),
                        "rows": len(cited_tmax),
                        "nonmissing_tmax_rows": int(cited_tmax["tmax_c"].notna().sum()),
                        "first_date": cited_tmax["date"].min(),
                        "last_date": cited_tmax["date"].max(),
                        "transport": "validated_cache",
                    }
                ]
            )
            source_overlap = audit_available_cis_against_open_api(
                cited_tmax,
                args.cis_cache_dir.resolve(),
            )
            source_overlap.to_csv(
                output / "hko_cis_open_api_overlap_audit.csv",
                index=False,
                encoding="utf-8-sig",
            )
        cited_audit.to_csv(output / "li_hko_source_audit.csv", index=False, encoding="utf-8-sig")
        sensitivity = li_implementation_sensitivity(cited_tmax)
        sensitivity.to_csv(output / "li_definition_implementation_sensitivity.csv", index=False, encoding="utf-8-sig")
        columns = [
            "sample_scope", "day_index_policy", "quantile_interpolation", "comparison", "event_assembly",
            "documented_primary", "published_event_summary_matched", "published_trends_matched",
            "published_key_metrics_matched", "event_count_1980_2023",
            "post_2010_inclusive_event_count", "post_2010_inclusive_mean_intensity_c", "event_span_days_slope_per_year",
            "event_frequency_slope_per_year", "mean_duration_slope_days_per_year",
            "annual_mean_event_cumulative_intensity_slope_c_days_per_year",
        ]
        print(sensitivity.loc[sensitivity["documented_primary"] | sensitivity["published_key_metrics_matched"], columns].to_string(index=False))
        print(f"Li method reproducibility audit complete: {output / 'li_definition_implementation_sensitivity.csv'}")
        return

    weather, api_audit = load_weather(args.cache_dir.resolve(), args.allow_legacy_weather)
    api_audit.to_csv(output / "hko_api_station_audit.csv", index=False, encoding="utf-8-sig")
    weather.to_parquet(output / "hko_daily_temperature_all_available.parquet", index=False)
    coordinates = parse_station_coordinates(COORD_JS)
    stock = gpd.read_parquet(BUILDINGS)
    future_projection = analyse_wrf_future_projection(stock, weather, coordinates, output)
    if args.future_projection_only:
        print(f"01 future projection complete: {output}")
        print(json.dumps(future_projection, ensure_ascii=False, indent=2))
        return
    flags, annual, coverage, events = definition_metrics(weather)
    flags.to_parquet(output / "daily_definition_flags.parquet", index=False)
    annual.to_csv(output / "station_year_heat_metrics.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(output / "station_year_coverage.csv", index=False, encoding="utf-8-sig")
    events.to_csv(output / "heatwave_event_catalog.csv", index=False, encoding="utf-8-sig")
    published_counts = pd.DataFrame(HKO_PUBLISHED_ANNUAL_COUNTS)
    calculated_counts = annual.loc[
        annual["station_code"].eq("HKO"),
        ["year", "hko_very_hot_days", "hko_hot_nights", "hko_extremely_hot_days"],
    ]
    official_crosscheck = published_counts.merge(calculated_counts, on="year", how="left", validate="one_to_one")
    for metric in ["hko_very_hot_days", "hko_hot_nights", "hko_extremely_hot_days"]:
        official_crosscheck[f"{metric}_match"] = official_crosscheck[metric].eq(official_crosscheck[f"{metric}_published"])
    official_crosscheck["all_metrics_match"] = official_crosscheck[
        ["hko_very_hot_days_match", "hko_hot_nights_match", "hko_extremely_hot_days_match"]
    ].all(axis=1)
    official_crosscheck.to_csv(output / "hko_official_annual_count_crosscheck.csv", index=False, encoding="utf-8-sig")
    trends = trend_table(annual)
    trends.to_csv(output / "trend_reconciliation.csv", index=False, encoding="utf-8-sig")
    li_sensitivity = li_implementation_sensitivity(weather)
    li_sensitivity.to_csv(output / "li_definition_implementation_sensitivity.csv", index=False, encoding="utf-8-sig")

    cells, station_features, building_index = grid_predictors(stock, coordinates, args.cell_m, output / "derived_spatial")
    geo_features = ["x_km", "y_km", "terrain_elevation_proxy_m"]
    spatial_pattern = r"(building_coverage|building_volume_density|facade_area_density|building_count_density|building_use_(residential|commercial|education|health|industrial)_coverage|worldcover_(tree|vegetation|built|water)_fraction|road_(all|major)_length_density_proxy)_\d+m"
    all_features = geo_features + [c for c in station_features.columns if re.fullmatch(spatial_pattern, c)] + ["distance_to_mapped_water_m", "distance_to_major_route_m"]
    target_specs = {
        "very_hot_days": {
            "annual_metric": "hko_very_hot_days",
            "prediction_field": "predicted_mean_annual_very_hot_days_2011_2025",
            "percentile_field": "very_hot_day_hazard_percentile_0_1",
            "definition": "2011-2025 mean annual days with daily maximum temperature >=33 C",
        },
        "hot_nights": {
            "annual_metric": "hko_hot_nights",
            "prediction_field": "predicted_mean_annual_hot_nights_2011_2025",
            "percentile_field": "hot_night_hazard_percentile_0_1",
            "definition": "2011-2025 mean annual days with daily minimum temperature >=28 C",
        },
    }
    target_runs = {}
    all_station_inputs = []
    all_cv_predictions = []
    all_cv_metrics = []
    all_selection_tests = []
    for target_name, spec in target_specs.items():
        target = contemporary_station_target(annual, spec["annual_metric"])
        stations = target.merge(station_features, on="station_code", how="inner")
        stations = stations.loc[~stations["station_code"].isin(OFFSHORE_EXCLUSIONS)].dropna(subset=["x_m", "y_m", "target"]).reset_index(drop=True)
        if len(stations) < 20:
            raise RuntimeError(f"Only {len(stations)} stations support {target_name}; at least 20 are required for a fine-surface attempt.")
        cv_predictions, cv_metrics = spatial_cv(stations, geo_features, all_features)
        selected, status, canonical, selection_tests = select_model(cv_metrics, cv_predictions)
        stations["target_metric"] = target_name
        cv_predictions["target_metric"] = target_name
        cv_metrics["target_metric"] = target_name
        selection_tests["target_metric"] = target_name
        all_station_inputs.append(stations)
        all_cv_predictions.append(cv_predictions)
        all_cv_metrics.append(cv_metrics)
        all_selection_tests.append(selection_tests)
        target_runs[target_name] = {
            "stations": stations,
            "selected_model": selected,
            "status": status,
            "canonical_morphology_surface": canonical,
            "station_count": len(stations),
            "definition": spec["definition"],
        }
    pd.concat(all_station_inputs, ignore_index=True).to_csv(output / "station_spatial_model_inputs.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_cv_predictions, ignore_index=True).to_csv(output / "spatial_model_leave_one_station_out_predictions.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_cv_metrics, ignore_index=True).to_csv(output / "spatial_model_cv.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_selection_tests, ignore_index=True).to_csv(output / "spatial_model_paired_superiority_tests.csv", index=False, encoding="utf-8-sig")

    canonical = all(run["canonical_morphology_surface"] for run in target_runs.values())
    status = "validated_day_and_night_morphology_surfaces" if canonical else "fine_surface_withheld"
    write_map = canonical or args.allow_unvalidated_map
    if not write_map:
        for stale_map in [output / "building_heat_hazard.parquet", output / f"hazard_grid_{args.cell_m}m.parquet"]:
            stale_map.unlink(missing_ok=True)
    if write_map:
        cell_fields = ["gx", "gy"]
        for target_name, spec in target_specs.items():
            stations = target_runs[target_name]["stations"]
            selected = target_runs[target_name]["selected_model"]
            if selected == "idw_p2":
                prediction = idw_predict(stations[["x_m", "y_m"]].to_numpy(float), stations["target"].to_numpy(float), cells[["x_m", "y_m"]].to_numpy(float))
            elif selected == "nearest_station":
                prediction = nearest_predict(stations[["x_m", "y_m"]].to_numpy(float), stations["target"].to_numpy(float), cells[["x_m", "y_m"]].to_numpy(float))
            else:
                features = geo_features if selected == "geographic_ridge" else all_features
                estimator = model_factories(all_features)[selected]
                estimator.fit(stations[features], stations["target"])
                prediction = estimator.predict(cells[features])
            cells[spec["prediction_field"]] = np.maximum(prediction, 0.0)
            cells[spec["percentile_field"]] = pd.Series(cells[spec["prediction_field"]]).rank(pct=True).to_numpy()
            cells[f"{target_name}_spatial_model"] = selected
            cell_fields.extend([spec["prediction_field"], spec["percentile_field"], f"{target_name}_spatial_model"])
        cells["spatial_evidence_status"] = status
        cell_fields.append("spatial_evidence_status")
        cell_values = cells[cell_fields]
        building_hazard = building_index.merge(cell_values, on=["gx", "gy"], how="left")
        building_hazard.to_parquet(output / "building_heat_hazard.parquet", index=False)
        cells.to_parquet(output / f"hazard_grid_{args.cell_m}m.parquet", index=False)

    li_row = trends.loc[(trends["scope"] == "Li_exact_HKO_1980_2023") & (trends["metric"] == "li_event_span_days")]
    observed_li_slope = float(li_row["slope_per_year"].iloc[0]) if len(li_row) else math.nan
    observed_li_se = float(li_row["standard_error"].iloc[0]) if len(li_row) else math.nan
    li_events = events.loc[events["station_code"].eq("HKO") & events["start_year"].between(1980, 2023)].copy()
    li_after_2010 = li_events.loc[li_events["start_year"].gt(2010)]
    li_event_count = len(li_events)
    li_min_duration = int(li_events["duration_days"].min()) if li_event_count else None
    li_max_duration = int(li_events["duration_days"].max()) if li_event_count else None
    li_post_2010_count = len(li_after_2010)
    li_post_2010_mean_intensity = float(li_after_2010["mean_intensity_c"].mean()) if len(li_after_2010) else math.nan
    li_slope_compatible = bool(math.isfinite(observed_li_se) and observed_li_slope - 1.96 * observed_li_se <= 0.49 <= observed_li_slope + 1.96 * observed_li_se)
    li_replication_passed = bool(
        li_event_count == 57
        and li_min_duration == 3
        and li_max_duration == 20
        and li_post_2010_count == 41
        and math.isfinite(li_post_2010_mean_intensity)
        and abs(li_post_2010_mean_intensity - 0.90) <= 0.05
        and li_slope_compatible
    )
    validation = {
        "documented_hko_daily_station_count": len(DAILY_STATIONS),
        "weather_station_count_loaded": int(weather["station_code"].nunique()),
        "spatial_station_count": min(run["station_count"] for run in target_runs.values()),
        "output_grid_m": args.cell_m,
        "output_grid_is_observation_precision": False,
        "official_index_period": "calendar year",
        "li_event_analysis_period": "May-September",
        "spatial_target_period": "2011-2025 common-period empirical mean; not a 30-year climatological normal",
        "spatial_target_minimum_valid_years": 11,
        "spatial_target_temporal_extrapolation_applied": False,
        "data_completeness_rule": "WMO Climpact: no more than 3 missing days in any month and no more than 15 missing days in the reported period",
        "hko_published_annual_count_crosscheck_all_matched": bool(official_crosscheck["all_metrics_match"].all()),
        "hko_published_annual_count_crosscheck_years": official_crosscheck["year"].astype(int).tolist(),
        "selected_spatial_model": {name: run["selected_model"] for name, run in target_runs.items()},
        "spatial_evidence_status": status,
        "canonical_building_hazard_released": canonical,
        "future_projection": future_projection,
        "sensitivity_map_written": bool(write_map and not canonical),
        "spatial_targets": {
            name: {
                "definition": run["definition"],
                "station_count": run["station_count"],
                "selected_model": run["selected_model"],
                "status": run["status"],
                "canonical_morphology_surface": run["canonical_morphology_surface"],
            }
            for name, run in target_runs.items()
        },
        "li_2025_reported_heatwave_days_slope": 0.49,
        "li_2025_reported_heatwave_days_slope_se": 0.09,
        "this_run_li_replication_slope": observed_li_slope,
        "this_run_li_replication_slope_se": observed_li_se,
        "li_replication_absolute_difference": abs(observed_li_slope - 0.49) if math.isfinite(observed_li_slope) else None,
        "li_reported_event_count_1980_2023": 57,
        "this_run_event_count_1980_2023": li_event_count,
        "li_reported_duration_range_days": [3, 20],
        "this_run_duration_range_days": [li_min_duration, li_max_duration],
        "li_reported_post_2010_event_count": 41,
        "this_run_post_2010_event_count": li_post_2010_count,
        "li_reported_post_2010_mean_intensity_c": 0.90,
        "this_run_post_2010_mean_intensity_c": li_post_2010_mean_intensity,
        "li_slope_95pct_interval_contains_reported_value": li_slope_compatible,
        "li_exact_replication_passed": li_replication_passed,
        "hazard_building_characteristics_policy": "neighbourhood morphology may modify outdoor hazard; own envelope, AC, floor and orientation are excluded and handled as vulnerability/consequence",
        "sources": {
            "hko_api": "https://www.hko.gov.hk/en/weatherAPI/doc/files/HKO_Open_Data_API_Documentation.pdf",
            "hko_very_hot_day_definition": "https://www.hko.gov.hk/en/education/weather/forecasting/00553-forecasting-of-very-hot-weather.html",
            "hko_hot_night_definition": "https://www.info.gov.hk/gia/general/202107/21/P2021072100319.htm",
            "wmo_climpact_completeness": "https://github.com/ARCCSS-extremes/climpact/blob/master/www/user_guide/Climpact_user_guide.md",
            "li_definition": "https://doi.org/10.1016/j.atmosres.2024.107845",
            "hong_kong_spatial_sota": "https://doi.org/10.1016/j.scs.2025.106763",
            "hong_kong_lur": "https://doi.org/10.1016/j.scitotenv.2017.08.252",
        },
    }
    (output / "hazard_validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"01 complete: {output}")
    print(json.dumps({k: validation[k] for k in ["weather_station_count_loaded", "spatial_station_count", "selected_spatial_model", "spatial_evidence_status", "canonical_building_hazard_released", "this_run_li_replication_slope"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
