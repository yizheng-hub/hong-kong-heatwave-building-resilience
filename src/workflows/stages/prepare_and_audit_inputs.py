"""Prepare and audit the sourced inputs used by the numbered paper pipeline."""

from __future__ import annotations

import argparse
import calendar
import concurrent.futures
import hashlib
import io
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
CONFIG_PATH = ROOT / "config" / "foundation_analysis.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
HAZARD_CONFIG = CONFIG["hazard"]
THERMAL_CONFIG = CONFIG["thermal"]
OUTPUT = ROOT / CONFIG["output_root"] / "00_audit"
BUILDING_STOCK = ROOT / "output" / "heatwave_stock_foundation_v0_3" / "building_stock_repaired.geoparquet"
RESIDENTIAL_CONTEXT = ROOT / "output" / "observational_evidence" / "residential_building_census_context.geoparquet"
EXTERNAL_RAW = ROOT / "data" / "external_validation_raw"
HKUST_ARCHIVE = EXTERNAL_RAW / "hkust" / "All_Data.zip"
POLYU_DIR = ROOT / "data" / "polyu_operational"
POLYU_FULL_CAMPUS = POLYU_DIR / "campus_building_hourly_mapping_v22_20250709_20260709_v1.zip"
POLYU_WEATHER_RAW = ROOT / "data" / "weather" / "raw" / "nasa_power_polyu_20250709_20260708.json"
POLYU_HOURLY_WEATHER = ROOT / "intermediate" / "source_derived" / "weather" / "polyu_hourly_weather_hko_anchored.parquet"
BG_SOURCE_TABLE = ROOT / "data" / "official" / "census2021_HMA_BG_21C" / "HMA_BG_21C" / "BuildingGroup.CSV"
BG_MEMBERSHIP = ROOT / "intermediate" / "source_derived" / "census2021_building_group_members.parquet"
BG_MEMBERSHIP_METADATA = BG_MEMBERSHIP.with_suffix(".metadata.json")
ALS_URL = "https://www.als.gov.hk/data/ALS-GeoJSON.zip"
ALS_ZIP = ROOT / "data" / "official" / "als_2026_08_13" / "ALS-GeoJSON.zip"
ALS_BUILDINGS = ROOT / "intermediate" / "source_derived" / "als_building_addresses_2026_08_13.parquet"
ALS_BUILDINGS_METADATA = ALS_BUILDINGS.with_suffix(".metadata.json")
HMA_BOUNDARIES = ROOT / "intermediate" / "source_derived" / "census2021_housing_market_areas.geoparquet"
HMA_BOUNDARIES_METADATA = HMA_BOUNDARIES.with_suffix(".metadata.json")
SSUG_SOURCE_TABLE = ROOT / "data" / "official" / "census2021_SSUG_21C" / "SSUG_21C.csv"
SSUG_BOUNDARIES = ROOT / "intermediate" / "source_derived" / "census2021_small_subunit_groups.geoparquet"
SSUG_BOUNDARIES_METADATA = SSUG_BOUNDARIES.with_suffix(".metadata.json")
LAND_USE_URL = (
    "https://static.csdi.gov.hk/csdi-webpage/download/common/"
    "33c4bb4d4a0cac30c5afa16512ae0c474d16ab295bc3a0d48a145bd0d8ed95de"
)
LAND_USE_DIR = ROOT / "data" / "official" / "land_utilization_2021"
LAND_USE_ZIP = LAND_USE_DIR / "2021_Raster_Grids_on_Land_Utilization_GEOTIFF.zip"
LAND_USE_TIFF = LAND_USE_DIR / "LUMHK_RasterGrid_2021" / "LUM_end2021.tif"
LAND_USE_CODE_TABLE = LAND_USE_DIR / "LUMHK_RasterGrid_2021" / "CODE TABLE.xlsx"
CSDI_BG_QUERY = (
    "https://portal.csdi.gov.hk/server/rest/services/common/"
    "censtatd_rcd_1728978338390_76872/MapServer/0/query"
)
CSDI_HMA_QUERY = (
    "https://portal.csdi.gov.hk/server/rest/services/common/"
    "censtatd_rcd_1728978338390_76872/MapServer/1/query"
)
CSDI_SSUG_QUERY = (
    "https://portal.csdi.gov.hk/server/rest/services/common/"
    "censtatd_rcd_1635933365202_19171/MapServer/0/query"
)
CENTAMAP_MODEL = "https://hkdata.centanet.com/BigData/api/PageData/GetModel"
HKO_CIS_CACHE = ROOT / "data" / "weather" / "hko_cis_daily_yearly"
HKO_OPEN_DATA_CACHE = ROOT / "data" / "weather" / "hko_open_data_daily"
HKO_CIS_YEAR_URL = "https://www.hko.gov.hk/cis/individual_day/daily_{year}.xml"
HKO_OPEN_DATA_API = "https://data.weather.gov.hk/weatherAPI/opendata/opendata.php"
HKO_CIS_YEARS = range(int(HAZARD_CONFIG["hko_cis_start_year"]), int(HAZARD_CONFIG["hko_cis_end_year"]) + 1)
HKO_DAILY_STATIONS = [str(code) for code in HAZARD_CONFIG["hko_daily_stations"]]
HKO_DAILY_DATA_TYPES = ("CLMMAXT", "CLMMINT")
REFERENCE_DAILY_WEATHER = ROOT / "intermediate" / "source_derived" / "weather" / "hko_kp_daily_2016_2025.csv"
REFERENCE_WEATHER_OUTPUT = ROOT / "output" / "heatwave_day3"
NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
NASA_POWER_PARAMETERS = ("T2M", "RH2M", "ALLSKY_SFC_SW_DWN", "WS10M", "PRECTOTCORR")
REFERENCE_EVENT_SPINUP_DAYS = 2
REFERENCE_EVENT_RECOVERY_DAYS = 2
REFERENCE_EVENT_COUNT = 3
REFERENCE_WEATHER_POINT = (22.311, 114.172)
WRF_HEATWAVE_DIR = ROOT / "data" / "weather" / "cmip6_wrf_prd"
WRF_HEATWAVE_ARCHIVE = WRF_HEATWAVE_DIR / "heatwave_data.zip"
WRF_HEATWAVE_DATA = WRF_HEATWAVE_DIR / "data"
WRF_HEATWAVE_ZENODO_MD5 = "db4b980d3e6296d660fbbd007f0da0c3"
CMIP6_CATALOG = ROOT / "data" / "climate" / "cmip6-zarr-consolidated-stores.csv"
CMIP6_POINT_ROOT = ROOT / "data" / "climate" / "cmip6_hong_kong_point_raw"
CMIP6_REFERENCE = (22.311897, 114.172878)
CMIP6_MODELS = [
    "ACCESS-CM2",
    "BCC-CSM2-MR",
    "GFDL-ESM4",
    "MIROC6",
    "MPI-ESM1-2-HR",
]
CMIP6_EXPERIMENTS = ["historical", "ssp245", "ssp585"]
CMIP6_GCS_REQUEST_TIMEOUT_SECONDS = 90
CMIP6_BLOCK_MAX_ATTEMPTS = 5


REQUIRED_INPUTS = {
    "analysis_config": CONFIG_PATH,
    "building_stock": BUILDING_STOCK,
    "residential_census_context": RESIDENTIAL_CONTEXT,
    "legacy_weather_cache": ROOT / "intermediate" / "source_derived" / "weather" / "hko_multistation_daily_2016_2025.csv",
    "hko_station_coordinates": ROOT / "data" / "weather" / "raw" / "hko_multistation" / "hko_station_coordinates.js",
    "hko_station_periods": ROOT / "data" / "weather" / "raw" / "hko_multistation" / "hko_station_periods.json",
    "hko_open_data_hko_tmax": HKO_OPEN_DATA_CACHE / "CLMMAXT_HKO.json",
    "hko_open_data_hko_tmin": HKO_OPEN_DATA_CACHE / "CLMMINT_HKO.json",
    "wrf_heatwave_archive": WRF_HEATWAVE_ARCHIVE,
    "cmip6_catalog": CMIP6_CATALOG,
    "census_building_group": BG_SOURCE_TABLE,
    "census_building_group_membership": BG_MEMBERSHIP,
    "address_lookup_service_archive": ALS_ZIP,
    "address_lookup_service_buildings": ALS_BUILDINGS,
    "census_housing_market_area_boundaries": HMA_BOUNDARIES,
    "census_small_subunit_group_table": SSUG_SOURCE_TABLE,
    "census_small_subunit_group_boundaries": SSUG_BOUNDARIES,
    "planning_department_land_utilization_archive": LAND_USE_ZIP,
    "planning_department_land_utilization_grid": LAND_USE_TIFF,
    "planning_department_land_utilization_codes": LAND_USE_CODE_TABLE,
    "hourly_heatwave_weather": ROOT / "output" / "heatwave_day3" / "heatwave_hourly_bias_corrected.csv",
    "energyplus_engine": ROOT / "software" / "energyplus" / "EnergyPlus-26.1.0-6f2e40d102-Windows-x86_64" / "energyplus.exe",
}


THRESHOLDS = [
    {
        "id": "hko_very_hot_day",
        "variable": "daily maximum outdoor air temperature",
        "operator": ">=",
        "value": HAZARD_CONFIG["hko_very_hot_day_c"],
        "unit": "degC",
        "source": "Hong Kong Observatory",
        "source_url": "https://www.hko.gov.hk/en/climate_change/proj_hk_annual_vhot_days.htm",
        "role": "official local fixed-threshold indicator",
        "is_official_hko_threshold": True,
        "is_main_heatwave_duration_definition": False,
    },
    {
        "id": "hko_extremely_hot_day",
        "variable": "daily maximum outdoor air temperature",
        "operator": ">=",
        "value": HAZARD_CONFIG["hko_extremely_hot_day_c"],
        "unit": "degC",
        "source": "Hong Kong Observatory",
        "source_url": "https://www.hko.gov.hk/en/climate_change/proj_hk_annual_vhot_days.htm",
        "role": "official local severe-heat sensitivity",
        "is_official_hko_threshold": True,
        "is_main_heatwave_duration_definition": False,
    },
    {
        "id": "hko_hot_night",
        "variable": "daily minimum outdoor air temperature",
        "operator": ">=",
        "value": HAZARD_CONFIG["hko_hot_night_c"],
        "unit": "degC",
        "source": "Hong Kong Observatory",
        "source_url": "https://www.hko.gov.hk/en/climate_change/proj_hk_annual_vhot_days.htm",
        "role": "official local hot-night indicator; not an indoor threshold",
        "is_official_hko_threshold": True,
        "is_main_heatwave_duration_definition": False,
    },
    {
        "id": "li_2025_heatwave",
        "variable": "daily maximum outdoor air temperature in May-September",
        "operator": "> calendar-day 90th percentile",
        "value": HAZARD_CONFIG["li_percentile"],
        "unit": "percentile",
        "source": "Li et al. (2025), Atmospheric Research 315, 107845",
        "source_url": "https://doi.org/10.1016/j.atmosres.2024.107845",
        "role": "literature-anchored external-consistency definition: 15-day moving window, >=3 consecutive exceedance days, merge event gaps <=2 days; exact event catalogue is not reproducible from the published text alone",
        "is_official_hko_threshold": False,
        "is_main_heatwave_duration_definition": True,
    },
    {
        "id": "em_sd_ac_setpoint",
        "variable": "air-conditioned room temperature",
        "operator": "recommended range",
        "value": f"{min(THERMAL_CONFIG['setpoint_scenarios_c']):g}-{max(THERMAL_CONFIG['setpoint_scenarios_c']):g}",
        "unit": "degC",
        "source": "Hong Kong Electrical and Mechanical Services Department",
        "source_url": "https://ecib.emsd.gov.hk/index.php/en/energy-saving-advices/residential-sector",
        "role": "local setpoint scenario range; 25 degC is central scenario, not measured per building",
        "is_official_hko_threshold": False,
        "is_main_heatwave_duration_definition": False,
    },
    {
        "id": "hk_epd_summer_comfort_range",
        "variable": "air temperature in air-conditioned offices and public places during summer",
        "operator": "optimum reference range",
        "value": "24-26",
        "unit": "degC",
        "source": "Hong Kong Environmental Protection Department, Guidance Notes for the Management of Indoor Air Quality in Offices and Public Places",
        "source_url": "https://www.iaq.gov.hk/wp-content/uploads/2021/04/gn_officeandpublicplace_eng-2019.pdf",
        "role": "local summer comfort evidence supporting the 24, 25 and 26 degC setpoint scenarios; not used as an equipment hysteresis band",
        "is_official_hko_threshold": False,
        "is_main_heatwave_duration_definition": False,
    },
    {
        "id": "cibse_tm59_2017_bedroom",
        "variable": "bedroom operative temperature, 22:00-07:00",
        "operator": ">",
        "value": THERMAL_CONFIG["overheating_benchmark_c"],
        "unit": "degC",
        "source": "CIBSE TM59 (2017)",
        "source_url": "https://www.cibse.org/knowledge-research/knowledge-portal/technical-memorandum-59-design-methodology-for-the-assessment-of-overheating-risk-in-homes/",
        "role": "international overheating benchmark; not a Hong Kong health-safety limit",
        "is_official_hko_threshold": False,
        "is_main_heatwave_duration_definition": False,
    },
    {
        "id": "who_indoor_daytime_heat_safety_screen",
        "variable": "indoor air temperature during a heatwave",
        "operator": "<",
        "value": THERMAL_CONFIG["outage_safety_screen_c"],
        "unit": "degC",
        "source": "World Health Organization, Heatwaves: how to stay cool",
        "source_url": THERMAL_CONFIG["outage_safety_screen_source"],
        "role": "24-hour outage upper indoor heat-safety screen; nighttime overheating is reported separately against the 26 degC benchmark",
        "is_official_hko_threshold": False,
        "is_main_heatwave_duration_definition": False,
    },
]


METHOD_DECISIONS = [
    ("climate_period", "2016-2025 common window", "all available station-years; explicitly reproduce Li at HKO for 1980-2023", "avoids discarding older records merely because YLP began in 2015"),
    ("heatwave_definition", "HKO 33 degC threshold plus Li minimum 3-day duration", "Li percentile definition for the long-term trend; HKO fixed-threshold episodes for building stress tests", "keeps the long-term literature comparison and the locally interpretable building forcing explicit"),
    ("spatial_hazard", "nearest of 13 stations", "all documented daily-temperature stations; 250 m predictor grid; station-held-out model gate", "fine output is released only if it predicts unseen stations better than nearest-station assignment"),
    ("hazard_vs_vulnerability", "partly conflated in interpretation", "neighbourhood morphology only in hazard; envelope/AC/floor/orientation only in vulnerability", "prevents double counting"),
    ("population", "211 STPUG totals allocated by residential GFA", "2,342 official SSUG totals allocated by residential GFA inside official 10 m residential land-use cells", "raises the source-zone resolution, preserves represented SSUG totals, and is externally tested on all 540 major housing estates"),
    ("thermal_model", "single-zone 1R1C for stock ranking", "EnergyPlus reference cells plus held-out surrogate; 1R1C only for smoke/sensitivity", "aligns the main physics layer with current Hong Kong literature"),
    ("outage", "full-event loss of cooling", "6/24/full-event stress tests with resistance time, degree-hours, recovery and rebound", "consequence is estimated; outage probability is not"),
]


def sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def english_label(series: pd.Series) -> pd.Series:
    """Remove the bilingual suffix used in the released HK-BFETD labels."""
    return series.astype("string").str.split("_", n=1).str[0].str.strip()


def audit_building_stock(output: Path) -> dict:
    columns = [
        "BUILDINGSTRUCTUREID",
        "hkbfetd_main_class",
        "hkbfetd_sub_class",
        "floors_value",
        "height_value_m",
        "gfa_value_m2",
        "residential_gfa_estimate_m2",
        "nonresidential_gfa_estimate_m2",
        "hkbfetd_residential_share_published",
        "geometry",
    ]
    stock = gpd.read_parquet(BUILDING_STOCK, columns=columns)
    stock["main_class"] = english_label(stock["hkbfetd_main_class"])
    stock["sub_class"] = english_label(stock["hkbfetd_sub_class"])
    valid_geometry = stock.geometry.notna() & ~stock.geometry.is_empty & stock.geometry.is_valid
    target_classes = set(CONFIG["study_scope"]["modeled_main_classes"])
    target = stock["main_class"].isin(target_classes)
    classified = stock["main_class"].notna() & stock["main_class"].ne("Non-assessed")
    mixed = stock["main_class"].eq("Mixed-use")
    mixed_share = pd.to_numeric(stock["hkbfetd_residential_share_published"], errors="coerce")
    mixed_share_valid = ~mixed | mixed_share.between(0.0, 1.0, inclusive="both")
    floors = pd.to_numeric(stock["floors_value"], errors="coerce")
    height = pd.to_numeric(stock["height_value_m"], errors="coerce")
    gfa = pd.to_numeric(stock["gfa_value_m2"], errors="coerce")
    attributes_valid = floors.between(1, 120) & height.between(2, 500) & gfa.gt(0)
    component_valid = (
        ~mixed
        | (
            pd.to_numeric(stock["residential_gfa_estimate_m2"], errors="coerce").gt(0)
            & pd.to_numeric(stock["nonresidential_gfa_estimate_m2"], errors="coerce").gt(0)
        )
    )
    final_candidate = valid_geometry & target & mixed_share_valid & attributes_valid & component_valid

    published = CONFIG["study_scope"]["published_inventory_counts"]
    use_rows = []
    observed_counts = stock["main_class"].value_counts(dropna=False)
    for name in ["Residential", "Commercial", "Mixed-use", "Industrial", "Non-assessed"]:
        observed = int(observed_counts.get(name, 0))
        expected = int(published[name])
        use_rows.append(
            {
                "main_class": name,
                "observed_buildings": observed,
                "published_buildings": expected,
                "difference": observed - expected,
                "count_matches_published_inventory": observed == expected,
                "modeled_scope": name in target_classes,
            }
        )
    use_summary = pd.DataFrame(use_rows)
    use_summary.to_csv(output / "building_use_inventory.csv", index=False, encoding="utf-8-sig")

    steps = [
        ("released_repaired_stock", pd.Series(True, index=stock.index)),
        ("valid_geometry", valid_geometry),
        ("classified_buildings", valid_geometry & classified),
        ("target_scope_before_geometry_control", target),
        ("target_scope_with_valid_geometry", valid_geometry & target),
        ("valid_mixed_use_component_share", valid_geometry & target & mixed_share_valid),
        ("valid_simulation_attributes", valid_geometry & target & mixed_share_valid & attributes_valid),
        (
            "valid_residential_commercial_component_area",
            valid_geometry & target & mixed_share_valid & attributes_valid & component_valid,
        ),
        ("final_pre_weather_simulation_candidate", final_candidate),
    ]
    flow_rows = []
    previous_count = len(stock)
    for step, mask in steps:
        count = int(mask.sum())
        flow_rows.append(
            {
                "step": step,
                "building_count": count,
                "excluded_from_previous_step": previous_count - count,
            }
        )
        previous_count = count
    flow = pd.DataFrame(flow_rows)
    flow.to_csv(output / "building_sample_flow.csv", index=False, encoding="utf-8-sig")

    completeness_rows = []
    for field in columns[:-1]:
        values = stock.loc[target, field]
        completeness_rows.append(
            {
                "field": field,
                "target_scope_rows": int(len(values)),
                "non_missing_rows": int(values.notna().sum()),
                "completeness_pct": float(values.notna().mean() * 100.0),
            }
        )
    pd.DataFrame(completeness_rows).to_csv(
        output / "building_field_completeness.csv", index=False, encoding="utf-8-sig"
    )
    result = {
        "released_stock_rows": int(len(stock)),
        "published_inventory_rows": int(published["all_buildings"]),
        "inventory_count_matches": int(len(stock)) == int(published["all_buildings"]),
        "residential_plus_mixed_use_rows": int(stock["main_class"].isin(["Residential", "Mixed-use"]).sum()),
        "target_scope_before_qc_rows": int(target.sum()),
        "final_pre_weather_simulation_candidate_rows": int(final_candidate.sum()),
        "mixed_use_rows": int(mixed.sum()),
        "mixed_use_component_rows_valid": int((mixed & mixed_share_valid & component_valid).sum()),
        "population_assignment": "residential_component_only",
        "commercial_exposure_assignment": "employee_and_visitor_person_hours_not_resident_population",
    }
    (output / "building_stock_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def zip_inventory(path: Path, deep: bool) -> tuple[int | None, bool | None]:
    if not path.exists():
        return None, None
    with zipfile.ZipFile(path) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        crc_ok = archive.testzip() is None if deep else None
    return len(members), crc_ok


def md5(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def audit_wrf_heatwave_data(output: Path, deep: bool) -> dict:
    """Audit the Zuo et al. 1 km PRD CMIP6-WRF temperature product."""
    if not WRF_HEATWAVE_ARCHIVE.exists():
        raise FileNotFoundError(f"Missing WRF heatwave archive: {WRF_HEATWAVE_ARCHIVE}")
    archive_md5 = md5(WRF_HEATWAVE_ARCHIVE)
    if archive_md5.lower() != WRF_HEATWAVE_ZENODO_MD5:
        raise ValueError(
            "WRF heatwave archive MD5 differs from Zenodo 7206739: "
            f"{archive_md5} != {WRF_HEATWAVE_ZENODO_MD5}"
        )
    with zipfile.ZipFile(WRF_HEATWAVE_ARCHIVE) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        crc_ok = archive.testzip() is None if deep else None
        if not WRF_HEATWAVE_DATA.exists():
            archive.extractall(WRF_HEATWAVE_DIR)

    periods = {
        "SSP126": ["4049", "9099"],
        "SSP245": ["1120", "4049", "9099"],
        "SSP585": ["4049", "9099"],
    }
    rows: list[dict] = []
    grid_spacing_x_km = None
    grid_spacing_y_km = None
    for scenario, period_codes in periods.items():
        for period_code in period_codes:
            for statistic, folder in [("dailymax", "daily_max"), ("dailymin", "daily_min")]:
                path = (
                    WRF_HEATWAVE_DATA
                    / "daily_max_data"
                    / scenario
                    / folder
                    / f"{scenario.lower()}_t2_{period_code}_era5_corrected_{statistic}.nc"
                )
                if not path.exists():
                    raise FileNotFoundError(f"Missing canonical WRF file: {path}")
                with xr.open_dataset(path, engine="netcdf4") as dataset:
                    data_variables = list(dataset.data_vars)
                    if len(data_variables) != 1:
                        raise ValueError(f"Expected one temperature variable in {path}, found {data_variables}")
                    variable = data_variables[0]
                    values = dataset[variable]
                    if dict(dataset.sizes) != {"Time": 1220, "south_north": 162, "west_east": 213}:
                        raise ValueError(f"Unexpected WRF dimensions in {path}: {dict(dataset.sizes)}")
                    latitude = dataset["XLAT"].values
                    longitude = dataset["XLONG"].values
                    if grid_spacing_x_km is None:
                        center_y, center_x = latitude.shape[0] // 2, latitude.shape[1] // 2
                        latitude0 = float(latitude[center_y, center_x])
                        grid_spacing_x_km = float(
                            np.hypot(
                                (longitude[center_y, center_x + 1] - longitude[center_y, center_x])
                                * np.cos(np.deg2rad(latitude0))
                                * 111.32,
                                (latitude[center_y, center_x + 1] - latitude[center_y, center_x]) * 110.57,
                            )
                        )
                        grid_spacing_y_km = float(
                            np.hypot(
                                (longitude[center_y + 1, center_x] - longitude[center_y, center_x])
                                * np.cos(np.deg2rad(latitude0))
                                * 111.32,
                                (latitude[center_y + 1, center_x] - latitude[center_y, center_x]) * 110.57,
                            )
                        )
                    scan = values if deep else values.isel(Time=slice(None, None, 61), south_north=slice(None, None, 12), west_east=slice(None, None, 12))
                    minimum = float(scan.min(skipna=True).values)
                    maximum = float(scan.max(skipna=True).values)
                    rows.append(
                        {
                            "scenario": scenario.lower(),
                            "period_code": period_code,
                            "statistic": statistic,
                            "path": str(path.resolve()),
                            "bytes": path.stat().st_size,
                            "variable": variable,
                            "time_steps": int(dataset.sizes["Time"]),
                            "first_date": pd.Timestamp(dataset["Time"].values[0]).date().isoformat(),
                            "last_date": pd.Timestamp(dataset["Time"].values[-1]).date().isoformat(),
                            "south_north": int(dataset.sizes["south_north"]),
                            "west_east": int(dataset.sizes["west_east"]),
                            "latitude_min": float(np.nanmin(latitude)),
                            "latitude_max": float(np.nanmax(latitude)),
                            "longitude_min": float(np.nanmin(longitude)),
                            "longitude_max": float(np.nanmax(longitude)),
                            "netcdf_units_attribute": values.attrs.get("units"),
                            "documented_unit": "degC",
                            "minimum_scanned_c": minimum,
                            "maximum_scanned_c": maximum,
                            "scan_depth": "all_values" if deep else "systematic_sample",
                        }
                    )
    inventory = pd.DataFrame(rows)
    inventory.to_csv(output / "wrf_heatwave_netcdf_inventory.csv", index=False, encoding="utf-8-sig")

    duplicate_a = (
        WRF_HEATWAVE_DATA / "daily_max_data" / "SSP126" / "daily_max"
        / "ssp126_t2_4049_era5_corrected_dailymin.nc"
    )
    duplicate_b = (
        WRF_HEATWAVE_DATA / "daily_max_data" / "SSP126" / "daily_min"
        / "ssp126_t2_4049_era5_corrected_dailymin.nc"
    )
    duplicate_identical = bool(
        duplicate_a.exists()
        and duplicate_b.exists()
        and duplicate_a.stat().st_size == duplicate_b.stat().st_size
        and sha256(duplicate_a) == sha256(duplicate_b)
    )
    result = {
        "source_dataset": "Zuo et al. Pearl River Delta CMIP6-WRF heatwave data",
        "source_doi": "10.5281/zenodo.7206739",
        "article_doi": "10.1175/JAMC-D-22-0201.1",
        "license": "CC-BY-4.0",
        "archive_bytes": WRF_HEATWAVE_ARCHIVE.stat().st_size,
        "archive_md5": archive_md5,
        "zenodo_md5_match": True,
        "archive_files": len(members),
        "archive_crc_ok": crc_ok,
        "canonical_daily_temperature_files": len(inventory),
        "driver_model": "MPI-ESM1-2-HR",
        "scenarios": ["ssp126", "ssp245", "ssp585"],
        "baseline": "ssp245 2011-2020",
        "future_periods": ["2040-2049", "2090-2099"],
        "months": [6, 7, 8, 9],
        "grid_shape": [162, 213],
        "grid_spacing_km": [grid_spacing_y_km, grid_spacing_x_km],
        "temperature_unit_note": "NetCDF variables omit a units attribute; degree Celsius is established by the companion paper, CSV labels and physically plausible values.",
        "ssp126_duplicate_dailymin_identical": duplicate_identical,
        "absolute_temperature_policy": "Do not use unadjusted WRF absolute daily temperatures as building weather; 01 independently validates against HKO and uses the product as a spatial change signal.",
        "hourly_weather_available": False,
        "humidity_radiation_wind_available": False,
        "multimodel_ensemble_available": False,
    }
    (output / "wrf_heatwave_archive_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def selected_cmip6_point_stores(variable: str) -> pd.DataFrame:
    catalog = pd.read_csv(CMIP6_CATALOG)
    selected = catalog.loc[
        catalog["source_id"].isin(CMIP6_MODELS)
        & catalog["experiment_id"].isin(CMIP6_EXPERIMENTS)
        & catalog["member_id"].eq("r1i1p1f1")
        & catalog["table_id"].eq("day")
        & catalog["variable_id"].eq(variable)
    ].copy()
    selected = selected.sort_values("version").drop_duplicates(
        ["source_id", "experiment_id"], keep="last"
    )
    expected = pd.MultiIndex.from_product(
        [CMIP6_MODELS, CMIP6_EXPERIMENTS], names=["source_id", "experiment_id"]
    )
    observed = pd.MultiIndex.from_frame(selected[["source_id", "experiment_id"]])
    missing = expected.difference(observed)
    if len(missing):
        raise RuntimeError(f"CMIP6 catalog lacks {variable} stores: {missing.tolist()}")
    return selected.sort_values(["source_id", "experiment_id"]).reset_index(drop=True)


def cmip6_point_cache_path(model: str, experiment: str, variable: str) -> Path:
    return CMIP6_POINT_ROOT / f"{model}_{experiment}_r1i1p1f1_{variable}.csv"


def validate_cmip6_point_cache(path: Path, model: str, experiment: str, variable: str) -> dict:
    raw_column = f"{variable}_raw_c"
    frame = pd.read_csv(path)
    required = {
        "source_id", "experiment_id", "year", "month", "day", raw_column,
        "grid_latitude", "grid_longitude", "zstore", "version",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path.name} lacks required columns: {missing}")
    if not frame["source_id"].eq(model).all() or not frame["experiment_id"].eq(experiment).all():
        raise ValueError(f"{path.name} model or experiment labels do not match its filename")
    minimum_rows = 23 * 360 if experiment == "historical" else 40 * 360
    values = pd.to_numeric(frame[raw_column], errors="coerce")
    if len(frame) < minimum_rows or not np.isfinite(values).all():
        raise ValueError(f"{path.name} is incomplete or contains non-finite temperature values")
    duplicate_days = int(frame.duplicated(["year", "month", "day"]).sum())
    if duplicate_days:
        raise ValueError(f"{path.name} contains {duplicate_days} duplicate calendar days")
    return {
        "variable": variable,
        "source_id": model,
        "experiment_id": experiment,
        "path": str(path.resolve()),
        "status": "valid",
        "rows": int(len(frame)),
        "first_year": int(frame["year"].min()),
        "last_year": int(frame["year"].max()),
        "minimum_c": float(values.min()),
        "maximum_c": float(values.max()),
        "grid_latitude": float(frame["grid_latitude"].iloc[0]),
        "grid_longitude": float(frame["grid_longitude"].iloc[0]),
    }


def download_cmip6_point_store(row: pd.Series, variable: str) -> Path:
    import fsspec
    import gcsfs

    path = cmip6_point_cache_path(str(row.source_id), str(row.experiment_id), variable)
    partial_path = path.with_suffix(path.suffix + ".part")
    print(f"[CMIP6 {variable}] downloading {row.source_id}/{row.experiment_id}", flush=True)
    filesystem = gcsfs.GCSFileSystem(
        token="anon",
        requests_timeout=CMIP6_GCS_REQUEST_TIMEOUT_SECONDS,
    )
    mapper = filesystem.get_mapper(str(row.zstore).replace("gs://", ""))
    dataset = xr.open_zarr(mapper, consolidated=True)
    cached_dataset = None
    try:
        latitude_values = dataset["lat"].values.astype(float)
        longitude_values = dataset["lon"].values.astype(float)
        latitude_index = int(np.abs(latitude_values - CMIP6_REFERENCE[0]).argmin())
        longitude_difference = np.abs(
            (longitude_values - CMIP6_REFERENCE[1] + 180.0) % 360.0 - 180.0
        )
        longitude_index = int(longitude_difference.argmin())
        year_values = dataset["time"].dt.year.values.astype(int)
        if row.experiment_id == "historical":
            requested_periods = [(1992, 2014)]
        else:
            requested_periods = [(2041, 2060), (2081, 2100)]
        blocks = [
            (block_start, min(block_start + 4, period_end))
            for period_start, period_end in requested_periods
            for block_start in range(period_start, period_end + 1, 5)
        ]
        completed = pd.read_csv(partial_path) if partial_path.exists() else pd.DataFrame()
        raw_column = f"{variable}_raw_c"
        for block_number, (start_year, end_year) in enumerate(blocks, start=1):
            indices = np.flatnonzero(
                (year_values >= start_year) & (year_values <= end_year)
            )
            if len(indices) == 0:
                raise RuntimeError(
                    f"{row.zstore} has no {variable} data for {start_year}-{end_year}"
                )
            cached_block = (
                completed.loc[completed["year"].between(start_year, end_year)]
                if len(completed) and "year" in completed
                else pd.DataFrame()
            )
            cache_complete = bool(
                len(cached_block) == len(indices)
                and not cached_block.duplicated(["year", "month", "day"]).any()
                and raw_column in cached_block
                and np.isfinite(
                    pd.to_numeric(cached_block[raw_column], errors="coerce")
                ).all()
            )
            if cache_complete:
                print(
                    f"[CMIP6 {variable}] {row.source_id}/{row.experiment_id} "
                    f"{start_year}-{end_year} cached ({block_number}/{len(blocks)})",
                    flush=True,
                )
                continue
            for attempt in range(1, CMIP6_BLOCK_MAX_ATTEMPTS + 1):
                try:
                    source_dataset = dataset
                    if attempt > 1:
                        if cached_dataset is None:
                            cache_directory = (
                                CMIP6_POINT_ROOT
                                / ".zarr_chunk_cache"
                                / f"{row.source_id}_{row.experiment_id}_{variable}"
                            )
                            cache_directory.mkdir(parents=True, exist_ok=True)
                            cached_filesystem = fsspec.filesystem(
                                "simplecache",
                                target_protocol="gcs",
                                target_options={
                                    "token": "anon",
                                    "requests_timeout": max(
                                        600, CMIP6_GCS_REQUEST_TIMEOUT_SECONDS
                                    ),
                                },
                                cache_storage=str(cache_directory),
                            )
                            cached_mapper = cached_filesystem.get_mapper(
                                str(row.zstore).replace("gs://", "")
                            )
                            cached_dataset = xr.open_zarr(
                                cached_mapper, consolidated=True
                            )
                            print(
                                f"[CMIP6 {variable}] {row.source_id}/{row.experiment_id} "
                                "using local Zarr chunk cache after direct-read failure",
                                flush=True,
                            )
                        source_dataset = cached_dataset
                    if cached_dataset is not None and source_dataset is cached_dataset:
                        yearly_points = []
                        for requested_year in range(start_year, end_year + 1):
                            year_indices = np.flatnonzero(year_values == requested_year)
                            year_point = source_dataset[variable].isel(
                                time=slice(
                                    int(year_indices[0]), int(year_indices[-1]) + 1
                                ),
                                lat=latitude_index,
                                lon=longitude_index,
                            ).load()
                            if not np.isfinite(
                                year_point.values.astype(float)
                            ).all():
                                raise ValueError(
                                    f"cached remote {variable} year {requested_year} "
                                    "contains non-finite values"
                                )
                            yearly_points.append(year_point)
                            print(
                                f"[CMIP6 {variable}] {row.source_id}/{row.experiment_id} "
                                f"{requested_year} cached-chunk read complete",
                                flush=True,
                            )
                        point = xr.concat(yearly_points, dim="time")
                    else:
                        point = source_dataset[variable].isel(
                            time=slice(int(indices[0]), int(indices[-1]) + 1),
                            lat=latitude_index,
                            lon=longitude_index,
                        ).load()
                    if not np.isfinite(point.values.astype(float)).all():
                        raise ValueError(
                            f"remote {variable} block {start_year}-{end_year} "
                            "contains non-finite values"
                        )
                    break
                except Exception as exc:
                    if attempt == CMIP6_BLOCK_MAX_ATTEMPTS:
                        raise RuntimeError(
                            f"[CMIP6 {variable}] {row.source_id}/{row.experiment_id} "
                            f"{start_year}-{end_year} failed after "
                            f"{CMIP6_BLOCK_MAX_ATTEMPTS} attempts; the existing .part "
                            "cache was retained"
                        ) from exc
                    wait_seconds = 2**attempt
                    print(
                        f"[CMIP6 {variable}] {row.source_id}/{row.experiment_id} "
                        f"{start_year}-{end_year} request failed "
                        f"({type(exc).__name__}); retry {attempt + 1}/"
                        f"{CMIP6_BLOCK_MAX_ATTEMPTS} in {wait_seconds}s",
                        flush=True,
                    )
                    time.sleep(wait_seconds)
            units = str(point.attrs.get("units", "K")).lower()
            values = point.values.astype(float)
            if units in {"k", "kelvin"}:
                values -= 273.15
            elif units not in {"degc", "degree_celsius", "degrees_celsius", "c"}:
                raise RuntimeError(f"Unsupported {variable} units in {row.zstore}: {units}")
            block_frame = pd.DataFrame(
                {
                    "source_id": row.source_id,
                    "experiment_id": row.experiment_id,
                    "member_id": row.member_id,
                    "grid_label": row.grid_label,
                    "year": point["time"].dt.year.values.astype(int),
                    "month": point["time"].dt.month.values.astype(int),
                    "day": point["time"].dt.day.values.astype(int),
                    "sequence_index": 0,
                    f"{variable}_raw_c": values,
                    "grid_latitude": float(point["lat"].values),
                    "grid_longitude": float(point["lon"].values),
                    "calendar": str(point["time"].dt.calendar),
                    "zstore": row.zstore,
                    "version": row.version,
                }
            )
            completed = completed.loc[
                ~completed["year"].between(start_year, end_year)
            ] if len(completed) else completed
            completed = pd.concat([completed, block_frame], ignore_index=True).sort_values(
                ["year", "month", "day"]
            )
            completed["sequence_index"] = np.arange(len(completed), dtype=int)
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = partial_path.with_suffix(partial_path.suffix + ".tmp")
            completed.to_csv(temporary_path, index=False)
            temporary_path.replace(partial_path)
            print(
                f"[CMIP6 {variable}] {row.source_id}/{row.experiment_id} "
                f"{start_year}-{end_year} ready ({block_number}/{len(blocks)})",
                flush=True,
            )
    finally:
        dataset.close()
        if cached_dataset is not None:
            cached_dataset.close()
    frame = pd.read_csv(partial_path).sort_values(["year", "month", "day"]).reset_index(drop=True)
    frame["sequence_index"] = np.arange(len(frame), dtype=int)
    CMIP6_POINT_ROOT.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary_path, index=False)
    validate_cmip6_point_cache(
        temporary_path, str(row.source_id), str(row.experiment_id), variable
    )
    temporary_path.replace(path)
    partial_path.unlink()
    print(f"[CMIP6 {variable}] ready {row.source_id}/{row.experiment_id}: {len(frame):,} days", flush=True)
    return path


def prepare_cmip6_point_temperatures(output: Path, offline: bool) -> dict:
    rows: list[dict] = []
    missing_downloads: list[tuple[pd.Series, str]] = []
    for variable in ["tasmax", "tasmin"]:
        for _, store in selected_cmip6_point_stores(variable).iterrows():
            path = cmip6_point_cache_path(str(store.source_id), str(store.experiment_id), variable)
            if path.exists():
                try:
                    rows.append(
                        validate_cmip6_point_cache(
                            path, str(store.source_id), str(store.experiment_id), variable
                        )
                    )
                    continue
                except ValueError as exc:
                    if offline:
                        invalid = pd.read_csv(path)
                        rows.append(
                            {
                                "variable": variable,
                                "source_id": str(store.source_id),
                                "experiment_id": str(store.experiment_id),
                                "path": str(path.resolve()),
                                "status": "invalid_cache",
                                "rows": int(len(invalid)),
                                "first_year": int(invalid["year"].min()),
                                "last_year": int(invalid["year"].max()),
                                "validation_error": str(exc),
                            }
                        )
                        continue
                    partial_path = path.with_suffix(path.suffix + ".part")
                    if not partial_path.exists():
                        path.replace(partial_path)
                        print(
                            f"[CMIP6 {variable}] adopted invalid final cache as .part; "
                            "only invalid blocks will be downloaded again",
                            flush=True,
                        )
                    missing_downloads.append((store, variable))
            elif offline:
                partial_path = path.with_suffix(path.suffix + ".part")
                partial = pd.read_csv(partial_path) if partial_path.exists() else pd.DataFrame()
                rows.append(
                    {
                        "variable": variable,
                        "source_id": str(store.source_id),
                        "experiment_id": str(store.experiment_id),
                        "path": str(path.resolve()),
                        "status": "partial_resume_available" if len(partial) else "missing_offline",
                        "rows": int(len(partial)) if len(partial) else None,
                        "first_year": int(partial["year"].min()) if len(partial) else None,
                        "last_year": int(partial["year"].max()) if len(partial) else None,
                    }
                )
            else:
                missing_downloads.append((store, variable))
    if missing_downloads:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(download_cmip6_point_store, store, variable): (store, variable)
                for store, variable in missing_downloads
            }
            for future in concurrent.futures.as_completed(futures):
                store, variable = futures[future]
                path = future.result()
                rows.append(
                    validate_cmip6_point_cache(
                        path, str(store.source_id), str(store.experiment_id), variable
                    )
                )
    audit = pd.DataFrame(rows).sort_values(["variable", "source_id", "experiment_id"])
    audit.to_csv(output / "cmip6_point_temperature_input_audit.csv", index=False, encoding="utf-8-sig")
    expected = len(CMIP6_MODELS) * len(CMIP6_EXPERIMENTS) * 2
    valid = int(audit["status"].eq("valid").sum())
    return {
        "variables": ["tasmax", "tasmin"],
        "models": CMIP6_MODELS,
        "experiments": CMIP6_EXPERIMENTS,
        "expected_caches": expected,
        "valid_caches": valid,
        "missing_caches": int(expected - valid),
        "all_daily_maximum_and_minimum_caches_ready": valid == expected,
        "download_policy": "00 downloads only missing point caches; --offline only audits frozen caches",
    }


def audit_hkust_brick_hierarchy(output: Path) -> dict:
    import rdflib

    ttl_member = "All Data/Clean Dataset/HKUST_Meter_Metadata.ttl"
    if not HKUST_ARCHIVE.exists():
        return {"status": "missing", "archive": str(HKUST_ARCHIVE.resolve())}
    with zipfile.ZipFile(HKUST_ARCHIVE) as archive:
        names = set(archive.namelist())
        if ttl_member not in names:
            raise FileNotFoundError(f"HKUST archive lacks {ttl_member}")
        graph = rdflib.Graph()
        graph.parse(data=archive.read(ttl_member).decode("utf-8"), format="turtle")
        normalized_name = lambda name: re.sub(
            r"^GUI_NO\.", "", Path(name).name, flags=re.IGNORECASE
        )
        clean_files = {
            normalized_name(name): Path(name).parent.name
            for name in names
            if "Clean Dataset/Resappled data/" in name and name.lower().endswith(".xlsx")
        }
        raw_files = {
            normalized_name(name)
            for name in names
            if "Raw Dataset/Time-series data/" in name and name.lower().endswith(".xlsx")
        }

    brick = rdflib.Namespace("https://brickschema.org/schema/Brick#")
    extension = rdflib.Namespace(
        "https://www.HKUST_Electric_Meter.com/schema/BrickExtension#"
    )
    reference = rdflib.Namespace("https://brickschema.org/schema/Brick/ref#")
    meters = sorted(
        set(graph.subjects(rdflib.RDF.type, brick.Electrical_Meter)), key=str
    )
    buildings = set(graph.subjects(rdflib.RDF.type, brick.Building))
    parent_of: dict[rdflib.term.Node, set[rdflib.term.Node]] = {}
    for parent, child in graph.subject_objects(brick.hasPart):
        parent_of.setdefault(child, set()).add(parent)

    def local_name(node: rdflib.term.Node) -> str:
        value = str(node)
        return value.rsplit("#", 1)[-1].rsplit("/", 1)[-1]

    def ancestors(nodes: set[rdflib.term.Node]) -> set[rdflib.term.Node]:
        seen = set(nodes)
        queue = list(nodes)
        while queue:
            node = queue.pop()
            for parent in parent_of.get(node, set()):
                if parent not in seen:
                    seen.add(parent)
                    queue.append(parent)
        return seen

    rows = []
    upstream_by_meter: dict[rdflib.term.Node, set[rdflib.term.Node]] = {}
    for meter in meters:
        usage_nodes = list(graph.objects(meter, extension.usageType))
        usage_values = sorted(
            {
                str(value)
                for node in usage_nodes
                for value in graph.objects(node, brick.value)
            }
        )
        reference_nodes = list(graph.objects(meter, reference.hasExternalReference))
        files = sorted(
            {
                str(value)
                for node in reference_nodes
                for value in graph.objects(node, reference.hasTimeseriesData)
            }
        )
        metered_entities = set(graph.subjects(brick.isMeteredBy, meter))
        containing_entities = set(graph.subjects(brick.hasPart, meter))
        entity_ancestors = ancestors(metered_entities)
        building_ancestors = sorted(buildings.intersection(entity_ancestors), key=str)
        upstream = set()
        for entity in entity_ancestors.difference(metered_entities):
            upstream.update(graph.objects(entity, brick.isMeteredBy))
        upstream.discard(meter)
        upstream_by_meter[meter] = upstream.intersection(meters)
        entity_types = sorted(
            {
                local_name(entity_type)
                for entity in metered_entities
                for entity_type in graph.objects(entity, rdflib.RDF.type)
                if str(entity_type).startswith(str(brick))
                or str(entity_type).startswith(str(extension))
            }
        )
        rows.append(
            {
                "meter_id": local_name(meter),
                "timeseries_files": "|".join(files),
                "timeseries_file_count": len(files),
                "clean_timeseries_file_present": bool(
                    files and all(Path(name).name in clean_files for name in files)
                ),
                "raw_timeseries_file_present": bool(
                    files and all(Path(name).name in raw_files for name in files)
                ),
                "clean_interval_minutes": "|".join(
                    sorted(
                        clean_files[Path(name).name].removeprefix("T")
                        for name in files
                        if Path(name).name in clean_files
                    )
                ),
                "unit": "|".join(
                    sorted(local_name(value) for value in graph.objects(meter, brick.hasUnit))
                ),
                "usage_type": "|".join(usage_values),
                "metered_entity_count": len(metered_entities),
                "metered_entity_types": "|".join(entity_types),
                "metered_entities": "|".join(
                    sorted(local_name(value) for value in metered_entities)
                ),
                "containing_electrical_entity_count": len(containing_entities),
                "containing_electrical_entities": "|".join(
                    sorted(local_name(value) for value in containing_entities)
                ),
                "building_count": len(building_ancestors),
                "buildings": "|".join(local_name(value) for value in building_ancestors),
                "upstream_meter_count": len(upstream_by_meter[meter]),
                "upstream_meters": "|".join(
                    sorted(local_name(value) for value in upstream_by_meter[meter])
                ),
            }
        )
    audit = pd.DataFrame(rows)
    downstream_counts = {meter: 0 for meter in meters}
    for upstream in upstream_by_meter.values():
        for meter in upstream:
            downstream_counts[meter] += 1
    audit["downstream_meter_count"] = audit["meter_id"].map(
        {local_name(meter): downstream_counts[meter] for meter in meters}
    )
    audit["candidate_top_level_for_nonoverlapping_cut"] = (
        audit["upstream_meter_count"].eq(0)
        & audit["building_count"].eq(1)
        & audit["timeseries_file_count"].eq(1)
        & audit["clean_timeseries_file_present"]
    )
    audit.to_csv(
        output / "hkust_meter_hierarchy_audit.csv", index=False, encoding="utf-8-sig"
    )
    result = {
        "status": "audited",
        "ttl_member": ttl_member,
        "graph_triples": int(len(graph)),
        "electrical_meters": int(len(audit)),
        "buildings": int(len(buildings)),
        "meters_with_one_timeseries_file": int(audit["timeseries_file_count"].eq(1).sum()),
        "meters_with_clean_timeseries_file": int(
            audit["clean_timeseries_file_present"].sum()
        ),
        "meters_with_raw_timeseries_file": int(
            audit["raw_timeseries_file_present"].sum()
        ),
        "meters_missing_both_clean_and_raw_timeseries": int(
            (~audit["clean_timeseries_file_present"] & ~audit["raw_timeseries_file_present"]).sum()
        ),
        "meters_mapped_to_exactly_one_building": int(audit["building_count"].eq(1).sum()),
        "meters_with_upstream_meter": int(audit["upstream_meter_count"].gt(0).sum()),
        "meters_with_downstream_meter": int(audit["downstream_meter_count"].gt(0).sum()),
        "top_level_single_building_meter_candidates": int(
            audit["candidate_top_level_for_nonoverlapping_cut"].sum()
        ),
        "aggregation_rule": "Do not add all 1,432 meters. Candidate top-level meters still require time-series overlap and energy-conservation checks before a non-overlapping cut is accepted; retain use-specific submeters only for end-use calibration.",
    }
    (output / "hkust_meter_hierarchy_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def audit_external_validation_data(output: Path, deep: bool) -> pd.DataFrame:
    datasets = [
        {
            "dataset_id": "HKUST",
            "path": EXTERNAL_RAW / "hkust" / "All_Data.zip",
            "geography": "Hong Kong",
            "building_scope": "campus; academic, research, staff quarters and student halls",
            "resolution": "raw plus 60-minute and daily resampled files",
            "variables": "meter readings and Brick/TTL meter-building hierarchy",
            "evidence_role": "local non-residential and hall load-shape calibration/holdout validation",
            "status_if_present": "downloaded_hierarchy_audited_pending_timeseries_quality_and_nonoverlap_selection",
            "source_url": "https://doi.org/10.5061/dryad.k3j9kd5h6",
        },
        {
            "dataset_id": "CU_BEMS",
            "path": EXTERNAL_RAW / "cubems" / "CU-BEMS_v6.zip",
            "geography": "Bangkok, Thailand",
            "building_scope": "seven-floor university office building",
            "resolution": "1 minute; 2018-07 to 2019-12 with floor-specific gaps",
            "variables": "AC, lighting, plug loads, indoor temperature/RH and outdoor conditions",
            "evidence_role": "hot-humid coupled load-temperature-moisture structural validation",
            "status_if_present": "downloaded_ready_for_synchronization",
            "source_url": "https://doi.org/10.1038/s41597-020-00582-3",
        },
        {
            "dataset_id": "SMETER30",
            "path": EXTERNAL_RAW / "smeter30" / "856978_data_and_documentation.zip",
            "geography": "United Kingdom",
            "building_scope": "30 occupied homes",
            "resolution": "30 minute; approximately one year per home",
            "variables": "electricity, gas, indoor temperature/RH, weather and dwelling surveys",
            "evidence_role": "residential behavior and parameter-identification structural test only",
            "status_if_present": "downloaded_ready_for_nested_archive_audit",
            "source_url": "https://reshare.ukdataservice.ac.uk/856978/",
        },
        {
            "dataset_id": "NIST_NZERTF",
            "path": EXTERNAL_RAW / "nist",
            "geography": "Maryland, USA",
            "building_scope": "instrumented detached net-zero house",
            "resolution": "hourly extracts from minute-level monitoring; two years",
            "variables": "HVAC, indoor and outdoor environmental variables",
            "evidence_role": "residential HVAC-indoor-temperature dynamic structure test only",
            "status_if_present": "downloaded_ready_for_synchronization",
            "source_url": "https://pages.nist.gov/netzero/data.html",
        },
        {
            "dataset_id": "Singapore_BCA",
            "path": EXTERNAL_RAW / "bca_singapore" / "bca-bebr-data.xlsx",
            "geography": "Singapore",
            "building_scope": "commercial building benchmark disclosure",
            "resolution": "annual 2022-2024",
            "variables": "EUI, GFA, conditioned area, AC system and use fractions",
            "evidence_role": "hot-humid commercial annual-EUI plausibility check only",
            "status_if_present": "downloaded_ready_for_benchmark_extraction",
            "source_url": "https://www1.bca.gov.sg/sustainability/legislation-on-environmental-sustainability-for-buildings/existing-buildings/building-energy-benchmarking-report/",
        },
        {
            "dataset_id": "PolyU_full_campus",
            "path": POLYU_FULL_CAMPUS,
            "geography": "Hong Kong",
            "building_scope": "45 phase-specific campus building units constructed from 538 mapped meters; mappings are not certified physical-building ground truth",
            "resolution": "hourly; 2025-07-09 to 2026-07-09; 8760 timestamps per building unit",
            "variables": "observed allocated electricity, complete-input electricity, meter counts, coverage and quality status",
            "evidence_role": "local operational load-shape calibration and building/time holdout validation after quality and mapping filters",
            "status_if_present": "downloaded_integrity_verified_pending_weather_join_and_holdout_selection",
            "source_url": "local_authorized_PolyU_full_campus_data",
        },
        {
            "dataset_id": "IDEAL_metadata_only",
            "path": EXTERNAL_RAW / "ideal" / "14096993.zip",
            "geography": "United Kingdom",
            "building_scope": "metadata record only; raw household sensors not downloaded",
            "resolution": "not_applicable",
            "variables": "citation metadata",
            "evidence_role": "not used while raw files are unavailable",
            "status_if_present": "metadata_only_not_analysis_ready",
            "source_url": "https://doi.org/10.7488/ds/2836",
        },
    ]
    rows = []
    for item in datasets:
        path = item["path"]
        exists = path.exists()
        file_count = None
        archive_crc_ok = None
        size_bytes = None
        if exists and path.is_file():
            size_bytes = path.stat().st_size
            if path.suffix.lower() == ".zip":
                file_count, archive_crc_ok = zip_inventory(path, deep)
        elif exists:
            files = [candidate for candidate in path.rglob("*") if candidate.is_file()]
            file_count = len(files)
            size_bytes = sum(candidate.stat().st_size for candidate in files)
            exists = file_count > 0
        rows.append(
            {
                **{key: value for key, value in item.items() if key != "path" and key != "status_if_present"},
                "path": str(path.resolve()),
                "exists": exists,
                "size_bytes": size_bytes,
                "file_count": file_count,
                "archive_crc_ok": archive_crc_ok,
                "audit_depth": "full_archive_crc" if deep and path.suffix.lower() == ".zip" else "inventory",
                "current_status": item["status_if_present"] if exists else "pending_download",
            }
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(output / "external_validation_data_audit.csv", index=False, encoding="utf-8-sig")
    return audit


def frame_time_profile(
    dataset_id: str,
    file_id: str,
    frame: pd.DataFrame,
    time_column: str,
    dayfirst: bool = False,
) -> dict:
    timestamps = pd.to_datetime(frame[time_column], errors="coerce", dayfirst=dayfirst, utc=True)
    valid_time = timestamps.dropna().sort_values()
    intervals = valid_time.diff().dropna().dt.total_seconds().div(60)
    values = frame.drop(columns=[time_column], errors="ignore")
    return {
        "dataset_id": dataset_id,
        "file_id": file_id,
        "rows": int(len(frame)),
        "start_utc": valid_time.min().isoformat() if len(valid_time) else None,
        "end_utc": valid_time.max().isoformat() if len(valid_time) else None,
        "median_interval_minutes": float(intervals.median()) if len(intervals) else None,
        "variable_columns": int(len(values.columns)),
        "value_missing_pct": float(values.isna().to_numpy().mean() * 100.0) if values.size else None,
    }


def profile_external_validation_data(output: Path) -> pd.DataFrame:
    rows: list[dict] = []
    hkust_path = EXTERNAL_RAW / "hkust" / "All_Data.zip"
    if hkust_path.exists():
        with zipfile.ZipFile(hkust_path) as archive:
            names = [item.filename for item in archive.infolist() if not item.is_dir()]
        for resolution in ["T15", "T30", "T60", "T1440"]:
            members = [
                name for name in names
                if f"/Resappled data/{resolution}/" in name and name.lower().endswith(".xlsx")
            ]
            rows.append(
                {
                    "dataset_id": "HKUST",
                    "file_id": resolution,
                    "rows": None,
                    "start_utc": None,
                    "end_utc": None,
                    "median_interval_minutes": float(resolution[1:]),
                    "variable_columns": None,
                    "value_missing_pct": None,
                    "source_files": len(members),
                    "profile_note": "meter-level time and missingness require Brick hierarchy audit before aggregation",
                }
            )

    cubems_path = EXTERNAL_RAW / "cubems" / "CU-BEMS_v6.zip"
    if cubems_path.exists():
        with zipfile.ZipFile(cubems_path) as archive:
            for name in sorted(archive.namelist()):
                if name.lower().endswith(".csv"):
                    frame = pd.read_csv(archive.open(name))
                    row = frame_time_profile("CU_BEMS", name, frame, "Date")
                    row["source_files"] = 1
                    row["profile_note"] = "one-minute floor-year file"
                    rows.append(row)

    smeter_path = EXTERNAL_RAW / "smeter30" / "856978_data_and_documentation.zip"
    if smeter_path.exists():
        with zipfile.ZipFile(smeter_path) as outer:
            inner_name = next(name for name in outer.namelist() if name.endswith("30homes.zip"))
            with zipfile.ZipFile(io.BytesIO(outer.read(inner_name))) as inner:
                for name in sorted(inner.namelist()):
                    if name.lower().endswith(".csv"):
                        frame = pd.read_csv(inner.open(name))
                        time_column = frame.columns[0]
                        row = frame_time_profile("SMETER30", name, frame, time_column, dayfirst=True)
                        row["source_files"] = 1
                        row["profile_note"] = "occupied UK home; structural validation only"
                        rows.append(row)

    nist_dir = EXTERNAL_RAW / "nist"
    if nist_dir.exists():
        for path in sorted(nist_dir.glob("y[12]_*.csv")):
            if "Metadata" in path.name:
                continue
            frame = pd.read_csv(path)
            row = frame_time_profile("NIST_NZERTF", path.name, frame, "Timestamp")
            row["source_files"] = 1
            row["profile_note"] = "US detached test house; structural validation only"
            rows.append(row)

    bca_path = EXTERNAL_RAW / "bca_singapore" / "bca-bebr-data.xlsx"
    if bca_path.exists():
        workbook = pd.ExcelFile(bca_path)
        dataset = pd.read_excel(workbook, sheet_name="Dataset")
        rows.append(
            {
                "dataset_id": "Singapore_BCA",
                "file_id": "Dataset",
                "rows": int(len(dataset)),
                "start_utc": None,
                "end_utc": None,
                "median_interval_minutes": None,
                "variable_columns": int(len(dataset.columns)),
                "value_missing_pct": float(dataset.isna().to_numpy().mean() * 100.0),
                "source_files": 1,
                "profile_note": "annual commercial EUI benchmark; sheets=" + "|".join(workbook.sheet_names),
            }
        )

    if POLYU_FULL_CAMPUS.exists():
        with zipfile.ZipFile(POLYU_FULL_CAMPUS) as archive:
            parquet_member = next(
                name for name in archive.namelist() if name.endswith("/data/building_hourly.parquet")
            )
            manifest_member = next(
                name for name in archive.namelist() if name.endswith("/metadata/dataset_manifest.json")
            )
            frame = pd.read_parquet(io.BytesIO(archive.read(parquet_member)))
            manifest = json.loads(archive.read(manifest_member).decode("utf-8"))
        timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
        numeric_values = frame[["energy_kwh", "complete_input_energy_kwh", "meter_coverage_pct"]]
        rows.append(
            {
                "dataset_id": "PolyU_full_campus",
                "file_id": POLYU_FULL_CAMPUS.name,
                "rows": int(len(frame)),
                "start_utc": timestamps.min().isoformat() if timestamps.notna().any() else None,
                "end_utc": timestamps.max().isoformat() if timestamps.notna().any() else None,
                "median_interval_minutes": 60.0,
                "variable_columns": int(len(frame.columns)),
                "value_missing_pct": float(numeric_values.isna().to_numpy().mean() * 100.0),
                "source_files": int(frame["building_id"].nunique()),
                "profile_note": (
                    f"{manifest['building_units']} phase-specific units; complete-input hours "
                    f"{manifest['quality_counts']['complete_inputs']:,}/{manifest['building_hour_rows']:,}; "
                    "use complete_inputs for primary temporal calibration and retain buildings/phases as grouped holdouts; mapping and Wh-to-kWh convention require sensitivity checks"
                ),
            }
        )

    profile = pd.DataFrame(rows)
    profile.to_csv(output / "external_validation_file_profiles.csv", index=False, encoding="utf-8-sig")
    return profile


def write_model_registries(output: Path) -> None:
    prototypes = pd.DataFrame(CONFIG["prototypes"])
    prototypes.to_csv(output / "prototype_registry.csv", index=False, encoding="utf-8-sig")
    indicator_rows = []
    for scope, indicators in CONFIG["resilience"].items():
        if scope.endswith("_indicators"):
            indicator_rows.extend({"indicator_scope": scope, "indicator": value} for value in indicators)
    pd.DataFrame(indicator_rows).to_csv(
        output / "resilience_indicator_registry.csv", index=False, encoding="utf-8-sig"
    )
    map_rows = [
        {"map_panel": panel, "field": field}
        for panel, fields in CONFIG["map_output_fields"].items()
        for field in fields
    ]
    pd.DataFrame(map_rows).to_csv(output / "map_field_registry.csv", index=False, encoding="utf-8-sig")


def request_json(url: str, payload: dict | None = None, attempts: int = 4) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {
        "User-Agent": "HKBFETD-paper-reproducibility/0.3",
        "Accept": "application/json",
    }
    if body is not None:
        headers["Content-Type"] = "application/json"
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, data=body, headers=headers, method="POST" if body is not None else "GET")
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _numeric_weather_value(value: object) -> float:
    text = str(value).strip().replace(",", "")
    text = re.sub(r"[#*]+$", "", text).strip()
    if text in {"", "-", "--", "Trace", "Tr", "N/A", "None"}:
        return np.nan
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text)
    return float(match.group()) if match else np.nan


def _reference_heatwave_catalog(daily: pd.DataFrame) -> pd.DataFrame:
    """Identify the HKO very-hot-day stress suite by cumulative threshold exceedance."""
    threshold = float(HAZARD_CONFIG["hko_very_hot_day_c"])
    minimum_days = int(HAZARD_CONFIG["li_minimum_consecutive_days"])
    frame = daily.sort_values("date").reset_index(drop=True).copy()
    hot = frame["tmax_c"].ge(threshold)
    continuous = frame["date"].diff().eq(pd.Timedelta(days=1))
    groups = ((~hot) | (hot & ~continuous)).cumsum()
    records = []
    for _, block in frame.loc[hot].groupby(groups.loc[hot]):
        if len(block) < minimum_days:
            continue
        excess = block["tmax_c"].sub(threshold)
        records.append(
            {
                "start_date": block["date"].min(),
                "end_date": block["date"].max(),
                "duration_days": int(len(block)),
                "peak_tmax_c": float(block["tmax_c"].max()),
                "mean_tmax_c": float(block["tmax_c"].mean()),
                "cumulative_excess_c_days": float(excess.sum()),
            }
        )
    events = pd.DataFrame(records)
    if events.empty:
        raise ValueError("No reference heatwave events were identified")
    # Cumulative exceedance already combines duration and intensity in a
    # directly interpretable physical unit (degree-days).  A second weighted
    # composite would double-count the same quantities and add an arbitrary
    # coefficient without changing the selected top-three events.
    events["severity_score"] = events["cumulative_excess_c_days"]
    events["severity_definition"] = (
        f"sum_over_event(max(daily_Tmax-{threshold:g}_C,0)); unit=C_day"
    )
    events = events.sort_values(
        ["severity_score", "peak_tmax_c", "start_date"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    events["event_id"] = [
        f"HW_{row.start_date:%Y%m%d}_{row.end_date:%Y%m%d}"
        for row in events.itertuples()
    ]
    events["selected_for_hourly_simulation"] = False
    events.loc[: REFERENCE_EVENT_COUNT - 1, "selected_for_hourly_simulation"] = True
    return events


def _nasa_utc_query_dates(start_date: pd.Timestamp, end_date: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_hkt = pd.Timestamp(start_date).tz_localize("Asia/Hong_Kong")
    end_exclusive_hkt = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).tz_localize(
        "Asia/Hong_Kong"
    )
    query_start = start_hkt.tz_convert("UTC").normalize().tz_localize(None)
    query_end = (
        end_exclusive_hkt.tz_convert("UTC") - pd.Timedelta(nanoseconds=1)
    ).normalize().tz_localize(None)
    return query_start, query_end


def _load_nasa_reference_window(
    event_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    offline: bool,
) -> tuple[pd.DataFrame, Path, dict]:
    query_start, query_end = _nasa_utc_query_dates(start_date, end_date)
    query = {
        "parameters": ",".join(NASA_POWER_PARAMETERS),
        "community": "RE",
        "longitude": REFERENCE_WEATHER_POINT[1],
        "latitude": REFERENCE_WEATHER_POINT[0],
        "start": query_start.strftime("%Y%m%d"),
        "end": query_end.strftime("%Y%m%d"),
        "format": "JSON",
        "time-standard": "UTC",
    }
    url = f"{NASA_POWER_URL}?{urllib.parse.urlencode(query)}"
    cache = (
        ROOT
        / "data"
        / "weather"
        / "raw"
        / f"nasa_power_{event_id}_{query_start:%Y%m%d}_{query_end:%Y%m%d}_utc.json"
    )
    if cache.exists():
        payload = json.loads(cache.read_text(encoding="utf-8"))
    else:
        if offline:
            raise FileNotFoundError(f"Missing frozen NASA POWER cache: {cache}")
        print(f"[reference weather] downloading {event_id} UTC hourly fields", flush=True)
        payload = request_json(url)
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix(cache.suffix + ".part")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(cache)
    time_standard = payload.get("header", {}).get("time_standard")
    if time_standard != "UTC":
        raise ValueError(f"NASA POWER {event_id} time standard is {time_standard!r}, expected 'UTC'")
    parameters = payload.get("properties", {}).get("parameter", {})
    missing_parameters = set(NASA_POWER_PARAMETERS) - set(parameters)
    if missing_parameters:
        raise ValueError(f"NASA POWER {event_id} lacks {sorted(missing_parameters)}")
    keys = sorted(parameters["T2M"])
    frame = pd.DataFrame(
        {
            "timestamp_hkt": pd.to_datetime(keys, format="%Y%m%d%H", utc=True).tz_convert(
                "Asia/Hong_Kong"
            )
        }
    )
    for parameter in NASA_POWER_PARAMETERS:
        frame[parameter] = [
            _numeric_weather_value(parameters[parameter].get(key)) for key in keys
        ]
        frame[parameter] = frame[parameter].replace(-999.0, np.nan)
    start_hkt = pd.Timestamp(start_date).tz_localize("Asia/Hong_Kong")
    end_exclusive_hkt = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).tz_localize(
        "Asia/Hong_Kong"
    )
    frame = frame.loc[
        frame["timestamp_hkt"].ge(start_hkt)
        & frame["timestamp_hkt"].lt(end_exclusive_hkt)
    ].reset_index(drop=True)
    expected_hours = int((end_exclusive_hkt - start_hkt) / pd.Timedelta(hours=1))
    if len(frame) != expected_hours:
        raise ValueError(f"NASA POWER {event_id} has {len(frame)} HKT hours, expected {expected_hours}")
    if not frame["timestamp_hkt"].diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise ValueError(f"NASA POWER {event_id} is not continuous after UTC-to-HKT conversion")
    if frame[list(NASA_POWER_PARAMETERS)].isna().any().any():
        raise ValueError(f"NASA POWER {event_id} contains missing core hourly values")
    return frame, cache, payload


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
        delta = (lo + hi) / 2.0
        if np.mean(np.clip(x + delta, 0.0, 100.0)) < target_mean:
            lo = delta
        else:
            hi = delta
    return pd.Series(np.clip(x + (lo + hi) / 2.0, 0.0, 100.0), index=values.index)


def _calendar_day_reference(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy()
    frame["month_day"] = frame["date"].dt.strftime("%m-%d")
    return (
        frame.groupby("month_day", as_index=False)[["tmin_c", "tmean_c", "tmax_c", "rh_mean_pct"]]
        .median()
        .rename(
            columns={
                "tmin_c": "typical_tmin_c",
                "tmean_c": "typical_tmean_c",
                "tmax_c": "typical_tmax_c",
                "rh_mean_pct": "typical_rh_mean_pct",
            }
        )
    )


def _constrain_reference_window(
    hourly: pd.DataFrame, daily: pd.DataFrame, typical: pd.DataFrame
) -> pd.DataFrame:
    frame = hourly.copy()
    frame["date"] = frame["timestamp_hkt"].dt.tz_localize(None).dt.normalize()
    targets = daily[["date", "tmin_c", "tmean_c", "tmax_c", "rh_mean_pct"]].drop_duplicates(
        "date"
    )
    frame = frame.merge(targets, on="date", how="left", validate="many_to_one")
    frame["month_day"] = frame["date"].dt.strftime("%m-%d")
    frame = frame.merge(typical, on="month_day", how="left", validate="many_to_one")
    for column in [
        "temperature_c",
        "relative_humidity_pct",
        "counterfactual_temperature_c",
        "counterfactual_relative_humidity_pct",
    ]:
        frame[column] = np.nan
    for _, index in frame.groupby("date").groups.items():
        idx = list(index)
        target = frame.loc[idx[0]]
        frame.loc[idx, "temperature_c"] = _monotone_daily_temperature(
            frame.loc[idx, "T2M"], target["tmin_c"], target["tmean_c"], target["tmax_c"]
        ).to_numpy()
        frame.loc[idx, "relative_humidity_pct"] = _bounded_humidity_shift(
            frame.loc[idx, "RH2M"], target["rh_mean_pct"]
        ).to_numpy()
        frame.loc[idx, "counterfactual_temperature_c"] = _monotone_daily_temperature(
            frame.loc[idx, "T2M"],
            target["typical_tmin_c"],
            target["typical_tmean_c"],
            target["typical_tmax_c"],
        ).to_numpy()
        frame.loc[idx, "counterfactual_relative_humidity_pct"] = _bounded_humidity_shift(
            frame.loc[idx, "RH2M"], target["typical_rh_mean_pct"]
        ).to_numpy()
    return frame


def prepare_reference_event_weather(offline: bool) -> dict:
    """Build the canonical hourly stress suite used by the EnergyPlus experiment."""
    daily = pd.read_csv(REFERENCE_DAILY_WEATHER, parse_dates=["date"])
    required = {"date", "tmin_c", "tmean_c", "tmax_c", "rh_mean_pct"}
    if required - set(daily):
        raise ValueError(f"Reference HKO daily file lacks {sorted(required - set(daily))}")
    if daily["date"].duplicated().any() or not daily["date"].is_monotonic_increasing:
        raise ValueError("Reference HKO daily dates must be unique and sorted")
    events = _reference_heatwave_catalog(daily)
    selected = events.loc[events["selected_for_hourly_simulation"]].copy()
    typical = _calendar_day_reference(daily)
    hourly_windows = []
    cache_paths = []
    last_payload = None
    for event in selected.itertuples():
        start = event.start_date - pd.Timedelta(days=REFERENCE_EVENT_SPINUP_DAYS)
        end = event.end_date + pd.Timedelta(days=REFERENCE_EVENT_RECOVERY_DAYS)
        raw, cache, payload = _load_nasa_reference_window(
            event.event_id, start, end, offline
        )
        corrected = _constrain_reference_window(raw, daily, typical)
        corrected["event_id"] = event.event_id
        corrected["event_phase"] = np.select(
            [
                corrected["date"].between(event.start_date, event.end_date),
                corrected["date"].lt(event.start_date),
            ],
            ["heatwave", "spinup"],
            default="recovery",
        )
        hourly_windows.append(corrected)
        cache_paths.append(cache)
        last_payload = payload
    hourly = pd.concat(hourly_windows, ignore_index=True)
    export_columns = [
        "timestamp_hkt", "date", "event_id", "event_phase", "temperature_c",
        "relative_humidity_pct", "counterfactual_temperature_c",
        "counterfactual_relative_humidity_pct", "ALLSKY_SFC_SW_DWN", "WS10M",
        "PRECTOTCORR", "tmin_c", "tmean_c", "tmax_c", "rh_mean_pct",
        "typical_tmin_c", "typical_tmean_c", "typical_tmax_c", "typical_rh_mean_pct",
    ]
    daily_check = hourly.groupby(["event_id", "date"]).agg(
        hours=("temperature_c", "size"),
        corrected_min=("temperature_c", "min"),
        corrected_mean=("temperature_c", "mean"),
        corrected_max=("temperature_c", "max"),
        corrected_rh_mean=("relative_humidity_pct", "mean"),
        hko_min=("tmin_c", "first"),
        hko_mean=("tmean_c", "first"),
        hko_max=("tmax_c", "first"),
        hko_rh_mean=("rh_mean_pct", "first"),
    )
    recovery_hours = hourly.loc[hourly["event_phase"].eq("recovery")].groupby("event_id").size()
    validation = {
        "hko_expected_days": int((daily["date"].max() - daily["date"].min()).days + 1),
        "hko_observed_days": int(len(daily)),
        "hko_temperature_complete_days": int(daily[["tmin_c", "tmean_c", "tmax_c"]].notna().all(axis=1).sum()),
        "heatwave_events_found": int(len(events)),
        "selected_events": int(len(selected)),
        "selected_hourly_rows": int(len(hourly)),
        "nasa_time_standard": "UTC",
        "spinup_hours_per_event": int(24 * REFERENCE_EVENT_SPINUP_DAYS),
        "recovery_hours_per_event": {str(key): int(value) for key, value in recovery_hours.items()},
        "daily_windows_with_24_hours": int(daily_check["hours"].eq(24).sum()),
        "daily_windows_total": int(len(daily_check)),
        "hourly_core_missing_values": int(hourly[export_columns].isna().sum().sum()),
        "temperature_min_mae_c": float((daily_check["corrected_min"] - daily_check["hko_min"]).abs().mean()),
        "temperature_mean_mae_c": float((daily_check["corrected_mean"] - daily_check["hko_mean"]).abs().mean()),
        "temperature_max_mae_c": float((daily_check["corrected_max"] - daily_check["hko_max"]).abs().mean()),
        "relative_humidity_mean_mae_pct": float((daily_check["corrected_rh_mean"] - daily_check["hko_rh_mean"]).abs().mean()),
    }
    validation["gates"] = {
        "nasa_utc_converted_to_hkt": bool(str(hourly["timestamp_hkt"].dt.tz) == "Asia/Hong_Kong"),
        "hko_daily_completeness_ge_0_99": bool(validation["hko_temperature_complete_days"] / validation["hko_expected_days"] >= 0.99),
        "at_least_three_heatwave_events": bool(validation["heatwave_events_found"] >= REFERENCE_EVENT_COUNT),
        "all_selected_days_have_24_hours": bool(validation["daily_windows_with_24_hours"] == validation["daily_windows_total"]),
        "no_missing_hourly_core_fields": bool(validation["hourly_core_missing_values"] == 0),
        "daily_temperature_targets_mae_le_0_05_c": bool(max(validation["temperature_min_mae_c"], validation["temperature_mean_mae_c"], validation["temperature_max_mae_c"]) <= 0.05),
        "daily_rh_target_mae_le_0_05_pct": bool(validation["relative_humidity_mean_mae_pct"] <= 0.05),
        "two_full_recovery_days_per_event": bool((recovery_hours == 48).all()),
    }
    validation["gates"]["reference_event_weather_ready"] = bool(all(validation["gates"].values()))
    if not validation["gates"]["reference_event_weather_ready"]:
        raise ValueError(f"Reference event weather validation failed: {validation['gates']}")
    REFERENCE_WEATHER_OUTPUT.mkdir(parents=True, exist_ok=True)
    writes = {
        REFERENCE_WEATHER_OUTPUT / "weather_event_catalog.csv": events,
        REFERENCE_WEATHER_OUTPUT / "heatwave_hourly_bias_corrected.csv": hourly[export_columns],
        ROOT / "intermediate" / "source_derived" / "weather" / "nasa_power_hourly_selected_events.csv": hourly[["timestamp_hkt", "event_id", *NASA_POWER_PARAMETERS]],
    }
    for path, frame in writes.items():
        temporary = path.with_suffix(path.suffix + ".part")
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    (REFERENCE_WEATHER_OUTPUT / "weather_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    units = {
        key: value.get("units")
        for key, value in (last_payload or {}).get("parameters", {}).items()
        if key in NASA_POWER_PARAMETERS
    }
    metadata = {
        "weather_product_name": "HKO-daily-constrained NASA POWER hourly heatwave forcing",
        "analysis_period": [int(daily["date"].dt.year.min()), int(daily["date"].dt.year.max())],
        "heatwave_operational_definition": {
            "threshold_tmax_c": float(HAZARD_CONFIG["hko_very_hot_day_c"]),
            "minimum_consecutive_days": int(HAZARD_CONFIG["li_minimum_consecutive_days"]),
            "event_selection": "three events with the largest cumulative Tmax exceedance above 33 C and complete multivariable hourly forcing",
        },
        "event_window": {
            "spinup_days": REFERENCE_EVENT_SPINUP_DAYS,
            "recovery_days": REFERENCE_EVENT_RECOVERY_DAYS,
            "recovery_rationale": "supports a 24 h outage followed by a complete 24 h recovery assessment even when outage onset occurs on the final event day",
        },
        "station": {"code": "KP", "name": "King's Park", "nasa_power_point": list(REFERENCE_WEATHER_POINT)},
        "provenance": {
            "HKO": "Observed daily values determine event selection and exact daily temperature and relative-humidity targets.",
            "NASA_POWER": "UTC hourly shapes are converted explicitly to Hong Kong time; shortwave radiation and wind are retained for EnergyPlus weather conversion.",
            "counterfactual": "Calendar-day median HKO daily targets over 2016-2025 are mapped to the identical hourly shape.",
            "paired_design": "Calendar time, solar radiation, wind and precipitation are identical within each heatwave-counterfactual pair.",
        },
        "spatial_role": "Common reference forcing is used to isolate building-stock susceptibility; the independently validated 1-km hazard surface represents intra-urban exposure variation.",
        "raw_file_sha256": {path.name: sha256(path) for path in [REFERENCE_DAILY_WEATHER, *cache_paths]},
        "nasa_parameter_units": units,
    }
    (REFERENCE_WEATHER_OUTPUT / "weather_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return validation


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "HKBFETD-paper-reproducibility/0.3"},
    )
    with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as stream:
        downloaded = 0
        while block := response.read(1024 * 1024):
            stream.write(block)
            downloaded += len(block)
            if downloaded % (10 * 1024 * 1024) < len(block):
                print(f"[download] {destination.name}: {downloaded / 1024 / 1024:.1f} MB", flush=True)
    temporary.replace(destination)


def prepare_polyu_hourly_weather(offline: bool) -> dict:
    """Freeze POWER hourly shape and anchor every day to official King's Park extrema."""
    url = (
        "https://power.larc.nasa.gov/api/temporal/hourly/point?"
        "parameters=T2M%2CRH2M%2CALLSKY_SFC_SW_DWN%2CWS10M%2CPRECTOTCORR&"
        "community=RE&longitude=114.1796&latitude=22.305&start=20250709&end=20260708&format=JSON"
    )
    if not POLYU_WEATHER_RAW.exists():
        if offline:
            return {
                "status": "missing_hourly_weather_cache",
                "required_for": "PolyU hourly temperature-load calibration",
                "missing": str(POLYU_WEATHER_RAW.resolve()),
            }
        print("[PolyU weather] downloading NASA POWER hourly shape", flush=True)
        last_error = None
        for attempt in range(4):
            try:
                download_file(url, POLYU_WEATHER_RAW)
                break
            except Exception as error:
                last_error = error
                if attempt == 3:
                    raise
                print(f"[PolyU weather] retry {attempt + 2}/4 after {type(error).__name__}", flush=True)
                time.sleep(2 ** attempt)
        if last_error is not None and not POLYU_WEATHER_RAW.exists():
            raise last_error
    payload = json.loads(POLYU_WEATHER_RAW.read_text(encoding="utf-8"))
    parameters = payload.get("properties", {}).get("parameter", {})
    required = {"T2M", "RH2M", "ALLSKY_SFC_SW_DWN", "WS10M", "PRECTOTCORR"}
    if required - set(parameters):
        raise ValueError(f"NASA POWER PolyU cache lacks {sorted(required - set(parameters))}")
    keys = sorted(parameters["T2M"])
    if len(keys) != 8760:
        raise ValueError(f"NASA POWER PolyU cache has {len(keys)} hours, expected 8760")
    weather = pd.DataFrame(
        {
            "timestamp_hkt": pd.to_datetime(keys, format="%Y%m%d%H").tz_localize("Asia/Hong_Kong"),
            "temperature_nasa_raw_c": [parameters["T2M"][key] for key in keys],
            "relative_humidity_pct": [parameters["RH2M"][key] for key in keys],
            "solar_radiation_w_m2": [parameters["ALLSKY_SFC_SW_DWN"][key] for key in keys],
            "wind_speed_m_s": [parameters["WS10M"][key] for key in keys],
            "precipitation_mm_h": [parameters["PRECTOTCORR"][key] for key in keys],
        }
    )
    value_columns = [column for column in weather if column != "timestamp_hkt"]
    weather[value_columns] = weather[value_columns].replace(-999.0, np.nan)
    missing_before = weather[value_columns].isna().sum().to_dict()
    required_for_temperature_load = ["temperature_nasa_raw_c", "relative_humidity_pct"]
    if weather[required_for_temperature_load].isna().any().any():
        raise ValueError(
            "NASA POWER temperature-load calibration variables contain missing values: "
            f"{weather[required_for_temperature_load].isna().sum().to_dict()}"
        )
    weather["solar_radiation_imputed"] = weather["solar_radiation_w_m2"].isna()
    month_hour_median = weather.groupby(
        [weather["timestamp_hkt"].dt.month, weather["timestamp_hkt"].dt.hour]
    )["solar_radiation_w_m2"].transform("median")
    weather["solar_radiation_w_m2"] = weather["solar_radiation_w_m2"].fillna(month_hour_median)
    hour_median = weather.groupby(weather["timestamp_hkt"].dt.hour)["solar_radiation_w_m2"].transform("median")
    weather["solar_radiation_w_m2"] = weather["solar_radiation_w_m2"].fillna(hour_median)
    if weather["solar_radiation_w_m2"].isna().any():
        raise ValueError("NASA POWER solar-radiation month-hour imputation is incomplete")
    weather["date"] = weather["timestamp_hkt"].dt.tz_localize(None).dt.normalize()
    hko_frames = []
    for data_type, output_name in [("CLMMAXT", "hko_tmax_c"), ("CLMMINT", "hko_tmin_c")]:
        source = ROOT / "data" / "weather" / "hko_open_data_daily" / f"{data_type}_KP.json"
        data = json.loads(source.read_text(encoding="utf-8"))["data"]
        frame = pd.DataFrame(data, columns=["year", "month", "day", output_name, "completeness"])
        frame["date"] = pd.to_datetime(frame[["year", "month", "day"]], errors="coerce")
        frame[output_name] = pd.to_numeric(frame[output_name], errors="coerce")
        hko_frames.append(frame[["date", output_name]])
    hko_daily = hko_frames[0].merge(hko_frames[1], on="date", how="inner", validate="one_to_one")
    weather = weather.merge(hko_daily, on="date", how="left", validate="many_to_one")
    raw_daily = weather.groupby("date")["temperature_nasa_raw_c"].agg(nasa_tmin_c="min", nasa_tmax_c="max")
    weather = weather.merge(raw_daily, on="date", how="left", validate="many_to_one")
    raw_range = (weather["nasa_tmax_c"] - weather["nasa_tmin_c"]).replace(0.0, np.nan)
    fraction = (weather["temperature_nasa_raw_c"] - weather["nasa_tmin_c"]) / raw_range
    anchored = weather["hko_tmin_c"] + fraction * (weather["hko_tmax_c"] - weather["hko_tmin_c"])
    weather["temperature_c"] = anchored.fillna(weather["temperature_nasa_raw_c"])
    start = pd.Timestamp("2025-07-09 00:00", tz="Asia/Hong_Kong")
    end = pd.Timestamp("2026-07-09 00:00", tz="Asia/Hong_Kong")
    weather = weather.loc[weather["timestamp_hkt"].ge(start) & weather["timestamp_hkt"].lt(end)].copy()
    if weather["timestamp_hkt"].duplicated().any() or weather.empty:
        raise ValueError("NOAA King's Park hourly weather is empty or has duplicate timestamps")
    weather.to_parquet(POLYU_HOURLY_WEATHER, index=False)
    expected = int((end - start).total_seconds() / 3600)
    anchored_days = int(weather.loc[weather["hko_tmin_c"].notna() & weather["hko_tmax_c"].notna(), "date"].nunique())
    return {
        "status": "ready" if len(weather) == expected and anchored_days >= 360 else "partial",
        "path": str(POLYU_HOURLY_WEATHER.resolve()),
        "rows": int(len(weather)),
        "expected_hours": expected,
        "coverage_pct": float(100 * len(weather) / expected),
        "start_hkt": weather["timestamp_hkt"].min().isoformat(),
        "end_hkt": weather["timestamp_hkt"].max().isoformat(),
        "hko_anchored_days": anchored_days,
        "raw_missing_hours": {key: int(value) for key, value in missing_before.items()},
        "solar_radiation_imputed_hours": int(weather["solar_radiation_imputed"].sum()),
        "solar_imputation_rule": "same-month same-hour median, falling back to all-month same-hour median when an entire month is unavailable; exclude in primary observed-weather calibration and retain only for EnergyPlus sensitivity",
        "source": "NASA POWER hourly diurnal shape anchored day-by-day to HKO King's Park CLMMINT/CLMMAXT",
        "source_url": url,
    }


def hko_number_is_present(value: object) -> bool:
    return re.search(r"[-+]?\d+(?:\.\d+)?", str(value).replace(",", "")) is not None


def validate_hko_cis_payload(payload: dict, year: int) -> dict:
    elements = payload.get("stn", {}).get("data", [])
    maximum = next((element for element in elements if element.get("code") == "MAX_TEMP"), None)
    if maximum is None:
        raise ValueError(f"MAX_TEMP is absent from HKO CIS daily file for {year}")
    rows_by_day = {}
    for row in maximum.get("dayData", []):
        if not row:
            continue
        match = re.search(r"\d+", str(row[0]))
        if match:
            rows_by_day[int(match.group())] = row
    expected_days = 366 if calendar.isleap(year) else 365
    observed_days = 0
    numeric_days = 0
    for month in range(1, 13):
        for day in range(1, calendar.monthrange(year, month)[1] + 1):
            row = rows_by_day.get(day)
            if row is None or len(row) <= month:
                raise ValueError(f"HKO CIS {year} is missing {year:04d}-{month:02d}-{day:02d}")
            observed_days += 1
            numeric_days += int(hko_number_is_present(row[month]))
    if observed_days != expected_days or numeric_days != expected_days:
        raise ValueError(
            f"HKO CIS {year} is incomplete: expected={expected_days}, observed={observed_days}, numeric={numeric_days}"
        )
    return {"rows": observed_days, "nonmissing_rows": numeric_days}


def validate_hko_open_data_payload(payload: dict, station: str, data_type: str) -> dict:
    fields = payload.get("fields", [])
    data = payload.get("data", [])
    if not isinstance(fields, list) or not fields:
        raise ValueError(f"HKO Open Data {station}/{data_type} has no fields")
    if not isinstance(data, list) or not data:
        raise ValueError(f"HKO Open Data {station}/{data_type} has no rows")
    normalized_fields = [str(field).strip().lower() for field in fields]
    value_candidates = [
        index
        for index, field in enumerate(normalized_fields)
        if "value" in field or "temperature" in field
    ]
    value_index = value_candidates[-1] if value_candidates else 3
    valid_dates = 0
    nonmissing_values = 0
    excluded_noncalendar_rows = []
    for row in data:
        try:
            datetime(int(row[0]), int(row[1]), int(row[2]))
            valid_dates += 1
        except (TypeError, ValueError, IndexError):
            excluded_noncalendar_rows.append(row)
            continue
        if len(row) > value_index:
            nonmissing_values += int(hko_number_is_present(row[value_index]))
    unexpected_noncalendar_rows = [
        row
        for row in excluded_noncalendar_rows
        if not (
            len(row) > value_index
            and str(row[0]).strip() == "1900"
            and str(row[1]).strip() == "2"
            and str(row[2]).strip() == "29"
            and not hko_number_is_present(row[value_index])
        )
    ]
    if unexpected_noncalendar_rows:
        raise ValueError(
            f"HKO Open Data {station}/{data_type} contains unexpected non-calendar rows: "
            f"{unexpected_noncalendar_rows[:3]}"
        )
    return {
        "raw_rows": len(data),
        "rows": valid_dates,
        "nonmissing_rows": nonmissing_values,
        "excluded_noncalendar_rows": len(excluded_noncalendar_rows),
    }


def validate_hko_json_file(
    path: Path,
    source_kind: str,
    *,
    year: int | None = None,
    station: str | None = None,
    data_type: str | None = None,
) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot parse JSON file {path}: {exc}") from exc
    if source_kind == "cis":
        if year is None:
            raise ValueError("year is required for HKO CIS validation")
        summary = validate_hko_cis_payload(payload, year)
    elif source_kind == "open_data":
        if station is None or data_type is None:
            raise ValueError("station and data_type are required for HKO Open Data validation")
        summary = validate_hko_open_data_payload(payload, station, data_type)
    else:
        raise ValueError(f"Unknown HKO source kind: {source_kind}")
    summary["size_bytes"] = path.stat().st_size
    return summary


def adopt_previous_hko_partial(destination: Path) -> Path:
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists():
        return partial
    legacy = [
        path
        for path in destination.parent.glob(f"{destination.stem}_*{destination.suffix}")
        if path.is_file() and path != destination
    ]
    if legacy:
        largest = max(legacy, key=lambda path: path.stat().st_size)
        largest.replace(partial)
        print(f"  adopted previous partial file: {largest.name} -> {partial.name}", flush=True)
    return partial


def download_hko_json(
    url: str,
    destination: Path,
    label: str,
    source_kind: str,
    *,
    year: int | None = None,
    station: str | None = None,
    data_type: str | None = None,
) -> tuple[dict, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    validation = {"year": year, "station": station, "data_type": data_type}
    if destination.exists():
        summary = validate_hko_json_file(destination, source_kind, **validation)
        print(f"{label} cache valid ({summary['size_bytes']:,} bytes)", flush=True)
        return summary, "validated_cache"

    partial = adopt_previous_hko_partial(destination)
    if partial.exists():
        try:
            summary = validate_hko_json_file(partial, source_kind, **validation)
        except ValueError:
            print(f"{label} resume from {partial.stat().st_size:,} bytes", flush=True)
        else:
            partial.replace(destination)
            print(f"{label} recovered complete partial ({summary['size_bytes']:,} bytes)", flush=True)
            return summary, "recovered_complete_partial"
    else:
        print(f"{label} download from 0 bytes", flush=True)

    curl = shutil.which("curl.exe") or shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl is required for resumable HKO downloads")
    command = [
        curl,
        "--fail",
        "--location",
        "--show-error",
        "--progress-bar",
        "--retry",
        "10",
        "--retry-all-errors",
        "--retry-delay",
        "3",
        "--retry-max-time",
        "1800",
        "--connect-timeout",
        "45",
        "--speed-limit",
        "1",
        "--speed-time",
        "90",
        "--max-time",
        "600",
        "--user-agent",
        "HKBFETD-paper-reproducibility/0.3",
        "--continue-at",
        "-",
    ]
    if os.name == "nt":
        command.append("--ssl-no-revoke")
    command.extend(["--output", str(partial), url])
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} download stopped with curl exit code {completed.returncode}. "
            f"The partial file was kept at {partial}; rerun 00 to resume."
        )
    summary = validate_hko_json_file(partial, source_kind, **validation)
    partial.replace(destination)
    print(f"{label} complete ({summary['size_bytes']:,} bytes)", flush=True)
    return summary, "downloaded_and_validated"


def prepare_hko_weather(offline: bool, download_cis_archive: bool = False) -> pd.DataFrame:
    rows = []
    print("Preparing canonical HKO all-year weather caches in 00 (download -> validate -> freeze)", flush=True)
    tasks = [(station, data_type) for station in HKO_DAILY_STATIONS for data_type in HKO_DAILY_DATA_TYPES]
    for index, (station, data_type) in enumerate(tasks, start=1):
        destination = HKO_OPEN_DATA_CACHE / f"{data_type}_{station}.json"
        params = urllib.parse.urlencode({"dataType": data_type, "station": station, "rformat": "json"})
        url = f"{HKO_OPEN_DATA_API}?{params}"
        label = f"[HKO Open Data {index:02d}/{len(tasks):02d}] {station}/{data_type}"
        if offline and not destination.exists():
            raise FileNotFoundError(f"Missing HKO Open Data cache: {destination}. Run 00 without --offline first.")
        if offline:
            summary = validate_hko_json_file(
                destination,
                "open_data",
                station=station,
                data_type=data_type,
            )
            transport = "validated_cache"
            print(f"{label} cache valid ({summary['size_bytes']:,} bytes)", flush=True)
        else:
            summary, transport = download_hko_json(
                url,
                destination,
                label,
                "open_data",
                station=station,
                data_type=data_type,
            )
        rows.append(
            {
                "source": "hko_open_data",
                "year": None,
                "station_code": station,
                "data_type": data_type,
                "source_url": url,
                "cache_file": str(destination),
                "status": transport,
                **summary,
            }
        )

    years = list(HKO_CIS_YEARS)
    if download_cis_archive:
        print("Preparing optional HKO CIS annual archive", flush=True)
    else:
        print("Auditing available HKO CIS files only (annual archive is optional)", flush=True)
    for index, year in enumerate(years, start=1):
        destination = HKO_CIS_CACHE / f"daily_{year}.json"
        label = f"[HKO CIS {index:02d}/{len(years):02d}] {year}"
        if destination.exists():
            summary = validate_hko_json_file(destination, "cis", year=year)
            transport = "validated_cache"
            print(f"{label} cache valid ({summary['size_bytes']:,} bytes)", flush=True)
        elif download_cis_archive:
            if offline:
                raise FileNotFoundError(
                    f"Missing optional HKO CIS archive file: {destination}. "
                    "Remove --offline or omit --download-cis-archive."
                )
            summary, transport = download_hko_json(
                HKO_CIS_YEAR_URL.format(year=year),
                destination,
                label,
                "cis",
                year=year,
            )
        else:
            continue
        rows.append(
            {
                "source": "hko_cis",
                "year": year,
                "station_code": "HKO",
                "data_type": "MAX_TEMP",
                "source_url": HKO_CIS_YEAR_URL.format(year=year),
                "cache_file": str(destination),
                "status": transport,
                **summary,
            }
        )
    return pd.DataFrame(rows)


def nested(mapping: dict, *keys: str) -> str:
    value: object = mapping
    for key in keys:
        if not isinstance(value, dict):
            return ""
        value = value.get(key, "")
    return "" if value is None else str(value).strip()


def prepare_als_buildings(refresh: bool, offline: bool) -> dict:
    if ALS_BUILDINGS.exists() and not refresh:
        return json.loads(ALS_BUILDINGS_METADATA.read_text(encoding="utf-8"))
    if not ALS_ZIP.exists():
        if offline:
            raise FileNotFoundError(f"Offline run requires {ALS_ZIP}")
        download_file(ALS_URL, ALS_ZIP)

    rows: list[dict] = []
    source_features = 0
    with zipfile.ZipFile(ALS_ZIP) as archive:
        district_files = [name for name in archive.namelist() if "_3d_" not in name.lower()]
        for name in district_files:
            collection = json.load(archive.open(name))
            features = collection.get("features", [])
            source_features += len(features)
            for feature in features:
                properties = feature.get("properties", {})
                premises = properties.get("Address", {}).get("PremisesAddress", {})
                csuid = nested(premises, "BuildingCsuInformation", "CsuId")
                if not csuid:
                    continue
                eng = premises.get("EngPremisesAddress", {})
                chi = premises.get("ChiPremisesAddress", {})
                rows.append(
                    {
                        "building_csuid": csuid,
                        "easting_m": pd.to_numeric(properties.get("Easting"), errors="coerce"),
                        "northing_m": pd.to_numeric(properties.get("Northing"), errors="coerce"),
                        "building_name_en": nested(eng, "BuildingName"),
                        "building_name_tc": nested(chi, "BuildingName"),
                        "estate_name_en": nested(eng, "EngEstate", "EstateName"),
                        "estate_name_tc": nested(chi, "ChiEstate", "EstateName"),
                        "block_descriptor_en": nested(eng, "EngBlock", "BlockDescriptor"),
                        "block_no_en": nested(eng, "EngBlock", "BlockNo"),
                        "block_descriptor_tc": nested(chi, "ChiBlock", "BlockDescriptor"),
                        "block_no_tc": nested(chi, "ChiBlock", "BlockNo"),
                        "street_name_en": nested(eng, "EngStreet", "StreetName"),
                        "street_name_tc": nested(chi, "ChiStreet", "StreetName"),
                        "building_no_from": nested(eng, "EngStreet", "BuildingNoFrom"),
                        "geoaddress": nested(premises, "GeoAddress"),
                        "source_file": name,
                    }
                )
    frame = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    if frame.empty or frame["building_csuid"].isna().any():
        raise ValueError("ALS building-address extraction produced invalid records")
    rows_with_coordinates = frame[["easting_m", "northing_m"]].notna().all(axis=1)
    ALS_BUILDINGS.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(ALS_BUILDINGS, index=False)
    metadata = {
        "retrieved_archive_last_modified": "2026-08-13",
        "source_features_2d": source_features,
        "address_rows": int(len(frame)),
        "unique_building_csuids": int(frame["building_csuid"].nunique()),
        "rows_with_hk80_coordinates": int(rows_with_coordinates.sum()),
        "rows_without_hk80_coordinates": int((~rows_with_coordinates).sum()),
        "rows_with_english_building_or_estate_name": int(frame[["building_name_en", "estate_name_en"]].ne("").any(axis=1).sum()),
        "rows_with_chinese_building_or_estate_name": int(frame[["building_name_tc", "estate_name_tc"]].ne("").any(axis=1).sum()),
        "archive_sha256": sha256(ALS_ZIP),
        "archive_size_bytes": ALS_ZIP.stat().st_size,
        "source": ALS_URL,
        "scope": "2D premises addresses only; the flat-level public-rental-housing 3D file is excluded to avoid duplicate unit records",
    }
    ALS_BUILDINGS_METADATA.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def prepare_hma_boundaries(refresh: bool, offline: bool) -> dict:
    if HMA_BOUNDARIES.exists() and not refresh:
        return json.loads(HMA_BOUNDARIES_METADATA.read_text(encoding="utf-8"))
    if offline:
        raise FileNotFoundError(f"Offline run requires {HMA_BOUNDARIES}")
    query = urllib.parse.urlencode(
        {
            "where": "1=1",
            "outFields": "hma,hma_eng,hma_chi",
            "returnGeometry": "true",
            "outSR": 2326,
            "resultRecordCount": 1000,
            "orderByFields": "OBJECTID",
            "f": "geojson",
        }
    )
    collection = request_json(f"{CSDI_HMA_QUERY}?{query}")
    frame = gpd.GeoDataFrame.from_features(collection["features"], crs="EPSG:2326")
    if len(frame) != 173 or frame["hma"].nunique() != 173 or frame.geometry.isna().any():
        raise ValueError(
            f"Unexpected HMA boundary coverage: rows={len(frame)}, unique={frame['hma'].nunique()}"
        )
    HMA_BOUNDARIES.parent.mkdir(parents=True, exist_ok=True)
    frame[["hma", "hma_eng", "hma_chi", "geometry"]].sort_values("hma").to_parquet(
        HMA_BOUNDARIES, index=False
    )
    metadata = {
        "housing_market_areas": 173,
        "crs": "EPSG:2326",
        "all_geometries_present": True,
        "source": CSDI_HMA_QUERY.rsplit("/query", 1)[0],
    }
    HMA_BOUNDARIES_METADATA.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def prepare_ssug_boundaries(refresh: bool, offline: bool) -> dict:
    """Cross-check the official SSUG table against the official polygon API."""
    if SSUG_BOUNDARIES.exists() and SSUG_BOUNDARIES_METADATA.exists() and not refresh:
        return json.loads(SSUG_BOUNDARIES_METADATA.read_text(encoding="utf-8"))
    if offline:
        raise FileNotFoundError(f"Offline run requires {SSUG_BOUNDARIES}")

    records: list[dict] = []
    offset = 0
    while True:
        query = urllib.parse.urlencode(
            {
                "where": "1=1",
                "outFields": "ssbg,t_pop,dh,oq",
                "returnGeometry": "true",
                "outSR": 2326,
                "resultOffset": offset,
                "resultRecordCount": 3000,
                "orderByFields": "OBJECTID",
                "f": "geojson",
            }
        )
        page = request_json(f"{CSDI_SSUG_QUERY}?{query}")
        features = page.get("features", [])
        records.extend(features)
        if not page.get("exceededTransferLimit", False):
            break
        offset += len(features)
    boundaries = gpd.GeoDataFrame.from_features(records, crs="EPSG:2326")
    boundaries["ssbg"] = boundaries["ssbg"].astype(str)
    for column in ("t_pop", "dh", "oq"):
        boundaries[column] = pd.to_numeric(boundaries[column], errors="coerce")
    if (
        len(boundaries) != 2342
        or boundaries["ssbg"].nunique() != 2342
        or boundaries.geometry.isna().any()
        or boundaries.geometry.is_empty.any()
    ):
        raise ValueError(
            "Unexpected SSUG boundary coverage: "
            f"rows={len(boundaries)}, unique={boundaries['ssbg'].nunique()}"
        )

    official = pd.read_csv(
        SSUG_SOURCE_TABLE,
        skiprows=2,
        dtype=str,
        encoding="big5hkscs",
        low_memory=False,
    )
    required = {"ssbg", "t_pop", "dh", "oq"}
    if not required.issubset(official.columns):
        raise ValueError(f"SSUG table is missing columns: {sorted(required - set(official))}")
    official = official.loc[official["ssbg"].isin(set(boundaries["ssbg"]))].copy()
    official["t_pop"] = pd.to_numeric(official["t_pop"], errors="raise")
    crosscheck = boundaries[["ssbg", "t_pop"]].merge(
        official[["ssbg", "t_pop"]],
        on="ssbg",
        how="outer",
        validate="one_to_one",
        suffixes=("_api", "_csv"),
    )
    if len(crosscheck) != 2342 or not crosscheck["t_pop_api"].eq(crosscheck["t_pop_csv"]).all():
        raise ValueError("Official SSUG CSV and polygon API population values do not match")
    total_population = int(crosscheck["t_pop_api"].sum())
    if total_population != 7_411_945:
        raise ValueError(f"Unexpected SSUG land-population total: {total_population}")

    boundaries = boundaries.rename(
        columns={"t_pop": "official_population", "dh": "domestic_households", "oq": "occupied_quarters"}
    )
    SSUG_BOUNDARIES.parent.mkdir(parents=True, exist_ok=True)
    boundaries[
        ["ssbg", "official_population", "domestic_households", "occupied_quarters", "geometry"]
    ].sort_values("ssbg").to_parquet(SSUG_BOUNDARIES, index=False)
    metadata = {
        "small_subunit_groups": 2342,
        "official_land_population": total_population,
        "crs": "EPSG:2326",
        "csv_api_population_crosscheck_passed": True,
        "source_csv": "https://www.census2021.gov.hk/doc/SSUG_21C.zip",
        "source_polygons": CSDI_SSUG_QUERY.rsplit("/query", 1)[0],
        "definition": "Each SSUG contains at least 400 persons; LSUG aggregation begins at 1,000 persons.",
    }
    SSUG_BOUNDARIES_METADATA.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def prepare_land_utilization(offline: bool) -> dict:
    """Verify the official 2021 10 m land-utilization raster and code table."""
    if not LAND_USE_ZIP.exists():
        if offline:
            raise FileNotFoundError(f"Offline run requires {LAND_USE_ZIP}")
        download_file(LAND_USE_URL, LAND_USE_ZIP)
    if not LAND_USE_TIFF.exists() or not LAND_USE_CODE_TABLE.exists():
        with zipfile.ZipFile(LAND_USE_ZIP) as archive:
            required_members = {
                "LUMHK_RasterGrid_2021/LUM_end2021.tif",
                "LUMHK_RasterGrid_2021/CODE TABLE.xlsx",
            }
            if not required_members.issubset(set(archive.namelist())):
                raise ValueError("Official land-utilization archive is missing the raster or code table")
            for member in required_members:
                archive.extract(member, LAND_USE_DIR)

    code_table = pd.read_excel(LAND_USE_CODE_TABLE, header=None, names=["code", "land_use"])
    code_table["code"] = pd.to_numeric(code_table["code"], errors="coerce")
    classes = {
        int(row.code): str(row.land_use).strip()
        for row in code_table.dropna(subset=["code", "land_use"]).itertuples(index=False)
    }
    expected_residential = {
        1: "Private Residential",
        2: "Public Residential",
        3: "Rural Settlement",
    }
    if {code: classes.get(code) for code in expected_residential} != expected_residential:
        raise ValueError("Official land-utilization residential codes do not match 1/2/3")
    with Image.open(LAND_USE_TIFF) as image:
        tiepoint = tuple(float(value) for value in image.tag_v2.get(33922, ()))
        scale = tuple(float(value) for value in image.tag_v2.get(33550, ()))
        size = image.size
        mode = image.mode
    if size != (6375, 4800) or mode != "L" or scale[:2] != (10.0, 10.0):
        raise ValueError(
            f"Unexpected land-utilization raster metadata: size={size}, mode={mode}, scale={scale}"
        )
    if len(tiepoint) < 6 or tiepoint[3:5] != (800000.0, 848000.0):
        raise ValueError(f"Unexpected land-utilization raster tiepoint: {tiepoint}")
    return {
        "grid_cells": int(size[0] * size[1]),
        "width": size[0],
        "height": size[1],
        "cell_size_m": 10.0,
        "crs": "EPSG:2326",
        "upper_left_edge_m": [tiepoint[3], tiepoint[4]],
        "residential_codes": expected_residential,
        "archive_sha256": sha256(LAND_USE_ZIP),
        "source": LAND_USE_URL,
        "catalogue": "https://data.gov.hk/en-data/dataset/hk-pland-pland1-land-utilization-in-hong-kong-raster-grid",
        "scope_note": "Planning Department describes the raster as broad-brush land utilization; it is used only as a dasymetric mask with building geometry, not as a building classifier.",
    }


def read_building_group_table() -> pd.DataFrame:
    table = pd.read_csv(BG_SOURCE_TABLE, skiprows=4, dtype=str, encoding="utf-8-sig", low_memory=False)
    required = {"bg", "hma", "t_pop", "bg_ind", "bg_eng", "bg_chi"}
    missing = required - set(table)
    if missing:
        raise ValueError(f"Building Group table is missing columns: {sorted(missing)}")
    table = table[list(required)].copy()
    # C&SD publishes 3,286 B records as Building Groups.  O records are
    # intentionally separate residual "other buildings" aggregates.
    table = table.loc[table["bg_ind"].eq("B")].copy()
    table["t_pop"] = pd.to_numeric(table["t_pop"], errors="raise")
    if table["bg"].duplicated().any():
        raise ValueError("Official Building Group table contains duplicate group codes")
    return table


def fetch_csdi_building_groups() -> pd.DataFrame:
    records: list[dict] = []
    offset = 0
    while True:
        query = urllib.parse.urlencode(
            {
                "where": "1=1",
                "outFields": "bg,hma,t_pop,bg_ind,bg_eng,bg_chi",
                "returnGeometry": "true",
                "resultOffset": offset,
                "resultRecordCount": 3000,
                "orderByFields": "OBJECTID",
                "f": "json",
            }
        )
        page = request_json(f"{CSDI_BG_QUERY}?{query}")
        if "error" in page:
            raise RuntimeError(f"CSDI Building Group query failed: {page['error']}")
        features = page.get("features", [])
        for feature in features:
            row = dict(feature["attributes"])
            row["point_x_m"] = feature["geometry"]["x"]
            row["point_y_m"] = feature["geometry"]["y"]
            records.append(row)
        if not page.get("exceededTransferLimit", False):
            break
        offset += len(features)
    frame = pd.DataFrame(records)
    frame = frame.loc[frame["bg_ind"].eq("B")].copy()
    frame["t_pop"] = pd.to_numeric(frame["t_pop"], errors="raise")
    if frame["bg"].duplicated().any():
        raise ValueError("CSDI Building Group layer contains duplicate group codes")
    return frame


def fetch_centamap_hma(hma: str) -> list[dict]:
    payload = {
        "Year": "2016",
        "Type": "hma",
        "Code": hma,
        "DetailCode": "",
        "Sort": "asc",
        "Field": "WP_FOOT",
        "Platform": "",
        "Lang": "en-US",
    }
    response = request_json(CENTAMAP_MODEL, payload=payload)
    groups = response.get("ItemsCensus")
    if not isinstance(groups, list) or not groups:
        raise ValueError(f"Centamap returned no 2021 Building Groups for {hma}")
    return [
        {
            "bg": group["Code"],
            "centamap_hma": hma,
            "centamap_population": pd.to_numeric(group.get("T_POP"), errors="raise"),
            "major_building_names_en": str(group.get("CustomE") or "").strip(),
            "major_building_names_tc": str(group.get("Custom") or "").strip(),
        }
        for group in groups
    ]


def prepare_building_group_membership(refresh: bool, offline: bool) -> dict:
    if BG_MEMBERSHIP.exists() and not refresh:
        return json.loads(BG_MEMBERSHIP_METADATA.read_text(encoding="utf-8"))
    if offline:
        raise FileNotFoundError(f"Offline run requires {BG_MEMBERSHIP}")

    official = read_building_group_table()
    csdi = fetch_csdi_building_groups()
    hma_codes = sorted(official["hma"].dropna().unique())
    member_rows: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for rows in executor.map(fetch_centamap_hma, hma_codes):
            member_rows.extend(rows)
    members = pd.DataFrame(member_rows)
    members = members.loc[members["bg"].isin(set(official["bg"]))].copy()
    if members["bg"].duplicated().any():
        raise ValueError("Centamap Building Group response contains duplicate group codes")

    expected_codes = set(official["bg"])
    csdi_codes = set(csdi["bg"])
    member_codes = set(members["bg"])
    if expected_codes != csdi_codes or expected_codes != member_codes:
        raise ValueError(
            "Building Group code coverage differs: "
            f"official={len(expected_codes)}, CSDI={len(csdi_codes)}, Centamap={len(member_codes)}"
        )

    merged = official.merge(
        csdi[["bg", "hma", "t_pop", "point_x_m", "point_y_m"]],
        on="bg",
        how="left",
        validate="one_to_one",
        suffixes=("_official", "_csdi"),
    ).merge(members, on="bg", how="left", validate="one_to_one")
    hma_match = merged["hma_official"].eq(merged["hma_csdi"]) & merged["hma_official"].eq(merged["centamap_hma"])
    population_match = (
        merged["t_pop_official"].eq(merged["t_pop_csdi"])
        & merged["t_pop_official"].eq(merged["centamap_population"])
    )
    if not bool(hma_match.all() and population_match.all()):
        raise ValueError(
            f"Building Group three-source check failed: HMA mismatches={int((~hma_match).sum())}, "
            f"population mismatches={int((~population_match).sum())}"
        )

    merged = merged.rename(
        columns={
            "hma_official": "hma",
            "t_pop_official": "official_population",
            "bg_eng": "bg_name_en",
            "bg_chi": "bg_name_tc",
        }
    )
    merged["major_building_count_en"] = merged["major_building_names_en"].map(lambda value: len([name for name in value.split(",") if name.strip()]))
    merged["major_building_count_tc"] = merged["major_building_names_tc"].map(lambda value: len([name for name in value.split(",") if name.strip()]))
    columns = [
        "bg", "hma", "official_population", "bg_ind", "bg_name_en", "bg_name_tc",
        "point_x_m", "point_y_m", "major_building_names_en", "major_building_names_tc",
        "major_building_count_en", "major_building_count_tc",
    ]
    BG_MEMBERSHIP.parent.mkdir(parents=True, exist_ok=True)
    merged[columns].sort_values("bg").to_parquet(BG_MEMBERSHIP, index=False)
    metadata = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "building_groups": int(len(merged)),
        "housing_market_areas": int(merged["hma"].nunique()),
        "all_group_codes_match": True,
        "all_hma_codes_match": True,
        "all_2021_population_values_match": True,
        "groups_without_any_major_building_name": int(((merged["major_building_count_en"] == 0) & (merged["major_building_count_tc"] == 0)).sum()),
        "sources": {
            "official_table": "https://www.census2021.gov.hk/doc/HMA_BG_21C.zip",
            "official_points": CSDI_BG_QUERY.rsplit("/query", 1)[0],
            "major_building_names": CENTAMAP_MODEL,
            "dataset_catalogue": "https://data.gov.hk/en-data/dataset/hk-censtatd-census_geo-2021-population-census-by-hmabg",
        },
        "major_building_note": "The public Centamap Building Group detail page labels Custom/CustomE as major buildings (主要樓宇). The list is descriptive, not an exhaustive Building Group membership or boundary, and is used only for address-linkage auditing.",
    }
    BG_MEMBERSHIP_METADATA.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--hash-large-files", action="store_true", help="Also hash files larger than 250 MB.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Do not access the network; validate and use only already frozen input caches.",
    )
    parser.add_argument(
        "--download-cis-archive",
        action="store_true",
        help="Optionally download all 44 annual CIS files for source archiving; not required for analysis.",
    )
    parser.add_argument("--refresh-building-groups", action="store_true", help="Refresh official Building Group points and Centamap member names.")
    parser.add_argument("--refresh-als", action="store_true", help="Rebuild the local ALS building-address table from the official archive.")
    parser.add_argument("--refresh-hma", action="store_true", help="Refresh the official HMA polygon layer.")
    parser.add_argument("--refresh-ssug", action="store_true", help="Refresh and cross-check the official SSUG polygon layer.")
    parser.add_argument(
        "--deep-external-audit",
        action="store_true",
        help="Read every member of downloaded ZIP archives to verify CRC; slower but independent of the weather pipeline.",
    )
    parser.add_argument(
        "--e8-preflight",
        action="store_true",
        help="Compile the solver-free E8 G0-G2 audit and dry-run manifests, then exit.",
    )
    parser.add_argument(
        "--e8-download-hourly",
        action="store_true",
        help="Allow E8 preflight to download candidate NASA POWER hourly weather caches.",
    )
    args = parser.parse_args()
    if args.e8_preflight:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from utils.e8_preflight import run_preflight

        if args.offline and args.e8_download_hourly:
            parser.error("--offline and --e8-download-hourly cannot be used together")
        summary = run_preflight(
            ROOT,
            ROOT / "intermediate/simulation_outputs/upstream",
            download_hourly=bool(args.e8_download_hourly),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    hko_weather_audit = prepare_hko_weather(args.offline, args.download_cis_archive)
    hko_weather_audit.to_csv(output / "hko_weather_cache_audit.csv", index=False, encoding="utf-8-sig")
    reference_event_weather_metadata = prepare_reference_event_weather(args.offline)
    als_metadata = prepare_als_buildings(args.refresh_als, args.offline)
    hma_metadata = prepare_hma_boundaries(args.refresh_hma, args.offline)
    ssug_metadata = prepare_ssug_boundaries(args.refresh_ssug, args.offline)
    land_utilization_metadata = prepare_land_utilization(args.offline)
    building_group_metadata = prepare_building_group_membership(args.refresh_building_groups, args.offline)
    building_stock_metadata = audit_building_stock(output)
    wrf_heatwave_metadata = audit_wrf_heatwave_data(output, args.deep_external_audit)
    cmip6_point_temperature_metadata = prepare_cmip6_point_temperatures(output, args.offline)
    external_validation_audit = audit_external_validation_data(output, args.deep_external_audit)
    polyu_hourly_weather_metadata = prepare_polyu_hourly_weather(args.offline)
    hkust_hierarchy_metadata = audit_hkust_brick_hierarchy(output)
    external_validation_profiles = (
        profile_external_validation_data(output) if args.deep_external_audit else pd.DataFrame()
    )
    write_model_registries(output)

    rows = []
    for key, path in REQUIRED_INPUTS.items():
        exists = path.exists()
        size = path.stat().st_size if exists else None
        should_hash = bool(exists and (args.hash_large_files or size <= 250_000_000))
        rows.append(
            {
                "input_id": key,
                "path": str(path.resolve()),
                "exists": exists,
                "size_bytes": size,
                "sha256": sha256(path) if should_hash else None,
                "hash_status": "computed" if should_hash else ("skipped_large" if exists else "missing"),
            }
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(output / "input_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(THRESHOLDS).to_csv(output / "threshold_registry.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(METHOD_DECISIONS, columns=["topic", "v0_2", "v0_3", "reason"]).to_csv(
        output / "method_decisions.csv", index=False, encoding="utf-8-sig"
    )

    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: package_version(name)
            for name in ["pandas", "numpy", "scipy", "statsmodels", "geopandas", "scikit-learn", "pyarrow", "shapely", "matplotlib", "pillow", "xarray", "netCDF4", "rdflib"]
        },
        "all_required_inputs_exist": bool(audit["exists"].all()),
        "missing_inputs": audit.loc[~audit["exists"], "input_id"].tolist(),
        "canonical_rules": {
            "unsourced_thresholds_permitted": False,
            "fine_grid_equals_observation_precision": False,
            "spatial_model_requires_station_held_out_improvement": True,
            "population_is_building_level_observation": False,
            "outage_probability_is_estimated": False,
        },
        "building_group_membership": building_group_metadata,
        "building_stock": building_stock_metadata,
        "wrf_heatwave": wrf_heatwave_metadata,
        "cmip6_point_temperature": cmip6_point_temperature_metadata,
        "reference_event_weather": reference_event_weather_metadata,
        "polyu_hourly_weather": polyu_hourly_weather_metadata,
        "hkust_meter_hierarchy": hkust_hierarchy_metadata,
        "external_validation": {
            "datasets_listed": int(len(external_validation_audit)),
            "downloaded_or_partially_available": int(external_validation_audit["exists"].sum()),
            "pending_download": external_validation_audit.loc[
                external_validation_audit["current_status"].eq("pending_download"), "dataset_id"
            ].tolist(),
            "full_polyu_campus_is_distinct_from_block_z": True,
            "profile_rows": int(len(external_validation_profiles)),
        },
        "hko_weather_cache": {
            "files": int(len(hko_weather_audit)),
            "cis_year_files": int((hko_weather_audit["source"] == "hko_cis").sum()),
            "open_data_files": int((hko_weather_audit["source"] == "hko_open_data").sum()),
            "cis_archive_requested": bool(args.download_cis_archive),
            "cis_archive_complete": bool(
                (hko_weather_audit["source"] == "hko_cis").sum() == len(HKO_CIS_YEARS)
            ),
            "all_files_validated": True,
        },
        "address_lookup_service": als_metadata,
        "housing_market_area_boundaries": hma_metadata,
        "small_subunit_group_boundaries": ssug_metadata,
        "land_utilization_grid": land_utilization_metadata,
    }
    (output / "environment_and_gate.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not environment["all_required_inputs_exist"]:
        raise SystemExit(f"Missing required inputs: {environment['missing_inputs']}")
    if not cmip6_point_temperature_metadata[
        "all_daily_maximum_and_minimum_caches_ready"
    ]:
        print(
            "00 audit complete, but CMIP6 point-temperature caches are incomplete: "
            f"{cmip6_point_temperature_metadata['valid_caches']}/"
            f"{cmip6_point_temperature_metadata['expected_caches']} valid. "
            "Rerun without --offline to resume the retained .part cache."
        )
    else:
        print(f"00 complete: {output}")


if __name__ == "__main__":
    main()
