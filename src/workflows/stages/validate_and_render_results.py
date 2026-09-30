"""Render paper figures from frozen Result 1-3 tables.

This script performs no scientific simulation and does not fill missing values
with demo data. The animation product aligns building type, hourly electricity,
indoor thermal safety, baseline reference-suite energy resilience and aggregate load.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

_CONDA_SHARE = Path(sys.prefix) / "Library" / "share"
if (_CONDA_SHARE / "gdal").is_dir():
    os.environ.setdefault("GDAL_DATA", str(_CONDA_SHARE / "gdal"))
if (_CONDA_SHARE / "proj").is_dir():
    os.environ.setdefault("PROJ_LIB", str(_CONDA_SHARE / "proj"))

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import dates as mdates
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image
from pyarrow import parquet as pq


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.building_composition import FLOOR_ROLES, floor_role_weight_table
from utils.figure_style import panel_title, save_figure


CONFIG = json.loads((ROOT / "config" / "foundation_analysis.json").read_text(encoding="utf-8"))
PIPELINE = ROOT / CONFIG["output_root"]
FULL_STOCK = ROOT / "output" / "heatwave_stock_foundation_v0_3" / "building_stock_repaired.geoparquet"
MODEL_INPUTS = PIPELINE / "02_building_response" / "all_sector_building_model_inputs.parquet"
RESULTS = PIPELINE / "03_energy_resilience" / "building_heatwave_energy_resilience.parquet"
REFERENCE_BERI_RESULTS = PIPELINE / "03_energy_resilience" / "building_reference_beri.parquet"
EVENT_RESPONSE_RESULTS = PIPELINE / "03_energy_resilience" / "building_event_response_score.parquet"
OUT = PIPELINE / "04_figures"
AUDIT_DIR = PIPELINE / "00_audit"
HAZARD_DIR = PIPELINE / "01_hazard"
RESPONSE_DIR = PIPELINE / "02_building_response"
RESULT_DIR = PIPELINE / "03_energy_resilience"
OUTAGE_TIMING_AUDIT = RESPONSE_DIR / "outage_timing_contract_audit.csv"
ANIMATION_DIR = OUT / "animations"
ANIMATION_LAYOUT_VERSION = "five_panel_v8_reference_beri_state_labelled_scope_load"
ROAD_DIR = ROOT / "data" / "spatial" / "csdi_hk_transport_road_network_2026_08_31"
HMA_BOUNDARIES = ROOT / "intermediate" / "source_derived" / "census2021_housing_market_areas.geoparquet"
HOMANTIN_BOUNDS_2326 = (836126.18034143, 818910.80931283, 837345.03745461, 820422.94111362)
HOMANTIN_BOUNDS_SOURCE = "fixed inset within official Census 2021 HMA058 Ho Man Tin; EPSG:2326"
REGION_BOUNDS_2326 = {
    "Ho Man Tin": HOMANTIN_BOUNDS_2326,
    "Mong Kok": (835132.28, 819514.11, 836016.10, 820302.95),
    "Tsim Sha Tsui": (835001.12, 817109.65, 836317.21, 818280.89),
}
REGION_BOUNDS_SOURCE = {
    "Ho Man Tin": HOMANTIN_BOUNDS_SOURCE,
    "Mong Kok": "official Census 2021 HMA068 Mong Kok bounding extent; EPSG:2326",
    "Tsim Sha Tsui": "official Census 2021 HMA172 Tsim Sha Tsui bounding extent; EPSG:2326",
}

EVENT_LABELS = {
    "HW_20180526_20180601": "26 May–1 June 2018",
    "HW_20220722_20220729": "22–29 July 2022",
    "HW_20230724_20230728": "24–28 July 2023",
}

EVENT_SELECTION_NOTES = {
    "HW_20180526_20180601": "high-end early-summer reference window within the 2018 record event",
    "HW_20220722_20220729": "prolonged midsummer reference window within the 2022 event",
    "HW_20230724_20230728": "moderate midsummer transfer window with a shorter thermal sequence",
}

EVENT_PARENT_HKO = {
    "HW_20180526_20180601": "HKO_LI_0068",
    "HW_20220722_20220729": "HKO_LI_0085",
    "HW_20230724_20230728": "HKO_LI_0090",
}

TYPE_COLORS = {
    "Residential": "#3b82c4",
    "Commercial": "#d95f4a",
    "Mixed-use": "#e68a00",
    "Industrial": "#78716c",
    "Non-assessed": "#d6d3d1",
}

ANIMATION_TYPE_LABELS = {
    "res_public_rental": "Public rental",
    "res_subsidised_sale": "Subsidised sale",
    "res_private_highrise": "Private high-rise",
    "res_old_lowrise_tenement": "Old low-rise",
    "res_village_house": "Village house",
    "com_office": "Office",
    "com_retail": "Retail",
    "com_restaurant": "Restaurant",
    "com_accommodation": "Accommodation",
    "com_education": "Education",
    "com_human_health": "Healthcare",
    "com_data_centre": "Data centre",
    "com_other": "Other commercial",
    "mixed_residential_commercial": "Mixed-use",
}

ANIMATION_TYPE_COLORS = {
    "Public rental": "#174A7E",
    "Subsidised sale": "#2F75B5",
    "Private high-rise": "#5B9BD5",
    "Old low-rise": "#75B7D3",
    "Village house": "#A6CEE3",
    "Office": "#A50F15",
    "Retail": "#DE2D26",
    "Restaurant": "#FB6A4A",
    "Accommodation": "#C51B7D",
    "Education": "#88419D",
    "Healthcare": "#E6550D",
    "Data centre": "#F16913",
    "Other commercial": "#FDAE6B",
    "Mixed-use": "#4D4D4D",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_frozen_numeric_artifact_catalog() -> pd.DataFrame:
    """Index reusable numerical products independently of figure styling."""
    specifications = [
        (HAZARD_DIR / "heatwave_event_catalog.csv", "one row per station-level identified historical heatwave", "event dates, duration and percentile-threshold intensity", "Figure 1; historical trends; supplementary events"),
        (ROOT / "output" / "heatwave_day3" / "weather_event_catalog.csv", "one row per complete-hourly fixed-threshold stress candidate", "pre-outcome event ranking and selection by cumulative exceedance above the HKO 33 C very-hot-day threshold", "stress-event selection; Methods; presentation"),
        (HAZARD_DIR / "station_year_heat_metrics.csv", "station-year", "annual heatwave frequency, duration and intensity metrics", "Figure 1; historical trends"),
        (HAZARD_DIR / "trend_reconciliation.csv", "definition-metric trend estimate", "Li-aligned and HKO-threshold trend reconciliation", "Figure 1; manuscript trend results"),
        (HAZARD_DIR / "spatial_model_cv.csv", "spatial-model validation fold", "station-holdout performance of the hazard surface", "Figure 1; validation"),
        (HAZARD_DIR / "spatial_model_paired_superiority_tests.csv", "paired spatial benchmark comparison", "paired evidence for spatial-model selection", "Figure 1; validation"),
        (HAZARD_DIR / "wrf_distributional_spatial_bias_validation.csv", "held-out variable and statistic", "raw and corrected error at unseen stations and years", "Figure 1; validation"),
        (HAZARD_DIR / "wrf_tmax_distributional_spatiotemporal_holdout.csv", "held-out station-month", "paired raw and corrected Tmax quantile, P95 and threshold-frequency errors", "Figure 1d; independent validation"),
        (HAZARD_DIR / "wrf_tmin_distributional_spatiotemporal_holdout.csv", "held-out station-month", "paired raw and corrected Tmin quantile, P95 and threshold-frequency errors", "Figure 1e; independent validation"),
        (HAZARD_DIR / "building_heat_hazard_wrf1km.parquet", "one row per HK-BFETD footprint", "validated 1-km historical heat-hazard fields", "Figure 1; spatial overlays"),
        (HAZARD_DIR / "cmip6_tmax_future_model_summary.csv", "GCM-scenario-period", "model-level future maximum-temperature change", "Figures 1 and 6"),
        (HAZARD_DIR / "cmip6_tmin_future_model_summary.csv", "GCM-scenario-period", "model-level future minimum-temperature change", "Figures 1 and 6"),
        (HAZARD_DIR / "cmip6_compound_change_factors.csv", "GCM-scenario-period", "humidity, radiation, wind and pressure morphing factors", "future EnergyPlus weather construction"),
        (HAZARD_DIR / "cmip6_wrf_multimodel_tmax_change_field.parquet", "GCM-scenario-period-WRF grid cell", "future maximum-temperature change field", "Figure 1; future maps"),
        (HAZARD_DIR / "cmip6_wrf_multimodel_tmin_change_field.parquet", "GCM-scenario-period-WRF grid cell", "future minimum-temperature change field", "Figure 1; future maps"),
        (RESPONSE_DIR / "all_sector_building_model_inputs.parquet", "one row per target-scope building", "frozen geometry, type, component area, population and WRF-grid keys", "Figures 3 and 4; animations"),
        (RESPONSE_DIR / "final_building_sample_audit.json", "final building-sample flow", "target scope, exclusions and unique-building/component distinction", "Methods; Supplementary Figure S1"),
        (RESPONSE_DIR / "final_building_sample_by_type.csv", "final top-level building type", "quality-controlled unique-building counts", "Methods; Supplementary Figure S1"),
        (RESPONSE_DIR / "final_component_sample_by_prototype.csv", "simulated component prototype", "residential and non-residential component counts and areas", "Methods; Supplementary Figure S1"),
        (RESPONSE_DIR / "building_population_and_archetypes.parquet", "one row per residential or mixed-use footprint", "primary and sensitivity population allocations", "Figure 2; exposure analyses"),
        (RESPONSE_DIR / "major_housing_estate_population_validation.csv", "major housing estate and allocation scheme", "building population allocation against 540 external estate totals", "Figure 2; population validation"),
        (RESPONSE_DIR / "population_validation.json", "population allocation workflow", "conservation, sensitivity and estate-validation audit", "Methods; release audit"),
        (RESPONSE_DIR / "prototype_physics_parameters.csv", "one row per EnergyPlus prototype", "frozen envelope, internal-gain, ventilation and cooling-capacity parameters", "Methods; model interpretation"),
        (RESPONSE_DIR / "prototype_parameter_provenance.csv", "prototype and parameter", "value, evidence tier, source and downstream calibration", "Methods; release audit"),
        (RESPONSE_DIR / "experiment_setting_provenance.csv", "pre-specified experiment setting", "value, scientific role, basis and source", "Methods; release audit"),
        (RESPONSE_DIR / "energyplus_multizone_results.csv", "prototype-design-event-climate-supply case", "direct EnergyPlus event summaries", "Figures 2, 4 and 5"),
        (RESPONSE_DIR / "energyplus_multizone_hourly_results.parquet", "prototype-design-event-climate-supply hour", "direct EnergyPlus hourly trajectories", "validation; derived metrics"),
        (RESPONSE_DIR / "energyplus_diagnostics_audit.json", "historical EnergyPlus run suite", "severe-error and warning-classification audit", "model validation; release audit"),
        (RESPONSE_DIR / "building_physics_consistency_audit.json", "building-response physical identities", "cooling balance, control, finite-value and timing checks", "model validation; release audit"),
        (RESPONSE_DIR / "paired_counterfactual_contract_audit.json", "paired event-hour contract", "weather-only attribution identity checks", "model validation; manuscript attribution"),
        (RESPONSE_DIR / "outage_timing_contract_audit.csv", "event-design-outage stage", "outage start, duration, control identity and zero-grid-load checks", "Methods; resilience validation"),
        (RESPONSE_DIR / "parameter_and_experiment_audit.json", "model parameter and experiment contract", "evidence ranges and complete design checks", "Methods; release audit"),
        (RESPONSE_DIR / "result2_validation.json", "building-response workflow", "complete sample, model, calibration and validation record", "Methods; release audit"),
        (RESPONSE_DIR / "prototype_event_hourly_interface.parquet", "prototype-floor-event-supply hour", "compact calibrated hourly response interface", "Figures 3 and 4; animations"),
        (RESPONSE_DIR / "building_dynamic_results.parquet", "building-event-supply stage", "propagated building consequences and uncertainty summaries", "Figures 3–6; Result 3"),
        (RESPONSE_DIR / "citywide_coincident_hourly_response.parquet", "event-supply-climate hour", "territory-wide coincident electricity trajectory", "Figures 4–6; animations"),
        (RESPONSE_DIR / "annual_2023_citywide_hourly_end_use.parquet", "2023 territory hour", "calibrated cooling, lighting, equipment and total electricity", "annual load/end-use analyses"),
        (RESPONSE_DIR / "official_domestic_monthly_shape_holdout_validation.csv", "calendar month", "official and modelled domestic electricity shares", "Figure 2; temporal validation"),
        (RESPONSE_DIR / "official_energy_calibration_factors.csv", "official sector-calibration group", "cooling and non-cooling mass-balance factors", "model calibration; Figure 2"),
        (RESPONSE_DIR / "official_energy_mass_balance_validation.csv", "official sector and end use", "official-to-model annual mass-balance validation", "Figure 2; calibration"),
        (RESPONSE_DIR / "polyu_campus_balance_point_blocked_cv.csv", "PolyU building holdout fold", "PolyU predictive load validation", "Figure 2"),
        (RESPONSE_DIR / "hkust_whole_building_daily_validation.parquet", "HKUST building-day", "HKUST predictive load validation", "Figure 2"),
        (RESPONSE_DIR / "local_observed_vs_energyplus_heat_sensitivity.csv", "dataset-sector-prototype comparison", "observed and simulated temperature-load sensitivity", "Figure 2; validation"),
        (RESPONSE_DIR / "bca_2024_hot_humid_commercial_eui_plausibility.csv", "commercial use class", "Hong Kong modelled EUI against hot-humid BCA distribution", "Figure 2; external plausibility"),
        (RESPONSE_DIR / "validation_evidence_matrix.csv", "validation evidence layer", "dataset, split, metric, result and scientific role", "Figure 2; Methods; release audit"),
        (RESPONSE_DIR / "external_dynamic_structure_validation.csv", "external dataset and thermal-response metric", "hot-humid and residential dynamic-structure checks", "Figure 2; supplementary validation"),
        (RESPONSE_DIR / "future_energyplus_results.csv", "GCM-scenario-period-node-prototype-event-supply case", "future event summary ensemble", "Figure 6; future analyses"),
        (RESPONSE_DIR / "future_energyplus_hourly_profiles.parquet", "GCM-scenario-period-node-prototype-event-supply hour", "future event hourly ensemble", "future diagnostics and supplementary results"),
        (RESPONSE_DIR / "future_energyplus_diagnostics_audit.json", "future EnergyPlus run suite", "future severe-error and warning-classification audit", "future validation; release audit"),
        (RESPONSE_DIR / "future_energyplus_validation.json", "future-response workflow", "future run completeness and finite-value validation", "Figure 6; release audit"),
        (RESULT_DIR / "building_heatwave_energy_resilience.parquet", "building-event-supply stage", "analysis-ready energy, thermal safety and recovery results", "Figures 3–6"),
        (RESULT_DIR / "building_event_response_score.parquet", "building-event", "event-conditioned four-component diagnostic score", "supplementary event comparisons"),
        (RESULT_DIR / "building_reference_beri.parquet", "one row per modelled building", "baseline reference-suite BERI and weighting sensitivities", "Figures 3–5; animations"),
        (RESULT_DIR / "building_energy_resilience_index_summary.csv", "main class and prototype", "reference BERI distributions and component medians", "Figure 5; manuscript results"),
        (RESULT_DIR / "type_event_stage_summary.csv", "prototype-event-supply stage-outcome", "type-resolved direct physical outcomes", "Figures 5 and 6; manuscript results"),
        (RESULT_DIR / "class_event_stage_summary.csv", "main class-event-supply stage-outcome", "class-resolved direct physical outcomes", "Figure 5; manuscript results"),
        (RESULT_DIR / "city_event_stage_summary.csv", "event-supply stage-outcome", "territory-wide direct physical outcomes", "Figures 5 and 6; manuscript results"),
        (RESULT_DIR / "city_coincident_peak_summary.csv", "event-supply stage", "hourly coincident territory peak and timing", "Figures 4–6"),
        (RESULT_DIR / "beri_weight_sensitivity_summary.csv", "BERI weighting scenario", "reference-suite ranking and magnitude sensitivity", "Figure 5; supplementary uncertainty"),
        (RESULT_DIR / "beri_lowest_top_fraction_capture.csv", "event-stage-burden outcome", "burden captured by the lowest-BERI fraction", "Figure 5; manuscript results"),
        (RESULT_DIR / "beri_leave_one_event_out_summary.csv", "held-out reference event", "rank and low-score-cohort transfer of a BERI constructed from the other two events", "Figure 5; manuscript results"),
        (RESULT_DIR / "beri_leave_one_event_out_capture.csv", "selection rule-held-out event-burden outcome", "held-out burden capture by BERI, stage scores, class and age comparators", "Figure 5; supplementary validation"),
        (RESULT_DIR / "beri_definition_sensitivity.csv", "BERI definition scenario", "threshold, horizon, score-map and aggregation sensitivity", "Figure 5; supplementary validation"),
        (RESULT_DIR / "cross_stage_ranking_audit.csv", "pair of BERI stage components", "all-building rank correlation and tie-aware low-score cohort overlap", "Figure 5d; manuscript results"),
        (RESULT_DIR / "cross_stage_cutoff_sensitivity.csv", "screening fraction and pair of BERI stage components", "tie-aware cross-stage cohort overlap at 5%, 10% and 20% cutoffs", "Figure 5d; manuscript robustness"),
        (RESULT_DIR / "driver_event_holdout_performance.csv", "target and held-out event", "whole-event holdout performance for driver analysis", "Figure 5; driver eligibility"),
        (RESULT_DIR / "driver_decomposition.csv", "target and driver", "cross-event permutation importance summary", "Figure 5"),
        (RESULT_DIR / "class_hourly_end_use_attribution.parquet", "event-main class-hour", "pointwise stock load and end-use increments", "Figure 6a; event-clock attribution"),
        (RESULT_DIR / "event_timing_decomposition.csv", "historical stress event", "exact weather and counterfactual-timing decomposition of the difference between maxima", "Figure 6a; manuscript results"),
        (RESULT_DIR / "class_event_timing_decomposition.csv", "historical stress event-main class", "class and end-use contributions to event electricity and peak weather increments", "Figure 6a; manuscript results"),
        (RESULT_DIR / "class_hourly_attribution_audit.json", "hourly attribution workflow", "city-trajectory reconstruction and peak-decomposition closure", "Figure 6a; release audit"),
        (RESULT_DIR / "district_spatial_summary.csv", "district-event-main class", "building count, event electricity, relative response, thermal endurance and BERI", "Figures 3 and 4; manuscript results"),
        (RESULT_DIR / "historical_future_recovery_comparability.csv", "observed event or future representative node", "recovery comparison on matched central prototypes", "Figure 6; manuscript results"),
        (RESULT_DIR / "historical_future_recovery_comparability_audit.json", "historical-future recovery workflow", "matched-prototype and common-definition contract", "Figure 6; release audit"),
        (RESULT_DIR / "future_ensemble_statistics_audit.csv", "scenario-period", "pooled-case and climate-model-median estimands", "Figure 6c; manuscript results"),
        (RESULT_DIR / "future_prototype_stage_summary.csv", "scenario-period-prototype-stage", "future response uncertainty summary", "Figure 6"),
        (RESULT_DIR / "dynamic_result_field_contract.csv", "analysis-ready result field", "field unit, time grain and interpretation", "all figures; downstream replotting"),
        (RESULT_DIR / "beri_contract_audit.json", "BERI formula and reference-suite contract", "component, geometric-mean and event-median identities", "Figures 3–5; release audit"),
        (RESULT_DIR / "result3_validation.json", "result aggregation workflow", "complete result and finding validation record", "manuscript; release audit"),
        (RESULT_DIR / "manuscript_key_findings.json", "manuscript result payload", "machine-readable exact values used in prose", "Results; captions; presentation"),
        (OUT / "figure_panel_data_contract.csv", "one row per formal figure panel", "panel-level data source, selection and scientific role", "all formal figures; manuscript cross-reference"),
    ]
    parameter_audit_path = RESPONSE_DIR / "parameter_and_experiment_audit.json"
    parameter_audit = (
        json.loads(parameter_audit_path.read_text(encoding="utf-8"))
        if parameter_audit_path.exists()
        else {}
    )
    model_version = str(parameter_audit.get("model_version", "unavailable"))
    configuration_sha256 = file_sha256(ROOT / "config" / "foundation_analysis.json")
    records = []
    for path, grain, role, consumers in specifications:
        if not path.exists():
            continue
        if path.suffix.lower() == ".parquet":
            parquet = pq.ParquetFile(path)
            row_count = int(parquet.metadata.num_rows)
            columns = parquet.schema_arrow.names
        elif path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                row_count = 1
                columns = list(payload.keys())
            elif isinstance(payload, list):
                row_count = len(payload)
                columns = sorted(
                    {key for item in payload if isinstance(item, dict) for key in item}
                )
            else:
                row_count = 1
                columns = ["value"]
        else:
            columns = pd.read_csv(path, nrows=0).columns.tolist()
            with path.open("rb") as handle:
                row_count = max(sum(1 for _ in handle) - 1, 0)
        records.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "format": path.suffix.lower().lstrip("."),
                "grain": grain,
                "scientific_role": role,
                "figure_consumers": consumers,
                "model_version": model_version,
                "configuration_sha256": configuration_sha256,
                "rows": row_count,
                "columns": len(columns),
                "column_names": "|".join(columns),
                "bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
            }
        )
    catalog = pd.DataFrame(records)
    catalog.to_csv(
        OUT / "frozen_numeric_artifact_catalog.csv", index=False, encoding="utf-8-sig"
    )
    readme = "\n".join(
        [
            "FROZEN NUMERIC RESULTS FOR PAPER FIGURES AND ANIMATIONS",
            "",
            "Scientific simulations are completed by 00-03. Script 04 reads the saved",
            "tables listed in frozen_numeric_artifact_catalog.csv and changes only",
            "figure layout, typography, colours and video encoding.",
            "",
            f"Model version: {model_version}",
            f"Configuration SHA-256: {configuration_sha256}",
            f"Indexed numerical files: {len(catalog)}",
            "",
            "Re-render static figures without animations:",
            "  python .\\src\\04_VALIDATE_AND_RENDER_RESULTS.py --skip-animations",
            "",
            "Re-render animations from the same frozen data:",
            "  python .\\src\\04_VALIDATE_AND_RENDER_RESULTS.py --animations-only",
            "",
            "Do not edit the numerical files when changing figure appearance. A changed",
            "SHA-256 in the catalogue identifies a changed scientific result.",
        ]
    )
    (OUT / "FROZEN_NUMERIC_DATA_README.txt").write_text(readme, encoding="utf-8")
    return catalog


def write_results_release_manifest() -> None:
    """Freeze code, result tables, figures and encoded videos for the paper release."""
    write_frozen_numeric_artifact_catalog()
    candidates = [
        ROOT / "config" / "foundation_analysis.json",
        *(ROOT / "src").glob("0[0-4]_*.py"),
        *(ROOT / "utils").glob("*.py"),
        ROOT / "MANUSCRIPT_DRAFT.md",
        ROOT / "COVER_LETTER_DRAFT.md",
        ROOT / "FIGURE_CAPTIONS.md",
        ROOT / "PPT_4_PAGE_STORYBOARD_中文.md",
        ROOT / "RESULTS_RELEASE_v0.3.3.md",
        ROOT / "RELEASE_PACKAGE_CONTENTS.md",
        ROOT / "GPT_REVIEW_PACKAGE_README_中文.txt",
        ROOT / "EDITORIAL_AUDIT_v0.3.3_中文.md",
        ROOT / "README_中文.md",
        ROOT / "PROJECT_OVERVIEW_中文.md",
        ROOT / "RUN_ORDER_手动运行顺序.md",
        ROOT / "CODE_TASK_MATRIX_中文.md",
        ROOT / "DATA_SOURCES_中文.md",
        ROOT / "外部数据来源与手动下载说明.txt",
        ROOT / "data" / "00_来源索引" / "外部校准与验证数据来源.txt",
        ROOT / "environment.yml",
        ROOT / "environment.lock.yml",
        ROOT / "pytest.ini",
        ROOT / "tests" / "test_paper_pipeline.py",
        ROOT / "MANUSCRIPT_WITH_FIGURES.docx",
        ROOT / "evidence" / "METHOD_BENCHMARKS_研究对照.md",
        ROOT / "evidence" / "comparable_study_rigor_matrix.csv",
        AUDIT_DIR / "building_sample_flow.csv",
        AUDIT_DIR / "building_stock_audit.json",
        AUDIT_DIR / "environment_and_gate.json",
        AUDIT_DIR / "external_validation_data_audit.csv",
        AUDIT_DIR / "method_decisions.csv",
        AUDIT_DIR / "prototype_registry.csv",
        AUDIT_DIR / "resilience_indicator_registry.csv",
        AUDIT_DIR / "threshold_registry.csv",
        AUDIT_DIR / "wrf_heatwave_archive_audit.json",
        ROOT / "output" / "heatwave_day3" / "weather_event_catalog.csv",
        ROOT / "output" / "heatwave_day3" / "weather_metadata.json",
        ROOT / "output" / "heatwave_day3" / "weather_validation.json",
        HAZARD_DIR / "hazard_validation.json",
        HAZARD_DIR / "cmip6_multimodel_validation.json",
        HAZARD_DIR / "cmip6_tmax_future_ensemble_summary.csv",
        HAZARD_DIR / "cmip6_tmin_future_ensemble_summary.csv",
        HAZARD_DIR / "li_definition_implementation_sensitivity.csv",
        HAZARD_DIR / "trend_reconciliation.csv",
        HAZARD_DIR / "wrf_distributional_spatial_bias_validation.json",
        HAZARD_DIR / "wrf_future_projection_validation.json",
        HAZARD_DIR / "building_heat_hazard_wrf1km.parquet",
        HAZARD_DIR / "building_wrf_land_grid_mapping.parquet",
        HAZARD_DIR / "cmip6_wrf_multimodel_tmax_change_field.parquet",
        HAZARD_DIR / "cmip6_wrf_multimodel_tmin_change_field.parquet",
        RESPONSE_DIR / "result2_validation.json",
        RESPONSE_DIR / "building_dynamic_results_validation.json",
        RESPONSE_DIR / "building_physics_consistency_audit.json",
        RESPONSE_DIR / "building_hourly_interface_manifest.json",
        RESPONSE_DIR / "final_building_sample_audit.json",
        RESPONSE_DIR / "final_building_sample_by_type.csv",
        RESPONSE_DIR / "model_implementation_audit.csv",
        RESPONSE_DIR / "experiment_setting_provenance.csv",
        RESPONSE_DIR / "parameter_and_experiment_audit.json",
        RESPONSE_DIR / "prototype_parameter_provenance.csv",
        RESPONSE_DIR / "population_validation.json",
        RESPONSE_DIR / "prototype_coverage.csv",
        RESPONSE_DIR / "prototype_physics_parameters.csv",
        RESPONSE_DIR / "result2_model_readiness.csv",
        RESPONSE_DIR / "validation_evidence_matrix.csv",
        RESPONSE_DIR / "external_dynamic_structure_validation.csv",
        RESPONSE_DIR / "external_dynamic_structure_validation.json",
        RESPONSE_DIR / "citywide_coincident_peak_summary.csv",
        RESPONSE_DIR / "official_domestic_monthly_shape_holdout_validation.csv",
        RESPONSE_DIR / "official_energy_calibration_validation.json",
        RESPONSE_DIR / "official_energy_mass_balance_validation.csv",
        RESPONSE_DIR / "citywide_paired_trajectory_uncertainty_summary.csv",
        RESPONSE_DIR / "data_centre_outage_physics_audit.csv",
        RESPONSE_DIR / "outage_timing_contract_audit.csv",
        RESPONSE_DIR / "paired_counterfactual_contract_audit.json",
        RESPONSE_DIR / "solar_decomposition_audit.json",
        RESPONSE_DIR / "local_observed_vs_energyplus_heat_sensitivity.csv",
        RESPONSE_DIR / "hkust_whole_building_validation.json",
        RESPONSE_DIR / "all_sector_building_model_inputs.parquet",
        RESPONSE_DIR / "building_population_and_archetypes.parquet",
        RESPONSE_DIR / "building_dynamic_results.parquet",
        RESPONSE_DIR / "energyplus_multizone_results.csv",
        RESPONSE_DIR / "energyplus_multizone_hourly_results.parquet",
        RESPONSE_DIR / "prototype_event_hourly_interface.parquet",
        RESPONSE_DIR / "annual_2023_prototype_hourly_results.parquet",
        RESPONSE_DIR / "annual_2023_citywide_hourly_end_use.parquet",
        RESPONSE_DIR / "citywide_coincident_hourly_response.parquet",
        RESPONSE_DIR / "official_energy_calibration_factors.csv",
        RESPONSE_DIR / "future_energyplus_results.csv",
        RESPONSE_DIR / "future_energyplus_hourly_profiles.parquet",
        RESULT_DIR / "result3_validation.json",
        RESULT_DIR / "beri_contract_audit.json",
        RESULT_DIR / "dynamic_result_field_contract.csv",
        RESULT_DIR / "result3_analysis_plan.csv",
        RESULT_DIR / "class_event_stage_summary.csv",
        RESULT_DIR / "type_event_stage_summary.csv",
        RESULT_DIR / "city_event_stage_summary.csv",
        RESULT_DIR / "city_coincident_peak_summary.csv",
        RESULT_DIR / "driver_decomposition.csv",
        RESULT_DIR / "driver_event_holdout_performance.csv",
        RESULT_DIR / "driver_stability_summary.csv",
        RESULT_DIR / "driver_finding_candidates.csv",
        RESULT_DIR / "cross_stage_ranking_audit.csv",
        RESULT_DIR / "cross_stage_cutoff_sensitivity.csv",
        RESULT_DIR / "class_hourly_end_use_attribution.parquet",
        RESULT_DIR / "event_timing_decomposition.csv",
        RESULT_DIR / "class_event_timing_decomposition.csv",
        RESULT_DIR / "class_hourly_attribution_audit.json",
        RESULT_DIR / "district_spatial_summary.csv",
        RESULT_DIR / "historical_future_recovery_comparability.csv",
        RESULT_DIR / "historical_future_recovery_comparability_audit.json",
        RESULT_DIR / "building_reference_beri.parquet",
        RESULT_DIR / "building_event_response_score.parquet",
        RESULT_DIR / "building_heatwave_energy_resilience.parquet",
        RESULT_DIR / "building_energy_resilience_index_summary.csv",
        RESULT_DIR / "beri_weight_sensitivity_summary.csv",
        RESULT_DIR / "beri_definition_sensitivity.csv",
        RESULT_DIR / "beri_leave_one_event_out_capture.csv",
        RESULT_DIR / "beri_leave_one_event_out_summary.csv",
        RESULT_DIR / "beri_lowest_top_fraction_capture.csv",
        RESULT_DIR / "recovery_elapsed_vs_active_cooling_audit.csv",
        RESULT_DIR / "future_ensemble_statistics_audit.csv",
        RESULT_DIR / "future_prototype_stage_summary.csv",
        RESULT_DIR / "manuscript_key_findings.json",
        OUT / "figure_data_contract.csv",
        OUT / "figure_panel_data_contract.csv",
        OUT / "frozen_numeric_artifact_catalog.csv",
        OUT / "FROZEN_NUMERIC_DATA_README.txt",
        OUT / "publication_figure_manifest.json",
        OUT / "historical_event_and_building_stress_selection_audit.csv",
        *(OUT.glob("Figure_*.png")),
        *(OUT.glob("Figure_*.pdf")),
        *(OUT.glob("Figure_*.style.json")),
        *(OUT.glob("Supplementary_Figure_*.png")),
        *(OUT.glob("Supplementary_Figure_*.pdf")),
        *(OUT.glob("Supplementary_Figure_*.style.json")),
        *(ANIMATION_DIR.rglob("*_HW_*_*.mp4")),
        ANIMATION_DIR / "animation_manifest.json",
    ]
    files = []
    for path in sorted({item.resolve() for item in candidates if item.exists()}):
        files.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
            }
        )
    frame_sets = []
    for directory in sorted(ANIMATION_DIR.glob(f"*_HW_*_{ANIMATION_LAYOUT_VERSION}_frames")):
        frames = sorted(directory.glob("frame_*.png"))
        frame_sets.append(
            {
                "path": directory.relative_to(ROOT).as_posix(),
                "frame_count": len(frames),
                "first_frame": frames[0].name if frames else None,
                "last_frame": frames[-1].name if frames else None,
            }
        )
    scientific_results_path = RESULT_DIR / "manuscript_key_findings.json"
    scientific_results = json.loads(scientific_results_path.read_text(encoding="utf-8"))
    payload = {
        "release_label": "paper-results-v0.3.3",
        "configuration": "config/foundation_analysis.json",
        "random_seed": int(CONFIG["random_seed"]),
        "scope": "330761 quality-controlled residential, commercial and mixed-use buildings",
        "scientific_results_source": scientific_results_path.relative_to(ROOT).as_posix(),
        "scientific_results_sha256": file_sha256(scientific_results_path),
        "scientific_results": scientific_results,
        "files": files,
        "frame_sets": frame_sets,
        "note": "Raw hourly PNG frames are recorded by deterministic filenames and counts; encoded MP4 files are SHA256 hashed.",
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    (PIPELINE / "PAPER_RESULTS_v0_3.json").write_text(text, encoding="utf-8")
    (ROOT / "PAPER_RESULTS_v0_3.json").write_text(text, encoding="utf-8")


def english_label(series: pd.Series) -> pd.Series:
    return series.astype("string").str.split("_", n=1).str[0].str.strip()


def write_figure_contract(output: Path) -> None:
    rows = [
        {
            "figure": "Figure 1",
            "content": "historical_future_heatwave_and_station_holdout_spatial_hazard",
            "required_inputs": "trend_reconciliation|future_scenarios|spatial_model_cv|validated_hazard_surface",
            "map": True,
        },
        {
            "figure": "Figure 2",
            "content": "layered_observation_constraints_and_held_out_validation",
            "required_inputs": "official_energy_shape|polyu_hkust_holdout|population_holdout|local_heat_sensitivity|bca_eui|emsd_mass_balance",
            "map": False,
        },
        {
            "figure": "Figure 3",
            "content": "aligned_citywide_building_type_energy_response_and_direct_resilience_maps",
            "required_inputs": "full_stock|all_sector_model_inputs|building_heatwave_energy_resilience",
            "map": True,
        },
        {
            "figure": "Figure 4",
            "content": "three_district_building_type_hourly_energy_thermal_safety_and_pre_event_beri",
            "required_inputs": "model_inputs|prototype_hourly_response|building_resilience",
            "map": True,
        },
        {
            "figure": "Figure 5",
            "content": "type_heterogeneity_cross_stage_divergence_event_holdout_and_selection_consequences",
            "required_inputs": "type_event_stage_summary|cross_stage_ranking|beri_leave_one_event_out|paired_trajectory_uncertainty",
            "map": False,
        },
        {
            "figure": "Figure 6",
            "content": "normal_constrained_outage_recovery_and_future_change",
            "required_inputs": "city_event_stage_summary|future_scenario_comparison",
            "map": False,
        },
        {
            "figure": "Supplementary Figures S1-S6",
            "content": "methods_audit_and_same_layout_support_for_nonfeatured_events",
            "required_inputs": "sample_flow|prototype_registry|2018_2023_citywide_and_three_district_views|paired_event_response",
            "map": True,
        },
    ]
    pd.DataFrame(rows).to_csv(output / "figure_data_contract.csv", index=False, encoding="utf-8-sig")
    panel_rows = [
        ("Figure 1", "a", "Observed HKO heatwaves, 1980–2025", "01_hazard/heatwave_event_catalog.csv", "HKO events; bubble area=duration; colour=mean Tmax exceedance", "establish observed event timing, duration and intensity"),
        ("Figure 1", "b", "Annual frequency, span and cumulative intensity", "01_hazard/heatwave_event_catalog.csv|01_hazard/trend_reconciliation.csv", "annual HKO aggregates with pre-specified linear trends", "quantify three independent dimensions of historical heatwave growth"),
        ("Figure 1", "c", "Future heat-tail amplification across five models", "01_hazard/cmip6_tmax_future_model_summary.csv", "joint ΔP95 Tmax, very-hot-day frequency and maximum-duration trajectories under SSP2-4.5/SSP5-8.5", "show future forcing magnitude, event consequences and ensemble spread"),
        ("Figure 1", "d", "Independent Tmax validation", "01_hazard/wrf_tmax_distributional_spatiotemporal_holdout.csv", "paired raw/corrected quantile, P95 and threshold-frequency errors for 140 held-out station-months", "test maximum-temperature bias correction outside its fitting stations and years"),
        ("Figure 1", "e", "Independent Tmin validation", "01_hazard/wrf_tmin_distributional_spatiotemporal_holdout.csv", "paired raw/corrected quantile, P95 and threshold-frequency errors for 140 held-out station-months", "test minimum-temperature bias correction outside its fitting stations and years"),
        ("Figure 1", "f", "Observation-constrained 1-km P95 Tmax, 2011–2020", "01_hazard/building_heat_hazard_wrf1km.parquet", "2011–2020 warm-season P95 Tmax on 942 land cells", "resolve intra-urban heat hazard beyond nearest-station zones"),
        ("Figure 2", "a", "Official Hong Kong monthly demand shape", "02_building_response/official_domestic_monthly_shape_holdout_validation.csv", "official versus model monthly annual shares", "test residential seasonal timing"),
        ("Figure 2", "b", "Unseen Hong Kong campus loads", "02_building_response/result2_validation.json|02_building_response/polyu_campus_balance_point_blocked_cv.csv|02_building_response/hkust_whole_building_daily_validation.parquet", "held-out buildings and final 28-day block", "test local non-residential and campus load prediction"),
        ("Figure 2", "c", "Population allocation against major estates", "02_building_response/major_housing_estate_population_validation.csv", "540 estates reserved for external agreement testing", "test building-level population allocation while conserving census totals"),
        ("Figure 2", "d", "Observed and simulated heat sensitivity", "02_building_response/local_observed_vs_energyplus_heat_sensitivity.csv", "observed P05–P95 versus EnergyPlus prototype ranges", "test transfer of temperature-load response across local building cohorts"),
        ("Figure 2", "e", "Hot-humid commercial EUI plausibility", "02_building_response/bca_2024_hot_humid_commercial_eui_plausibility.csv", "Hong Kong model versus Singapore BCA P05–P95", "test annual magnitude across comparable hot-humid commercial uses"),
        ("Figure 2", "f", "EMSD 2023 annual stock-energy constraint", "02_building_response/official_energy_mass_balance_validation.csv", "cooling and non-cooling stock totals by calibration group", "document exact annual energy mass balance before event attribution"),
        ("Figure 3", "a", "Detailed Hong Kong building types", "02_building_response/all_sector_building_model_inputs.parquet", "one symbol per HK-BFETD footprint; 2022 inset", "locate detailed residential, commercial and mixed-use stock"),
        ("Figure 3", "b", "Event electricity", "03_energy_resilience/building_heatwave_energy_resilience.parquet", "normal supply; 22–29 July 2022 event energy", "map accumulated electricity demand during the featured heatwave"),
        ("Figure 3", "c", "Outage thermal safety", "03_energy_resilience/building_heatwave_energy_resilience.parquet", "time from outage onset to first 32 °C crossing, capped at 24 h", "map loss-of-cooling thermal endurance"),
        ("Figure 3", "d", "Baseline reference-suite Building Energy Resilience Index", "03_energy_resilience/building_reference_beri.parquet", "componentwise median across the three reference events", "map the common reference-suite energy-resilience rating"),
        ("Figure 5", "a", "Building-level electricity sensitivity", "03_energy_resilience/building_heatwave_energy_resilience.parquet", "class distributions by event", "measure heterogeneous relative electricity response"),
        ("Figure 5", "b", "Event energy and coincident peak", "03_energy_resilience/city_event_stage_summary.csv|03_energy_resilience/city_coincident_peak_summary.csv", "integrated GWh and hourly coincident peak change", "separate accumulated demand from system-timed peak stress"),
        ("Figure 5", "c", "Complete-event holdout of two-event BERI", "03_energy_resilience/beri_leave_one_event_out_summary.csv", "rank agreement and tie-aware low-10% overlap for each held-out event", "test whether a rating constructed from two events transfers to a complete event excluded from construction"),
        ("Figure 5", "d", "Cross-stage rank and low-score overlap", "03_energy_resilience/cross_stage_ranking_audit.csv|03_energy_resilience/cross_stage_cutoff_sensitivity.csv|03_energy_resilience/building_event_response_score.parquet", "lower triangle=all-building Spearman correlation; upper triangle=tie-aware expected low-10% cohort overlap; diagonal=raw physical medians; 5/10/20% cutoff robustness in Supplementary Data", "test whether normal demand, cooling adequacy, outage tolerance and recovery identify the same buildings"),
        ("Figure 5", "e", "Outage-attributable heat", "03_energy_resilience/class_event_stage_summary.csv", "24-h outage degree-hours above 26 °C minus the matched 100% rated-capacity reference, clipped hourly at zero", "quantify event-dependent thermal consequences"),
        ("Figure 5", "f", "Held-out 2023 burden captured by alternative fixed cohorts", "03_energy_resilience/beri_leave_one_event_out_capture.csv", "tie-aware expected share of five physical burdens captured by fixed 10% cohorts selected from 2018 and 2022", "compare multistage BERI with stage-specific and stock-attribute selection rules on an unseen event"),
        ("Figure 6", "a", "Hourly weather and peak-timing attribution", "02_building_response/citywide_coincident_hourly_response.parquet|03_energy_resilience/event_timing_decomposition.csv", "paired heatwave and weather-only counterfactual; exact weather/timing decomposition of differences between maxima", "isolate hourly load caused by heatwave weather and explain event peak ordering"),
        ("Figure 6", "b", "Prototype outage and recovery temperature", "02_building_response/energyplus_multizone_hourly_results.parquet", "24-h outage and matched 100% rated-capacity, grid-available temperature trajectories", "show heat accumulation and recovery against the physical comparator used for outage-attributable heat"),
        ("Figure 6", "c", "Five-model future electricity response", "02_building_response/future_energyplus_results.csv", "SSP2-4.5/SSP5-8.5 and mid-/late-century ensemble", "quantify future event response and model uncertainty"),
        ("Figure 6", "d", "Compound future stress", "03_energy_resilience/future_prototype_stage_summary.csv", "electricity increment versus outage-attributable heat", "show joint escalation of electricity and thermal stress"),
    ]
    regional_panel_specs = [
        ("Building type", "02_building_response/all_sector_building_model_inputs.parquet", "actual footprint map", "provide the local physical stock context"),
        ("Hourly electricity demand", "02_building_response/prototype_event_hourly_interface.parquet", "featured event local-peak hour mapped to each building", "show dynamic building electricity demand"),
        ("Indoor thermal safety", "02_building_response/prototype_event_hourly_interface.parquet", "hourly safety margin within the common-clock 24-h outage", "show dynamic loss-of-cooling thermal safety on one synchronized city time axis"),
        ("Building Energy Resilience Index", "03_energy_resilience/building_reference_beri.parquet", "fixed baseline reference-suite BERI", "show building-specific preparedness under the frozen stress suite"),
        ("Area-weighted load-intensity profile", "02_building_response/all_sector_building_model_inputs.parquet|02_building_response/prototype_event_hourly_interface.parquet", "area-weighted regional event trajectory with local peak marked", "link building maps to the time of aggregate demand"),
    ]
    regional_letters = "abcdefghijklmno"
    for region_index, region_name in enumerate(REGION_BOUNDS_2326):
        for panel_index, (title, frozen_data, statistic, role) in enumerate(regional_panel_specs):
            panel_rows.append(
                (
                    "Figure 4", regional_letters[5 * region_index + panel_index],
                    f"{region_name}: {title}", frozen_data,
                    f"{region_name}; {statistic}", role,
                )
            )
    pd.DataFrame(
        panel_rows,
        columns=["figure", "panel", "title", "frozen_data", "selection_or_statistic", "scientific_role"],
    ).to_csv(output / "figure_panel_data_contract.csv", index=False, encoding="utf-8-sig")


def select_result_slice(
    frame: pd.DataFrame,
    event_id: str | None,
    climate_period: str | None,
    climate_scenario: str | None,
    supply_stage: str | None,
    behavior_seed: int | None,
) -> tuple[pd.DataFrame, dict]:
    filters = {
        "event_id": event_id,
        "climate_period": climate_period,
        "climate_scenario": climate_scenario,
        "supply_stage": supply_stage,
        "behavior_seed": behavior_seed,
    }
    selected = frame.copy()
    for field, value in filters.items():
        if value is not None:
            selected = selected.loc[selected[field].astype(str).eq(str(value))].copy()
    grain = ["event_id", "climate_period", "climate_scenario", "supply_stage", "behavior_seed"]
    if selected.empty:
        raise ValueError("No building results remain after the requested map filters")
    if supply_stage is None:
        preferred = "outage_peak_24h"
        available = set(selected["supply_stage"].astype(str))
        selected = selected.loc[selected["supply_stage"].astype(str).eq(
            preferred if preferred in available else sorted(available)[0]
        )].copy()
    for field in ["climate_period", "climate_scenario", "behavior_seed"]:
        if filters[field] is None:
            value = sorted(selected[field].astype(str).unique())[0]
            selected = selected.loc[selected[field].astype(str).eq(value)].copy()
    if event_id is None:
        event_totals = selected.groupby("event_id", observed=True)["heatwave_energy_kwh"].sum()
        chosen_event = event_totals.sort_values(ascending=False, kind="mergesort").index[0]
        selected = selected.loc[selected["event_id"].eq(chosen_event)].copy()
    combinations = selected[grain].drop_duplicates()
    if len(combinations) != 1:
        examples = combinations.head(10).to_dict(orient="records")
        raise ValueError(
            "Map requires exactly one event/period/scenario/stage/behavior slice. "
            f"Remaining combinations={len(combinations)}; examples={examples}"
        )
    if selected["BUILDINGSTRUCTUREID"].duplicated().any():
        raise ValueError("Selected map slice contains duplicate building results")
    selection = combinations.iloc[0].to_dict()
    return selected, selection


def finite_limits(series: pd.Series) -> tuple[float, float]:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        raise ValueError(f"No finite values available for map field {series.name}")
    lower = float(values.quantile(0.02))
    upper = float(values.quantile(0.98))
    if not upper > lower:
        lower = float(values.min())
        upper = float(values.max())
    if not upper > lower:
        upper = lower + 1.0
    return lower, upper


def draw_continuous_map(
    axis: plt.Axes,
    colorbar_axis: plt.Axes,
    frame: gpd.GeoDataFrame,
    field: str,
    cmap: str,
    label: str,
    limits: tuple[float, float] | None = None,
) -> None:
    lower, upper = limits or finite_limits(frame[field])
    plot_arguments = {
        "ax": axis,
        "column": field,
        "cmap": cmap,
        "norm": Normalize(lower, upper),
        "linewidth": 0,
        "rasterized": True,
        "missing_kwds": {"color": "#e5e7eb"},
    }
    if frame.geometry.geom_type.eq("Point").all():
        plot_arguments["markersize"] = 0.72
        plot_arguments["alpha"] = 0.88
    frame.plot(**plot_arguments)
    scalar = plt.cm.ScalarMappable(norm=Normalize(lower, upper), cmap=cmap)
    scalar.set_array([])
    colorbar = axis.figure.colorbar(scalar, cax=colorbar_axis)
    colorbar.set_label(label, labelpad=2)
    colorbar.ax.tick_params(pad=2)
    colorbar.outline.set_visible(False)


def render_citywide_triptych(
    full_stock: gpd.GeoDataFrame,
    results: gpd.GeoDataFrame,
    energy_field: str,
    resilience_field: str,
    beri_field: str,
    selection: dict,
    destination: Path,
) -> None:
    for field in [energy_field, resilience_field, beri_field]:
        if field not in results.columns:
            raise ValueError(f"Map field missing from Result 3: {field}")
    fig = plt.figure(figsize=(13.8, 10.6), constrained_layout=False)
    # Use a common map column width for all four panels.  The upper-left side
    # column holds the building-type key; the lower-left side column holds the
    # panel-c colorbar.  Their left edges therefore align without shrinking
    # panel (a) relative to the three continuous maps.
    outer = fig.add_gridspec(
        2, 5, left=0.040, right=0.985, bottom=0.085, top=0.945,
        width_ratios=[1.0, 0.28, 0.055, 1.0, 0.055],
        wspace=0.045, hspace=0.18,
    )
    axes = [
        fig.add_subplot(outer[0, 0]),
        fig.add_subplot(outer[0, 3]),
        fig.add_subplot(outer[1, 0]),
        fig.add_subplot(outer[1, 3]),
    ]
    colorbar_axes = [
        fig.add_subplot(outer[0, 1]),
        fig.add_subplot(outer[0, 4]),
        fig.add_subplot(outer[1, 1]),
        fig.add_subplot(outer[1, 4]),
    ]
    land = gpd.read_parquet(HMA_BOUNDARIES).to_crs(full_stock.crs)
    common_bounds = map_extent(full_stock, pad_fraction=0.01)
    for axis in axes:
        axis.set_facecolor("#CBD3D0")
        land.plot(ax=axis, color="#D5DBD7", edgecolor="#687274", linewidth=0.22, zorder=0)

    full_stock = full_stock.copy()
    full_stock["main_class"] = english_label(full_stock["hkbfetd_main_class"])
    full_stock["_context_color"] = full_stock["main_class"].map(
        {
            "Industrial": TYPE_COLORS["Industrial"],
            "Non-assessed": TYPE_COLORS["Non-assessed"],
        }
    ).fillna("#BFC5C3")
    detailed = animation_display_type(results)
    full_stock_points = full_stock.copy()
    full_stock_points.geometry = full_stock_points.geometry.centroid
    detailed_points = detailed.copy()
    detailed_points.geometry = detailed_points.geometry.centroid
    full_stock_points.plot(
        ax=axes[0], color=full_stock_points["_context_color"], linewidth=0,
        markersize=0.55, alpha=0.76, rasterized=True, zorder=1,
    )
    detailed_points.plot(
        ax=axes[0], color=detailed_points["_color"], linewidth=0,
        markersize=0.62, alpha=0.90, rasterized=True, zorder=2,
    )
    handles = [
        Patch(facecolor=color, edgecolor="none", label=label)
        for label, color in ANIMATION_TYPE_COLORS.items()
        if label in set(detailed["_display_type"].dropna())
    ]
    handles.extend([
        Patch(facecolor=TYPE_COLORS["Industrial"], edgecolor="none", label="Industrial context"),
        Patch(facecolor=TYPE_COLORS["Non-assessed"], edgecolor="none", label="Non-assessed context"),
    ])
    panel_title(axes[0], "a", "Detailed Hong Kong building types")

    draw_continuous_map(
        axes[1],
        colorbar_axes[1],
        gpd.GeoDataFrame(results.drop(columns="geometry"), geometry=results.geometry.centroid, crs=results.crs),
        energy_field,
        "YlOrRd",
        "Heatwave event electricity (kWh building⁻¹)",
    )
    event_id = str(selection["event_id"])
    event_label = EVENT_LABELS.get(event_id, event_id)
    panel_title(axes[1], "b", f"Event electricity, {event_label}")

    resilience_is_time = resilience_field in {
        "time_to_thermal_threshold_hours", "safety_resistance_hours_to_32c",
    }
    resilience_cmap = "YlGn" if resilience_is_time else "YlOrRd"
    resilience_label = (
        "Time to first 32 °C crossing (h)"
        if resilience_field == "safety_resistance_hours_to_32c"
        else "Time to 26 °C during 24-h outage (h)"
        if resilience_field == "time_to_thermal_threshold_hours"
        else "Outage-attributable thermal stress (°C h)"
    )
    draw_continuous_map(
        axes[2],
        colorbar_axes[2],
        gpd.GeoDataFrame(results.drop(columns="geometry"), geometry=results.geometry.centroid, crs=results.crs),
        resilience_field,
        resilience_cmap,
        resilience_label,
        limits=(0.0, 24.0)
        if resilience_field == "safety_resistance_hours_to_32c" else None,
    )
    panel_title(axes[2], "c", f"Time to first 32 °C crossing, {event_label}")
    draw_continuous_map(
        axes[3],
        colorbar_axes[3],
        gpd.GeoDataFrame(results.drop(columns="geometry"), geometry=results.geometry.centroid, crs=results.crs),
        beri_field, "RdYlGn", "BERI (0–100)", limits=(0.0, 100.0),
    )
    panel_title(axes[3], "d", "Reference-suite Building Energy Resilience Index")

    for axis in axes:
        axis.set_xlim(common_bounds[0], common_bounds[2])
        axis.set_ylim(common_bounds[1], common_bounds[3])
        axis.set_axis_off()
        axis.set_aspect("equal")
    fig.canvas.draw()
    legend_axis = colorbar_axes[0]
    legend_axis.set_axis_off()
    legend_axis.legend(
        handles=handles, loc="center left", ncol=1, frameon=False,
        bbox_to_anchor=(0.0, 0.50), borderaxespad=0.0,
        handlelength=0.95, handletextpad=0.35, labelspacing=0.28,
        columnspacing=0.0,
    )
    for panel_index in range(1, 4):
        slot = colorbar_axes[panel_index].get_position()
        reference = axes[panel_index].get_position()
        width_fraction = 0.18 if panel_index == 2 else 0.38
        colorbar_axes[panel_index].set_position(
            [slot.x0 + 0.02 * slot.width, reference.y0,
             width_fraction * slot.width, reference.height]
        )
    fig.text(
        0.035, 0.042,
        f"Territory view: {EVENT_SELECTION_NOTES.get(event_id, 'reference stress event')} ({event_label}). "
        "Panels (b,c) are event outcomes; panel (d) is fixed across events.",
        ha="left", va="center", color="#4B5563",
    )
    save_figure(fig, destination)
    plt.close(fig)


def outcome_table(path: Path, outcome: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame.loc[frame["outcome"].eq(outcome)].copy()


def render_figure_1(full_stock: gpd.GeoDataFrame, destination: Path) -> None:
    """Observed events, annual trends, model spread, holdout skill and hazard."""
    trend = pd.read_csv(HAZARD_DIR / "trend_reconciliation.csv")
    events = pd.read_csv(HAZARD_DIR / "heatwave_event_catalog.csv")
    model_future = pd.read_csv(HAZARD_DIR / "cmip6_tmax_future_model_summary.csv")
    validation = pd.read_csv(HAZARD_DIR / "wrf_distributional_spatial_bias_validation.csv")
    hazard = pd.read_parquet(HAZARD_DIR / "building_heat_hazard_wrf1km.parquet")

    fig = plt.figure(figsize=(13.8, 13.4), constrained_layout=False)
    layout = fig.add_gridspec(
        3, 2, left=0.075, right=0.975, bottom=0.060, top=0.965,
        wspace=0.29, hspace=0.36,
    )
    event_grid = layout[0, 0].subgridspec(1, 2, width_ratios=[1.0, 0.055], wspace=0.035)
    event_axis = fig.add_subplot(event_grid[0, 0])
    event_bar_axis = fig.add_subplot(event_grid[0, 1])
    trend_grid = layout[0, 1].subgridspec(3, 1, hspace=0.14)
    trend_axes = [fig.add_subplot(trend_grid[row, 0]) for row in range(3)]
    projection_axis = fig.add_subplot(layout[1, 0])
    validation_tmax_grid = layout[1, 1].subgridspec(3, 1, hspace=0.25)
    validation_tmax_axes = [fig.add_subplot(validation_tmax_grid[row, 0]) for row in range(3)]
    validation_tmin_grid = layout[2, 0].subgridspec(3, 1, hspace=0.25)
    validation_tmin_axes = [fig.add_subplot(validation_tmin_grid[row, 0]) for row in range(3)]
    hazard_axis = fig.add_subplot(layout[2, 1])

    hko_events = events.loc[
        events["station_code"].eq("HKO") & events["start_year"].between(1980, 2025)
    ].copy()
    hko_events["start_date"] = pd.to_datetime(hko_events["start_date"])
    duration_size = lambda values: 10.0 + 7.0 * np.asarray(values, dtype=float)
    event_scatter = event_axis.scatter(
        hko_events["start_year"], hko_events["start_date"].dt.month,
        s=duration_size(hko_events["duration_days"]),
        c=hko_events["mean_intensity_c"], cmap="magma", alpha=0.88,
        edgecolor="white", linewidth=0.42,
    )
    selected_parent_ids = set(EVENT_PARENT_HKO.values())
    selected_events = hko_events.loc[hko_events["event_id"].isin(selected_parent_ids)].copy()
    event_axis.scatter(
        selected_events["start_year"], selected_events["start_date"].dt.month,
        s=duration_size(selected_events["duration_days"]) + 58,
        facecolor="none", edgecolor="#00A6A6", linewidth=2.1, zorder=5,
    )
    event_axis.set_yticks([5, 6, 7, 8, 9, 10], ["May", "Jun", "Jul", "Aug", "Sep", "Oct"])
    event_axis.set_xlim(1978, 2027)
    event_axis.set_ylim(4.55, 10.95)
    event_axis.set_xlabel("Event start year")
    event_axis.set_ylabel("Start month")
    event_bar = fig.colorbar(event_scatter, cax=event_bar_axis)
    # A vertical colorbar label competes directly with the three compact
    # y-axis labels in panel (b); place the descriptor below its own bar.
    event_bar.set_label("")
    event_bar.ax.text(
        0.5, -0.035, "Mean Tmax excess\n(°C)",
        transform=event_bar.ax.transAxes, ha="center", va="top",
    )
    event_bar.outline.set_visible(False)
    duration_handles = [
        Line2D([], [], linestyle="", marker="o", markerfacecolor="#8b4b80",
               markeredgecolor="white", markersize=np.sqrt(duration_size([days])[0]),
               label=f"{days} d")
        for days in [3, 10, 20]
    ]
    duration_handles.append(
        Line2D([], [], linestyle="", marker="o", markerfacecolor="none",
               markeredgecolor="#00A6A6", markeredgewidth=2.0, markersize=8.5,
               label="Hourly building stress suite")
    )
    event_axis.legend(
        handles=duration_handles, title="Bubble area: event duration", ncol=2,
        loc="upper left", frameon=False, handletextpad=0.25, columnspacing=0.8,
    )
    panel_title(event_axis, "a", "Observed HKO heatwaves, 1980–2025")

    years = np.arange(1980, 2026)
    annual = hko_events.groupby("start_year", observed=True).agg(
        event_count=("event_id", "nunique"),
        event_span_days=("duration_days", "sum"),
        cumulative_intensity=("cumulative_intensity_c_days", "sum"),
    ).reindex(years, fill_value=0)
    trend_specs = [
        ("event_count", "li_event_count", "Events\n(count yr⁻¹)", "events yr⁻²", "#4C78A8"),
        ("event_span_days", "li_event_span_days", "Event span\n(days yr⁻¹)", "days yr⁻¹", "#D95F59"),
        ("cumulative_intensity", "li_cumulative_intensity_c_days", "Cumulative\nintensity (°C d)", "°C d yr⁻¹", "#7A5195"),
    ]
    for row, (field, metric, ylabel, units, color) in enumerate(trend_specs):
        axis = trend_axes[row]
        values = annual[field].to_numpy(dtype=float)
        axis.plot(years, values, color=color, lw=0.85, alpha=0.45)
        axis.scatter(years, values, s=9, color=color, alpha=0.62, edgecolor="none")
        coefficients = np.polyfit(years, values, 1)
        axis.plot(years, np.polyval(coefficients, years), color=color, lw=2.0)
        estimate = trend.loc[
            trend["scope"].eq("HKO_extended_1980_2025") & trend["metric"].eq(metric)
        ].iloc[0]
        ci = 1.96 * float(estimate["standard_error"])
        axis.text(
            0.018, 0.82,
            f"slope {float(estimate['slope_per_year']):+.3f} ± {ci:.3f} {units}",
            transform=axis.transAxes, ha="left", va="top", color=color, zorder=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
        )
        axis.set_ylabel(ylabel, labelpad=3)
        axis.set_xlim(1979, 2026)
        axis.grid(axis="y", color="#d9dbdc", linewidth=0.5, alpha=0.65)
        axis.spines[["top", "right"]].set_visible(False)
        if row < 2:
            axis.tick_params(axis="x", labelbottom=False)
        else:
            axis.set_xlabel("Year")
    panel_title(trend_axes[0], "b", "Annual frequency, span and cumulative intensity")

    future = model_future.copy()
    scenario_order = ["ssp245", "ssp585"]
    period_order = ["mid_century_2041_2060", "late_century_2081_2100"]
    scenario_colors = {"ssp245": "#2C7FB8", "ssp585": "#D1495B"}
    period_markers = {
        "mid_century_2041_2060": "o",
        "late_century_2081_2100": "^",
    }
    duration_size = lambda values: 16.0 + 1.45 * np.asarray(values, dtype=float)
    for scenario in scenario_order:
        color = scenario_colors[scenario]
        scenario_frame = future.loc[future["experiment_id"].eq(scenario)].copy()
        for source_id, model in scenario_frame.groupby("source_id", observed=True):
            pair = model.set_index("future_period").reindex(period_order)
            if pair[["delta_p95_tmax_c", "very_hot_days_per_summer"]].isna().any().any():
                continue
            projection_axis.annotate(
                "",
                xy=(pair.iloc[1]["delta_p95_tmax_c"], pair.iloc[1]["very_hot_days_per_summer"]),
                xytext=(pair.iloc[0]["delta_p95_tmax_c"], pair.iloc[0]["very_hot_days_per_summer"]),
                arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.05, "alpha": 0.42},
                zorder=1,
            )
        for period in period_order:
            group = scenario_frame.loc[scenario_frame["future_period"].eq(period)]
            projection_axis.scatter(
                group["delta_p95_tmax_c"], group["very_hot_days_per_summer"],
                s=duration_size(group["maximum_heatwave_duration_days"]),
                marker=period_markers[period], color=color, alpha=0.64,
                edgecolor="white", linewidth=0.55, zorder=3,
            )
            projection_axis.scatter(
                [group["delta_p95_tmax_c"].median()],
                [group["very_hot_days_per_summer"].median()],
                s=duration_size([group["maximum_heatwave_duration_days"].median()])[0] + 42,
                marker="D", facecolor="white", edgecolor=color, linewidth=1.7, zorder=4,
            )
    projection_axis.axvline(0, color="#6b7280", lw=0.75)
    projection_axis.set_xlim(0.55, 5.55)
    projection_axis.set_ylim(24, 117)
    projection_axis.set_xlabel("Δ summer P95 Tmax vs 1992–2014 (°C)")
    projection_axis.set_ylabel("Very hot days\nper summer")
    projection_axis.grid(color="#d9dbdc", linewidth=0.5, alpha=0.62)
    projection_handles = [
        Line2D([], [], color=scenario_colors["ssp245"], marker="o", linestyle="-", label="SSP2–4.5"),
        Line2D([], [], color=scenario_colors["ssp585"], marker="o", linestyle="-", label="SSP5–8.5"),
        Line2D([], [], color="#475569", marker="o", linestyle="", label="2041–2060"),
        Line2D([], [], color="#475569", marker="^", linestyle="", label="2081–2100"),
        Line2D([], [], color="#475569", marker="D", markerfacecolor="white", linestyle="", label="Ensemble median"),
    ]
    for days in [10, 30, 60]:
        projection_handles.append(
            Line2D([], [], color="#64748B", marker="o", markerfacecolor="#CBD5E1", linestyle="",
                   markersize=np.sqrt(duration_size([days])[0]), label=f"{days} d max event")
        )
    projection_axis.legend(
        handles=projection_handles, frameon=False, loc="lower right", ncol=2,
        handletextpad=0.35, columnspacing=0.8, labelspacing=0.38,
    )
    panel_title(projection_axis, "c", "Future heat-tail amplification across five models")

    station_holdouts = {
        "tmax": pd.read_csv(HAZARD_DIR / "wrf_tmax_distributional_spatiotemporal_holdout.csv"),
        "tmin": pd.read_csv(HAZARD_DIR / "wrf_tmin_distributional_spatiotemporal_holdout.csv"),
    }
    validation_specs = [
        (validation_tmax_axes, "tmax", "d", "Independent Tmax validation"),
        (validation_tmin_axes, "tmin", "e", "Independent Tmin validation"),
    ]
    for axes, variable, letter, title in validation_specs:
        part = station_holdouts[variable]
        metric_arrays = [
            (part["raw_quantile_mae_c"].to_numpy(float),
             part["corrected_quantile_mae_c"].to_numpy(float), "Quantile MAE\n(°C)"),
            (part["raw_p95_error_c"].abs().to_numpy(float),
             part["corrected_p95_error_c"].abs().to_numpy(float), "|P95 error|\n(°C)"),
            ((part["raw_threshold_fraction"] - part["observed_threshold_fraction"]).abs().to_numpy(float) * 100,
             (part["corrected_threshold_fraction"] - part["observed_threshold_fraction"]).abs().to_numpy(float) * 100,
             "Threshold error\n(percentage points)"),
        ]
        for row, (axis, (raw_values, corrected_values, ylabel)) in enumerate(zip(axes, metric_arrays)):
            violins = axis.violinplot(
                [raw_values, corrected_values], positions=[0, 1], widths=0.66,
                showmeans=False, showmedians=False, showextrema=False, bw_method=0.28,
            )
            for body, color in zip(violins["bodies"], ["#94A3B8", "#0F766E"]):
                body.set_facecolor(color)
                body.set_edgecolor("none")
                body.set_alpha(0.34)
            for raw_value, corrected_value in zip(raw_values, corrected_values):
                axis.plot(
                    [0, 1], [raw_value, corrected_value], color="#64748B",
                    lw=0.36, alpha=0.075, zorder=1,
                )
            for position, values, color in zip(
                [0, 1], [raw_values, corrected_values], ["#64748B", "#0F766E"]
            ):
                q10, q25, median, q75, q90 = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90])
                axis.vlines(position, q10, q90, color=color, lw=1.1, zorder=3)
                axis.vlines(position, q25, q75, color=color, lw=5.0, zorder=4)
                axis.scatter(
                    [position], [median], s=24, facecolor="white",
                    edgecolor=color, linewidth=1.0, zorder=5,
                )
            raw_median = float(np.median(raw_values))
            corrected_median = float(np.median(corrected_values))
            median_change = 100 * (corrected_median / raw_median - 1)
            improved_share = 100 * float(np.mean(corrected_values < raw_values))
            axis.text(
                0.985, 0.96, f"median {median_change:+.0f}%  |  {improved_share:.0f}% improved",
                transform=axis.transAxes, ha="right", va="top", color="#334155", zorder=10,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.2},
            )
            axis.set_xlim(-0.43, 1.43)
            axis.set_ylim(0, max(raw_values.max(), corrected_values.max()) * 1.30)
            axis.set_xticks([0, 1])
            if row < 2:
                axis.tick_params(axis="x", labelbottom=False)
            else:
                axis.set_xticklabels(["Raw WRF", "Bias corrected"])
                axis.get_xticklabels()[0].set_color("#64748B")
                axis.get_xticklabels()[1].set_color("#0F766E")
            axis.set_ylabel(ylabel, labelpad=4)
            axis.grid(axis="y", color="#d9dbdc", linewidth=0.5, alpha=0.58)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
            if row == 0:
                axis.text(
                    0.015, 0.96, "35 unseen stations × Jun–Sep\n2016–2020 (n=140 station-months)",
                    transform=axis.transAxes, ha="left", va="top", color="#475569", zorder=10,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.2},
                )
                panel_title(axis, letter, title)

    grid = hazard.drop_duplicates(["wrf_south_north", "wrf_west_east"])
    land = gpd.read_parquet(HMA_BOUNDARIES).to_crs(4326)
    hazard_axis.set_facecolor("#CBD3D0")
    land.plot(
        ax=hazard_axis, color="#D5DBD7", edgecolor="#596466",
        linewidth=0.28, rasterized=True, zorder=0,
    )
    sc = hazard_axis.scatter(
        grid["wrf_grid_longitude"], grid["wrf_grid_latitude"],
        c=grid["bias_corrected_tmax_p95_c"], s=18, marker="s",
        linewidth=0, cmap="inferno", zorder=1,
    )
    cb = fig.colorbar(sc, ax=hazard_axis, fraction=0.027, pad=0.015)
    cb.set_label("Warm-season P95 Tmax (°C)")
    cb.outline.set_visible(False)
    hazard_axis.text(
        0.015, 0.025, "Static June–September distribution, 2011–2020\n942 land cells; buildings linked to nearest land cell",
        transform=hazard_axis.transAxes, ha="left", va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 3.0},
    )
    hazard_axis.annotate(
        "N", xy=(0.95, 0.92), xytext=(0.95, 0.80), xycoords="axes fraction",
        ha="center", arrowprops={"arrowstyle": "-|>", "color": "#263238", "lw": 1.1},
    )
    hazard_axis.set_xlabel("Longitude (°E)")
    hazard_axis.set_ylabel("Latitude (°N)")
    hazard_axis.set_aspect("equal", adjustable="box")
    panel_title(hazard_axis, "f", "Observation-constrained 1-km P95 Tmax, 2011–2020")
    candidate_path = ROOT / "output" / "heatwave_day3" / "weather_event_catalog.csv"
    candidates = pd.read_csv(candidate_path)
    candidates["start_date"] = pd.to_datetime(candidates["start_date"])
    candidates["end_date"] = pd.to_datetime(candidates["end_date"])
    percentile_events = hko_events.copy()
    percentile_events["end_date"] = pd.to_datetime(percentile_events["end_date"])
    parent_ids = []
    for candidate in candidates.itertuples(index=False):
        overlap = percentile_events.loc[
            percentile_events["start_date"].le(candidate.end_date)
            & percentile_events["end_date"].ge(candidate.start_date)
        ].copy()
        if overlap.empty:
            parent_ids.append(None)
            continue
        overlap["overlap_days"] = overlap.apply(
            lambda row: (
                min(row["end_date"], candidate.end_date)
                - max(row["start_date"], candidate.start_date)
            ).days + 1,
            axis=1,
        )
        parent_ids.append(str(overlap.sort_values("overlap_days", ascending=False).iloc[0]["event_id"]))
    candidates["parent_percentile_event_id"] = parent_ids
    candidates["fixed_threshold_severity_rank"] = candidates["severity_score"].rank(
        ascending=False, method="min"
    ).astype(int)
    candidates["selection_was_pre_outcome"] = True
    candidates["candidate_scope"] = (
        "all 2016–2025 HKO >=33 C runs with complete multivariable hourly forcing"
    )
    candidates.to_csv(
        destination.parent / "historical_event_and_building_stress_selection_audit.csv",
        index=False, encoding="utf-8-sig",
    )
    save_figure(fig, destination)
    plt.close(fig)


def render_figure_2(destination: Path) -> None:
    """Observation constraints, held-out tests and operating-response evidence."""
    validation = json.loads((RESPONSE_DIR / "result2_validation.json").read_text(encoding="utf-8"))
    fig = plt.figure(figsize=(13.6, 14.0), constrained_layout=False)
    layout = fig.add_gridspec(
        3, 2, left=0.12, right=0.965, bottom=0.065, top=0.965,
        hspace=0.40, wspace=0.38,
    )
    monthly_axis = fig.add_subplot(layout[0, 0])
    campus_axis = fig.add_subplot(layout[0, 1])
    population_axis = fig.add_subplot(layout[1, 0])
    sensitivity_axis = fig.add_subplot(layout[1, 1])
    bca_axis = fig.add_subplot(layout[2, 0])
    mass_axis = fig.add_subplot(layout[2, 1])

    monthly = pd.read_csv(RESPONSE_DIR / "official_domestic_monthly_shape_holdout_validation.csv")
    month = monthly["month"].to_numpy(dtype=int)
    observed_share = 100 * monthly["official_share"].to_numpy(dtype=float)
    modeled_share = 100 * monthly["modeled_share"].to_numpy(dtype=float)
    monthly_axis.fill_between(
        month, observed_share, modeled_share, color="#B8C4CE", alpha=0.32,
    )
    monthly_axis.plot(month, observed_share, color="#1F4E79", lw=2.2, marker="o",
                      ms=4.2, label="Official 2023")
    monthly_axis.plot(month, modeled_share, color="#D97706", lw=2.2, marker="D",
                      ms=3.8, label="Model holdout")
    correlation = float(np.corrcoef(observed_share, modeled_share)[0, 1])
    cvrmse = 100 * float(np.sqrt(np.mean((modeled_share - observed_share) ** 2)) / observed_share.mean())
    monthly_axis.text(
        0.03, 0.96, f"r = {correlation:.3f}\nCVRMSE = {cvrmse:.1f}%",
        transform=monthly_axis.transAxes, ha="left", va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.5},
    )
    monthly_axis.set_xticks(month)
    monthly_axis.set_xlabel("Month")
    monthly_axis.set_ylabel("Share of annual domestic\nelectricity (%)")
    monthly_axis.set_ylim(0, 1.18 * max(observed_share.max(), modeled_share.max()))
    monthly_axis.legend(
        frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.76),
        borderaxespad=0.0, alignment="left",
    )
    monthly_axis.grid(axis="y", color="#d9dbdc", linewidth=0.5, alpha=0.65)
    panel_title(monthly_axis, "a", "Official Hong Kong monthly demand shape")

    polyu = validation["polyu_campus"]
    hkust = validation["hkust_whole_building"]
    campus_points = pd.DataFrame(
        [
            {"label": "PolyU\nbuildings", **polyu["building_holdout_validation"]},
            {"label": "PolyU\nfinal 28 d", **polyu["final_28_day_time_holdout_validation"]},
            {"label": "HKUST\nbuildings", "nmbe_pct": hkust["held_out_nmbe_pct"],
             "cvrmse_pct": hkust["held_out_cvrmse_pct"]},
        ]
    )
    campus_colors = ["#00798C", "#6A994E", "#8E5EA2"]
    for color, record in zip(campus_colors, campus_points.itertuples(index=False)):
        campus_axis.scatter(
            abs(float(record.nmbe_pct)), float(record.cvrmse_pct), s=115,
            color=color, edgecolor="white", linewidth=0.8, zorder=3,
        )
        campus_axis.annotate(
            str(record.label), (abs(float(record.nmbe_pct)), float(record.cvrmse_pct)),
            xytext=(6, 5), textcoords="offset points", ha="left", va="bottom",
        )
    campus_axis.set_xlim(0, max(4.0, 1.35 * campus_points["nmbe_pct"].abs().max()))
    campus_axis.set_ylim(0, 1.22 * campus_points["cvrmse_pct"].max())
    campus_axis.set_xlabel("Absolute normalized mean bias, |NMBE| (%)  ← lower is better")
    campus_axis.set_ylabel("Load-shape dispersion,\nCVRMSE (%)")
    campus_axis.text(
        0.03, 0.05, "all three holdouts: |NMBE| < 3%, CVRMSE < 24%",
        transform=campus_axis.transAxes, ha="left", va="bottom", color="#18705C",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.0},
    )
    campus_axis.grid(color="#d9dbdc", linewidth=0.5, alpha=0.65)
    panel_title(campus_axis, "b", "Unseen Hong Kong campus loads")

    estate = pd.read_csv(RESPONSE_DIR / "major_housing_estate_population_validation.csv")
    observed = estate["official_population"].to_numpy(dtype=float)
    predicted = estate["allocated_population__ssug_landuse_residential_gfa"].to_numpy(dtype=float)
    positive = (observed > 0) & (predicted > 0)
    observed, predicted = observed[positive], predicted[positive]
    population_hex = population_axis.hexbin(
        observed, predicted, gridsize=27, bins="log", mincnt=1,
        xscale="log", yscale="log", cmap="viridis", linewidths=0.12,
    )
    bounds = [min(observed.min(), predicted.min()), max(observed.max(), predicted.max())]
    population_axis.plot(bounds, bounds, color="#D1495B", lw=1.3, ls="--", label="1:1")
    population_axis.set_xscale("log")
    population_axis.set_yscale("log")
    population_axis.set_xlim(bounds)
    population_axis.set_ylim(bounds)
    population_axis.set_xlabel("Official estate population")
    population_axis.set_ylabel("Allocated building population")
    population_axis.text(
        0.04, 0.96, "540 estates\nRMAE 4.13%\nSpearman 0.986",
        transform=population_axis.transAxes, ha="left", va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.5},
    )
    population_axis.legend(frameon=False, loc="lower right")
    population_colorbar = fig.colorbar(
        population_hex, ax=population_axis, fraction=0.042, pad=0.012,
    )
    population_colorbar.set_label("Estate density")
    population_colorbar.outline.set_visible(False)
    panel_title(population_axis, "c", "Population allocation against major estates")

    sensitivity = pd.read_csv(RESPONSE_DIR / "local_observed_vs_energyplus_heat_sensitivity.csv")
    sensitivity_labels = {
        ("PolyU full campus", "nonresidential campus buildings"): "PolyU non-residential",
        ("HKUST full archive", "campus residence proxy"): "HKUST residence proxy",
        ("HKUST full archive", "campus nonresidential"): "HKUST non-residential",
    }
    sensitivity["label"] = [
        sensitivity_labels.get((str(row.dataset), str(row.observed_scope)), str(row.observed_scope))
        for row in sensitivity.itertuples(index=False)
    ]
    y_sensitivity = np.arange(len(sensitivity))
    for row, record in enumerate(sensitivity.itertuples(index=False)):
        sensitivity_axis.plot(
            [record.observed_slope_p05, record.observed_slope_p95],
            [row + 0.11, row + 0.11], color="#167D8D", lw=5.2,
            solid_capstyle="round", alpha=0.80,
        )
        sensitivity_axis.scatter(
            record.observed_slope_median, row + 0.11, s=42, color="white",
            edgecolor="#167D8D", linewidth=1.4, zorder=4,
        )
        sensitivity_axis.plot(
            [record.model_slope_min, record.model_slope_max],
            [row - 0.11, row - 0.11], color="#D55E4C", lw=5.2,
            solid_capstyle="round", alpha=0.76,
        )
        sensitivity_axis.scatter(
            record.model_slope_median, row - 0.11, marker="D", s=38,
            color="white", edgecolor="#D55E4C", linewidth=1.4, zorder=4,
        )
    sensitivity["display_label"] = sensitivity["label"].replace(
        {
            "PolyU non-residential": "PolyU\nnon-residential",
            "HKUST residence proxy": "HKUST residence\nproxy",
            "HKUST non-residential": "HKUST\nnon-residential",
        }
    )
    sensitivity_axis.set_yticks(y_sensitivity, sensitivity["display_label"])
    sensitivity_axis.invert_yaxis()
    sensitivity_axis.set_ylim(len(sensitivity) - 0.45, -1.15)
    sensitivity_axis.set_xscale("log")
    sensitivity_axis.set_xlabel("Relative daily electricity response per °C above 24 °C")
    sensitivity_axis.grid(axis="x", color="#d9dbdc", linewidth=0.5, alpha=0.65)
    sensitivity_axis.legend(
        handles=[
            Line2D([], [], color="#167D8D", lw=5, marker="o", markerfacecolor="white", label="Observed P05–P95; median"),
            Line2D([], [], color="#D55E4C", lw=5, marker="D", markerfacecolor="white", label="EnergyPlus prototype range; median"),
        ],
        frameon=False, loc="upper center", ncol=2,
        bbox_to_anchor=(0.50, 0.99), columnspacing=0.9, handletextpad=0.35,
    )
    panel_title(sensitivity_axis, "d", "Observed and simulated heat sensitivity")

    bca = pd.read_csv(RESPONSE_DIR / "bca_2024_hot_humid_commercial_eui_plausibility.csv")
    bca_names = {
        "Office": "Office", "Retail": "Retail", "Hotel": "Hotel",
        "Autonomous University": "University", "Hospital or Specialist Clinic": "Healthcare",
        "Mixed Development": "Mixed development",
    }
    bca["label"] = bca["bca_building_type"].map(bca_names).fillna(bca["bca_building_type"])
    bca = bca.drop_duplicates("label").sort_values("modeled_calibrated_eui_kwh_m2_year")
    y_bca = np.arange(len(bca))
    for row, record in enumerate(bca.itertuples(index=False)):
        bca_axis.plot([record.bca_p05_eui, record.bca_p95_eui], [row, row],
                      color="#A7B4BE", lw=5.0, solid_capstyle="round")
        bca_axis.scatter(record.bca_median_eui, row, marker="|", s=110,
                         color="#374151", linewidth=1.8, zorder=3)
        bca_axis.scatter(
            record.modeled_calibrated_eui_kwh_m2_year, row, marker="D", s=50,
            color="#17806D" if record.inside_bca_p05_p95 else "#C2413B",
            edgecolor="white", linewidth=0.6, zorder=4,
        )
    bca_axis.set_yticks(y_bca, bca["label"])
    bca_axis.set_xlabel("Annual EUI (kWh m⁻² yr⁻¹)")
    bca_axis.legend(
        handles=[
            Line2D([], [], color="#A7B4BE", lw=5, label="Singapore BCA P05–P95"),
            Line2D([], [], linestyle="", marker="|", color="#374151", markersize=10, label="BCA median"),
            Line2D([], [], linestyle="", marker="D", color="#17806D", markersize=6, label="Hong Kong model"),
        ],
        frameon=False, loc="lower right",
    )
    bca_axis.grid(axis="x", color="#d9dbdc", linewidth=0.5, alpha=0.65)
    panel_title(bca_axis, "e", "Hot-humid commercial EUI plausibility")

    mass = pd.read_csv(RESPONSE_DIR / "official_energy_mass_balance_validation.csv")
    factors = pd.read_csv(RESPONSE_DIR / "official_energy_calibration_factors.csv")
    mass_labels = {
        "commercial_sector_total": "Commercial",
        "public_housing": "Public rental",
        "subsidised_sale": "Subsidised sale",
        "private_housing": "Private housing",
        "other_housing": "Other housing",
    }
    mass["label"] = mass["calibration_group"].map(mass_labels)
    raw = factors.assign(
        raw_stock_total_kwh=(
            factors["raw_stock_cooling_kwh"] + factors["raw_stock_non_cooling_kwh"]
        )
    ).groupby("calibration_group", as_index=False, observed=True)[
        "raw_stock_total_kwh"
    ].sum()
    mass = mass.merge(raw, on="calibration_group", how="left", validate="one_to_one")
    mass = mass.sort_values("official_total_kwh", ascending=True).reset_index(drop=True)
    y_mass = np.arange(len(mass))
    official_total = mass["official_total_kwh"].to_numpy(dtype=float) / 1e9
    cooling_share = (
        mass["official_cooling_kwh"].to_numpy(dtype=float)
        / mass["official_total_kwh"].to_numpy(dtype=float)
    )
    calibrated_total = mass["calibrated_total_kwh"].to_numpy(dtype=float) / 1e9
    raw_total = mass["raw_stock_total_kwh"].to_numpy(dtype=float) / 1e9
    sizes = 75 + 440 * cooling_share
    mass_axis.scatter(
        official_total, y_mass, s=sizes, color="#305F72", alpha=0.78,
        edgecolor="white", linewidth=0.7, label="Official target; area = cooling share",
    )
    for raw_value, official_value, y_value in zip(raw_total, official_total, y_mass):
        mass_axis.plot(
            [raw_value, official_value], [y_value, y_value], color="#A7B4BE",
            lw=1.2, zorder=1,
        )
    mass_axis.scatter(
        raw_total, y_mass, s=34, marker="s", facecolor="white", edgecolor="#6B7280",
        linewidth=1.0, zorder=3, label="Uncalibrated stock model",
    )
    mass_axis.scatter(
        calibrated_total, y_mass, s=28, marker="D", color="#E07A5F",
        edgecolor="white", linewidth=0.6, zorder=4, label="Calibrated model",
    )
    mass_axis.set_xscale("log")
    mass_axis.set_yticks(y_mass, mass["label"])
    mass_axis.set_xlabel("Annual electricity (TWh, log scale)")
    mass_axis.grid(axis="x", color="#d9dbdc", linewidth=0.5, alpha=0.65)
    legend_handles = [
        Line2D([], [], linestyle="", marker="o", color="#305F72", markersize=7,
               label="Official target"),
        Line2D([], [], linestyle="", marker="s", markerfacecolor="white",
               markeredgecolor="#6B7280", markersize=5, label="Uncalibrated model"),
        Line2D([], [], linestyle="", marker="D", color="#E07A5F", markersize=5,
               label="Calibrated model"),
    ]
    for share in [0.2, 0.4, 0.6]:
        legend_handles.append(
            Line2D([], [], linestyle="", marker="o", markerfacecolor="none",
                   markeredgecolor="#305F72", markersize=np.sqrt(75 + 440 * share) / 1.5,
                   label=f"{share:.0%} cooling share")
        )
    mass_axis.legend(handles=legend_handles, frameon=False, loc="lower right", ncol=2)
    panel_title(mass_axis, "f", "EMSD 2023 annual calibration constraint")
    save_figure(fig, destination)
    plt.close(fig)


def render_figure_5(destination: Path) -> None:
    """Class responses, cross-stage divergence, event holdout and selection consequences."""
    summary_path = RESULT_DIR / "type_event_stage_summary.csv"
    class_summary_path = RESULT_DIR / "class_event_stage_summary.csv"
    fig, grid_axes = plt.subplots(3, 2, figsize=(13.8, 14.2), constrained_layout=False)
    fig.subplots_adjust(left=0.105, right=0.965, bottom=0.06, top=0.965, wspace=0.29, hspace=0.40)
    axes = np.empty((2, 3), dtype=object)
    axes[0, 0], axes[0, 1] = grid_axes[0, 0], grid_axes[0, 1]
    axes[0, 2], axes[1, 0] = grid_axes[1, 0], grid_axes[1, 1]
    axes[1, 1], axes[1, 2] = grid_axes[2, 0], grid_axes[2, 1]
    class_colors = {"Residential": "#3b82c4", "Commercial": "#d95f4a", "Mixed-use": "#e68a00"}
    class_order = ["Residential", "Mixed-use", "Commercial"]

    event_scores = pd.read_parquet(
        EVENT_RESPONSE_RESULTS,
        columns=[
            "BUILDINGSTRUCTUREID", "main_class", "event_id",
            "heatwave_relative_energy_increase", "score_capacity",
            "safety_resistance_hours_to_32c", "recovery_time_hours",
        ],
    )
    event_order = sorted(event_scores["event_id"].unique())
    event_years = [str(value)[3:7] for value in event_order]
    x_event = np.arange(len(event_order))
    offsets = {"Residential": -0.23, "Mixed-use": 0.0, "Commercial": 0.23}
    for main_class in class_order:
        for event_position, event_id in enumerate(event_order):
            values = 100 * event_scores.loc[
                event_scores["main_class"].eq(main_class)
                & event_scores["event_id"].eq(event_id),
                "heatwave_relative_energy_increase",
            ].dropna().to_numpy(dtype=float)
            if len(values) > 12000:
                ordered = np.sort(values)
                values = ordered[np.linspace(0, len(ordered) - 1, 12000).astype(int)]
            position = event_position + offsets[main_class]
            violin = axes[0, 0].violinplot(
                [values], positions=[position], widths=0.20,
                showmeans=False, showmedians=False, showextrema=False,
            )
            for body in violin["bodies"]:
                body.set_facecolor(class_colors[main_class])
                body.set_edgecolor("none")
                body.set_alpha(0.34)
            q05, q25, median, q75, q95 = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
            axes[0, 0].plot([position, position], [q05, q95], color=class_colors[main_class], lw=1.0)
            axes[0, 0].plot([position, position], [q25, q75], color=class_colors[main_class], lw=3.5,
                            solid_capstyle="round")
            axes[0, 0].scatter(position, median, s=24, color="white",
                               edgecolor=class_colors[main_class], linewidth=1.1, zorder=4)
    axes[0, 0].set_xticks(x_event, event_years)
    axes[0, 0].set_ylabel("Building-level event electricity increase (%)")
    axes[0, 0].legend(
        handles=[Patch(facecolor=class_colors[name], edgecolor="none", label=name) for name in class_order],
        frameon=False, loc="upper right",
    )
    axes[0, 0].grid(axis="y", color="#d9dbdc", linewidth=0.5, alpha=0.65)
    panel_title(axes[0, 0], "a", "Building-level electricity sensitivity")

    peaks = pd.read_csv(RESULT_DIR / "city_coincident_peak_summary.csv")
    peaks = peaks.loc[peaks["supply_stage"].eq("normal")].sort_values("event_id")
    required_uncertainty = {
        "heatwave_incremental_energy_kwh_p10",
        "heatwave_incremental_energy_kwh_median",
        "heatwave_incremental_energy_kwh_p90",
        "coincident_peak_change_pct_p10",
        "coincident_peak_change_pct_median",
        "coincident_peak_change_pct_p90",
    }
    if missing := sorted(required_uncertainty - set(peaks.columns)):
        raise ValueError(f"Figure 5b lacks paired trajectory uncertainty fields: {missing}")
    event_energy = peaks["heatwave_incremental_energy_kwh_median"].to_numpy(dtype=float) / 1e6
    event_energy_low = peaks["heatwave_incremental_energy_kwh_p10"].to_numpy(dtype=float) / 1e6
    event_energy_high = peaks["heatwave_incremental_energy_kwh_p90"].to_numpy(dtype=float) / 1e6
    peak_change = peaks["coincident_peak_change_pct_median"].to_numpy(dtype=float)
    peak_change_low = peaks["coincident_peak_change_pct_p10"].to_numpy(dtype=float)
    peak_change_high = peaks["coincident_peak_change_pct_p90"].to_numpy(dtype=float)
    years = peaks["event_id"].str.extract(r"HW_(\d{4})", expand=False).tolist()
    x = np.arange(len(peaks))
    bars = axes[0, 1].bar(
        x, event_energy, width=0.55, color="#315D7D", alpha=0.82,
        yerr=np.vstack([event_energy - event_energy_low, event_energy_high - event_energy]),
        error_kw={"ecolor": "#1F4058", "elinewidth": 1.1, "capsize": 3.0},
        label="Event electricity",
    )
    axes[0, 1].set_xticks(x, years)
    axes[0, 1].set_ylabel("Heatwave-attributable electricity (GWh)", color="#315D7D", labelpad=7)
    axes[0, 1].tick_params(axis="y", colors="#315D7D")
    peak_axis = axes[0, 1].twinx()
    peak_axis.vlines(x, 0, peak_change, color="#D1495B", lw=1.5, alpha=0.72)
    peak_axis.errorbar(
        x, peak_change,
        yerr=np.vstack([peak_change - peak_change_low, peak_change_high - peak_change]),
        fmt="D", color="#D1495B", markersize=6.5, markeredgecolor="white",
        markeredgewidth=0.65, elinewidth=1.0, capsize=3.0, zorder=4,
        label="Coincident peak",
    )
    peak_axis.set_ylabel("Change in territory coincident peak (%)", color="#D1495B", labelpad=7)
    peak_axis.tick_params(axis="y", colors="#D1495B")
    axes[0, 1].set_ylim(0, float(event_energy.max()) * 1.42)
    peak_axis.set_ylim(0, float(peak_change_high.max()) * 1.18)
    for rectangle, value in zip(bars, event_energy):
        axes[0, 1].text(
            rectangle.get_x() + rectangle.get_width() / 2, value + 0.55,
            f"{value:.1f}", ha="center", va="bottom", color="#315D7D", zorder=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.7},
        )
    for position, value in zip(x, peak_change):
        # Place peak-change labels outside the bars and lollipop stems.
        x_offset = 26 if position == x[0] else 0
        y_offset = 7 if value >= 0 else -7
        peak_axis.annotate(
            f"{value:+.2f}%", (position, value), xytext=(x_offset, y_offset),
            textcoords="offset points", ha="left" if x_offset else "center",
            va="top" if y_offset < 0 else "bottom", color="#D1495B", zorder=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.7},
        )
    axes[0, 1].legend(
        handles=[
            Patch(facecolor="#315D7D", edgecolor="none", label="Event electricity (median; P10–P90)"),
            Line2D([], [], color="#D1495B", marker="D", label="Coincident peak (median; P10–P90)"),
        ],
        frameon=False, loc="upper left", ncol=2,
        columnspacing=0.9, handletextpad=0.4,
    )
    axes[0, 1].grid(axis="y", color="#d9dbdc", linewidth=0.5, alpha=0.65)
    panel_title(axes[0, 1], "b", "Event energy and territory-wide coincident peak")

    loeo = pd.read_csv(RESULT_DIR / "beri_leave_one_event_out_summary.csv").sort_values("held_out_event")
    loeo_years = loeo["held_out_event"].str.extract(r"HW_(\d{4})", expand=False).tolist()
    loeo_values = np.vstack([
        loeo["spearman_training_beri_vs_held_out_event_score"].to_numpy(dtype=float),
        loeo["tie_aware_expected_lowest_decile_overlap_with_held_out_event_response"].to_numpy(dtype=float),
    ])
    loeo_labels = ["Rank agreement\n(Spearman ρ)", "Low-10% cohort\noverlap"]
    loeo_norm = Normalize(0.0, 1.0)
    for row in range(loeo_values.shape[0]):
        for column in range(loeo_values.shape[1]):
            value = float(loeo_values[row, column])
            axes[0, 2].scatter(
                column, row, s=110 + 640 * value,
                c=[value], cmap="viridis", norm=loeo_norm,
                edgecolor="white", linewidth=0.9, zorder=3,
            )
            label = f"{value:.2f}" if row == 0 else f"{100 * value:.1f}%"
            axes[0, 2].text(
                column, row, label, ha="center", va="center",
                color="white" if value > 0.43 else "#17212B",
                fontweight="bold", zorder=4,
            )
    axes[0, 2].set_xticks(np.arange(len(loeo_years)), loeo_years)
    axes[0, 2].set_yticks(np.arange(len(loeo_labels)), loeo_labels)
    axes[0, 2].set_xlim(-0.55, len(loeo_years) - 0.45)
    axes[0, 2].set_ylim(len(loeo_labels) - 0.40, -0.60)
    axes[0, 2].tick_params(length=0)
    axes[0, 2].text(
        0.02, 0.04, "circle area and colour encode the held-out value",
        transform=axes[0, 2].transAxes, ha="left", va="bottom", color="#555555",
    )
    for spine in axes[0, 2].spines.values():
        spine.set_visible(False)
    panel_title(axes[0, 2], "c", "Complete-event holdout of two-event BERI")

    component_fields = ["score_normal", "score_capacity", "score_outage", "score_recovery"]
    component_labels = ["Normal\nsupply", "Cooling\nservice", "Outage\ntolerance", "Recovery"]
    cross_stage = pd.read_csv(RESULT_DIR / "cross_stage_ranking_audit.csv")
    per_building_physical = event_scores.assign(
        unmet_cooling_fraction=(1.0 - event_scores["score_capacity"]).clip(0.0, 1.0),
    ).groupby("BUILDINGSTRUCTUREID", observed=True)[
        [
            "heatwave_relative_energy_increase", "unmet_cooling_fraction",
            "safety_resistance_hours_to_32c", "recovery_time_hours",
        ]
    ].median()
    physical_diagonal = [
        f"median\nΔE {100 * per_building_physical['heatwave_relative_energy_increase'].median():.2f}%",
        f"median\nunmet {100 * per_building_physical['unmet_cooling_fraction'].median():.2f}%",
        f"median\nsafe {per_building_physical['safety_resistance_hours_to_32c'].median():.2f} h",
        f"median\nrecover {per_building_physical['recovery_time_hours'].median():.1f} h",
    ]
    field_index = {field: index for index, field in enumerate(component_fields)}
    pair_values = {}
    for record in cross_stage.itertuples(index=False):
        left = field_index[record.left_component]
        right = field_index[record.right_component]
        pair_values[(max(left, right), min(left, right))] = float(record.spearman_rank_correlation)
        pair_values[(min(left, right), max(left, right))] = float(record.tie_aware_expected_low_fraction_overlap)
    for row in range(4):
        for column in range(4):
            if row == column:
                facecolor = "#F2F0EC"
                label = physical_diagonal[row]
                text_color = "#1F2933"
            elif row > column:
                value = pair_values[(row, column)]
                facecolor = mpl.colormaps["Blues"](0.18 + 0.68 * value)
                label = f"ρ = {value:.2f}"
                text_color = "white" if value > 0.58 else "#17324D"
            else:
                value = pair_values[(row, column)]
                facecolor = mpl.colormaps["Oranges"](0.12 + 0.76 * value)
                label = f"{100 * value:.1f}%"
                text_color = "white" if value > 0.50 else "#65310C"
            axes[1, 0].add_patch(
                mpl.patches.Rectangle(
                    (column - 0.48, row - 0.48), 0.96, 0.96,
                    facecolor=facecolor, edgecolor="white", linewidth=1.1,
                )
            )
            axes[1, 0].text(
                column, row, label, ha="center", va="center",
                color=text_color, fontsize=9.2 if row == column else 10.2,
                fontweight="bold" if row != column else "normal",
            )
    axes[1, 0].set_xticks(np.arange(4), component_labels)
    axes[1, 0].set_yticks(np.arange(4), component_labels)
    axes[1, 0].set_xlim(-0.52, 3.52)
    axes[1, 0].set_ylim(3.52, -0.82)
    axes[1, 0].tick_params(length=0)
    for spine in axes[1, 0].spines.values():
        spine.set_visible(False)
    panel_title(
        axes[1, 0], "d",
        "Stage agreement (rank ρ below; tie-aware low-10% overlap above)",
    )

    thermal = outcome_table(class_summary_path, "outage_attributable_degree_hours_above_26c")
    thermal = thermal.loc[thermal["supply_stage"].eq("outage_peak_24h")].copy()
    thermal["event_year"] = thermal["event_id"].str.extract(r"HW_(\d{4})", expand=False)
    thermal_matrix = thermal.pivot(index="main_class", columns="event_year", values="median").reindex(class_order)
    thermal_values = thermal_matrix.to_numpy(dtype=float)
    thermal_norm = Normalize(np.nanmin(thermal_values), np.nanmax(thermal_values))
    for row, main_class in enumerate(class_order):
        for column, year in enumerate(event_years):
            value = float(thermal_matrix.loc[main_class, year])
            axes[1, 1].scatter(column, row, s=60 + 2.1 * value, c=[value], cmap="magma_r",
                               norm=thermal_norm, edgecolor="white", linewidth=0.65)
            axes[1, 1].annotate(
                f"{value:.1f}", (column, row), xytext=(0, 9), textcoords="offset points",
                ha="center", va="bottom", color="#17212B",
            )
    axes[1, 1].set_xticks(np.arange(3), event_years)
    axes[1, 1].set_yticks(np.arange(3), class_order)
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlim(-0.55, 2.55)
    axes[1, 1].set_ylim(2.60, -0.95)
    axes[1, 1].text(0.98, 0.03, "values: °C h", transform=axes[1, 1].transAxes,
                    ha="right", va="bottom", color="#555555")
    axes[1, 1].grid(color="#e5e7eb", linewidth=0.5)
    panel_title(axes[1, 1], "e", "24-h outage-attributable heat by event")

    capture = pd.read_csv(RESULT_DIR / "beri_leave_one_event_out_capture.csv")
    capture = capture.loc[capture["held_out_event"].eq("HW_20230724_20230728")].copy()
    selection_order = [
        "lowest_two_event_BERI",
        "lowest_two_event_normal_score",
        "lowest_two_event_capacity_score",
        "lowest_two_event_outage_score",
        "lowest_two_event_recovery_score",
        "lowest_training_detailed_use_floor_exposure_median_BERI",
        "oldest_construction_year",
    ]
    selection_labels = [
        "Two-event BERI", "Normal-supply score", "Cooling-service score",
        "Outage-tolerance score", "Recovery score", "Detailed use + floor", "Oldest buildings",
    ]
    burden_order = [
        "normal_incremental_electricity_kwh",
        "capacity_unmet_cooling_kwh_th",
        "outage_degree_hours_above_32c",
        "outage_population_degree_hours_above_32c",
        "recovery_time_hours",
    ]
    burden_labels = [
        "Electricity\nincrement", "Unmet\ncooling", "Outage\n>32 °C", "Resident-weighted\noutage >32 °C", "Recovery\ntime",
    ]
    capture_matrix = capture.pivot(
        index="selection_rule", columns="captured_metric",
        values="tie_adjusted_expected_captured_share",
    ).reindex(index=selection_order, columns=burden_order)
    if capture_matrix.isna().any().any():
        raise ValueError("Figure 5f selection-rule capture matrix is incomplete")
    capture_values = capture_matrix.to_numpy(dtype=float)
    capture_norm = Normalize(0.0, 0.55)
    for row in range(capture_values.shape[0]):
        for column in range(capture_values.shape[1]):
            value = float(capture_values[row, column])
            axes[1, 2].add_patch(
                mpl.patches.Rectangle(
                    (column - 0.49, row - 0.49), 0.98, 0.98,
                    facecolor=mpl.colormaps["YlGnBu"](capture_norm(value)),
                    edgecolor="white", linewidth=0.8,
                )
            )
            axes[1, 2].text(
                column, row, f"{100 * value:.1f}%", ha="center", va="center",
                color="white" if value >= 0.29 else "#17324D",
                fontsize=8.4, fontweight="bold" if value >= 0.10 else "normal",
            )
    axes[1, 2].set_xticks(np.arange(len(burden_order)), burden_labels)
    axes[1, 2].set_yticks(np.arange(len(selection_order)), selection_labels)
    axes[1, 2].set_xlim(-0.52, len(burden_order) - 0.48)
    axes[1, 2].set_ylim(len(selection_order) - 0.48, -0.72)
    axes[1, 2].tick_params(length=0)
    axes[1, 2].text(
        0.99, -0.18, "fixed 10% cohorts; random expectation = 10%",
        transform=axes[1, 2].transAxes, ha="right", va="top", color="#555555",
        clip_on=False,
    )
    for spine in axes[1, 2].spines.values():
        spine.set_visible(False)
    panel_title(axes[1, 2], "f", "Held-out 2023 burden captured by alternative cohorts")
    save_figure(fig, destination)
    plt.close(fig)


def render_figure_6(destination: Path) -> None:
    """Paired event attribution, supply stages and multimodel future response."""
    hourly = pd.read_parquet(RESPONSE_DIR / "citywide_coincident_hourly_response.parquet")
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.2), constrained_layout=False)
    fig.subplots_adjust(left=0.095, right=0.965, bottom=0.085, top=0.955, wspace=0.25, hspace=0.38)
    panel_a_box = axes[0, 0].get_position()
    axes[0, 0].set_position(
        [
            panel_a_box.x0,
            panel_a_box.y0 + 0.43 * panel_a_box.height,
            panel_a_box.width,
            0.57 * panel_a_box.height,
        ]
    )
    peak_terms_axis = fig.add_axes(
        [
            panel_a_box.x0,
            panel_a_box.y0,
            panel_a_box.width,
            0.30 * panel_a_box.height,
        ]
    )
    event = "HW_20220722_20220729"
    part = hourly.loc[hourly["event_id"].eq(event) & hourly["supply_stage"].eq("normal")].copy()
    part["timestamp_hkt"] = pd.to_datetime(part["timestamp_hkt"])
    paired_load = part.pivot(
        index="timestamp_hkt", columns="climate_case", values="city_average_load_kw"
    ).sort_index() / 1e6
    load_index = pd.DatetimeIndex(paired_load.index)
    event_start = pd.Timestamp("2022-07-22 00:00:00")
    event_end_exclusive = pd.Timestamp("2022-07-30 00:00:00")
    if load_index.tz is not None:
        event_start = event_start.tz_localize(load_index.tz)
        event_end_exclusive = event_end_exclusive.tz_localize(load_index.tz)
    paired_load = paired_load.loc[
        (load_index >= event_start) & (load_index < event_end_exclusive)
    ]
    axes[0, 0].plot(pd.to_datetime(paired_load.index), paired_load["counterfactual"],
                    lw=2.0, color="#607080", label="Matched weather baseline")
    axes[0, 0].plot(pd.to_datetime(paired_load.index), paired_load["heatwave"],
                    lw=2.1, color="#C44741", label="Observed-weather heatwave simulation")
    if {"heatwave", "counterfactual"}.issubset(paired_load.columns):
        axes[0, 0].fill_between(
            pd.to_datetime(paired_load.index), paired_load["counterfactual"], paired_load["heatwave"],
            where=paired_load["heatwave"].ge(paired_load["counterfactual"]),
            color="#F2A65A", alpha=0.32, interpolate=True, label="Heatwave-attributable load",
        )
    axes[0, 0].set_ylabel("Hong Kong building load (GW)")
    axes[0, 0].set_xlabel("22–29 July 2022 (HKT)")
    axes[0, 0].xaxis.set_major_locator(mdates.DayLocator(interval=2))
    axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    load_min = float(np.nanmin(paired_load.to_numpy(dtype=float)))
    load_max = float(np.nanmax(paired_load.to_numpy(dtype=float)))
    load_span = max(load_max - load_min, 0.5)
    axes[0, 0].set_ylim(load_min - 0.04 * load_span, load_max + 0.24 * load_span)
    axes[0, 0].legend(frameon=False, loc="upper left")
    axes[0, 0].tick_params(axis="x", rotation=0)
    axes[0, 0].text(
        0.985, 0.965, "matched stock, calendar and behaviour;\nweather differs",
        transform=axes[0, 0].transAxes, ha="right", va="top", color="#555555",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 2.0},
    )
    timing = pd.read_csv(RESULT_DIR / "event_timing_decomposition.csv")
    timing["year"] = timing["event_id"].str.extract(r"HW_(\d{4})", expand=False)
    timing = timing.sort_values("event_id")
    y_timing = np.arange(len(timing))
    weather_mw = timing["weather_increment_at_heatwave_peak_kw"].to_numpy(dtype=float) / 1000.0
    clock_mw = timing["counterfactual_timing_term_kw"].to_numpy(dtype=float) / 1000.0
    net_mw = timing["coincident_peak_increment_kw"].to_numpy(dtype=float) / 1000.0
    peak_terms_axis.barh(
        y_timing - 0.13, weather_mw, height=0.22, color="#D95F4A", alpha=0.86,
        label="Weather at heatwave peak",
    )
    peak_terms_axis.barh(
        y_timing + 0.13, clock_mw, height=0.22, color="#8B98A5", alpha=0.86,
        label="Counterfactual timing",
    )
    peak_terms_axis.scatter(
        net_mw, y_timing, marker="D", s=30, color="#213B53",
        edgecolor="white", linewidth=0.55, zorder=4, label="Difference between maxima",
    )
    peak_terms_axis.axvline(0, color="#4B5563", lw=0.65)
    peak_terms_axis.set_yticks(y_timing, timing["year"])
    peak_terms_axis.set_ylim(-0.55, len(timing) + 0.25)
    peak_terms_axis.set_xlabel("Exact peak-change terms (MW)", fontsize=8.7, labelpad=1)
    peak_terms_axis.tick_params(axis="both", labelsize=8.2, length=2.0, pad=1.5)
    peak_terms_axis.legend(
        frameon=False, loc="upper center", ncol=3,
        fontsize=7.4, handlelength=1.2, handletextpad=0.35,
        columnspacing=0.65, borderaxespad=0.15,
    )
    for spine in ["top", "right"]:
        peak_terms_axis.spines[spine].set_visible(False)
    axes[0, 0].grid(axis="y", color="#d9dbdc", linewidth=0.5, alpha=0.65)
    panel_title(axes[0, 0], "a", "Hourly heatwave attribution and peak decomposition")

    raw_profiles = pd.read_parquet(
        RESPONSE_DIR / "energyplus_multizone_hourly_results.parquet",
        columns=[
            "event_id", "prototype_id", "base_case_id", "climate_case",
            "supply_stage", "timestamp_hkt", "exposure_temperature_c",
            "total_electricity_kwh_m2",
        ],
        filters=[
            ("event_id", "==", event),
            ("climate_case", "==", "heatwave"),
        ],
    )
    raw_profiles = raw_profiles.loc[
        raw_profiles["supply_stage"].isin(["capacity_100pct", "outage_peak_24h"])
    ].copy()
    prototype_styles = {
        "res_old_lowrise_tenement": ("Old low-rise residential", "#2C7BB6"),
        "com_office": ("Office", "#D95F02"),
        "com_human_health": ("Healthcare", "#1B9E77"),
        "com_data_centre": ("Data centre", "#7B3294"),
    }
    for prototype_id, (label, color) in prototype_styles.items():
        prototype = raw_profiles.loc[raw_profiles["prototype_id"].eq(prototype_id)].copy()
        aligned = []
        aligned_rated_capacity = []
        outage_profiles = prototype.loc[
            prototype["supply_stage"].eq("outage_peak_24h")
        ]
        rated_capacity_profiles = prototype.loc[
            prototype["supply_stage"].eq("capacity_100pct")
        ]
        for base_case_id, case in outage_profiles.groupby("base_case_id", observed=True):
            case = case.sort_values("timestamp_hkt").copy()
            case["timestamp_hkt"] = pd.to_datetime(case["timestamp_hkt"])
            zero_grid = pd.to_numeric(
                case["total_electricity_kwh_m2"], errors="coerce"
            ).abs().le(1e-10)
            if int(zero_grid.sum()) != 24:
                continue
            outage_start = case.loc[zero_grid, "timestamp_hkt"].iloc[0]
            case["elapsed_h"] = (
                case["timestamp_hkt"] - outage_start
            ).dt.total_seconds() / 3600.0
            aligned.append(case.loc[case["elapsed_h"].between(-12, 48), [
                "elapsed_h", "exposure_temperature_c",
            ]])
            matched_rated = rated_capacity_profiles.loc[
                rated_capacity_profiles["base_case_id"].eq(base_case_id)
            ].sort_values("timestamp_hkt").copy()
            matched_rated["timestamp_hkt"] = pd.to_datetime(
                matched_rated["timestamp_hkt"]
            )
            matched_rated["elapsed_h"] = (
                matched_rated["timestamp_hkt"] - outage_start
            ).dt.total_seconds() / 3600.0
            aligned_rated_capacity.append(
                matched_rated.loc[
                    matched_rated["elapsed_h"].between(-12, 48),
                    ["elapsed_h", "exposure_temperature_c"],
                ]
            )
        if not aligned:
            raise ValueError(f"No complete 24-h outage trajectory for {prototype_id}")
        aligned = pd.concat(aligned, ignore_index=True)
        quantiles = aligned.groupby("elapsed_h", observed=True)[
            "exposure_temperature_c"
        ].quantile([0.10, 0.50, 0.90]).unstack()
        axes[0, 1].fill_between(
            quantiles.index, quantiles[0.10], quantiles[0.90],
            color=color, alpha=0.10, linewidth=0,
        )
        axes[0, 1].plot(
            quantiles.index, quantiles[0.50], color=color, lw=1.8, label=label,
        )
        rated_quantiles = pd.concat(aligned_rated_capacity, ignore_index=True).groupby(
            "elapsed_h", observed=True
        )["exposure_temperature_c"].quantile([0.10, 0.50, 0.90]).unstack()
        axes[0, 1].plot(
            rated_quantiles.index, rated_quantiles[0.50],
            color=color, lw=1.0, ls="--", alpha=0.72,
        )
    axes[0, 1].axvspan(0, 24, color="#4B5563", alpha=0.08, zorder=0)
    axes[0, 1].axvline(0, color="#4B5563", lw=0.9, ls="--")
    axes[0, 1].axvline(24, color="#4B5563", lw=0.9, ls="--")
    axes[0, 1].axhline(32, color="#B2182B", lw=1.0, ls=":")
    axes[0, 1].set_ylim(23.5, 45.5)
    axes[0, 1].text(-10.8, 43.0, "power lost", ha="left", va="top", color="#4B5563")
    axes[0, 1].text(24.7, 43.0, "restored", ha="left", va="top", color="#4B5563")
    axes[0, 1].set_xlim(-12, 48)
    axes[0, 1].set_xlabel("Hours from outage onset")
    axes[0, 1].set_ylabel("Hottest occupied-zone temperature (°C)")
    axes[0, 1].legend(
        handles=[
            *[
                Line2D([], [], color=color, lw=1.8, label=label)
                for label, color in prototype_styles.values()
            ],
            Line2D([], [], color="#4B5563", lw=1.0, ls="--", label="100% rated-capacity reference"),
        ],
        frameon=False, loc="upper center", ncol=3,
        handlelength=1.5, columnspacing=0.7, handletextpad=0.35,
    )
    axes[0, 1].grid(color="#d9dbdc", linewidth=0.45, alpha=0.55)
    panel_title(axes[0, 1], "b", "Indoor temperature through outage and restoration")

    future_path = RESPONSE_DIR / "future_energyplus_results.csv"
    if future_path.exists():
        future = pd.read_csv(future_path)
        future["future_period"] = pd.Categorical(
            future["future_period"],
            ["mid_century_2041_2060", "late_century_2081_2100"],
            ordered=True,
        )
        normal = future.loc[future["supply_stage"].eq("normal")].copy()
        ensemble = normal.groupby(
            ["source_id", "experiment_id", "future_period"], observed=True
        )["future_incremental_energy_kwh_m2"].median().reset_index()
        group_order = [
            ("ssp245", "mid_century_2041_2060"), ("ssp245", "late_century_2081_2100"),
            ("ssp585", "mid_century_2041_2060"), ("ssp585", "late_century_2081_2100"),
        ]
        xlabels = ["SSP2–4.5\n2041–2060", "SSP2–4.5\n2081–2100",
                   "SSP5–8.5\n2041–2060", "SSP5–8.5\n2081–2100"]
        scenario_map = {"ssp245": "#2878B5", "ssp585": "#D94841"}
        model_names = sorted(ensemble["source_id"].unique())
        offsets = dict(zip(model_names, np.linspace(-0.10, 0.10, len(model_names))))
        for position, (scenario, period) in enumerate(group_order):
            group = ensemble.loc[
                ensemble["experiment_id"].eq(scenario) & ensemble["future_period"].eq(period)
            ]
            values = group["future_incremental_energy_kwh_m2"].to_numpy(dtype=float)
            for record in group.itertuples(index=False):
                axes[1, 0].scatter(
                    position + offsets[str(record.source_id)], record.future_incremental_energy_kwh_m2,
                    s=40, color=scenario_map[scenario], alpha=0.62,
                    edgecolor="white", linewidth=0.5, zorder=3,
                )
            p10, median, p90 = np.quantile(values, [0.1, 0.5, 0.9])
            axes[1, 0].plot([position, position], [p10, p90], color=scenario_map[scenario], lw=3.0)
            axes[1, 0].scatter(position, median, marker="D", s=62, facecolor="white",
                               edgecolor=scenario_map[scenario], linewidth=1.5, zorder=4)
        axes[1, 0].set_xticks(np.arange(4), xlabels)
        axes[1, 0].set_ylabel("Event electricity increment (kWh m⁻²)")
        axes[1, 0].text(0.02, 0.97, "dots: model medians   diamond: ensemble median   line: P10–P90",
                        transform=axes[1, 0].transAxes, ha="left", va="top", color="#555555")
        axes[1, 0].grid(axis="y", color="#d9dbdc", linewidth=0.5, alpha=0.65)

        outage = future.loc[future["supply_stage"].eq("outage_peak_24h")].groupby(
            ["source_id", "experiment_id", "future_period"], observed=True
        ).agg(
            heat_stress=("outage_attributable_degree_hours_above_26c", "median"),
            recovery=("recovery_to_26c_hours", "median"),
        ).reset_index()
        joint = ensemble.merge(
            outage, on=["source_id", "experiment_id", "future_period"], validate="one_to_one"
        )
        period_markers = {"mid_century_2041_2060": "o", "late_century_2081_2100": "D"}
        period_order = ["mid_century_2041_2060", "late_century_2081_2100"]
        for (scenario, source), group in joint.groupby(["experiment_id", "source_id"], observed=True):
            group = group.assign(
                _period_order=pd.Categorical(group["future_period"], period_order, ordered=True)
            ).sort_values("_period_order")
            if len(group) == 2:
                axes[1, 1].annotate(
                    "", xy=(group.iloc[1]["future_incremental_energy_kwh_m2"], group.iloc[1]["heat_stress"]),
                    xytext=(group.iloc[0]["future_incremental_energy_kwh_m2"], group.iloc[0]["heat_stress"]),
                    arrowprops={"arrowstyle": "->", "color": scenario_map[str(scenario)],
                                "lw": 0.9, "alpha": 0.45},
                )
        for record in joint.itertuples(index=False):
            comparator = (
                str(record.experiment_id) == "ssp585"
                and str(record.future_period) == "mid_century_2041_2060"
                and str(record.source_id) == "BCC-CSM2-MR"
            )
            axes[1, 1].scatter(
                record.future_incremental_energy_kwh_m2, record.heat_stress,
                s=28 + 11 * float(record.recovery), marker=period_markers[str(record.future_period)],
                facecolor=("white" if comparator else scenario_map[str(record.experiment_id)]),
                alpha=0.86, edgecolor=scenario_map[str(record.experiment_id)],
                linewidth=(1.2 if comparator else 0.65), zorder=3,
            )
        axes[1, 1].set_xlabel("Event electricity increment (kWh m⁻²)")
        axes[1, 1].set_ylabel("24-h outage-attributable heat (°C h)")
        recovery_legend_hours = [1, 2]
        joint_handles = [
                Line2D([], [], linestyle="", marker="o", color="#2878B5", label="SSP2–4.5"),
                Line2D([], [], linestyle="", marker="o", color="#D94841", label="SSP5–8.5"),
                Line2D([], [], linestyle="", marker="o", color="#555555", label="2041–2060"),
                Line2D([], [], linestyle="", marker="D", color="#555555", label="2081–2100"),
        ]
        joint_handles.extend(
            Line2D(
                [], [], linestyle="", marker="o", markerfacecolor="none",
                markeredgecolor="#555555", markersize=np.sqrt(28 + 11 * hours),
                label=f"{hours} h recovery",
            )
            for hours in recovery_legend_hours
        )
        axes[1, 1].legend(
            handles=joint_handles, frameon=False, loc="lower right", ncol=2,
        )
        axes[1, 1].text(0.02, 0.97, "filled: representative nodes   hollow: BCC same-model comparator",
                        transform=axes[1, 1].transAxes, ha="left", va="top", color="#555555")
        axes[1, 1].grid(color="#d9dbdc", linewidth=0.55, alpha=0.55)
    else:
        axes[1, 0].text(0.5, 0.5, "Future EnergyPlus response pending", ha="center", va="center", transform=axes[1, 0].transAxes)
        axes[1, 1].text(0.5, 0.5, "Future outage response pending", ha="center", va="center", transform=axes[1, 1].transAxes)
    panel_title(axes[1, 0], "c", "Five-model future electricity response")
    panel_title(axes[1, 1], "d", "Representative future nodes and same-model change")
    save_figure(fig, destination)
    plt.close(fig)


def region_subset(
    frame: gpd.GeoDataFrame,
    bounds_2326: tuple[float, float, float, float],
) -> gpd.GeoDataFrame:
    xmin, ymin, xmax, ymax = bounds_2326
    return frame.cx[xmin:xmax, ymin:ymax].copy()


def homantin_subset(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return region_subset(frame, HOMANTIN_BOUNDS_2326)


def render_figure_4(
    model_inputs: gpd.GeoDataFrame,
    results: pd.DataFrame,
    destination: Path,
    event: str = "HW_20220722_20220729",
) -> None:
    """Peak-hour electricity, thermal safety and BERI in three dense urban districts."""
    timestamps, profiles, _ = hourly_prototype_profiles(event)
    timing = pd.read_csv(OUTAGE_TIMING_AUDIT)
    timing = timing.loc[
        timing["event_id"].astype(str).eq(event)
        & timing["supply_stage"].astype(str).eq("outage_common_24h")
    ].copy()
    if timing.empty:
        raise ValueError(f"Figure 4 requires a synchronized outage timing audit for {event}")
    outage_starts = pd.to_datetime(timing["outage_start_hkt"], utc=True).dt.tz_convert(
        "Asia/Hong_Kong"
    ).drop_duplicates()
    outage_ends = pd.to_datetime(
        timing["outage_end_exclusive_hkt"], utc=True
    ).dt.tz_convert("Asia/Hong_Kong").drop_duplicates()
    if len(outage_starts) != 1 or len(outage_ends) != 1:
        raise ValueError(f"Figure 4 synchronized outage clock is not unique for {event}")
    outage_start = pd.Timestamp(outage_starts.iloc[0])
    outage_end = pd.Timestamp(outage_ends.iloc[0])
    thermal_timestamp = outage_start + pd.Timedelta(hours=12)
    if thermal_timestamp not in pd.DatetimeIndex(profiles["timestamp_hkt"].unique()):
        raise ValueError(f"Figure 4 outage +12 h is outside the hourly profile for {event}")
    candidates = model_inputs.loc[model_inputs["pre_weather_simulation_candidate"]].copy()
    beri = building_reference_beri()
    payloads: list[dict] = []
    energy_pool: list[pd.Series] = []
    thermal_pool: list[pd.Series] = []
    for region_name, bounds_2326 in REGION_BOUNDS_2326.items():
        local = animation_display_type(region_subset(candidates, bounds_2326))
        local["building_energy_resilience_index"] = local["BUILDINGSTRUCTUREID"].map(beri)
        if local.empty or local["building_energy_resilience_index"].isna().any():
            raise ValueError(f"{region_name} Figure 4 view requires complete geometry and BERI coverage")
        area = (
            pd.to_numeric(local["residential_gfa_m2"], errors="coerce").fillna(0)
            + pd.to_numeric(local["commercial_gfa_m2"], errors="coerce").fillna(0)
        )
        timeline_rows: list[dict] = []
        region_energy: list[pd.Series] = []
        for timestamp in timestamps:
            load, thermal = map_hourly_state(local, profiles, timestamp)
            region_energy.append(load.dropna())
            timeline_rows.append(
                {
                    "timestamp_hkt": timestamp,
                    "scope_load_w_m2": float((load.fillna(0) * area).sum() / area.sum()),
                }
            )
        timeline = pd.DataFrame(timeline_rows)
        peak_timestamp = pd.Timestamp(
            timeline.loc[timeline["scope_load_w_m2"].idxmax(), "timestamp_hkt"]
        )
        load, _ = map_hourly_state(local, profiles, peak_timestamp)
        _, thermal = map_hourly_state(local, profiles, thermal_timestamp)
        local["hourly_load_w_m2"] = load.to_numpy()
        local["thermal_safety_margin_c"] = thermal.to_numpy()
        energy_pool.extend(region_energy)
        thermal_pool.append(thermal.dropna())
        payloads.append(
            {
                "name": region_name,
                "frame": local.to_crs(3857),
                "timeline": timeline,
                "peak_timestamp": peak_timestamp,
                "thermal_timestamp": thermal_timestamp,
                "outage_start": outage_start,
                "outage_end": outage_end,
                "roads": load_animation_roads(bounds_2326),
            }
        )

    energy_limits = finite_limits(pd.concat(energy_pool, ignore_index=True))
    thermal_values = pd.concat(thermal_pool, ignore_index=True)
    thermal_limits = (
        min(float(thermal_values.quantile(0.02)), -0.1),
        max(float(thermal_values.quantile(0.98)), 0.1),
    )
    thermal_norm = TwoSlopeNorm(vmin=thermal_limits[0], vcenter=0.0, vmax=thermal_limits[1])
    letters = "abcdefghijklmno"
    fig = plt.figure(figsize=(17.5, 16.0), constrained_layout=False)
    outer = fig.add_gridspec(
        3, 1, left=0.028, right=0.987, bottom=0.045, top=0.975, hspace=0.17,
    )
    for region_index, payload in enumerate(payloads):
        block = outer[region_index].subgridspec(
            2, 1, height_ratios=[5.0, 1.0], hspace=0.09,
        )
        top = block[0].subgridspec(
            1, 8,
            width_ratios=[2.55, 4.10, 4.10, 0.70, 4.10, 0.70, 4.10, 0.70],
            wspace=0.032,
        )
        legend_axis = fig.add_subplot(top[0])
        maps = [
            fig.add_subplot(top[1]), fig.add_subplot(top[2]),
            fig.add_subplot(top[4]), fig.add_subplot(top[6]),
        ]
        energy_bar = fig.add_subplot(top[3])
        thermal_bar = fig.add_subplot(top[5])
        beri_bar = fig.add_subplot(top[7])
        curve = fig.add_subplot(block[1])
        local = payload["frame"]
        roads = payload["roads"]
        bounds = map_extent(local)
        for axis in maps:
            setup_animation_map(axis, bounds, roads, None)
            # South anchoring gives every district the same visible map-to-profile gap
            # while preserving true map aspect ratios.
            axis.set_anchor("S")
        draw_animation_layer(maps[0], local, colors=local["_color"])
        draw_animation_layer(
            maps[1], local, field="hourly_load_w_m2", cmap="YlOrRd", limits=energy_limits,
        )
        draw_animation_layer(
            maps[2], local, field="thermal_safety_margin_c", cmap="RdYlGn",
            limits=thermal_limits, norm=thermal_norm,
        )
        draw_animation_layer(
            maps[3], local, field="building_energy_resilience_index",
            cmap="RdYlGn", limits=(0.0, 100.0),
        )
        for axis in maps:
            if not roads.empty:
                roads.plot(ax=axis, color="#485256", linewidth=0.18, alpha=0.38, zorder=3)
        base = region_index * 5
        panel_title(maps[0], letters[base], f"{payload['name']} · building type")
        panel_title(maps[1], letters[base + 1], "Normal-supply hourly electricity")
        panel_title(
            maps[2], letters[base + 2],
            "Indoor thermal margin · outage +12 h",
        )
        panel_title(maps[3], letters[base + 3], "Reference-suite BERI")
        for axis in maps:
            axis.title.set_position((0.5, 1.02))

        legend_axis.set_axis_off()
        if region_index == 1:
            legend_axis.legend(
                handles=[
                    Patch(facecolor=color, edgecolor="none", label=name)
                    for name, color in ANIMATION_TYPE_COLORS.items()
                ],
                title="Building type", loc="center right", ncol=1,
                handlelength=1.0, handletextpad=0.38,
                labelspacing=0.28, borderaxespad=0.0, frameon=False,
            )
        add_animation_colorbar(fig, energy_bar, maps[1], "YlOrRd", energy_limits, "Load (W m⁻²)")
        add_animation_colorbar(
            fig, thermal_bar, maps[2], "RdYlGn", thermal_limits,
            f"Safety margin to {CONFIG['thermal']['outage_safety_screen_c']:g} °C (°C)",
            norm=thermal_norm,
        )
        add_animation_colorbar(fig, beri_bar, maps[3], "RdYlGn", (0.0, 100.0), "BERI (0–100)")
        fig.canvas.draw()
        common_title_y = max(axis.get_position().y1 for axis in maps) + 0.004
        for axis in maps:
            box = axis.get_position()
            title_y = (common_title_y - box.y0) / box.height
            # Passing y through set_title disables Matplotlib's automatic
            # per-axis title repositioning, which otherwise reintroduces drift.
            axis.set_title(axis.get_title(), y=title_y, fontsize=12, fontweight="bold")

        timeline = payload["timeline"].sort_values("timestamp_hkt")
        timestamp = payload["peak_timestamp"]
        curve.plot(timeline["timestamp_hkt"], timeline["scope_load_w_m2"], color="#3A7CA5", lw=1.7)
        curve.fill_between(
            timeline["timestamp_hkt"], 0, timeline["scope_load_w_m2"], color="#3A7CA5", alpha=0.09,
        )
        curve.axvline(timestamp, color="#D97706", lw=1.25)
        curve.axvspan(
            payload["outage_start"], payload["outage_end"],
            color="#6B7280", alpha=0.10, lw=0,
        )
        current = timeline.loc[timeline["timestamp_hkt"].eq(timestamp), "scope_load_w_m2"]
        if not current.empty:
            curve.scatter([timestamp], [float(current.iloc[0])], color="#D97706", s=25, zorder=4)
        curve.set_ylim(0.0, float(timeline["scope_load_w_m2"].max()) * 1.40)
        curve.set_ylabel("Area-weighted load\n(W m⁻²)", labelpad=2)
        curve.set_xlabel("Event time (HKT)", labelpad=1)
        curve.grid(axis="y", color="#d9dbdc", linewidth=0.55, alpha=0.65)
        curve.spines[["top", "right"]].set_visible(False)
        curve.tick_params(axis="x", labelrotation=0, pad=2)
        curve.text(
            0.995, 0.96,
            f"{len(local):,} buildings · load map: {timestamp.strftime('%d %b %Y %H:%M HKT')}\n"
            f"thermal map: common outage +12 h ({thermal_timestamp.strftime('%d %b %Y %H:%M HKT')})",
            transform=curve.transAxes, ha="right", va="top", color="#4B5563", zorder=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
        )
        panel_title(curve, letters[base + 4], f"{payload['name']} area-weighted load intensity")
        fig.canvas.draw()
        curve_box = curve.get_position()
        curve_left = maps[0].get_position().x0
        curve_right = maps[3].get_position().x1
        curve.set_position([curve_left, curve_box.y0, curve_right - curve_left, curve_box.height])
    fig.text(
        0.985, 0.012,
        f"{EVENT_LABELS.get(event, event)} · Road basemap: Hong Kong Transport Department / CSDI",
        ha="right", color="#666666",
    )
    save_figure(fig, destination)
    plt.close(fig)


def hourly_prototype_profiles(event_id: str) -> tuple[pd.DatetimeIndex, pd.DataFrame, pd.DataFrame]:
    """Return compact prototype-hour profiles without expanding a city-year table."""
    hourly = pd.read_parquet(RESPONSE_DIR / "energyplus_multizone_hourly_results.parquet")
    available = set(
        hourly.loc[hourly["event_id"].eq(event_id), "supply_stage"].astype(str)
    )
    outage_stage = (
        "outage_common_24h"
        if "outage_common_24h" in available
        else "outage_peak_24h"
    )
    hourly = hourly.loc[
        hourly["event_id"].eq(event_id)
        & hourly["climate_case"].eq("heatwave")
        & hourly["supply_stage"].isin(["normal", outage_stage])
    ].copy()
    hourly["timestamp_hkt"] = pd.to_datetime(hourly["timestamp_hkt"])
    profiles = hourly.groupby(
        ["timestamp_hkt", "prototype_id", "floor_position", "supply_stage"], observed=True
    ).agg(
        load_w_m2=("total_electric_w_m2", "median"),
        exposure_c=("maximum_zone_temperature_c", "median"),
        outdoor_c=("outdoor_temperature_c", "median"),
    ).reset_index()
    event_timestamps = hourly.loc[
        hourly["event_phase"].eq("heatwave"), "timestamp_hkt"
    ]
    timestamps = pd.DatetimeIndex(sorted(event_timestamps.unique()))
    context = profiles.loc[profiles["timestamp_hkt"].isin(timestamps)].groupby(
        ["timestamp_hkt", "supply_stage"], observed=True
    ).agg(
        outdoor_c=("outdoor_c", "median"), load_w_m2=("load_w_m2", "median")
    ).reset_index()
    return timestamps, profiles, context


def building_reference_beri() -> pd.Series:
    if not REFERENCE_BERI_RESULTS.exists():
        raise FileNotFoundError(
            f"Reference-suite BERI missing: {REFERENCE_BERI_RESULTS}. Run Result 3 first."
        )
    frame = pd.read_parquet(
        REFERENCE_BERI_RESULTS,
        columns=["BUILDINGSTRUCTUREID", "building_energy_resilience_index"],
    )
    if frame["BUILDINGSTRUCTUREID"].duplicated().any():
        raise ValueError("Reference-suite BERI is not unique by building")
    if frame["building_energy_resilience_index"].isna().any():
        raise ValueError("Reference-suite BERI contains missing values")
    return frame.set_index("BUILDINGSTRUCTUREID")["building_energy_resilience_index"]


def hourly_outage_stage(profiles: pd.DataFrame) -> str:
    """Return the synchronized outage profile when present."""
    available = set(profiles["supply_stage"].astype(str))
    if "outage_common_24h" in available:
        return "outage_common_24h"
    if "outage_peak_24h" in available:
        return "outage_peak_24h"
    raise ValueError("Hourly profiles contain no 24-hour outage stage")


def map_hourly_state(building: pd.DataFrame, profiles: pd.DataFrame, timestamp: pd.Timestamp) -> tuple[pd.Series, pd.Series]:
    current = profiles.loc[profiles["timestamp_hkt"].eq(timestamp)]
    normal = current.loc[current["supply_stage"].eq("normal")].set_index(["prototype_id", "floor_position"])
    outage = current.loc[
        current["supply_stage"].eq(hourly_outage_stage(profiles))
    ].set_index(["prototype_id", "floor_position"])
    role_weights = floor_role_weight_table(building["floors_value"])

    def component_profile(prototype_field: str, lookup: pd.DataFrame, value_field: str) -> pd.Series:
        values = pd.Series(0.0, index=building.index)
        prototypes = building[prototype_field]
        for floor_position in FLOOR_ROLES:
            weight = role_weights[floor_position]
            active = prototypes.notna() & weight.gt(0)
            if not active.any():
                continue
            keys = pd.MultiIndex.from_arrays(
                [
                    prototypes.loc[active].astype(str),
                    pd.Series(floor_position, index=prototypes.loc[active].index),
                ]
            )
            mapped = lookup[value_field].reindex(keys)
            if mapped.isna().any():
                missing = sorted(set(keys[mapped.isna()]))
                raise ValueError(f"Missing hourly floor-conditioned profiles: {missing[:5]}")
            values.loc[active] += weight.loc[active].to_numpy() * mapped.to_numpy(dtype=float)
        return values

    res_area = pd.to_numeric(building["residential_gfa_m2"], errors="coerce").fillna(0)
    com_area = pd.to_numeric(building["commercial_gfa_m2"], errors="coerce").fillna(0)
    res_area = res_area.where(building["residential_prototype_id"].notna(), 0.0)
    com_area = com_area.where(building["commercial_prototype_id"].notna(), 0.0)
    denominator = (res_area + com_area).replace(0, np.nan)
    res_load = component_profile("residential_prototype_id", normal, "load_w_m2")
    com_load = component_profile("commercial_prototype_id", normal, "load_w_m2")
    load = (res_area * res_load + com_area * com_load) / denominator
    res_temp = component_profile("residential_prototype_id", outage, "exposure_c")
    com_temp = component_profile("commercial_prototype_id", outage, "exposure_c")
    safety_screen = float(CONFIG["thermal"]["outage_safety_screen_c"])
    thermal = safety_screen - (res_area * res_temp + com_area * com_temp) / denominator
    return load, thermal


def animation_display_type(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    result = frame.copy()
    result["_display_type"] = np.where(
        result["main_class"].eq("Mixed-use"),
        "Mixed-use",
        result["residential_prototype_id"].map(ANIMATION_TYPE_LABELS),
    )
    result["_display_type"] = result["_display_type"].fillna(
        result["commercial_prototype_id"].map(ANIMATION_TYPE_LABELS)
    ).fillna("Other commercial")
    result["_color"] = result["_display_type"].map(ANIMATION_TYPE_COLORS)
    return result


def map_extent(frame: gpd.GeoDataFrame, pad_fraction: float = 0.02) -> tuple[float, float, float, float]:
    left, bottom, right, top = (float(value) for value in frame.total_bounds)
    x_pad = max((right - left) * pad_fraction, 1.0)
    y_pad = max((top - bottom) * pad_fraction, 1.0)
    return left - x_pad, bottom - y_pad, right + x_pad, top + y_pad


def load_animation_roads(bounds_2326: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    parts = []
    for path in sorted(ROAD_DIR.glob("centerline_*_simplified5m.geojson")):
        part = gpd.read_file(path, bbox=bounds_2326)
        if not part.empty:
            parts.append(part[["geometry"]])
    if not parts:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=3857)
    return gpd.GeoDataFrame(
        pd.concat(parts, ignore_index=True), geometry="geometry", crs=2326
    ).to_crs(3857)


def setup_animation_map(
    axis: plt.Axes,
    bounds: tuple[float, float, float, float],
    roads: gpd.GeoDataFrame,
    land: gpd.GeoDataFrame | None,
) -> None:
    axis.set_facecolor("#e3e4e0")
    if land is not None and not land.empty:
        land.plot(ax=axis, color="#d9dcd8", edgecolor="#90989a", linewidth=0.55, zorder=0)
    if not roads.empty:
        roads.plot(ax=axis, color="#8e9699", linewidth=0.48, alpha=0.86, zorder=1)
    axis.set_xlim(bounds[0], bounds[2])
    axis.set_ylim(bounds[1], bounds[3])
    axis.set_axis_off()
    axis.set_aspect("equal")


def draw_animation_layer(
    axis: plt.Axes,
    frame: gpd.GeoDataFrame,
    *,
    colors: pd.Series | np.ndarray | None = None,
    field: str | None = None,
    cmap: str | None = None,
    limits: tuple[float, float] | None = None,
    norm: mpl.colors.Normalize | None = None,
    point_size: float | None = None,
) -> None:
    if point_size is not None:
        points = frame.geometry
        if colors is None:
            assert field is not None and cmap is not None and limits is not None
            axis.scatter(
                points.x, points.y, c=frame[field], s=point_size, marker="o",
                cmap=cmap, norm=norm or Normalize(*limits), linewidths=0, alpha=0.84, zorder=2,
            )
        else:
            axis.scatter(
                points.x, points.y, c=np.asarray(colors), s=point_size, marker="o",
                linewidths=0, alpha=0.84, zorder=2,
            )
        return
    if colors is not None:
        frame.plot(ax=axis, color=colors, linewidth=0.08, edgecolor="white", alpha=0.84, zorder=2)
    else:
        assert field is not None and cmap is not None and limits is not None
        frame.plot(
            ax=axis, column=field, cmap=cmap, norm=norm or Normalize(*limits),
            linewidth=0.08, edgecolor="white", alpha=0.84, zorder=2,
        )


def add_animation_colorbar(
    fig: plt.Figure,
    axis: plt.Axes,
    reference_axis: plt.Axes,
    cmap: str,
    limits: tuple[float, float],
    label: str,
    norm: mpl.colors.Normalize | None = None,
) -> None:
    box = axis.get_position()
    reference_box = reference_axis.get_position()
    bar_width = box.width * 0.22
    axis.set_position([
        box.x0 + box.width * 0.04, reference_box.y0,
        bar_width, reference_box.height,
    ])
    scalar = mpl.cm.ScalarMappable(norm=norm or Normalize(*limits), cmap=cmap)
    colorbar = fig.colorbar(scalar, cax=axis, orientation="vertical")
    colorbar.set_label(label, labelpad=2)
    colorbar.ax.tick_params(labelsize=11, length=2, pad=2)
    colorbar.outline.set_visible(False)


def encode_mp4(
    frame_dir: Path,
    destination: Path,
    fps: int,
    start_number: int = 0,
    frame_count: int | None = None,
) -> bool:
    executable = shutil.which("ffmpeg")
    if executable is None:
        try:
            import imageio_ffmpeg
            executable = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            return False
    command = [
        executable, "-y", "-framerate", str(fps), "-start_number", str(start_number),
        "-i", str(frame_dir / "frame_%04d.png"),
    ]
    if frame_count is not None:
        command.extend(["-frames:v", str(frame_count)])
    command.extend([
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(destination),
    ])
    completed = subprocess.run(
        command,
        capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"MP4 encoding failed: {completed.stderr[-2000:]}")
    return True


def valid_animation_frame(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 10_000:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def render_animation_frame(
    map_frame: gpd.GeoDataFrame,
    type_frame: gpd.GeoDataFrame,
    energy_field: str,
    thermal_field: str,
    beri_field: str,
    context: pd.DataFrame,
    timestamp: pd.Timestamp,
    destination: Path,
    title_scope: str,
    energy_limits: tuple[float, float],
    thermal_limits: tuple[float, float],
    roads: gpd.GeoDataFrame,
    land: gpd.GeoDataFrame | None,
    timeline_field: str,
    timeline_label: str,
    timeline_title: str,
    point_size: float | None,
    thermal_panel_title: str,
    legend_width_ratio: float = 3.10,
    timeline_labelpad: float = -2.0,
) -> None:
    fig = plt.figure(figsize=(17.5, 7.2), constrained_layout=False)
    outer = fig.add_gridspec(
        2, 1, height_ratios=[6.15, 0.92],
        left=0.035, right=0.986, bottom=0.105, top=0.955,
        hspace=0.105,
    )
    top = outer[0].subgridspec(
        1, 8,
        width_ratios=[legend_width_ratio, 4.10, 4.10, 0.54, 4.10, 0.54, 4.10, 0.54],
        wspace=0.035,
    )
    legend_axis = fig.add_subplot(top[0])
    maps = [
        fig.add_subplot(top[1]),
        fig.add_subplot(top[2]),
        fig.add_subplot(top[4]),
        fig.add_subplot(top[6]),
    ]
    energy_bar = fig.add_subplot(top[3])
    thermal_bar = fig.add_subplot(top[5])
    beri_bar = fig.add_subplot(top[7])
    curve = fig.add_subplot(outer[1])
    bounds = map_extent(map_frame)
    for axis in maps:
        setup_animation_map(axis, bounds, roads, land)
    draw_animation_layer(maps[0], type_frame, colors=type_frame["_color"], point_size=point_size)
    draw_animation_layer(
        maps[1], map_frame, field=energy_field, cmap="YlOrRd",
        limits=energy_limits, point_size=point_size,
    )
    thermal_norm = TwoSlopeNorm(vmin=thermal_limits[0], vcenter=0.0, vmax=thermal_limits[1])
    draw_animation_layer(
        maps[2], map_frame, field=thermal_field, cmap="RdYlGn",
        limits=thermal_limits, norm=thermal_norm, point_size=point_size,
    )
    draw_animation_layer(
        maps[3], map_frame, field=beri_field, cmap="RdYlGn",
        limits=(0.0, 100.0), point_size=point_size,
    )
    for axis in maps:
        if not roads.empty:
            roads.plot(ax=axis, color="#485256", linewidth=0.18, alpha=0.38, zorder=3)
    panel_title(maps[0], "a", "Building type")
    panel_title(maps[1], "b", "Normal-supply hourly electricity")
    panel_title(maps[2], "c", thermal_panel_title)
    panel_title(maps[3], "d", "Reference-suite Building Energy Resilience Index")

    legend_axis.set_axis_off()
    present = [
        name for name in ANIMATION_TYPE_COLORS
        if name in set(type_frame["_display_type"].dropna())
    ]
    legend_axis.legend(
        handles=[
            Patch(facecolor=ANIMATION_TYPE_COLORS[name], edgecolor="none", label=name)
            for name in present
        ],
        loc="center left", ncol=1,
        handlelength=1.0, handletextpad=0.42, labelspacing=0.48,
        borderaxespad=0.0, frameon=False,
    )
    add_animation_colorbar(
        fig, energy_bar, maps[1], "YlOrRd", energy_limits, "Electric load (W m⁻²)"
    )
    add_animation_colorbar(
        fig, thermal_bar, maps[2], "RdYlGn", thermal_limits,
        f"Safety margin to {CONFIG['thermal']['outage_safety_screen_c']:g} °C (°C)",
        norm=thermal_norm,
    )
    add_animation_colorbar(
        fig, beri_bar, maps[3], "RdYlGn", (0.0, 100.0), "BERI (0–100)"
    )

    timeline = context.sort_values("timestamp_hkt")
    curve.plot(timeline["timestamp_hkt"], timeline[timeline_field], color="#3A7CA5", lw=1.7)
    curve.fill_between(
        timeline["timestamp_hkt"], 0, timeline[timeline_field], color="#3A7CA5", alpha=0.09,
    )
    current = timeline.loc[timeline["timestamp_hkt"].eq(timestamp), timeline_field]
    curve.axvline(timestamp, color="#D97706", lw=1.25)
    if not current.empty:
        curve.scatter([timestamp], [float(current.iloc[0])], color="#D97706", s=25, zorder=4)
    curve.set_ylabel(timeline_label, labelpad=timeline_labelpad)
    curve.set_xlabel("Event time (HKT)", labelpad=1)
    curve.grid(axis="y", color="#d9dbdc", linewidth=0.55, alpha=0.65)
    curve.spines[["top", "right"]].set_visible(False)
    curve.tick_params(axis="x", labelrotation=0, pad=2)
    panel_title(curve, "e", timeline_title)
    fig.text(
        0.018, 0.025, f"{title_scope} | {timestamp.strftime('%d %b %Y %H:%M HKT')}",
        ha="left", fontsize=11, color="#303638",
    )
    fig.text(
        0.985, 0.025, "Road basemap: Hong Kong Transport Department / CSDI",
        ha="right", fontsize=11, color="#666666",
    )
    try:
        if destination.name.startswith(("Figure_", "Supplementary_Figure_")):
            save_figure(fig, destination)
        else:
            fig.savefig(destination, dpi=180, facecolor="white")
    finally:
        plt.close(fig)
        fig.clear()


def animation_clips(
    frame_dir: Path,
    prefix: str,
    timestamps: pd.DatetimeIndex,
    timeline: pd.DataFrame,
    timeline_field: str,
    fps: int,
) -> dict:
    missing_or_invalid = [
        number for number in range(len(timestamps))
        if not valid_animation_frame(frame_dir / f"frame_{number:04d}.png")
    ]
    if missing_or_invalid:
        raise RuntimeError(
            f"Animation frame set is incomplete: {frame_dir}; "
            f"missing_or_invalid={missing_or_invalid[:10]}"
        )
    peak_timestamp = pd.Timestamp(
        timeline.loc[timeline[timeline_field].idxmax(), "timestamp_hkt"]
    )
    peak_index = int(np.argmin(np.abs(timestamps - peak_timestamp)))
    peak_start = max(0, min(peak_index - 24, len(timestamps) - min(48, len(timestamps))))
    day_start = max(0, min(peak_index - peak_index % 24, len(timestamps) - min(24, len(timestamps))))
    specifications = {
        "full_event": (0, len(timestamps)),
        "peak_48h": (peak_start, min(48, len(timestamps) - peak_start)),
        "peak_day": (day_start, min(24, len(timestamps) - day_start)),
    }
    outputs = {}
    for name, (start, count) in specifications.items():
        destination = ANIMATION_DIR / f"{prefix}_{name}.mp4"
        outputs[name] = {
            "mp4": encode_mp4(frame_dir, destination, fps, start, count),
            "file": destination.name,
            "start_hkt": str(timestamps[start]),
            "end_hkt": str(timestamps[start + count - 1]),
            "frames": int(count),
        }
    return outputs


def render_hourly_animations(
    model_inputs: gpd.GeoDataFrame,
    event_id: str,
    hour_step: int,
    fps: int,
) -> dict:
    """Create one-point-per-building Hong Kong and footprint-level Ho Man Tin animations."""
    ANIMATION_DIR.mkdir(parents=True, exist_ok=True)
    timestamps, profiles, context = hourly_prototype_profiles(event_id)
    timestamps = timestamps[::max(1, hour_step)]
    indexed = model_inputs.loc[model_inputs["pre_weather_simulation_candidate"]].copy()
    indexed = indexed.set_index("BUILDINGSTRUCTUREID", drop=False)
    if "wrf_south_north" not in indexed or "wrf_west_east" not in indexed:
        mapping = pd.read_parquet(
            HAZARD_DIR / "building_wrf_land_grid_mapping.parquet",
            columns=["BUILDINGSTRUCTUREID", "wrf_south_north", "wrf_west_east"],
        ).drop_duplicates("BUILDINGSTRUCTUREID")
        mapping["BUILDINGSTRUCTUREID"] = pd.to_numeric(mapping["BUILDINGSTRUCTUREID"], errors="raise").astype("int64")
        mapping = mapping.set_index("BUILDINGSTRUCTUREID")
        indexed = indexed.join(mapping, how="left", validate="one_to_one")
    indexed = animation_display_type(indexed)
    beri = building_reference_beri()
    indexed["building_energy_resilience_index"] = indexed["BUILDINGSTRUCTUREID"].map(beri)
    if indexed["building_energy_resilience_index"].isna().any():
        raise ValueError(f"BERI does not cover all animation buildings for {event_id}")
    normal_values = profiles.loc[profiles["supply_stage"].eq("normal"), "load_w_m2"]
    outage_stage = hourly_outage_stage(profiles)
    outage_values = (
        float(CONFIG["thermal"]["outage_safety_screen_c"])
        - profiles.loc[profiles["supply_stage"].eq(outage_stage), "exposure_c"]
    )
    energy_limits = finite_limits(normal_values.rename("hourly_load_w_m2"))
    thermal_values = outage_values.replace([np.inf, -np.inf], np.nan).dropna()
    thermal_limits = (
        min(float(thermal_values.quantile(0.02)), -0.1),
        max(float(thermal_values.quantile(0.98)), 0.1),
    )
    outputs = {}
    common_timing = pd.read_csv(OUTAGE_TIMING_AUDIT)
    common_timing = common_timing.loc[
        common_timing["event_id"].eq(event_id)
        & common_timing["supply_stage"].eq("outage_common_24h")
    ]
    if common_timing.empty:
        raise ValueError(f"Missing synchronized outage timing for {event_id}")
    outage_starts = pd.to_datetime(common_timing["outage_start_hkt"], utc=True).dt.tz_convert("Asia/Hong_Kong")
    outage_ends = pd.to_datetime(common_timing["outage_end_exclusive_hkt"], utc=True).dt.tz_convert("Asia/Hong_Kong")
    if outage_starts.nunique() != 1 or outage_ends.nunique() != 1:
        raise ValueError(f"Synchronized outage timing differs among prototypes for {event_id}")
    common_outage_start = pd.Timestamp(outage_starts.iloc[0])
    common_outage_end = pd.Timestamp(outage_ends.iloc[0])

    def thermal_title_at(timestamp: pd.Timestamp) -> str:
        current = pd.Timestamp(timestamp)
        if current.tzinfo is None:
            current = current.tz_localize("Asia/Hong_Kong")
        else:
            current = current.tz_convert("Asia/Hong_Kong")
        if current < common_outage_start:
            return "Indoor thermal safety (pre-outage)"
        if current < common_outage_end:
            hour = int((current - common_outage_start).total_seconds() // 3600) + 1
            return f"Indoor thermal safety (outage hour {hour}/24)"
        hour = int((current - common_outage_end).total_seconds() // 3600) + 1
        return f"Indoor thermal safety (recovery hour {hour})"

    local = homantin_subset(indexed).copy()
    local_dir = ANIMATION_DIR / f"Ho_Man_Tin_{event_id}_{ANIMATION_LAYOUT_VERSION}_frames"
    local_dir.mkdir(parents=True, exist_ok=True)
    local_area = (
        pd.to_numeric(local["residential_gfa_m2"], errors="coerce").fillna(0)
        + pd.to_numeric(local["commercial_gfa_m2"], errors="coerce").fillna(0)
    )
    local_timeline_rows = []
    local_energy_values = []
    for timestamp in timestamps:
        load, _ = map_hourly_state(local, profiles, timestamp)
        local_energy_values.append(load.dropna())
        local_timeline_rows.append({
            "timestamp_hkt": timestamp,
            "scope_load_w_m2": float((load.fillna(0) * local_area).sum() / local_area.sum()),
        })
    local_timeline = pd.DataFrame(local_timeline_rows)
    local_energy = pd.concat(local_energy_values, ignore_index=True)
    local_energy_limits = finite_limits(local_energy.rename("hourly_load_w_m2"))
    local_bounds_2326 = map_extent(local)
    local_roads = load_animation_roads(local_bounds_2326)
    local_3857 = local.to_crs(3857)
    for number, timestamp in enumerate(timestamps):
        destination = local_dir / f"frame_{number:04d}.png"
        if valid_animation_frame(destination):
            if number % 12 == 0 or number == len(timestamps) - 1:
                print(
                    f"[Animation Ho Man Tin {event_id}] "
                    f"{number + 1}/{len(timestamps)} frame cached",
                    flush=True,
                )
            continue
        load, thermal = map_hourly_state(local, profiles, timestamp)
        local_3857["hourly_load_w_m2"] = load.to_numpy()
        local_3857["thermal_safety_margin_c"] = thermal.to_numpy()
        render_animation_frame(
            local_3857, local_3857, "hourly_load_w_m2", "thermal_safety_margin_c",
            "building_energy_resilience_index",
            local_timeline, timestamp, destination,
            f"Ho Man Tin, {len(local):,} buildings", local_energy_limits, thermal_limits,
            local_roads, None, "scope_load_w_m2", "HMT load (W m⁻²)",
            "Normal-supply area-weighted load intensity", None,
            thermal_title_at(timestamp),
            legend_width_ratio=2.55, timeline_labelpad=-2.0,
        )
        if number % 12 == 0:
            gc.collect()
            print(
                f"[Animation Ho Man Tin {event_id}] "
                f"{number + 1}/{len(timestamps)} frames rendered",
                flush=True,
            )
    outputs["ho_man_tin"] = animation_clips(
        local_dir, f"Ho_Man_Tin_{event_id}", timestamps, local_timeline,
        "scope_load_w_m2", fps,
    )

    full = indexed.dropna(subset=["wrf_south_north", "wrf_west_east"]).copy()
    full_points = full.copy()
    full_points["geometry"] = full_points.geometry.centroid
    full_points = full_points.to_crs(3857)
    full_bounds_2326 = map_extent(full, 0.025)
    full_roads = load_animation_roads(full_bounds_2326)
    land = gpd.read_parquet(HMA_BOUNDARIES).to_crs(3857)
    city = pd.read_parquet(RESPONSE_DIR / "citywide_coincident_hourly_response.parquet")
    city["timestamp_hkt"] = pd.to_datetime(city["timestamp_hkt"])
    city_timeline = city.loc[
        city["event_id"].eq(event_id)
        & city["climate_case"].eq("heatwave")
        & city["supply_stage"].eq("normal")
        & city["event_phase"].eq("heatwave"),
        ["timestamp_hkt", "city_average_load_kw"],
    ].copy()
    city_timeline["city_load_gw"] = city_timeline["city_average_load_kw"] / 1e6
    full_energy_limits = energy_limits
    hk_dir = ANIMATION_DIR / f"Hong_Kong_{event_id}_{ANIMATION_LAYOUT_VERSION}_frames"
    hk_dir.mkdir(parents=True, exist_ok=True)
    for number, timestamp in enumerate(timestamps):
        destination = hk_dir / f"frame_{number:04d}.png"
        if valid_animation_frame(destination):
            if number % 6 == 0 or number == len(timestamps) - 1:
                print(
                    f"[Animation Hong Kong {event_id}] "
                    f"{number + 1}/{len(timestamps)} frame cached",
                    flush=True,
                )
            continue
        load, thermal = map_hourly_state(full, profiles, timestamp)
        full_points["hourly_load_w_m2"] = load.to_numpy()
        full_points["thermal_safety_margin_c"] = thermal.to_numpy()
        render_animation_frame(
            full_points, full_points, "hourly_load_w_m2", "thermal_safety_margin_c",
            "building_energy_resilience_index",
            city_timeline, timestamp, destination,
            f"Hong Kong, {len(full):,} individual buildings", full_energy_limits, thermal_limits,
            full_roads, land, "city_load_gw", "Hong Kong load (GW)",
            "Normal-supply aggregate load", 1.15,
            thermal_title_at(timestamp),
        )
        if number % 6 == 0:
            gc.collect()
            print(
                f"[Animation Hong Kong {event_id}] "
                f"{number + 1}/{len(timestamps)} frames rendered",
                flush=True,
            )
    outputs["hong_kong"] = animation_clips(
        hk_dir, f"Hong_Kong_{event_id}", timestamps, city_timeline,
        "city_load_gw", fps,
    )
    outputs.update({
        "event_id": event_id,
        "hour_step": int(hour_step),
        "frames": int(len(timestamps)),
        "hong_kong_buildings": int(len(full)),
        "ho_man_tin_buildings": int(len(local)),
        "display_note": "Hong Kong uses one centroid point per simulated building; Ho Man Tin retains building footprints.",
        "layout": "single-column building-type key, four aligned maps with adjacent vertical keys, and one compact scope-specific load timeline",
        "layout_version": ANIMATION_LAYOUT_VERSION,
        "hourly_dynamic_panels": ["hourly electricity demand", "indoor thermal safety", "scope-specific load profile"],
        "static_panels": ["building type", "baseline reference-suite Building Energy Resilience Index"],
        "beri_time_basis": "one baseline rating per building from the frozen multi-event reference stress suite; constant across historical-event videos for rating vintage HK_BERI_2025_v1",
        "building_types": list(ANIMATION_TYPE_COLORS),
    })
    return outputs


def assemble_event_map_slice(
    model_inputs: gpd.GeoDataFrame,
    results: pd.DataFrame,
    event_id: str,
    energy_field: str = "heatwave_energy_kwh",
    resilience_field: str = "safety_resistance_hours_to_32c",
) -> tuple[gpd.GeoDataFrame, dict]:
    """Create one observed event map slice with normal-supply energy and 24-h outage safety."""
    selected, selection = select_result_slice(
        results, event_id, "observed_2018_2023", "observed",
        "outage_peak_24h", int(CONFIG["random_seed"]),
    )
    energy_slice = results.loc[
        results["event_id"].eq(event_id)
        & results["climate_period"].eq("observed_2018_2023")
        & results["climate_scenario"].eq("observed")
        & results["supply_stage"].eq("normal")
        & results["behavior_seed"].eq(int(CONFIG["random_seed"])),
        ["BUILDINGSTRUCTUREID", energy_field, "heatwave_relative_energy_increase"],
    ].copy()
    if energy_slice["BUILDINGSTRUCTUREID"].duplicated().any():
        raise ValueError(f"Duplicate normal-supply energy rows for {event_id}")
    selected = selected.drop(
        columns=[energy_field, "heatwave_relative_energy_increase"], errors="ignore"
    ).merge(energy_slice, on="BUILDINGSTRUCTUREID", how="left", validate="one_to_one")
    selected = selected.drop(columns=["main_class"], errors="ignore")
    selected = model_inputs.merge(
        selected, on="BUILDINGSTRUCTUREID", how="inner", validate="one_to_one"
    )
    selected = gpd.GeoDataFrame(selected, geometry="geometry", crs=model_inputs.crs)
    selected["building_energy_resilience_index"] = selected["BUILDINGSTRUCTUREID"].map(
        building_reference_beri()
    )
    required = [energy_field, resilience_field, "building_energy_resilience_index"]
    if selected[required].isna().any().any():
        missing = selected[required].isna().sum().to_dict()
        raise ValueError(f"Incomplete event map fields for {event_id}: {missing}")
    selection["energy_supply_stage"] = "normal"
    selection["resilience_supply_stage"] = selection.pop("supply_stage")
    return selected, selection


def render_supplementary_methods(destination: Path) -> None:
    """Move stock-flow and prototype-design evidence out of the main Results figures."""
    flow = pd.read_csv(AUDIT_DIR / "building_sample_flow.csv")
    registry = pd.read_csv(AUDIT_DIR / "prototype_registry.csv")
    design = pd.read_csv(RESPONSE_DIR / "thermal_experiment_design.csv")
    by_type = pd.read_csv(RESPONSE_DIR / "final_building_sample_by_type.csv")
    fig, axes = plt.subplots(2, 2, figsize=(16.2, 11.2), constrained_layout=False)
    fig.subplots_adjust(left=0.19, right=0.97, bottom=0.09, top=0.95, wspace=0.36, hspace=0.36)

    flow_label_map = {
        "released_repaired_stock": "Released repaired\nstock",
        "valid_geometry": "Valid geometry",
        "classified_buildings": "Classified\nbuildings",
        "target_scope_before_geometry_control": "Residential, commercial or mixed-use\nbefore geometry control",
        "target_scope_with_valid_geometry": "Target scope with\nvalid geometry",
        "valid_mixed_use_component_share": "Valid mixed-use\ncomponent shares",
        "valid_simulation_attributes": "Complete simulation\nattributes",
        "valid_residential_commercial_component_area": "Valid component\nfloor-area split",
        "final_pre_weather_simulation_candidate": "Final simulation\nsample",
    }
    flow_labels = flow["step"].map(flow_label_map)
    if flow_labels.isna().any():
        raise ValueError(f"Unlabelled sample-flow steps: {flow.loc[flow_labels.isna(), 'step'].tolist()}")
    y = np.arange(len(flow))
    axes[0, 0].hlines(y, 0, flow["building_count"], color="#C7D2D9", lw=2.2)
    axes[0, 0].scatter(flow["building_count"], y, s=70, color="#1F6F8B", zorder=3)
    for row, record in enumerate(flow.itertuples(index=False)):
        axes[0, 0].text(
            record.building_count + 3200, row, f"{record.building_count:,}",
            va="center", ha="left", zorder=6,
        )
    axes[0, 0].set_yticks(y, flow_labels.tolist())
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlim(300000, 380000)
    axes[0, 0].set_xlabel("Buildings")
    axes[0, 0].grid(axis="x", color="#E3E7E8", lw=0.55)
    panel_title(axes[0, 0], "a", "Traceable stock-to-simulation flow")

    registry = registry.assign(
        sector_label=registry["sector"].str.title(),
        prototype_label=registry["source_subclass"].replace(
            {
                "Human Health": "Healthcare",
                "Public Rental Housing": "Public rental\nhousing",
                "Subsidised Sale Flats": "Subsidised sale\nflats",
                "Old Tenement / Low-rise Housing": "Old tenement /\nlow-rise housing",
                "Mixed-use non-residential component": "Mixed-use\ncommercial part",
            }
        ),
    )
    registry = registry.sort_values(["sector", "prototype_label"]).reset_index(drop=True)
    y = np.arange(len(registry))
    sector_color = {"Residential": "#2F75B5", "Commercial": "#D94841"}
    for row, record in enumerate(registry.itertuples(index=False)):
        axes[0, 1].scatter(
            0 if record.sector_label == "Residential" else 1, row, s=95,
            color=sector_color[record.sector_label], edgecolor="white", linewidth=0.7,
        )
    axes[0, 1].set_xticks([0, 1], ["Residential\n5 prototypes", "Commercial\n9 prototypes"])
    axes[0, 1].set_yticks(y, registry["prototype_label"])
    axes[0, 1].tick_params(axis="y", pad=5)
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlim(-0.55, 1.55)
    axes[0, 1].grid(axis="x", color="#E3E7E8", lw=0.55)
    panel_title(axes[0, 1], "b", "Use-specific prototype taxonomy")

    behaviour_order = sorted(design["behaviour"].astype(str).unique())
    behaviour_coverage = design.groupby(
        ["sector", "behaviour", "setpoint_c"], observed=True
    ).size().rename("cases").reset_index()
    sector_offset = {"residential": -0.14, "commercial": 0.14}
    maximum_cases = max(float(behaviour_coverage["cases"].max()), 1.0)
    for record in behaviour_coverage.itertuples(index=False):
        axes[1, 0].scatter(
            behaviour_order.index(str(record.behaviour)) + sector_offset[str(record.sector)],
            float(record.setpoint_c), s=35 + 230 * float(record.cases) / maximum_cases,
            color=sector_color[str(record.sector).title()], alpha=0.68,
            edgecolor="white", linewidth=0.6,
        )
    axes[1, 0].set_xticks(np.arange(len(behaviour_order)), [name.replace("_", " ") for name in behaviour_order], rotation=18)
    axes[1, 0].set_yticks([24, 25, 26])
    axes[1, 0].set_xlabel("Use-specific occupancy and operating schedule")
    axes[1, 0].set_ylabel("Cooling setpoint (°C)")
    axes[1, 0].set_ylim(23.0, 27.0)
    axes[1, 0].legend(
        handles=[Patch(facecolor=color, edgecolor="none", label=label)
                 for label, color in sector_color.items()],
        frameon=False, loc="lower left", ncol=2,
    )
    axes[1, 0].grid(color="#E3E7E8", lw=0.55)
    panel_title(axes[1, 0], "c", f"Frozen factorial design ({len(design)} cases)")

    counts = design.groupby(["prototype_id", "floor_position"], observed=True).size().unstack(fill_value=0)
    prototype_order = registry["prototype_id"].tolist()
    floor_order = [name for name in FLOOR_ROLES if name in counts.columns]
    counts = counts.reindex(index=prototype_order, columns=floor_order, fill_value=0)
    maximum = max(float(counts.to_numpy().max()), 1.0)
    for row, prototype in enumerate(prototype_order):
        for column, floor in enumerate(floor_order):
            value = int(counts.loc[prototype, floor])
            axes[1, 1].scatter(
                column, row, s=22 + 260 * value / maximum,
                color="#6A51A3" if value else "#E5E7EB", alpha=0.75,
                edgecolor="white", linewidth=0.5,
            )
            axes[1, 1].annotate(
                str(value), (column, row), xytext=(-13, 0), textcoords="offset points",
                ha="right", va="center", zorder=6,
            )
    axes[1, 1].set_xticks(np.arange(len(floor_order)), [name.title() for name in floor_order])
    axes[1, 1].set_yticks(np.arange(len(prototype_order)), registry["prototype_label"])
    axes[1, 1].tick_params(axis="y", pad=5)
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlim(-0.80, len(floor_order) - 0.45)
    axes[1, 1].set_ylim(len(prototype_order) - 0.45, -0.55)
    axes[1, 1].grid(color="#E3E7E8", lw=0.55)
    panel_title(axes[1, 1], "d", "Prototype and floor-position coverage")
    save_figure(fig, destination)
    plt.close(fig)


def render_supplementary_event_trajectories(destination: Path) -> None:
    """Same outcome views as Figure 6a-b for the two non-featured observed events."""
    hourly = pd.read_parquet(RESPONSE_DIR / "citywide_coincident_hourly_response.parquet")
    event_ids = ["HW_20180526_20180601", "HW_20230724_20230728"]
    stage_order = ["normal", "capacity_70pct", "outage_peak_6h", "outage_peak_24h", "outage_full_event"]
    stage_labels = ["Normal", "70% capacity", "6-h outage", "24-h outage", "Full-event outage"]
    stage_colors = ["#20639B", "#3CAEA3", "#F6D55C", "#ED553B", "#7A5195"]
    fig, axes = plt.subplots(2, 2, figsize=(15.8, 8.4), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.105, top=0.94, wspace=0.16, hspace=0.34)
    for column, event_id in enumerate(event_ids):
        frame = hourly.loc[hourly["event_id"].eq(event_id)].copy()
        frame["timestamp_hkt"] = pd.to_datetime(frame["timestamp_hkt"])
        paired = frame.loc[
            frame["supply_stage"].eq("normal") & frame["event_phase"].eq("heatwave")
        ].pivot(index="timestamp_hkt", columns="climate_case", values="city_average_load_kw") / 1e6
        axes[0, column].plot(paired.index, paired["counterfactual"], color="#8DA0AE", lw=1.35,
                             label="Matched counterfactual")
        axes[0, column].plot(paired.index, paired["heatwave"], color="#C23B22", lw=1.8,
                             label="Heatwave")
        axes[0, column].fill_between(
            paired.index, paired["counterfactual"], paired["heatwave"],
            where=paired["heatwave"].ge(paired["counterfactual"]),
            color="#F2A65A", alpha=0.32, label="Heatwave-attributable load",
        )
        paired_min = float(np.nanmin(paired.to_numpy(dtype=float)))
        paired_max = float(np.nanmax(paired.to_numpy(dtype=float)))
        paired_span = max(paired_max - paired_min, 0.5)
        axes[0, column].set_ylim(
            paired_min - 0.04 * paired_span,
            paired_max + 0.25 * paired_span,
        )
        axes[0, column].set_ylabel("Hong Kong load (GW)")
        axes[0, column].xaxis.set_major_locator(mdates.DayLocator(interval=2))
        axes[0, column].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        axes[0, column].legend(frameon=False, loc="upper left", ncol=3)
        axes[0, column].grid(color="#E3E7E8", lw=0.55)
        panel_title(axes[0, column], chr(ord("a") + column), EVENT_LABELS[event_id])

        heatwave = frame.loc[frame["climate_case"].eq("heatwave")].copy()
        normal_peak = heatwave.loc[heatwave["supply_stage"].eq("normal"), "city_average_load_kw"].max()
        for stage, label, color in zip(stage_order, stage_labels, stage_colors):
            group = heatwave.loc[heatwave["supply_stage"].eq(stage)].sort_values("timestamp_hkt")
            axes[1, column].plot(
                group["timestamp_hkt"], group["city_average_load_kw"] / normal_peak,
                color=color, lw=1.45, alpha=0.9, label=label,
            )
        normalized_max = max(float(line.get_ydata().max()) for line in axes[1, column].lines)
        axes[1, column].axhline(1.0, color="#6B7280", lw=0.7, ls="--")
        axes[1, column].set_ylim(-0.04, max(1.15, 1.24 * normalized_max))
        axes[1, column].set_ylabel("Load relative to event\nnormal-supply peak")
        axes[1, column].set_xlabel("Hong Kong time")
        axes[1, column].xaxis.set_major_locator(mdates.DayLocator(interval=2))
        axes[1, column].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        axes[1, column].legend(frameon=False, loc="upper left", ncol=3)
        axes[1, column].grid(color="#E3E7E8", lw=0.55)
        panel_title(axes[1, column], chr(ord("c") + column), "Supply-stage trajectory")
    for axis in axes.flat:
        axis.tick_params(axis="x", rotation=18)
    save_figure(fig, destination)
    plt.close(fig)


def render_supplementary_figures(
    full_stock: gpd.GeoDataFrame,
    model_inputs: gpd.GeoDataFrame,
    results: pd.DataFrame,
    output: Path,
) -> None:
    render_supplementary_methods(output / "Supplementary_Figure_S1_methods_and_sample_audit.png")
    for number, event_id in [(2, "HW_20180526_20180601"), (3, "HW_20230724_20230728")]:
        event_map, selection = assemble_event_map_slice(model_inputs, results, event_id)
        render_citywide_triptych(
            full_stock, event_map, "heatwave_energy_kwh", "safety_resistance_hours_to_32c",
            "building_energy_resilience_index", selection,
            output / f"Supplementary_Figure_S{number}_{event_id}_Hong_Kong_maps.png",
        )
    render_figure_4(
        model_inputs, results,
        output / "Supplementary_Figure_S4_HW_20180526_20180601_three_districts.png",
        "HW_20180526_20180601",
    )
    render_figure_4(
        model_inputs, results,
        output / "Supplementary_Figure_S5_HW_20230724_20230728_three_districts.png",
        "HW_20230724_20230728",
    )
    render_supplementary_event_trajectories(
        output / "Supplementary_Figure_S6_nonfeatured_event_trajectories.png"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--full-stock", type=Path, default=FULL_STOCK)
    parser.add_argument("--model-inputs", type=Path, default=MODEL_INPUTS)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--write-contract-only", action="store_true")
    parser.add_argument(
        "--event-id", default="HW_20220722_20220729",
        help="Featured observed event for Figures 3 and 6; 2018 and 2023 are rendered in SI.",
    )
    parser.add_argument("--climate-period")
    parser.add_argument("--climate-scenario")
    parser.add_argument("--supply-stage")
    parser.add_argument("--behavior-seed", type=int)
    parser.add_argument("--energy-field", default="heatwave_energy_kwh")
    parser.add_argument("--resilience-field", default="safety_resistance_hours_to_32c")
    parser.add_argument("--skip-animations", action="store_true")
    parser.add_argument("--animations-only", action="store_true")
    parser.add_argument("--animation-event", help="Render one event only; overrides --animation-events.")
    parser.add_argument(
        "--animation-events", nargs="+",
        default=["HW_20180526_20180601", "HW_20220722_20220729", "HW_20230724_20230728"],
        help="Observed events used to produce separate full-event, peak-48-h and peak-day clips.",
    )
    parser.add_argument("--animation-hour-step", type=int, default=1)
    parser.add_argument("--animation-fps", type=int, default=12)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_figure_contract(output)
    if args.write_contract_only:
        print(f"04 figure contract complete: {output}")
        return
    if args.animations_only:
        model_columns = [
            "BUILDINGSTRUCTUREID", "geometry", "main_class", "residential_gfa_m2",
            "commercial_gfa_m2", "floors_value", "residential_prototype_id", "commercial_prototype_id",
            "pre_weather_simulation_candidate",
        ]
        model_inputs = gpd.read_parquet(args.model_inputs.resolve(), columns=model_columns).drop_duplicates("BUILDINGSTRUCTUREID")
        event_ids = [args.animation_event] if args.animation_event else args.animation_events
        animations = {
            event_id: render_hourly_animations(
                model_inputs, event_id, args.animation_hour_step, args.animation_fps
            )
            for event_id in event_ids
        }
        animation = {
            "events": animations,
            "event_count": len(animations),
            "total_rendered_frames_per_scope": int(sum(item["frames"] for item in animations.values())),
        }
        (ANIMATION_DIR / "animation_manifest.json").write_text(
            json.dumps(animation, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_results_release_manifest()
        print(
            f"04 animations complete: {output} | events={len(animations)} | "
            f"frames_per_scope={animation['total_rendered_frames_per_scope']}"
        )
        return
    for path in [args.full_stock.resolve(), args.model_inputs.resolve(), args.results.resolve()]:
        if not path.exists():
            raise FileNotFoundError(f"Required frozen figure input does not exist: {path}")
    full_stock = gpd.read_parquet(
        args.full_stock.resolve(), columns=["BUILDINGSTRUCTUREID", "hkbfetd_main_class", "geometry"]
    )
    results = pd.read_parquet(args.results.resolve())
    selected, selection = select_result_slice(
        results,
        args.event_id,
        args.climate_period,
        args.climate_scenario,
        args.supply_stage,
        args.behavior_seed,
    )
    energy_slice = results.copy()
    for field in ["event_id", "climate_period", "climate_scenario", "behavior_seed"]:
        energy_slice = energy_slice.loc[energy_slice[field].astype(str).eq(str(selection[field]))]
    energy_slice = energy_slice.loc[
        energy_slice["supply_stage"].eq("normal"),
        ["BUILDINGSTRUCTUREID", args.energy_field, "heatwave_relative_energy_increase"],
    ].copy()
    if energy_slice["BUILDINGSTRUCTUREID"].duplicated().any():
        raise ValueError("Normal-supply energy map slice contains duplicate buildings")
    selected = selected.drop(
        columns=[args.energy_field, "heatwave_relative_energy_increase"], errors="ignore"
    ).merge(
        energy_slice, on="BUILDINGSTRUCTUREID", how="left", validate="one_to_one"
    )
    selection["energy_supply_stage"] = "normal"
    selection["resilience_supply_stage"] = selection.pop("supply_stage")
    model_columns = [
        "BUILDINGSTRUCTUREID", "geometry", "main_class", "residential_gfa_m2",
        "commercial_gfa_m2", "floors_value", "residential_prototype_id", "commercial_prototype_id",
        "pre_weather_simulation_candidate",
    ]
    model_inputs = gpd.read_parquet(args.model_inputs.resolve(), columns=model_columns).drop_duplicates("BUILDINGSTRUCTUREID")
    selected = selected.drop(columns=["main_class"], errors="ignore")
    selected = model_inputs.merge(
        selected, on="BUILDINGSTRUCTUREID", how="inner", validate="one_to_one"
    )
    selected = gpd.GeoDataFrame(selected, geometry="geometry", crs=model_inputs.crs)
    beri = building_reference_beri()
    selected["building_energy_resilience_index"] = selected["BUILDINGSTRUCTUREID"].map(beri)
    if selected["building_energy_resilience_index"].isna().any():
        raise ValueError("BERI does not cover every building in the selected Figure 3 slice")
    render_citywide_triptych(
        full_stock,
        selected,
        args.energy_field,
        args.resilience_field,
        "building_energy_resilience_index",
        selection,
        output / "Figure_3_Hong_Kong_building_energy_resilience_maps.png",
    )
    (output / "Figure_3_selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    render_figure_1(full_stock, output / "Figure_1_heatwave_hazard_validation_and_projection.png")
    render_figure_2(output / "Figure_2_layered_validation.png")
    render_figure_4(
        model_inputs, results, output / "Figure_4_three_district_energy_resilience.png",
        args.event_id,
    )
    render_figure_5(output / "Figure_5_building_heterogeneity_and_drivers.png")
    render_figure_6(output / "Figure_6_supply_disruption_and_future_response.png")
    render_supplementary_figures(full_stock, model_inputs, results, output)
    animation_manifest_path = ANIMATION_DIR / "animation_manifest.json"
    animation = {"skipped": True}
    if args.skip_animations and animation_manifest_path.exists():
        animation = json.loads(animation_manifest_path.read_text(encoding="utf-8"))
        animation["reused_existing"] = True
    elif not args.skip_animations:
        event_ids = [args.animation_event] if args.animation_event else args.animation_events
        animations = {
            event_id: render_hourly_animations(
                model_inputs, event_id, args.animation_hour_step, args.animation_fps
            )
            for event_id in event_ids
        }
        animation = {
            "events": animations,
            "event_count": len(animations),
            "total_rendered_frames_per_scope": int(sum(item["frames"] for item in animations.values())),
        }
        animation_manifest_path.write_text(
            json.dumps(animation, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    manifest = {
        "figures": sorted(path.name for path in output.glob("Figure_*.png")),
        "figure_count": len(list(output.glob("Figure_*.png"))),
        "supplementary_figures": sorted(path.name for path in output.glob("Supplementary_Figure_*.png")),
        "supplementary_figure_count": len(list(output.glob("Supplementary_Figure_*.png"))),
        "animations": animation,
        "font": "Times New Roman",
        "body_font_pt": 11,
        "panel_title_font_pt": 12,
        "panel_labels": "lowercase_parenthesized",
        "regional_bounds_epsg2326": REGION_BOUNDS_2326,
        "regional_bounds_source": REGION_BOUNDS_SOURCE,
    }
    (output / "publication_figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_results_release_manifest()
    print(f"04 complete: {output} | figures={manifest['figure_count']} | animations_skipped={args.skip_animations}")


if __name__ == "__main__":
    main()
