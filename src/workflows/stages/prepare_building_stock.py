# -*- coding: utf-8 -*-
"""Prepare an uncertainty-aware Hong Kong residential stock for heat studies.

This pipeline deliberately does not alter the published HK-BFETD workflow. It
uses raw LandsD/BD evidence, optionally fuses the published v1.1 CSV, and keeps
observations, allocations, model estimates, and unusable raw fields separate.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import pyogrio
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
LANDSD = DATA / "LandsD"
DEFAULT_OUTPUT = ROOT / "intermediate" / "prepared_stock" / "source"
CURRENT_YEAR = 2026

FILES = {
    "buildings": LANDSD / "1.Building_Footprint_Public_20250206.gdb_BUILDING_STRUCTURE_converted.json",
    "names": LANDSD / "2.Building_Footprint_Public_20250206.gdb_BUILDING_NAME_converted.json",
    "info": LANDSD / "3.Building_Footprint_Public_20250206.gdb_BUILDING_INFO_converted.json",
    "works": LANDSD / "4.Building_Footprint_Public_20250206.gdb_BUILDING_WORKS_HISTORY_converted.json",
    "permits": LANDSD / "6.Building_Footprint_Public_20250206.gdb_OCCUPATION_PERMIT_converted.json",
    "permit_links": LANDSD / "7.Building_Footprint_Public_20250206.gdb_OP_BUILDING_STRUCTURE_converted.json",
    "bdbiar": DATA / "BDBIAR.gdb_converted.csv",
    "zones": DATA / "OZP_Zones",
    "dtm": ROOT / "intermediate" / "source_derived" / "landsd_dtm_building_points.csv",
}

HKBFETD_CANDIDATES = (
    DATA / "building_stock" / "hk_bfetd" / "HK_UBEM_Buildings_Public_v1_1.csv",
)

POSITIVE_USE = re.compile(
    r"\b(?:apartment|residential|domestic|dwelling|house|housing|residence|"
    r"home ownership|hostel|quarters|village house)\b",
    re.IGNORECASE,
)
NEGATIVE_USE = re.compile(
    r"\b(?:car park|substation|pump house|school|office|commercial|industrial|"
    r"factory|hospital|clinic|market|toilet|warehouse|temple|church|hotel|"
    r"restaurant|retail|utility|plant room)\b",
    re.IGNORECASE,
)


def read_table(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    return pyogrio.read_dataframe(path, columns=columns, read_geometry=False)


def to_year(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip().str.slice(0, 4)
    year = pd.to_numeric(text, errors="coerce")
    return year.where(year.between(1840, CURRENT_YEAR))


def join_unique(values: pd.Series) -> str:
    cleaned = sorted({str(v).strip() for v in values.dropna() if str(v).strip()})
    return " | ".join(cleaned)


def first_existing(columns: pd.Index, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in columns:
            return name
    return None


def load_buildings() -> gpd.GeoDataFrame:
    gdf = pyogrio.read_dataframe(FILES["buildings"])
    gdf = gdf.rename(columns={"SHAPE_Area": "footprint_area_m2", "SHAPE_Length": "perimeter_m"})
    gdf["BUILDINGSTRUCTUREID"] = pd.to_numeric(gdf["BUILDINGSTRUCTUREID"], errors="raise").astype("int64")
    gdf["footprint_area_m2"] = pd.to_numeric(gdf["footprint_area_m2"], errors="coerce")
    gdf["perimeter_m"] = pd.to_numeric(gdf["perimeter_m"], errors="coerce")
    return gdf


def attach_dtm_evidence(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if FILES["dtm"].exists():
        dtm = pd.read_csv(
            FILES["dtm"],
            usecols=["BUILDINGSTRUCTUREID", "dtm_elevation_mpd"],
        )
        dtm["BUILDINGSTRUCTUREID"] = pd.to_numeric(
            dtm["BUILDINGSTRUCTUREID"], errors="raise"
        ).astype("int64")
        dtm = dtm.drop_duplicates("BUILDINGSTRUCTUREID", keep="last")
        gdf = gdf.merge(dtm, on="BUILDINGSTRUCTUREID", how="left", validate="one_to_one")
    else:
        gdf["dtm_elevation_mpd"] = np.nan

    top = pd.to_numeric(gdf["TOPHEIGHT"], errors="coerce")
    dtm_value = pd.to_numeric(gdf["dtm_elevation_mpd"], errors="coerce")
    net = top - dtm_value
    valid = top.notna() & dtm_value.notna() & net.between(2, 500)
    gdf["net_height_evidence_m"] = net.where(valid)
    gdf["dtm_height_evidence_status"] = np.select(
        [top.isna(), dtm_value.isna(), ~net.between(2, 500)],
        ["roof_elevation_missing", "dtm_sample_missing", "roof_minus_dtm_out_of_range"],
        default="available_point_sample",
    )
    return gdf


def attach_names_and_info(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    names = read_table(FILES["names"])
    if not names.empty:
        names = names.sort_values(["BUILDINGSTRUCTUREID", "NAMESTATUS"], na_position="last")
        names = names.drop_duplicates("BUILDINGSTRUCTUREID")
        cols = ["BUILDINGSTRUCTUREID", "BUILDINGNAMEEN", "BUILDINGNAMETC"]
        gdf = gdf.merge(names[cols], on="BUILDINGSTRUCTUREID", how="left")

    info = read_table(FILES["info"])
    info["INFODESCRIPTION"] = info["INFODESCRIPTION"].astype("string")
    info_agg = info.groupby("BUILDINGSTRUCTUREID", as_index=False).agg(
        intended_use_text=("INFODESCRIPTION", join_unique),
        info_residential=("INFODESCRIPTION", lambda x: bool(x.fillna("").str.contains(POSITIVE_USE).any())),
        info_nonresidential=("INFODESCRIPTION", lambda x: bool(x.fillna("").str.contains(NEGATIVE_USE).any())),
    )
    return gdf.merge(info_agg, on="BUILDINGSTRUCTUREID", how="left")


def load_permit_evidence() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    permits = read_table(FILES["permits"])
    links = read_table(FILES["permit_links"])
    permits["permit_year"] = to_year(permits["OPDATE"])
    for col in ("DOMESTICGFA", "NONDOMESTICGFA"):
        permits[col] = pd.to_numeric(permits[col], errors="coerce").where(lambda x: x > 0)
    linked = links.merge(permits, on="OPNO", how="left", validate="many_to_one")
    total = linked["DOMESTICGFA"].fillna(0) + linked["NONDOMESTICGFA"].fillna(0)
    linked["permit_residential_share_signal"] = linked["DOMESTICGFA"].fillna(0).div(total.where(total > 0))
    building = linked.groupby("BUILDINGSTRUCTUREID", as_index=False).agg(
        occupation_permit_year=("permit_year", "min"),
        permit_residential_share_signal=("permit_residential_share_signal", "max"),
        linked_permit_count=("OPNO", "nunique"),
    )
    return permits, links, building


def load_work_years() -> pd.DataFrame:
    works = read_table(FILES["works"])
    works["works_year"] = to_year(works["WORKSDATE"])
    works = works[works["WORKSTYPE"].isin(["CD", "CC"])]
    pivot = works.pivot_table(
        index="BUILDINGSTRUCTUREID", columns="WORKSTYPE", values="works_year", aggfunc="min"
    ).reset_index()
    return pivot.rename(columns={"CD": "completion_date_year", "CC": "compliance_certificate_year"})


def load_zones(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    frames: list[gpd.GeoDataFrame] = []
    for path in sorted(FILES["zones"].glob("*.json")):
        frame = pyogrio.read_dataframe(path, columns=["ZONE_LABEL"])
        if frame.empty or "ZONE_LABEL" not in frame.columns:
            continue
        frame["ozp_plan"] = path.stem
        frame["ozp_polygon_area_m2"] = frame.geometry.area
        frames.append(frame)
    if not frames:
        return pd.DataFrame({"BUILDINGSTRUCTUREID": gdf["BUILDINGSTRUCTUREID"]})

    zones = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs).to_crs(gdf.crs)
    points = gdf[["BUILDINGSTRUCTUREID", "geometry"]].copy()
    points.geometry = points.geometry.representative_point()
    joined = gpd.sjoin(points, zones, how="left", predicate="within")
    joined = joined.sort_values(["BUILDINGSTRUCTUREID", "ozp_polygon_area_m2"], na_position="last")
    joined = joined.drop_duplicates("BUILDINGSTRUCTUREID")
    label = joined["ZONE_LABEL"].astype("string").fillna("").str.upper().str.strip()
    joined["ozp_context"] = np.select(
        [label.str.match(r"^R($|\(|/)|RESIDENT"), label.str.match(r"^V($|\()"), label.str.match(r"^(C|I|G/IC)($|\(|/)")],
        ["residential", "village", "non_residential"],
        default="other_or_missing",
    )
    return joined[["BUILDINGSTRUCTUREID", "ZONE_LABEL", "ozp_plan", "ozp_context"]]


def match_bdbiar(gdf: gpd.GeoDataFrame, max_distance: float = 30.0) -> pd.DataFrame:
    bd = pd.read_csv(FILES["bdbiar"], encoding="utf-8-sig", encoding_errors="replace", low_memory=False)
    bd["bdbiar_year"] = to_year(bd["NSEARCH3_E"])
    bd["bdbiar_usage"] = bd["NSEARCH5_E"].astype("string")
    bd["bdbiar_part"] = bd["NSEARCH4_E"].astype("string")
    points = gpd.GeoDataFrame(
        bd,
        geometry=gpd.points_from_xy(bd["LONGITUDE"], bd["LATITUDE"]),
        crs="EPSG:4326",
    ).to_crs(gdf.crs)
    candidates = gdf[["BUILDINGSTRUCTUREID", "BUILDINGSTRUCTURETYPE", "geometry"]]

    part = points["bdbiar_part"].str.lower()
    specs = ((part.str.contains("tower", na=False), "T"), (part.str.contains("podium", na=False), "P"))
    matched_parts: list[pd.DataFrame] = []
    assigned = pd.Series(False, index=points.index)
    for mask, structure_type in specs:
        subset = points[mask]
        assigned |= mask
        if subset.empty:
            continue
        target = candidates[candidates["BUILDINGSTRUCTURETYPE"] == structure_type]
        matched_parts.append(
            gpd.sjoin_nearest(subset, target, how="left", max_distance=max_distance, distance_col="bdbiar_distance_m")
        )
    other = points[~assigned]
    if not other.empty:
        matched_parts.append(
            gpd.sjoin_nearest(other, candidates, how="left", max_distance=max_distance, distance_col="bdbiar_distance_m")
        )
    matched = pd.concat(matched_parts, ignore_index=True)
    matched = matched.dropna(subset=["BUILDINGSTRUCTUREID"])
    matched["BUILDINGSTRUCTUREID"] = matched["BUILDINGSTRUCTUREID"].astype("int64")
    matched = matched.sort_values(["OBJECTID", "bdbiar_distance_m", "BUILDINGSTRUCTUREID"]).drop_duplicates("OBJECTID")
    usage = matched["bdbiar_usage"].str.lower()
    matched["bdbiar_residential"] = usage.str.contains("residential", na=False)
    matched["bdbiar_composite"] = usage.str.contains("composite", na=False)
    matched["bdbiar_nonresidential"] = ~matched["bdbiar_residential"]
    matched["bdbiar_residential_value"] = np.select(
        [matched["bdbiar_composite"], matched["bdbiar_residential"]], [0.70, 0.95], default=0.05
    )
    return matched.groupby("BUILDINGSTRUCTUREID", as_index=False).agg(
        bdbiar_record_ids=("OBJECTID", lambda x: " | ".join(map(str, sorted(set(x.astype(int)))))) ,
        bdbiar_usage=("bdbiar_usage", join_unique),
        bdbiar_year=("bdbiar_year", "median"),
        bdbiar_distance_m=("bdbiar_distance_m", "min"),
        bdbiar_residential=("bdbiar_residential", "max"),
        bdbiar_nonresidential=("bdbiar_nonresidential", "max"),
        bdbiar_residential_value=("bdbiar_residential_value", "max"),
    )


def parse_residential_share(value: object) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        try:
            parsed = json.loads(str(value))
        except (TypeError, json.JSONDecodeError):
            pairs = re.findall(r"([^,;:]+):\s*([0-9]*\.?[0-9]+)", str(value))
            if not pairs:
                return np.nan
            return float(sum(float(number) for label, number in pairs if re.search(r"residential|housing|房屋|住宅", label, re.I)))
    if not isinstance(parsed, dict):
        return np.nan
    return float(sum(float(v) for k, v in parsed.items() if re.search(r"residential|housing|房屋|住宅", str(k), re.I)))


def load_hkbfetd(path: Path | None) -> tuple[pd.DataFrame | None, str]:
    if path is None:
        path = next((candidate for candidate in HKBFETD_CANDIDATES if candidate.exists()), None)
    if path is None or not path.exists():
        return None, "hkbfetd_release_unavailable"

    release = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    id_col = first_existing(release.columns, ("Building_ID_建筑ID", "Building_ID_建筑物ID", "Building_ID", "BUILDINGSTRUCTUREID"))
    class_col = first_existing(release.columns, ("UBEM_Main_Class_主类别", "UBEM_Main_Class", "Final_Main_Class"))
    subclass_col = first_existing(release.columns, ("UBEM_Sub_Class_子类别", "UBEM_Sub_Class", "Final_Sub_Class"))
    probability_col = first_existing(release.columns, ("ML_Prob_Residential_住宅概率", "proba_住宅类别"))
    confidence_col = first_existing(release.columns, ("ML_Prediction_Confidence_机器学习置信度", "ML_Confidence"))
    source_col = first_existing(release.columns, ("Classification_Source_分类来源", "Classification_Source"))
    proportions_col = first_existing(release.columns, ("UBEM_Mixed_Proportions_混合比例", "UBEM_Mixed_Proportions_混合用途比例", "Final_Sub_Proportions"))
    legacy_floors_col = first_existing(release.columns, ("Floors_Count_地上层数", "Estimated_Floors"))
    legacy_height_col = first_existing(release.columns, ("Height_m_建筑高度", "Estimated_Height"))
    if id_col is None or class_col is None:
        raise ValueError("HK-BFETD release is missing its building ID or main-class field")

    result = pd.DataFrame({"BUILDINGSTRUCTUREID": pd.to_numeric(release[id_col], errors="coerce")})
    result = result.dropna(subset=["BUILDINGSTRUCTUREID"])
    result["BUILDINGSTRUCTUREID"] = result["BUILDINGSTRUCTUREID"].astype("int64")
    classes = release.loc[result.index, class_col].astype("string")
    result["hkbfetd_main_class"] = classes.to_numpy()
    result["hkbfetd_sub_class"] = (
        release.loc[result.index, subclass_col].astype("string").to_numpy()
        if subclass_col else pd.NA
    )
    if probability_col:
        result["hkbfetd_residential_value"] = pd.to_numeric(release.loc[result.index, probability_col], errors="coerce").to_numpy()
    else:
        result["hkbfetd_residential_value"] = np.select(
            [classes.str.contains("Residential|住宅", case=False, na=False), classes.str.contains("Mixed|混合", case=False, na=False)],
            [1.0, 0.5],
            default=0.0,
        )
    if proportions_col:
        parsed = release.loc[result.index, proportions_col].apply(parse_residential_share)
        result["hkbfetd_residential_value"] = parsed.fillna(result["hkbfetd_residential_value"]).to_numpy()
    result["hkbfetd_confidence"] = (
        pd.to_numeric(release.loc[result.index, confidence_col], errors="coerce").to_numpy() if confidence_col else np.nan
    )
    result["hkbfetd_source"] = release.loc[result.index, source_col].astype("string").to_numpy() if source_col else "unknown"
    if legacy_floors_col:
        result["hkbfetd_legacy_floors"] = pd.to_numeric(release.loc[result.index, legacy_floors_col], errors="coerce").to_numpy()
    if legacy_height_col:
        result["hkbfetd_legacy_height_m"] = pd.to_numeric(release.loc[result.index, legacy_height_col], errors="coerce").to_numpy()
    result = result.drop_duplicates("BUILDINGSTRUCTUREID")
    return result, f"fused:{path}"


def add_weighted_evidence(score_sum: np.ndarray, weight_sum: np.ndarray, mask: pd.Series, value: object, weight: object) -> None:
    valid = mask.fillna(False).to_numpy(dtype=bool)
    values = np.broadcast_to(np.asarray(value, dtype=float), score_sum.shape) if np.ndim(value) == 0 else np.asarray(value, dtype=float)
    weights = np.broadcast_to(np.asarray(weight, dtype=float), score_sum.shape) if np.ndim(weight) == 0 else np.asarray(weight, dtype=float)
    score_sum[valid] += values[valid] * weights[valid]
    weight_sum[valid] += weights[valid]


def qualify_residential(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    n = len(gdf)
    score_sum = np.zeros(n, dtype=float)
    weight_sum = np.zeros(n, dtype=float)
    category = gdf["CATEGORY"].astype("string")
    structure = gdf["BUILDINGSTRUCTURETYPE"].astype("string")

    category_value = category.map({"1": 0.50, "2": 0.95, "3": 0.90, "4": 0.05, "5": 0.05, "9": 0.25}).fillna(0.25)
    category_weight = category.map({"1": 0.5, "2": 3.0, "3": 3.0, "4": 3.0, "5": 1.5, "9": 0.5}).fillna(0.5)
    category_value = category_value.mask((category == "3") & (structure == "P"), 0.50)
    add_weighted_evidence(score_sum, weight_sum, pd.Series(True, index=gdf.index), category_value, category_weight)

    permit_share = pd.to_numeric(gdf["permit_residential_share_signal"], errors="coerce")
    add_weighted_evidence(score_sum, weight_sum, permit_share.notna(), permit_share.fillna(0), 6.0)
    bd_value = pd.to_numeric(gdf["bdbiar_residential_value"], errors="coerce")
    bd_weight = np.clip(6.0 - gdf["bdbiar_distance_m"].fillna(30).to_numpy() / 10.0, 3.0, 6.0)
    add_weighted_evidence(score_sum, weight_sum, bd_value.notna(), bd_value.fillna(0), bd_weight)

    if "hkbfetd_residential_value" in gdf:
        hk_value = pd.to_numeric(gdf["hkbfetd_residential_value"], errors="coerce").clip(0, 1)
        hk_conf = pd.to_numeric(gdf["hkbfetd_confidence"], errors="coerce").fillna(0.8).clip(0.25, 1.0)
        hk_weight = 6.0 * hk_conf.to_numpy()
        add_weighted_evidence(score_sum, weight_sum, hk_value.notna(), hk_value.fillna(0), hk_weight)

    info_res = gdf["info_residential"].astype("boolean").fillna(False).astype(bool)
    info_nonres = gdf["info_nonresidential"].astype("boolean").fillna(False).astype(bool)
    add_weighted_evidence(score_sum, weight_sum, info_res & ~info_nonres, 0.95, 4.0)
    add_weighted_evidence(score_sum, weight_sum, info_nonres & ~info_res, 0.05, 4.0)
    add_weighted_evidence(score_sum, weight_sum, info_res & info_nonres, 0.50, 4.0)

    names = (
        gdf["OFFICIALBUILDINGNAMEEN"].fillna("").astype(str)
        + " " + gdf["BUILDINGNAMEEN"].fillna("").astype(str)
    )
    name_res = names.str.contains(POSITIVE_USE)
    name_nonres = names.str.contains(NEGATIVE_USE)
    add_weighted_evidence(score_sum, weight_sum, name_res & ~name_nonres, 0.85, 1.5)
    add_weighted_evidence(score_sum, weight_sum, name_nonres & ~name_res, 0.05, 1.5)

    ozp = gdf["ozp_context"].fillna("other_or_missing")
    add_weighted_evidence(score_sum, weight_sum, ozp.eq("residential"), 0.75, 1.0)
    add_weighted_evidence(score_sum, weight_sum, ozp.eq("village"), 0.65, 1.0)
    add_weighted_evidence(score_sum, weight_sum, ozp.eq("non_residential"), 0.15, 1.0)

    score = np.divide(score_sum, weight_sum, out=np.full(n, np.nan), where=weight_sum > 0)
    gdf["residential_evidence_score"] = np.clip(score, 0, 1)
    gdf["residential_score_is_calibrated"] = False

    bd_near = pd.to_numeric(gdf["bdbiar_distance_m"], errors="coerce").le(15)
    permit_positive = permit_share.ge(0.5)
    permit_mixed = permit_share.gt(0.10) & permit_share.lt(0.50)
    bdbiar_positive = gdf["bdbiar_residential"].fillna(False) & bd_near
    info_positive = info_res & ~info_nonres
    category_positive = category.eq("2") | (category.eq("3") & ~structure.eq("P"))
    direct_positive = permit_positive | bdbiar_positive | info_positive | category_positive

    hk_value = pd.to_numeric(
        gdf.get("hkbfetd_residential_value", pd.Series(np.nan, index=gdf.index)),
        errors="coerce",
    )
    hk_source = gdf.get("hkbfetd_source", pd.Series("unavailable", index=gdf.index)).astype("string")
    rule_positive = hk_source.str.startswith("Rule_Engine", na=False) & hk_value.ge(0.75)
    ml_only_positive = (
        hk_source.str.contains("Machine_Learning|ML_Low", case=False, na=False)
        & hk_value.ge(0.75)
    )
    direct_negative = (
        (permit_share.notna() & permit_share.le(0.10))
        | (gdf["bdbiar_nonresidential"].fillna(False) & bd_near & ~gdf["bdbiar_residential"].fillna(False))
        | (info_nonres & ~info_res)
        | category.eq("4")
    )
    conflict = direct_positive & direct_negative
    ambiguous_category = category.isin(["5", "9"])
    active = gdf["STATUS"].astype("string").eq("A")

    gdf["res_evidence_permit_direct"] = permit_positive
    gdf["res_evidence_permit_mixed"] = permit_mixed
    gdf["res_evidence_bdbiar_spatial"] = bdbiar_positive
    gdf["res_evidence_info_direct"] = info_positive
    gdf["res_evidence_category_direct"] = category_positive
    gdf["res_evidence_hkbfetd_rule"] = rule_positive
    gdf["res_evidence_hkbfetd_ml_only"] = ml_only_positive
    gdf["res_evidence_negative_direct"] = direct_negative
    gdf["res_evidence_conflict"] = conflict
    gdf["residential_direct_evidence_count"] = pd.concat(
        [permit_positive, bdbiar_positive, info_positive, category_positive], axis=1
    ).sum(axis=1).astype("int8")

    status = np.full(n, "review_insufficient_direct_evidence", dtype=object)
    status[~active.to_numpy()] = "exclude_inactive"
    status[(active & direct_negative).to_numpy()] = "exclude_nonresidential_direct"
    status[(active & rule_positive & ~direct_negative).to_numpy()] = "include_residential_rule_supported"
    status[(active & direct_positive & ~direct_negative).to_numpy()] = "include_residential_direct"
    status[(active & permit_mixed & ~direct_negative).to_numpy()] = "mixed_use_direct_permit"
    status[(active & ml_only_positive & ~direct_positive).to_numpy()] = "review_ml_only"
    status[(active & ambiguous_category).to_numpy()] = "review_category_5_9"
    status[conflict.to_numpy()] = "review_conflicting_evidence"
    gdf["residential_eligibility_status"] = status
    gdf["thermal_stock_eligible"] = pd.Series(status, index=gdf.index).isin(
        ["include_residential_direct", "include_residential_rule_supported", "mixed_use_direct_permit"]
    )
    gdf["manual_review_priority"] = np.select(
        [conflict, ambiguous_category & (direct_positive | rule_positive | ml_only_positive), ml_only_positive, pd.Series(status, index=gdf.index).str.startswith("review")],
        ["high_conflict", "high_category_5_9_positive", "high_ml_only", "standard_review"],
        default="not_required_by_rules",
    )
    gdf["ac_presence_status"] = "unknown_no_building_level_observation"
    return gdf


def resolve_year_evidence(gdf: pd.DataFrame) -> pd.DataFrame:
    sources = [
        ("occupation_permit_year", "occupation_permit"),
        ("completion_date_year", "landsd_completion_date"),
        ("compliance_certificate_year", "landsd_compliance_certificate"),
        ("bdbiar_year", "bdbiar_completion"),
    ]
    evidence = pd.concat(
        [pd.to_numeric(gdf[column], errors="coerce").rename(label) for column, label in sources], axis=1
    )
    valid = evidence.where(evidence.apply(lambda x: x.between(1840, CURRENT_YEAR)))
    lower = valid.min(axis=1)
    upper = valid.max(axis=1)
    selected = pd.Series(np.nan, index=gdf.index, dtype=float)
    selected_source = pd.Series("missing", index=gdf.index, dtype="string")
    for column, label in sources:
        values = pd.to_numeric(gdf[column], errors="coerce").where(lambda x: x.between(1840, CURRENT_YEAR))
        use = selected.isna() & values.notna()
        selected.loc[use] = values.loc[use]
        selected_source.loc[use] = label
    return pd.DataFrame(
        {
            "construction_year_observed": selected,
            "construction_year_observed_source": selected_source,
            "construction_year_evidence_lower": lower,
            "construction_year_evidence_upper": upper,
            "construction_year_conflict": lower.notna() & upper.notna() & upper.sub(lower).gt(5),
        },
        index=gdf.index,
    )


def feature_matrix(
    gdf: gpd.GeoDataFrame, include_floors: bool, include_observed_year: bool = True
) -> tuple[pd.DataFrame, pd.Series]:
    point = gdf.geometry.representative_point()
    area = gdf["footprint_area_m2"].clip(lower=1)
    perimeter = gdf["perimeter_m"].clip(lower=1)
    features = pd.DataFrame(
        {
            "log_footprint_area": np.log1p(area),
            "compactness": np.clip(4 * np.pi * area / perimeter.pow(2), 0, 1),
            "centroid_x_km": point.x / 1000.0,
            "centroid_y_km": point.y / 1000.0,
            "residential_evidence_score": gdf["residential_evidence_score"].fillna(0.5),
        },
        index=gdf.index,
    )
    net_height = pd.to_numeric(
        gdf.get("net_height_evidence_m", pd.Series(np.nan, index=gdf.index)),
        errors="coerce",
    )
    features["log_roof_minus_dtm_height"] = np.log1p(net_height.clip(lower=0)).fillna(0)
    features["roof_minus_dtm_height_missing"] = net_height.isna().astype(int)
    if include_observed_year:
        features["observed_year"] = gdf.get(
            "construction_year_observed", pd.Series(np.nan, index=gdf.index)
        )
        features["observed_year_missing"] = features["observed_year"].isna().astype(int)
        features["observed_year"] = features["observed_year"].fillna(features["observed_year"].median())
    if include_floors:
        features["floors_value"] = gdf["floors_value"]
    categories = pd.DataFrame(
        {
            "category": gdf["CATEGORY"].fillna("missing").astype(str),
            "structure_type": gdf["BUILDINGSTRUCTURETYPE"].fillna("missing").astype(str),
            "ozp_context": gdf["ozp_context"].fillna("missing").astype(str),
        },
        index=gdf.index,
    )
    features = pd.concat([features, pd.get_dummies(categories, dtype=float)], axis=1).astype(float)
    groups = (point.x // 2000).astype(int).astype(str) + "_" + (point.y // 2000).astype(int).astype(str)
    return features, groups


def fit_quantile_models(
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    transform: str,
    random_state: int,
) -> tuple[dict[str, HistGradientBoostingRegressor], dict[str, float], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    valid = target.notna()
    x = features.loc[valid]
    y = target.loc[valid].astype(float)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=random_state)
    train_pos, test_pos = next(splitter.split(x, y, groups.loc[valid]))
    x_train, x_test = x.iloc[train_pos], x.iloc[test_pos]
    y_train, y_test = y.iloc[train_pos], y.iloc[test_pos]

    def forward(values: pd.Series | np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        return np.log1p(arr) if transform == "log1p" else arr

    def inverse(values: np.ndarray) -> np.ndarray:
        return np.expm1(values) if transform == "log1p" else values

    validation_model = HistGradientBoostingRegressor(
        loss="squared_error", max_iter=220, max_leaf_nodes=31, min_samples_leaf=30,
        l2_regularization=1.0, random_state=random_state,
    ).fit(x_train, forward(y_train))
    validation_prediction = inverse(validation_model.predict(x_test))
    metrics = {
        "n_observed": int(valid.sum()),
        "n_validation": int(len(y_test)),
        "mae": float(mean_absolute_error(y_test, validation_prediction)),
        "rmse": float(mean_squared_error(y_test, validation_prediction) ** 0.5),
        "r2": float(r2_score(y_test, validation_prediction)),
    }

    models = {
        "median": HistGradientBoostingRegressor(
            loss="squared_error", max_iter=260, max_leaf_nodes=31, min_samples_leaf=30,
            l2_regularization=1.0, random_state=random_state,
        ),
        "lower": HistGradientBoostingRegressor(
            loss="quantile", quantile=0.10, max_iter=260, max_leaf_nodes=31,
            min_samples_leaf=30, l2_regularization=1.0, random_state=random_state,
        ),
        "upper": HistGradientBoostingRegressor(
            loss="quantile", quantile=0.90, max_iter=260, max_leaf_nodes=31,
            min_samples_leaf=30, l2_regularization=1.0, random_state=random_state,
        ),
    }
    for model in models.values():
        model.fit(x, forward(y))
    predictions = tuple(inverse(models[name].predict(features)) for name in ("median", "lower", "upper"))
    return models, metrics, predictions


def repair_floors(gdf: gpd.GeoDataFrame, random_state: int) -> tuple[gpd.GeoDataFrame, dict, dict]:
    raw = pd.to_numeric(gdf["NUMABOVEGROUNDSTOREYS"], errors="coerce")
    official = raw.where(raw.between(1, 120))
    features, groups = feature_matrix(gdf, include_floors=False)
    models, metrics, (median, lower, upper) = fit_quantile_models(features, official, groups, "log1p", random_state)
    median = np.clip(np.rint(median), 1, 120)
    lower = np.clip(np.floor(lower), 1, 120)
    upper = np.clip(np.ceil(upper), 1, 120)
    lower = np.minimum(lower, median)
    upper = np.maximum(upper, median)
    observed = official.notna().to_numpy()
    gdf["floors_value"] = np.where(observed, official, median).astype(float)
    gdf["floors_lower"] = np.where(observed, official, lower).astype(float)
    gdf["floors_upper"] = np.where(observed, official, upper).astype(float)
    height_informed = gdf["net_height_evidence_m"].notna().to_numpy() & ~observed
    gdf["floors_source"] = np.select(
        [observed, height_informed],
        ["landsd_official_storeys", "roof_dtm_height_informed_spatial_model"],
        default="spatial_quantile_model",
    )
    width = gdf["floors_upper"] - gdf["floors_lower"]
    gdf["floors_quality"] = np.select(
        [observed, height_informed & width.le(5), height_informed, width.le(5), width.le(15)],
        ["A_official", "B_height_model_narrow", "C_height_model_wide", "C_spatial_model_narrow", "D_spatial_model_moderate"],
        default="E_spatial_model_wide",
    )
    return gdf, models, metrics


def repair_years(gdf: gpd.GeoDataFrame, random_state: int) -> tuple[gpd.GeoDataFrame, dict, dict]:
    resolved = resolve_year_evidence(gdf)
    for column in resolved:
        gdf[column] = resolved[column]
    features, groups = feature_matrix(gdf, include_floors=True, include_observed_year=False)
    target = gdf["construction_year_observed"]
    models, metrics, (median, lower, upper) = fit_quantile_models(features, target, groups, "identity", random_state)
    median = np.clip(np.rint(median), 1840, CURRENT_YEAR)
    lower = np.clip(np.floor(lower), 1840, CURRENT_YEAR)
    upper = np.clip(np.ceil(upper), 1840, CURRENT_YEAR)
    lower = np.minimum(lower, median)
    upper = np.maximum(upper, median)
    observed = target.notna().to_numpy()
    conflict = gdf["construction_year_conflict"].to_numpy()
    gdf["construction_year_value"] = np.where(observed, target, median).astype(float)
    gdf["construction_year_lower"] = np.where(observed, gdf["construction_year_evidence_lower"], lower).astype(float)
    gdf["construction_year_upper"] = np.where(observed, gdf["construction_year_evidence_upper"], upper).astype(float)
    gdf["construction_year_source"] = np.where(observed, gdf["construction_year_observed_source"], "spatial_quantile_model")
    source = gdf["construction_year_source"].astype("string")
    gdf["construction_year_quality"] = np.select(
        [
            conflict,
            source.eq("occupation_permit"),
            source.isin(["landsd_completion_date", "landsd_compliance_certificate"]),
            source.eq("bdbiar_completion"),
            (upper - lower) <= 15,
        ],
        ["C_evidence_conflict", "A_direct_permit", "A_direct_landsd", "B_bdbiar_spatial_match", "C_model_era"],
        default="D_model_wide",
    )
    gdf["building_age_2026"] = CURRENT_YEAR - gdf["construction_year_value"]
    return gdf, models, metrics


def repair_height(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    top = pd.to_numeric(gdf["TOPHEIGHT"], errors="coerce")
    base = pd.to_numeric(gdf["BASEHEIGHT"], errors="coerce")
    net = top - base
    valid_base_net = top.notna() & base.notna() & net.between(2, 500)
    dtm_net = pd.to_numeric(
        gdf.get("net_height_evidence_m", pd.Series(np.nan, index=gdf.index)),
        errors="coerce",
    )
    valid_dtm_net = dtm_net.notna()
    residential = gdf["residential_evidence_score"].ge(0.5)
    podium = gdf["BUILDINGSTRUCTURETYPE"].astype("string").eq("P")
    floor_height = np.select([podium, residential], [3.6, 3.0], default=3.3)
    floor_height_lower = np.select([podium, residential], [2.8, 2.6], default=2.7)
    floor_height_upper = np.select([podium, residential], [5.0, 3.4], default=4.5)
    estimated = gdf["floors_value"].to_numpy() * floor_height
    lower = gdf["floors_lower"].to_numpy() * floor_height_lower
    upper = gdf["floors_upper"].to_numpy() * floor_height_upper
    gdf["roof_elevation_mpd"] = top
    measured_or_sampled = np.where(valid_base_net, net, dtm_net)
    usable_net = valid_base_net | valid_dtm_net
    gdf["height_value_m"] = np.where(usable_net, measured_or_sampled, estimated)
    gdf["height_lower_m"] = np.where(usable_net, measured_or_sampled, lower)
    gdf["height_upper_m"] = np.where(usable_net, measured_or_sampled, upper)
    gdf["height_source"] = np.select(
        [valid_base_net, valid_dtm_net],
        ["top_minus_base_official", "landsd_roof_minus_open3dhk_dtm_point"],
        default="floors_x_typology_floor_height",
    )
    gdf["height_quality"] = np.select(
        [valid_base_net, valid_dtm_net],
        ["A_official_net", "B_official_roof_dtm_point_sample"],
        default="C_typology_estimate",
    )
    gdf["topheight_is_net_height"] = False
    gdf["terrain_elevation_status"] = np.select(
        [
            base.notna(),
            gdf.get("dtm_elevation_mpd", pd.Series(np.nan, index=gdf.index)).notna(),
        ],
        ["landsd_baseheight_available", "open3dhk_dtm_point_sample"],
        default="unavailable",
    )
    official_floors = pd.to_numeric(
        gdf.get("NUMABOVEGROUNDSTOREYS", pd.Series(np.nan, index=gdf.index)),
        errors="coerce",
    ).where(lambda x: x.between(1, 120))
    floor_to_floor = pd.to_numeric(gdf["height_value_m"], errors="coerce") / official_floors
    gdf["height_floor_consistency_status"] = np.select(
        [official_floors.isna(), floor_to_floor.between(2, 6)],
        ["not_testable_official_floors_missing", "consistent_2_to_6_m_per_floor"],
        default="conflict_outside_2_to_6_m_per_floor",
    )
    return gdf


def allocate_permit_gfa(
    permits: pd.DataFrame, links: pd.DataFrame, buildings: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, float]]:
    linked = links.merge(
        permits[["OPNO", "DOMESTICGFA", "NONDOMESTICGFA"]], on="OPNO", how="left", validate="many_to_one"
    ).merge(
        buildings[["BUILDINGSTRUCTUREID", "gfa_capacity_proxy_m2", "residential_evidence_score"]],
        on="BUILDINGSTRUCTUREID", how="left", validate="many_to_one",
    )
    linked["gfa_capacity_proxy_m2"] = linked["gfa_capacity_proxy_m2"].fillna(1).clip(lower=1)
    residential = linked["residential_evidence_score"].fillna(0.5).clip(0.05, 0.95)
    linked["domestic_weight"] = linked["gfa_capacity_proxy_m2"] * residential
    linked["nondomestic_weight"] = linked["gfa_capacity_proxy_m2"] * (1 - residential)
    linked["permit_link_count"] = linked.groupby("OPNO")["BUILDINGSTRUCTUREID"].transform("nunique")
    for kind, source, weight in (
        ("domestic", "DOMESTICGFA", "domestic_weight"),
        ("nondomestic", "NONDOMESTICGFA", "nondomestic_weight"),
    ):
        denominator = linked.groupby("OPNO")[weight].transform("sum")
        linked[f"allocated_{kind}_gfa_m2"] = linked[source] * linked[weight] / denominator
        linked[f"equal_share_{kind}_gfa_m2"] = linked[source] / linked["permit_link_count"]
        linked[f"allocated_{kind}_gfa_lower_m2"] = linked[
            [f"allocated_{kind}_gfa_m2", f"equal_share_{kind}_gfa_m2"]
        ].min(axis=1)
        linked[f"allocated_{kind}_gfa_upper_m2"] = linked[
            [f"allocated_{kind}_gfa_m2", f"equal_share_{kind}_gfa_m2"]
        ].max(axis=1)

    checks = linked.groupby("OPNO", as_index=False).agg(
        permit_domestic_gfa_m2=("DOMESTICGFA", "first"),
        allocated_domestic_gfa_m2=("allocated_domestic_gfa_m2", "sum"),
        permit_nondomestic_gfa_m2=("NONDOMESTICGFA", "first"),
        allocated_nondomestic_gfa_m2=("allocated_nondomestic_gfa_m2", "sum"),
    )
    dom_error = (checks["permit_domestic_gfa_m2"] - checks["allocated_domestic_gfa_m2"]).abs().dropna()
    non_error = (checks["permit_nondomestic_gfa_m2"] - checks["allocated_nondomestic_gfa_m2"]).abs().dropna()
    metrics = {
        "permits_with_domestic_gfa": int(checks["permit_domestic_gfa_m2"].notna().sum()),
        "permits_with_nondomestic_gfa": int(checks["permit_nondomestic_gfa_m2"].notna().sum()),
        "max_domestic_conservation_error_m2": float(dom_error.max()) if len(dom_error) else 0.0,
        "max_nondomestic_conservation_error_m2": float(non_error.max()) if len(non_error) else 0.0,
    }
    return linked, metrics


def repair_gfa(
    gdf: gpd.GeoDataFrame, permits: pd.DataFrame, links: pd.DataFrame
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, dict[str, float]]:
    gdf["gfa_capacity_proxy_m2"] = gdf["footprint_area_m2"] * gdf["floors_value"]
    allocations, metrics = allocate_permit_gfa(permits, links, gdf)
    allocation_fields = [
        "allocated_domestic_gfa_m2", "allocated_nondomestic_gfa_m2",
        "allocated_domestic_gfa_lower_m2", "allocated_domestic_gfa_upper_m2",
        "allocated_nondomestic_gfa_lower_m2", "allocated_nondomestic_gfa_upper_m2",
    ]
    building_alloc = allocations.groupby("BUILDINGSTRUCTUREID", as_index=False)[allocation_fields].sum(min_count=1)
    building_alloc["permit_gfa_allocation_count"] = allocations.groupby("BUILDINGSTRUCTUREID")["OPNO"].nunique().to_numpy()
    gdf = gdf.merge(building_alloc, on="BUILDINGSTRUCTUREID", how="left")

    raw = pd.to_numeric(gdf["GROSSFLOORAREA"], errors="coerce").where(lambda x: x > 0)
    allocated_total = gdf["allocated_domestic_gfa_m2"].fillna(0) + gdf["allocated_nondomestic_gfa_m2"].fillna(0)
    allocated_total = allocated_total.where(allocated_total > 0)
    capacity = gdf["gfa_capacity_proxy_m2"]
    gdf["gfa_observed_m2"] = raw
    gdf["gfa_allocated_m2"] = allocated_total
    gdf["gfa_value_m2"] = raw.fillna(allocated_total).fillna(capacity)
    gdf["gfa_source"] = np.select(
        [raw.notna(), allocated_total.notna()],
        ["landsd_registered_gfa", "occupation_permit_constrained_allocation"],
        default="footprint_x_repaired_floors_capacity_proxy",
    )
    gdf["gfa_measurement_class"] = np.select(
        [raw.notna(), allocated_total.notna()],
        ["observed_building_record", "permit_constrained_allocation"],
        default="capacity_proxy",
    )
    gdf["gfa_capacity_ratio"] = gdf["gfa_value_m2"].div(capacity.where(capacity.gt(0)))
    measured_or_allocated = gdf["gfa_measurement_class"].isin(
        ["observed_building_record", "permit_constrained_allocation"]
    )
    plausible = measured_or_allocated & gdf["gfa_capacity_ratio"].between(0.25, 2.0)
    gdf["gfa_physical_plausibility_status"] = np.select(
        [
            ~measured_or_allocated,
            gdf["gfa_capacity_ratio"].lt(0.25),
            gdf["gfa_capacity_ratio"].gt(2.0),
            plausible,
        ],
        [
            "not_testable_capacity_proxy",
            "implausibly_low_below_0_25_capacity",
            "implausibly_high_above_2_capacity",
            "plausible_0_25_to_2_capacity",
        ],
        default="unavailable_or_invalid",
    )
    gdf["thermal_floor_area_m2"] = gdf["gfa_value_m2"].where(plausible)
    gdf["thermal_floor_area_source"] = np.where(
        plausible,
        gdf["gfa_source"],
        "unavailable_due_to_gfa_plausibility_gate",
    )
    allocated_lower = gdf["allocated_domestic_gfa_lower_m2"].fillna(0) + gdf["allocated_nondomestic_gfa_lower_m2"].fillna(0)
    allocated_upper = gdf["allocated_domestic_gfa_upper_m2"].fillna(0) + gdf["allocated_nondomestic_gfa_upper_m2"].fillna(0)
    capacity_lower = gdf["footprint_area_m2"] * gdf["floors_lower"]
    capacity_upper = gdf["footprint_area_m2"] * gdf["floors_upper"]
    gdf["gfa_lower_m2"] = np.select([raw.notna(), allocated_total.notna()], [raw, allocated_lower], default=capacity_lower)
    gdf["gfa_upper_m2"] = np.select([raw.notna(), allocated_total.notna()], [raw, allocated_upper], default=capacity_upper)

    permit_share = gdf["allocated_domestic_gfa_m2"].div(allocated_total)
    observed_residential = (raw * permit_share).where(raw.notna() & permit_share.notna())
    allocated_residential = gdf["allocated_domestic_gfa_m2"].where(allocated_total.notna())
    gdf["residential_gfa_observed_or_allocated_m2"] = observed_residential.fillna(allocated_residential)
    proxy_share = gdf["residential_evidence_score"].clip(0, 1)
    gdf["residential_gfa_capacity_proxy_m2"] = capacity * proxy_share
    gdf["residential_gfa_estimate_m2"] = gdf["residential_gfa_observed_or_allocated_m2"].fillna(
        gdf["residential_gfa_capacity_proxy_m2"]
    )
    gdf["nonresidential_gfa_estimate_m2"] = gdf["gfa_value_m2"] - gdf["residential_gfa_estimate_m2"]
    gdf["residential_gfa_source"] = np.select(
        [observed_residential.notna(), allocated_residential.notna()],
        ["registered_gfa_x_permit_domestic_share", "occupation_permit_constrained_domestic_allocation"],
        default="capacity_proxy_x_uncalibrated_residential_score",
    )
    return gdf, allocations, metrics


def assign_analysis_tiers(gdf: pd.DataFrame) -> pd.DataFrame:
    direct_residential = gdf["residential_eligibility_status"].eq("include_residential_direct")
    category_allowed = ~gdf.get(
        "CATEGORY", pd.Series("", index=gdf.index)
    ).astype("string").isin(["5", "9"])
    not_ml_only = ~gdf.get(
        "res_evidence_hkbfetd_ml_only", pd.Series(False, index=gdf.index)
    ).fillna(False).astype(bool)
    official_floors = gdf["floors_source"].eq("landsd_official_storeys")
    observed_year = ~gdf["construction_year_source"].eq("spatial_quantile_model")
    year_consistent = ~gdf["construction_year_conflict"].fillna(True)
    height_spatial_evidence = gdf["height_quality"].isin(
        ["A_official_net", "B_official_roof_dtm_point_sample"]
    )
    height_floor_consistent = gdf["height_floor_consistency_status"].eq(
        "consistent_2_to_6_m_per_floor"
    )
    gfa_constrained = gdf["gfa_measurement_class"].isin(
        ["observed_building_record", "permit_constrained_allocation"]
    )
    gfa_physically_plausible = gdf["gfa_physical_plausibility_status"].eq(
        "plausible_0_25_to_2_capacity"
    )

    gdf["gate_direct_residential"] = direct_residential
    gdf["gate_official_floors"] = official_floors
    gdf["gate_observed_year_no_conflict"] = observed_year & year_consistent
    gdf["gate_height_spatial_floor_consistent"] = height_spatial_evidence & height_floor_consistent
    gdf["gate_gfa_observed_or_constrained"] = gfa_constrained
    gdf["gate_gfa_physically_plausible"] = gfa_physically_plausible

    archetype = direct_residential & official_floors & observed_year & year_consistent
    core = (
        archetype
        & height_spatial_evidence
        & height_floor_consistent
        & gfa_constrained
        & gfa_physically_plausible
        & category_allowed
        & not_ml_only
    )
    eligible = gdf["thermal_stock_eligible"].fillna(False)
    review = gdf["residential_eligibility_status"].str.startswith("review")
    gdf["analysis_tier"] = np.select(
        [core, archetype, review, eligible],
        ["core_simulation_stock", "archetype_training", "manual_review", "uncertainty_propagation"],
        default="excluded",
    )
    gdf["day3_archetype_eligible"] = core | archetype
    gdf["citywide_propagation_only"] = gdf["analysis_tier"].eq("uncertainty_propagation")
    return gdf


def build_manual_audit_sample(gdf: pd.DataFrame, sample_size: int, random_state: int) -> pd.DataFrame:
    candidates = gdf[
        gdf["residential_eligibility_status"].str.startswith("review")
        | gdf["CATEGORY"].astype(str).isin(["5", "9"])
        | gdf["floors_source"].eq("spatial_quantile_model")
    ].copy()
    candidates["audit_stratum"] = (
        candidates["CATEGORY"].astype(str) + "|" + candidates["residential_eligibility_status"].astype(str)
        + "|" + candidates["manual_review_priority"].astype(str)
    )
    groups = candidates.groupby("audit_stratum", group_keys=False)
    minimum = max(1, sample_size // max(groups.ngroups, 1))
    sampled = groups.apply(lambda x: x.sample(min(len(x), minimum), random_state=random_state), include_groups=False)
    if len(sampled) < sample_size:
        remaining = candidates.drop(index=sampled.index, errors="ignore")
        extra = remaining.sample(min(sample_size - len(sampled), len(remaining)), random_state=random_state)
        sampled = pd.concat([sampled, extra])
    sampled = sampled.head(sample_size).copy()
    sampled["audit_label_status"] = "unreviewed"
    sampled["audit_manual_residential_label"] = pd.NA
    sampled["audit_manual_mixed_use_label"] = pd.NA
    sampled["audit_reviewer"] = pd.NA
    sampled["audit_review_date"] = pd.NA
    sampled["audit_notes"] = pd.NA
    return sampled


def quarantine_table(gdf: pd.DataFrame) -> pd.DataFrame:
    raw_floors = pd.to_numeric(gdf["NUMABOVEGROUNDSTOREYS"], errors="coerce")
    raw_top = pd.to_numeric(gdf["TOPHEIGHT"], errors="coerce")
    raw_base = pd.to_numeric(gdf["BASEHEIGHT"], errors="coerce")
    raw_gfa = pd.to_numeric(gdf["GROSSFLOORAREA"], errors="coerce")
    dtm = pd.to_numeric(gdf.get("dtm_elevation_mpd", pd.Series(np.nan, index=gdf.index)), errors="coerce")
    legacy_floors = pd.to_numeric(
        gdf.get("hkbfetd_legacy_floors", pd.Series(np.nan, index=gdf.index)), errors="coerce"
    )
    legacy_height = pd.to_numeric(
        gdf.get("hkbfetd_legacy_height_m", pd.Series(np.nan, index=gdf.index)), errors="coerce"
    )
    reasons = pd.Series("", index=gdf.index, dtype="string")
    reasons = reasons.mask(raw_top.notna() & raw_base.isna() & dtm.isna(), reasons + "TOPHEIGHT_without_terrain_evidence;")
    reasons = reasons.mask(raw_floors.notna() & ~raw_floors.between(1, 120), reasons + "official_floors_invalid;")
    reasons = reasons.mask(raw_gfa.notna() & raw_gfa.le(0), reasons + "registered_gfa_invalid;")
    reasons = reasons.mask(legacy_floors.notna(), reasons + "hkbfetd_legacy_floors_not_reused;")
    reasons = reasons.mask(legacy_height.notna(), reasons + "hkbfetd_legacy_height_quarantined;")
    reasons = reasons.mask(
        gdf["gfa_physical_plausibility_status"].isin(
            [
                "implausibly_low_below_0_25_capacity",
                "implausibly_high_above_2_capacity",
                "unavailable_or_invalid",
            ]
        ),
        reasons + "observed_or_allocated_gfa_outside_physical_capacity_gate;",
    )
    result = pd.DataFrame(
        {
            "BUILDINGSTRUCTUREID": gdf["BUILDINGSTRUCTUREID"],
            "raw_topheight_mpd": raw_top,
            "raw_baseheight_mpd": raw_base,
            "raw_official_floors": raw_floors,
            "raw_registered_gfa_m2": raw_gfa,
            "raw_record_creation_date": gdf["RECORDCREATIONDATE"],
            "hkbfetd_legacy_floors": legacy_floors,
            "hkbfetd_legacy_height_m": legacy_height,
            "quarantine_reasons": reasons.str.rstrip(";"),
        }
    )
    return result[result["quarantine_reasons"].ne("")]


FIELD_META = {
    "residential_evidence_score": ("multisource evidence fusion", "Uncalibrated residential-use evidence score [0,1]", "Screening and stratification only", "Never interpret as AC probability"),
    "residential_eligibility_status": ("deterministic audit rules", "Residential thermal-stock inclusion status", "Select include/mixed records or manual review", "Review conflicts and Category 5/9 positives"),
    "ac_presence_status": ("no local observation", "Building-level AC presence state", "Scenario assignment only", "Always unknown in current sources"),
    "roof_elevation_mpd": ("LandsD TOPHEIGHT", "Roof elevation relative to Hong Kong Principal Datum", "Terrain subtraction after a DTM is added", "Must not be used as net building height"),
    "floors_value": ("LandsD or spatial quantile model", "Observed or repaired above-ground storeys", "Stock modeling with floors_source and interval", "Predictions are not official records"),
    "floors_lower": ("model interval or official value", "Lower storey estimate", "Uncertainty propagation", "Not a legal floor count"),
    "floors_upper": ("model interval or official value", "Upper storey estimate", "Uncertainty propagation", "Not a legal floor count"),
    "height_value_m": ("net official height or floor-height typology", "Net-height estimate", "Heat model geometry with source and interval", "Current data have no BASEHEIGHT/DTM"),
    "height_lower_m": ("floor and typology interval", "Lower net-height estimate", "Uncertainty propagation", "Not surveyed height"),
    "height_upper_m": ("floor and typology interval", "Upper net-height estimate", "Uncertainty propagation", "Not surveyed height"),
    "construction_year_value": ("OP/BDBIAR/CD/CC or spatial model", "Selected construction/completion year", "Era assignment with source and conflict flag", "Modeled years are era proxies"),
    "construction_year_lower": ("official evidence range or model interval", "Lower construction-year estimate", "Era uncertainty", "Conflicting official evidence requires review"),
    "construction_year_upper": ("official evidence range or model interval", "Upper construction-year estimate", "Era uncertainty", "Conflicting official evidence requires review"),
    "gfa_capacity_proxy_m2": ("footprint x repaired floors", "Floor-area capacity proxy", "Weighting and fallback only", "Not legal or conditioned GFA"),
    "gfa_value_m2": ("registered GFA, constrained permit allocation, or capacity proxy", "Best available building-level total GFA", "Stock aggregation with gfa_source", "Allocated/proxy values are estimates"),
    "gfa_lower_m2": ("allocation or floor interval", "Lower GFA estimate", "Uncertainty propagation", "Not a statutory bound"),
    "gfa_upper_m2": ("allocation or floor interval", "Upper GFA estimate", "Uncertainty propagation", "Not a statutory bound"),
    "gfa_capacity_ratio": ("selected GFA / footprint-floor capacity", "Physical plausibility ratio for measured or allocated GFA", "Core-stock quality gate", "Values outside [0.25, 2.0] are excluded from the core"),
    "thermal_floor_area_m2": ("plausible observed or allocated GFA", "Thermal floor area admitted by the physical plausibility gate", "Core heat simulation scaling", "Missing values failed the GFA plausibility gate"),
    "residential_gfa_estimate_m2": ("permit domestic share or evidence-score allocation", "Estimated residential portion of GFA", "Residential stock aggregation", "Evidence-score split is not calibrated"),
    "hkbfetd_legacy_floors": ("HK-BFETD v1.1", "Published legacy floor field retained for traceability", "Comparison only", "Not reused because its fallback lineage mixes official, elevation-derived, and five-storey values"),
    "hkbfetd_legacy_height_m": ("HK-BFETD v1.1", "Published legacy height field retained for traceability", "Comparison only", "Quarantined because TOPHEIGHT was used without terrain/base elevation"),
}


LANDSD_FIELD_MEANINGS = {
    "BUILDINGSTRUCTUREID": "Stable LandsD building-structure identifier",
    "BUILDINGCSUID": "LandsD common spatial-unit identifier",
    "CATEGORY": "LandsD building category code; 5 and 9 are not residential proof",
    "BUILDINGSTRUCTURETYPE": "LandsD structure type code such as tower or podium",
    "STATUS": "LandsD building record status code",
    "OFFICIALBUILDINGNAMEEN": "Official English building name from the structure layer",
    "OFFICIALBUILDINGNAMETC": "Official Traditional Chinese building name from the structure layer",
    "BUILDINGNAMEEN": "Selected English name from the LandsD building-name table",
    "BUILDINGNAMETC": "Selected Traditional Chinese name from the LandsD building-name table",
    "NUMABOVEGROUNDSTOREYS": "Raw official above-ground storey count",
    "TOPHEIGHT": "Raw roof elevation relative to Hong Kong Principal Datum",
    "BASEHEIGHT": "Raw building base elevation relative to Hong Kong Principal Datum",
    "GROSSFLOORAREA": "Raw LandsD registered gross floor area where supplied",
    "footprint_area_m2": "Planar building footprint area in EPSG:2326",
    "perimeter_m": "Planar building footprint perimeter in EPSG:2326",
    "geometry": "Building footprint polygon in EPSG:2326",
}


def dictionary_metadata(column: str) -> tuple[str, str, str, str, str, str, str]:
    unit = "none"
    if column.endswith("_m2") or column in {"GROSSFLOORAREA", "footprint_area_m2"}:
        unit = "m2"
    elif column.endswith("_m") or column in {"TOPHEIGHT", "BASEHEIGHT", "perimeter_m"}:
        unit = "m"
    elif "year" in column or column == "building_age_2026":
        unit = "year"
    elif "floor" in column.lower() or column == "NUMABOVEGROUNDSTOREYS":
        unit = "storey"
    elif column.endswith("score") or column.endswith("value") or column.endswith("_ratio") or "share" in column:
        unit = "fraction"

    if column in LANDSD_FIELD_MEANINGS:
        return (
            unit,
            "Lands Department Building Footprint Public 2025-02-06",
            column if column not in {"footprint_area_m2", "perimeter_m", "geometry"} else "SHAPE_Area / SHAPE_Length / geometry",
            "Direct field; area and perimeter were renamed only",
            LANDSD_FIELD_MEANINGS[column],
            "official_source",
            "Use as raw observation with code tables and CRS; TOPHEIGHT requires terrain subtraction",
        )
    if column in {"ZONE_LABEL", "ozp_plan", "ozp_context"}:
        return (unit, "Planning Department Outline Zoning Plans", "ZONE_LABEL and source plan", "Representative-point spatial join; ozp_context is a grouped label", column.replace("_", " "), "spatial_context", "Context and stratification only; not proof of building use")
    if column == "intended_use_text":
        return (unit, "LandsD BUILDING_INFO", "INFODESCRIPTION", "Unique descriptions joined per building", "Official intended-use description text", "official_text", "Use as direct evidence only when positive and negative terms do not conflict")
    if column.startswith("bdbiar_"):
        return (unit, "Buildings Department BDBIAR", "OBJECTID / NSEARCH3_E / NSEARCH5_E / coordinates", "Nearest structure-type-aware match within 30 m", column.replace("_", " "), "B_spatial_match", "Use with bdbiar_distance_m; never describe as a direct building-key join")
    if column in {"occupation_permit_year", "permit_gfa_allocation_count"}:
        return (unit, "LandsD occupation permits and OP-building links", "OPDATE / OPNO", "Permit records joined through official OPNO-building links", column.replace("_", " "), "official_linked", "Permit-level evidence; allocation count is not a building measurement")
    if column in {"completion_date_year", "compliance_certificate_year"}:
        return (unit, "LandsD BUILDING_WORKS_HISTORY", "WORKSDATE and WORKSTYPE", "Earliest valid CD or CC year per building", column.replace("_", " "), "official_source", "Completion/compliance year evidence; conflicts are flagged")
    if column.startswith("hkbfetd_"):
        return (unit, "Published HK-BFETD v1.1", column.removeprefix("hkbfetd_"), "One-to-one fusion by BUILDINGSTRUCTUREID; legacy geometry fields are quarantined", column.replace("_", " "), "published_derived", "Classification is supporting evidence; ML-only rows are excluded from the core")
    if column in {"dtm_elevation_mpd", "net_height_evidence_m", "dtm_height_evidence_status"}:
        return (unit, "LandsD Open3Dhk DTM endpoint", "DTM point response and LandsD TOPHEIGHT", "DTM sampled at footprint representative point; net evidence = TOPHEIGHT - DTM", column.replace("_", " "), "B_spatial_point_sample", "Not a surveyed BASEHEIGHT; use consistency and source flags")
    if column.startswith("res_evidence_") or column in {"residential_direct_evidence_count", "residential_eligibility_status", "thermal_stock_eligible", "manual_review_priority", "ac_presence_status", "residential_score_is_calibrated"}:
        return (unit, "Residential qualification audit", "Explicit official, spatial, rule, and ML evidence fields", "Deterministic evidence flags and inclusion rules", column.replace("_", " "), "audit_flag", "Category 5/9 and ML-only evidence cannot enter the core; AC remains unknown")
    if column == "residential_evidence_score":
        source, meaning, allowed, warning = FIELD_META[column]
        return (unit, source, "weighted evidence inputs", "Manual weighted average; not statistically calibrated", meaning, "screening_score", f"{allowed}; {warning}")
    if column.startswith("floors_"):
        return (unit, "LandsD storeys plus repair model", "NUMABOVEGROUNDSTOREYS, morphology, DTM height evidence", "Official value retained; missing value predicted with spatial quantile models", column.replace("_", " "), "observed_or_modeled", "Core requires landsd_official_storeys; propagate modeled intervals")
    if column in {"roof_elevation_mpd", "topheight_is_net_height", "terrain_elevation_status"} or column.startswith("height_"):
        return (unit, "LandsD roof/base, Open3Dhk DTM, and floor typology", "TOPHEIGHT, BASEHEIGHT, DTM, repaired floors", "Prefer TOPHEIGHT-BASEHEIGHT, then TOPHEIGHT-DTM, else floors x typology", column.replace("_", " "), "A_B_or_C_by_height_quality", "TOPHEIGHT alone is never net height; DTM subtraction is B-grade")
    if column.startswith("construction_year_") or column == "building_age_2026":
        return (unit, "OP, LandsD works, BDBIAR, and year repair model", "OPDATE, WORKSDATE, BDBIAR completion year", "Priority OP > LandsD CD > LandsD CC > BDBIAR; missing years modeled", column.replace("_", " "), "A_B_or_modeled", "BDBIAR is a spatial match; model years are era proxies")
    if column.startswith("gfa_") or column.startswith("residential_gfa_") or column == "nonresidential_gfa_estimate_m2":
        return (unit, "LandsD GFA, occupation permits, and geometry proxy", "GROSSFLOORAREA, permit GFA, footprint and floors", "Observed, permit-constrained allocation, and capacity proxy stored separately", column.replace("_", " "), "observed_allocated_or_proxy", "Do not mix capacity_proxy with observed/allocated GFA without uncertainty propagation")
    if column in {"thermal_floor_area_m2", "thermal_floor_area_source"}:
        return (unit, "GFA physical plausibility gate", "gfa_value_m2 and gfa_capacity_proxy_m2", "Admit observed or allocated GFA only when its capacity ratio is within [0.25, 2.0]", column.replace("_", " "), "core_simulation_input", "Missing means the record is excluded from core heat-simulation scaling")
    if column.startswith("gate_") or column in {"analysis_tier", "day3_archetype_eligible", "citywide_propagation_only"}:
        return (unit, "Analysis readiness contract", "Qualified residential, floor, year, height, and GFA fields", "Deterministic conjunction of explicit quality gates", column.replace("_", " "), "readiness_flag", "Use only for the analysis stage named by analysis_tier")
    raise KeyError(f"No publication-grade dictionary metadata for field: {column}")


def write_dictionary(gdf: pd.DataFrame, path: Path) -> None:
    rows = []
    for column in gdf.columns:
        unit, dataset, source_field, transformation, meaning, quality, warning = dictionary_metadata(column)
        missing = int(gdf[column].isna().sum())
        rows.append(
            {
                "field": column,
                "dtype": str(gdf[column].dtype),
                "unit": unit,
                "source_dataset": dataset,
                "source_field": source_field,
                "transformation": transformation,
                "meaning": meaning,
                "quality_class": quality,
                "allowed_use_and_prohibition": warning,
                "missing_count": missing,
                "missing_pct": round(100 * missing / len(gdf), 6),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def write_unavailable_field_manifest(path: Path) -> None:
    rows = [
        {
            "field_or_concept": "TOPHEIGHT used alone",
            "status": "prohibited",
            "reason": "Roof elevation is relative to HKPD, not net building height",
            "replacement": "TOPHEIGHT minus valid BASEHEIGHT or Open3Dhk DTM point sample",
        },
        {
            "field_or_concept": "RECORDCREATIONDATE as construction year",
            "status": "prohibited",
            "reason": "Database record creation is not building completion",
            "replacement": "OPDATE, LandsD CD/CC, BDBIAR spatial match, or modeled era",
        },
        {
            "field_or_concept": "HK-BFETD legacy height and floors",
            "status": "comparison_only",
            "reason": "Legacy lineage includes TOPHEIGHT without terrain subtraction and five-storey fallback",
            "replacement": "Repaired fields with source, quality, and interval companions",
        },
        {
            "field_or_concept": "Building-level air-conditioning presence",
            "status": "unavailable",
            "reason": "No fused source observes AC presence for individual buildings",
            "replacement": "Aggregate adoption scenarios with uncertainty; never infer from residential class",
        },
        {
            "field_or_concept": "Capacity-proxy GFA as legal or conditioned area",
            "status": "prohibited",
            "reason": "Footprint times repaired floors is a geometric capacity proxy",
            "replacement": "Observed GFA or conserved permit allocation for the core; proxy only for uncertainty propagation",
        },
        {
            "field_or_concept": "Observed or allocated GFA outside 0.25-2.0 times geometric capacity",
            "status": "quarantined_from_core",
            "reason": "Permit conservation alone does not prove a physically plausible building-level allocation",
            "replacement": "Retain source GFA for audit; use only records passing the explicit physical plausibility gate",
        },
        {
            "field_or_concept": "Category 5/9 or ML-only record as core residential",
            "status": "prohibited",
            "reason": "Ambiguous category or prediction is not direct residential-use evidence",
            "replacement": "Direct permit, near BDBIAR, intended-use, Category 2, or HA tower evidence",
        },
        {
            "field_or_concept": "BDBIAR matched year as direct official building key",
            "status": "prohibited",
            "reason": "BDBIAR is joined spatially within 30 m and can mismatch dense sites",
            "replacement": "Quality B spatial-match evidence with distance; direct OP/LandsD works take priority",
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def write_report(summary: dict, path: Path) -> None:
    lines = [
        "# Heatwave residential stock: day 1-2 validation report",
        "",
        "The published HK-BFETD workflow remains unchanged. This output is a derived, uncertainty-aware research layer.",
        "",
        f"- Buildings processed: {summary['buildings']:,}",
        f"- HK-BFETD release status: {summary['hkbfetd_release_status']}",
        f"- Official floor records retained: {summary['official_floors']:,}",
        f"- Floor records modeled: {summary['modeled_floors']:,}",
        f"- TOPHEIGHT values isolated as roof elevation: {summary['topheight_isolated']:,}",
        f"- Records with a DTM/base elevation: {summary['baseheight_available']:,}",
        f"- LandsD Open3Dhk DTM point samples fused: {summary['dtm_point_samples']:,}",
        f"- Valid roof-minus-DTM height evidence: {summary['valid_roof_minus_dtm_heights']:,}",
        f"- Official construction-year evidence: {summary['official_years']:,}",
        f"- Registered building-level GFA records: {summary['registered_gfa']:,}",
        f"- Buildings receiving constrained permit GFA: {summary['permit_gfa_buildings']:,}",
        f"- Observed/allocated GFA records outside the physical capacity gate: {summary['gfa_plausibility_failures']:,}",
        "",
        "## Validation",
        "",
        f"- Floor spatial holdout: MAE={summary['floor_model']['mae']:.3f}, RMSE={summary['floor_model']['rmse']:.3f}, R2={summary['floor_model']['r2']:.3f}",
        f"- Year spatial holdout: MAE={summary['year_model']['mae']:.3f}, RMSE={summary['year_model']['rmse']:.3f}, R2={summary['year_model']['r2']:.3f}",
        f"- Maximum permit domestic-GFA conservation error: {summary['gfa_allocation']['max_domestic_conservation_error_m2']:.9f} m2",
        f"- Maximum permit non-domestic-GFA conservation error: {summary['gfa_allocation']['max_nondomestic_conservation_error_m2']:.9f} m2",
        f"- Floor MAE <= 2-storey gate: {'PASS' if summary['validation_gates']['floor_spatial_mae_le_2'] else 'FAIL'}",
        f"- Year MAE <= 10-year era gate: {'PASS' if summary['validation_gates']['year_spatial_mae_le_10'] else 'FAIL'}",
        f"- Permit GFA conservation <= 1e-6 m2 gate: {'PASS' if summary['validation_gates']['permit_gfa_conservation_le_1e-6'] else 'FAIL'}",
        f"- Core GFA physical-plausibility gate: {'PASS' if summary['validation_gates']['core_gfa_physically_plausible'] else 'FAIL'}",
        f"- Net-height spatial-evidence gate: {'PASS' if summary['validation_gates']['net_height_spatial_evidence_available'] else 'FAIL'}",
        f"- Archetype-training sample >= 1,000: {'PASS' if summary['validation_gates']['archetype_training_at_least_1000'] else 'FAIL'}",
        f"- Core simulation stock >= 500: {'PASS' if summary['validation_gates']['core_simulation_stock_at_least_500'] else 'FAIL'}",
        f"- Day 3 data-foundation gate: {'PASS' if summary['validation_gates']['day3_data_foundation_ready'] else 'FAIL'}",
        "",
        "## Required interpretation",
        "",
        "- TOPHEIGHT is never treated as net height; a DTM point sample is subtracted when available.",
        "- Open3Dhk DTM values are B-grade point samples, not surveyed building base heights.",
        "- Model estimates do not overwrite raw official fields.",
        "- Permit GFA is conserved at permit level and allocated once across linked structures.",
        "- Residential evidence is not a building-level observation of air-conditioning presence.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hkbfetd-path", type=Path, default=None)
    parser.add_argument("--audit-sample-size", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    print("[1/8] Loading LandsD buildings, names, intended use, permits, and works")
    gdf = attach_dtm_evidence(attach_names_and_info(load_buildings()))
    permits, links, permit_building = load_permit_evidence()
    gdf = gdf.merge(permit_building, on="BUILDINGSTRUCTUREID", how="left")
    gdf = gdf.merge(load_work_years(), on="BUILDINGSTRUCTUREID", how="left")

    print("[2/8] Fusing OZP and BDBIAR official evidence")
    gdf = gdf.merge(load_zones(gdf), on="BUILDINGSTRUCTUREID", how="left")
    gdf = gdf.merge(match_bdbiar(gdf), on="BUILDINGSTRUCTUREID", how="left")

    print("[3/8] Fusing the published HK-BFETD release when available")
    release, release_status = load_hkbfetd(args.hkbfetd_path)
    if release is not None:
        gdf = gdf.merge(release, on="BUILDINGSTRUCTUREID", how="left", validate="one_to_one")

    for column in ("info_residential", "info_nonresidential", "bdbiar_residential", "bdbiar_nonresidential"):
        if column not in gdf:
            gdf[column] = False
    for column in ("permit_residential_share_signal", "bdbiar_residential_value", "bdbiar_distance_m"):
        if column not in gdf:
            gdf[column] = np.nan
    print("[4/8] Auditing residential eligibility without inferring AC presence")
    gdf = qualify_residential(gdf)

    year_evidence = resolve_year_evidence(gdf)
    for column in year_evidence:
        gdf[column] = year_evidence[column]

    print("[5/8] Repairing floors and construction year with spatial holdout validation")
    gdf, floor_models, floor_metrics = repair_floors(gdf, args.random_state)
    gdf, year_models, year_metrics = repair_years(gdf, args.random_state)

    print("[6/8] Isolating TOPHEIGHT and deriving uncertainty-aware net-height estimates")
    gdf = repair_height(gdf)

    print("[7/8] Conserving and allocating occupation-permit GFA")
    gdf, allocations, allocation_metrics = repair_gfa(gdf, permits, links)
    gdf = assign_analysis_tiers(gdf)

    print("[8/8] Writing repaired stock, audit sample, quarantine table, and reports")
    audit_columns = [
        "BUILDINGSTRUCTUREID", "CATEGORY", "BUILDINGSTRUCTURETYPE", "OFFICIALBUILDINGNAMEEN",
        "BUILDINGNAMEEN", "intended_use_text", "ZONE_LABEL", "bdbiar_usage", "bdbiar_distance_m",
        "permit_residential_share_signal", "residential_evidence_score", "residential_eligibility_status",
        "res_evidence_permit_direct", "res_evidence_permit_mixed", "res_evidence_bdbiar_spatial",
        "res_evidence_info_direct", "res_evidence_category_direct", "res_evidence_hkbfetd_rule",
        "res_evidence_hkbfetd_ml_only", "res_evidence_negative_direct", "res_evidence_conflict",
        "manual_review_priority", "floors_value", "floors_lower", "floors_upper", "floors_source",
        "construction_year_value", "construction_year_source", "height_value_m", "height_source",
        "gfa_observed_m2", "gfa_allocated_m2", "gfa_capacity_proxy_m2", "gfa_value_m2", "gfa_source",
        "gfa_measurement_class", "gfa_capacity_ratio", "gfa_physical_plausibility_status", "analysis_tier",
    ]
    audit_columns = [column for column in audit_columns if column in gdf]
    audit_sample = build_manual_audit_sample(gdf[audit_columns + ["CATEGORY"]].loc[:, ~gdf[audit_columns + ["CATEGORY"]].columns.duplicated()], args.audit_sample_size, args.random_state)
    audit_sample.to_csv(output / "residential_manual_audit_sample.csv", index=False, encoding="utf-8-sig")
    quarantine_table(gdf).to_csv(output / "quarantined_source_fields.csv", index=False, encoding="utf-8-sig")

    allocation_export = allocations[[
        "OPNO", "BUILDINGSTRUCTUREID", "permit_link_count", "DOMESTICGFA", "NONDOMESTICGFA",
        "allocated_domestic_gfa_m2", "allocated_nondomestic_gfa_m2",
        "allocated_domestic_gfa_lower_m2", "allocated_domestic_gfa_upper_m2",
        "allocated_nondomestic_gfa_lower_m2", "allocated_nondomestic_gfa_upper_m2",
    ]]
    allocation_export.to_csv(output / "occupation_permit_gfa_allocation.csv", index=False, encoding="utf-8-sig")

    export_columns = [
        "BUILDINGSTRUCTUREID", "BUILDINGCSUID", "CATEGORY", "BUILDINGSTRUCTURETYPE", "STATUS",
        "OFFICIALBUILDINGNAMEEN", "OFFICIALBUILDINGNAMETC", "BUILDINGNAMEEN", "BUILDINGNAMETC",
        "footprint_area_m2", "perimeter_m", "ZONE_LABEL", "ozp_plan", "ozp_context",
        "intended_use_text", "bdbiar_record_ids", "bdbiar_usage", "bdbiar_distance_m",
        "occupation_permit_year", "completion_date_year", "compliance_certificate_year",
        "residential_evidence_score", "residential_score_is_calibrated", "residential_eligibility_status",
        "thermal_stock_eligible", "manual_review_priority", "ac_presence_status",
        "res_evidence_permit_direct", "res_evidence_permit_mixed", "res_evidence_bdbiar_spatial",
        "res_evidence_info_direct", "res_evidence_category_direct", "res_evidence_hkbfetd_rule",
        "res_evidence_hkbfetd_ml_only", "res_evidence_negative_direct", "res_evidence_conflict",
        "residential_direct_evidence_count",
        "NUMABOVEGROUNDSTOREYS", "floors_value", "floors_lower", "floors_upper", "floors_source", "floors_quality",
        "TOPHEIGHT", "BASEHEIGHT", "dtm_elevation_mpd", "net_height_evidence_m", "dtm_height_evidence_status",
        "roof_elevation_mpd", "topheight_is_net_height", "terrain_elevation_status",
        "height_value_m", "height_lower_m", "height_upper_m", "height_source", "height_quality",
        "height_floor_consistency_status",
        "construction_year_value", "construction_year_lower", "construction_year_upper",
        "construction_year_source", "construction_year_quality", "construction_year_conflict", "building_age_2026",
        "GROSSFLOORAREA", "gfa_observed_m2", "gfa_allocated_m2", "gfa_capacity_proxy_m2",
        "gfa_value_m2", "gfa_lower_m2", "gfa_upper_m2", "gfa_source", "gfa_measurement_class",
        "gfa_capacity_ratio", "gfa_physical_plausibility_status", "thermal_floor_area_m2",
        "thermal_floor_area_source",
        "residential_gfa_observed_or_allocated_m2", "residential_gfa_capacity_proxy_m2",
        "residential_gfa_estimate_m2", "nonresidential_gfa_estimate_m2", "residential_gfa_source",
        "permit_gfa_allocation_count", "gate_direct_residential", "gate_official_floors",
        "gate_observed_year_no_conflict", "gate_height_spatial_floor_consistent",
        "gate_gfa_observed_or_constrained", "gate_gfa_physically_plausible", "analysis_tier", "day3_archetype_eligible",
        "citywide_propagation_only", "geometry",
    ]
    export_columns += [column for column in (
        "hkbfetd_main_class", "hkbfetd_sub_class", "hkbfetd_residential_value", "hkbfetd_confidence", "hkbfetd_source",
        "hkbfetd_legacy_floors", "hkbfetd_legacy_height_m",
    ) if column in gdf]
    export_columns = [column for column in export_columns if column in gdf]
    result = gdf[export_columns].copy()
    result.to_parquet(output / "building_stock_repaired.geoparquet", index=False)
    result.drop(columns="geometry").to_csv(output / "building_stock_repaired.csv", index=False, encoding="utf-8-sig")
    result[result["analysis_tier"].eq("core_simulation_stock")].to_parquet(
        output / "building_stock_core_simulation.geoparquet", index=False
    )
    result[result["day3_archetype_eligible"]].to_parquet(
        output / "building_stock_archetype_training.geoparquet", index=False
    )
    result[result["analysis_tier"].eq("uncertainty_propagation")].to_parquet(
        output / "building_stock_uncertainty_propagation.geoparquet", index=False
    )
    result[result["analysis_tier"].eq("manual_review")].drop(columns="geometry").to_csv(
        output / "residential_review_layer.csv", index=False, encoding="utf-8-sig"
    )
    write_dictionary(result, output / "field_data_dictionary.csv")
    write_unavailable_field_manifest(output / "unavailable_field_manifest.csv")
    joblib.dump({"models": floor_models, "feature_columns": feature_matrix(gdf, False)[0].columns.tolist()}, output / "floor_models.joblib")
    joblib.dump(
        {"models": year_models, "feature_columns": feature_matrix(gdf, True, False)[0].columns.tolist()},
        output / "year_models.joblib",
    )

    summary = {
        "buildings": int(len(gdf)),
        "hkbfetd_release_status": release_status,
        "eligibility_counts": gdf["residential_eligibility_status"].value_counts(dropna=False).to_dict(),
        "category_counts": gdf["CATEGORY"].value_counts(dropna=False).to_dict(),
        "official_floors": int(gdf["floors_source"].eq("landsd_official_storeys").sum()),
        "modeled_floors": int(gdf["floors_source"].ne("landsd_official_storeys").sum()),
        "topheight_isolated": int(gdf["roof_elevation_mpd"].notna().sum()),
        "baseheight_available": int(pd.to_numeric(gdf["BASEHEIGHT"], errors="coerce").notna().sum()),
        "official_years": int(gdf["construction_year_source"].ne("spatial_quantile_model").sum()),
        "year_conflicts": int(gdf["construction_year_conflict"].sum()),
        "registered_gfa": int(pd.to_numeric(gdf["GROSSFLOORAREA"], errors="coerce").gt(0).sum()),
        "permit_gfa_buildings": int(gdf["gfa_source"].eq("occupation_permit_constrained_allocation").sum()),
        "gfa_plausibility_failures": int(
            gdf["gfa_physical_plausibility_status"].isin(
                [
                    "implausibly_low_below_0_25_capacity",
                    "implausibly_high_above_2_capacity",
                    "unavailable_or_invalid",
                ]
            ).sum()
        ),
        "dtm_point_samples": int(gdf["dtm_elevation_mpd"].notna().sum()),
        "valid_roof_minus_dtm_heights": int(gdf["net_height_evidence_m"].notna().sum()),
        "analysis_tier_counts": gdf["analysis_tier"].value_counts(dropna=False).to_dict(),
        "gfa_measurement_class_counts": gdf["gfa_measurement_class"].value_counts(dropna=False).to_dict(),
        "floor_model": floor_metrics,
        "year_model": year_metrics,
        "gfa_allocation": allocation_metrics,
    }
    summary["validation_gates"] = {
        "floor_spatial_mae_le_2": bool(floor_metrics["mae"] <= 2.0),
        "year_spatial_mae_le_10": bool(year_metrics["mae"] <= 10.0),
        "permit_gfa_conservation_le_1e-6": bool(
            allocation_metrics["max_domestic_conservation_error_m2"] <= 1e-6
            and allocation_metrics["max_nondomestic_conservation_error_m2"] <= 1e-6
        ),
        "net_height_spatial_evidence_available": bool(gdf["net_height_evidence_m"].notna().any()),
        "archetype_training_at_least_1000": bool(gdf["day3_archetype_eligible"].sum() >= 1000),
        "core_simulation_stock_at_least_500": bool(gdf["analysis_tier"].eq("core_simulation_stock").sum() >= 500),
        "core_excludes_category_5_9": bool(
            ~gdf.loc[gdf["analysis_tier"].eq("core_simulation_stock"), "CATEGORY"].astype(str).isin(["5", "9"]).any()
        ),
        "core_excludes_ml_only": bool(
            ~gdf.loc[gdf["analysis_tier"].eq("core_simulation_stock"), "res_evidence_hkbfetd_ml_only"].any()
        ),
        "core_gfa_physically_plausible": bool(
            gdf.loc[
                gdf["analysis_tier"].eq("core_simulation_stock"),
                "gate_gfa_physically_plausible",
            ].all()
        ),
    }
    gates = summary["validation_gates"]
    gates["day3_data_foundation_ready"] = bool(
        gates["year_spatial_mae_le_10"]
        and gates["permit_gfa_conservation_le_1e-6"]
        and gates["net_height_spatial_evidence_available"]
        and gates["archetype_training_at_least_1000"]
        and gates["core_simulation_stock_at_least_500"]
        and gates["core_excludes_category_5_9"]
        and gates["core_excludes_ml_only"]
        and gates["core_gfa_physically_plausible"]
    )
    gates["citywide_floor_propagation_ready"] = gates["floor_spatial_mae_le_2"]
    (output / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, output / "validation_report.md")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
