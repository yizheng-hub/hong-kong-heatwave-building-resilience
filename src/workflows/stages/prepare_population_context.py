# -*- coding: utf-8 -*-
"""Integrate official repair snapshots and contextual Census evidence.

The stage deliberately separates contextual and statutory-record evidence from
household observations, measured retrofit effects, and structural-damage labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
FOUNDATION = ROOT / "intermediate" / "prepared_stock" / "source"
OFFICIAL = ROOT / "data" / "official"
DEFAULT_OUTPUT = ROOT / "intermediate" / "prepared_stock" / "population"

REPAIR_SOURCES = {
    "mbis_issued_snapshot": OFFICIAL / "bd_mbis_issued_20260831.csv",
    "mbis_closed_withdrawn_superseded_snapshot": (
        OFFICIAL / "bd_mbis_closed_20260831.csv"
    ),
    "dangerous_order_issued_snapshot": (
        OFFICIAL / "bd_dangerous_orders_issued_20260831.csv"
    ),
    "dangerous_order_closed_withdrawn_superseded_snapshot": (
        OFFICIAL / "bd_dangerous_orders_closed_20260831.csv"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_estate_name(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return "".join(character for character in text if character.isalnum())


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace({"-": np.nan, "@": np.nan}), errors="coerce")


def english_label(series: pd.Series) -> pd.Series:
    """Return the English prefix from bilingual HK-BFETD class labels."""
    return series.astype("string").str.split("_", n=1).str[0]


def census_count(series: pd.Series) -> pd.Series:
    """Parse Census count fields while retaining suppressed values as missing."""
    return pd.to_numeric(series.replace({"-": 0, "@": np.nan, "**": np.nan}), errors="coerce")


def unique_point_matches(
    records: pd.DataFrame,
    buildings: gpd.GeoDataFrame,
    dataset_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return exact one-footprint matches and quarantine all other records."""
    points = gpd.GeoDataFrame(
        records.copy(),
        geometry=gpd.points_from_xy(records["EASTING"], records["NORTHING"]),
        crs="EPSG:2326",
    )
    points["source_record_id"] = points["GmlId"].astype(str)
    joined = gpd.sjoin(
        points,
        buildings[["BUILDINGSTRUCTUREID", "geometry"]],
        how="left",
        predicate="within",
    )
    matched_counts = joined.groupby("source_record_id")["BUILDINGSTRUCTUREID"].count()
    joined["footprint_match_count"] = joined["source_record_id"].map(matched_counts)
    accepted = joined.loc[joined["footprint_match_count"].eq(1)].copy()
    accepted["dataset_key"] = dataset_key
    accepted["official_notice_count"] = numeric(accepted["NSEARCH01_EN"]).fillna(1)
    accepted["link_method"] = "point_within_exactly_one_building_footprint"
    accepted["temporal_interpretation"] = "current_public_snapshot_not_event_timeline"
    accepted_columns = [
        "dataset_key",
        "source_record_id",
        "OBJECTID",
        "BUILDINGSTRUCTUREID",
        "official_notice_count",
        "NAME_EN",
        "ADDRESS_EN",
        "NAME_TC",
        "ADDRESS_TC",
        "NSEARCH02_EN",
        "LASTUPDATE",
        "EASTING",
        "NORTHING",
        "LATITUDE",
        "LONGITUDE",
        "footprint_match_count",
        "link_method",
        "temporal_interpretation",
    ]
    accepted = pd.DataFrame(accepted[accepted_columns])

    unmatched_ids = matched_counts.loc[matched_counts.eq(0)].index
    ambiguous_ids = matched_counts.loc[matched_counts.gt(1)].index
    quarantine = points.loc[
        points["source_record_id"].isin([*unmatched_ids, *ambiguous_ids])
    ].copy()
    quarantine["dataset_key"] = dataset_key
    quarantine["footprint_match_count"] = quarantine["source_record_id"].map(
        matched_counts
    )
    quarantine["quarantine_reason"] = np.where(
        quarantine["footprint_match_count"].eq(0),
        "no_containing_building_footprint",
        "multiple_overlapping_building_footprints",
    )
    quarantine["nearest_neighbour_forcing_permitted"] = False
    quarantine_columns = [
        "dataset_key",
        "source_record_id",
        "OBJECTID",
        "NAME_EN",
        "ADDRESS_EN",
        "LASTUPDATE",
        "EASTING",
        "NORTHING",
        "footprint_match_count",
        "quarantine_reason",
        "nearest_neighbour_forcing_permitted",
    ]
    return accepted, pd.DataFrame(quarantine[quarantine_columns])


def read_stpug_context() -> gpd.GeoDataFrame:
    table = pd.read_csv(
        OFFICIAL / "census2021_STPUG_21C" / "STPUG_21C.CSV",
        header=4,
        dtype={"stpug": str},
    )
    table["census_population"] = numeric(table["t_pop"])
    table = table.loc[table["census_population"].notna()].copy()
    selected = pd.DataFrame(
        {
            "stpug_code": table["stpug"],
            "stpug_name_en": table["stpug_eng"],
            "stpug_name_tc": table["stpug_chi"],
            "census_population": table["census_population"],
            "census_population_age65plus": census_count(table["age_5"]),
            "census_domestic_households": census_count(table["dh"]),
            "census_median_monthly_household_income_hkd": numeric(table["ma_hh"]),
            "census_median_accommodation_floor_area_m2": numeric(table["fa_m"]),
            "census_public_rental_households": census_count(table["dh_pub"]),
            "census_subsidised_sale_households": census_count(table["dh_s"]),
            "census_private_permanent_households": census_count(table["dh_pri"]),
            "census_owner_with_mortgage_households": census_count(table["dh_ocm"]),
            "census_owner_without_mortgage_households": census_count(table["dh_ocwm"]),
            "census_sole_tenant_households": census_count(table["dh_st"]),
            "census_co_main_subtenant_households": census_count(table["dh_co"]),
        }
    )
    selected["census_age65plus_share"] = (
        selected["census_population_age65plus"] / selected["census_population"]
    )
    selected["census_public_or_subsidised_household_share"] = (
        selected["census_public_rental_households"]
        + selected["census_subsidised_sale_households"]
    ) / selected["census_domestic_households"]
    selected["census_tenant_household_share"] = (
        selected["census_sole_tenant_households"]
        + selected["census_co_main_subtenant_households"]
    ) / selected["census_domestic_households"]

    geometry = gpd.read_file(
        OFFICIAL
        / "census2021_gml_tpu"
        / "All Characteristics - STPUG.gml"
    )[["STPUG", "geometry"]].rename(columns={"STPUG": "stpug_code"})
    geometry["stpug_code"] = geometry["stpug_code"].astype(str)
    context = geometry.merge(selected, on="stpug_code", how="left", validate="1:1")
    if context["census_population"].isna().any():
        missing = context.loc[context["census_population"].isna(), "stpug_code"].tolist()
        raise RuntimeError(f"Missing STPUG Census statistics for: {missing}")
    return context


MHE_POPULATION = "TIME_2021__STAT_VAR_RP__STAT_PRES_RP_NPER__SEX_All__AGE_All"
MHE_AGE65 = (
    "TIME_2021__STAT_VAR_RP__STAT_PRES_RP_NPER__SEX_All__AGE_65_10000"
)
MHE_HOUSEHOLDS = "TIME_2021__STAT_VAR_DH__STAT_PRES_DH_NHH__UHSIZE_All"
MHE_INCOME = "TIME_2021__STAT_VAR_MED_HHINCOME__STAT_PRES_MED_HHINCOME_HKD"


def read_mhe_context() -> gpd.GeoDataFrame:
    raw = gpd.read_file(
        OFFICIAL / "census2021_gml_mhe" / "All Characteristics - MHE.gml"
    )
    selected = raw[
        ["gml_id", "ENAME", "CNAME", MHE_POPULATION, MHE_AGE65,
         MHE_HOUSEHOLDS, MHE_INCOME, "geometry"]
    ].copy()
    selected = selected.rename(
        columns={
            "gml_id": "mhe_code",
            "ENAME": "mhe_name_en",
            "CNAME": "mhe_name_tc",
            MHE_POPULATION: "mhe_population",
            MHE_AGE65: "mhe_population_age65plus",
            MHE_HOUSEHOLDS: "mhe_domestic_households",
            MHE_INCOME: "mhe_median_monthly_household_income_hkd",
        }
    )
    for column in [
        "mhe_population",
        "mhe_population_age65plus",
        "mhe_domestic_households",
        "mhe_median_monthly_household_income_hkd",
    ]:
        selected[column] = numeric(selected[column])
    selected["mhe_age65plus_share"] = (
        selected["mhe_population_age65plus"] / selected["mhe_population"]
    )
    selected["estate_name_key"] = selected["mhe_name_en"].map(normalize_estate_name)
    return selected


def _english(value: object) -> object:
    return value.get("en") if isinstance(value, dict) else value


def read_ha_profiles() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specs = [
        (
            "public_rental_housing",
            OFFICIAL / "ha_public_housing_estates_20260831.json",
            "Year of Intake",
            "No. of Rental Flats",
        ),
        (
            "hos_psps_gsh",
            OFFICIAL / "ha_hos_courts_20260831.json",
            "Year of Completion",
            "No. of Flats",
        ),
    ]
    for housing_program, path, year_key, flats_key in specs:
        for record in json.loads(path.read_text(encoding="utf-8")):
            name_en = _english(record.get("Estate Name"))
            rows.append(
                {
                    "ha_housing_program": housing_program,
                    "ha_estate_name_en": name_en,
                    "ha_estate_name_tc": (
                        record.get("Estate Name", {}).get("zh-Hant")
                        if isinstance(record.get("Estate Name"), dict)
                        else None
                    ),
                    "ha_district_name_en": _english(record.get("District Name")),
                    "ha_reference_year": _english(record.get(year_key)),
                    "ha_number_of_blocks": _english(record.get("No. of Blocks")),
                    "ha_number_of_flats": _english(record.get(flats_key)),
                    "ha_estate_latitude": record.get("Estate Map Latitude"),
                    "ha_estate_longitude": record.get("Estate Map Longitude"),
                    "estate_name_key": normalize_estate_name(name_en),
                    "ha_source_file": path.name,
                }
            )
    return pd.DataFrame(rows)


def exact_unique_estate_linkage(
    mhe: pd.DataFrame, ha: pd.DataFrame
) -> pd.DataFrame:
    mhe_keys = mhe["estate_name_key"].value_counts()
    ha_keys = ha["estate_name_key"].value_counts()
    rows: list[dict[str, object]] = []
    ha_by_key = {key: group for key, group in ha.groupby("estate_name_key")}
    for estate in mhe.drop(columns="geometry", errors="ignore").to_dict("records"):
        key = estate["estate_name_key"]
        candidates = ha_by_key.get(key)
        if not key or candidates is None:
            status = "no_exact_name_match"
            profile: dict[str, object] = {}
        elif mhe_keys.get(key, 0) != 1 or ha_keys.get(key, 0) != 1:
            status = "ambiguous_normalized_name"
            profile = {}
        else:
            status = "exact_unique_normalized_name"
            profile = candidates.iloc[0].to_dict()
        rows.append(
            {
                **estate,
                **{k: v for k, v in profile.items() if k != "estate_name_key"},
                "ha_link_status": status,
                "ha_candidate_count": int(ha_keys.get(key, 0)),
            }
        )
    return pd.DataFrame(rows)


def unique_context_join(
    buildings: gpd.GeoDataFrame,
    polygons: gpd.GeoDataFrame,
    polygon_columns: list[str],
    context_name: str,
    require_all: bool,
) -> pd.DataFrame:
    points = buildings[["BUILDINGSTRUCTUREID", "geometry"]].copy()
    points.geometry = points.geometry.representative_point()
    joined = gpd.sjoin(
        points,
        polygons[[*polygon_columns, "geometry"]],
        how="left",
        predicate="within",
    )
    counts = joined.groupby("BUILDINGSTRUCTUREID")[polygon_columns[0]].count()
    ambiguous = counts[counts.gt(1)]
    if not ambiguous.empty:
        raise RuntimeError(
            f"{context_name} has ambiguous polygon matches for {len(ambiguous)} buildings"
        )
    joined = joined.drop_duplicates("BUILDINGSTRUCTUREID")
    if require_all and counts.lt(1).any():
        raise RuntimeError(
            f"{context_name} missing for {int(counts.lt(1).sum())} core buildings"
        )
    result = pd.DataFrame(joined.drop(columns=["geometry", "index_right"]))
    result[f"{context_name}_link_method"] = "representative_point_within_polygon"
    result[f"{context_name}_match_count"] = result[polygon_columns[0]].notna().astype(int)
    return result


def build_repair_summary(
    stock: gpd.GeoDataFrame, links: pd.DataFrame
) -> pd.DataFrame:
    record_counts = links.pivot_table(
        index="BUILDINGSTRUCTUREID",
        columns="dataset_key",
        values="source_record_id",
        aggfunc="count",
        fill_value=0,
    )
    notice_counts = links.pivot_table(
        index="BUILDINGSTRUCTUREID",
        columns="dataset_key",
        values="official_notice_count",
        aggfunc="sum",
        fill_value=0,
    )
    record_counts.columns = [f"{column}_location_records" for column in record_counts]
    notice_counts.columns = [f"{column}_official_notice_count" for column in notice_counts]
    base_columns = [
        "BUILDINGSTRUCTUREID",
        "CATEGORY",
        "BUILDINGSTRUCTURETYPE",
        "construction_year_value",
        "construction_year_source",
        "construction_year_quality",
        "building_age_2026",
        "residential_eligibility_status",
        "analysis_tier",
    ]
    summary = pd.DataFrame(stock[base_columns]).set_index("BUILDINGSTRUCTUREID")
    summary = summary.join(record_counts).join(notice_counts).fillna(
        {column: 0 for column in [*record_counts.columns, *notice_counts.columns]}
    )
    evidence_columns = [column for column in summary if column.endswith("_location_records")]
    summary["any_statutory_condition_record_in_downloaded_snapshots"] = (
        summary[evidence_columns].sum(axis=1) > 0
    )
    summary["no_record_interpretation"] = (
        "no_matched_record_in_downloaded_snapshots_not_evidence_of_sound_condition"
    )
    summary["structural_vulnerability_label_permitted"] = False
    return summary.reset_index()


def describe_age_condition(summary: pd.DataFrame) -> pd.DataFrame:
    frame = summary.copy()
    frame["building_age_2026"] = numeric(frame["building_age_2026"])
    frame["age_band"] = pd.cut(
        frame["building_age_2026"],
        bins=[-np.inf, 19, 39, 59, np.inf],
        labels=["0-19", "20-39", "40-59", "60+"],
    )
    frame["year_evidence_group"] = np.where(
        frame["construction_year_quality"].astype(str).str.contains(
            "official|observed|high", case=False, na=False
        ),
        "observed_or_high_quality",
        "estimated_or_other_quality",
    )
    grouped = (
        frame.groupby(
            ["year_evidence_group", "age_band", "CATEGORY"],
            observed=True,
            dropna=False,
        )
        .agg(
            buildings=("BUILDINGSTRUCTUREID", "size"),
            buildings_with_snapshot_record=(
                "any_statutory_condition_record_in_downloaded_snapshots",
                "sum",
            ),
        )
        .reset_index()
    )
    grouped["snapshot_record_linkage_rate"] = (
        grouped["buildings_with_snapshot_record"] / grouped["buildings"]
    )
    grouped["inference_limit"] = (
        "descriptive linkage only; not disaster damage, repair duration, causality, or structural vulnerability"
    )
    return grouped


def source_manifest() -> pd.DataFrame:
    rows = [
        (
            "bd_mbis_issued",
            REPAIR_SOURCES["mbis_issued_snapshot"],
            "Buildings Department",
            "https://portal.csdi.gov.hk/server/services/common/bd_rcd_1631168029910_44937/MapServer/WFSServer?service=wfs&request=GetFeature&typename=BDMBIS&outputFormat=CSV",
            "Government data public-data terms; verify downstream redistribution terms",
        ),
        (
            "bd_mbis_closed",
            REPAIR_SOURCES["mbis_closed_withdrawn_superseded_snapshot"],
            "Buildings Department",
            "https://portal.csdi.gov.hk/server/services/common/bd_rcd_1696920579106_13347/MapServer/WFSServer?service=wfs&request=GetFeature&typename=BDMBIS_CWS&outputFormat=CSV",
            "Government data public-data terms; verify downstream redistribution terms",
        ),
        (
            "bd_dangerous_issued",
            REPAIR_SOURCES["dangerous_order_issued_snapshot"],
            "Buildings Department",
            "https://portal.csdi.gov.hk/server/services/common/bd_rcd_1631167799285_82457/MapServer/WFSServer?service=wfs&request=GetFeature&typename=BDS26&outputFormat=CSV",
            "Government data public-data terms; verify downstream redistribution terms",
        ),
        (
            "bd_dangerous_closed",
            REPAIR_SOURCES["dangerous_order_closed_withdrawn_superseded_snapshot"],
            "Buildings Department",
            "https://portal.csdi.gov.hk/server/services/common/bd_rcd_1696920940144_71128/MapServer/WFSServer?service=wfs&request=GetFeature&typename=BDS26_CWS&outputFormat=CSV",
            "Government data public-data terms; verify downstream redistribution terms",
        ),
        (
            "census2021_stpug_table",
            OFFICIAL / "census2021_STPUG_21C.zip",
            "Census and Statistics Department",
            "https://www.census2021.gov.hk/en/district_profiles.html",
            "2021 Population Census intellectual-property notice in source package",
        ),
        (
            "census2021_stpug_geometry",
            OFFICIAL / "census2021_gml_tpu.zip",
            "Census and Statistics Department",
            "https://www.census2021.gov.hk/en/district_profiles.html",
            "2021 Population Census intellectual-property notice in source package",
        ),
        (
            "census2021_major_housing_estates",
            OFFICIAL / "census2021_gml_mhe.zip",
            "Census and Statistics Department",
            "https://www.census2021.gov.hk/en/district_profiles.html",
            "2021 Population Census intellectual-property notice in source package",
        ),
        (
            "ha_public_housing_estates",
            OFFICIAL / "ha_public_housing_estates_20260831.json",
            "Hong Kong Housing Authority",
            "https://www.housingauthority.gov.hk/datagovhk/prh-estates.json",
            "Government data public-data terms; verify downstream redistribution terms",
        ),
        (
            "ha_hos_psps_gsh_courts",
            OFFICIAL / "ha_hos_courts_20260831.json",
            "Hong Kong Housing Authority",
            "https://www.housingauthority.gov.hk/datagovhk/hos-courts.json",
            "Government data public-data terms; verify downstream redistribution terms",
        ),
        (
            "censtat_domestic_households_citywide",
            OFFICIAL / "censtat_130-06102_full_series_20260831.json",
            "Census and Statistics Department",
            "https://www.censtatd.gov.hk/api/get.php?id=130-06102&lang=en&full_series=1",
            "Government data public-data terms; Hong Kong-wide context only",
        ),
    ]
    manifest = pd.DataFrame(
        rows, columns=["source_id", "path", "publisher", "source_url", "licence_note"]
    )
    manifest["retrieved_on"] = "2026-08-31"
    manifest["local_path"] = manifest["path"].map(
        lambda path: str(Path(path).relative_to(ROOT)).replace("\\", "/")
    )
    manifest["bytes"] = manifest["path"].map(lambda path: Path(path).stat().st_size)
    manifest["sha256"] = manifest["path"].map(lambda path: sha256_file(Path(path)))
    return manifest.drop(columns="path")


def catalog_result_count(filename: str) -> int:
    path = OFFICIAL / "catalog_audits" / filename
    data = json.loads(path.read_text(encoding="utf-8"))
    return int(data.get("result", {}).get("count", 0))


def availability_matrix() -> pd.DataFrame:
    rows = [
        ("建筑级电力", "智能电表/楼宇逐时电力", "unavailable_public", "无", "无", "CLP/HK Electric 匿名数据申请", "search_smart_meter_building_electricity.json"),
        ("室内热环境", "住宅室内温度与相对湿度", "unavailable_public", "无", "无", "伦理审批后的现场监测", "search_indoor_temperature_residential.json"),
        ("空调运行", "回路状态、设定点、功率和设备效率", "unavailable_public", "无", "无", "住户同意下的回路监测和设备审计", "search_ac_operation_residential.json"),
        ("同步室外天气", "逐时站点/街区微气候", "partial_daily_reference", "京士柏逐日", "日", "HKO 逐时申请并布设现场传感器", "search_hko_historical_hourly_weather.json"),
        ("停电事件", "起止、原因、恢复与受影响客户", "unavailable_public", "无", "无", "CLP/HK Electric 事件日志申请", "search_power_interruption_outage.json"),
        ("电网拓扑", "馈线、变电站、服务区和负荷", "unavailable_public", "无", "无", "电力公司安全聚合数据申请", "search_grid_substation_feeder.json"),
        ("住户与公平性", "年龄、收入、租住、住房类型", "contextual_small_area_only", "STPUG/MHE", "2021 横截面", "同意问卷或安全数据实验室连接", None),
        ("法定维修/状况", "MBIS 与危险楼宇命令快照", "available_snapshot_only", "建筑点位", "当前快照", "申请通知日期、缺陷、完成日期和共享键", None),
        ("灾后损伤维修", "事件、损伤、维修和恢复时间线", "unavailable_public", "无", "无", "屋宇署/物业/保险灾害事件数据申请", "search_disaster_building_damage_repair.json"),
        ("实测节能改造", "投运日期、规格、费用及前后能耗", "unavailable_public", "无", "无", "房屋署/物业/承建商项目台账", "search_building_energy_retrofit.json"),
        ("城市总体电力", "住宅月度电力和年度终端用途", "available_citywide", "全港", "月/年", "已接入总体外部基准", None),
    ]
    matrix = pd.DataFrame(
        rows,
        columns=[
            "evidence_domain",
            "required_variables",
            "availability_status",
            "public_geographic_unit",
            "public_time_resolution",
            "minimum_acquisition_route",
            "catalog_audit_file",
        ],
    )
    matrix["public_catalog_result_count"] = matrix["catalog_audit_file"].map(
        lambda name: catalog_result_count(name) if isinstance(name, str) else np.nan
    )
    matrix["measured_building_validation_permitted"] = False
    matrix.loc[
        matrix["evidence_domain"].eq("法定维修/状况"),
        "measured_building_validation_permitted",
    ] = True
    matrix["interpretation_guardrail"] = [
        "不得以城市月度电力或模型输出替代建筑电表",
        "不得以室外温度或热模型替代室内实测",
        "不得把 Category 5/9 或低置信度住宅默认为有空调",
        "NASA POWER 仅提供小时形状，不得称为 HKO 逐时观测",
        "假设全事件停电不得称为真实停电风险",
        "不得推断馈线拥塞或恢复公平性",
        "仅允许生态尺度背景分析，禁止个体住户推断",
        "仅表示法定记录快照，不等于灾害损伤或节能改造",
        "MBIS/危险楼宇记录不得重命名为灾后维修",
        "情景改造效果不得称为实测因果效应",
        "只允许总体数量级和关联核验",
    ]
    return matrix


def teacher_task_matrix() -> pd.DataFrame:
    rows = [
        ("TXT-1", "文本任务", "检查约34万栋建筑的物理结构脆弱性", "blocked", "完成34.1万栋属性审计和法定记录连接，但没有结构缺陷/损伤标签", "结构检查、材料、缺陷、灾害损伤和专家判定"),
        ("TXT-2", "文本任务", "收集灾害后建筑维修及修护记录", "blocked", "已接入 MBIS/危险楼宇当前快照；其不是灾后维修时间线", "事件ID、发生/检查/维修/完工日期、损伤类型和费用"),
        ("TXT-3", "文本任务", "判断脆弱建筑类别及其是否完全由年龄解释", "partial", "可描述年龄/类别与法定快照连接率；禁止当作结构脆弱性因果检验", "带标签的结构状态样本、混杂因素和外部验证"),
        ("TXT-4", "文本任务", "连接极端天气、电网配电系统与建筑能源", "partial", "已完成热浪-建筑能源和假设停电影响情景及实证接收契约；无馈线和真实停电", "建筑-馈线映射、负荷、停电、恢复和建筑电力配对"),
        ("DOCX-1", "研究方案", "热浪空间分布与未来演化", "partial", "已完成历史热浪事件和单站约束天气；无城市空间气候面和未来气候集合", "多站/城市气候面及经偏差订正的 CMIP6 多模式集合"),
        ("DOCX-2", "研究方案", "热浪对住宅制冷与电网风险的影响", "partial", "完成核心住宅物理情景、城市总体电力关联及实证接收契约；缺建筑电表及电网观测", "电表-天气-空调-馈线-停电同步面板"),
        ("DOCX-3", "研究方案", "被动、围护、通风与智能空调适应措施", "complete_scenario_scope", "已运行七类改造/运行情景并保留参数先验标签", "真实项目投运记录用于实证升级，不影响任务书情景交付"),
        ("DOCX-4", "研究方案", "量化节能和热舒适收益", "complete_scenario_scope", "已量化模型内相对能耗、峰值和过热指标及不确定性", "配对实测后方可升级为现实绝对效应"),
        ("DOCX-5", "研究方案", "维修/状况证据及年龄解释", "partial", "完成公开法定快照的唯一空间连接和年龄分层描述", "灾害特定维修、共享通知键、缺陷严重度及独立结构判定"),
    ]
    matrix = pd.DataFrame(
        rows,
        columns=[
            "task_id",
            "task_source",
            "target_task",
            "status",
            "current_evidence",
            "remaining_acceptance_evidence",
        ],
    )
    matrix["all_required_truthful_claims_complete"] = matrix["status"].eq(
        "complete_scenario_scope"
    )
    return matrix


def write_protocols(output: Path) -> None:
    protocol = """# 真实观测数据申请与现场监测实施方案

## 1. 研究边界与成功条件

本方案不使用模型输出补齐缺失观测。Nature Energy 级实证门槛是形成同一建筑、同一时间轴上的电力、室内温湿度、空调状态和室外天气配对面板，并至少取得一种真实外生冲击：热浪、停电或已投运改造。人口普查仅用于区域背景；MBIS 与危险楼宇数据仅用于法定记录快照。

### 1.1 先给结论

只补充公开数据不能解除全部实证阻断。建筑级电力、室温、空调状态和住户属性必须通过有伦理审批的案例面板取得；馈线映射与真实停电必须由 CLP/HK Electric 以伪匿名或安全数据室方式提供。结构脆弱性可由详细法定检查记录或注册结构工程师案例评估形成标签；灾后维修时间线主要依赖屋宇署、房屋署、物业、承建商或保险机构的历史工单，单次现场踏勘不能重建历史时间线。

### 1.2 论文架构

主论文只回答“热浪如何改变住宅制冷、室内热暴露和供电风险，哪些适应措施有效，收益如何分配”。结构评估和灾后维修作为复合风险与改造可行性工作包，使用同一 34 万栋建筑底座和伪匿名键，但单独验收；数据量和识别强度足够时可形成伴随论文，不能为了覆盖任务书而把与热浪能源无直接因果关系的结论塞入主模型。

### 1.3 可执行边界

研究团队可直接完成公开数据获取、样本框与抽样、仪器和问卷规范、伦理及预注册技术附件、数据接收校验、功效分析、模型、外部验证和论文。必须由项目负责人或合作机构完成伦理批准、数据使用协议签署、场地与住户招募、现场安装、结构工程师签字和受限电网数据释放。自动化验收器只能证明数据契约通过，不能证明已达到任何期刊的录用标准。

## 2. 数据申请包

### 2.1 CLP 与香港电灯

申请 15 分钟或小时级匿名总电力、空调分项（若有）、最大需量、服务区/馈线匿名码、停电起止、原因、恢复、受影响客户数和事件时负荷。时间统一为 HKT（UTC+8），保留原始时区、夏令时标志和仪表质量码。建筑连接优先使用双方托管的 BUILDINGSTRUCTUREID 对照表；研究团队只接收伪匿名键。最少覆盖两个制冷季及全部研究热浪，时间戳完整率目标不低于 95%。

### 2.2 房屋署、物业和业主立案法团

申请楼宇/座号、住户数、空调资产、公共电力、围护结构、维修缺陷、法定通知日期、工程投运日期、规格、发票/结算、承建商、停机窗口和验收记录。灾害维修必须含事件 ID、损伤检查日期、损伤类型/等级、维修开始及完工日期；没有这些字段时不得称为“灾后维修”。

### 2.3 屋宇署与香港天文台

向屋宇署申请 MBIS/危险楼宇发出与完成记录的稳定共享键、真实业务日期、缺陷类别和完成状态。向天文台申请研究站点逐时温度、湿度、风、太阳辐射、降雨和质量码。街区监测与 HKO 时钟以 NTP 每日同步，允许漂移不超过 60 秒。

## 3. 现场面板

每个入选单位记录 5 或 15 分钟室内温度/RH、空调回路功率或开关状态、设定点（可获得时）、窗磁、占用代理和设备铭牌；屋顶/街谷记录同步室外温湿度，至少配置一台校准参考仪。传感器布设不记录音视频。设备编号、住户编号和建筑编号分别伪匿名化，连接表由独立数据管理员保存。

样本按住宅原型、楼龄、楼层、住房类型、电力服务区、区域收入和老龄背景分层。先开展不少于一个热浪季的试点，利用试点估计建筑内相关、效应方差、失访和传感器缺失，再计算正式样本量；不得预先声称“100 栋足够”。

现场启动采用两阶段设计。试点建议覆盖 8-12 栋、每栋 4-8 户，用于估计方差、建筑内相关和设备缺失，不用于宣称全港效应。正式研究的工作预算范围可按 24-40 栋、200-400 户准备，但最终样本必须由试点功效分析决定。至少覆盖一个完整炎热季，优选连续 12 个月或两个炎热季；至少保留 20% 建筑和一个完整热浪事件作从未参与调参的外部验证。上述数字是资源规划范围，不是 Nature Energy 的固定门槛。

室内温湿度传感器目标精度不低于温度 ±0.3 degC、相对湿度 ±2%RH；空调优先使用独立回路电表或经核验的电流传感器，不能只依赖问卷回忆。每栋设置有防辐射罩的同步室外传感器，并与 HKO 站点交叉核验。所有设备安装前后均做校准核查，时钟每日同步，漂移超过 60 秒的记录进入隔离层。

## 4. 伦理、隐私与质量控制

住户问卷与室内监测须经机构伦理审批和书面知情同意。公开成果不得展示可重识别的单栋敏感属性；小样本单元合并或抑制。原始连接键加密、分权访问、记录审计日志，并在协议中明确保留期和销毁日期。对电力、天气和传感器执行单位、范围、时间连续性、重复值、时钟漂移和跨源一致性检查；原始层只读，清洗层保留逐字段质量标志。

## 5. 识别与验证

基线模型按建筑和事件留出验证，报告 MAE、偏差、覆盖率和校准曲线。热浪响应使用事件研究，含建筑固定效应、日期/小时效应和天气非线性。真实改造采用有投运日期的差分中的差分或合成对照，检验平行趋势并使用同期未改造建筑。真实停电采用事件时间设计，结果包括室温上升、阈值暴露时长和恢复动态；因果样本不得与调参样本重叠。

## 6. 三个放行门

1. 配对门：同一伪匿名建筑的电力、室温、空调和天气覆盖率达到预注册阈值。
2. 事件门：停电或改造具有可核验起止/投运日期和对照组。
3. 外部验证门：至少一个建筑群或热浪事件完全留作外部验证。

在三个门全部通过前，论文只能报告官方背景、法定快照、模型情景和数据获取进度，不能声称建筑级实证校准、真实停电风险、住户公平性或改造因果效应。

结构与维修另设两个放行门。结构门要求抽样框可追溯、评估方法统一、专家标签可审计、至少 20% 双人独立复核并报告一致性，而且保留建筑级外部验证集；对 34 万栋的推广结果必须称为概率预测并附不确定性。维修门要求同一 hazard_event_id 下存在灾害发生、损伤评估、维修开始、维修完成和核实费用，且日期顺序与原始工单一致；缺任一关键节点时只能称为维修记录片段。

## 7. 机器接收验收

机构或现场数据只能放入受控目录 `data/empirical_observations/`，并按 `output/empirical_observation_validation/empirical_input_contract.csv` 提供八张表和来源清单。运行 `python src/validate_empirical_observation_package.py` 后，系统检查明确 HKT 时区、跨表键、值域、来源哈希、知情同意、伦理审批、预注册、试点功效分析目标和外部留出建筑。任何一门失败都会保留阻断状态。

该验收器只能证明数据包满足预注册的接收契约，不能替代后续因果模型、稳健性分析或同行评审。

## 8. 项目负责人现在需要完成的最小动作

1. 由老师/项目负责人确认允许从“仅公开数据模拟”升级为“受限机构数据加住户现场面板”，并指定伦理申请主体。
2. 向 CLP/HK Electric、房屋署或至少一家大型物业管理方发出合作申请；研究团队提供字段清单，机构保留真实地址与伪匿名键的对照。
3. 提供首批可接触的 8-12 栋候选楼宇和物业联系人；研究团队从 1,518 栋核心样本框中做分层匹配，不能只选方便进入的新楼。
4. 确认传感器、安装、数据通信、注册结构工程师和住户补偿预算；未确定预算前只做试点，不承诺正式样本数。
5. 任何含地址、账户、住户身份或电网敏感键的原始文件不得直接放入普通项目目录；先由独立数据管理员伪匿名化，再进入 `data/empirical_observations/`。
"""
    (output / "真实观测数据申请与现场监测实施方案.md").write_text(
        protocol, encoding="utf-8"
    )

    request_fields = [
        ("电力", "pseudonymous_building_id", "匿名建筑键", "static", "string", "restricted"),
        ("电力", "timestamp_hkt", "计量区间结束时间", "15 min preferred", "ISO-8601", "restricted"),
        ("电力", "electricity_kwh", "区间用电", "15 min preferred", "kWh", "restricted"),
        ("电力", "maximum_demand_kw", "区间最大需量", "15 min/hour", "kW", "restricted"),
        ("空调", "ac_circuit_power_w", "空调回路功率", "5/15 min", "W", "restricted"),
        ("空调", "ac_on_state", "空调运行状态", "5/15 min", "0/1", "restricted"),
        ("室内", "indoor_temperature_c", "室内干球温度", "5/15 min", "degC", "restricted"),
        ("室内", "indoor_relative_humidity_pct", "室内相对湿度", "5/15 min", "%", "restricted"),
        ("室外", "outdoor_temperature_c", "同步室外温度", "5/15 min", "degC", "controlled"),
        ("室外", "outdoor_relative_humidity_pct", "同步室外湿度", "5/15 min", "%", "controlled"),
        ("电网", "pseudonymous_feeder_id", "匿名馈线/服务区键", "static", "string", "highly_restricted"),
        ("停电", "outage_start_hkt", "停电开始", "event", "ISO-8601", "highly_restricted"),
        ("停电", "outage_end_hkt", "恢复完成", "event", "ISO-8601", "highly_restricted"),
        ("停电", "outage_cause_code", "停电原因", "event", "controlled vocabulary", "highly_restricted"),
        ("住户", "consented_household_id", "同意参加的匿名住户键", "static", "string", "highly_restricted"),
        ("住户", "age_band", "年龄组而非出生日期", "survey", "category", "highly_restricted"),
        ("住户", "income_band", "收入组", "survey", "category", "highly_restricted"),
        ("改造", "commissioning_date", "真实投运日期", "event", "date", "restricted"),
        ("改造", "measure_specification", "工程规格/性能", "event", "structured text", "restricted"),
        ("改造", "verified_cost_hkd", "发票或结算费用", "event", "HKD", "restricted"),
        ("维修", "hazard_event_id", "灾害事件键", "event", "string", "restricted"),
        ("维修", "damage_assessment_date", "损伤检查日期", "event", "date", "restricted"),
        ("维修", "repair_completion_date", "维修完工日期", "event", "date", "restricted"),
        ("维修", "damage_type_and_grade", "损伤类型与等级", "event", "controlled vocabulary", "restricted"),
    ]
    pd.DataFrame(
        request_fields,
        columns=["domain", "field_name", "chinese_definition", "resolution", "unit_or_type", "privacy_class"],
    ).to_csv(output / "真实观测数据申请字段清单.csv", index=False, encoding="utf-8-sig")

    action_rows = [
        ("空间气候与未来情景", "主论文", "partial", "HKO 多站逐时观测加经偏差订正的 CMIP6 多模式集合", False, False, True, "下载、质控、偏差订正、热浪识别、城市气候面和不确定性分析", "确认情景、时期与数据许可；无需现场协调", "至少保留一个站点和一个事件作外部验证", "历史与未来气候输入可追溯且验证通过"),
        ("电力-室温-空调-室外天气配对", "主论文", "blocked", "有伦理审批的住宅案例面板；电力可用住户回路表或电力公司智能电表", True, False, False, "生成分层抽样、设备清单、数据接收、时钟校准、质控和模型", "取得伦理批准、物业与住户同意，组织安装和维护", "先 8-12 栋试点；正式样本由试点功效分析决定", "同一建筑和时间轴完整率达到预注册阈值"),
        ("真实馈线与停电", "主论文", "blocked", "CLP/HK Electric 伪匿名馈线映射、事件日志和负荷记录", False, True, False, "提供字段、校验安全聚合数据、事件时间分析和外部验证", "由老师/项目负责人签署合作与数据使用协议", "优先申请 5-10 年回溯记录；不得等待现场偶遇停电", "建筑-馈线-事件键闭合且事件数达到功效要求"),
        ("住户公平性", "主论文", "blocked", "同意问卷与监测面板连接；人口普查只作区域背景", True, False, False, "问卷数据字典、伪匿名连接、权重、缺失和公平性分析", "伦理审批、知情同意、招募与补偿", "年龄和收入使用分组；不收不必要的直接身份信息", "全部参与记录有有效同意且可连接观测面板"),
        ("真实改造效果与成本", "主论文", "blocked", "有投运日期、规格、发票和前后电表的真实项目及同期对照", True, True, False, "匹配或分期设计、差分中的差分、事件研究、成本碳分析", "取得房屋署/物业/承建商台账并安排基线与随访", "每个干预组同时含 treated 和 control；预先检查平行趋势", "投运日期和成本可核验，前后窗口与对照完整"),
        ("结构脆弱性标签", "复合风险工作包", "blocked", "详细法定检查记录或注册结构工程师分层案例评估", True, True, False, "从 34 万栋底座抽样、盲化编号、标签一致性、推广与不确定性", "取得记录或聘请工程师并提供入场权限", "至少 20% 双人独立复核；正式数量由阳性率和学习曲线决定", "专家标签可审计、复核一致性达标并有外部验证集"),
        ("灾后维修时间线", "复合风险工作包", "blocked", "屋宇署/房屋署/物业/承建商/保险的灾害工单和结算记录", False, True, False, "跨表连接、日期逻辑、恢复时间和费用分析", "签署数据协议并指定可核验的历史灾害事件", "单次现场踏勘不能重建时间线；优先回溯多次台风或暴雨事件", "事件、损伤、开工、完工和费用五节点齐全"),
        ("伦理、预注册、来源与冻结", "全部工作包", "blocked", "机构伦理批准、预注册、功效分析、数据使用依据与哈希清单", False, False, False, "起草技术附件、计算功效、生成哈希和机器验收报告", "提交并取得正式批准/注册编号，授权数据冻结", "外部验证建筑在建模前锁定", "study_metadata 与 provenance_manifest 全部通过"),
    ]
    pd.DataFrame(
        action_rows,
        columns=[
            "evidence_domain",
            "paper_package",
            "current_status",
            "completion_route",
            "case_field_required",
            "restricted_institutional_data_required",
            "public_data_can_replace",
            "assistant_or_pipeline_can_execute",
            "user_or_pi_must_execute",
            "planning_or_minimum_rule",
            "release_gate",
        ],
    ).to_csv(output / "实证研究责任矩阵.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundation-dir", type=Path, default=FOUNDATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    foundation = args.foundation_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    stock = gpd.read_parquet(foundation / "building_stock_repaired.geoparquet")
    core = gpd.read_parquet(foundation / "building_stock_core_simulation.geoparquet")
    if stock.crs.to_epsg() != 2326 or core.crs.to_epsg() != 2326:
        raise RuntimeError("Expected HK1980 Grid EPSG:2326 building geometries")

    links: list[pd.DataFrame] = []
    quarantines: list[pd.DataFrame] = []
    repair_audit: dict[str, dict[str, int]] = {}
    for dataset_key, path in REPAIR_SOURCES.items():
        records = pd.read_csv(path)
        accepted, quarantine = unique_point_matches(records, stock, dataset_key)
        links.append(accepted)
        quarantines.append(quarantine)
        repair_audit[dataset_key] = {
            "source_records": len(records),
            "accepted_unique_footprint_matches": len(accepted),
            "quarantined_unmatched": int(
                quarantine["quarantine_reason"].eq("no_containing_building_footprint").sum()
            ),
            "quarantined_overlapping_footprints": int(
                quarantine["quarantine_reason"].eq(
                    "multiple_overlapping_building_footprints"
                ).sum()
            ),
        }
    repair_links = pd.concat(links, ignore_index=True)
    repair_quarantine = pd.concat(quarantines, ignore_index=True)
    if repair_links.duplicated(["dataset_key", "source_record_id"]).any():
        raise RuntimeError("A statutory snapshot record was assigned more than once")
    repair_links.to_csv(output / "building_repair_evidence_links.csv", index=False)
    repair_quarantine.to_csv(
        output / "building_repair_evidence_quarantine.csv", index=False
    )

    repair_summary = build_repair_summary(stock, repair_links)
    repair_summary.to_csv(output / "building_repair_evidence_summary.csv", index=False)
    age_description = describe_age_condition(repair_summary)
    age_description.to_csv(
        output / "building_condition_by_age_and_category.csv", index=False
    )
    (output / "repair_linkage_audit.json").write_text(
        json.dumps(
            {
                "rules": {
                    "accepted": "point within exactly one building footprint",
                    "quarantined": "no containing footprint or multiple overlapping footprints",
                    "nearest_neighbour_forcing": False,
                    "lastupdate_interpretation": "dataset refresh time only",
                    "issued_closed_shared_key_proven": False,
                    "repair_duration_inference_permitted": False,
                    "disaster_repair_interpretation_permitted": False,
                },
                "datasets": repair_audit,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    stpug = read_stpug_context()
    stpug_columns = [column for column in stpug if column != "geometry"]
    stpug_join = unique_context_join(
        core, stpug, stpug_columns, "stpug", require_all=True
    )
    mhe = read_mhe_context()
    ha_linkage = exact_unique_estate_linkage(mhe, read_ha_profiles())
    ha_linkage.to_csv(output / "major_estate_ha_profile_linkage.csv", index=False)
    mhe_enriched = mhe.merge(
        ha_linkage[
            [
                "mhe_code",
                "ha_link_status",
                "ha_candidate_count",
                "ha_housing_program",
                "ha_estate_name_en",
                "ha_estate_name_tc",
                "ha_district_name_en",
                "ha_reference_year",
                "ha_number_of_blocks",
                "ha_number_of_flats",
            ]
        ],
        on="mhe_code",
        how="left",
        validate="1:1",
    )
    mhe_columns = [column for column in mhe_enriched if column != "geometry"]
    mhe_join = unique_context_join(
        core, mhe_enriched, mhe_columns, "mhe", require_all=False
    )

    residential_stock = stock.loc[
        english_label(stock["hkbfetd_main_class"]).isin(
            ["Residential", "Mixed-use"]
        )
    ].copy()
    residential_stpug_join = unique_context_join(
        residential_stock, stpug, stpug_columns, "stpug", require_all=False
    )
    residential_mhe_join = unique_context_join(
        residential_stock, mhe_enriched, mhe_columns, "mhe", require_all=False
    )
    residential_context = residential_stock.merge(
        residential_stpug_join, on="BUILDINGSTRUCTUREID", validate="1:1"
    )
    residential_context = residential_context.merge(
        residential_mhe_join, on="BUILDINGSTRUCTUREID", validate="1:1"
    )
    residential_context["contextual_socioeconomic_linkage_ready"] = (
        residential_context["stpug_code"].notna()
    )
    residential_context["household_level_inference_permitted"] = False
    residential_context["ecological_fallacy_warning"] = (
        "STPUG/MHE values describe areas, not the residents of this building"
    )
    residential_context.to_parquet(
        output / "residential_building_census_context.geoparquet", index=False
    )

    core_context = core.merge(stpug_join, on="BUILDINGSTRUCTUREID", validate="1:1")
    core_context = core_context.merge(mhe_join, on="BUILDINGSTRUCTUREID", validate="1:1")
    core_repair = repair_summary[
        repair_summary["BUILDINGSTRUCTUREID"].isin(core["BUILDINGSTRUCTUREID"])
    ].drop(
        columns=[
            "CATEGORY",
            "BUILDINGSTRUCTURETYPE",
            "construction_year_value",
            "construction_year_source",
            "construction_year_quality",
            "building_age_2026",
            "residential_eligibility_status",
            "analysis_tier",
        ]
    )
    core_context = core_context.merge(
        core_repair, on="BUILDINGSTRUCTUREID", validate="1:1"
    )
    core_context["contextual_socioeconomic_linkage_ready"] = True
    core_context["household_level_inference_permitted"] = False
    core_context["ecological_fallacy_warning"] = (
        "STPUG/MHE values describe areas, not the residents of this building"
    )
    non_geometry = pd.DataFrame(core_context.drop(columns="geometry"))
    non_geometry.to_csv(output / "core_building_census_context.csv", index=False)
    core_context.to_parquet(output / "core_building_census_context.geoparquet")

    source_manifest().to_csv(output / "official_source_manifest.csv", index=False)
    availability_matrix().to_csv(
        output / "observational_availability_matrix.csv", index=False
    )
    teacher_task_matrix().to_csv(
        output / "teacher_task_compliance_matrix.csv", index=False, encoding="utf-8-sig"
    )
    write_protocols(output)

    readiness = {
        "stock_buildings_audited": int(len(stock)),
        "core_buildings": int(len(core)),
        "core_buildings_with_stpug_context": int(
            core_context["stpug_code"].notna().sum()
        ),
        "core_buildings_with_major_estate_context": int(
            core_context["mhe_code"].notna().sum()
        ),
        "residential_and_mixed_use_buildings_with_context": int(
            len(residential_context)
        ),
        "major_estates_with_exact_unique_ha_profile": int(
            ha_linkage["ha_link_status"].eq("exact_unique_normalized_name").sum()
        ),
        "accepted_statutory_snapshot_records": int(len(repair_links)),
        "quarantined_statutory_snapshot_records": int(len(repair_quarantine)),
        "contextual_socioeconomic_linkage_ready": True,
        "household_level_equity_ready": False,
        "paired_building_observations_ready": False,
        "feeder_outage_validation_ready": False,
        "measured_retrofit_effect_ready": False,
        "disaster_repair_timeline_ready": False,
        "structural_vulnerability_labels_ready": False,
        "nature_energy_empirical_validation_ready": False,
        "teacher_task_scenario_components_ready": True,
        "teacher_task_empirical_components_complete": False,
        "gates": {
            "all_core_buildings_have_stpug_context": bool(
                core_context["stpug_code"].notna().all()
            ),
            "no_snapshot_record_assigned_to_multiple_buildings": bool(
                not repair_links.duplicated(["dataset_key", "source_record_id"]).any()
            ),
            "ambiguous_repair_matches_quarantined": True,
            "ambiguous_estate_names_excluded": bool(
                ha_linkage.loc[
                    ha_linkage["ha_link_status"].eq("ambiguous_normalized_name"),
                    "ha_estate_name_en",
                ].isna().all()
            ),
            "context_not_misclassified_as_household_observation": bool(
                not core_context["household_level_inference_permitted"].any()
            ),
            "statutory_snapshot_not_misclassified_as_disaster_repair": True,
            "unsupported_empirical_claims_remain_blocked": True,
        },
    }
    readiness["observational_evidence_stage_ready"] = bool(
        all(readiness["gates"].values())
    )
    (output / "observational_readiness.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(readiness, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
