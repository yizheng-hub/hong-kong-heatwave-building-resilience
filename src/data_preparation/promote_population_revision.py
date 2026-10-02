"""Promote a verified population-aligned analysis into notebook inputs.

This command does not run EnergyPlus.  It copies the completed building-event
tables, regenerates population-dependent compact summaries and panel data, and
leaves all population-independent figure sources unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd


EVENTS = (
    "HKO_LI_0049", "HKO_LI_0052", "HKO_LI_0054", "HKO_LI_0058",
    "HKO_LI_0066", "HKO_LI_0068", "HKO_LI_0085", "HKO_LI_0090",
)
YEARS = dict(zip(EVENTS, (2005, 2009, 2012, 2014, 2017, 2018, 2022, 2023)))
OUTCOMES = (
    "cooling_shortfall_kwh", "absolute_dh32", "extra_dh26",
    "population_degree_hours", "recovery_hours",
)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".writing")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def main(argv: list[str] | None = None) -> int:
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path,
        default=project / "result/population_revision_run",
    )
    args = parser.parse_args(argv)
    source = args.source_root.resolve()

    audit_path = source / "00_protocol/solver_free_analysis_audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(f"completed solver-free audit is missing: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "complete" or audit.get("energyplus_calls") != 0:
        raise ValueError("source analysis is not a completed zero-EnergyPlus run")

    event_destination = project / "result/building_event_outcomes"
    for event in EVENTS:
        event_source = source / f"03_stock/events/{event}.parquet"
        if not event_source.is_file():
            raise FileNotFoundError(event_source)
        atomic_copy(event_source, event_destination / event_source.name)

    component_source = source / "03_stock/building_component_map.parquet"
    atomic_copy(
        component_source,
        project / "intermediate/prepared_stock/building_component_map.parquet",
    )

    validation_source = source / "04_validation"
    compact_validation = project / "result/validation"
    for name in ("event_holdout_screening_capture.csv", "event_holdout_rank_metrics.csv"):
        atomic_copy(validation_source / name, compact_validation / name)

    analysis_destination = project / "result/analysis"
    old_class = pd.read_csv(analysis_destination / "building_event_class_outcomes.csv")
    population_rows: list[dict] = []
    for event in EVENTS:
        frame = pd.read_parquet(
            source / f"03_stock/events/{event}.parquet",
            columns=[
                "main_class", "absolute_dh32", "population_reference_2021",
                "population_degree_hours",
            ],
        )
        for main_class, group in frame.groupby("main_class"):
            population_rows.append({
                "event_id": event,
                "main_class": main_class,
                "population_in_buildings_dh32_gt0": float(
                    group.loc[group.absolute_dh32.gt(0), "population_reference_2021"].sum()
                ),
                "population_degree_hours": float(group.population_degree_hours.sum()),
            })
    replacement = pd.DataFrame(population_rows)
    class_summary = old_class.drop(
        columns=["population_in_buildings_dh32_gt0", "population_degree_hours"]
    ).merge(replacement, on=["event_id", "main_class"], validate="one_to_one")
    atomic_csv(class_summary, analysis_destination / "building_event_class_outcomes.csv")

    lower_rank = pd.read_csv(validation_source / "event_holdout_rank_metrics.csv")
    lower_rank = lower_rank[
        lower_rank.fold_id.astype(str).str.startswith("LOEO_")
        & lower_rank.score_endpoint_scenario.eq("lower")
    ]
    rank_summary = lower_rank.groupby(["method", "outcome"]).agg(
        folds=("fold_id", "size"),
        median_spearman=("spearman", "median"),
        min_spearman=("spearman", "min"),
        max_spearman=("spearman", "max"),
        median_kendall_tau_b=("kendall_tau_b", "median"),
    ).reset_index()
    atomic_csv(rank_summary, analysis_destination / "loeo8_lower_rank_summary.csv")

    first_rank = pd.read_csv(validation_source / "lockbox_first_look_rank.csv")
    first_rank_summary = first_rank.groupby(["method", "outcome"]).agg(
        comparisons=("fold_id", "size"),
        median_spearman=("spearman", "median"),
        min_spearman=("spearman", "min"),
        max_spearman=("spearman", "max"),
    ).reset_index()
    atomic_csv(first_rank_summary, analysis_destination / "first_look_rank_summary.csv")

    first_capture = pd.read_csv(validation_source / "lockbox_first_look_capture.csv")
    first_capture = first_capture[first_capture.fraction.eq(0.10)]
    first_capture_summary = first_capture.groupby(["method", "outcome"]).agg(
        comparisons=("fold_id", "size"),
        median_capture=("expected_burden_capture", "median"),
        min_capture=("expected_burden_capture", "min"),
        max_capture=("expected_burden_capture", "max"),
    ).reset_index()
    atomic_csv(
        first_capture_summary,
        analysis_destination / "first_look_top10_capture_summary.csv",
    )

    figure_data = project / "result/figure_source_data"
    estate_source = (
        project
        / "intermediate/verification/clean_chain/model_inputs/major_housing_estate_population_validation.csv"
    )
    atomic_copy(estate_source, figure_data / "Figure2_estate_population.csv")

    featured = pd.read_parquet(
        source / "03_stock/events/HKO_LI_0085.parquet",
        columns=["BUILDINGSTRUCTUREID", "population_degree_hours"],
    )
    figure3_path = figure_data / "Figure3_2022_territory_buildings.csv"
    figure3 = pd.read_csv(figure3_path).drop(columns=["population_degree_hours"])
    figure3 = figure3.merge(featured, on="BUILDINGSTRUCTUREID", validate="one_to_one")
    atomic_csv(figure3, figure3_path)

    figure5_class = class_summary.copy()
    figure5_class["year"] = figure5_class.event_id.map(YEARS)
    atomic_csv(figure5_class, figure_data / "Figure5_class.csv")
    atomic_csv(rank_summary, figure_data / "Figure5_rank.csv")

    capture = pd.read_csv(validation_source / "event_holdout_screening_capture.csv")
    loeo_lower = capture[
        capture.fold_id.astype(str).str.startswith("LOEO_")
        & capture.score_endpoint_scenario.eq("lower")
    ]
    top10 = loeo_lower[loeo_lower.fraction.eq(0.10)]
    capture_summary = top10.groupby(["method", "outcome"]).agg(
        folds=("fold_id", "size"),
        median_top10_capture=("expected_burden_capture", "median"),
        min_top10_capture=("expected_burden_capture", "min"),
        max_top10_capture=("expected_burden_capture", "max"),
    ).reset_index()
    atomic_csv(capture_summary, figure_data / "Figure5_capture.csv")

    sensitivity_methods = {"BERI", "detailed_use_floor", "stage_C", "stage_O", "stage_R"}
    sensitivity = loeo_lower[
        loeo_lower.method.isin(sensitivity_methods)
        & loeo_lower.outcome.isin(OUTCOMES)
    ].groupby(["outcome", "fraction", "method"]).expected_burden_capture.agg(
        median="median", min="min", max="max"
    ).reset_index()
    atomic_csv(
        sensitivity,
        figure_data / "Supplementary_Figure_S16_LOEO8_lower_fraction_sensitivity.csv",
    )

    report = {
        "source_root": source.relative_to(project).as_posix(),
        "energyplus_calls": 0,
        "population_field": "allocated_population__ssug_landuse_residential_gfa",
        "event_tables": len(EVENTS),
        "population_total": float(
            pd.read_parquet(
                source / "03_stock/fixed_stock_buildings.parquet",
                columns=["population_reference_2021"],
            ).population_reference_2021.sum()
        ),
        "updated_figures": [
            "Figure 2", "Figure 3", "Figure 5", "Supplementary Figure S2",
            "Supplementary Figure S5", "Supplementary Figure S11",
            "Supplementary Figure S16",
        ],
    }
    report_path = project / "result/validation/population_revision_promotion.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
