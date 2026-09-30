"""Shared building-composition calculations for result aggregation and maps."""

from __future__ import annotations

import numpy as np
import pandas as pd


FLOOR_ROLES = ("single", "bottom", "middle", "top")


def floor_role_weight_table(floors: pd.Series) -> pd.DataFrame:
    """Return equal-floor-area weights for simulated thermal positions."""
    values = pd.to_numeric(floors, errors="coerce")
    if values.isna().any() or values.lt(1).any():
        raise ValueError("Floor-role composition requires finite floor counts of at least one")
    rounded = values.round().astype("int64")
    if not np.allclose(values.to_numpy(dtype=float), rounded.to_numpy(dtype=float)):
        raise ValueError("Floor-role composition requires integer-valued floor counts")

    weights = pd.DataFrame(0.0, index=floors.index, columns=FLOOR_ROLES)
    one = rounded.eq(1)
    multi = rounded.ge(2)
    weights.loc[one, "single"] = 1.0
    weights.loc[multi, "bottom"] = 1.0 / rounded.loc[multi]
    weights.loc[multi, "top"] = 1.0 / rounded.loc[multi]
    middle = rounded.ge(3)
    weights.loc[middle, "middle"] = (rounded.loc[middle] - 2.0) / rounded.loc[middle]
    if not np.allclose(weights.sum(axis=1).to_numpy(dtype=float), 1.0):
        raise ValueError("Floor-role weights do not conserve whole-building floor area")
    return weights
