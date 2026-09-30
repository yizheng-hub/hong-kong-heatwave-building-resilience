from __future__ import annotations

import numpy as np
import pandas as pd


def normalized_absolute_error(observed, estimated) -> float:
    observed = np.asarray(observed, dtype=float)
    estimated = np.asarray(estimated, dtype=float)
    return 100.0 * np.abs(estimated - observed).sum() / observed.sum()


def normalized_root_square_error(observed, estimated) -> float:
    observed = np.asarray(observed, dtype=float)
    estimated = np.asarray(estimated, dtype=float)
    return 100.0 * np.sqrt(
        np.square(estimated - observed).sum() / np.square(observed).sum()
    )


def spearman_rank_correlation(observed, estimated) -> float:
    observed_rank = pd.Series(observed).rank(method="average")
    estimated_rank = pd.Series(estimated).rank(method="average")
    return float(observed_rank.corr(estimated_rank, method="pearson"))


def tie_aware_selection_probability(scores, fraction: float, *, lowest: bool = True):
    scores = pd.Series(scores, dtype=float)
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1].")
    valid = scores.notna()
    target = fraction * valid.sum()
    order = scores[valid].sort_values(ascending=lowest)
    probabilities = pd.Series(0.0, index=scores.index)
    selected = 0.0
    for value, members in order.groupby(order, sort=False).groups.items():
        member_index = list(members)
        remaining = target - selected
        if remaining <= 0:
            break
        probability = min(1.0, remaining / len(member_index))
        probabilities.loc[member_index] = probability
        selected += probability * len(member_index)
    return probabilities


def tie_aware_capture(scores, burden, fraction: float, *, lowest: bool = True) -> float:
    burden = pd.Series(burden, dtype=float).fillna(0.0)
    denominator = burden.sum()
    if denominator <= 0:
        return float("nan")
    probability = tie_aware_selection_probability(scores, fraction, lowest=lowest)
    return float((probability * burden).sum() / denominator)
