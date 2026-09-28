"""Label-agnostic building separation metrics.

Predicted labels are compared with annotated building instances without
requiring the same identifiers, so the scores measure *separation*: whether
one real building ends up under one label, and one label under one building.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Ignore slivers when counting merge/split events.
EVENT_MINIMUM_PIXELS = 500
EVENT_MINIMUM_SHARE = 0.2


def _contingency(truth: np.ndarray, predicted: np.ndarray, region: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth_values, truth_index = np.unique(truth[region], return_inverse=True)
    predicted_values, predicted_index = np.unique(predicted[region], return_inverse=True)
    table = np.zeros((len(truth_values), len(predicted_values)), dtype=np.int64)
    np.add.at(table, (truth_index, predicted_index), 1)
    return table, truth_values, predicted_values


def separation_scores(truth: np.ndarray, predicted: np.ndarray, region: np.ndarray | None = None) -> dict[str, float]:
    """Scores for one panorama.

    ``truth``: annotated instance ids (0 = no building, -1 = ignored).
    ``predicted``: predicted labels (0 = none). ``region``: pixels the
    annotation covers exhaustively; defaults to the whole panorama.
    """
    region = np.ones(truth.shape, dtype=bool) if region is None else region
    region = region & (truth >= 0)
    table, truth_values, predicted_values = _contingency(truth, predicted, region)
    buildings = truth_values > 0
    labels = predicted_values > 0
    building_table = table[buildings][:, labels]
    building_sizes = table[buildings].sum(axis=1)
    # Completeness: share of each building under its single dominant label.
    dominant = building_table.max(axis=1, initial=0)
    completeness = float(dominant.sum() / max(building_sizes.sum(), 1))
    detected = float(building_table.sum() / max(building_sizes.sum(), 1))
    # Purity: share of each label's building pixels inside one building.
    label_on_buildings = building_table.sum(axis=0)
    purity = float(building_table.max(axis=0, initial=0).sum() / max(label_on_buildings.sum(), 1))
    label_sizes = table[:, labels].sum(axis=0)
    background = table[~buildings][:, labels].sum()
    leak = float(background / max(label_sizes.sum(), 1))

    def events(matrix: np.ndarray) -> int:
        count = 0
        for row in matrix:
            ordered = np.sort(row)[::-1]
            if len(ordered) > 1 and ordered[1] >= EVENT_MINIMUM_PIXELS and ordered[1] >= EVENT_MINIMUM_SHARE * row.sum():
                count += 1
        return count

    top_detected = _top_detected(truth, predicted, region)
    merges = events(building_table.T)  # a label spanning two buildings
    splits = events(building_table)  # a building spread over two labels
    f_score = 2 * purity * completeness / max(purity + completeness, 1e-9)
    return {
        "completeness": completeness, "purity": purity, "separation_f": f_score,
        "detected": detected, "top_detected": top_detected, "leak": leak,
        "merges": merges, "splits": splits, "buildings": int(buildings.sum()),
    }


TOP_BAND = 0.25


def _top_detected(truth: np.ndarray, predicted: np.ndarray, region: np.ndarray) -> float:
    """Share of each building's upper quarter (per column) that carries a label.

    Catches masks that stop below the roof line, which whole-building recall
    hides because the tall lower floors dominate the pixel count.
    """
    rows = np.arange(truth.shape[0])[:, None]
    covered = total = 0
    for building in np.unique(truth[region & (truth > 0)]):
        mask = (truth == building) & region
        columns = mask.any(axis=0)
        top = np.where(mask, rows, truth.shape[0]).min(axis=0)
        bottom = np.where(mask, rows, -1).max(axis=0)
        band = mask & (rows <= top + TOP_BAND * (bottom - top)) & columns
        covered += int((band & (predicted > 0)).sum())
        total += int(band.sum())
    return covered / max(total, 1)


def summarize(results: list[dict[str, Any]]) -> dict[str, float]:
    if not results:
        return {}
    summary: dict[str, float] = {}
    for key in ("completeness", "purity", "separation_f", "detected", "top_detected", "leak"):
        summary[key] = round(float(np.mean([result[key] for result in results])), 4)
    for key in ("merges", "splits", "buildings"):
        summary[key] = int(sum(result[key] for result in results))
    return summary
