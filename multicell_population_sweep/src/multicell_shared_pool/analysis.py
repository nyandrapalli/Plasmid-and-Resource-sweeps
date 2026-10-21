from __future__ import annotations

from typing import Mapping

import numpy as np


def _basic_metrics(values: np.ndarray) -> dict[str, float | int]:
    flat = np.asarray(values, dtype=float).ravel()
    flat = flat[np.isfinite(flat)]
    count = int(flat.size)
    if count == 0:
        return {
            "mean": np.nan,
            "sd": np.nan,
            "variance": np.nan,
            "cv": np.nan,
            "cv2": np.nan,
            "fano": np.nan,
            "n": 0,
        }
    mean = float(np.mean(flat))
    variance = float(np.var(flat, ddof=1)) if count > 1 else np.nan
    sd = float(np.sqrt(variance)) if np.isfinite(variance) else np.nan
    if mean == 0 or not np.isfinite(variance):
        cv = np.nan
        cv2 = np.nan
        fano = np.nan
    else:
        cv = sd / mean
        cv2 = variance / (mean * mean)
        fano = variance / mean
    return {
        "mean": mean,
        "sd": sd,
        "variance": variance,
        "cv": cv,
        "cv2": cv2,
        "fano": fano,
        "n": count,
    }


def _windowed_valid_values(
    values: np.ndarray,
    times: np.ndarray,
    window: tuple[int, int],
    valid_replicates: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    times = np.asarray(times)
    valid_replicates = np.asarray(valid_replicates, dtype=bool)
    if values.ndim != 3:
        raise ValueError("reporter values must have shape [replicate, cell, time]")
    if values.shape[0] != len(valid_replicates) or values.shape[2] != len(times):
        raise ValueError("reporter values, replicate mask, and times are incompatible")
    window_mask = (times >= window[0]) & (times <= window[1])
    if not np.any(window_mask):
        raise ValueError("analysis window contains no sampled times")
    return values[valid_replicates][:, :, window_mask]


def _mean_cell_correlation(block: np.ndarray) -> float:
    if block.shape[0] == 0 or block.shape[1] < 2:
        return np.nan
    correlations: list[float] = []
    for replicate in block:
        variable_cells = np.std(replicate, axis=1) > 0
        cell_observations = replicate[variable_cells]
        if cell_observations.shape[0] < 2:
            continue
        correlation = np.corrcoef(cell_observations)
        upper = correlation[np.triu_indices(cell_observations.shape[0], k=1)]
        correlations.extend(upper[np.isfinite(upper)].tolist())
    return float(np.mean(correlations)) if correlations else np.nan


def summarize_reporter(
    values: np.ndarray,
    times: np.ndarray,
    window: tuple[int, int],
    valid_replicates: np.ndarray,
) -> dict[str, float | int]:
    block = _windowed_valid_values(values, times, window, valid_replicates)
    single_cell = _basic_metrics(block)
    population_means = np.mean(block, axis=1)
    population = _basic_metrics(population_means)

    within_cv2: list[float] = []
    if block.shape[1] > 1:
        for replicate_index in range(block.shape[0]):
            for time_index in range(block.shape[2]):
                cell_values = block[replicate_index, :, time_index]
                mean = float(np.mean(cell_values))
                if mean > 0:
                    variance = float(np.var(cell_values, ddof=1))
                    within_cv2.append(variance / (mean * mean))

    output: dict[str, float | int] = {"valid_replicates": int(block.shape[0])}
    for prefix, metrics in (("single_cell", single_cell), ("population_mean", population)):
        for key, value in metrics.items():
            output[f"{prefix}_{key}"] = value
    output["within_population_cv2_mean"] = (
        float(np.mean(within_cv2)) if within_cv2 else np.nan
    )
    output["within_population_cv2_sd"] = (
        float(np.std(within_cv2, ddof=1)) if len(within_cv2) > 1 else np.nan
    )
    output["mean_cell_correlation"] = _mean_cell_correlation(block)
    return output


def cross_reporter_population_correlation(
    first: np.ndarray,
    second: np.ndarray,
    times: np.ndarray,
    window: tuple[int, int],
    valid_replicates: np.ndarray,
) -> float:
    first_block = _windowed_valid_values(first, times, window, valid_replicates)
    second_block = _windowed_valid_values(second, times, window, valid_replicates)
    if first_block.shape[0] == 0 or second_block.shape[0] == 0:
        return np.nan
    correlations: list[float] = []
    first_means = np.mean(first_block, axis=1)
    second_means = np.mean(second_block, axis=1)
    for first_mean, second_mean in zip(first_means, second_means):
        if first_mean.size < 2 or np.std(first_mean) == 0 or np.std(second_mean) == 0:
            continue
        correlations.append(float(np.corrcoef(first_mean, second_mean)[0, 1]))
    return float(np.mean(correlations)) if correlations else np.nan


def summarize_condition(
    reporter_data: Mapping[str, np.ndarray],
    reporter_labels: Mapping[str, str],
    times: np.ndarray,
    window: tuple[int, int],
    valid_replicates: np.ndarray,
) -> list[dict[str, float | int | str]]:
    reporter_keys = list(reporter_data)
    cross_correlation = np.nan
    if len(reporter_keys) == 2:
        cross_correlation = cross_reporter_population_correlation(
            reporter_data[reporter_keys[0]],
            reporter_data[reporter_keys[1]],
            times,
            window,
            valid_replicates,
        )

    rows: list[dict[str, float | int | str]] = []
    for reporter_key in reporter_keys:
        row: dict[str, float | int | str] = {
            "reporter_key": reporter_key,
            "reporter_label": reporter_labels[reporter_key],
            "cross_reporter_population_correlation": cross_correlation,
        }
        row.update(
            summarize_reporter(
                reporter_data[reporter_key],
                times,
                window,
                valid_replicates,
            )
        )
        rows.append(row)
    return rows
