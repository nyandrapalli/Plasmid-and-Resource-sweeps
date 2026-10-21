from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from multicell_shared_pool.config import SimulationConfig
from multicell_shared_pool.models.base import PopulationModel


@dataclass
class TrajectoryResult:
    times: np.ndarray
    reporters: dict[str, np.ndarray]
    final_time: float
    event_count: int
    hit_event_cap: bool


def _write_sample(storage: dict[str, np.ndarray], values: Mapping[str, np.ndarray], index: int) -> None:
    for reporter, reporter_values in values.items():
        storage[reporter][:, index] = reporter_values


def run_trajectory(
    model: PopulationModel,
    config: SimulationConfig,
    seed: int,
) -> TrajectoryResult:
    rng = np.random.default_rng(seed)
    state = model.initial_state(config)
    model.validate(state)

    times = config.sample_times
    initial_reporters = model.reporters(state)
    reporter_storage = {
        name: np.full((len(values), len(times)), -1, dtype=np.int64)
        for name, values in initial_reporters.items()
    }
    _write_sample(reporter_storage, initial_reporters, 0)

    time = 0.0
    event_count = 0
    sample_index = 1

    while time < config.t_end and event_count < config.max_events:
        propensities, reaction_keys = model.reactions(state)
        propensities = np.asarray(propensities, dtype=float)
        if len(propensities) != len(reaction_keys):
            raise ValueError("propensity and reaction-key counts differ")
        if np.any(propensities < 0) or not np.all(np.isfinite(propensities)):
            raise ValueError("propensities must be finite and non-negative")

        total_propensity = float(propensities.sum())
        if total_propensity <= 0:
            while sample_index < len(times):
                _write_sample(reporter_storage, model.reporters(state), sample_index)
                sample_index += 1
            time = float(config.t_end)
            break

        next_time = time - math.log(rng.random()) / total_propensity
        while sample_index < len(times) and times[sample_index] <= min(next_time, config.t_end):
            _write_sample(reporter_storage, model.reporters(state), sample_index)
            sample_index += 1

        if next_time > config.t_end:
            time = next_time
            break

        threshold = rng.random() * total_propensity
        reaction_index = int(np.searchsorted(np.cumsum(propensities), threshold, side="right"))
        if reaction_index >= len(reaction_keys):
            reaction_index = len(reaction_keys) - 1
        model.apply(state, reaction_keys[reaction_index])
        time = next_time
        event_count += 1
        if event_count % config.validation_interval == 0:
            model.validate(state)

    hit_event_cap = event_count >= config.max_events and time < config.t_end
    model.validate(state)
    return TrajectoryResult(
        times=times,
        reporters=reporter_storage,
        final_time=time,
        event_count=event_count,
        hit_event_cap=hit_event_cap,
    )
