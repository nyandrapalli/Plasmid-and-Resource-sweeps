import unittest

import numpy as np

from multicell_shared_pool.config import SimulationConfig
from multicell_shared_pool.ssa import run_trajectory


class BirthDeathModel:
    reporter_labels = {"X": "X"}

    def initial_state(self, config):
        return np.array([0], dtype=np.int64)

    def reactions(self, state):
        return np.array([2.0, 0.5 * state[0]], dtype=float), ("birth", "death")

    def apply(self, state, reaction_key):
        if reaction_key == "birth":
            state[0] += 1
        elif reaction_key == "death":
            state[0] -= 1

    def reporters(self, state):
        return {"X": np.array([state[0]], dtype=np.int64)}

    def validate(self, state):
        if state[0] < 0:
            raise ValueError("negative state")


class CountingValidationModel(BirthDeathModel):
    def __init__(self):
        self.validation_calls = 0

    def validate(self, state):
        self.validation_calls += 1
        super().validate(state)


class ExactSSATests(unittest.TestCase):
    def test_integer_sampling_is_deterministic_and_nonnegative(self):
        config = SimulationConfig(
            condition="autocrine_nc3",
            copies=5,
            cells_per_population=1,
            t_end=5,
            window=(2, 5),
            volume_scale=1,
            max_events=1000,
        )

        first = run_trajectory(BirthDeathModel(), config, seed=123)
        second = run_trajectory(BirthDeathModel(), config, seed=123)

        np.testing.assert_array_equal(first.times, np.arange(6))
        np.testing.assert_array_equal(first.reporters["X"], second.reporters["X"])
        self.assertEqual(first.reporters["X"].shape, (1, 6))
        self.assertTrue(np.all(first.reporters["X"] >= 0))
        self.assertFalse(first.hit_event_cap)
        self.assertGreaterEqual(first.final_time, config.t_end)

    def test_event_cap_is_reported_instead_of_treated_as_complete(self):
        config = SimulationConfig(
            condition="autocrine_nc3",
            copies=5,
            cells_per_population=1,
            t_end=100,
            window=(50, 100),
            volume_scale=1,
            max_events=2,
        )

        result = run_trajectory(BirthDeathModel(), config, seed=99)

        self.assertTrue(result.hit_event_cap)
        self.assertEqual(result.event_count, 2)
        self.assertLess(result.final_time, config.t_end)

    def test_expensive_full_state_validation_is_periodic_and_final(self):
        config = SimulationConfig(
            condition="autocrine_nc3",
            copies=5,
            cells_per_population=1,
            t_end=20,
            window=(10, 20),
            volume_scale=1,
            validation_interval=10,
            max_events=1000,
        )
        model = CountingValidationModel()

        result = run_trajectory(model, config, seed=321)

        self.assertGreater(result.event_count, 10)
        self.assertGreaterEqual(model.validation_calls, 2)
        self.assertLess(model.validation_calls, result.event_count / 2)


if __name__ == "__main__":
    unittest.main()
