import unittest

from multicell_shared_pool.config import SimulationConfig, derive_seed


class SimulationConfigTests(unittest.TestCase):
    def test_defaults_match_approved_population_experiment(self):
        config = SimulationConfig(condition="paracrine_c6", copies=40)

        self.assertEqual(config.cells_per_population, 10)
        self.assertEqual(config.rnap_total, 50)
        self.assertEqual(config.ribo_total, 200)
        self.assertEqual(config.t_end, 3500)
        self.assertEqual(config.window, (3000, 3500))
        self.assertEqual(config.sample_interval, 1)
        self.assertEqual(config.transfer_time_ahl, 240.0)
        self.assertEqual(config.transfer_time_s, 240.0)
        self.assertEqual(config.volume_scale, 10.0)
        self.assertEqual(config.validation_interval, 10_000)

    def test_six_plasmid_noncommunicating_condition_uses_nc6_identifier(self):
        config = SimulationConfig(condition="autocrine_nc6", copies=5)

        self.assertEqual(config.condition, "autocrine_nc6")

    def test_invalid_scientific_settings_are_rejected(self):
        invalid_cases = [
            {"condition": "unknown", "copies": 5},
            {"condition": "autocrine_nc3", "copies": 10},
            {"condition": "autocrine_nc3", "copies": 5, "cells_per_population": 0},
            {"condition": "autocrine_nc3", "copies": 5, "rnap_total": 0},
            {"condition": "autocrine_nc3", "copies": 5, "ribo_total": -1},
            {"condition": "paracrine_c3", "copies": 5, "transfer_time_ahl": 0},
            {"condition": "paracrine_c6", "copies": 5, "transfer_time_s": -1},
            {"condition": "autocrine_nc3", "copies": 5, "window": (3500, 3000)},
            {"condition": "autocrine_nc3", "copies": 5, "window": (3000, 3501)},
            {"condition": "autocrine_nc3", "copies": 5, "sample_interval": 0},
            {"condition": "autocrine_nc3", "copies": 5, "max_events": 0},
            {"condition": "autocrine_nc3", "copies": 5, "validation_interval": 0},
        ]

        for kwargs in invalid_cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    SimulationConfig(**kwargs)

    def test_seed_derivation_is_stable_and_separates_replicates(self):
        seed = derive_seed(20261019, "paracrine_c6", 40, 17)

        self.assertEqual(seed, 5595571789577904499)
        self.assertNotEqual(seed, derive_seed(20261019, "paracrine_c6", 40, 18))
        self.assertNotEqual(seed, derive_seed(20261019, "paracrine_c3", 40, 17))

    def test_nc6_rename_preserves_the_completed_run_seed_stream(self):
        corrected = derive_seed(20261019, "autocrine_nc6", 40, 17)

        self.assertEqual(corrected, 4854838953329405237)


if __name__ == "__main__":
    unittest.main()
