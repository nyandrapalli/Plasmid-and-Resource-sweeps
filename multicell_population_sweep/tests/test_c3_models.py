import unittest

import numpy as np

from multicell_shared_pool.config import SimulationConfig
from multicell_shared_pool.models.autocrine_nc3 import AutocrineNC3Model
from multicell_shared_pool.models.paracrine_c3 import ParacrineC3Model


class ThreePlasmidPopulationModelTests(unittest.TestCase):
    def test_autocrine_initializes_ten_complete_cells_and_one_shared_pool(self):
        config = SimulationConfig(condition="autocrine_nc3", copies=5)
        model = AutocrineNC3Model(config)
        state = model.initial_state(config)

        self.assertEqual(model.reporters(state)["pLux_eGFP"].shape, (10,))
        self.assertEqual(state[model.species_index("pool.AHL")], 0)
        for cell_index in range(10):
            prefix = f"cell{cell_index}"
            self.assertEqual(state[model.species_index(f"{prefix}.RNAP_free")], 50)
            self.assertEqual(state[model.species_index(f"{prefix}.Ribo_free")], 200)
            self.assertEqual(state[model.species_index(f"{prefix}.pCon_luxI.promoter_free")], 5)
            self.assertEqual(state[model.species_index(f"{prefix}.pCon_luxR.promoter_free")], 5)
            self.assertEqual(state[model.species_index(f"{prefix}.pLux_eGFP.promoter_free")], 5)

    def test_paracrine_initializes_ten_producers_and_ten_responders(self):
        config = SimulationConfig(condition="paracrine_c3", copies=40)
        model = ParacrineC3Model(config)
        state = model.initial_state(config)

        self.assertEqual(model.reporters(state)["pLux_eGFP"].shape, (10,))
        self.assertEqual(len(model.cell_names("producer")), 10)
        self.assertEqual(len(model.cell_names("responder")), 10)
        self.assertFalse(model.has_species("producer0.LuxR"))
        self.assertFalse(model.has_species("responder0.LuxI"))
        self.assertEqual(state[model.species_index("producer_pool.AHL")], 0)
        self.assertEqual(state[model.species_index("responder_pool.AHL")], 0)

    def test_one_cell_limit_uses_existing_mass_action_propensities(self):
        config = SimulationConfig(
            condition="autocrine_nc3",
            copies=5,
            cells_per_population=1,
            volume_scale=1,
        )
        model = AutocrineNC3Model(config)
        state = model.initial_state(config)

        self.assertAlmostEqual(model.reaction_propensity(state, "cell0.pCon_luxI.rnap_bind"), 0.125)
        self.assertAlmostEqual(model.reaction_propensity(state, "cell0.pCon_luxR.rnap_bind"), 0.125)
        self.assertAlmostEqual(model.reaction_propensity(state, "cell0.pLux_eGFP.rnap_bind_inactive"), 0.125)

        state[model.species_index("cell0.LuxI")] = 4
        self.assertAlmostEqual(model.reaction_propensity(state, "cell0.LuxI.synthesize_AHL"), 0.08)

    def test_volume_scaling_preserves_per_cell_binding_propensity(self):
        single_config = SimulationConfig(
            condition="autocrine_nc3",
            copies=5,
            cells_per_population=1,
            volume_scale=1,
        )
        population_config = SimulationConfig(condition="autocrine_nc3", copies=5)
        single = AutocrineNC3Model(single_config)
        population = AutocrineNC3Model(population_config)
        single_state = single.initial_state(single_config)
        population_state = population.initial_state(population_config)

        single_state[single.species_index("cell0.LuxR")] = 2
        single_state[single.species_index("pool.AHL")] = 3
        population_state[population.species_index("cell0.LuxR")] = 2
        population_state[population.species_index("pool.AHL")] = 30

        single_rate = single.reaction_propensity(single_state, "cell0.RL.bind")
        population_rate = population.reaction_propensity(population_state, "cell0.RL.bind")
        self.assertAlmostEqual(single_rate, population_rate)
        self.assertAlmostEqual(single_rate, 0.006)

    def test_paracrine_transfer_moves_one_signal_and_conserves_total(self):
        config = SimulationConfig(
            condition="paracrine_c3",
            copies=5,
            cells_per_population=1,
            volume_scale=1,
        )
        model = ParacrineC3Model(config)
        state = model.initial_state(config)
        source = model.species_index("producer_pool.AHL")
        destination = model.species_index("responder_pool.AHL")
        state[source] = 2

        before = state[source] + state[destination]
        model.apply_named(state, "AHL.transfer_producer_to_responder")

        self.assertEqual(state[source], 1)
        self.assertEqual(state[destination], 1)
        self.assertEqual(state[source] + state[destination], before)
        self.assertAlmostEqual(
            model.reaction_propensity(state, "AHL.transfer_producer_to_responder"),
            1 / 240,
        )

    def test_promoter_and_resource_invariants_detect_corruption(self):
        config = SimulationConfig(
            condition="autocrine_nc3",
            copies=5,
            cells_per_population=1,
            volume_scale=1,
        )
        model = AutocrineNC3Model(config)
        state = model.initial_state(config)

        model.apply_named(state, "cell0.pCon_luxI.rnap_bind")
        model.validate(state)
        self.assertEqual(state[model.species_index("cell0.RNAP_free")], 49)

        corrupted = state.copy()
        corrupted[model.species_index("cell0.pCon_luxI.promoter_free")] += 1
        with self.assertRaises(ValueError):
            model.validate(corrupted)

        negative = state.copy()
        negative[model.species_index("pool.AHL")] = -1
        with self.assertRaises(ValueError):
            model.validate(negative)


if __name__ == "__main__":
    unittest.main()
