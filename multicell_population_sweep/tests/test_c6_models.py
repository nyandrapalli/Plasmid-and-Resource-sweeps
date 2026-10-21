import unittest

from multicell_shared_pool.config import SimulationConfig
from multicell_shared_pool.models.autocrine_nc6 import AutocrineNC6Model
from multicell_shared_pool.models.paracrine_c6 import ParacrineC6Model


class SixPlasmidPopulationModelTests(unittest.TestCase):
    def test_autocrine_has_ten_cells_with_both_correctly_labeled_reporters(self):
        config = SimulationConfig(condition="autocrine_nc6", copies=5)
        model = AutocrineNC6Model(config)
        state = model.initial_state(config)
        reporters = model.reporters(state)

        self.assertEqual(reporters["pLux_eGFP"].shape, (10,))
        self.assertEqual(reporters["pLas_reporter"].shape, (10,))
        self.assertEqual(model.reporter_labels["pLux_eGFP"], "pLux-eGFP")
        self.assertEqual(model.reporter_labels["pLas_reporter"], "pLas-mScarlet")
        self.assertEqual(len(model.cell_names("autocrine")), 10)

        expected_promoters = (
            "pCon_luxI",
            "pCon_luxR",
            "pCon_lasR",
            "pLas_reporter",
            "pLux_eGFP",
            "pLux_lasI",
        )
        for promoter in expected_promoters:
            self.assertEqual(
                state[model.species_index(f"cell0.{promoter}.promoter_free")],
                5,
            )

    def test_paracrine_has_two_ten_cell_populations_and_separate_reporters(self):
        config = SimulationConfig(condition="paracrine_c6", copies=40)
        model = ParacrineC6Model(config)
        state = model.initial_state(config)
        reporters = model.reporters(state)

        self.assertEqual(len(model.cell_names("population_a")), 10)
        self.assertEqual(len(model.cell_names("population_b")), 10)
        self.assertEqual(reporters["pLas_reporter"].shape, (10,))
        self.assertEqual(reporters["pLux_eGFP"].shape, (10,))
        self.assertEqual(model.reporter_labels["pLas_reporter"], "pLas-eGFP")
        self.assertTrue(model.has_species("population_a0.G_Las"))
        self.assertFalse(model.has_species("population_a0.G_Lux"))
        self.assertTrue(model.has_species("population_b0.G_Lux"))
        self.assertFalse(model.has_species("population_b0.G_Las"))

    def test_one_cell_limit_preserves_six_plasmid_rnap_binding_equations(self):
        config = SimulationConfig(
            condition="autocrine_nc6",
            copies=5,
            cells_per_population=1,
            volume_scale=1,
        )
        model = AutocrineNC6Model(config)
        state = model.initial_state(config)

        reaction_names = (
            "cell0.pCon_luxI.rnap_bind",
            "cell0.pCon_luxR.rnap_bind",
            "cell0.pCon_lasR.rnap_bind",
            "cell0.pLas_reporter.rnap_bind_inactive",
            "cell0.pLux_eGFP.rnap_bind_inactive",
            "cell0.pLux_lasI.rnap_bind_inactive",
        )
        for reaction_name in reaction_names:
            with self.subTest(reaction_name=reaction_name):
                self.assertAlmostEqual(model.reaction_propensity(state, reaction_name), 0.125)

    def test_paracrine_uses_correct_signal_for_each_receptor(self):
        config = SimulationConfig(
            condition="paracrine_c6",
            copies=5,
            cells_per_population=1,
            volume_scale=1,
        )
        model = ParacrineC6Model(config)
        state = model.initial_state(config)

        state[model.species_index("population_a0.LasR")] = 2
        state[model.species_index("population_a.S")] = 3
        state[model.species_index("population_b0.LuxR")] = 4
        state[model.species_index("population_b.AHL")] = 5

        self.assertAlmostEqual(model.reaction_propensity(state, "population_a0.RS.bind"), 0.006)
        self.assertAlmostEqual(model.reaction_propensity(state, "population_b0.RL.bind"), 0.020)
        self.assertFalse(model.has_reaction("population_a0.RL.bind"))
        self.assertFalse(model.has_reaction("population_b0.RS.bind"))

    def test_ahl_and_las_signal_transfer_independently_at_240_seconds(self):
        config = SimulationConfig(
            condition="paracrine_c6",
            copies=5,
            cells_per_population=1,
            volume_scale=1,
        )
        model = ParacrineC6Model(config)
        state = model.initial_state(config)
        state[model.species_index("population_a.AHL")] = 2
        state[model.species_index("population_b.S")] = 3

        self.assertAlmostEqual(
            model.reaction_propensity(state, "AHL.transfer_a_to_b"),
            2 / 240,
        )
        self.assertAlmostEqual(
            model.reaction_propensity(state, "S.transfer_b_to_a"),
            3 / 240,
        )

        model.apply_named(state, "AHL.transfer_a_to_b")
        model.apply_named(state, "S.transfer_b_to_a")
        self.assertEqual(state[model.species_index("population_a.AHL")], 1)
        self.assertEqual(state[model.species_index("population_b.AHL")], 1)
        self.assertEqual(state[model.species_index("population_b.S")], 2)
        self.assertEqual(state[model.species_index("population_a.S")], 1)

    def test_reporter_values_are_not_swapped(self):
        auto_config = SimulationConfig(
            condition="autocrine_nc6",
            copies=5,
            cells_per_population=1,
            volume_scale=1,
        )
        auto = AutocrineNC6Model(auto_config)
        auto_state = auto.initial_state(auto_config)
        auto_state[auto.species_index("cell0.G_Lux")] = 11
        auto_state[auto.species_index("cell0.G_Las")] = 22

        self.assertEqual(auto.reporters(auto_state)["pLux_eGFP"][0], 11)
        self.assertEqual(auto.reporters(auto_state)["pLas_reporter"][0], 22)

        para_config = SimulationConfig(
            condition="paracrine_c6",
            copies=5,
            cells_per_population=1,
            volume_scale=1,
        )
        para = ParacrineC6Model(para_config)
        para_state = para.initial_state(para_config)
        para_state[para.species_index("population_a0.G_Las")] = 33
        para_state[para.species_index("population_b0.G_Lux")] = 44

        self.assertEqual(para.reporters(para_state)["pLas_reporter"][0], 33)
        self.assertEqual(para.reporters(para_state)["pLux_eGFP"][0], 44)


if __name__ == "__main__":
    unittest.main()
