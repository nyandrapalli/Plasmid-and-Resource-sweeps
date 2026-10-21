from __future__ import annotations

from multicell_shared_pool.config import SimulationConfig
from multicell_shared_pool.models.network import CircuitBuilder, CompiledPopulationModel


class ParacrineC6Model(CompiledPopulationModel):
    def __init__(self, config: SimulationConfig):
        if config.condition != "paracrine_c6":
            raise ValueError("ParacrineC6Model requires condition='paracrine_c6'")
        builder = CircuitBuilder(config)
        for pool in (
            "population_a.AHL",
            "population_b.AHL",
            "population_a.S",
            "population_b.S",
        ):
            builder.add_pool(pool)

        for cell_index in range(config.cells_per_population):
            cell = f"population_a{cell_index}"
            builder.add_cell(cell, "population_a")
            builder.add_constitutive_gene(cell, "pCon_luxI", "LuxI")
            builder.add_constitutive_gene(cell, "pCon_lasR", "LasR")
            builder.add_receptor_binding(
                cell,
                "LasR",
                "population_a.S",
                "RS",
                builder.kinetics.kon_RS,
                builder.kinetics.koff_RS,
            )
            builder.add_regulated_gene(
                cell,
                "pLas_reporter",
                "RS",
                "G_Las",
                reporter_key="pLas_reporter",
            )
            builder.add_signal_synthesis(
                cell,
                "LuxI",
                "population_a.AHL",
                "AHL",
                builder.kinetics.k_I,
            )

        for cell_index in range(config.cells_per_population):
            cell = f"population_b{cell_index}"
            builder.add_cell(cell, "population_b")
            builder.add_constitutive_gene(cell, "pCon_luxR", "LuxR")
            builder.add_receptor_binding(
                cell,
                "LuxR",
                "population_b.AHL",
                "RL",
                builder.kinetics.kon_RL,
                builder.kinetics.koff_RL,
            )
            builder.add_regulated_gene(
                cell,
                "pLux_eGFP",
                "RL",
                "G_Lux",
                reporter_key="pLux_eGFP",
            )
            builder.add_regulated_gene(cell, "pLux_lasI", "RL", "LasI")
            builder.add_signal_synthesis(
                cell,
                "LasI",
                "population_b.S",
                "S",
                builder.kinetics.k_S,
            )

        transfer_ahl = 1.0 / config.transfer_time_ahl
        transfer_s = 1.0 / config.transfer_time_s
        builder.add_transfer(
            "AHL.transfer_a_to_b",
            "population_a.AHL",
            "population_b.AHL",
            transfer_ahl,
        )
        builder.add_transfer(
            "AHL.transfer_b_to_a",
            "population_b.AHL",
            "population_a.AHL",
            transfer_ahl,
        )
        builder.add_transfer(
            "S.transfer_a_to_b",
            "population_a.S",
            "population_b.S",
            transfer_s,
        )
        builder.add_transfer(
            "S.transfer_b_to_a",
            "population_b.S",
            "population_a.S",
            transfer_s,
        )
        builder.add_pool_decay("population_a.AHL", "AHL", builder.kinetics.gamma_AHL)
        builder.add_pool_decay("population_b.AHL", "AHL", builder.kinetics.gamma_AHL)
        builder.add_pool_decay("population_a.S", "S", builder.kinetics.gamma_S)
        builder.add_pool_decay("population_b.S", "S", builder.kinetics.gamma_S)
        super().__init__(
            config,
            builder,
            {"pLux_eGFP": "pLux-eGFP", "pLas_reporter": "pLas-eGFP"},
        )
