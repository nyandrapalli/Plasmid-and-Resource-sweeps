from __future__ import annotations

from multicell_shared_pool.config import SimulationConfig
from multicell_shared_pool.models.network import CircuitBuilder, CompiledPopulationModel


class AutocrineNC6Model(CompiledPopulationModel):
    def __init__(self, config: SimulationConfig):
        if config.condition != "autocrine_nc6":
            raise ValueError("AutocrineNC6Model requires condition='autocrine_nc6'")
        builder = CircuitBuilder(config)
        builder.add_pool("pool.AHL")
        builder.add_pool("pool.S")

        for cell_index in range(config.cells_per_population):
            cell = f"cell{cell_index}"
            builder.add_cell(cell, "autocrine")
            builder.add_constitutive_gene(cell, "pCon_luxI", "LuxI")
            builder.add_constitutive_gene(cell, "pCon_luxR", "LuxR")
            builder.add_constitutive_gene(cell, "pCon_lasR", "LasR")
            builder.add_receptor_binding(
                cell,
                "LuxR",
                "pool.AHL",
                "RL",
                builder.kinetics.kon_RL,
                builder.kinetics.koff_RL,
            )
            builder.add_receptor_binding(
                cell,
                "LasR",
                "pool.S",
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
                "LuxI",
                "pool.AHL",
                "AHL",
                builder.kinetics.k_I,
            )
            builder.add_signal_synthesis(
                cell,
                "LasI",
                "pool.S",
                "S",
                builder.kinetics.k_S,
            )

        builder.add_pool_decay("pool.AHL", "AHL", builder.kinetics.gamma_AHL)
        builder.add_pool_decay("pool.S", "S", builder.kinetics.gamma_S)
        super().__init__(
            config,
            builder,
            {"pLux_eGFP": "pLux-eGFP", "pLas_reporter": "pLas-mScarlet"},
        )
