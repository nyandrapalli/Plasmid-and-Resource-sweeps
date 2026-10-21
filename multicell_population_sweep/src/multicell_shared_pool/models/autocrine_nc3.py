from __future__ import annotations

from multicell_shared_pool.config import SimulationConfig
from multicell_shared_pool.models.network import (
    CircuitBuilder,
    CompiledPopulationModel,
)


class AutocrineNC3Model(CompiledPopulationModel):
    def __init__(self, config: SimulationConfig):
        if config.condition != "autocrine_nc3":
            raise ValueError("AutocrineNC3Model requires condition='autocrine_nc3'")
        builder = CircuitBuilder(config)
        builder.add_pool("pool.AHL")
        for cell_index in range(config.cells_per_population):
            cell = f"cell{cell_index}"
            builder.add_cell(cell, "autocrine")
            builder.add_constitutive_gene(cell, "pCon_luxI", "LuxI")
            builder.add_constitutive_gene(cell, "pCon_luxR", "LuxR")
            builder.add_receptor_binding(
                cell,
                "LuxR",
                "pool.AHL",
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
            builder.add_signal_synthesis(
                cell,
                "LuxI",
                "pool.AHL",
                "AHL",
                builder.kinetics.k_I,
            )
        builder.add_pool_decay("pool.AHL", "AHL", builder.kinetics.gamma_AHL)
        super().__init__(config, builder, {"pLux_eGFP": "pLux-eGFP"})
