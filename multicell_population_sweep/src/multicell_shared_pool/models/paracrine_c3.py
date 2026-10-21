from __future__ import annotations

from multicell_shared_pool.config import SimulationConfig
from multicell_shared_pool.models.network import (
    CircuitBuilder,
    CompiledPopulationModel,
)


class ParacrineC3Model(CompiledPopulationModel):
    def __init__(self, config: SimulationConfig):
        if config.condition != "paracrine_c3":
            raise ValueError("ParacrineC3Model requires condition='paracrine_c3'")
        builder = CircuitBuilder(config)
        builder.add_pool("producer_pool.AHL")
        builder.add_pool("responder_pool.AHL")

        for cell_index in range(config.cells_per_population):
            cell = f"producer{cell_index}"
            builder.add_cell(cell, "producer")
            builder.add_constitutive_gene(cell, "pCon_luxI", "LuxI")
            builder.add_signal_synthesis(
                cell,
                "LuxI",
                "producer_pool.AHL",
                "AHL",
                builder.kinetics.k_I,
            )

        for cell_index in range(config.cells_per_population):
            cell = f"responder{cell_index}"
            builder.add_cell(cell, "responder")
            builder.add_constitutive_gene(cell, "pCon_luxR", "LuxR")
            builder.add_receptor_binding(
                cell,
                "LuxR",
                "responder_pool.AHL",
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

        transfer_rate = 1.0 / config.transfer_time_ahl
        builder.add_transfer(
            "AHL.transfer_producer_to_responder",
            "producer_pool.AHL",
            "responder_pool.AHL",
            transfer_rate,
        )
        builder.add_transfer(
            "AHL.transfer_responder_to_producer",
            "responder_pool.AHL",
            "producer_pool.AHL",
            transfer_rate,
        )
        builder.add_pool_decay("producer_pool.AHL", "AHL", builder.kinetics.gamma_AHL)
        builder.add_pool_decay("responder_pool.AHL", "AHL", builder.kinetics.gamma_AHL)
        super().__init__(config, builder, {"pLux_eGFP": "pLux-eGFP"})
