from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping

import numpy as np

from multicell_shared_pool.config import SimulationConfig


@dataclass(frozen=True)
class KineticParameters:
    kon_act: float = 1e-3
    koff_act: float = 1e-3
    kon_RL: float = 1e-3
    koff_RL: float = 1e-2
    kon_RS: float = 1e-3
    koff_RS: float = 1e-2
    kon_RNAP: float = 5e-4
    kcat_tx_active: float = 0.08
    kcat_tx_basal: float = 0.008
    kon_ribo: float = 1e-3
    kcat_tl: float = 0.5
    gamma_m: float = 0.01
    gamma_p: float = 5e-4
    kmat: float = 0.01
    k_I: float = 0.02
    k_S: float = 0.02
    gamma_AHL: float = 0.001
    gamma_S: float = 0.001
    gamma_G: float = 0.01


@dataclass(frozen=True)
class ReactionNetwork:
    species_names: tuple[str, ...]
    initial_state: np.ndarray
    reaction_names: tuple[str, ...]
    rates: np.ndarray
    reactant_a: np.ndarray
    reactant_b: np.ndarray
    scales: np.ndarray
    stoichiometry: tuple[tuple[tuple[int, int], ...], ...]

    def propensities(self, state: np.ndarray) -> np.ndarray:
        values = self.rates * state[self.reactant_a]
        second_order = self.reactant_b >= 0
        if np.any(second_order):
            values[second_order] *= (
                state[self.reactant_b[second_order]] / self.scales[second_order]
            )
        return values

    def apply(self, state: np.ndarray, reaction_index: int) -> None:
        for species_index, delta in self.stoichiometry[reaction_index]:
            state[species_index] += delta


class NetworkBuilder:
    def __init__(self) -> None:
        self.species_names: list[str] = []
        self.initial_values: list[int] = []
        self.species_lookup: dict[str, int] = {}
        self.reaction_names: list[str] = []
        self.rates: list[float] = []
        self.reactant_a: list[int] = []
        self.reactant_b: list[int] = []
        self.scales: list[float] = []
        self.stoichiometry: list[tuple[tuple[int, int], ...]] = []

    def add_species(self, name: str, initial: int = 0) -> int:
        if name in self.species_lookup:
            raise ValueError(f"Duplicate species: {name}")
        index = len(self.species_names)
        self.species_names.append(name)
        self.initial_values.append(initial)
        self.species_lookup[name] = index
        return index

    def index(self, name: str) -> int:
        return self.species_lookup[name]

    def add_reaction(
        self,
        name: str,
        rate: float,
        reactant_a: str,
        changes: Mapping[str, int],
        reactant_b: str | None = None,
        scale: float = 1.0,
    ) -> int:
        if name in self.reaction_names:
            raise ValueError(f"Duplicate reaction: {name}")
        if rate < 0 or scale <= 0:
            raise ValueError("Reaction rate must be non-negative and scale positive")
        self.reaction_names.append(name)
        self.rates.append(rate)
        self.reactant_a.append(self.index(reactant_a))
        self.reactant_b.append(-1 if reactant_b is None else self.index(reactant_b))
        self.scales.append(scale)
        self.stoichiometry.append(
            tuple((self.index(species), delta) for species, delta in changes.items())
        )
        return len(self.reaction_names) - 1

    def compile(self) -> ReactionNetwork:
        return ReactionNetwork(
            species_names=tuple(self.species_names),
            initial_state=np.asarray(self.initial_values, dtype=np.int64),
            reaction_names=tuple(self.reaction_names),
            rates=np.asarray(self.rates, dtype=float),
            reactant_a=np.asarray(self.reactant_a, dtype=np.int64),
            reactant_b=np.asarray(self.reactant_b, dtype=np.int64),
            scales=np.asarray(self.scales, dtype=float),
            stoichiometry=tuple(self.stoichiometry),
        )


class CircuitBuilder:
    def __init__(self, config: SimulationConfig, kinetics: KineticParameters | None = None):
        self.config = config
        self.kinetics = kinetics or KineticParameters()
        self.network = NetworkBuilder()
        self.promoter_invariants: list[tuple[str, tuple[int, ...], int]] = []
        self.rnap_members: dict[str, list[int]] = {}
        self.ribo_members: dict[str, list[int]] = {}
        self.reporter_indices: dict[str, list[int]] = {}
        self.cell_groups: dict[str, list[str]] = {}

    def add_pool(self, name: str) -> None:
        self.network.add_species(name, 0)

    def add_cell(self, cell: str, group: str) -> None:
        rnap = self.network.add_species(f"{cell}.RNAP_free", self.config.rnap_total)
        ribo = self.network.add_species(f"{cell}.Ribo_free", self.config.ribo_total)
        self.rnap_members[cell] = [rnap]
        self.ribo_members[cell] = [ribo]
        self.cell_groups.setdefault(group, []).append(cell)

    def add_constitutive_gene(self, cell: str, gene: str, protein: str) -> None:
        k = self.kinetics
        prefix = f"{cell}.{gene}"
        free = self.network.add_species(f"{prefix}.promoter_free", self.config.copies)
        bound = self.network.add_species(f"{prefix}.promoter_RNAP", 0)
        mrna = self.network.add_species(f"{prefix}.mRNA", 0)
        ribo_bound = self.network.add_species(f"{prefix}.ribo_bound", 0)
        product_name = f"{cell}.{protein}"
        self.network.add_species(product_name, 0)
        rnap_free = f"{cell}.RNAP_free"
        ribo_free = f"{cell}.Ribo_free"

        self.network.add_reaction(
            f"{prefix}.rnap_bind",
            k.kon_RNAP,
            rnap_free,
            {rnap_free: -1, f"{prefix}.promoter_free": -1, f"{prefix}.promoter_RNAP": 1},
            f"{prefix}.promoter_free",
        )
        self.network.add_reaction(
            f"{prefix}.transcription",
            k.kcat_tx_active,
            f"{prefix}.promoter_RNAP",
            {
                f"{prefix}.promoter_RNAP": -1,
                f"{prefix}.promoter_free": 1,
                rnap_free: 1,
                f"{prefix}.mRNA": 1,
            },
        )
        self.network.add_reaction(
            f"{prefix}.ribo_bind",
            k.kon_ribo,
            ribo_free,
            {ribo_free: -1, f"{prefix}.mRNA": -1, f"{prefix}.ribo_bound": 1},
            f"{prefix}.mRNA",
        )
        self.network.add_reaction(
            f"{prefix}.translation",
            k.kcat_tl,
            f"{prefix}.ribo_bound",
            {f"{prefix}.ribo_bound": -1, ribo_free: 1, product_name: 1},
        )
        self.network.add_reaction(
            f"{prefix}.mrna_decay",
            k.gamma_m,
            f"{prefix}.mRNA",
            {f"{prefix}.mRNA": -1},
        )
        self.network.add_reaction(
            f"{prefix}.protein_decay",
            k.gamma_p,
            product_name,
            {product_name: -1},
        )

        self.promoter_invariants.append((prefix, (free, bound), self.config.copies))
        self.rnap_members[cell].append(bound)
        self.ribo_members[cell].append(ribo_bound)

    def add_regulated_gene(
        self,
        cell: str,
        gene: str,
        regulator_complex: str,
        product: str,
        reporter_key: str | None = None,
    ) -> None:
        k = self.kinetics
        prefix = f"{cell}.{gene}"
        free = self.network.add_species(f"{prefix}.promoter_free", self.config.copies)
        active = self.network.add_species(f"{prefix}.promoter_active", 0)
        rnap_inactive = self.network.add_species(f"{prefix}.RNAP_inactive", 0)
        rnap_active = self.network.add_species(f"{prefix}.RNAP_active", 0)
        mrna = self.network.add_species(f"{prefix}.mRNA", 0)
        ribo_bound = self.network.add_species(f"{prefix}.ribo_bound", 0)
        complex_name = f"{cell}.{regulator_complex}"
        rnap_free = f"{cell}.RNAP_free"
        ribo_free = f"{cell}.Ribo_free"

        if reporter_key is None:
            product_name = f"{cell}.{product}"
            self.network.add_species(product_name, 0)
        else:
            immature_name = f"{cell}.{product}_immature"
            product_name = f"{cell}.{product}"
            self.network.add_species(immature_name, 0)
            reporter_index = self.network.add_species(product_name, 0)
            self.reporter_indices.setdefault(reporter_key, []).append(reporter_index)

        self.network.add_reaction(
            f"{prefix}.activate",
            k.kon_act,
            complex_name,
            {complex_name: -1, f"{prefix}.promoter_free": -1, f"{prefix}.promoter_active": 1},
            f"{prefix}.promoter_free",
        )
        self.network.add_reaction(
            f"{prefix}.deactivate",
            k.koff_act,
            f"{prefix}.promoter_active",
            {f"{prefix}.promoter_active": -1, f"{prefix}.promoter_free": 1, complex_name: 1},
        )
        self.network.add_reaction(
            f"{prefix}.rnap_bind_inactive",
            k.kon_RNAP,
            rnap_free,
            {rnap_free: -1, f"{prefix}.promoter_free": -1, f"{prefix}.RNAP_inactive": 1},
            f"{prefix}.promoter_free",
        )
        self.network.add_reaction(
            f"{prefix}.rnap_bind_active",
            k.kon_RNAP,
            rnap_free,
            {rnap_free: -1, f"{prefix}.promoter_active": -1, f"{prefix}.RNAP_active": 1},
            f"{prefix}.promoter_active",
        )
        self.network.add_reaction(
            f"{prefix}.transcription_inactive",
            k.kcat_tx_basal,
            f"{prefix}.RNAP_inactive",
            {
                f"{prefix}.RNAP_inactive": -1,
                f"{prefix}.promoter_free": 1,
                rnap_free: 1,
                f"{prefix}.mRNA": 1,
            },
        )
        self.network.add_reaction(
            f"{prefix}.transcription_active",
            k.kcat_tx_active,
            f"{prefix}.RNAP_active",
            {
                f"{prefix}.RNAP_active": -1,
                f"{prefix}.promoter_active": 1,
                rnap_free: 1,
                f"{prefix}.mRNA": 1,
            },
        )
        self.network.add_reaction(
            f"{prefix}.ribo_bind",
            k.kon_ribo,
            ribo_free,
            {ribo_free: -1, f"{prefix}.mRNA": -1, f"{prefix}.ribo_bound": 1},
            f"{prefix}.mRNA",
        )
        translation_changes = {f"{prefix}.ribo_bound": -1, ribo_free: 1}
        if reporter_key is None:
            translation_changes[product_name] = 1
        else:
            translation_changes[immature_name] = 1
        self.network.add_reaction(
            f"{prefix}.translation",
            k.kcat_tl,
            f"{prefix}.ribo_bound",
            translation_changes,
        )
        self.network.add_reaction(
            f"{prefix}.mrna_decay",
            k.gamma_m,
            f"{prefix}.mRNA",
            {f"{prefix}.mRNA": -1},
        )
        if reporter_key is None:
            self.network.add_reaction(
                f"{prefix}.protein_decay",
                k.gamma_p,
                product_name,
                {product_name: -1},
            )
        else:
            self.network.add_reaction(
                f"{prefix}.maturation",
                k.kmat,
                immature_name,
                {immature_name: -1, product_name: 1},
            )
            self.network.add_reaction(
                f"{prefix}.reporter_decay",
                k.gamma_G,
                product_name,
                {product_name: -1},
            )

        self.promoter_invariants.append(
            (prefix, (free, active, rnap_inactive, rnap_active), self.config.copies)
        )
        self.rnap_members[cell].extend((rnap_inactive, rnap_active))
        self.ribo_members[cell].append(ribo_bound)

    def add_receptor_binding(
        self,
        cell: str,
        receptor: str,
        signal_pool: str,
        complex_name: str,
        on_rate: float,
        off_rate: float,
    ) -> None:
        receptor_name = f"{cell}.{receptor}"
        complex_species = f"{cell}.{complex_name}"
        self.network.add_species(complex_species, 0)
        self.network.add_reaction(
            f"{cell}.{complex_name}.bind",
            on_rate,
            receptor_name,
            {receptor_name: -1, signal_pool: -1, complex_species: 1},
            signal_pool,
            scale=float(self.config.volume_scale),
        )
        self.network.add_reaction(
            f"{cell}.{complex_name}.unbind",
            off_rate,
            complex_species,
            {complex_species: -1, receptor_name: 1, signal_pool: 1},
        )

    def add_signal_synthesis(
        self,
        cell: str,
        synthase: str,
        signal_pool: str,
        signal_name: str,
        rate: float,
    ) -> None:
        self.network.add_reaction(
            f"{cell}.{synthase}.synthesize_{signal_name}",
            rate,
            f"{cell}.{synthase}",
            {signal_pool: 1},
        )

    def add_pool_decay(self, pool: str, signal_name: str, rate: float) -> None:
        self.network.add_reaction(
            f"{signal_name}.decay.{pool}",
            rate,
            pool,
            {pool: -1},
        )

    def add_transfer(self, name: str, source: str, destination: str, rate: float) -> None:
        self.network.add_reaction(
            name,
            rate,
            source,
            {source: -1, destination: 1},
        )


class CompiledPopulationModel:
    def __init__(
        self,
        config: SimulationConfig,
        builder: CircuitBuilder,
        reporter_labels: Mapping[str, str],
    ) -> None:
        self.config = config
        self.network = builder.network.compile()
        self.reporter_indices = {
            key: np.asarray(indices, dtype=np.int64)
            for key, indices in builder.reporter_indices.items()
        }
        self.reporter_labels = dict(reporter_labels)
        self._promoter_invariants = tuple(builder.promoter_invariants)
        self._rnap_invariants = tuple(
            (cell, tuple(indices), config.rnap_total)
            for cell, indices in builder.rnap_members.items()
        )
        self._ribo_invariants = tuple(
            (cell, tuple(indices), config.ribo_total)
            for cell, indices in builder.ribo_members.items()
        )
        self._cell_groups = {group: tuple(cells) for group, cells in builder.cell_groups.items()}
        self._species_lookup = {name: i for i, name in enumerate(self.network.species_names)}
        self._reaction_lookup = {name: i for i, name in enumerate(self.network.reaction_names)}
        self._reaction_keys = tuple(range(len(self.network.reaction_names)))

    def initial_state(self, config: SimulationConfig) -> np.ndarray:
        if config != self.config:
            raise ValueError("Model and simulation configurations differ")
        return self.network.initial_state.copy()

    def reactions(self, state: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
        return self.network.propensities(state), self._reaction_keys

    def apply(self, state: np.ndarray, reaction_key: Hashable) -> None:
        self.network.apply(state, int(reaction_key))

    def reporters(self, state: np.ndarray) -> dict[str, np.ndarray]:
        return {key: state[indices].copy() for key, indices in self.reporter_indices.items()}

    def validate(self, state: np.ndarray) -> None:
        if np.any(state < 0):
            raise ValueError("State contains a negative molecular count")
        for label, indices, expected in self._promoter_invariants:
            if int(state[list(indices)].sum()) != expected:
                raise ValueError(f"Promoter conservation failed for {label}")
        for cell, indices, expected in self._rnap_invariants:
            if int(state[list(indices)].sum()) != expected:
                raise ValueError(f"RNAP conservation failed for {cell}")
        for cell, indices, expected in self._ribo_invariants:
            if int(state[list(indices)].sum()) != expected:
                raise ValueError(f"Ribosome conservation failed for {cell}")

    def species_index(self, name: str) -> int:
        return self._species_lookup[name]

    def has_species(self, name: str) -> bool:
        return name in self._species_lookup

    def reaction_index(self, name: str) -> int:
        return self._reaction_lookup[name]

    def has_reaction(self, name: str) -> bool:
        return name in self._reaction_lookup

    def reaction_propensity(self, state: np.ndarray, name: str) -> float:
        return float(self.network.propensities(state)[self.reaction_index(name)])

    def apply_named(self, state: np.ndarray, name: str) -> None:
        self.network.apply(state, self.reaction_index(name))

    def cell_names(self, group: str) -> tuple[str, ...]:
        return self._cell_groups[group]
