from __future__ import annotations

from typing import Any, Hashable, Mapping, Protocol, Sequence

import numpy as np

from multicell_shared_pool.config import SimulationConfig


class PopulationModel(Protocol):
    reporter_labels: Mapping[str, str]

    def initial_state(self, config: SimulationConfig) -> Any:
        ...

    def reactions(self, state: Any) -> tuple[np.ndarray, Sequence[Hashable]]:
        ...

    def apply(self, state: Any, reaction_key: Hashable) -> None:
        ...

    def reporters(self, state: Any) -> Mapping[str, np.ndarray]:
        ...

    def validate(self, state: Any) -> None:
        ...
