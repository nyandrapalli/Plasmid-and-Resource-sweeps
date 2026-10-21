from __future__ import annotations

import hashlib
from dataclasses import dataclass


CONDITIONS = (
    "autocrine_nc3",
    "paracrine_c3",
    "autocrine_nc6",
    "paracrine_c6",
)

SEED_CONDITION_IDENTITIES = {
    "autocrine_nc6": "autocrine_c6",
}


@dataclass(frozen=True)
class SimulationConfig:
    condition: str
    copies: int
    cells_per_population: int = 10
    rnap_total: int = 50
    ribo_total: int = 200
    t_end: int = 3500
    window: tuple[int, int] = (3000, 3500)
    sample_interval: int = 1
    transfer_time_ahl: float = 240.0
    transfer_time_s: float = 240.0
    volume_scale: float | None = None
    max_events: int = 100_000_000
    validation_interval: int = 10_000

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS:
            raise ValueError(f"Unsupported condition: {self.condition}")
        if self.copies not in (5, 40):
            raise ValueError("copies must be 5 or 40")
        if self.cells_per_population <= 0:
            raise ValueError("cells_per_population must be positive")
        if self.rnap_total <= 0 or self.ribo_total <= 0:
            raise ValueError("resource totals must be positive")
        if self.t_end <= 0:
            raise ValueError("t_end must be positive")
        if self.sample_interval <= 0:
            raise ValueError("sample_interval must be positive")
        if self.t_end % self.sample_interval != 0:
            raise ValueError("t_end must be divisible by sample_interval")
        window_start, window_end = self.window
        if window_start < 0 or window_start > window_end or window_end > self.t_end:
            raise ValueError("window must satisfy 0 <= start <= end <= t_end")
        if self.transfer_time_ahl <= 0 or self.transfer_time_s <= 0:
            raise ValueError("transfer times must be positive")
        if self.max_events <= 0:
            raise ValueError("max_events must be positive")
        if self.validation_interval <= 0:
            raise ValueError("validation_interval must be positive")
        if self.volume_scale is None:
            object.__setattr__(self, "volume_scale", float(self.cells_per_population))
        elif self.volume_scale <= 0:
            raise ValueError("volume_scale must be positive")

    @property
    def sample_times(self):
        import numpy as np

        return np.arange(0, self.t_end + 1, self.sample_interval, dtype=np.int64)


def derive_seed(base_seed: int, condition: str, copies: int, replicate_id: int) -> int:
    seed_condition = SEED_CONDITION_IDENTITIES.get(condition, condition)
    payload = f"{base_seed}|{seed_condition}|{copies}|{replicate_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFF_FFFF_FFFF_FFFF
