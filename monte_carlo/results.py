"""Output object returned by the manager."""

from dataclasses import dataclass

from numpy.typing import NDArray
import numpy as np

from .config import SimulationConfig


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SimulationResult:
    config: SimulationConfig
    paths: FloatArray
    raw_payoffs: FloatArray
    discounted_payoffs: FloatArray
    option_price: float
    payoff_variance: float
    standard_error: float
    elapsed_seconds: float

    @property
    def time_grid(self) -> FloatArray:
        return np.linspace(0.0, self.config.maturity, self.config.num_steps + 1)
