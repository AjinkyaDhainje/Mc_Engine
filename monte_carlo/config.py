"""Validated inputs and supported choices for a simulation."""

from dataclasses import dataclass
from enum import Enum
import math


class ModelType(str, Enum):
    GBM = "Geometric Brownian Motion"
    MERTON_JUMP = "Merton Jump Model"


class DiscretizationType(str, Enum):
    EULER = "Euler"
    MILSTEIN = "Milstein"


class PayoffType(str, Enum):
    VANILLA = "Vanilla"
    ASIAN = "Asian"


class OptionType(str, Enum):
    CALL = "Call"
    PUT = "Put"


class SamplingType(str, Enum):
    STANDARD = "Standard"
    QUASI = "Quasi"
    QUASI_RANDOM = "Quasi Random"


@dataclass(frozen=True)
class SimulationConfig:
    """All market, product, and numerical inputs required for one run."""

    model: ModelType
    discretization: DiscretizationType
    payoff: PayoffType
    option_type: OptionType
    sampling_type: SamplingType
    start_price: float
    strike: float
    maturity: float
    risk_free_rate: float
    volatility: float
    num_paths: int
    num_steps: int
    asian_averaging_months: float = 12.0
    random_seed: int | None = 42
    jump_intensity: float = 0.0
    jump_mean: float = 0.0
    jump_volatility: float = 0.0

    def __post_init__(self) -> None:
        finite_values = {
            "start_price": self.start_price,
            "strike": self.strike,
            "maturity": self.maturity,
            "risk_free_rate": self.risk_free_rate,
            "volatility": self.volatility,
            "asian_averaging_months": self.asian_averaging_months,
            "jump_intensity": self.jump_intensity,
            "jump_mean": self.jump_mean,
            "jump_volatility": self.jump_volatility,
        }
        for name, value in finite_values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number.")

        if self.start_price <= 0:
            raise ValueError("Start price must be greater than zero.")
        if self.strike <= 0:
            raise ValueError("Strike must be greater than zero.")
        if self.maturity <= 0:
            raise ValueError("Maturity must be greater than zero.")
        if self.volatility < 0:
            raise ValueError("Volatility cannot be negative.")
        if self.num_paths < 2:
            raise ValueError("At least two paths are required to estimate variance.")
        if self.num_steps < 1:
            raise ValueError("At least one time step is required.")
        if self.payoff is PayoffType.ASIAN:
            if self.asian_averaging_months <= 0:
                raise ValueError("Asian averaging months must be greater than zero.")
            if self.asian_averaging_months > self.maturity * 12:
                raise ValueError(
                    "Asian averaging months cannot exceed the option maturity."
                )
        # Each path stores every simulated time point as a float64. Guarding the
        # total size prevents an accidental multi-gigabyte allocation in the UI.
        if self.num_paths * (self.num_steps + 1) > 50_000_000:
            raise ValueError(
                "This paths × steps combination is too large. Reduce either "
                "input so that stored path points do not exceed 50 million."
            )
        if self.jump_intensity < 0:
            raise ValueError("Jump intensity cannot be negative.")
        if self.jump_volatility < 0:
            raise ValueError("Jump volatility cannot be negative.")
