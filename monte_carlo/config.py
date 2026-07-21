"""Validated inputs, component choices, and extensible input definitions."""

from collections.abc import Mapping
from dataclasses import dataclass, field
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
class ParameterSpec:
    """Describe one component-specific number without coupling it to the UI."""

    key: str
    label: str
    default: float
    min_value: float | None = None
    step: float = 0.01
    number_format: str | None = None
    help: str | None = None
    maturity_days_limit: bool = False


MODEL_PARAMETER_SPECS: dict[ModelType, tuple[ParameterSpec, ...]] = {
    ModelType.GBM: (),
    ModelType.MERTON_JUMP: (
        ParameterSpec(
            "jump_intensity",
            "Jump intensity (expected jumps/year)",
            0.75,
            min_value=0.0,
            step=0.05,
        ),
        ParameterSpec(
            "jump_mean", "Mean log-jump size", -0.10, step=0.01, number_format="%.4f"
        ),
        ParameterSpec(
            "jump_volatility",
            "Log-jump volatility",
            0.20,
            min_value=0.0,
            step=0.01,
            number_format="%.4f",
        ),
    ),
}


PAYOFF_PARAMETER_SPECS: dict[PayoffType, tuple[ParameterSpec, ...]] = {
    PayoffType.VANILLA: (),
    PayoffType.ASIAN: (
        ParameterSpec(
            "averaging_days",
            "Average over final days",
            365.0,
            min_value=0.01,
            step=1.0,
            help=(
                "The Asian payoff uses simulated prices from this many days "
                "immediately before maturity."
            ),
            maturity_days_limit=True,
        ),
    ),
}


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
    model_parameters: Mapping[str, float] = field(default_factory=dict)
    payoff_parameters: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        finite_values = {
            "start_price": self.start_price,
            "strike": self.strike,
            "maturity": self.maturity,
            "risk_free_rate": self.risk_free_rate,
            "volatility": self.volatility,
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
            raise ValueError("At least two paths are required to estimate uncertainty.")
        if self.num_paths > 1_000_000:
            raise ValueError("The number of paths cannot exceed 1,000,000.")
        if self.num_steps < 1:
            raise ValueError("At least one time step is required.")

        self._validate_component_parameters(
            self.model_parameters, MODEL_PARAMETER_SPECS[self.model], "model"
        )
        self._validate_component_parameters(
            self.payoff_parameters, PAYOFF_PARAMETER_SPECS[self.payoff], "payoff"
        )

    def _validate_component_parameters(
        self,
        values: Mapping[str, float],
        specs: tuple[ParameterSpec, ...],
        component_name: str,
    ) -> None:
        expected = {spec.key for spec in specs}
        received = set(values)
        if received != expected:
            missing = sorted(expected - received)
            unexpected = sorted(received - expected)
            details = []
            if missing:
                details.append(f"missing {missing}")
            if unexpected:
                details.append(f"unexpected {unexpected}")
            raise ValueError(
                f"Invalid {component_name} parameters: {', '.join(details)}."
            )

        for spec in specs:
            value = values[spec.key]
            if not math.isfinite(value):
                raise ValueError(f"{spec.label} must be a finite number.")
            if spec.min_value is not None and value < spec.min_value:
                raise ValueError(
                    f"{spec.label} must be at least {spec.min_value}."
                )
            if spec.maturity_days_limit and value > self.maturity * 365.0:
                raise ValueError(
                    f"{spec.label} cannot exceed the option maturity in days."
                )
