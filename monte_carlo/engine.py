"""Memory-bounded path generation and registry-based model dispatch."""

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .config import DiscretizationType, ModelType, SamplingType, SimulationConfig
from .models import Model
from .sampling import Sampling


FloatArray = NDArray[np.float64]
StepMethod = Callable[..., FloatArray]
RandomInputMethod = Callable[
    [np.random.Generator, Mapping[str, float], float, int], dict[str, object]
]


class SamplingStream(Protocol):
    def draw(self, num_paths: int) -> FloatArray: ...


SamplingStreamFactory = Callable[[int], SamplingStream]


@dataclass(frozen=True)
class ModelDefinition:
    step_methods: Mapping[DiscretizationType, StepMethod]
    random_inputs: RandomInputMethod


class Engine:
    """Generate paths by selecting the requested model equation."""

    _MAX_BATCH_PATH_POINTS = 5_000_000

    _MODELS: dict[ModelType, ModelDefinition] = {
        ModelType.GBM: ModelDefinition(
            step_methods={
                DiscretizationType.EULER: Model.geometric_brownian_euler,
                DiscretizationType.MILSTEIN: Model.geometric_brownian_milstein,
            },
            random_inputs=lambda _rng, _parameters, _dt, _count: {},
        ),
        ModelType.MERTON_JUMP: ModelDefinition(
            step_methods={
                DiscretizationType.EULER: Model.merton_jump_euler,
                DiscretizationType.MILSTEIN: Model.merton_jump_milstein,
            },
            random_inputs=lambda rng, parameters, dt, count: Engine._merton_inputs(
                rng, parameters, dt, count
            ),
        ),
    }

    _SAMPLING_STREAMS: dict[SamplingType, SamplingStreamFactory] = {
        SamplingType.STANDARD: Sampling.standard_stream,
        SamplingType.QUASI: Sampling.quasi_stream,
        SamplingType.QUASI_RANDOM: Sampling.quasi_random_stream,
    }

    def generate_paths(self, config: SimulationConfig) -> FloatArray:
        """Return all paths; the Manager uses batches to keep UI runs bounded."""
        return np.vstack(tuple(self.generate_path_batches(config)))

    def generate_path_batches(
        self, config: SimulationConfig, batch_size: int | None = None
    ) -> Iterator[FloatArray]:
        """Yield complete path batches while one sampling stream stays active.

        The automatic batch size caps the two-dimensional working arrays even
        when the user combines many paths with many time steps.
        """
        if batch_size is None:
            batch_size = min(
                10_000,
                max(1, self._MAX_BATCH_PATH_POINTS // (config.num_steps + 1)),
            )
        if batch_size < 1:
            raise ValueError("Batch size must be at least one path.")
        try:
            sampling_stream = self._SAMPLING_STREAMS[config.sampling_type](
                config.num_steps
            )
        except KeyError as error:
            raise ValueError(
                f"Unsupported sampling type: {config.sampling_type.value}."
            ) from error

        try:
            model_definition = self._MODELS[config.model]
            step_method = model_definition.step_methods[config.discretization]
        except KeyError as error:
            raise ValueError(
                f"Unsupported combination: {config.model.value} with "
                f"{config.discretization.value}."
            ) from error

        jump_rng = np.random.default_rng()
        dt = config.maturity / config.num_steps
        completed = 0
        while completed < config.num_paths:
            count = min(batch_size, config.num_paths - completed)
            normal_draws = sampling_stream.draw(count)
            paths = np.empty((count, config.num_steps + 1), dtype=float)
            paths[:, 0] = config.start_price

            for step in range(1, config.num_steps + 1):
                step_inputs = model_definition.random_inputs(
                    jump_rng, config.model_parameters, dt, count
                )
                paths[:, step] = step_method(
                    current_price=paths[:, step - 1],
                    risk_free_rate=config.risk_free_rate,
                    volatility=config.volatility,
                    dt=dt,
                    normal_draw=normal_draws[:, step - 1],
                    **config.model_parameters,
                    **step_inputs,
                )

            completed += count
            yield paths

    @staticmethod
    def _merton_inputs(
        rng: np.random.Generator,
        parameters: Mapping[str, float],
        dt: float,
        count: int,
    ) -> dict[str, object]:
        """Generate the model-specific compound jump for one time step."""
        jump_counts = rng.poisson(parameters["jump_intensity"] * dt, count)
        jump_normal = rng.standard_normal(count)
        jump_log_sum = (
            jump_counts * parameters["jump_mean"]
            + np.sqrt(jump_counts)
            * parameters["jump_volatility"]
            * jump_normal
        )
        return {"jump_log_sum": jump_log_sum}
