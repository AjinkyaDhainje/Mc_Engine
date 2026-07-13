"""Path generation and model-method dispatch."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from .config import DiscretizationType, ModelType, SimulationConfig
from .models import Model


FloatArray = NDArray[np.float64]
StepMethod = Callable[..., FloatArray]


class Engine:
    """Generate paths by selecting the requested model equation."""

    _METHODS: dict[tuple[ModelType, DiscretizationType], StepMethod] = {
        (ModelType.GBM, DiscretizationType.EULER): Model.geometric_brownian_euler,
        (ModelType.GBM, DiscretizationType.MILSTEIN): Model.geometric_brownian_milstein,
        (ModelType.MERTON_JUMP, DiscretizationType.EULER): Model.merton_jump_euler,
        (ModelType.MERTON_JUMP, DiscretizationType.MILSTEIN): Model.merton_jump_milstein,
    }

    def generate_paths(self, config: SimulationConfig) -> FloatArray:
        """Return an array shaped ``(num_paths, num_steps + 1)``."""
        rng = np.random.default_rng(config.random_seed)
        dt = config.maturity / config.num_steps
        paths = np.empty((config.num_paths, config.num_steps + 1), dtype=float)
        paths[:, 0] = config.start_price

        try:
            step_method = self._METHODS[(config.model, config.discretization)]
        except KeyError as error:
            raise ValueError(
                f"Unsupported combination: {config.model.value} with "
                f"{config.discretization.value}."
            ) from error

        for step in range(1, config.num_steps + 1):
            normal_draw = rng.standard_normal(config.num_paths)
            step_inputs: dict[str, object] = {}

            if config.model is ModelType.MERTON_JUMP:
                # Conditional on N jumps, the sum of N independent Normal jump
                # log-sizes is Normal(N*mu_J, N*sigma_J^2). This avoids an inner
                # loop over individual jumps while preserving the distribution.
                jump_counts = rng.poisson(config.jump_intensity * dt, config.num_paths)
                jump_normal = rng.standard_normal(config.num_paths)
                jump_log_sum = (
                    jump_counts * config.jump_mean
                    + np.sqrt(jump_counts) * config.jump_volatility * jump_normal
                )
                step_inputs = {
                    "jump_log_sum": jump_log_sum,
                    "jump_intensity": config.jump_intensity,
                    "jump_mean": config.jump_mean,
                    "jump_volatility": config.jump_volatility,
                }

            paths[:, step] = step_method(
                current_price=paths[:, step - 1],
                risk_free_rate=config.risk_free_rate,
                volatility=config.volatility,
                dt=dt,
                normal_draw=normal_draw,
                **step_inputs,
            )

        return paths
