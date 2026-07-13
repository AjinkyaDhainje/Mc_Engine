"""High-level orchestration for simulations and charts."""

from time import perf_counter

import numpy as np

from .config import SimulationConfig
from .engine import Engine
from .payoff import Payoff
from .results import SimulationResult
from .visualisation import Visualisation


class Manager:
    """Call the engine, payoff calculator, and visualisation in order."""

    def __init__(
        self,
        engine: Engine | None = None,
        payoff: Payoff | None = None,
        visualisation: Visualisation | None = None,
    ) -> None:
        self.engine = engine or Engine()
        self.payoff = payoff or Payoff()
        self.visualisation = visualisation or Visualisation()

    def start(self, config: SimulationConfig) -> SimulationResult:
        """Run one pricing simulation and return all numerical outputs."""
        started_at = perf_counter()
        paths = self.engine.generate_paths(config)
        raw_payoffs = self.payoff.calculate(paths, config)
        discounted_payoffs = self.payoff.discount(
            raw_payoffs, config.risk_free_rate, config.maturity
        )

        option_price = float(np.mean(discounted_payoffs))
        payoff_variance = float(np.var(discounted_payoffs, ddof=1))
        standard_error = float(np.sqrt(payoff_variance / config.num_paths))
        elapsed_seconds = perf_counter() - started_at

        return SimulationResult(
            config=config,
            paths=paths,
            raw_payoffs=raw_payoffs,
            discounted_payoffs=discounted_payoffs,
            option_price=option_price,
            payoff_variance=payoff_variance,
            standard_error=standard_error,
            elapsed_seconds=elapsed_seconds,
        )

    def visualise(self, result: SimulationResult) -> dict[str, object]:
        """Create every chart requested by the UI."""
        return self.visualisation.create_all(result)
