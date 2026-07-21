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
        """Price every path while retaining at most 10,000 paths for display."""
        started_at = perf_counter()
        display_limit = min(config.num_paths, 10_000)
        display_paths = np.empty(
            (display_limit, config.num_steps + 1), dtype=float
        )
        terminal_prices = np.empty(config.num_paths, dtype=float)
        discounted_payoffs = np.empty(config.num_paths, dtype=float)

        offset = 0
        display_offset = 0
        for paths in self.engine.generate_path_batches(config):
            count = len(paths)
            raw_payoffs = self.payoff.calculate(paths, config)
            batch_payoffs = self.payoff.discount(
                raw_payoffs, config.risk_free_rate, config.maturity
            )
            terminal_prices[offset : offset + count] = paths[:, -1]
            discounted_payoffs[offset : offset + count] = batch_payoffs

            retained = min(count, display_limit - display_offset)
            if retained > 0:
                display_paths[display_offset : display_offset + retained] = paths[
                    :retained
                ]
                display_offset += retained
            offset += count

        option_price = float(np.mean(discounted_payoffs))
        standard_error = float(
            np.std(discounted_payoffs, ddof=1) / np.sqrt(config.num_paths)
        )
        confidence_half_width = 1.96 * standard_error
        elapsed_seconds = perf_counter() - started_at

        return SimulationResult(
            config=config,
            display_paths=display_paths,
            terminal_prices=terminal_prices,
            discounted_payoffs=discounted_payoffs,
            option_price=option_price,
            standard_error=standard_error,
            confidence_interval_low=option_price - confidence_half_width,
            confidence_interval_high=option_price + confidence_half_width,
            elapsed_seconds=elapsed_seconds,
        )

    def visualise(self, result: SimulationResult) -> dict[str, object]:
        """Create every chart requested by the UI."""
        return self.visualisation.create_all(result)
