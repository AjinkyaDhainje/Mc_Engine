"""Matplotlib visualisations for simulation diagnostics."""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from .results import SimulationResult


class Visualisation:
    """Create charts from only the arrays each chart actually needs."""

    def create_all(self, result: SimulationResult) -> dict[str, Figure]:
        return {
            "paths": self.plot_paths(result),
            "terminal_distribution": self.plot_terminal_distribution(result),
            "payoff_distribution": self.plot_payoff_distribution(result),
            "price_convergence": self.plot_price_convergence(result),
        }

    @staticmethod
    def _new_figure(title: str, xlabel: str, ylabel: str) -> tuple[Figure, object]:
        figure, axes = plt.subplots(figsize=(9, 5))
        axes.set_title(title)
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.grid(alpha=0.25)
        return figure, axes

    def plot_paths(self, result: SimulationResult, max_paths: int = 100) -> Figure:
        figure, axes = self._new_figure(
            "Sample simulated asset paths", "Time (years)", "Asset price"
        )
        count = min(max_paths, len(result.display_paths))
        # The result contains at most the first 10,000 paths. Select 100 evenly
        # spaced rows from that UI-safe subset rather than sending every path.
        indices = np.linspace(0, len(result.display_paths) - 1, count, dtype=int)
        axes.plot(
            result.time_grid,
            result.display_paths[indices].T,
            alpha=0.35,
            linewidth=0.8,
        )
        axes.axhline(
            result.config.strike,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=f"Strike = {result.config.strike:.2f}",
        )
        axes.legend()
        figure.tight_layout()
        return figure

    def plot_terminal_distribution(self, result: SimulationResult) -> Figure:
        figure, axes = self._new_figure(
            "Final asset-price distribution", "Asset price at maturity", "Frequency"
        )
        terminal_prices = result.terminal_prices
        axes.hist(terminal_prices, bins=50, color="#4169E1", alpha=0.8)
        axes.axvline(
            np.mean(terminal_prices),
            color="darkred",
            linestyle="--",
            label=f"Mean = {np.mean(terminal_prices):.2f}",
        )
        axes.legend()
        figure.tight_layout()
        return figure

    def plot_payoff_distribution(self, result: SimulationResult) -> Figure:
        figure, axes = self._new_figure(
            "Discounted payoff distribution", "Present value of payoff", "Frequency"
        )
        axes.hist(result.discounted_payoffs, bins=50, color="#2E8B57", alpha=0.8)
        axes.axvline(
            result.option_price,
            color="darkred",
            linestyle="--",
            label=f"Estimated price = {result.option_price:.4f}",
        )
        axes.legend()
        figure.tight_layout()
        return figure

    def plot_price_convergence(self, result: SimulationResult) -> Figure:
        figure, axes = self._new_figure(
            "Monte Carlo price convergence", "Number of paths", "Option-price estimate"
        )
        values = result.discounted_payoffs
        sample_counts = np.arange(1, len(values) + 1)
        running_mean = np.cumsum(values) / sample_counts

        cumulative_sum_squares = np.cumsum(values**2)
        running_std = np.zeros_like(running_mean)
        if len(values) > 1:
            centered_sum_squares = (
                cumulative_sum_squares[1:]
                - sample_counts[1:] * running_mean[1:] ** 2
            )
            running_std[1:] = np.sqrt(
                np.maximum(centered_sum_squares, 0.0) / (sample_counts[1:] - 1)
            )
        confidence_half_width = 1.96 * running_std / np.sqrt(sample_counts)

        shown = self._sample_indices(len(values))
        axes.plot(sample_counts[shown], running_mean[shown], color="#4169E1")
        axes.fill_between(
            sample_counts[shown],
            (running_mean - confidence_half_width)[shown],
            (running_mean + confidence_half_width)[shown],
            color="#4169E1",
            alpha=0.18,
            label="Approx. 95% confidence interval",
        )
        axes.axhline(
            result.option_price,
            color="darkred",
            linestyle="--",
            label="Final estimate",
        )
        axes.legend()
        figure.tight_layout()
        return figure

    @staticmethod
    def _sample_indices(
        length: int, max_points: int = 1500, start: int = 0
    ) -> np.ndarray:
        """Downsample long convergence series without altering calculations."""
        if length - start <= max_points:
            return np.arange(start, length)
        return np.unique(np.linspace(start, length - 1, max_points, dtype=int))
