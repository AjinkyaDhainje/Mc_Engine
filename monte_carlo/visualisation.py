"""Matplotlib visualisations for simulation diagnostics."""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from .results import SimulationResult


class Visualisation:
    """Create paths, distributions, convergence, and variance charts."""

    def create_all(self, result: SimulationResult) -> dict[str, Figure]:
        return {
            "paths": self.plot_paths(result),
            "terminal_distribution": self.plot_terminal_distribution(result),
            "payoff_distribution": self.plot_payoff_distribution(result),
            "price_convergence": self.plot_price_convergence(result),
            "variance_convergence": self.plot_variance_of_mean_convergence(result),
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
        count = min(max_paths, result.config.num_paths)
        # Evenly spaced indices make the displayed sample deterministic.
        indices = np.linspace(0, result.config.num_paths - 1, count, dtype=int)
        axes.plot(result.time_grid, result.paths[indices].T, alpha=0.35, linewidth=0.8)
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
        terminal_prices = result.paths[:, -1]
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
        running_variance = np.zeros_like(running_mean)
        if len(values) > 1:
            running_variance[1:] = (
                cumulative_sum_squares[1:]
                - sample_counts[1:] * running_mean[1:] ** 2
            ) / (sample_counts[1:] - 1)
            running_variance[1:] = np.maximum(running_variance[1:], 0.0)
        confidence_half_width = 1.96 * np.sqrt(
            running_variance / np.maximum(sample_counts, 1)
        )

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
        axes.axhline(result.option_price, color="darkred", linestyle="--", label="Final estimate")
        axes.legend()
        figure.tight_layout()
        return figure

    def plot_variance_convergence(self, result: SimulationResult) -> Figure:
        figure, axes = self._new_figure(
            "Payoff-variance convergence", "Number of paths", "Sample variance"
        )
        values = result.discounted_payoffs
        sample_counts = np.arange(1, len(values) + 1)
        running_mean = np.cumsum(values) / sample_counts
        cumulative_sum_squares = np.cumsum(values**2)
        running_variance = np.zeros_like(values)
        if len(values) > 1:
            running_variance[1:] = (
                cumulative_sum_squares[1:]
                - sample_counts[1:] * running_mean[1:] ** 2
            ) / (sample_counts[1:] - 1)
            running_variance[1:] = np.maximum(running_variance[1:], 0.0)

        shown = self._sample_indices(len(values), start=1)
        axes.plot(sample_counts[shown], running_variance[shown], color="#8A2BE2")
        axes.axhline(
            result.payoff_variance,
            color="darkred",
            linestyle="--",
            label=f"Final variance = {result.payoff_variance:.4f}",
        )
        axes.legend()
        figure.tight_layout()
        return figure
    
    def plot_variance_of_mean_convergence(
        self, result: SimulationResult
    ) -> Figure:
        figure, axes = self._new_figure(
            "Variance of Monte Carlo mean convergence",
            "Number of paths",
            "Estimated variance of mean",
        )

        values = result.discounted_payoffs
        sample_counts = np.arange(1, len(values) + 1)

        running_mean = np.cumsum(values) / sample_counts
        cumulative_sum_squares = np.cumsum(values**2)

        # Sample variance of the individual discounted payoffs.
        running_sample_variance = np.zeros_like(values, dtype=float)

        if len(values) > 1:
            running_sample_variance[1:] = (
                cumulative_sum_squares[1:]
                - sample_counts[1:] * running_mean[1:] ** 2
            ) / (sample_counts[1:] - 1)

            # Protect against tiny negative values caused by floating-point errors.
            running_sample_variance[1:] = np.maximum(
                running_sample_variance[1:], 0.0
            )

        # Variance of the Monte Carlo mean:
        #
        #     Var(mean) = sample variance / number of paths
        #
        # This is also the square of the Monte Carlo standard error.
        running_variance_of_mean = (
            running_sample_variance / sample_counts
        )

        shown = self._sample_indices(len(values), start=1)

        axes.plot(
            sample_counts[shown],
            running_variance_of_mean[shown],
            color="#8A2BE2",
        )

        final_variance_of_mean = (
            result.payoff_variance / result.config.num_paths
        )

        axes.axhline(
            final_variance_of_mean,
            color="darkred",
            linestyle="--",
            label=(
                "Final variance of mean = "
                f"{final_variance_of_mean:.8f}"
            ),
        )

        axes.legend()
        figure.tight_layout()

        return figure

    @staticmethod
    def _sample_indices(length: int, max_points: int = 1500, start: int = 0) -> np.ndarray:
        """Downsample long convergence series without altering calculations."""
        if length - start <= max_points:
            return np.arange(start, length)
        return np.unique(np.linspace(start, length - 1, max_points, dtype=int))
