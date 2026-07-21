"""Option payoff equations and discounting."""

import numpy as np
from numpy.typing import NDArray

from .config import OptionType, PayoffType, SimulationConfig


FloatArray = NDArray[np.float64]


class Payoff:
    """Calculate one payoff per simulated path."""

    _METHOD_NAMES = {
        PayoffType.VANILLA: "vanilla",
        PayoffType.ASIAN: "asian",
    }
    _DIRECTIONS = {OptionType.CALL: 1.0, OptionType.PUT: -1.0}

    def calculate(self, paths: FloatArray, config: SimulationConfig) -> FloatArray:
        """Run the registered payoff through one uniform calling convention."""
        try:
            method = getattr(self, self._METHOD_NAMES[config.payoff])
        except KeyError as error:
            raise ValueError(f"Unsupported payoff: {config.payoff.value}.") from error
        return method(
            paths=paths,
            strike=config.strike,
            option_type=config.option_type,
            maturity=config.maturity,
            **config.payoff_parameters,
        )

    @staticmethod
    def vanilla(
        paths: FloatArray,
        strike: float,
        option_type: OptionType,
        **_: object,
    ) -> FloatArray:
        """Return the European vanilla payoff for every path.

        Only the terminal price S_T matters. At maturity:

            Call payoff = max(S_T - K, 0)
            Put payoff  = max(K - S_T, 0)

        where K is the strike. The max with zero means the holder exercises the
        option only when exercising is beneficial.
        """
        terminal_prices = paths[:, -1]
        direction = Payoff._DIRECTIONS[option_type]
        return np.maximum(direction * (terminal_prices - strike), 0.0)

    @staticmethod
    def asian(
        paths: FloatArray,
        strike: float,
        option_type: OptionType,
        maturity: float,
        averaging_days: float,
    ) -> FloatArray:
        """Return an arithmetic-average Asian payoff for every path.

        The user chooses how many days immediately before maturity form the
        averaging window. If dt = maturity / num_steps, the number of simulated
        observations in that window is

            M = ceil((averaging_days / 365) / dt).

        The calculation takes the final M observations from each path:

            A = (S_(T-M+1) + ... + S_T) / M.

        The fixed-strike payoff is then

            Asian call = max(A - K, 0)
            Asian put  = max(K - A, 0).

        Using ceil means the selected window is never shortened when its length
        does not divide exactly into the simulation grid. When the selected
        window is the full maturity, the known time-zero price is still excluded
        and all simulated post-inception observations are used.
        """
        num_steps = paths.shape[1] - 1
        dt = maturity / num_steps
        averaging_years = averaging_days / 365.0
        observations = min(num_steps, max(1, int(np.ceil(averaging_years / dt))))
        averaging_window = paths[:, -observations:]
        arithmetic_average = np.mean(averaging_window, axis=1)
        direction = Payoff._DIRECTIONS[option_type]
        return np.maximum(direction * (arithmetic_average - strike), 0.0)

    @staticmethod
    def discount(payoffs: FloatArray, risk_free_rate: float, maturity: float) -> FloatArray:
        """Discount maturity cashflows: PV = payoff * exp(-rT)."""
        return payoffs * np.exp(-risk_free_rate * maturity)
