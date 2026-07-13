"""One-step stochastic model equations used by the simulation engine."""

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


class Model:
    """Model/discretization equations.

    Every public method advances all simulated paths by one time step. Adding a
    future model follows the same pattern: implement one Euler method and one
    Milstein method, then register both methods in ``Engine._METHODS``.
    """

    @staticmethod
    def geometric_brownian_euler(
        current_price: FloatArray,
        risk_free_rate: float,
        volatility: float,
        dt: float,
        normal_draw: FloatArray,
        **_: object,
    ) -> FloatArray:
        """Advance GBM by Euler-Maruyama.

        Risk-neutral GBM is

            dS_t = r S_t dt + sigma S_t dW_t.

        Over a small step, dW is approximated by sqrt(dt) * Z, where Z follows
        a standard normal distribution. Euler-Maruyama therefore gives

            S_(t+dt) = S_t + r S_t dt + sigma S_t sqrt(dt) Z.

        Euler can rarely produce a negative asset value for a large step. Since
        an equity price cannot be negative, such values are floored at zero.
        """
        next_price = current_price * (
            1.0
            + risk_free_rate * dt
            + volatility * np.sqrt(dt) * normal_draw
        )
        return np.maximum(next_price, 0.0)

    @staticmethod
    def geometric_brownian_milstein(
        current_price: FloatArray,
        risk_free_rate: float,
        volatility: float,
        dt: float,
        normal_draw: FloatArray,
        **_: object,
    ) -> FloatArray:
        """Advance GBM using the Milstein discretization.

        For dS = a(S)dt + b(S)dW, Milstein adds

            0.5 * b(S) * b'(S) * ((dW)^2 - dt)

        to the Euler step. For GBM, b(S) = sigma*S and b'(S) = sigma, so

            S_(t+dt) = S_t + rS_t dt + sigma S_t sqrt(dt) Z
                       + 0.5 sigma^2 S_t dt (Z^2 - 1).

        This correction captures part of the curvature missed by Euler and has
        stronger pathwise convergence for one-dimensional diffusions.
        """
        diffusion = volatility * np.sqrt(dt) * normal_draw
        correction = 0.5 * volatility**2 * dt * (normal_draw**2 - 1.0)
        next_price = current_price * (
            1.0 + risk_free_rate * dt + diffusion + correction
        )
        return np.maximum(next_price, 0.0)

    @staticmethod
    def merton_jump_euler(
        current_price: FloatArray,
        risk_free_rate: float,
        volatility: float,
        dt: float,
        normal_draw: FloatArray,
        jump_log_sum: FloatArray,
        jump_intensity: float,
        jump_mean: float,
        jump_volatility: float,
        **_: object,
    ) -> FloatArray:
        """Advance the Merton jump-diffusion with an Euler diffusion step.

        Under the risk-neutral measure the model is

            dS_t/S_(t-) = (r - lambda*kappa)dt + sigma dW_t + (J-1)dN_t,

        where N is Poisson with intensity lambda, log(J) is Normal(mu_J,
        sigma_J^2), and

            kappa = E[J-1] = exp(mu_J + 0.5*sigma_J^2) - 1.

        The ``-lambda*kappa`` drift compensation makes the discounted asset a
        martingale. First Euler advances the continuous part:

            S_cont = S_t[1 + (r-lambda*kappa)dt + sigma*sqrt(dt)*Z].

        If N jumps occur in the step, their product is exp(sum(log J_i)); hence

            S_(t+dt) = S_cont * exp(jump_log_sum).
        """
        expected_relative_jump = (
            np.exp(jump_mean + 0.5 * jump_volatility**2) - 1.0
        )
        compensated_drift = risk_free_rate - jump_intensity * expected_relative_jump
        continuous_price = current_price * (
            1.0
            + compensated_drift * dt
            + volatility * np.sqrt(dt) * normal_draw
        )
        next_price = continuous_price * np.exp(jump_log_sum)
        return np.maximum(next_price, 0.0)

    @staticmethod
    def merton_jump_milstein(
        current_price: FloatArray,
        risk_free_rate: float,
        volatility: float,
        dt: float,
        normal_draw: FloatArray,
        jump_log_sum: FloatArray,
        jump_intensity: float,
        jump_mean: float,
        jump_volatility: float,
        **_: object,
    ) -> FloatArray:
        """Advance Merton jump-diffusion with a Milstein diffusion step.

        The jump process and compensator are the same as in Merton Euler. The
        continuous part receives the GBM Milstein correction:

            S_cont = S_t[1 + (r-lambda*kappa)dt + sigma*sqrt(dt)*Z
                         + 0.5*sigma^2*dt*(Z^2-1)].

        The compound jump multiplier is then applied exactly within the step:

            S_(t+dt) = S_cont * exp(sum(log J_i)).

        This is a jump-adapted approximation in which jump sizes are exact and
        only the continuous diffusion is discretized.
        """
        expected_relative_jump = (
            np.exp(jump_mean + 0.5 * jump_volatility**2) - 1.0
        )
        compensated_drift = risk_free_rate - jump_intensity * expected_relative_jump
        diffusion = volatility * np.sqrt(dt) * normal_draw
        correction = 0.5 * volatility**2 * dt * (normal_draw**2 - 1.0)
        continuous_price = current_price * (
            1.0 + compensated_drift * dt + diffusion + correction
        )
        next_price = continuous_price * np.exp(jump_log_sum)
        return np.maximum(next_price, 0.0)
