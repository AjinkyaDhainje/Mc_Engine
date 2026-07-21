"""Normal random-number sampling methods used by the path engine."""

import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm, qmc


FloatArray = NDArray[np.float64]


class _StandardStream:
    def __init__(self, num_steps: int) -> None:
        self.num_steps = num_steps
        self.rng = np.random.default_rng()

    def draw(self, num_paths: int) -> FloatArray:
        return self.rng.standard_normal((num_paths, self.num_steps))


class _SobolStream:
    def __init__(self, num_steps: int, scramble: bool, skip_first: bool) -> None:
        self.sobol = qmc.Sobol(d=num_steps, scramble=scramble)
        if skip_first:
            self.sobol.fast_forward(1)

    def draw(self, num_paths: int) -> FloatArray:
        uniforms = Sampling._sobol_sample(self.sobol, num_paths)
        return Sampling._uniform_to_normal(uniforms)


class Sampling:
    """Generate only the standard-normal shocks ``Z`` used by a model.

    Every public method returns an array with shape ``(num_paths, num_steps)``.
    The class does not calculate prices, drift, volatility, time increments, or
    payoffs. Those responsibilities remain in Model, Engine, and Payoff. Keeping
    sampling independent means the same random-number method can be reused by
    every current or future stochastic model.

    The columns represent time steps and the rows represent simulated paths.
    All returned values have an approximately standard normal distribution:

        Z ~ Normal(0, 1).
    """

    @staticmethod
    def standard(num_paths: int, num_steps: int) -> FloatArray:
        """Return independent pseudo-random standard-normal values.

        NumPy's random generator first produces pseudo-random uniform bits and
        transforms them into independent Normal(0, 1) values.

        This is ordinary Monte Carlo sampling. Its estimator error normally
        decreases at a rate proportional to 1/sqrt(num_paths).
        """
        return Sampling.standard_stream(num_steps).draw(num_paths)

    @staticmethod
    def quasi(num_paths: int, num_steps: int) -> FloatArray:
        """Return deterministic quasi-Monte Carlo normal values.

        A non-scrambled Sobol sequence fills the unit hypercube more evenly than
        independent pseudo-random points. Each Sobol dimension represents one
        simulation time step. The inverse normal CDF then transforms every
        uniform coordinate U into a standard-normal shock:

            Z = Phi^(-1)(U).

        The first deterministic Sobol point is all zeros. It is skipped because
        Phi^(-1)(0) is negative infinity. This method intentionally ignores the
        Sobol balance is strongest when ``num_paths`` is a power of two,
        although arbitrary positive path counts are supported here.
        """
        return Sampling.quasi_stream(num_steps).draw(num_paths)

    @staticmethod
    def quasi_random(num_paths: int, num_steps: int) -> FloatArray:
        """Return randomized quasi-Monte Carlo normal values.

        This uses a scrambled Sobol sequence. Scrambling retains the improved
        space-filling behaviour of quasi-Monte Carlo while adding randomness,
        which permits independent repeated runs and conventional error studies.
        As in ``quasi``, the inverse normal CDF maps Sobol uniforms to Normal(0,
        1) shocks.

        This option is called "Quasi Random" in the UI to distinguish it from
        the completely deterministic non-scrambled Sobol sequence.
        """
        return Sampling.quasi_random_stream(num_steps).draw(num_paths)

    @staticmethod
    def standard_stream(num_steps: int) -> _StandardStream:
        """Return a stateful standard-Monte-Carlo stream for batched runs."""
        return _StandardStream(num_steps)

    @staticmethod
    def quasi_stream(num_steps: int) -> _SobolStream:
        """Return one continuous deterministic Sobol stream."""
        return _SobolStream(num_steps, scramble=False, skip_first=True)

    @staticmethod
    def quasi_random_stream(num_steps: int) -> _SobolStream:
        """Return one continuous scrambled Sobol stream."""
        return _SobolStream(num_steps, scramble=True, skip_first=False)

    @staticmethod
    def _sobol_sample(sobol: qmc.Sobol, num_paths: int) -> FloatArray:
        """Draw an arbitrary number of Sobol points without padding the matrix.

        SciPy warns when a Sobol sample size is not a power of two because the
        strongest balance guarantee then does not apply. The output is still a
        valid Sobol sequence, so the warning is suppressed locally. This avoids
        allocating and then discarding a potentially large padded matrix.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The balance properties of Sobol.*",
                category=UserWarning,
            )
            return sobol.random(num_paths)

    @staticmethod
    def _uniform_to_normal(uniforms: FloatArray) -> FloatArray:
        """Transform uniform Sobol coordinates to finite normal values.

        The inverse normal CDF is infinite at exactly zero or one. Clipping only
        those numerical endpoints keeps every generated Z finite without
        materially changing interior Sobol points.
        """
        epsilon = np.finfo(float).eps
        safe_uniforms = np.clip(uniforms, epsilon, 1.0 - epsilon)
        return norm.ppf(safe_uniforms)
