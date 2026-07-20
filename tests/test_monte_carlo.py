"""Numerical smoke tests for all supported simulation combinations."""

import math
import unittest
from dataclasses import replace

import numpy as np

from monte_carlo import (
    DiscretizationType,
    Manager,
    ModelType,
    OptionType,
    PayoffType,
    SamplingType,
    SimulationConfig,
)
from monte_carlo.payoff import Payoff
from monte_carlo.sampling import Sampling


class MonteCarloTests(unittest.TestCase):
    def make_config(self, model: ModelType, method: DiscretizationType) -> SimulationConfig:
        return SimulationConfig(
            model=model,
            discretization=method,
            payoff=PayoffType.VANILLA,
            option_type=OptionType.CALL,
            sampling_type=SamplingType.STANDARD,
            start_price=100.0,
            strike=100.0,
            maturity=1.0,
            risk_free_rate=0.05,
            volatility=0.2,
            num_paths=2_000,
            num_steps=64,
            asian_averaging_months=12.0,
            random_seed=7,
            jump_intensity=0.75,
            jump_mean=-0.1,
            jump_volatility=0.2,
        )

    def test_every_supported_combination_runs(self) -> None:
        manager = Manager()
        for model in ModelType:
            for method in DiscretizationType:
                for payoff in PayoffType:
                    for option_type in OptionType:
                        for sampling_type in SamplingType:
                            with self.subTest(
                                model=model,
                                method=method,
                                payoff=payoff,
                                option_type=option_type,
                                sampling_type=sampling_type,
                            ):
                                config = replace(
                                    self.make_config(model, method),
                                    payoff=payoff,
                                    option_type=option_type,
                                    sampling_type=sampling_type,
                                )
                                result = manager.start(config)
                                self.assertEqual(result.paths.shape, (2_000, 65))
                                self.assertEqual(
                                    result.discounted_payoffs.shape, (2_000,)
                                )
                                self.assertTrue(np.all(np.isfinite(result.paths)))
                                self.assertTrue(np.all(result.paths >= 0.0))
                                self.assertGreaterEqual(result.option_price, 0.0)
                                self.assertGreaterEqual(result.payoff_variance, 0.0)

    def test_sampling_methods_return_finite_z_matrices(self) -> None:
        methods = [Sampling.standard, Sampling.quasi, Sampling.quasi_random]
        for method in methods:
            with self.subTest(method=method.__name__):
                values = method(num_paths=128, num_steps=12, random_seed=11)
                self.assertEqual(values.shape, (128, 12))
                self.assertTrue(np.all(np.isfinite(values)))

    def test_asian_payoff_uses_only_selected_final_months(self) -> None:
        # Twelve monthly observations contain prices 1 through 12. Selecting the
        # final three months averages 10, 11, and 12, producing a call payoff of 1.
        paths = np.arange(13, dtype=float).reshape(1, 13)
        payoff = Payoff.asian(
            paths=paths,
            strike=10.0,
            option_type=OptionType.CALL,
            maturity=1.0,
            averaging_months=3.0,
        )
        np.testing.assert_allclose(payoff, np.array([1.0]))

    def test_gbm_price_is_close_to_black_scholes(self) -> None:
        config = replace(
            self.make_config(ModelType.GBM, DiscretizationType.MILSTEIN),
            num_paths=50_000,
            num_steps=128,
        )
        result = Manager().start(config)

        # Black-Scholes benchmark for S=K=100, r=5%, sigma=20%, T=1.
        normal_cdf = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
        d1 = 0.35
        d2 = 0.15
        benchmark = 100.0 * normal_cdf(d1) - 100.0 * math.exp(-0.05) * normal_cdf(d2)
        self.assertLess(abs(result.option_price - benchmark), 0.35)

    def test_invalid_input_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            replace(
                self.make_config(ModelType.GBM, DiscretizationType.EULER),
                volatility=-0.2,
            )

        with self.assertRaises(ValueError):
            replace(
                self.make_config(ModelType.GBM, DiscretizationType.EULER),
                payoff=PayoffType.ASIAN,
                maturity=1.0,
                asian_averaging_months=13.0,
            )


if __name__ == "__main__":
    unittest.main()
