# Configurable Monte Carlo Option-Pricing Engine

A modular Python application in which a user selects:

- Model: Geometric Brownian Motion or Merton Jump-Diffusion
- Discretization: Euler or Milstein
- Payoff: European vanilla or arithmetic-average Asian
- Option direction: call or put
- Sampling: standard Monte Carlo, deterministic quasi-Monte Carlo, or
  randomized quasi-Monte Carlo

It returns the discounted Monte Carlo option price, payoff variance, standard
error, approximate 95% interval, runtime, and five diagnostic charts.

## Project structure

```text
configurable_monte_carlo_engine/
├── app.py                         # Streamlit UI
├── monte_carlo/
│   ├── config.py                  # Validated inputs and enums
│   ├── models.py                  # Four commented one-step equations
│   ├── sampling.py                # Standard, Sobol, and scrambled Sobol Z values
│   ├── engine.py                  # Path generation and method dispatch
│   ├── payoff.py                  # Vanilla/Asian payoff mathematics
│   ├── manager.py                 # Runs the components in order
│   ├── results.py                 # Typed simulation output
│   └── visualisation.py           # Charts and convergence diagnostics
├── tests/test_monte_carlo.py      # Numerical and smoke tests
└── requirements.txt
```

The call flow is:

```text
UI -> Manager -> Engine -> selected Model equation -> Payoff -> Result
                   Result -> Visualisation -> UI
```

## Run the application

Python 3.10 or newer is recommended.

```bash
cd configurable_monte_carlo_engine
python -m venv .venv
```

Activate the environment:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

## Inputs

All models use start price, strike, maturity, risk-free rate, volatility, number
of paths, number of time steps, and random seed. The Merton model additionally
uses:

- Jump intensity: expected number of jumps per year, lambda
- Mean log-jump size: mu_J in `log(J) ~ Normal(mu_J, sigma_J^2)`
- Log-jump volatility: sigma_J

For an Asian payoff, `asian_averaging_months` selects the final part of the path
used in the arithmetic average. For example, a value of `3` averages simulated
prices over the final three months before maturity.

The sampling choices are:

- Standard: independent pseudo-random Normal(0, 1) shocks
- Quasi: deterministic non-scrambled Sobol points transformed to Normal(0, 1)
- Quasi Random: scrambled Sobol points transformed to Normal(0, 1)

Sobol sequences have their strongest balance properties when the number of
paths is a power of two.

Rates and volatilities use decimal notation: enter `0.05` for 5%.

## Adding another model

1. Add its name to `ModelType` in `monte_carlo/config.py`.
2. Add Euler and Milstein one-step methods to `Model` in `models.py`.
3. Register both combinations in `Engine._METHODS` in `engine.py`.
4. Add any model-specific inputs to `SimulationConfig` and the UI.

The Manager, Payoff, Result, and Visualisation classes do not need to change
when a new path model is introduced.

## Tests

From the project directory:

```bash
python -m unittest discover -v
```

The tests execute all 48 model/discretization/payoff/direction/sampling
combinations, validate the Asian averaging window, reject invalid input, and
compare a large GBM run with Black-Scholes.

## Numerical notes

- Pricing assumes a constant risk-free rate and risk-neutral dynamics.
- The reported variance is the sample variance of discounted path payoffs.
- The standard error is `sqrt(variance / number_of_paths)`.
- Euler and Milstein are educational discretizations. GBM also has an exact
  transition, but it is intentionally not used because the application compares
  the two requested numerical methods.
- Euler-style price updates are floored at zero when a large step would produce
  a negative value.
