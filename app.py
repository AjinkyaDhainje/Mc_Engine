"""Streamlit user interface for the configurable Monte Carlo engine."""

import matplotlib.pyplot as plt
import streamlit as st

from monte_carlo import (
    DiscretizationType,
    Manager,
    ModelType,
    OptionType,
    PayoffType,
    SamplingType,
    SimulationConfig,
)


st.set_page_config(page_title="Configurable Monte Carlo Engine", layout="wide")
st.title("Configurable Monte Carlo Option-Pricing Engine")
st.caption(
    "Select a stochastic model, discretization method, and payoff, then inspect "
    "the price and simulation diagnostics."
)

with st.sidebar.form("simulation_inputs"):
    st.header("Simulation configuration")
    model = st.selectbox("Model", list(ModelType), format_func=lambda item: item.value)
    discretization = st.selectbox(
        "Discretization", list(DiscretizationType), format_func=lambda item: item.value
    )
    payoff = st.selectbox("Payoff", list(PayoffType), format_func=lambda item: item.value)
    option_type = st.selectbox(
        "Option type", list(OptionType), format_func=lambda item: item.value
    )
    sampling_type = st.selectbox(
        "Sampling type", list(SamplingType), format_func=lambda item: item.value
    )

    st.subheader("Market and option inputs")
    start_price = st.number_input("Start price", min_value=0.01, value=100.0, step=1.0)
    strike = st.number_input("Strike", min_value=0.01, value=100.0, step=1.0)
    maturity = st.number_input(
        "Maturity (years)", min_value=0.01, value=1.0, step=0.25
    )
    risk_free_rate = st.number_input(
        "Risk-free rate", value=0.05, step=0.005, format="%.4f"
    )
    volatility = st.number_input(
        "Volatility", min_value=0.0, value=0.20, step=0.01, format="%.4f"
    )

    asian_averaging_months = 12.0
    if payoff is PayoffType.ASIAN:
        max_averaging_months = float(maturity) * 12.0
        asian_averaging_months = st.number_input(
            "Average over final months",
            min_value=0.01,
            max_value=max_averaging_months,
            value=min(12.0, max_averaging_months),
            step=1.0,
            help=(
                "The Asian payoff uses simulated prices from this many months "
                "immediately before maturity."
            ),
        )

    jump_intensity = 0.0
    jump_mean = 0.0
    jump_volatility = 0.0
    if model is ModelType.MERTON_JUMP:
        st.subheader("Merton jump inputs")
        jump_intensity = st.number_input(
            "Jump intensity (expected jumps/year)", min_value=0.0, value=0.75, step=0.05
        )
        jump_mean = st.number_input(
            "Mean log-jump size", value=-0.10, step=0.01, format="%.4f"
        )
        jump_volatility = st.number_input(
            "Log-jump volatility", min_value=0.0, value=0.20, step=0.01, format="%.4f"
        )

    st.subheader("Numerical inputs")
    num_paths = st.number_input(
        "Number of paths", min_value=100, max_value=200_000, value=10_000, step=1_000
    )
    num_steps = st.number_input(
        "Time steps", min_value=1, max_value=2_000, value=252, step=1
    )
    random_seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    submitted = st.form_submit_button("Run simulation", type="primary")

if not submitted:
    st.info("Choose the inputs in the sidebar and click **Run simulation**.")
    st.stop()

try:
    config = SimulationConfig(
        model=model,
        discretization=discretization,
        payoff=payoff,
        option_type=option_type,
        sampling_type=sampling_type,
        start_price=float(start_price),
        strike=float(strike),
        maturity=float(maturity),
        risk_free_rate=float(risk_free_rate),
        volatility=float(volatility),
        num_paths=int(num_paths),
        num_steps=int(num_steps),
        asian_averaging_months=float(asian_averaging_months),
        random_seed=int(random_seed),
        jump_intensity=float(jump_intensity),
        jump_mean=float(jump_mean),
        jump_volatility=float(jump_volatility),
    )
    with st.spinner("Generating paths and calculating the option price..."):
        manager = Manager()
        result = manager.start(config)
        figures = manager.visualise(result)
except (ValueError, MemoryError) as error:
    st.error(f"Simulation could not be completed: {error}")
    st.stop()

ci_half_width = 1.96 * result.standard_error
metric_columns = st.columns(4)
metric_columns[0].metric("Option price", f"{result.option_price:.4f}")
metric_columns[1].metric("Payoff variance", f"{result.payoff_variance:.4f}")
metric_columns[2].metric("Standard error", f"{result.standard_error:.4f}")
metric_columns[3].metric("Simulation time", f"{result.elapsed_seconds:.3f} s")
st.caption(
    f"Approximate 95% Monte Carlo interval: "
    f"[{result.option_price - ci_half_width:.4f}, "
    f"{result.option_price + ci_half_width:.4f}]"
)

tabs = st.tabs(
    ["Paths", "Final prices", "Payoffs", "Price convergence", "Variance"]
)
figure_keys = [
    "paths",
    "terminal_distribution",
    "payoff_distribution",
    "price_convergence",
    "variance_convergence",
]
for tab, key in zip(tabs, figure_keys, strict=True):
    with tab:
        st.pyplot(figures[key], use_container_width=True)
        plt.close(figures[key])

with st.expander("Selected configuration"):
    st.json(
        {
            "model": config.model.value,
            "discretization": config.discretization.value,
            "payoff": config.payoff.value,
            "option_type": config.option_type.value,
            "sampling_type": config.sampling_type.value,
            "asian_averaging_months": (
                config.asian_averaging_months
                if config.payoff is PayoffType.ASIAN
                else "Not applicable"
            ),
            "paths": config.num_paths,
            "steps": config.num_steps,
            "seed": config.random_seed,
        }
    )
