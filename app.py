"""Streamlit user interface for the configurable Monte Carlo engine."""

import matplotlib.pyplot as plt
import streamlit as st

from monte_carlo import (
    DiscretizationType,
    MODEL_PARAMETER_SPECS,
    Manager,
    ModelType,
    OptionType,
    PAYOFF_PARAMETER_SPECS,
    ParameterSpec,
    PayoffType,
    SamplingType,
    SimulationConfig,
)


def render_component_inputs(
    specs: tuple[ParameterSpec, ...], component: str, maturity: float
) -> dict[str, float]:
    """Render any registered component inputs through one reusable UI flow."""
    values: dict[str, float] = {}
    for spec in specs:
        default = (
            min(spec.default, maturity * 365.0)
            if spec.maturity_days_limit
            else spec.default
        )
        input_options: dict[str, object] = {
            "label": spec.label,
            "value": default,
            "step": spec.step,
            "key": f"{component}_{spec.key}",
        }
        if spec.min_value is not None:
            input_options["min_value"] = spec.min_value
        if spec.number_format is not None:
            input_options["format"] = spec.number_format
        if spec.help is not None:
            input_options["help"] = spec.help
        values[spec.key] = float(st.number_input(**input_options))
    return values


st.set_page_config(page_title="Configurable Monte Carlo Engine", layout="wide")
st.title("Configurable Monte Carlo Option-Pricing Engine")
st.caption(
    "Select a stochastic model, discretization method, and payoff, then inspect "
    "the price and simulation diagnostics."
)

with st.sidebar:
    st.header("Simulation configuration")
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

    st.subheader("Component choices")
    model = st.selectbox("Model", list(ModelType), format_func=lambda item: item.value)
    model_parameters = render_component_inputs(
        MODEL_PARAMETER_SPECS[model], "model", float(maturity)
    )
    discretization = st.selectbox(
        "Discretization", list(DiscretizationType), format_func=lambda item: item.value
    )
    payoff = st.selectbox("Payoff", list(PayoffType), format_func=lambda item: item.value)
    payoff_parameters = render_component_inputs(
        PAYOFF_PARAMETER_SPECS[payoff], "payoff", float(maturity)
    )
    option_type = st.selectbox(
        "Option type", list(OptionType), format_func=lambda item: item.value
    )
    sampling_type = st.selectbox(
        "Sampling type", list(SamplingType), format_func=lambda item: item.value
    )

    st.subheader("Numerical inputs")
    num_paths = st.number_input(
        "Number of paths",
        min_value=100,
        max_value=1_000_000,
        value=10_000,
        step=1_000,
    )
    num_steps = st.number_input(
        "Time steps", min_value=1, max_value=2_000, value=252, step=1
    )
    submitted = st.button("Run simulation", type="primary")

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
        model_parameters=model_parameters,
        payoff_parameters=payoff_parameters,
    )
    with st.spinner("Generating paths and calculating the option price..."):
        manager = Manager()
        result = manager.start(config)
        figures = manager.visualise(result)
except (ValueError, MemoryError) as error:
    st.error(f"Simulation could not be completed: {error}")
    st.stop()

metric_columns = st.columns(4)
metric_columns[0].metric("Option price", f"{result.option_price:.4f}")
metric_columns[1].metric(
    "95% confidence interval",
    f"[{result.confidence_interval_low:.4f}, {result.confidence_interval_high:.4f}]",
)
metric_columns[2].metric("Standard error", f"{result.standard_error:.4f}")
metric_columns[3].metric("Simulation time", f"{result.elapsed_seconds:.3f} s")

tabs = st.tabs(["Paths", "Final prices", "Payoffs", "Price convergence"])
figure_keys = [
    "paths",
    "terminal_distribution",
    "payoff_distribution",
    "price_convergence",
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
            "model_parameters": dict(config.model_parameters),
            "payoff_parameters": dict(config.payoff_parameters),
            "paths": config.num_paths,
            "steps": config.num_steps,
        }
    )
