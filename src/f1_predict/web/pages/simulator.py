"""Monte Carlo Race Simulation Interface for F1 Race Predictor.

Enables users to configure and run custom race simulations with parameter variations,
sensitivity analysis, and result visualization.
"""

import logging
from typing import Dict, List, Optional, Tuple

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from f1_predict.simulation.core.driver_state import DriverState, TireCompound
from f1_predict.simulation.core.race_state import CircuitContext
from f1_predict.simulation.engine.simulator import MonteCarloSimulator
from f1_predict.simulation.analysis.scenario_builder import (
    ScenarioBuilder,
    DriverStrategy,
)
from f1_predict.simulation.analysis.sensitivity_analyzer import (
    SensitivityAnalyzer,
    ParameterType,
    ParameterSweep,
)
from f1_predict.simulation.analysis.sensitivity_report import SensitivityReport

logger = logging.getLogger(__name__)

# Predefined F1 circuits
CIRCUITS = {
    "Bahrain": {"name": "Bahrain", "laps": 57},
    "Saudi Arabia": {"name": "Saudi Arabia", "laps": 50},
    "Australia": {"name": "Albert Park", "laps": 58},
    "Monaco": {"name": "Monaco", "laps": 78},
    "Spain": {"name": "Circuit de Barcelona", "laps": 66},
    "Austria": {"name": "Red Bull Ring", "laps": 71},
    "France": {"name": "Paul Ricard", "laps": 53},
    "Silverstone": {"name": "Silverstone", "laps": 52},
    "Hungary": {"name": "Hungaroring", "laps": 70},
    "Belgium": {"name": "Spa-Francorchamps", "laps": 44},
    "Italy": {"name": "Monza", "laps": 53},
    "Singapore": {"name": "Marina Bay", "laps": 62},
    "Japan": {"name": "Suzuka", "laps": 53},
    "Qatar": {"name": "Lusail", "laps": 57},
    "USA": {"name": "Circuit of Americas", "laps": 56},
    "Mexico": {"name": "Autodromo Hermanos Rodriguez", "laps": 71},
    "Brazil": {"name": "Interlagos", "laps": 71},
    "Abu Dhabi": {"name": "Yas Marina", "laps": 58},
}

# Default F1 drivers
DEFAULT_DRIVERS = {
    "VER": {"name": "Max Verstappen", "pace": 81.5},
    "LEC": {"name": "Charles Leclerc", "pace": 82.0},
    "SAI": {"name": "Carlos Sainz", "pace": 82.2},
    "HAM": {"name": "Lewis Hamilton", "pace": 82.1},
    "ALO": {"name": "Fernando Alonso", "pace": 82.5},
    "NOR": {"name": "Lando Norris", "pace": 82.3},
    "RUS": {"name": "George Russell", "pace": 82.0},
    "PIA": {"name": "Oscar Piastri", "pace": 82.2},
    "BOT": {"name": "Valtteri Bottas", "pace": 83.0},
    "MAG": {"name": "Kevin Magnussen", "pace": 83.2},
}


def initialize_session_state() -> None:
    """Initialize session state for simulator page."""
    if "race_config" not in st.session_state:
        st.session_state.race_config = {
            "circuit": "Australia",
            "weather": "Dry",
            "temperature": 25,
            "drivers": ["VER", "LEC", "HAM", "RUS"],
        }

    if "simulation_result" not in st.session_state:
        st.session_state.simulation_result = None

    if "sensitivity_result" not in st.session_state:
        st.session_state.sensitivity_result = None

    if "simulation_cache" not in st.session_state:
        st.session_state.simulation_cache = {}


def show_circuit_configuration() -> Tuple[str, str, int, float]:
    """Display circuit configuration section.

    Returns:
        Tuple of (circuit_name, weather, temperature, num_simulations)
    """
    st.subheader("🏁 Race Configuration")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        circuit = st.selectbox(
            "Circuit",
            options=list(CIRCUITS.keys()),
            index=list(CIRCUITS.keys()).index(st.session_state.race_config["circuit"]),
            help="Select F1 circuit for simulation",
        )
        st.session_state.race_config["circuit"] = circuit

    with col2:
        weather = st.selectbox(
            "Weather",
            options=["Dry", "Wet", "Intermediate"],
            index=["Dry", "Wet", "Intermediate"].index(
                st.session_state.race_config["weather"]
            ),
            help="Weather condition affects tire performance",
        )
        st.session_state.race_config["weather"] = weather

    with col3:
        temperature = st.slider(
            "Temperature (°C)",
            min_value=5,
            max_value=40,
            value=st.session_state.race_config["temperature"],
            help="Ambient temperature affects tire degradation",
        )
        st.session_state.race_config["temperature"] = temperature

    with col4:
        num_sims = st.slider(
            "# Simulations",
            min_value=10,
            max_value=10000,
            value=1000,
            step=10,
            help="Number of Monte Carlo simulations to run",
        )

    return circuit, weather, temperature, num_sims


def show_driver_configuration() -> List[DriverState]:
    """Display driver configuration section.

    Returns:
        List of configured DriverState objects
    """
    st.subheader("👥 Driver Setup")

    selected_drivers = st.multiselect(
        "Select Drivers",
        options=list(DEFAULT_DRIVERS.keys()),
        default=st.session_state.race_config.get("drivers", ["VER", "LEC", "HAM", "RUS"]),
        format_func=lambda x: f"{x} - {DEFAULT_DRIVERS[x]['name']}",
        help="Select drivers to include in simulation",
    )
    st.session_state.race_config["drivers"] = selected_drivers

    # Driver adjustment section
    st.markdown("**Driver Pace Adjustments**")

    drivers = []
    cols = st.columns(min(3, len(selected_drivers)))

    for idx, driver_id in enumerate(selected_drivers):
        with cols[idx % len(cols)]:
            driver_info = DEFAULT_DRIVERS[driver_id]
            pace_adjustment = st.slider(
                f"{driver_id} ({driver_info['name']})",
                min_value=-5.0,
                max_value=5.0,
                value=0.0,
                step=0.1,
                help="Adjust baseline pace (s/lap difference)",
            )

            adjusted_pace = driver_info["pace"] + pace_adjustment
            drivers.append(
                DriverState(
                    driver_id=driver_id,
                    driver_name=driver_info["name"],
                    expected_lap_time=adjusted_pace,
                    tire_compound=TireCompound.SOFT,
                    position=len(drivers) + 1,
                )
            )

    return drivers


def show_strategy_configuration() -> Dict[str, DriverStrategy]:
    """Display pit strategy configuration section.

    Returns:
        Dictionary mapping driver_id to strategy
    """
    st.subheader("🏎️ Strategy Configuration")

    strategies = {}

    col1, col2 = st.columns(2)

    with col1:
        pit_stop_count = st.radio(
            "Pit Stop Strategy",
            options=["One Stop", "Two Stops", "Three Stops"],
            horizontal=True,
            help="Expected pit stop strategy for all drivers",
        )

    with col2:
        tire_compound = st.selectbox(
            "Primary Tire Compound",
            options=["Soft", "Medium", "Hard"],
            help="Starting tire compound",
        )

    # Map strategy names to DriverStrategy enum
    strategy_map = {
        "One Stop": DriverStrategy.ONE_STOP,
        "Two Stops": DriverStrategy.TWO_STOP,
        "Three Stops": DriverStrategy.THREE_STOP,
    }

    return strategy_map.get(pit_stop_count, DriverStrategy.TWO_STOP)


def show_simulation_controls(
    drivers: List[DriverState],
    circuit_name: str,
    num_sims: int,
) -> Tuple[bool, bool]:
    """Display simulation control buttons.

    Returns:
        Tuple of (run_simulation, show_sensitivity)
    """
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        run_sim = st.button(
            "▶️ Run Simulation",
            use_container_width=True,
            type="primary",
            help="Execute Monte Carlo simulation",
        )

    with col2:
        show_sens = st.button(
            "📊 Sensitivity Analysis",
            use_container_width=True,
            help="Run sensitivity analysis on selected parameter",
        )

    with col3:
        st.info(
            f"ℹ️ {len(drivers)} drivers | {num_sims} simulations | {circuit_name}"
        )

    return run_sim, show_sens


def run_simulation(
    drivers: List[DriverState],
    circuit_name: str,
    num_sims: int,
    weather: str,
) -> None:
    """Execute Monte Carlo simulation and display results.

    Args:
        drivers: List of driver configurations
        circuit_name: Name of the F1 circuit
        num_sims: Number of simulations to run
        weather: Weather condition
    """
    # Create circuit context
    circuit_info = CIRCUITS[circuit_name]
    circuit = CircuitContext(
        circuit_name=circuit_info["name"],
        total_laps=circuit_info["laps"],
    )

    # Create scenario
    scenario = (
        ScenarioBuilder(f"sim_{circuit_name}", circuit)
        .with_drivers(drivers)
        .build()
    )

    # Run simulation with progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("🔄 Initializing simulation...")

        simulator = MonteCarloSimulator(circuit=circuit, random_state=42)
        status_text.text("🔄 Running Monte Carlo simulations...")

        result = simulator.run_simulations(
            drivers,
            n_simulations=num_sims,
        )

        progress_bar.progress(100)
        status_text.text("✅ Simulation complete!")

        # Store result in session state
        st.session_state.simulation_result = result

        # Display results
        display_simulation_results(result, drivers)

    except Exception as e:
        st.error(f"❌ Simulation failed: {str(e)}")
        logger.exception("Simulation error")
    finally:
        progress_bar.empty()
        status_text.empty()


def display_simulation_results(result, drivers: List[DriverState]) -> None:
    """Display simulation results with visualizations.

    Args:
        result: SimulationResult from simulator
        drivers: List of driver configurations
    """
    st.success("✅ Simulation Results Ready!")

    # Results summary table
    st.subheader("📊 Results Summary")

    results_data = []
    for driver in drivers:
        win_prob = result.get_winner_probability(driver.driver_id)
        podium_prob = result.get_podium_probability(driver.driver_id)
        results_data.append({
            "Driver": f"{driver.driver_id} - {driver.driver_name}",
            "Win Probability": f"{win_prob:.2%}",
            "Podium Probability": f"{podium_prob:.2%}",
        })

    st.dataframe(results_data, use_container_width=True)

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
        # Win probability chart
        win_probs = [
            result.get_winner_probability(driver.driver_id) for driver in drivers
        ]
        fig_win = go.Figure(
            data=[
                go.Bar(
                    x=[d.driver_id for d in drivers],
                    y=win_probs,
                    marker_color="rgba(31, 78, 140, 0.8)",
                )
            ]
        )
        fig_win.update_layout(
            title="Win Probability by Driver",
            xaxis_title="Driver",
            yaxis_title="Probability",
            showlegend=False,
            template="plotly_dark",
        )
        st.plotly_chart(fig_win, use_container_width=True)

    with col2:
        # Podium probability chart
        podium_probs = [
            result.get_podium_probability(driver.driver_id) for driver in drivers
        ]
        fig_podium = go.Figure(
            data=[
                go.Bar(
                    x=[d.driver_id for d in drivers],
                    y=podium_probs,
                    marker_color="rgba(40, 167, 69, 0.8)",
                )
            ]
        )
        fig_podium.update_layout(
            title="Podium Probability by Driver",
            xaxis_title="Driver",
            yaxis_title="Probability",
            showlegend=False,
            template="plotly_dark",
        )
        st.plotly_chart(fig_podium, use_container_width=True)


def show_sensitivity_analysis(
    drivers: List[DriverState],
    circuit_name: str,
    num_sims: int,
) -> None:
    """Display sensitivity analysis interface.

    Args:
        drivers: List of driver configurations
        circuit_name: Name of the F1 circuit
        num_sims: Number of simulations to run
    """
    st.subheader("📈 Sensitivity Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        parameter_type = st.selectbox(
            "Parameter Type",
            options=["Pace", "Grid Position"],
            help="Select which parameter to vary",
        )

    with col2:
        target_driver = st.selectbox(
            "Target Driver",
            options=[d.driver_id for d in drivers],
            help="Select driver to analyze",
        )

    with col3:
        sweep_type = st.selectbox(
            "Sweep Type",
            options=["Linear", "Logarithmic"],
            help="How to space parameter values",
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        min_val = st.number_input(
            "Min Value",
            value=-2.0,
            step=0.5,
            help="Minimum parameter value",
        )

    with col2:
        max_val = st.number_input(
            "Max Value",
            value=2.0,
            step=0.5,
            help="Maximum parameter value",
        )

    with col3:
        num_steps = st.slider(
            "Number of Steps",
            min_value=3,
            max_value=21,
            value=5,
            help="Number of sweep points",
        )

    if st.button("🚀 Run Sensitivity Analysis", use_container_width=True, type="primary"):
        run_sensitivity_analysis(
            drivers,
            circuit_name,
            num_sims,
            parameter_type,
            target_driver,
            sweep_type,
            min_val,
            max_val,
            num_steps,
        )


def run_sensitivity_analysis(
    drivers: List[DriverState],
    circuit_name: str,
    num_sims: int,
    parameter_type: str,
    target_driver: str,
    sweep_type: str,
    min_val: float,
    max_val: float,
    num_steps: int,
) -> None:
    """Execute sensitivity analysis.

    Args:
        drivers: List of driver configurations
        circuit_name: Name of the F1 circuit
        num_sims: Number of simulations
        parameter_type: Type of parameter (Pace/Grid)
        target_driver: Driver ID to analyze
        sweep_type: Linear or Logarithmic
        min_val: Minimum parameter value
        max_val: Maximum parameter value
        num_steps: Number of sweep points
    """
    try:
        # Create circuit and scenario
        circuit_info = CIRCUITS[circuit_name]
        circuit = CircuitContext(
            circuit_name=circuit_info["name"],
            total_laps=circuit_info["laps"],
        )

        scenario = (
            ScenarioBuilder(f"sensitivity_{circuit_name}", circuit)
            .with_drivers(drivers)
            .build()
        )

        # Create analyzer
        simulator = MonteCarloSimulator(circuit=circuit, random_state=42)
        analyzer = SensitivityAnalyzer(simulator, scenario, n_simulations=num_sims)

        # Run appropriate sensitivity analysis
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("🔄 Running sensitivity analysis...")

        if parameter_type == "Pace":
            pace_deltas = list(
                [min_val + (max_val - min_val) * i / (num_steps - 1) for i in range(num_steps)]
            )
            result = analyzer.vary_driver_pace(target_driver, pace_deltas)
        else:  # Grid Position
            position_offsets = [
                int(min_val + (max_val - min_val) * i / (num_steps - 1))
                for i in range(num_steps)
            ]
            result = analyzer.vary_grid_positions(target_driver, position_offsets)

        progress_bar.progress(100)
        status_text.text("✅ Sensitivity analysis complete!")

        # Store result
        st.session_state.sensitivity_result = result

        # Display results
        display_sensitivity_results(result)

        progress_bar.empty()
        status_text.empty()

    except Exception as e:
        st.error(f"❌ Sensitivity analysis failed: {str(e)}")
        logger.exception("Sensitivity analysis error")


def display_sensitivity_results(result) -> None:
    """Display sensitivity analysis results.

    Args:
        result: SensitivityResult from analyzer
    """
    st.success("✅ Sensitivity Analysis Results Ready!")

    # Generate report
    report = SensitivityReport(result)

    # Summary
    st.subheader("📋 Summary")
    st.markdown(report.generate_summary_text())

    # Tornado chart
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌪️ Tornado Chart")
        tornado_data = report.get_tornado_chart_data()

        # Create tornado chart
        drivers = list(tornado_data.drivers.keys())
        negative_impacts = [tornado_data.drivers[d][0] for d in drivers]
        positive_impacts = [tornado_data.drivers[d][1] for d in drivers]

        fig_tornado = go.Figure()
        fig_tornado.add_trace(
            go.Bar(
                y=drivers,
                x=[-x for x in negative_impacts],
                name="Negative Impact",
                orientation="h",
                marker_color="rgba(220, 53, 69, 0.8)",
            )
        )
        fig_tornado.add_trace(
            go.Bar(
                y=drivers,
                x=positive_impacts,
                name="Positive Impact",
                orientation="h",
                marker_color="rgba(40, 167, 69, 0.8)",
            )
        )
        fig_tornado.update_layout(
            title=f"Parameter Sensitivity: {result.parameter_name}",
            xaxis_title="Impact on Win Probability",
            barmode="relative",
            template="plotly_dark",
        )
        st.plotly_chart(fig_tornado, use_container_width=True)

    with col2:
        st.subheader("📊 Sensitivity Metrics")
        table_data = report.get_sensitivity_table_data()
        st.dataframe(table_data, use_container_width=True)

    # Probability curves
    st.subheader("📈 Probability Curves")
    curves = report.get_probability_curves()

    fig_curves = go.Figure()
    for driver_id, curve_data in curves.items():
        x_vals = [point[0] for point in curve_data]
        y_vals = [point[1] for point in curve_data]
        fig_curves.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines+markers",
                name=driver_id,
            )
        )

    fig_curves.update_layout(
        title="Win Probability vs Parameter Value",
        xaxis_title=result.parameter_name,
        yaxis_title="Win Probability",
        template="plotly_dark",
    )
    st.plotly_chart(fig_curves, use_container_width=True)

    # Key findings
    st.subheader("🔍 Key Findings")
    findings = report.get_key_findings()
    for finding in findings:
        st.write(f"• {finding}")


def show_simulator_page() -> None:
    """Main entry point for simulator page."""
    st.title("🏎️ Monte Carlo Race Simulator")

    st.markdown(
        """
        Configure and run custom F1 race simulations with Monte Carlo analysis.
        Analyze driver performance under different conditions and conduct sensitivity analysis.
        """
    )

    # Initialize session state
    initialize_session_state()

    # Configuration sections
    st.markdown("---")
    circuit, weather, temperature, num_sims = show_circuit_configuration()

    st.markdown("---")
    drivers = show_driver_configuration()

    st.markdown("---")
    strategy = show_strategy_configuration()

    # Control buttons
    st.markdown("---")
    run_sim, show_sens = show_simulation_controls(drivers, circuit, num_sims)

    # Execute simulation if requested
    if run_sim:
        run_simulation(drivers, circuit, num_sims, weather)

    # Display cached result if available
    if st.session_state.simulation_result is not None and not run_sim:
        st.info("Showing previous simulation result. Click 'Run Simulation' to run again.")
        display_simulation_results(st.session_state.simulation_result, drivers)

    # Sensitivity analysis
    st.markdown("---")
    show_sensitivity_analysis(drivers, circuit, num_sims)

    # Display cached sensitivity result if available
    if st.session_state.sensitivity_result is not None:
        st.markdown("---")
        display_sensitivity_results(st.session_state.sensitivity_result)
