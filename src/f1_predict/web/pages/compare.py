"""Driver and team comparison page for F1 Race Predictor web app."""

import logging

import pandas as pd
import streamlit as st

from f1_predict.web.utils.comparison import (
    DriverComparison,
    TeamComparison,
)

logger = logging.getLogger(__name__)


@st.cache_resource
def get_driver_comparison():
    """Get cached driver comparison instance."""
    return DriverComparison()


@st.cache_resource
def get_team_comparison():
    """Get cached team comparison instance."""
    return TeamComparison()


def show_comparison_page() -> None:
    """Display the driver/team comparison interface."""
    st.title("📊 Driver & Team Comparison")
    st.markdown("Compare drivers or teams head-to-head with performance analytics.")

    # Comparison type selector
    comparison_type = st.radio(
        "Comparison Type",
        options=["Drivers", "Teams"],
        horizontal=True,
    )

    if comparison_type == "Drivers":
        show_driver_comparison()
    else:
        show_team_comparison()


def show_driver_comparison() -> None:
    """Display driver comparison interface."""
    st.subheader("Driver Comparison")

    # Input section
    col1, col2 = st.columns(2)

    with col1:
        # Sample driver list - in production would load from data
        drivers = [
            "Lewis Hamilton",
            "Max Verstappen",
            "Charles Leclerc",
            "George Russell",
            "Sergio Perez",
            "Carlos Sainz",
            "Lando Norris",
            "Oscar Piastri",
        ]
        driver1 = st.selectbox("Driver 1", drivers, key="driver1_select")

    with col2:
        driver2 = st.selectbox("Driver 2", drivers, key="driver2_select")

    # Filters
    col3, col4 = st.columns(2)

    with col3:
        circuits = ["All Circuits", "Monaco", "Silverstone", "Monza", "Spa"]
        circuit_filter = st.selectbox("Circuit", circuits)

    with col4:
        seasons = [2024, 2023, 2022, 2021, 2020]
        season_filter = st.selectbox("Season", seasons)

    if st.button("Compare Drivers", type="primary", key="compare_drivers_btn"):
        with st.spinner("Analyzing performance..."):
            show_driver_results(driver1, driver2, circuit_filter, season_filter)


def show_driver_results(
    driver1: str, driver2: str, circuit_filter: str, season_filter: int
) -> None:
    """Display driver comparison results.

    Args:
        driver1: First driver name
        driver2: Second driver name
        circuit_filter: Circuit filter
        season_filter: Season filter
    """
    st.success("✅ Comparison ready!")

    # Head-to-Head Stats
    st.subheader("🏆 Head-to-Head Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Races Won",
            f"{driver1}: 0",
            delta=f"{driver2}: 0",
            delta_color="off",
        )

    with col2:
        st.metric(
            "Podiums",
            f"{driver1}: 0",
            delta=f"{driver2}: 0",
            delta_color="off",
        )

    with col3:
        st.metric(
            "Avg Position",
            f"{driver1}: —",
            delta=f"{driver2}: —",
            delta_color="off",
        )

    # Performance Charts
    st.subheader("📈 Performance Comparison")

    tab1, tab2, tab3 = st.tabs(["Race Results", "Points Trend", "Position Distribution"])

    with tab1:
        st.markdown("**Race Results Over Time**")
        # Placeholder chart
        sample_data = pd.DataFrame({
            "Race": [f"Race {i}" for i in range(1, 11)],
            f"{driver1}": list(range(1, 11)),
            f"{driver2}": list(range(2, 12)),
        })
        st.line_chart(sample_data, x="Race")

    with tab2:
        st.markdown("**Cumulative Points Trend**")
        # Placeholder chart
        st.line_chart(
            pd.DataFrame({
                "Race": [f"Race {i}" for i in range(1, 11)],
                f"{driver1}": [25 * i for i in range(1, 11)],
                f"{driver2}": [22 * i for i in range(1, 11)],
            }),
            x="Race"
        )

    with tab3:
        st.markdown("**Finishing Position Distribution**")
        st.info("Position distribution chart would appear here")

    # Circuit-specific performance
    if circuit_filter != "All Circuits":
        st.subheader(f"🏁 Performance at {circuit_filter}")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                f"{driver1} at {circuit_filter}",
                "Best: 1st",
                delta="Avg: 3rd",
                delta_color="off",
            )

        with col2:
            st.metric(
                f"{driver2} at {circuit_filter}",
                "Best: 2nd",
                delta="Avg: 4th",
                delta_color="off",
            )

    # Export options
    st.subheader("💾 Export Comparison")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📥 Download as CSV", key="export_csv"):
            st.info("CSV export feature coming soon")

    with col2:
        if st.button("📥 Download as PDF", key="export_pdf"):
            st.info("PDF export feature coming soon")


def show_team_comparison() -> None:
    """Display team comparison interface."""
    st.subheader("Team Comparison")

    # Input section
    col1, col2 = st.columns(2)

    with col1:
        teams = [
            "Mercedes",
            "Red Bull Racing",
            "Ferrari",
            "McLaren",
            "Alpine",
            "Aston Martin",
        ]
        team1 = st.selectbox("Team 1", teams, key="team1_select")

    with col2:
        team2 = st.selectbox("Team 2", teams, key="team2_select")

    # Filter
    season_filter = st.selectbox(
        "Season",
        [2024, 2023, 2022, 2021, 2020],
        key="team_season_select",
    )

    if st.button("Compare Teams", type="primary", key="compare_teams_btn"):
        with st.spinner("Analyzing team performance..."):
            show_team_results(team1, team2, season_filter)


def show_team_results(team1: str, team2: str, season_filter: int) -> None:
    """Display team comparison results.

    Args:
        team1: First team name
        team2: Second team name
        season_filter: Season filter
    """
    st.success("✅ Comparison ready!")

    # Team Statistics
    st.subheader("🏆 Team Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Points", f"{team1}: 0", delta=f"{team2}: 0", delta_color="off")

    with col2:
        st.metric("Wins", f"{team1}: 0", delta=f"{team2}: 0", delta_color="off")

    with col3:
        st.metric("Podiums", f"{team1}: 0", delta=f"{team2}: 0", delta_color="off")

    # Performance Visualizations
    st.subheader("📊 Performance Analysis")

    tab1, tab2, tab3 = st.tabs(
        ["Points Progression", "Reliability", "Qualifying vs Race"]
    )

    with tab1:
        st.markdown("**Points Progression Over Season**")
        sample_data = pd.DataFrame({
            "Race": [f"Race {i}" for i in range(1, 11)],
            f"{team1}": [25 * i for i in range(1, 11)],
            f"{team2}": [22 * i for i in range(1, 11)],
        })
        st.line_chart(sample_data, x="Race")

    with tab2:
        st.markdown("**Reliability Comparison**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(f"{team1} Finishes", "50", delta="DNF: 0")
        with col2:
            st.metric(f"{team2} Finishes", "48", delta="DNF: 2")

    with tab3:
        st.markdown("**Qualifying vs Race Performance**")
        st.info("Scatter plot comparing qualifying and race performance would appear here")

    # Driver Contribution
    st.subheader("👥 Driver Contribution")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**{team1} Drivers**")
        st.dataframe(
            pd.DataFrame({
                "Driver": ["Driver A", "Driver B"],
                "Points": [250, 200],
            }),
            use_container_width=True,
        )

    with col2:
        st.markdown(f"**{team2} Drivers**")
        st.dataframe(
            pd.DataFrame({
                "Driver": ["Driver C", "Driver D"],
                "Points": [220, 180],
            }),
            use_container_width=True,
        )

    # Export options
    st.subheader("💾 Export Comparison")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📥 Download as CSV", key="export_team_csv"):
            st.info("CSV export feature coming soon")

    with col2:
        if st.button("📥 Download as PDF", key="export_team_pdf"):
            st.info("PDF export feature coming soon")
