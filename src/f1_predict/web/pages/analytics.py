"""Analytics dashboard page for F1 Race Predictor web app."""

import logging

import pandas as pd
import streamlit as st

from f1_predict.web.utils.analytics import (
    AnalyticsCalculator,
    CircuitAnalyzer,
    PerformanceAnalyzer,
    StandingsCalculator,
    TrendAnalyzer,
    filter_by_time_period,
)
from f1_predict.web.utils.analytics_visualization import (
    create_circuit_heatmap,
    create_points_distribution_chart,
    create_points_progression_chart,
    create_qualifying_vs_race_chart,
    create_reliability_chart,
    create_standings_chart,
    create_win_rate_chart,
)

logger = logging.getLogger(__name__)


# Sample data generator for demonstration
def _get_sample_race_results() -> pd.DataFrame:
    """Generate sample race results for demonstration."""
    import numpy as np

    np.random.seed(42)

    teams = [
        "Mercedes",
        "Red Bull Racing",
        "Ferrari",
        "McLaren",
        "Alpine",
        "Aston Martin",
    ]
    drivers = [
        "Lewis Hamilton",
        "Max Verstappen",
        "Charles Leclerc",
        "Lando Norris",
        "Fernando Alonso",
        "Oscar Piastri",
    ]
    circuits = ["Monaco", "Silverstone", "Monza", "Spa", "Hungary"]

    races = []
    for season in [2023, 2024]:
        for round_num in range(1, 11):
            for team_idx, team in enumerate(teams):
                race = {
                    "race_id": f"race_{season}_{round_num}",
                    "season": season,
                    "round": round_num,
                    "circuit": circuits[round_num % len(circuits)],
                    "date": pd.Timestamp(f"{season}-{round_num+3:02d}-15"),
                    "team": team,
                    "driver_id": drivers[team_idx % len(drivers)],
                    "position": np.random.randint(1, 10),
                    "points": max(0, 30 - np.random.randint(0, 25)),
                    "status": "DNF" if np.random.random() < 0.1 else "Finished",
                    "grid_position": np.random.randint(1, 20),
                }
                races.append(race)

    return pd.DataFrame(races)


@st.cache_data(ttl=3600)
def get_sample_data() -> pd.DataFrame:
    """Get cached sample race results."""
    return _get_sample_race_results()


def _display_kpis(filtered_data: pd.DataFrame) -> None:
    """Display Key Performance Indicators."""
    st.subheader("📊 Key Performance Indicators")

    kpis = AnalyticsCalculator.calculate_kpis(filtered_data)
    accuracy_delta = AnalyticsCalculator.calculate_accuracy_delta(
        kpis["prediction_accuracy"]
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Races Analyzed", kpis["total_races"])

    with col2:
        st.metric(
            "Prediction Accuracy",
            f"{kpis['prediction_accuracy']:.1f}%",
            delta=f"{accuracy_delta:+.1f}%",
        )

    with col3:
        st.metric("Avg Confidence", f"{kpis['avg_confidence']:.1f}%")

    with col4:
        st.metric("Data Quality", f"{kpis['data_quality_score']:.1f}%")


def _display_championship_standings(filtered_data: pd.DataFrame) -> None:
    """Display championship standings."""
    st.subheader("🏆 Championship Standings")

    tab1, tab2 = st.tabs(["Drivers Championship", "Constructors Championship"])

    with tab1:
        driver_standings = StandingsCalculator.get_driver_standings(
            filtered_data, season=2024
        )

        if len(driver_standings) > 0:
            col1, col2 = st.columns([3, 1])

            with col1:
                st.dataframe(
                    driver_standings.head(10),
                    column_config={
                        "position": st.column_config.NumberColumn("POS", width="small"),
                        "driver_id": "Driver",
                        "points": st.column_config.ProgressColumn(
                            "Points",
                            min_value=0,
                            max_value=driver_standings["points"].max(),
                        ),
                        "races": "Races",
                        "wins": st.column_config.NumberColumn("Wins", width="small"),
                    },
                    hide_index=True,
                    use_container_width=True,
                )

            with col2:
                fig = create_standings_chart(
                    driver_standings.head(10), driver_standings=True
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No driver standings data available")

    with tab2:
        team_standings = StandingsCalculator.get_team_standings(
            filtered_data, season=2024
        )

        if len(team_standings) > 0:
            col1, col2 = st.columns([3, 1])

            with col1:
                st.dataframe(
                    team_standings.head(10),
                    column_config={
                        "position": st.column_config.NumberColumn("POS", width="small"),
                        "team": "Team",
                        "points": st.column_config.ProgressColumn(
                            "Points",
                            min_value=0,
                            max_value=team_standings["points"].max(),
                        ),
                        "races": st.column_config.NumberColumn("Races", width="small"),
                    },
                    hide_index=True,
                    use_container_width=True,
                )

            with col2:
                fig = create_standings_chart(team_standings, driver_standings=False)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No team standings data available")


def _display_performance_charts(filtered_data: pd.DataFrame) -> None:
    """Display performance visualizations."""
    st.subheader("📈 Performance Visualizations")

    tab1, tab2, tab3 = st.tabs([
        "Win Rate",
        "Reliability",
        "Qualifying vs Race",
    ])

    with tab1:
        win_rate_data = PerformanceAnalyzer.calculate_win_rate(filtered_data)
        if len(win_rate_data) > 0:
            fig = create_win_rate_chart(win_rate_data)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(win_rate_data, use_container_width=True, hide_index=True)
        else:
            st.info("No win rate data available")

    with tab2:
        reliability_data = PerformanceAnalyzer.calculate_reliability(filtered_data)
        if len(reliability_data) > 0:
            fig = create_reliability_chart(reliability_data)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(reliability_data, use_container_width=True, hide_index=True)
        else:
            st.info("No reliability data available")

    with tab3:
        perf_data = PerformanceAnalyzer.get_qualifying_vs_race_performance(
            filtered_data
        )
        if len(perf_data) > 0:
            fig = create_qualifying_vs_race_chart(perf_data)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No qualifying vs race data available")


def _display_circuit_heatmap(filtered_data: pd.DataFrame) -> None:
    """Display circuit performance heatmap."""
    st.subheader("🔥 Circuit Performance Heatmap")

    heatmap_data = CircuitAnalyzer.get_circuit_performance_heatmap(filtered_data)
    if len(heatmap_data) > 0:
        fig = create_circuit_heatmap(heatmap_data)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No circuit performance data available")


def _display_trends(filtered_data: pd.DataFrame) -> None:
    """Display historical trends."""
    st.subheader("📉 Historical Trends")

    tab1, tab2 = st.tabs(["Points Progression", "Points Distribution"])

    with tab1:
        standings = StandingsCalculator.get_team_standings(filtered_data)
        top_teams = standings.head(5)["team"].tolist() if len(standings) > 0 else []

        if top_teams:
            trend_data = TrendAnalyzer.get_cumulative_points_trend(
                filtered_data, teams=top_teams
            )
            if len(trend_data) > 0:
                fig = create_points_progression_chart(trend_data, teams=top_teams)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No points progression data available")
        else:
            st.info("No team data available for trends")

    with tab2:
        distribution_data = PerformanceAnalyzer.calculate_points_distribution(
            filtered_data
        )
        if len(distribution_data) > 0:
            fig = create_points_distribution_chart(distribution_data)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(distribution_data, use_container_width=True, hide_index=True)
        else:
            st.info("No distribution data available")


def _display_export_options(filtered_data: pd.DataFrame) -> None:
    """Display export options."""
    st.subheader("💾 Export Data")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📥 Export Standings CSV", key="export_standings"):
            standings = StandingsCalculator.get_driver_standings(filtered_data)
            csv = standings.to_csv(index=False)
            st.download_button(
                label="Download Standings",
                data=csv,
                file_name="f1_standings.csv",
                mime="text/csv",
            )

    with col2:
        if st.button("📥 Export Performance CSV", key="export_performance"):
            perf_data = PerformanceAnalyzer.calculate_win_rate(filtered_data)
            csv = perf_data.to_csv(index=False)
            st.download_button(
                label="Download Performance",
                data=csv,
                file_name="f1_performance.csv",
                mime="text/csv",
            )

    with col3:
        if st.button("📊 Export All Data", key="export_all"):
            csv = filtered_data.to_csv(index=False)
            st.download_button(
                label="Download All Data",
                data=csv,
                file_name="f1_analytics_export.csv",
                mime="text/csv",
            )


def show_analytics_page() -> None:
    """Display the analytics and visualization dashboard."""
    st.title("📈 F1 Analytics Dashboard")
    st.markdown(
        "Comprehensive F1 analytics with championship standings, performance metrics, and interactive visualizations."
    )

    # Load sample data
    race_results = get_sample_data()

    # Time period selector
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        time_period = st.selectbox(
            "Time Period",
            ["Last 5 Races", "Current Season", "Last 2 Seasons", "All Time"],
            value="Current Season",
            key="analytics_time_period",
        )

    with col2:
        st.checkbox("Auto Refresh", value=False, key="auto_refresh")

    with col3:
        if st.button("🔄 Refresh", key="manual_refresh"):
            st.cache_data.clear()
            st.rerun()

    # Filter data by time period
    filtered_data = filter_by_time_period(race_results, time_period)

    # Display info
    st.caption(
        f"Showing {filtered_data['race_id'].nunique()} races | "
        f"Total entries: {len(filtered_data)}"
    )

    # Display all sections
    _display_kpis(filtered_data)
    _display_championship_standings(filtered_data)
    _display_performance_charts(filtered_data)
    _display_circuit_heatmap(filtered_data)
    _display_trends(filtered_data)
    _display_export_options(filtered_data)
