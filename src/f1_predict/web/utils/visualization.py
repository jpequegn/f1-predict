"""Visualization utilities for comparison charts."""

import pandas as pd
import plotly.graph_objects as go


def create_race_results_comparison(driver1_data: pd.DataFrame, driver2_data: pd.DataFrame, driver1_name: str, driver2_name: str) -> go.Figure:
    """Create race results comparison chart.

    Args:
        driver1_data: Driver 1 race data
        driver2_data: Driver 2 race data
        driver1_name: Driver 1 name
        driver2_name: Driver 2 name

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    if len(driver1_data) > 0:
        fig.add_trace(go.Scatter(
            x=range(len(driver1_data)),
            y=driver1_data.get("position", []),
            name=driver1_name,
            line=dict(color="#1F4E8C", width=3),
            mode="lines+markers"
        ))

    if len(driver2_data) > 0:
        fig.add_trace(go.Scatter(
            x=range(len(driver2_data)),
            y=driver2_data.get("position", []),
            name=driver2_name,
            line=dict(color="#2762B3", width=3),
            mode="lines+markers"
        ))

    fig.update_layout(
        title="Race Results Comparison",
        xaxis_title="Race Number",
        yaxis_title="Finishing Position",
        yaxis=dict(autorange="reversed"),
        hovermode="x unified",
        template="plotly_dark",
        height=400
    )

    return fig


def create_points_trend(driver1_data: pd.DataFrame, driver2_data: pd.DataFrame, driver1_name: str, driver2_name: str) -> go.Figure:
    """Create cumulative points trend chart.

    Args:
        driver1_data: Driver 1 race data
        driver2_data: Driver 2 race data
        driver1_name: Driver 1 name
        driver2_name: Driver 2 name

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    if len(driver1_data) > 0:
        cumulative_points = driver1_data.get("points", []).cumsum()
        fig.add_trace(go.Scatter(
            x=range(len(cumulative_points)),
            y=cumulative_points,
            name=driver1_name,
            line=dict(color="#1F4E8C", width=3),
            fill="tozeroy"
        ))

    if len(driver2_data) > 0:
        cumulative_points = driver2_data.get("points", []).cumsum()
        fig.add_trace(go.Scatter(
            x=range(len(cumulative_points)),
            y=cumulative_points,
            name=driver2_name,
            line=dict(color="#2762B3", width=3),
            fill="tozeroy"
        ))

    fig.update_layout(
        title="Cumulative Points Trend",
        xaxis_title="Race Number",
        yaxis_title="Total Points",
        hovermode="x unified",
        template="plotly_dark",
        height=400
    )

    return fig


def create_position_distribution(driver1_data: pd.DataFrame, driver2_data: pd.DataFrame, driver1_name: str, driver2_name: str) -> go.Figure:
    """Create position distribution comparison.

    Args:
        driver1_data: Driver 1 race data
        driver2_data: Driver 2 race data
        driver1_name: Driver 1 name
        driver2_name: Driver 2 name

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    if len(driver1_data) > 0:
        fig.add_trace(go.Histogram(
            x=driver1_data.get("position", []),
            name=driver1_name,
            opacity=0.7,
            marker=dict(color="#1F4E8C")
        ))

    if len(driver2_data) > 0:
        fig.add_trace(go.Histogram(
            x=driver2_data.get("position", []),
            name=driver2_name,
            opacity=0.7,
            marker=dict(color="#2762B3")
        ))

    fig.update_layout(
        title="Finishing Position Distribution",
        xaxis_title="Position",
        yaxis_title="Frequency",
        barmode="overlay",
        template="plotly_dark",
        height=400
    )

    return fig


def create_stats_comparison(stats: dict) -> go.Figure:
    """Create head-to-head stats comparison bar chart.

    Args:
        stats: Comparison statistics dictionary

    Returns:
        Plotly figure
    """
    driver1_id = stats.get("driver1_id", "Driver 1")
    driver2_id = stats.get("driver2_id", "Driver 2")

    metrics = ["Wins", "Podiums", "Races"]
    d1_values = [
        stats.get("wins", {}).get(driver1_id, 0),
        stats.get("podiums", {}).get(driver1_id, 0),
        stats.get("races_competed", {}).get(driver1_id, 0)
    ]
    d2_values = [
        stats.get("wins", {}).get(driver2_id, 0),
        stats.get("podiums", {}).get(driver2_id, 0),
        stats.get("races_competed", {}).get(driver2_id, 0)
    ]

    fig = go.Figure(data=[
        go.Bar(name=driver1_id, x=metrics, y=d1_values, marker_color="#1F4E8C"),
        go.Bar(name=driver2_id, x=metrics, y=d2_values, marker_color="#2762B3")
    ])

    fig.update_layout(
        title="Head-to-Head Statistics",
        barmode="group",
        template="plotly_dark",
        height=400
    )

    return fig
