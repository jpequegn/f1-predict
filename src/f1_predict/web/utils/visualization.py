"""Visualization utilities for comparison charts."""

from typing import Any

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
            x=list(range(len(driver1_data))),
            y=driver1_data.get("position", []),
            name=driver1_name,
            line=dict(color="#1F4E8C", width=3),
            mode="lines+markers"
        ))

    if len(driver2_data) > 0:
        fig.add_trace(go.Scatter(
            x=list(range(len(driver2_data))),
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
            x=list(range(len(cumulative_points))),
            y=cumulative_points,
            name=driver1_name,
            line=dict(color="#1F4E8C", width=3),
            fill="tozeroy"
        ))

    if len(driver2_data) > 0:
        cumulative_points = driver2_data.get("points", []).cumsum()
        fig.add_trace(go.Scatter(
            x=list(range(len(cumulative_points))),
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


def create_stats_comparison(stats: dict[str, Any]) -> go.Figure:
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


def create_position_changes_chart(race_data: pd.DataFrame) -> go.Figure:
    """Create race position changes visualization.

    Args:
        race_data: DataFrame with lap, driver, and position columns

    Returns:
        Plotly figure showing position changes throughout race
    """
    if len(race_data) == 0:
        return _create_empty_chart("Race Position Changes")

    fig = go.Figure()

    for driver in race_data.get("driver", []).unique():
        driver_data = race_data[race_data["driver"] == driver]

        fig.add_trace(go.Scatter(
            x=driver_data.get("lap", []),
            y=driver_data.get("position", []),
            name=driver,
            mode="lines+markers",
            line={"width": 2},
            marker={"size": 6}
        ))

    fig.update_layout(
        title="Race Position Changes",
        xaxis_title="Lap Number",
        yaxis_title="Position",
        yaxis={"autorange": "reversed"},
        template="plotly_dark",
        height=400,
        hovermode="x unified"
    )

    return fig


def create_lap_time_heatmap(lap_times: pd.DataFrame) -> go.Figure:
    """Create lap time performance heatmap.

    Args:
        lap_times: DataFrame with driver, lap, and time columns

    Returns:
        Plotly heatmap showing lap time performance
    """
    if len(lap_times) == 0:
        return _create_empty_chart("Lap Time Heatmap")

    pivot = lap_times.pivot_table(
        index="driver",
        columns="lap",
        values="time",
        aggfunc="first"
    )

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale="RdYlGn_r",
        colorbar={"title": "Lap Time (s)"},
        text=pivot.values.round(2),
        texttemplate="%{text}",
        textfont={"size": 9}
    ))

    fig.update_layout(
        title="Lap Time Performance Heatmap",
        xaxis_title="Lap Number",
        yaxis_title="Driver",
        template="plotly_dark",
        height=500
    )

    return fig


def create_driver_radar_chart(driver: str, metrics: dict[str, Any]) -> go.Figure:
    """Create driver performance radar chart.

    Args:
        driver: Driver name
        metrics: Dictionary with performance metrics

    Returns:
        Plotly radar chart
    """
    categories = list(metrics.keys())
    values = list(metrics.values())

    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        name=driver,
        marker={"color": "#1F4E8C"}
    ))

    fig.update_layout(
        polar={"radialaxis": {"visible": True, "range": [0, 100]}},
        title=f"{driver} Performance Profile",
        template="plotly_dark",
        height=500
    )

    return fig


def _create_empty_chart(title: str) -> go.Figure:
    """Create empty placeholder chart.

    Args:
        title: Chart title

    Returns:
        Plotly figure with empty state message
    """
    fig = go.Figure()

    fig.add_annotation(
        text="No data available",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 20, "color": "gray"}
    )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=400,
        xaxis={"showgrid": False, "visible": False},
        yaxis={"showgrid": False, "visible": False}
    )

    return fig
