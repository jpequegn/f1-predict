"""Visualization utilities for analytics dashboard.

Generates interactive Plotly charts for:
- Championship standings
- Performance metrics
- Reliability analysis
- Circuit performance
- Historical trends
"""

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_win_rate_chart(win_rate_data: pd.DataFrame) -> go.Figure:
    """Create win rate by team bar chart.

    Args:
        win_rate_data: DataFrame with team and win_rate columns

    Returns:
        Plotly figure
    """
    if len(win_rate_data) == 0:
        return _create_empty_chart("Win Rate by Team")

    fig = go.Figure(data=[
        go.Bar(
            x=win_rate_data["team"],
            y=win_rate_data["win_rate"],
            marker={
                "color": win_rate_data["win_rate"],
                "colorscale": "Blues",
                "showscale": False
            },
            text=[f"{x:.1f}%" for x in win_rate_data["win_rate"]],
            textposition="auto",
        )
    ])

    fig.update_layout(
        title="Win Rate by Team",
        xaxis_title="Team",
        yaxis_title="Win Rate (%)",
        template="plotly_dark",
        height=400,
        xaxis_tickangle=-45,
        showlegend=False,
    )

    return fig


def create_reliability_chart(reliability_data: pd.DataFrame) -> go.Figure:
    """Create reliability (finishes vs DNFs) stacked bar chart.

    Args:
        reliability_data: DataFrame with team, finishes, and dnfs columns

    Returns:
        Plotly figure
    """
    if len(reliability_data) == 0:
        return _create_empty_chart("Reliability Analysis")

    fig = go.Figure(data=[
        go.Bar(
            name="Finishes",
            x=reliability_data["team"],
            y=reliability_data["finishes"],
            marker={"color": "#28A745"},
        ),
        go.Bar(
            name="DNFs",
            x=reliability_data["team"],
            y=reliability_data["dnfs"],
            marker={"color": "#DC3545"},
        ),
    ])

    fig.update_layout(
        title="Reliability Analysis: Finishes vs DNFs",
        xaxis_title="Team",
        yaxis_title="Count",
        barmode="stack",
        template="plotly_dark",
        height=400,
        xaxis_tickangle=-45,
    )

    return fig


def create_qualifying_vs_race_chart(perf_data: pd.DataFrame) -> go.Figure:
    """Create qualifying position vs race position scatter plot.

    Args:
        perf_data: DataFrame with grid_position, race_position, and team columns

    Returns:
        Plotly figure
    """
    if len(perf_data) == 0:
        return _create_empty_chart("Qualifying vs Race Performance")

    fig = go.Figure(data=[
        go.Scatter(
            x=perf_data.get("grid_position", []),
            y=perf_data.get("race_position", []),
            mode="markers",
            marker={
                "size": 8,
                "color": perf_data.get("team", "").astype("category").cat.codes,
                "colorscale": "Viridis",
                "showscale": True,
            },
            text=perf_data.get("team", ""),
            hovertemplate="<b>%{text}</b><br>Grid: %{x}<br>Finish: %{y}<extra></extra>",
        )
    ])

    # Add trendline reference (perfect performance)
    fig.add_trace(go.Scatter(
        x=[0, 22],
        y=[0, 22],
        mode="lines",
        name="Perfect Performance",
        line={"dash": "dash", "color": "gray", "width": 2},
        hoverinfo="skip",
    ))

    fig.update_layout(
        title="Qualifying vs Race Performance",
        xaxis_title="Grid Position (Qualifying)",
        yaxis_title="Race Finish Position",
        template="plotly_dark",
        height=400,
        hovermode="closest",
        yaxis={"autorange": "reversed"},
        xaxis={"autorange": "reversed"},
    )

    return fig


def create_circuit_heatmap(heatmap_data: pd.DataFrame) -> go.Figure:
    """Create circuit performance heatmap.

    Args:
        heatmap_data: DataFrame with teams as index, circuits as columns, positions as values

    Returns:
        Plotly figure
    """
    if len(heatmap_data) == 0 or len(heatmap_data.columns) == 0:
        return _create_empty_chart("Circuit Performance Heatmap")

    # Reverse colors so better position (lower number) is brighter
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale="RdYlGn_r",
        text=heatmap_data.values,
        texttemplate="<b>%{text:.1f}</b>",
        textfont={"size": 10},
        colorbar={"title": "Avg Position"},
    ))

    fig.update_layout(
        title="Circuit Performance Heatmap (Lower is Better)",
        xaxis_title="Circuit",
        yaxis_title="Team",
        template="plotly_dark",
        height=500,
    )

    return fig


def create_points_progression_chart(
    trend_data: pd.DataFrame,
    teams: Optional[list[str]] = None
) -> go.Figure:
    """Create cumulative points progression line chart.

    Args:
        trend_data: DataFrame with race_number, team, and cumulative points
        teams: Optional list of teams to highlight

    Returns:
        Plotly figure
    """
    if len(trend_data) == 0:
        return _create_empty_chart("Points Progression Over Season")

    fig = go.Figure()

    team_groups = trend_data.groupby(
        trend_data.get("team", "")
    ) if "team" in trend_data.columns else [(None, trend_data)]

    colors = px.colors.qualitative.Set2
    for idx, (team, data) in enumerate(team_groups):
        if teams and team not in teams:
            continue

        color = colors[idx % len(colors)]

        fig.add_trace(go.Scatter(
            x=data.get("race_number", range(len(data))),
            y=data[data.columns[2]] if len(data.columns) > 2 else data.get("points", []).cumsum(),
            name=team if team else "Points",
            mode="lines+markers",
            line={"width": 3, "color": color},
            marker={"size": 6},
        ))

    fig.update_layout(
        title="Cumulative Points Progression Over Season",
        xaxis_title="Race Number",
        yaxis_title="Total Points",
        template="plotly_dark",
        height=400,
        hovermode="x unified",
    )

    return fig


def create_points_distribution_chart(distribution_data: pd.DataFrame) -> go.Figure:
    """Create points distribution by team (box plot).

    Args:
        distribution_data: DataFrame with team and distribution statistics

    Returns:
        Plotly figure
    """
    if len(distribution_data) == 0:
        return _create_empty_chart("Points Distribution")

    fig = go.Figure()

    for _, row in distribution_data.iterrows():
        # Create box plot trace for each team
        fig.add_trace(go.Box(
            name=row["team"],
            y=[
                row.get("min_points", 0),
                row.get("avg_points", 0),
                row.get("max_points", 0),
            ],
            boxmean="sd",
            marker={"opacity": 0.7},
        ))

    fig.update_layout(
        title="Points Distribution by Team",
        yaxis_title="Points per Race",
        template="plotly_dark",
        height=400,
    )

    return fig


def create_standings_chart(standings_data: pd.DataFrame, driver_standings: bool = True) -> go.Figure:
    """Create championship standings bar chart.

    Args:
        standings_data: DataFrame with standings (position, name, points)
        driver_standings: True for drivers, False for teams

    Returns:
        Plotly figure
    """
    if len(standings_data) == 0:
        return _create_empty_chart("Championship Standings")

    # Get top 10 for visibility
    top_data = standings_data.head(10)

    # Determine colors: gold/silver/bronze for top 3, blue for others
    colors = []
    for i in range(len(top_data)):
        if i == 0:
            colors.append("#FFD700")  # Gold
        elif i == 1:
            colors.append("#C0C0C0")  # Silver
        elif i == 2:
            colors.append("#CD7F32")  # Bronze
        else:
            colors.append("#1F4E8C")  # Blue

    col_name = top_data.columns[1] if len(top_data.columns) > 1 else "driver_id"

    fig = go.Figure(data=[
        go.Bar(
            x=top_data[col_name],
            y=top_data["points"],
            marker={"color": colors},
            text=[f"<b>#{p}</b>" for p in top_data["position"]],
            textposition="outside",
        )
    ])

    title = "Driver Championship Standings" if driver_standings else "Constructor Championship Standings"

    fig.update_layout(
        title=title,
        xaxis_title="Driver" if driver_standings else "Team",
        yaxis_title="Points",
        template="plotly_dark",
        height=400,
        xaxis_tickangle=-45,
        showlegend=False,
    )

    return fig


def _create_empty_chart(title: str) -> go.Figure:
    """Create a placeholder empty chart.

    Args:
        title: Chart title

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    fig.add_annotation(
        text="No data available",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 20, "color": "gray"},
    )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=400,
        xaxis={"showgrid": False, "zeroline": False, "visible": False},
        yaxis={"showgrid": False, "zeroline": False, "visible": False},
    )

    return fig
