"""Home/Dashboard page for F1 Race Predictor web app."""

import streamlit as st
from datetime import datetime
from typing import Dict, List


def show_home_page() -> None:
    """Display the home/dashboard page with overview and quick actions."""
    st.title("🏎️ F1 Race Predictor")
    st.markdown("Welcome to the F1 Race Predictor - Your AI-powered F1 analysis companion")

    # Quick stats
    st.markdown("### 📊 Quick Stats")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Prediction Accuracy",
            value="87.3%",
            delta="+2.1%",
            help="Historical prediction accuracy over last 50 races",
        )

    with col2:
        st.metric(
            label="Races Analyzed",
            value="124",
            delta="+5",
            help="Total number of races in the dataset",
        )

    with col3:
        st.metric(
            label="Active Models",
            value="4",
            help="Number of trained ML models available",
        )

    with col4:
        st.metric(
            label="Data Quality",
            value="94.2%",
            delta="+1.3%",
            help="Overall data quality score",
        )

    st.markdown("---")

    # Upcoming races
    st.markdown("### 🏁 Upcoming Races")

    upcoming_races = get_upcoming_races()

    if upcoming_races:
        for race in upcoming_races[:3]:  # Show next 3 races
            with st.container():
                col1, col2, col3 = st.columns([3, 2, 1])

                with col1:
                    st.markdown(f"**{race['name']}**")
                    st.caption(f"{race['location']} • {race['circuit']}")

                with col2:
                    st.markdown(f"📅 {race['date']}")
                    st.caption(f"Round {race['round']}")

                with col3:
                    if st.button("Predict", key=f"predict_{race['round']}"):
                        st.session_state.selected_race = race
                        st.switch_page("pages/predict.py")

                st.markdown('<div class="nebula-divider"></div>', unsafe_allow_html=True)
    else:
        st.info("ℹ️ No upcoming races scheduled. Check back later!")

    # Quick actions
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Quick Actions")

        if st.button("🏆 Generate Prediction", type="primary", use_container_width=True):
            st.switch_page("pages/predict.py")

        if st.button("📊 Compare Drivers", use_container_width=True):
            st.switch_page("pages/compare.py")

        if st.button("📈 View Analytics", use_container_width=True):
            st.switch_page("pages/analytics.py")

    with col2:
        st.markdown("### 💬 AI Assistant")

        st.markdown(
            """
            Ask anything about F1 predictions, statistics, or race analysis:

            - "Who will win the next race?"
            - "Compare Hamilton and Verstappen"
            - "Show me Red Bull's performance"
            """
        )

        if st.button("🤖 Start Chat", type="primary", use_container_width=True):
            st.switch_page("pages/chat.py")

    st.markdown("---")

    # Recent predictions
    st.markdown("### 📝 Recent Predictions")

    recent = get_recent_predictions()

    if recent:
        for pred in recent[:5]:  # Show last 5 predictions
            with st.expander(f"{pred['race']} - {pred['date']}", expanded=False):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown("**Predicted Podium:**")
                    st.markdown(f"🥇 {pred['podium'][0]}")
                    st.markdown(f"🥈 {pred['podium'][1]}")
                    st.markdown(f"🥉 {pred['podium'][2]}")

                with col2:
                    st.markdown("**Accuracy:**")
                    st.metric("", f"{pred['accuracy']}%", delta=None)
                    st.caption(f"Model: {pred['model']}")
    else:
        st.info("ℹ️ No predictions yet. Create your first prediction!")

    # System status
    st.markdown("---")
    st.markdown("### 🔧 System Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.success("✅ API Connected")
        st.caption("Ergast F1 API")

    with col2:
        st.success("✅ Models Loaded")
        st.caption("4 ML models ready")

    with col3:
        st.success("✅ Data Up-to-date")
        st.caption(f"Last sync: {datetime.now().strftime('%H:%M')}")


def get_upcoming_races() -> List[Dict]:
    """Get list of upcoming F1 races.

    Returns:
        List of upcoming race dictionaries with details
    """
    # Mock data - replace with actual API call
    return [
        {
            "name": "Monaco Grand Prix",
            "location": "Monte Carlo, Monaco",
            "circuit": "Circuit de Monaco",
            "date": "May 26, 2024",
            "round": 8,
        },
        {
            "name": "Canadian Grand Prix",
            "location": "Montreal, Canada",
            "circuit": "Circuit Gilles Villeneuve",
            "date": "June 9, 2024",
            "round": 9,
        },
        {
            "name": "Spanish Grand Prix",
            "location": "Barcelona, Spain",
            "circuit": "Circuit de Barcelona-Catalunya",
            "date": "June 23, 2024",
            "round": 10,
        },
    ]


def get_recent_predictions() -> List[Dict]:
    """Get list of recent predictions made by the system.

    Returns:
        List of prediction dictionaries with results
    """
    # Mock data - replace with actual database query
    return [
        {
            "race": "Emilia Romagna Grand Prix",
            "date": "May 19, 2024",
            "podium": ["Max Verstappen", "Lando Norris", "Charles Leclerc"],
            "accuracy": 92.5,
            "model": "Ensemble",
        },
        {
            "race": "Miami Grand Prix",
            "date": "May 5, 2024",
            "podium": ["Lando Norris", "Max Verstappen", "Charles Leclerc"],
            "accuracy": 88.3,
            "model": "XGBoost",
        },
        {
            "race": "Chinese Grand Prix",
            "date": "April 21, 2024",
            "podium": ["Max Verstappen", "Lewis Hamilton", "Sergio Perez"],
            "accuracy": 85.7,
            "model": "Ensemble",
        },
    ]
