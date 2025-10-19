"""Race prediction page for F1 Race Predictor web app."""

import logging

import plotly.graph_objects as go
import streamlit as st

from f1_predict.web.utils.prediction import PredictionManager

logger = logging.getLogger(__name__)


@st.cache_resource
def get_prediction_manager():
    """Get or create prediction manager (cached across reruns)."""
    return PredictionManager()


def show_advanced_options() -> dict:
    """Display advanced prediction configuration options.

    Returns:
        Dictionary with advanced options
    """
    with st.expander("⚙️ Advanced Options"):
        col1, col2 = st.columns(2)

        with col1:
            weather_condition = st.selectbox(
                "Weather Condition",
                options=["Forecast", "Dry", "Wet", "Mixed"],
                help="Override weather forecast for what-if scenarios",
            )

            temperature = st.slider(
                "Temperature (°C)",
                min_value=10,
                max_value=40,
                value=25,
                help="Track temperature affects tire performance",
            )

        with col2:
            tire_strategy = st.selectbox(
                "Tire Strategy",
                options=["Optimal", "One-stop", "Two-stop", "Aggressive"],
                help="Expected pit stop strategy",
            )

            safety_car_prob = st.slider(
                "Safety Car Probability",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                format="%.2f",
                help="Likelihood of safety car affecting strategy",
            )

    return {
        "weather": weather_condition,
        "temperature": temperature,
        "tire_strategy": tire_strategy,
        "safety_car_prob": safety_car_prob,
    }


def _display_input_section() -> tuple:
    """Display prediction configuration inputs.

    Returns:
        Tuple of (selected_race, model_type, predict_button)
    """
    st.subheader("Prediction Configuration")
    col1, col2, col3 = st.columns([2, 2, 1])

    manager = get_prediction_manager()

    # Race selection
    with col1:
        races = manager.get_upcoming_races()
        if not races:
            st.error(
                "❌ Could not load upcoming races. Please ensure race data is collected."
            )
            st.info("Run: `python scripts/collect_historical_data.py`")
            return None, None, None

        race_names = [f"{r['name']} ({r['circuit']})" for r in races]
        selected_race_idx = st.selectbox(
            "Select Race",
            range(len(races)),
            format_func=lambda i: race_names[i],
            help="Choose the race for prediction",
        )
        selected_race = races[selected_race_idx]

    # Model selection
    with col2:
        model_type = st.selectbox(
            "Prediction Model",
            options=["ensemble", "xgboost", "lightgbm", "random_forest"],
            format_func=lambda x: x.replace("_", " ").title(),
            help="Choose the machine learning model for prediction",
        )

    # Predict button
    with col3:
        st.write("")  # Spacing
        st.write("")
        predict_button = st.button(
            "🎯 Generate Prediction",
            type="primary",
            use_container_width=True,
        )

    return selected_race, model_type, predict_button


def _generate_and_store_prediction(
    selected_race: dict, model_type: str, advanced_options: dict
) -> bool:
    """Generate prediction and store in session state.

    Args:
        selected_race: Selected race dictionary
        model_type: Type of model to use
        advanced_options: Advanced configuration options

    Returns:
        True if prediction successful, False otherwise
    """
    manager = get_prediction_manager()

    try:
        with st.spinner("🔄 Loading model..."):
            model, model_metadata = manager.load_model(model_type)

        st.success(f"✅ Model loaded: {model_type.title()}")

        with st.spinner("📊 Preparing race features..."):
            features = manager.prepare_race_features(
                selected_race["race_id"], selected_race["season"]
            )

        with st.spinner("🎯 Generating predictions..."):
            prediction = manager.generate_prediction(
                model, features, selected_race["name"]
            )

        st.session_state.last_prediction = prediction
        st.session_state.last_race = selected_race
        st.session_state.model_metadata = model_metadata
        st.session_state.advanced_options = advanced_options
        return True
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        logger.error(f"Prediction error: {e}")
        return False


def _display_model_info() -> None:
    """Display model metadata."""
    model_metadata = st.session_state.model_metadata
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Model Type",
            model_metadata["type"].replace("_", " ").title(),
        )
    with col2:
        st.metric("Model Accuracy", f"{model_metadata['accuracy']:.1%}")
    with col3:
        st.metric("Training Date", model_metadata["training_date"])


def _display_podium(prediction: dict) -> None:
    """Display predicted podium."""
    st.subheader("🏆 Predicted Podium")

    podium_cols = st.columns(3)
    for i, (driver_id, confidence) in enumerate(prediction["podium"]):
        with podium_cols[i]:
            position_emoji = ["🥇", "🥈", "🥉"][i]
            st.metric(
                label=f"{position_emoji} P{i+1}",
                value=driver_id,
                delta=f"{confidence:.1%} confidence",
                delta_color="off",
            )


def _display_full_results(manager: PredictionManager, prediction: dict) -> None:
    """Display full race prediction table."""
    st.subheader("📍 Complete Race Prediction")

    # Format results for display
    results_df = manager.format_prediction_results(prediction)
    if not results_df.empty:
        # Remove the confidence score column (used for sorting only)
        display_df = results_df.drop(columns=["Confidence Score"])

        st.dataframe(
            display_df,
            column_config={
                "Position": st.column_config.NumberColumn(
                    "Position",
                    width="small",
                    format="%d",
                ),
                "Driver": st.column_config.TextColumn(
                    "Driver",
                    width="medium",
                ),
                "Team": st.column_config.TextColumn(
                    "Team",
                    width="medium",
                ),
                "Confidence": st.column_config.TextColumn(
                    "Confidence",
                    width="small",
                ),
            },
            hide_index=True,
            use_container_width=True,
        )


def _display_confidence_chart(prediction: dict) -> None:
    """Display confidence distribution chart."""
    st.subheader("📈 Confidence Distribution")

    fig = go.Figure(
        data=[
            go.Bar(
                y=[p["driver_id"] for p in prediction["predictions"][:10]],
                x=[p["confidence"] for p in prediction["predictions"][:10]],
                orientation="h",
                marker={"color": "rgba(31, 78, 140, 0.8)"},
            )
        ]
    )

    fig.update_layout(
        title="Top 10 Predictions by Confidence",
        xaxis_title="Confidence Score",
        yaxis_title="Driver",
        height=400,
        hovermode="closest",
    )

    st.plotly_chart(fig, use_container_width=True)


def _display_export_options(
    manager: PredictionManager, prediction: dict, selected_race: dict
) -> None:
    """Display export options."""
    st.subheader("💾 Export Prediction")

    col1, col2 = st.columns(2)

    with col1:
        csv_data = manager.export_prediction(prediction, format="csv")
        if csv_data:
            st.download_button(
                label="📥 Download as CSV",
                data=csv_data,
                file_name=f"{selected_race['name'].replace(' ', '_')}_prediction.csv",
                mime="text/csv",
            )

    with col2:
        json_data = manager.export_prediction(prediction, format="json")
        if json_data:
            st.download_button(
                label="📥 Download as JSON",
                data=json_data,
                file_name=f"{selected_race['name'].replace(' ', '_')}_prediction.json",
                mime="application/json",
            )


def _display_explanations() -> None:
    """Display model explanations."""
    with st.expander("🔍 Model Explanation"):
        st.markdown(
            """
            #### How Predictions Work

            **Confidence Score**: Represents the model's certainty about a driver's
            position prediction, based on:
            - Driver form (recent race performance)
            - Team reliability (constructor performance)
            - Circuit performance history
            - Qualifying position and momentum

            **Top Performers**: Drivers at the top of the prediction list have the
            highest confidence scores, indicating the model believes they are most
            likely to finish in that position.

            **Model Ensemble**: This prediction uses multiple models to provide more
            robust results by combining predictions from XGBoost, LightGBM, and
            Random Forest models.
            """
        )

        st.info(
            "📊 Note: These are probabilistic predictions. Actual race results "
            "depend on many factors including pit strategy, weather, and incidents."
        )


def _display_prediction_results(manager: PredictionManager) -> None:
    """Display prediction results from session state.

    Args:
        manager: PredictionManager instance
    """
    if "last_prediction" not in st.session_state:
        return

    prediction = st.session_state.last_prediction
    selected_race = st.session_state.last_race

    st.success("✅ Prediction complete!")

    _display_model_info()
    _display_podium(prediction)
    _display_full_results(manager, prediction)
    _display_confidence_chart(prediction)
    _display_export_options(manager, prediction, selected_race)
    _display_explanations()


def show_prediction_page() -> None:
    """Display the race prediction interface."""
    st.title("🏁 Race Prediction")
    st.markdown(
        "Select a race and ML model to generate predictions with confidence scores."
    )

    manager = get_prediction_manager()

    # ========== INPUT SECTION ==========
    selected_race, model_type, predict_button = _display_input_section()

    if selected_race is None:
        return

    # Advanced options
    advanced_options = show_advanced_options()

    # ========== PREDICTION GENERATION ==========
    if predict_button:
        _generate_and_store_prediction(selected_race, model_type, advanced_options)

    # ========== RESULTS DISPLAY ==========
    _display_prediction_results(manager)
