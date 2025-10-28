"""Explainability dashboard page for F1 race predictions.

This page provides comprehensive model explanations using SHAP values,
feature importance visualizations, and interactive what-if analysis tools.
"""

from pathlib import Path

import pandas as pd
import streamlit as st
import structlog

from f1_predict.analysis.shap_explainer import SHAPExplainer
from f1_predict.analysis.shap_visualizations import SHAPVisualizer
from f1_predict.models.baseline import RuleBasedPredictor
from f1_predict.models.logistic import LogisticRacePredictor
from f1_predict.models.random_forest import RandomForestRacePredictor

logger = structlog.get_logger(__name__)


def show_explainability_page() -> None:
    """Display explainability dashboard page."""
    st.title("🔍 Model Explainability Dashboard")
    st.markdown(
        """
        Understand how F1 race predictions are made with SHAP (SHapley Additive exPlanations).
        Explore feature importance, analyze individual predictions, and run what-if scenarios.
        """
    )

    # Initialize session state
    if "shap_explainer" not in st.session_state:
        st.session_state.shap_explainer = None
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False
    if "sample_data" not in st.session_state:
        st.session_state.sample_data = None

    # Sidebar for model selection and configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Model selection
        model_type = st.selectbox(
            "Select Model",
            options=[
                "Rule-Based",
                "Logistic Regression",
                "Random Forest",
                "XGBoost",
                "LightGBM",
            ],
            help="Choose the model to explain",
        )

        # Model loading (simplified for demo)
        if st.button("Load Model", type="primary"):
            with st.spinner("Loading model and generating explanations..."):
                success = _load_model_and_explainer(model_type)
                if success:
                    st.success("Model loaded successfully!")
                    st.session_state.model_loaded = True
                else:
                    st.error("Failed to load model. Using demo mode.")
                    _initialize_demo_mode()

        # Cache management
        st.markdown("---")
        st.subheader("Cache Management")
        if st.button("Clear Cache"):
            if st.session_state.shap_explainer:
                st.session_state.shap_explainer.clear_cache()
                st.success("Cache cleared successfully!")

        # Display settings
        st.markdown("---")
        st.subheader("Display Settings")
        max_features = st.slider(
            "Max Features to Display",
            min_value=5,
            max_value=20,
            value=10,
            help="Number of features to show in visualizations",
        )

        dark_mode = st.checkbox("Dark Mode", value=True)

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊 Global Importance",
            "🎯 Single Prediction",
            "🔄 What-If Analysis",
            "📈 Model Comparison",
        ]
    )

    # Initialize visualizer
    visualizer = SHAPVisualizer(dark_mode=dark_mode)

    # Tab 1: Global Feature Importance
    with tab1:
        _show_global_importance_tab(visualizer, max_features)

    # Tab 2: Single Prediction Explanation
    with tab2:
        _show_single_prediction_tab(visualizer, max_features)

    # Tab 3: What-If Analysis
    with tab3:
        _show_whatif_analysis_tab(visualizer, max_features)

    # Tab 4: Model Comparison
    with tab4:
        _show_model_comparison_tab(visualizer, max_features)


def _load_model_and_explainer(model_type: str) -> bool:
    """Load model and initialize SHAP explainer.

    Args:
        model_type: Type of model to load

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("loading_model", model_type=model_type)

        # Map model type to implementation
        model_map = {
            "Rule-Based": ("rule_based", RuleBasedPredictor),
            "Logistic Regression": ("logistic", LogisticRacePredictor),
            "Random Forest": ("random_forest", RandomForestRacePredictor),
        }

        if model_type not in model_map:
            logger.warning("model_not_implemented", model_type=model_type)
            return False

        model_key, model_class = model_map[model_type]

        # For demo purposes, create a simple model
        # In production, load trained model from disk
        if model_key == "rule_based":
            model = RuleBasedPredictor()
        elif model_key == "logistic":
            model = LogisticRacePredictor(target="podium")
        else:  # random_forest
            model = RandomForestRacePredictor(target="podium", n_estimators=100)

        # Create sample feature names
        feature_names = [
            "qualifying_position",
            "driver_form_score",
            "team_reliability_score",
            "circuit_performance_score",
            "championship_position",
            "avg_finish_position",
            "podium_rate",
            "dnf_rate",
        ]

        # Initialize SHAP explainer
        cache_dir = Path.cwd() / "cache" / "shap"
        explainer = SHAPExplainer(
            model=model,
            model_type=model_key,
            feature_names=feature_names,
            cache_dir=cache_dir,
        )

        st.session_state.shap_explainer = explainer
        st.session_state.model_type = model_key
        st.session_state.feature_names = feature_names

        # Generate sample data for demonstrations
        _generate_sample_data()

        return True

    except Exception as e:
        logger.error("model_loading_failed", error=str(e))
        return False


def _initialize_demo_mode() -> None:
    """Initialize demo mode with mock explanations."""
    # Create mock feature names
    feature_names = [
        "qualifying_position",
        "driver_form_score",
        "team_reliability_score",
        "circuit_performance_score",
        "championship_position",
        "avg_finish_position",
        "podium_rate",
        "dnf_rate",
    ]

    st.session_state.model_type = "rule_based"
    st.session_state.feature_names = feature_names
    st.session_state.model_loaded = True

    # Generate sample data
    _generate_sample_data()


def _generate_sample_data() -> None:
    """Generate sample data for demonstrations."""
    # Create sample feature data
    sample_data = pd.DataFrame(
        {
            "qualifying_position": [1, 3, 5, 7, 2, 4, 6, 8],
            "driver_form_score": [85, 78, 72, 68, 82, 75, 70, 65],
            "team_reliability_score": [92, 88, 85, 82, 90, 87, 84, 80],
            "circuit_performance_score": [88, 82, 76, 72, 85, 80, 75, 70],
            "championship_position": [1, 2, 3, 4, 5, 6, 7, 8],
            "avg_finish_position": [2.5, 4.2, 5.8, 7.1, 3.2, 5.5, 6.9, 8.2],
            "podium_rate": [0.75, 0.60, 0.45, 0.30, 0.70, 0.50, 0.40, 0.25],
            "dnf_rate": [0.10, 0.12, 0.15, 0.18, 0.08, 0.14, 0.16, 0.20],
        }
    )

    st.session_state.sample_data = sample_data


def _show_global_importance_tab(visualizer: SHAPVisualizer, max_features: int) -> None:
    """Show global feature importance tab.

    Args:
        visualizer: SHAP visualizer instance
        max_features: Maximum features to display
    """
    st.header("Global Feature Importance")
    st.markdown(
        """
        See which features have the most impact on predictions across all races.
        Features are ranked by mean absolute SHAP value.
        """
    )

    if not st.session_state.model_loaded:
        st.warning("⚠️ Please load a model first using the sidebar.")
        return

    if st.session_state.shap_explainer is None:
        st.warning("⚠️ Explainer not initialized. Please load a model.")
        return

    with st.spinner("Calculating global feature importance..."):
        try:
            # Generate dataset explanation
            sample_data = st.session_state.sample_data
            if sample_data is None:
                st.error("No sample data available.")
                return

            dataset_explanation = st.session_state.shap_explainer.explain_dataset(
                sample_data,
                sample_size=100,
            )

            # Create summary plot
            fig = visualizer.summary_plot(dataset_explanation, max_display=max_features)
            st.plotly_chart(fig, use_container_width=True)

            # Show feature importance table
            st.subheader("Feature Importance Scores")
            importance_df = pd.DataFrame(
                {
                    "Feature": dataset_explanation["feature_names"],
                    "Mean |SHAP|": dataset_explanation["mean_abs_shap"],
                    "Normalized Importance": dataset_explanation["feature_importance"],
                }
            )
            importance_df = importance_df.sort_values("Mean |SHAP|", ascending=False)
            st.dataframe(importance_df.head(max_features), use_container_width=True)

        except Exception as e:
            logger.error("global_importance_failed", error=str(e))
            st.error(f"Error generating global importance: {str(e)}")


def _show_single_prediction_tab(visualizer: SHAPVisualizer, max_features: int) -> None:
    """Show single prediction explanation tab.

    Args:
        visualizer: SHAP visualizer instance
        max_features: Maximum features to display
    """
    st.header("Single Prediction Explanation")
    st.markdown(
        """
        Understand how individual features contribute to a specific prediction.
        Choose a driver or enter custom feature values.
        """
    )

    if not st.session_state.model_loaded:
        st.warning("⚠️ Please load a model first using the sidebar.")
        return

    # Driver selection
    col1, col2 = st.columns([2, 1])

    with col1:
        driver_options = [
            "Max Verstappen",
            "Lewis Hamilton",
            "Charles Leclerc",
            "Sergio Perez",
            "Custom Values",
        ]
        selected_driver = st.selectbox("Select Driver", driver_options)

    with col2:
        use_sample = st.checkbox("Use Sample Data", value=True)

    # Feature input
    if selected_driver == "Custom Values" and not use_sample:
        st.subheader("Enter Feature Values")
        features = {}
        col_left, col_right = st.columns(2)

        with col_left:
            features["qualifying_position"] = st.slider(
                "Qualifying Position", 1, 20, 3
            )
            features["driver_form_score"] = st.slider(
                "Driver Form Score", 0.0, 100.0, 80.0
            )
            features["team_reliability_score"] = st.slider(
                "Team Reliability", 0.0, 100.0, 90.0
            )
            features["circuit_performance_score"] = st.slider(
                "Circuit Performance", 0.0, 100.0, 85.0
            )

        with col_right:
            features["championship_position"] = st.slider(
                "Championship Position", 1, 20, 2
            )
            features["avg_finish_position"] = st.slider(
                "Avg Finish Position", 1.0, 20.0, 3.5
            )
            features["podium_rate"] = st.slider("Podium Rate", 0.0, 1.0, 0.70)
            features["dnf_rate"] = st.slider("DNF Rate", 0.0, 1.0, 0.10)

        prediction_features = pd.DataFrame([features])
    else:
        # Use sample data
        if st.session_state.sample_data is None:
            st.error("No sample data available.")
            return

        # Get first row as sample
        prediction_features = st.session_state.sample_data.head(1)

    # Generate explanation
    if st.button("Explain Prediction", type="primary"):
        with st.spinner("Generating explanation..."):
            try:
                explainer = st.session_state.shap_explainer
                explanation = explainer.explain_prediction(
                    prediction_features,
                    cache_key=f"{selected_driver}_prediction",
                )

                # Show prediction output
                st.success(
                    f"Model Output: **{explanation['model_output']:.3f}**"
                )

                # Visualization type selector
                viz_type = st.radio(
                    "Visualization Type",
                    ["Waterfall Plot", "Force Plot"],
                    horizontal=True,
                )

                # Show selected visualization
                if viz_type == "Waterfall Plot":
                    fig = visualizer.waterfall_plot(
                        explanation,
                        max_display=max_features,
                        title=f"Prediction Explanation - {selected_driver}",
                    )
                else:
                    fig = visualizer.force_plot(explanation, max_display=max_features)

                st.plotly_chart(fig, use_container_width=True)

                # Show top contributing features
                st.subheader("Top Contributing Features")
                top_features = explanation["top_features"]
                features_df = pd.DataFrame(top_features)
                st.dataframe(features_df, use_container_width=True)

            except Exception as e:
                logger.error("prediction_explanation_failed", error=str(e))
                st.error(f"Error generating explanation: {str(e)}")


def _show_whatif_analysis_tab(visualizer: SHAPVisualizer, max_features: int) -> None:
    """Show what-if analysis tab.

    Args:
        visualizer: SHAP visualizer instance
        max_features: Maximum features to display
    """
    st.header("What-If Analysis")
    st.markdown(
        """
        Explore counterfactual scenarios by changing feature values.
        See how predictions change when you modify specific factors.
        """
    )

    if not st.session_state.model_loaded:
        st.warning("⚠️ Please load a model first using the sidebar.")
        return

    if st.session_state.sample_data is None:
        st.error("No sample data available.")
        return

    # Base scenario
    st.subheader("Base Scenario")
    base_features = st.session_state.sample_data.head(1).copy()

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Current Feature Values:**")
        st.dataframe(base_features.T, use_container_width=True)

    # Feature modifications
    st.subheader("Modify Features")
    st.markdown("Select features to change and their new values:")

    feature_changes = {}
    available_features = st.session_state.feature_names

    # Allow up to 3 feature modifications
    for i in range(3):
        col_feat, col_val = st.columns([2, 1])

        with col_feat:
            feature = st.selectbox(
                f"Feature {i+1}",
                options=["None"] + available_features,
                key=f"whatif_feature_{i}",
            )

        if feature != "None":
            with col_val:
                current_val = float(base_features[feature].values[0])
                new_val = st.number_input(
                    "New Value",
                    value=current_val,
                    key=f"whatif_value_{i}",
                )
                feature_changes[feature] = new_val

    # Run what-if analysis
    if st.button("Run What-If Analysis", type="primary"):
        if not feature_changes:
            st.warning("Please select at least one feature to modify.")
            return

        with st.spinner("Running what-if analysis..."):
            try:
                explainer = st.session_state.shap_explainer
                whatif_result = explainer.what_if_analysis(
                    base_features,
                    feature_changes,
                )

                # Show prediction change
                pred_delta = whatif_result["prediction_delta"]

                st.markdown("### Prediction Impact")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Base Prediction",
                        f"{whatif_result['base_prediction']['model_output']:.3f}",
                    )

                with col2:
                    st.metric(
                        "Modified Prediction",
                        f"{whatif_result['modified_prediction']['model_output']:.3f}",
                    )

                with col3:
                    st.metric(
                        "Change",
                        f"{pred_delta:+.3f}",
                        delta=f"{pred_delta:+.3f}",
                    )

                # Show comparison visualization
                st.subheader("Feature Contribution Comparison")
                comparisons = [
                    whatif_result["base_prediction"],
                    whatif_result["modified_prediction"],
                ]
                labels = ["Base", "Modified"]

                fig = visualizer.comparison_plot(comparisons, labels)
                st.plotly_chart(fig, use_container_width=True)

                # Show feature changes
                st.subheader("Feature Changes")
                changes_df = pd.DataFrame(
                    [
                        {
                            "Feature": feat,
                            "Original": base_features[feat].values[0],
                            "Modified": val,
                            "Change": val - base_features[feat].values[0],
                        }
                        for feat, val in feature_changes.items()
                    ]
                )
                st.dataframe(changes_df, use_container_width=True)

            except Exception as e:
                logger.error("whatif_analysis_failed", error=str(e))
                st.error(f"Error in what-if analysis: {str(e)}")


def _show_model_comparison_tab(visualizer: SHAPVisualizer, max_features: int) -> None:
    """Show model comparison tab.

    Args:
        visualizer: SHAP visualizer instance
        max_features: Maximum features to display
    """
    st.header("Model Comparison")
    st.markdown(
        """
        Compare how different models explain the same prediction.
        See which features each model considers most important.
        """
    )

    st.info(
        "🚧 Model comparison feature coming soon! "
        "This will allow you to compare explanations across multiple model types."
    )

    # Placeholder for future implementation
    st.markdown(
        """
        **Planned Features:**
        - Side-by-side model explanations
        - Feature importance comparison across models
        - Prediction consistency analysis
        - Model agreement metrics
        """
    )
