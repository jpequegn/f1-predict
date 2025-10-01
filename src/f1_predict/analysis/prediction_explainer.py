"""Prediction explanation generator for translating ML predictions to plain English."""

from typing import Any, Optional

import structlog

from f1_predict.analysis.base import BaseAnalyzer
from f1_predict.llm.base import BaseLLMProvider
from f1_predict.llm.templates import PromptTemplateManager

logger = structlog.get_logger(__name__)


# Feature name translations to plain English
FEATURE_EXPLANATIONS = {
    "qualifying_position": "starting grid position",
    "driver_form_score": "recent race performance",
    "team_reliability_score": "car reliability",
    "circuit_performance_score": "historical performance at this track",
    "championship_position": "current championship standings",
    "avg_finish_position": "average finishing position",
    "podium_rate": "podium finish percentage",
    "dnf_rate": "retirement rate",
    "points_per_race": "average points scored per race",
    "overtakes_made": "overtaking ability",
    "fastest_laps": "pace on fresh tires",
    "tire_management": "tire conservation ability",
    "wet_weather_performance": "performance in rain",
    "street_circuit_performance": "performance on street tracks",
}

# Confidence level messaging
CONFIDENCE_MESSAGES = {
    (90, 100): "Very strong prediction - high probability",
    (75, 90): "Confident prediction - likely outcome",
    (60, 75): "Moderate prediction - reasonable chance",
    (50, 60): "Slight edge - could go either way",
    (0, 50): "Uncertain prediction - many variables at play",
}


class PredictionExplainer(BaseAnalyzer):
    """Generate plain English explanations for ML predictions.

    Translates complex ML model predictions into understandable explanations
    for non-technical users, including feature importance, historical context,
    and confidence level interpretation.
    """

    def __init__(self, llm_provider: BaseLLMProvider):
        """Initialize prediction explainer.

        Args:
            llm_provider: LLM provider for generating explanations
        """
        super().__init__(llm_provider)
        self.template_manager = PromptTemplateManager()

    async def generate(
        self,
        driver_name: str,
        position: int,
        confidence: float,
        model_name: str,
        top_features: list[tuple[str, float, str]],
        circuit: Optional[str] = None,
        past_performance: Optional[str] = None,
        recent_results: Optional[str] = None,
        head_to_head: Optional[str] = None,
        detail_level: str = "detailed",
    ) -> dict[str, Any]:
        """Generate prediction explanation.

        Args:
            driver_name: Driver name for prediction
            position: Predicted finishing position
            confidence: Confidence level (0-100)
            model_name: Name of ML model used
            top_features: List of (feature_name, importance, explanation) tuples
            circuit: Circuit name (optional)
            past_performance: Historical performance summary (optional)
            recent_results: Recent race results (optional)
            head_to_head: Head-to-head comparison (optional)
            detail_level: Level of detail ("simple", "detailed", "technical")

        Returns:
            Dictionary containing explanation and metadata

        Raises:
            ValueError: If required parameters are invalid
        """
        if not driver_name:
            msg = "Driver name is required"
            raise ValueError(msg)

        if not 1 <= position <= 20:
            msg = f"Position must be between 1 and 20, got {position}"
            raise ValueError(msg)

        if not 0 <= confidence <= 100:
            msg = f"Confidence must be between 0 and 100, got {confidence}"
            raise ValueError(msg)

        self.logger.info(
            "generating_prediction_explanation",
            driver=driver_name,
            position=position,
            confidence=confidence,
            detail_level=detail_level,
        )

        # Generate explanation based on detail level
        if detail_level == "simple":
            explanation = self._generate_simple_explanation(
                driver_name, position, confidence, top_features
            )
        elif detail_level == "technical":
            explanation = await self._generate_technical_explanation(
                driver_name, position, confidence, model_name, top_features
            )
        else:  # detailed
            explanation = await self._generate_detailed_explanation(
                driver_name,
                position,
                confidence,
                model_name,
                top_features,
                circuit,
                past_performance,
                recent_results,
                head_to_head,
            )

        result = {
            "driver_name": driver_name,
            "predicted_position": position,
            "confidence": confidence,
            "confidence_message": self._get_confidence_message(confidence),
            "explanation": explanation,
            "detail_level": detail_level,
            "top_features": self._format_features(top_features),
        }

        return self._add_metadata(result)

    def _generate_simple_explanation(
        self,
        driver_name: str,
        position: int,
        confidence: float,
        top_features: list[tuple[str, float, str]],
    ) -> str:
        """Generate simple 1-2 sentence explanation.

        Args:
            driver_name: Driver name
            position: Predicted position
            confidence: Confidence percentage
            top_features: Top contributing features

        Returns:
            Simple explanation string
        """
        position_text = "win" if position == 1 else f"finish P{position}"
        top_feature = (
            top_features[0] if top_features else ("qualifying", 0.5, "strong form")
        )

        return (
            f"We predict {driver_name} to {position_text} with {confidence:.0f}% confidence "
            f"based primarily on {self._translate_feature(top_feature[0])} {top_feature[2]}."
        )

    async def _generate_detailed_explanation(
        self,
        driver_name: str,
        position: int,
        confidence: float,
        model_name: str,
        top_features: list[tuple[str, float, str]],
        circuit: Optional[str],
        past_performance: Optional[str],
        recent_results: Optional[str],
        head_to_head: Optional[str],
    ) -> str:
        """Generate detailed explanation using LLM.

        Args:
            driver_name: Driver name
            position: Predicted position
            confidence: Confidence percentage
            model_name: Model name
            top_features: Top features with importance
            circuit: Circuit name
            past_performance: Past performance summary
            recent_results: Recent results
            head_to_head: Head-to-head data

        Returns:
            Detailed explanation text
        """
        try:
            prompt = self.template_manager.render(
                "prediction_explanation.jinja2",
                driver_name=driver_name,
                position=position,
                confidence=confidence,
                model_name=model_name,
                top_features=[
                    (self._translate_feature(f[0]), f[1], f[2]) for f in top_features
                ],
                circuit=circuit or "this circuit",
                past_performance=past_performance or "No historical data available",
                recent_results=recent_results or "No recent results available",
                head_to_head=head_to_head,
            )

            response = await self.llm_provider.generate(prompt=prompt)
            return response.content.strip()

        except Exception as e:
            self.logger.error("llm_generation_failed", error=str(e))
            # Fallback to simple explanation
            return self._generate_simple_explanation(
                driver_name, position, confidence, top_features
            )

    async def _generate_technical_explanation(
        self,
        driver_name: str,
        position: int,
        confidence: float,
        model_name: str,
        top_features: list[tuple[str, float, str]],
    ) -> str:
        """Generate technical explanation with ML details.

        Args:
            driver_name: Driver name
            position: Predicted position
            confidence: Confidence percentage
            model_name: Model name
            top_features: Top features with importance

        Returns:
            Technical explanation string
        """
        explanation_parts = [
            f"**Model**: {model_name}",
            f"**Prediction**: P{position} ({confidence:.1f}% confidence)",
            "",
            "**Feature Importance**:",
        ]

        for feature, importance, context in top_features[:5]:
            explanation_parts.append(f"- {feature}: {importance:.3f} - {context}")

        explanation_parts.extend(
            [
                "",
                "**Statistical Confidence**:",
                f"The model assigns a {confidence:.1f}% probability to this outcome based on "
                f"historical patterns and current form indicators.",
            ]
        )

        return "\n".join(explanation_parts)

    def _translate_feature(self, feature_name: str) -> str:
        """Translate technical feature name to plain English.

        Args:
            feature_name: Technical feature name

        Returns:
            Plain English translation
        """
        return FEATURE_EXPLANATIONS.get(feature_name, feature_name.replace("_", " "))

    def _get_confidence_message(self, confidence: float) -> str:
        """Get confidence level message.

        Args:
            confidence: Confidence percentage

        Returns:
            Human-readable confidence message
        """
        for (low, high), message in CONFIDENCE_MESSAGES.items():
            if low <= confidence < high:
                return message
        return "Uncertain prediction"

    def _format_features(
        self, top_features: list[tuple[str, float, str]]
    ) -> list[dict[str, Any]]:
        """Format features for output.

        Args:
            top_features: List of (name, importance, explanation) tuples

        Returns:
            List of formatted feature dictionaries
        """
        return [
            {
                "feature": self._translate_feature(name),
                "importance": float(importance),
                "explanation": explanation,
            }
            for name, importance, explanation in top_features
        ]
