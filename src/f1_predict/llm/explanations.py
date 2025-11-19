"""Natural language explanation generation for F1 predictions.

This module generates human-readable explanations for model predictions
using LLM providers, explaining prediction confidence and reasoning.
"""

from typing import Any, Optional

import structlog

from f1_predict.llm.base import BaseLLMProvider, LLMResponse
from f1_predict.llm.chat_session import ChatSession
from f1_predict.llm.templates import PromptTemplateManager

logger = structlog.get_logger(__name__)


class F1PredictionExplainer:
    """Generate natural language explanations for F1 race predictions.

    Uses LLM providers to create human-friendly explanations of prediction
    confidence, key factors, and reasoning.
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        template_manager: Optional[PromptTemplateManager] = None,
    ):
        """Initialize prediction explainer.

        Args:
            provider: LLM provider for generating explanations
            template_manager: Optional prompt template manager
        """
        self.provider = provider
        self.template_manager = template_manager or PromptTemplateManager()
        self.logger = structlog.get_logger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("explainer_initialized", provider=provider.name)

    async def explain_race_prediction(
        self,
        race_name: str,
        drivers: list[str],
        predicted_winner: str,
        confidence: float,
        key_factors: dict[str, Any],
        session: Optional[ChatSession] = None,
    ) -> str:
        """Generate explanation for race prediction.

        Args:
            race_name: Name of the race
            drivers: List of competing drivers
            predicted_winner: Predicted race winner
            confidence: Prediction confidence (0.0-1.0)
            key_factors: Dictionary of key prediction factors
            session: Optional chat session for context

        Returns:
            Natural language explanation of prediction

        Raises:
            ValueError: If confidence not in valid range
        """
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {confidence}")

        factors_text = "\n".join(
            [f"- {k}: {v}" for k, v in key_factors.items()]
        )

        prompt = f"""Generate a concise, natural explanation of an F1 race prediction:

Race: {race_name}
Drivers: {", ".join(drivers)}
Predicted Winner: {predicted_winner}
Confidence: {confidence:.1%}

Key Factors:
{factors_text}

Provide a brief, informative explanation (2-3 sentences) explaining why this driver is predicted to win and the key contributing factors."""

        self.logger.debug(
            "race_explanation_requested",
            race=race_name,
            predicted_winner=predicted_winner,
            confidence=confidence,
        )

        try:
            response = await self.provider.generate(
                prompt=prompt,
                system_prompt="You are an F1 expert analyst providing concise race predictions.",
            )

            if session:
                session.add_message("user", f"Explain prediction for {race_name}")
                session.add_message("assistant", response.content)

            self.logger.info(
                "race_explanation_generated",
                race=race_name,
                tokens_used=response.total_tokens,
            )

            return response.content
        except Exception as e:
            self.logger.error(
                "explanation_generation_failed",
                race=race_name,
                error=str(e),
            )
            raise

    async def explain_driver_performance(
        self,
        driver_name: str,
        recent_form: dict[str, Any],
        circuit_factors: dict[str, float],
        weather_forecast: dict[str, Any],
        session: Optional[ChatSession] = None,
    ) -> str:
        """Generate explanation for driver performance prediction.

        Args:
            driver_name: Name of driver
            recent_form: Driver's recent performance data
            circuit_factors: Circuit-specific factors
            weather_forecast: Weather forecast data
            session: Optional chat session

        Returns:
            Explanation of driver performance prediction
        """
        recent_text = "\n".join(
            [f"- {k}: {v}" for k, v in recent_form.items()]
        )
        circuit_text = "\n".join(
            [f"- {k}: {v:.2f}" for k, v in circuit_factors.items()]
        )
        weather_text = "\n".join(
            [f"- {k}: {v}" for k, v in weather_forecast.items()]
        )

        prompt = f"""Analyze and explain {driver_name}'s predicted performance:

Recent Form:
{recent_text}

Circuit Factors:
{circuit_text}

Weather Forecast:
{weather_text}

Provide a brief explanation (2-3 sentences) of how these factors affect the driver's expected performance."""

        self.logger.debug("driver_explanation_requested", driver=driver_name)

        try:
            response = await self.provider.generate(
                prompt=prompt,
                system_prompt="You are an F1 performance analyst.",
            )

            if session:
                session.add_message("user", f"Analyze {driver_name}'s performance")
                session.add_message("assistant", response.content)

            self.logger.info(
                "driver_explanation_generated",
                driver=driver_name,
                tokens=response.total_tokens,
            )

            return response.content
        except Exception as e:
            self.logger.error(
                "driver_explanation_failed",
                driver=driver_name,
                error=str(e),
            )
            raise

    async def explain_prediction_uncertainty(
        self,
        prediction: str,
        confidence: float,
        uncertainty_factors: list[str],
        session: Optional[ChatSession] = None,
    ) -> str:
        """Explain sources of uncertainty in predictions.

        Args:
            prediction: The prediction being explained
            confidence: Prediction confidence
            uncertainty_factors: Factors contributing to uncertainty
            session: Optional chat session

        Returns:
            Explanation of prediction uncertainty
        """
        factors_text = "\n".join([f"- {factor}" for factor in uncertainty_factors])

        prompt = f"""Explain the uncertainty in this F1 prediction:

Prediction: {prediction}
Confidence Level: {confidence:.1%}

Uncertainty Factors:
{factors_text}

Briefly explain (2-3 sentences) why this prediction has moderate-to-low confidence and what factors contribute to the uncertainty."""

        self.logger.debug(
            "uncertainty_explanation_requested",
            confidence=confidence,
            factor_count=len(uncertainty_factors),
        )

        try:
            response = await self.provider.generate(
                prompt=prompt,
                system_prompt="You are an expert at explaining prediction uncertainty.",
            )

            if session:
                session.add_message("user", f"Explain uncertainty: {prediction}")
                session.add_message("assistant", response.content)

            return response.content
        except Exception as e:
            self.logger.error(
                "uncertainty_explanation_failed",
                error=str(e),
            )
            raise

    async def generate_detailed_analysis(
        self,
        race_name: str,
        data: dict[str, Any],
        session: Optional[ChatSession] = None,
    ) -> str:
        """Generate detailed multi-paragraph analysis.

        Args:
            race_name: Name of the race
            data: Dictionary containing all relevant analysis data
            session: Optional chat session

        Returns:
            Detailed analysis text
        """
        data_text = "\n".join([f"{k}: {v}" for k, v in data.items()])

        prompt = f"""Provide a detailed analysis (4-5 paragraphs) of the {race_name} race prediction:

Data:
{data_text}

Structure your analysis with:
1. Race context and expectations
2. Top contenders analysis
3. Key variables and track conditions
4. Prediction and confidence reasoning
5. Potential surprises or wildcards"""

        self.logger.debug("detailed_analysis_requested", race=race_name)

        try:
            response = await self.provider.generate(
                prompt=prompt,
                system_prompt="You are a professional F1 race analyst providing comprehensive insights.",
                max_tokens=2000,
            )

            if session:
                session.add_message("user", f"Analyze {race_name} in detail")
                session.add_message("assistant", response.content)

            self.logger.info(
                "detailed_analysis_generated",
                race=race_name,
                tokens=response.total_tokens,
            )

            return response.content
        except Exception as e:
            self.logger.error(
                "detailed_analysis_failed",
                race=race_name,
                error=str(e),
            )
            raise

    def create_explanation_summary(
        self,
        explanation: str,
        confidence: float,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Create structured explanation summary.

        Args:
            explanation: Generated explanation text
            confidence: Prediction confidence
            metadata: Optional additional metadata

        Returns:
            Structured explanation summary
        """
        return {
            "explanation": explanation,
            "confidence": confidence,
            "provider": self.provider.name,
            "model": self.provider.config.model,
            "metadata": metadata or {},
        }
