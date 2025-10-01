"""Race preview generator for creating comprehensive pre-race analysis."""

from typing import Any, Optional

import structlog

from f1_predict.analysis.base import BaseAnalyzer
from f1_predict.analysis.historical_context import HistoricalContextProvider
from f1_predict.llm.base import BaseLLMProvider
from f1_predict.llm.templates import PromptTemplateManager

logger = structlog.get_logger(__name__)


class RacePreviewGenerator(BaseAnalyzer):
    """Generate comprehensive race previews using LLM and historical data.

    Creates professional-quality race previews (400-600 words) that combine
    ML predictions, historical context, and engaging narrative to provide
    value to F1 fans.
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        historical_provider: Optional[HistoricalContextProvider] = None,
    ):
        """Initialize race preview generator.

        Args:
            llm_provider: LLM provider for narrative generation
            historical_provider: Historical context provider (creates new if None)
        """
        super().__init__(llm_provider)
        self.template_manager = PromptTemplateManager()
        self.historical_provider = historical_provider or HistoricalContextProvider(
            llm_provider
        )

    async def generate(
        self,
        race_name: str,
        circuit_name: str,
        race_date: str,
        round_number: int,
        circuit_characteristics: str,
        past_winner: Optional[str] = None,
        lap_record: Optional[str] = None,
        interesting_facts: Optional[str] = None,
        top_drivers: Optional[list[dict[str, Any]]] = None,
        predicted_winner: Optional[str] = None,
        confidence: Optional[float] = None,
        predicted_podium: Optional[str] = None,
        key_factors: Optional[list[str]] = None,
        include_historical: bool = True,
    ) -> dict[str, Any]:
        """Generate race preview.

        Args:
            race_name: Name of the race (e.g., "Monaco Grand Prix")
            circuit_name: Name of the circuit
            race_date: Date of the race (ISO format)
            round_number: Round number in championship
            circuit_characteristics: Description of circuit features
            past_winner: Previous year's winner (optional)
            lap_record: Lap record details (optional)
            interesting_facts: Notable circuit facts (optional)
            top_drivers: List of championship leaders with points
            predicted_winner: ML model's predicted winner
            confidence: Prediction confidence (0-100)
            predicted_podium: Predicted podium finishers
            key_factors: Key prediction factors
            include_historical: Whether to include historical context

        Returns:
            Dictionary containing race preview and metadata

        Raises:
            ValueError: If required parameters are missing
        """
        if not race_name or not circuit_name:
            msg = "Race name and circuit name are required"
            raise ValueError(msg)

        self.logger.info(
            "generating_race_preview",
            race=race_name,
            circuit=circuit_name,
            round=round_number,
        )

        # Get historical context if requested
        historical_context = None
        if include_historical:
            driver_names = [d["name"] for d in (top_drivers or [])][:5]
            historical_context = await self.historical_provider.generate(
                circuit_name=circuit_name,
                driver_names=driver_names if driver_names else None,
                max_facts=5,
            )

        # Prepare template context
        template_context = {
            "race_name": race_name,
            "circuit_name": circuit_name,
            "race_date": race_date,
            "round_number": round_number,
            "circuit_characteristics": circuit_characteristics
            or "A challenging circuit requiring precision",
            "past_winner": past_winner or "Previous winner data unavailable",
            "lap_record": lap_record or "Lap record details unavailable",
            "interesting_facts": interesting_facts,
            "top_drivers": top_drivers or [],
            "predicted_winner": predicted_winner or "TBD",
            "confidence": confidence or 50,
            "predicted_podium": predicted_podium or "TBD",
            "key_factors": key_factors or [],
        }

        # Add historical facts to context
        if historical_context:
            circuit_facts = historical_context.get("circuit_facts", [])
            if circuit_facts:
                template_context["historical_facts"] = [
                    f["fact"] for f in circuit_facts[:3]
                ]

            patterns = historical_context.get("interesting_patterns", [])
            if patterns:
                template_context["patterns"] = patterns[:2]

        # Generate preview using LLM
        try:
            preview_text = await self._generate_preview_text(template_context)

            result = {
                "race_name": race_name,
                "circuit_name": circuit_name,
                "race_date": race_date,
                "round_number": round_number,
                "preview_text": preview_text,
                "predicted_winner": predicted_winner,
                "confidence": confidence,
                "predicted_podium": predicted_podium,
                "key_factors": key_factors or [],
                "circuit_facts": historical_context.get("circuit_facts", [])
                if historical_context
                else [],
                "interesting_patterns": historical_context.get(
                    "interesting_patterns", []
                )
                if historical_context
                else [],
            }

            # Add readability metrics
            if preview_text:
                result["readability"] = self._calculate_readability(preview_text)

            return self._add_metadata(result)

        except Exception as e:
            self.logger.error("preview_generation_failed", error=str(e), race=race_name)
            raise

    async def _generate_preview_text(self, context: dict[str, Any]) -> str:
        """Generate preview text using LLM.

        Args:
            context: Template context with race data

        Returns:
            Generated preview text

        Raises:
            Exception: If LLM generation fails
        """
        try:
            # Use race preview template
            prompt = self.template_manager.render(
                "race_preview.jinja2",
                **context,
            )

            # Generate with LLM
            response = await self.llm_provider.generate(
                prompt=prompt,
                temperature=0.7,  # Balance creativity and consistency
                max_tokens=800,  # ~600 words
            )

            preview_text = response.content.strip()

            # Validate word count
            word_count = len(preview_text.split())
            if word_count < 300:
                self.logger.warning(
                    "preview_too_short",
                    word_count=word_count,
                    race=context.get("race_name"),
                )
            elif word_count > 700:
                self.logger.warning(
                    "preview_too_long",
                    word_count=word_count,
                    race=context.get("race_name"),
                )
                # Truncate if too long
                words = preview_text.split()[:650]
                preview_text = " ".join(words) + "..."

            return preview_text

        except Exception as e:
            self.logger.error("llm_generation_failed", error=str(e))
            # Return fallback preview
            return self._generate_fallback_preview(context)

    def _generate_fallback_preview(self, context: dict[str, Any]) -> str:
        """Generate basic preview without LLM.

        Args:
            context: Race context data

        Returns:
            Basic preview text
        """
        race_name = context.get("race_name", "Grand Prix")
        circuit_name = context.get("circuit_name", "the circuit")
        predicted_winner = context.get("predicted_winner", "TBD")
        confidence = context.get("confidence", 50)

        parts = [
            f"The {race_name} returns to {circuit_name} for Round {context.get('round_number', 'X')} of the championship.",
            "",
            f"Circuit Characteristics: {context.get('circuit_characteristics', 'A challenging circuit')}",
            "",
        ]

        if context.get("past_winner"):
            parts.append(f"Last year's winner: {context['past_winner']}")
            parts.append("")

        if predicted_winner and predicted_winner != "TBD":
            parts.append(
                f"Our ML models predict {predicted_winner} to win with {confidence:.0f}% confidence."
            )
            parts.append("")

        if context.get("key_factors"):
            parts.append("Key factors in this prediction:")
            for factor in context["key_factors"][:3]:
                parts.append(f"• {factor}")
            parts.append("")

        parts.append(
            f"The {race_name} promises exciting racing and unpredictable outcomes. "
            "Stay tuned for qualifying and race day action!"
        )

        return "\n".join(parts)

    async def generate_drivers_to_watch(
        self,
        predictions: list[dict[str, Any]],
        max_drivers: int = 4,
    ) -> list[dict[str, Any]]:
        """Generate "drivers to watch" section.

        Args:
            predictions: List of driver predictions with confidence
            max_drivers: Maximum drivers to highlight

        Returns:
            List of driver highlight dictionaries
        """
        drivers_to_watch = []

        # Sort by confidence and select top drivers
        sorted_predictions = sorted(
            predictions,
            key=lambda x: x.get("confidence", 0),
            reverse=True,
        )

        for pred in sorted_predictions[:max_drivers]:
            driver = {
                "name": pred.get("driver_name", "Unknown"),
                "predicted_finish": f"P{pred.get('position', 'X')}",
                "confidence": pred.get("confidence", 0),
                "reasoning": pred.get("reasoning", "Strong form expected"),
            }
            drivers_to_watch.append(driver)

        return drivers_to_watch
