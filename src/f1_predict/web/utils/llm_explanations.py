"""LLM explanation utilities for web interface.

Provides natural language generation capabilities for the web UI,
integrating with the LLM providers and explanation generators.
"""

import asyncio
import os
from typing import Any, Optional

import structlog

from f1_predict.llm.base import LLMConfig
from f1_predict.llm.exceptions import (
    AuthenticationError,
    LLMError,
    ProviderUnavailableError,
    RateLimitError,
)
from f1_predict.llm.explanations import F1PredictionExplainer

logger = structlog.get_logger(__name__)

# Default provider configuration
DEFAULT_MODEL = "claude-3-haiku-20240307"
DEFAULT_PROVIDER = "anthropic"


def _get_provider(
    provider_name: str = DEFAULT_PROVIDER,
    model: Optional[str] = None,
) -> Optional[Any]:
    """Get an LLM provider instance.

    Args:
        provider_name: Name of provider ('anthropic', 'openai', 'local')
        model: Optional model override

    Returns:
        Provider instance or None if not available
    """
    config = LLMConfig(
        model=model or DEFAULT_MODEL,
        temperature=0.7,
        max_tokens=1024,
    )

    try:
        if provider_name == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                logger.warning("anthropic_api_key_not_set")
                return None
            from f1_predict.llm.anthropic_provider import AnthropicProvider

            return AnthropicProvider(config, api_key)

        if provider_name == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                logger.warning("openai_api_key_not_set")
                return None
            from f1_predict.llm.openai_provider import OpenAIProvider

            return OpenAIProvider(config, api_key)

        if provider_name == "local":
            from f1_predict.llm.local_provider import LocalProvider

            return LocalProvider(config)

        logger.warning("unknown_provider", provider=provider_name)
        return None

    except Exception as e:
        logger.error("provider_init_failed", provider=provider_name, error=str(e))
        return None


def _run_async(coro: Any) -> Any:
    """Run async coroutine in sync context.

    Args:
        coro: Coroutine to run

    Returns:
        Result of coroutine
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Create a new loop in a thread for nested async calls
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result(timeout=30)
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def generate_prediction_explanation(
    prediction: dict[str, Any],
    provider_name: str = DEFAULT_PROVIDER,
    model: Optional[str] = None,
) -> str:
    """Generate natural language explanation for a race prediction.

    Args:
        prediction: Prediction dictionary with podium, confidence scores
        provider_name: LLM provider to use
        model: Optional model override

    Returns:
        String explanation of the prediction
    """
    # Extract prediction data
    podium = prediction.get("podium", [])
    race_name = prediction.get("race_name", "Unknown Race")
    confidence = prediction.get("overall_confidence", 0.7)

    # Fallback to simple formatting if no LLM available
    provider = _get_provider(provider_name, model)
    if not provider:
        return _format_simple_explanation(podium)

    # Build context for LLM
    drivers = [entry.get("driver", "Unknown") for entry in podium]
    predicted_winner = drivers[0] if drivers else "Unknown"

    key_factors = {
        "Qualifying Performance": prediction.get("qualifying_factor", "Strong"),
        "Recent Form": prediction.get("form_factor", "Good"),
        "Circuit History": prediction.get("circuit_factor", "Favorable"),
        "Weather Conditions": prediction.get("weather_factor", "Dry"),
    }

    try:
        explainer = F1PredictionExplainer(provider)
        explanation = _run_async(
            explainer.explain_race_prediction(
                race_name=race_name,
                drivers=drivers[:10],  # Top 10 drivers
                predicted_winner=predicted_winner,
                confidence=confidence,
                key_factors=key_factors,
            )
        )
        logger.info("prediction_explanation_generated", race=race_name)
        return explanation

    except (RateLimitError, AuthenticationError, ProviderUnavailableError) as e:
        logger.warning("llm_explanation_failed", error=str(e))
        return _format_simple_explanation(podium)
    except LLMError as e:
        logger.error("llm_error", error=str(e))
        return _format_simple_explanation(podium)
    except Exception as e:
        logger.error("explanation_failed", error=str(e))
        return _format_simple_explanation(podium)


def _format_simple_explanation(podium: list[dict[str, Any]]) -> str:
    """Format a simple explanation without LLM.

    Args:
        podium: List of podium predictions

    Returns:
        Simple formatted explanation
    """
    if not podium:
        return "No prediction available."

    explanation_parts = []
    for entry in podium:
        driver = entry.get("driver", "Unknown")
        confidence = entry.get("confidence", 0.0)
        position = entry.get("position", 0)
        explanation_parts.append(
            f"Position {position}: {driver} ({confidence:.0%} confidence)"
        )

    return " | ".join(explanation_parts)


def generate_driver_comparison(
    driver1: str,
    driver2: str,
    stats1: Optional[dict[str, Any]] = None,
    stats2: Optional[dict[str, Any]] = None,
    provider_name: str = DEFAULT_PROVIDER,
    model: Optional[str] = None,
) -> str:
    """Generate comparison between two drivers.

    Args:
        driver1: First driver name
        driver2: Second driver name
        stats1: Optional stats for driver1
        stats2: Optional stats for driver2
        provider_name: LLM provider to use
        model: Optional model override

    Returns:
        String comparison of the two drivers
    """
    provider = _get_provider(provider_name, model)
    if not provider:
        return _format_simple_comparison(driver1, driver2, stats1, stats2)

    # Build comparison prompt
    stats1 = stats1 or {}
    stats2 = stats2 or {}

    prompt = f"""Compare the following two F1 drivers based on their statistics:

**{driver1}**
- Wins: {stats1.get('wins', 'N/A')}
- Podiums: {stats1.get('podiums', 'N/A')}
- Points: {stats1.get('points', 'N/A')}
- Qualifying Average: {stats1.get('qualifying_avg', 'N/A')}
- Recent Form: {stats1.get('recent_form', 'N/A')}

**{driver2}**
- Wins: {stats2.get('wins', 'N/A')}
- Podiums: {stats2.get('podiums', 'N/A')}
- Points: {stats2.get('points', 'N/A')}
- Qualifying Average: {stats2.get('qualifying_avg', 'N/A')}
- Recent Form: {stats2.get('recent_form', 'N/A')}

Provide a concise comparison (3-4 sentences) highlighting key differences in their performance and strengths."""

    try:
        response = _run_async(
            provider.generate(
                prompt=prompt,
                system_prompt="You are an expert F1 analyst comparing driver performances.",
            )
        )
        logger.info("driver_comparison_generated", driver1=driver1, driver2=driver2)
        return response.content

    except (RateLimitError, AuthenticationError, ProviderUnavailableError) as e:
        logger.warning("llm_comparison_failed", error=str(e))
        return _format_simple_comparison(driver1, driver2, stats1, stats2)
    except LLMError as e:
        logger.error("llm_error", error=str(e))
        return _format_simple_comparison(driver1, driver2, stats1, stats2)
    except Exception as e:
        logger.error("comparison_failed", error=str(e))
        return _format_simple_comparison(driver1, driver2, stats1, stats2)


def _format_simple_comparison(
    driver1: str,
    driver2: str,
    stats1: Optional[dict[str, Any]],
    stats2: Optional[dict[str, Any]],
) -> str:
    """Format a simple comparison without LLM.

    Args:
        driver1: First driver name
        driver2: Second driver name
        stats1: Stats for driver1
        stats2: Stats for driver2

    Returns:
        Simple formatted comparison
    """
    stats1 = stats1 or {}
    stats2 = stats2 or {}

    lines = [f"**{driver1} vs {driver2}**\n"]

    if stats1.get("wins") is not None and stats2.get("wins") is not None:
        lines.append(
            f"Wins: {driver1} ({stats1['wins']}) vs {driver2} ({stats2['wins']})"
        )

    if stats1.get("points") is not None and stats2.get("points") is not None:
        lines.append(
            f"Points: {driver1} ({stats1['points']}) vs {driver2} ({stats2['points']})"
        )

    if not lines[1:]:
        lines.append("Detailed statistics not available for comparison.")

    return "\n".join(lines)


def generate_race_preview(
    race_name: str,
    circuit_name: str,
    drivers: list[str],
    weather_forecast: Optional[dict[str, Any]] = None,
    historical_data: Optional[dict[str, Any]] = None,
    provider_name: str = DEFAULT_PROVIDER,
    model: Optional[str] = None,
) -> str:
    """Generate a race preview article.

    Args:
        race_name: Name of the race
        circuit_name: Name of the circuit
        drivers: List of competing drivers
        weather_forecast: Optional weather data
        historical_data: Optional historical race data
        provider_name: LLM provider to use
        model: Optional model override

    Returns:
        Race preview text
    """
    provider = _get_provider(provider_name, model)
    if not provider:
        return _format_simple_preview(race_name, circuit_name, drivers)

    weather_forecast = weather_forecast or {}
    historical_data = historical_data or {}

    prompt = f"""Write a race preview article for the {race_name} at {circuit_name}.

**Drivers**: {', '.join(drivers[:10])}

**Weather Forecast**:
- Temperature: {weather_forecast.get('temperature', 'N/A')}
- Conditions: {weather_forecast.get('conditions', 'N/A')}
- Rain Chance: {weather_forecast.get('rain_chance', 'N/A')}

**Historical Data**:
- Most Wins: {historical_data.get('most_wins_driver', 'N/A')}
- Track Record: {historical_data.get('track_record', 'N/A')}
- Overtaking Difficulty: {historical_data.get('overtaking', 'N/A')}

Write a 200-300 word preview covering:
1. Circuit characteristics and challenges
2. Drivers to watch
3. Key factors that could influence the race
4. Bold prediction"""

    try:
        response = _run_async(
            provider.generate(
                prompt=prompt,
                system_prompt="You are a professional F1 journalist writing an engaging race preview.",
                max_tokens=1500,
            )
        )
        logger.info("race_preview_generated", race=race_name)
        return response.content

    except (RateLimitError, AuthenticationError, ProviderUnavailableError) as e:
        logger.warning("llm_preview_failed", error=str(e))
        return _format_simple_preview(race_name, circuit_name, drivers)
    except LLMError as e:
        logger.error("llm_error", error=str(e))
        return _format_simple_preview(race_name, circuit_name, drivers)
    except Exception as e:
        logger.error("preview_failed", error=str(e))
        return _format_simple_preview(race_name, circuit_name, drivers)


def _format_simple_preview(
    race_name: str,
    circuit_name: str,
    drivers: list[str],
) -> str:
    """Format a simple preview without LLM.

    Args:
        race_name: Race name
        circuit_name: Circuit name
        drivers: List of drivers

    Returns:
        Simple formatted preview
    """
    top_drivers = drivers[:5] if drivers else ["No drivers available"]
    return f"""**{race_name} at {circuit_name}**

The grid lines up for another exciting race weekend.

**Top Contenders**: {', '.join(top_drivers)}

Stay tuned for qualifying and race predictions!"""


def generate_post_race_analysis(
    race_name: str,
    results: list[dict[str, Any]],
    key_moments: Optional[list[str]] = None,
    provider_name: str = DEFAULT_PROVIDER,
    model: Optional[str] = None,
) -> str:
    """Generate post-race analysis.

    Args:
        race_name: Name of the race
        results: Race results (position, driver, time/gap)
        key_moments: Optional list of key race moments
        provider_name: LLM provider to use
        model: Optional model override

    Returns:
        Post-race analysis text
    """
    provider = _get_provider(provider_name, model)
    if not provider:
        return _format_simple_post_race(race_name, results)

    key_moments = key_moments or []

    # Format results
    results_text = "\n".join(
        [
            f"{r.get('position', '?')}. {r.get('driver', 'Unknown')} - {r.get('gap', 'N/A')}"
            for r in results[:10]
        ]
    )

    moments_text = (
        "\n".join([f"- {moment}" for moment in key_moments]) if key_moments else "N/A"
    )

    prompt = f"""Write a post-race analysis for the {race_name}.

**Final Results (Top 10)**:
{results_text}

**Key Moments**:
{moments_text}

Write a 200-300 word analysis covering:
1. Race winner performance
2. Noteworthy battles and overtakes
3. Strategic decisions that made a difference
4. Championship implications"""

    try:
        response = _run_async(
            provider.generate(
                prompt=prompt,
                system_prompt="You are an F1 analyst providing insightful post-race analysis.",
                max_tokens=1500,
            )
        )
        logger.info("post_race_analysis_generated", race=race_name)
        return response.content

    except (RateLimitError, AuthenticationError, ProviderUnavailableError) as e:
        logger.warning("llm_analysis_failed", error=str(e))
        return _format_simple_post_race(race_name, results)
    except LLMError as e:
        logger.error("llm_error", error=str(e))
        return _format_simple_post_race(race_name, results)
    except Exception as e:
        logger.error("analysis_failed", error=str(e))
        return _format_simple_post_race(race_name, results)


def _format_simple_post_race(
    race_name: str,
    results: list[dict[str, Any]],
) -> str:
    """Format simple post-race summary without LLM.

    Args:
        race_name: Race name
        results: Race results

    Returns:
        Simple formatted analysis
    """
    if not results:
        return f"**{race_name}** - Results not available."

    winner = results[0].get("driver", "Unknown") if results else "Unknown"
    podium = [r.get("driver", "Unknown") for r in results[:3]]

    return f"""**{race_name} - Race Analysis**

**Winner**: {winner}

**Podium**: {', '.join(podium)}

Full analysis coming soon."""


def check_llm_availability(provider_name: str = DEFAULT_PROVIDER) -> dict[str, Any]:
    """Check if LLM provider is available and configured.

    Args:
        provider_name: Provider to check

    Returns:
        Dictionary with availability status and details
    """
    result = {
        "provider": provider_name,
        "available": False,
        "api_key_set": False,
        "error": None,
    }

    if provider_name == "anthropic":
        result["api_key_set"] = bool(os.environ.get("ANTHROPIC_API_KEY"))
    elif provider_name == "openai":
        result["api_key_set"] = bool(os.environ.get("OPENAI_API_KEY"))
    elif provider_name == "local":
        result["api_key_set"] = True  # No key needed for local

    provider = _get_provider(provider_name)
    if provider:
        result["available"] = True
        result["model"] = provider.config.model
    else:
        result["error"] = (
            "Provider initialization failed"
            if result["api_key_set"]
            else "API key not configured"
        )

    return result
