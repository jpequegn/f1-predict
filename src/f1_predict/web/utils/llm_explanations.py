"""LLM explanation utilities for web interface."""
from typing import Any

# TODO: Import actual LLM client from Issue #11
# from f1_predict.llm.client import LLMClient


def generate_prediction_explanation(prediction: dict[str, Any]) -> str:
    """Generate natural language explanation for a race prediction.

    Args:
        prediction: Prediction dictionary with podium, confidence scores

    Returns:
        String explanation of the prediction
    """
    # TODO: Integrate with LLM API from Issue #11

    podium = prediction.get("podium", [])
    explanation_parts = []

    for entry in podium:
        driver = entry.get("driver", "Unknown")
        confidence = entry.get("confidence", 0.0)
        position = entry.get("position", 0)

        explanation_parts.append(
            f"Position {position}: {driver} ({confidence:.0%} confidence)"
        )

    return " → ".join(explanation_parts) if explanation_parts else "No prediction available."


def generate_driver_comparison(driver1: str, driver2: str) -> str:
    """Generate comparison between two drivers.

    Args:
        driver1: First driver name
        driver2: Second driver name

    Returns:
        String comparison of the two drivers
    """
    # TODO: Integrate with LLM API from Issue #11

    return f"Comparison between {driver1} and {driver2} coming soon."
