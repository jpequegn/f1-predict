"""Rule-based prediction system for F1 race outcomes.

This module provides a simple baseline predictor using heuristic rules:
- Qualifying position as primary indicator
- Recent driver form
- Team reliability
- Circuit-specific performance
"""

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class RuleBasedPredictor:
    """Simple rule-based predictor for race outcomes.

    Uses heuristic rules to predict race finishing positions:
    1. Qualifying position (60% weight)
    2. Driver form score (20% weight)
    3. Team reliability (10% weight)
    4. Circuit performance (10% weight)
    """

    def __init__(
        self,
        quali_weight: float = 0.6,
        form_weight: float = 0.2,
        reliability_weight: float = 0.1,
        circuit_weight: float = 0.1,
    ):
        """Initialize rule-based predictor.

        Args:
            quali_weight: Weight for qualifying position
            form_weight: Weight for driver form
            reliability_weight: Weight for team reliability
            circuit_weight: Weight for circuit performance
        """
        if (
            not abs(
                quali_weight + form_weight + reliability_weight + circuit_weight - 1.0
            )
            < 0.01
        ):
            msg = "Weights must sum to 1.0"
            raise ValueError(msg)

        self.quali_weight = quali_weight
        self.form_weight = form_weight
        self.reliability_weight = reliability_weight
        self.circuit_weight = circuit_weight
        self.logger = logger.bind(model="rule_based")

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Predict race finishing positions based on features.

        Args:
            features: DataFrame with columns:
                - driver_id
                - qualifying_position
                - driver_form_score (0-100)
                - team_reliability_score (0-100)
                - circuit_performance_score (0-100)

        Returns:
            DataFrame with predicted positions and confidence scores
        """
        self.logger.info("predicting_positions", num_drivers=len(features))

        if features.empty:
            return pd.DataFrame(
                columns=["driver_id", "predicted_position", "confidence"]
            )

        # Validate required columns
        required_cols = [
            "driver_id",
            "qualifying_position",
            "driver_form_score",
            "team_reliability_score",
            "circuit_performance_score",
        ]
        missing = set(required_cols) - set(features.columns)
        if missing:
            msg = f"Missing required columns: {missing}"
            raise ValueError(msg)

        # Calculate composite score (lower is better)
        # Normalize qualifying position to 0-100 scale (inverse)
        max_pos = features["qualifying_position"].max()
        quali_score = (max_pos - features["qualifying_position"] + 1) / max_pos * 100

        # Combine scores with weights
        composite_score = (
            self.quali_weight * quali_score
            + self.form_weight * features["driver_form_score"]
            + self.reliability_weight * features["team_reliability_score"]
            + self.circuit_weight * features["circuit_performance_score"]
        )

        # Rank by composite score (higher score = better prediction)
        features = features.copy()
        features["composite_score"] = composite_score
        features = features.sort_values("composite_score", ascending=False)

        # Assign predicted positions
        features["predicted_position"] = range(1, len(features) + 1)

        # Calculate confidence based on score separation
        # Higher separation = higher confidence
        score_std = features["composite_score"].std()
        if score_std > 0:
            # Normalize confidence to 0-100 range
            features["confidence"] = (
                (features["composite_score"] - features["composite_score"].min())
                / (
                    features["composite_score"].max()
                    - features["composite_score"].min()
                )
                * 100
            )
        else:
            # All scores equal - low confidence
            features["confidence"] = 50.0

        result = features[["driver_id", "predicted_position", "confidence"]].copy()

        self.logger.info(
            "prediction_complete",
            num_predictions=len(result),
            avg_confidence=result["confidence"].mean(),
        )

        return result

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance weights.

        Returns:
            Dictionary mapping feature names to weights
        """
        return {
            "qualifying_position": self.quali_weight,
            "driver_form_score": self.form_weight,
            "team_reliability_score": self.reliability_weight,
            "circuit_performance_score": self.circuit_weight,
        }
