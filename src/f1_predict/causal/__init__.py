"""Causal inference module for F1 predictions.

This module provides causal inference techniques to understand the true impact
of various factors on race outcomes, going beyond correlation to identify causation.

Components:
- DAG: Directed Acyclic Graph representation of F1 causal relationships
- TreatmentEffects: Estimation of Average Treatment Effect (ATE) and CATE
- Counterfactuals: "What-if" scenario generation and analysis
- CausalDiscovery: Automated causal structure learning
"""

from f1_predict.causal.counterfactuals import (
    CounterfactualEngine,
    CounterfactualResult,
    Intervention,
)
from f1_predict.causal.dag import CausalEdge, CausalNode, F1CausalDAG
from f1_predict.causal.treatment_effects import (
    ATEResult,
    CATEResult,
    TreatmentEffectEstimator,
)

__all__ = [
    "F1CausalDAG",
    "CausalNode",
    "CausalEdge",
    "TreatmentEffectEstimator",
    "ATEResult",
    "CATEResult",
    "CounterfactualEngine",
    "CounterfactualResult",
    "Intervention",
]
