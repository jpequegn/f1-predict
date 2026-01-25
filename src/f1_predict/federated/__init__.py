"""Federated Learning & Privacy-Preserving ML for F1 Predictions.

This module provides:
- Federated averaging (FedAvg) algorithm
- Differential privacy mechanisms
- Secure aggregation protocols
- Privacy budget tracking
- Client-server architecture for distributed training
"""

from f1_predict.federated.aggregation import (
    AggregationResult,
    ByzantineRobustAggregator,
    FederatedAggregator,
    SecureAggregator,
)
from f1_predict.federated.client import (
    ClientConfig,
    ClientUpdate,
    FederatedClient,
)
from f1_predict.federated.compression import (
    CompressionMethod,
    ModelCompressor,
)
from f1_predict.federated.privacy import (
    DifferentialPrivacy,
    NoiseType,
    PrivacyAccountant,
    PrivacyBudget,
)
from f1_predict.federated.server import (
    FederatedModel,
    FederatedServer,
    ServerConfig,
    TrainingRound,
)

__all__ = [
    # Aggregation
    "FederatedAggregator",
    "SecureAggregator",
    "ByzantineRobustAggregator",
    "AggregationResult",
    # Client
    "FederatedClient",
    "ClientConfig",
    "ClientUpdate",
    # Privacy
    "DifferentialPrivacy",
    "PrivacyBudget",
    "PrivacyAccountant",
    "NoiseType",
    # Server
    "FederatedServer",
    "ServerConfig",
    "TrainingRound",
    "FederatedModel",
    # Compression
    "ModelCompressor",
    "CompressionMethod",
]
