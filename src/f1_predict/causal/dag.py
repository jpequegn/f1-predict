"""Directed Acyclic Graph (DAG) for F1 causal relationships.

This module provides tools for building and analyzing causal DAGs
specific to F1 race prediction, encoding domain knowledge about
causal relationships between variables.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)


class NodeType(Enum):
    """Types of nodes in the causal DAG."""

    TREATMENT = "treatment"  # Variable we can intervene on
    OUTCOME = "outcome"  # Variable we want to predict/explain
    CONFOUNDER = "confounder"  # Variable affecting both treatment and outcome
    MEDIATOR = "mediator"  # Variable on causal path between treatment and outcome
    INSTRUMENT = "instrument"  # Variable affecting treatment but not outcome directly
    COLLIDER = "collider"  # Variable affected by multiple causes
    OBSERVED = "observed"  # General observed variable
    UNOBSERVED = "unobserved"  # Latent/unobserved variable


class EdgeType(Enum):
    """Types of edges in the causal DAG."""

    CAUSAL = "causal"  # Direct causal relationship
    ASSOCIATION = "association"  # Non-causal association
    BIDIRECTIONAL = "bidirectional"  # Confounded relationship


@dataclass
class CausalNode:
    """A node in the causal DAG representing a variable."""

    name: str
    node_type: NodeType
    description: str = ""
    domain: Optional[list[Any]] = None  # Possible values for discrete variables
    continuous: bool = True
    observed: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CausalNode):
            return False
        return self.name == other.name


@dataclass
class CausalEdge:
    """An edge in the causal DAG representing a causal relationship."""

    source: str
    target: str
    edge_type: EdgeType = EdgeType.CAUSAL
    strength: Optional[float] = None  # Estimated causal strength (if known)
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash((self.source, self.target))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CausalEdge):
            return False
        return self.source == other.source and self.target == other.target


class F1CausalDAG:
    """Directed Acyclic Graph for F1 causal relationships.

    This class encodes domain knowledge about causal relationships
    in F1 racing, enabling causal inference and counterfactual analysis.

    Example:
        >>> dag = F1CausalDAG()
        >>> dag.build_default_dag()
        >>> parents = dag.get_parents("race_position")
        >>> confounders = dag.identify_confounders("qualifying_position", "race_position")
    """

    def __init__(self) -> None:
        """Initialize empty causal DAG."""
        self.nodes: dict[str, CausalNode] = {}
        self.edges: dict[tuple[str, str], CausalEdge] = {}
        self._adjacency: dict[str, set[str]] = {}  # parent -> children
        self._reverse_adjacency: dict[str, set[str]] = {}  # child -> parents
        self.logger = logger.bind(component="F1CausalDAG")

    def add_node(self, node: CausalNode) -> None:
        """Add a node to the DAG.

        Args:
            node: CausalNode to add

        Raises:
            ValueError: If node with same name already exists
        """
        if node.name in self.nodes:
            msg = f"Node '{node.name}' already exists in DAG"
            raise ValueError(msg)

        self.nodes[node.name] = node
        self._adjacency[node.name] = set()
        self._reverse_adjacency[node.name] = set()
        self.logger.debug("node_added", name=node.name, type=node.node_type.value)

    def add_edge(self, edge: CausalEdge) -> None:
        """Add an edge to the DAG.

        Args:
            edge: CausalEdge to add

        Raises:
            ValueError: If source or target node doesn't exist
            ValueError: If edge would create a cycle
        """
        if edge.source not in self.nodes:
            msg = f"Source node '{edge.source}' not in DAG"
            raise ValueError(msg)
        if edge.target not in self.nodes:
            msg = f"Target node '{edge.target}' not in DAG"
            raise ValueError(msg)

        # Check for cycles
        if self._would_create_cycle(edge.source, edge.target):
            msg = f"Edge {edge.source} -> {edge.target} would create a cycle"
            raise ValueError(msg)

        key = (edge.source, edge.target)
        self.edges[key] = edge
        self._adjacency[edge.source].add(edge.target)
        self._reverse_adjacency[edge.target].add(edge.source)
        self.logger.debug(
            "edge_added",
            source=edge.source,
            target=edge.target,
            type=edge.edge_type.value,
        )

    def _would_create_cycle(self, source: str, target: str) -> bool:
        """Check if adding edge source -> target would create a cycle."""
        # If target can reach source, adding source -> target creates a cycle
        visited = set()
        stack = [target]

        while stack:
            current = stack.pop()
            if current == source:
                return True
            if current in visited:
                continue
            visited.add(current)
            stack.extend(self._adjacency.get(current, []))

        return False

    def get_parents(self, node_name: str) -> list[str]:
        """Get parent nodes (direct causes) of a node.

        Args:
            node_name: Name of the node

        Returns:
            List of parent node names
        """
        return list(self._reverse_adjacency.get(node_name, []))

    def get_children(self, node_name: str) -> list[str]:
        """Get children nodes (direct effects) of a node.

        Args:
            node_name: Name of the node

        Returns:
            List of children node names
        """
        return list(self._adjacency.get(node_name, []))

    def get_ancestors(self, node_name: str) -> set[str]:
        """Get all ancestor nodes (all causes) of a node.

        Args:
            node_name: Name of the node

        Returns:
            Set of ancestor node names
        """
        ancestors = set()
        stack = list(self._reverse_adjacency.get(node_name, []))

        while stack:
            current = stack.pop()
            if current in ancestors:
                continue
            ancestors.add(current)
            stack.extend(self._reverse_adjacency.get(current, []))

        return ancestors

    def get_descendants(self, node_name: str) -> set[str]:
        """Get all descendant nodes (all effects) of a node.

        Args:
            node_name: Name of the node

        Returns:
            Set of descendant node names
        """
        descendants = set()
        stack = list(self._adjacency.get(node_name, []))

        while stack:
            current = stack.pop()
            if current in descendants:
                continue
            descendants.add(current)
            stack.extend(self._adjacency.get(current, []))

        return descendants

    def identify_confounders(self, treatment: str, outcome: str) -> list[str]:
        """Identify confounding variables between treatment and outcome.

        Confounders are variables that:
        1. Cause the treatment
        2. Cause the outcome
        3. Are not on the causal path from treatment to outcome

        Args:
            treatment: Treatment variable name
            outcome: Outcome variable name

        Returns:
            List of confounder variable names
        """
        treatment_ancestors = self.get_ancestors(treatment)
        outcome_ancestors = self.get_ancestors(outcome)
        treatment_descendants = self.get_descendants(treatment)

        # Confounders are common ancestors that are not descendants of treatment
        common_ancestors = treatment_ancestors & outcome_ancestors
        confounders = [
            node for node in common_ancestors if node not in treatment_descendants
        ]

        self.logger.info(
            "confounders_identified",
            treatment=treatment,
            outcome=outcome,
            confounders=confounders,
        )
        return confounders

    def identify_mediators(self, treatment: str, outcome: str) -> list[str]:
        """Identify mediating variables between treatment and outcome.

        Mediators are variables on the causal path from treatment to outcome.

        Args:
            treatment: Treatment variable name
            outcome: Outcome variable name

        Returns:
            List of mediator variable names
        """
        treatment_descendants = self.get_descendants(treatment)
        outcome_ancestors = self.get_ancestors(outcome)

        # Mediators are descendants of treatment that are ancestors of outcome
        mediators = list(treatment_descendants & outcome_ancestors)

        self.logger.info(
            "mediators_identified",
            treatment=treatment,
            outcome=outcome,
            mediators=mediators,
        )
        return mediators

    def get_adjustment_set(self, treatment: str, outcome: str) -> list[str]:
        """Get the minimal adjustment set for estimating causal effect.

        Uses the backdoor criterion to identify variables that need
        to be controlled for to get unbiased causal effect estimates.

        Args:
            treatment: Treatment variable name
            outcome: Outcome variable name

        Returns:
            List of variables to adjust for
        """
        # Start with confounders
        adjustment_set = set(self.identify_confounders(treatment, outcome))

        # Add parents of treatment that are observed
        for parent in self.get_parents(treatment):
            if self.nodes[parent].observed:
                adjustment_set.add(parent)

        # Remove mediators (we don't want to block the causal path)
        mediators = set(self.identify_mediators(treatment, outcome))
        adjustment_set -= mediators

        # Remove treatment and outcome
        adjustment_set.discard(treatment)
        adjustment_set.discard(outcome)

        self.logger.info(
            "adjustment_set_computed",
            treatment=treatment,
            outcome=outcome,
            adjustment_set=list(adjustment_set),
        )
        return list(adjustment_set)

    def build_default_dag(self) -> None:
        """Build the default F1 causal DAG with domain knowledge.

        This encodes expert knowledge about causal relationships
        in Formula 1 racing.
        """
        self.logger.info("building_default_f1_dag")

        # Add nodes for key variables
        # Driver/Team attributes
        self.add_node(
            CausalNode(
                name="driver_skill",
                node_type=NodeType.CONFOUNDER,
                description="Inherent driver skill and experience",
                observed=False,  # Latent variable
            )
        )
        self.add_node(
            CausalNode(
                name="team_performance",
                node_type=NodeType.CONFOUNDER,
                description="Team's overall performance level (car + operations)",
                observed=False,
            )
        )
        self.add_node(
            CausalNode(
                name="car_performance",
                node_type=NodeType.CONFOUNDER,
                description="Car's pace and reliability",
                continuous=True,
            )
        )

        # Race weekend variables
        self.add_node(
            CausalNode(
                name="qualifying_position",
                node_type=NodeType.TREATMENT,
                description="Grid position after qualifying",
                continuous=False,
                domain=list(range(1, 21)),
            )
        )
        self.add_node(
            CausalNode(
                name="practice_performance",
                node_type=NodeType.OBSERVED,
                description="Performance in practice sessions",
            )
        )

        # Strategy variables
        self.add_node(
            CausalNode(
                name="pit_stop_strategy",
                node_type=NodeType.TREATMENT,
                description="Number and timing of pit stops",
            )
        )
        self.add_node(
            CausalNode(
                name="tire_compound_choice",
                node_type=NodeType.TREATMENT,
                description="Selection of tire compounds",
            )
        )
        self.add_node(
            CausalNode(
                name="pit_stop_execution",
                node_type=NodeType.MEDIATOR,
                description="Quality of pit stop execution (time)",
            )
        )

        # Race conditions
        self.add_node(
            CausalNode(
                name="weather_conditions",
                node_type=NodeType.CONFOUNDER,
                description="Weather during the race",
            )
        )
        self.add_node(
            CausalNode(
                name="safety_car",
                node_type=NodeType.CONFOUNDER,
                description="Safety car deployment",
            )
        )
        self.add_node(
            CausalNode(
                name="track_temperature",
                node_type=NodeType.OBSERVED,
                description="Track surface temperature",
            )
        )

        # Circuit characteristics
        self.add_node(
            CausalNode(
                name="circuit_type",
                node_type=NodeType.CONFOUNDER,
                description="Type of circuit (street, high-speed, technical)",
            )
        )
        self.add_node(
            CausalNode(
                name="overtaking_difficulty",
                node_type=NodeType.CONFOUNDER,
                description="How difficult it is to overtake at this circuit",
            )
        )

        # In-race variables
        self.add_node(
            CausalNode(
                name="first_lap_positions",
                node_type=NodeType.MEDIATOR,
                description="Positions gained/lost on first lap",
            )
        )
        self.add_node(
            CausalNode(
                name="tire_degradation",
                node_type=NodeType.MEDIATOR,
                description="Rate of tire wear during race",
            )
        )
        self.add_node(
            CausalNode(
                name="incidents",
                node_type=NodeType.OBSERVED,
                description="Collisions, mechanical failures, penalties",
            )
        )

        # Outcome
        self.add_node(
            CausalNode(
                name="race_position",
                node_type=NodeType.OUTCOME,
                description="Final race finishing position",
                continuous=False,
                domain=list(range(1, 21)),
            )
        )

        # Add causal edges
        # Driver/Team -> Performance
        self.add_edge(CausalEdge("driver_skill", "qualifying_position"))
        self.add_edge(CausalEdge("driver_skill", "practice_performance"))
        self.add_edge(CausalEdge("driver_skill", "first_lap_positions"))
        self.add_edge(CausalEdge("driver_skill", "race_position"))

        self.add_edge(CausalEdge("team_performance", "car_performance"))
        self.add_edge(CausalEdge("team_performance", "pit_stop_strategy"))
        self.add_edge(CausalEdge("team_performance", "pit_stop_execution"))

        self.add_edge(CausalEdge("car_performance", "qualifying_position"))
        self.add_edge(CausalEdge("car_performance", "practice_performance"))
        self.add_edge(CausalEdge("car_performance", "tire_degradation"))
        self.add_edge(CausalEdge("car_performance", "race_position"))

        # Qualifying -> Race
        self.add_edge(CausalEdge("qualifying_position", "first_lap_positions"))
        self.add_edge(CausalEdge("qualifying_position", "race_position"))

        # Strategy
        self.add_edge(CausalEdge("pit_stop_strategy", "pit_stop_execution"))
        self.add_edge(CausalEdge("pit_stop_strategy", "tire_degradation"))
        self.add_edge(CausalEdge("pit_stop_strategy", "race_position"))

        self.add_edge(CausalEdge("tire_compound_choice", "tire_degradation"))
        self.add_edge(CausalEdge("tire_compound_choice", "race_position"))

        self.add_edge(CausalEdge("pit_stop_execution", "race_position"))

        # Conditions
        self.add_edge(CausalEdge("weather_conditions", "qualifying_position"))
        self.add_edge(CausalEdge("weather_conditions", "tire_degradation"))
        self.add_edge(CausalEdge("weather_conditions", "incidents"))
        self.add_edge(CausalEdge("weather_conditions", "race_position"))

        self.add_edge(CausalEdge("safety_car", "pit_stop_strategy"))
        self.add_edge(CausalEdge("safety_car", "race_position"))

        self.add_edge(CausalEdge("track_temperature", "tire_degradation"))

        # Circuit
        self.add_edge(CausalEdge("circuit_type", "overtaking_difficulty"))
        self.add_edge(CausalEdge("circuit_type", "tire_degradation"))
        self.add_edge(CausalEdge("circuit_type", "pit_stop_strategy"))

        self.add_edge(CausalEdge("overtaking_difficulty", "qualifying_position"))
        self.add_edge(CausalEdge("overtaking_difficulty", "race_position"))

        # In-race
        self.add_edge(CausalEdge("first_lap_positions", "race_position"))
        self.add_edge(CausalEdge("tire_degradation", "race_position"))
        self.add_edge(CausalEdge("incidents", "race_position"))

        self.logger.info(
            "default_dag_built",
            num_nodes=len(self.nodes),
            num_edges=len(self.edges),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert DAG to dictionary representation.

        Returns:
            Dictionary containing nodes and edges
        """
        return {
            "nodes": [
                {
                    "name": node.name,
                    "type": node.node_type.value,
                    "description": node.description,
                    "continuous": node.continuous,
                    "observed": node.observed,
                }
                for node in self.nodes.values()
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "type": edge.edge_type.value,
                    "strength": edge.strength,
                    "description": edge.description,
                }
                for edge in self.edges.values()
            ],
        }

    def summary(self) -> dict[str, Any]:
        """Get summary statistics of the DAG.

        Returns:
            Dictionary with summary statistics
        """
        node_types = {}
        for node in self.nodes.values():
            t = node.node_type.value
            node_types[t] = node_types.get(t, 0) + 1

        return {
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "node_types": node_types,
            "treatments": [
                n.name for n in self.nodes.values() if n.node_type == NodeType.TREATMENT
            ],
            "outcomes": [
                n.name for n in self.nodes.values() if n.node_type == NodeType.OUTCOME
            ],
            "confounders": [
                n.name
                for n in self.nodes.values()
                if n.node_type == NodeType.CONFOUNDER
            ],
        }
