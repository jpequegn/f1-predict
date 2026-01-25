"""Tests for F1 Causal DAG implementation."""

import pytest

from f1_predict.causal.dag import (
    CausalEdge,
    CausalNode,
    EdgeType,
    F1CausalDAG,
    NodeType,
)


class TestNodeType:
    """Test NodeType enum."""

    def test_node_types_exist(self):
        """Test all expected node types are defined."""
        assert NodeType.TREATMENT is not None
        assert NodeType.OUTCOME is not None
        assert NodeType.CONFOUNDER is not None
        assert NodeType.MEDIATOR is not None
        assert NodeType.INSTRUMENT is not None
        assert NodeType.COLLIDER is not None
        assert NodeType.OBSERVED is not None
        assert NodeType.UNOBSERVED is not None


class TestEdgeType:
    """Test EdgeType enum."""

    def test_edge_types_exist(self):
        """Test all expected edge types are defined."""
        assert EdgeType.CAUSAL is not None
        assert EdgeType.ASSOCIATION is not None
        assert EdgeType.BIDIRECTIONAL is not None


class TestCausalNode:
    """Test CausalNode dataclass."""

    def test_create_node(self):
        """Test creating a causal node."""
        node = CausalNode(
            name="test_node",
            node_type=NodeType.TREATMENT,
            description="A test node",
        )
        assert node.name == "test_node"
        assert node.node_type == NodeType.TREATMENT
        assert node.description == "A test node"
        assert node.observed is True  # default
        assert node.metadata == {}  # default

    def test_node_with_metadata(self):
        """Test creating node with metadata."""
        node = CausalNode(
            name="weather",
            node_type=NodeType.CONFOUNDER,
            metadata={"source": "openweather"},
        )
        assert node.metadata["source"] == "openweather"

    def test_unobserved_node(self):
        """Test creating unobserved latent variable."""
        node = CausalNode(
            name="driver_talent",
            node_type=NodeType.CONFOUNDER,
            observed=False,
        )
        assert node.observed is False

    def test_node_hash(self):
        """Test node hashing."""
        node1 = CausalNode("test", NodeType.TREATMENT)
        node2 = CausalNode("test", NodeType.OUTCOME)
        assert hash(node1) == hash(node2)  # Same name = same hash

    def test_node_equality(self):
        """Test node equality."""
        node1 = CausalNode("test", NodeType.TREATMENT)
        node2 = CausalNode("test", NodeType.OUTCOME)
        node3 = CausalNode("other", NodeType.TREATMENT)
        assert node1 == node2  # Same name
        assert node1 != node3  # Different name


class TestCausalEdge:
    """Test CausalEdge dataclass."""

    def test_create_edge(self):
        """Test creating a causal edge."""
        edge = CausalEdge(
            source="qualifying",
            target="race_position",
            edge_type=EdgeType.CAUSAL,
        )
        assert edge.source == "qualifying"
        assert edge.target == "race_position"
        assert edge.edge_type == EdgeType.CAUSAL

    def test_edge_with_strength(self):
        """Test creating edge with strength."""
        edge = CausalEdge(
            source="pit_strategy",
            target="race_position",
            edge_type=EdgeType.CAUSAL,
            strength=0.8,
        )
        assert edge.strength == 0.8

    def test_edge_hash(self):
        """Test edge hashing."""
        edge1 = CausalEdge("A", "B", EdgeType.CAUSAL)
        edge2 = CausalEdge("A", "B", EdgeType.ASSOCIATION)
        assert hash(edge1) == hash(edge2)  # Same source/target = same hash

    def test_edge_equality(self):
        """Test edge equality."""
        edge1 = CausalEdge("A", "B", EdgeType.CAUSAL)
        edge2 = CausalEdge("A", "B", EdgeType.ASSOCIATION)
        edge3 = CausalEdge("A", "C", EdgeType.CAUSAL)
        assert edge1 == edge2  # Same source/target
        assert edge1 != edge3  # Different target


class TestF1CausalDAG:
    """Test F1CausalDAG class."""

    @pytest.fixture
    def empty_dag(self):
        """Create empty DAG."""
        return F1CausalDAG()

    @pytest.fixture
    def simple_dag(self):
        """Create simple DAG with a few nodes."""
        dag = F1CausalDAG()
        dag.add_node(CausalNode("A", NodeType.TREATMENT))
        dag.add_node(CausalNode("B", NodeType.MEDIATOR))
        dag.add_node(CausalNode("C", NodeType.OUTCOME))
        dag.add_edge(CausalEdge("A", "B", EdgeType.CAUSAL))
        dag.add_edge(CausalEdge("B", "C", EdgeType.CAUSAL))
        return dag

    @pytest.fixture
    def confounded_dag(self):
        """Create DAG with confounding."""
        dag = F1CausalDAG()
        dag.add_node(CausalNode("treatment", NodeType.TREATMENT))
        dag.add_node(CausalNode("outcome", NodeType.OUTCOME))
        dag.add_node(CausalNode("confounder", NodeType.CONFOUNDER))

        dag.add_edge(CausalEdge("confounder", "treatment", EdgeType.CAUSAL))
        dag.add_edge(CausalEdge("confounder", "outcome", EdgeType.CAUSAL))
        dag.add_edge(CausalEdge("treatment", "outcome", EdgeType.CAUSAL))
        return dag

    def test_add_node(self, empty_dag):
        """Test adding a node."""
        node = CausalNode("test", NodeType.TREATMENT)
        empty_dag.add_node(node)
        assert "test" in empty_dag.nodes

    def test_add_duplicate_node_raises(self, empty_dag):
        """Test adding duplicate node raises error."""
        node = CausalNode("test", NodeType.TREATMENT)
        empty_dag.add_node(node)
        with pytest.raises(ValueError, match="already exists"):
            empty_dag.add_node(CausalNode("test", NodeType.OUTCOME))

    def test_add_edge(self, empty_dag):
        """Test adding an edge."""
        empty_dag.add_node(CausalNode("A", NodeType.TREATMENT))
        empty_dag.add_node(CausalNode("B", NodeType.OUTCOME))
        empty_dag.add_edge(CausalEdge("A", "B", EdgeType.CAUSAL))

        assert ("A", "B") in empty_dag.edges

    def test_add_edge_missing_source_raises(self, empty_dag):
        """Test adding edge with missing source raises error."""
        empty_dag.add_node(CausalNode("B", NodeType.OUTCOME))
        with pytest.raises(ValueError, match="Source node"):
            empty_dag.add_edge(CausalEdge("A", "B", EdgeType.CAUSAL))

    def test_add_edge_missing_target_raises(self, empty_dag):
        """Test adding edge with missing target raises error."""
        empty_dag.add_node(CausalNode("A", NodeType.TREATMENT))
        with pytest.raises(ValueError, match="Target node"):
            empty_dag.add_edge(CausalEdge("A", "B", EdgeType.CAUSAL))

    def test_add_edge_cycle_raises(self, simple_dag):
        """Test adding edge that creates cycle raises error."""
        with pytest.raises(ValueError, match="cycle"):
            simple_dag.add_edge(CausalEdge("C", "A", EdgeType.CAUSAL))

    def test_get_parents(self, simple_dag):
        """Test getting parent nodes."""
        parents = simple_dag.get_parents("B")
        assert "A" in parents

        parents = simple_dag.get_parents("C")
        assert "B" in parents

    def test_get_parents_root_node(self, simple_dag):
        """Test root node has no parents."""
        parents = simple_dag.get_parents("A")
        assert len(parents) == 0

    def test_get_children(self, simple_dag):
        """Test getting child nodes."""
        children = simple_dag.get_children("A")
        assert "B" in children

        children = simple_dag.get_children("B")
        assert "C" in children

    def test_get_children_leaf_node(self, simple_dag):
        """Test leaf node has no children."""
        children = simple_dag.get_children("C")
        assert len(children) == 0

    def test_get_ancestors(self, simple_dag):
        """Test getting all ancestors."""
        ancestors = simple_dag.get_ancestors("C")
        assert "A" in ancestors
        assert "B" in ancestors

    def test_get_descendants(self, simple_dag):
        """Test getting all descendants."""
        descendants = simple_dag.get_descendants("A")
        assert "B" in descendants
        assert "C" in descendants

    def test_identify_confounders(self, confounded_dag):
        """Test identifying confounders."""
        confounders = confounded_dag.identify_confounders("treatment", "outcome")
        assert "confounder" in confounders

    def test_identify_mediators(self, simple_dag):
        """Test identifying mediators."""
        mediators = simple_dag.identify_mediators("A", "C")
        assert "B" in mediators

    def test_get_adjustment_set(self, confounded_dag):
        """Test getting adjustment set for causal effect."""
        adjustment = confounded_dag.get_adjustment_set("treatment", "outcome")
        assert "confounder" in adjustment


class TestF1CausalDAGDefault:
    """Test default F1 DAG construction."""

    @pytest.fixture
    def f1_dag(self):
        """Create default F1 DAG."""
        dag = F1CausalDAG()
        dag.build_default_dag()
        return dag

    def test_default_dag_has_nodes(self, f1_dag):
        """Test default DAG has expected nodes."""
        assert len(f1_dag.nodes) > 0
        # Check some expected F1-specific nodes
        expected_nodes = [
            "qualifying_position",
            "race_position",
            "driver_skill",
            "team_performance",
        ]
        for node in expected_nodes:
            assert node in f1_dag.nodes, f"Missing node: {node}"

    def test_default_dag_has_edges(self, f1_dag):
        """Test default DAG has edges."""
        assert len(f1_dag.edges) > 0

    def test_default_dag_qualifying_affects_race(self, f1_dag):
        """Test qualifying position affects race position."""
        children = f1_dag.get_descendants("qualifying_position")
        assert "race_position" in children

    def test_default_dag_driver_skill_is_confounder(self, f1_dag):
        """Test driver skill is a confounder for treatment effects."""
        node = f1_dag.nodes.get("driver_skill")
        assert node is not None
        assert node.node_type == NodeType.CONFOUNDER

    def test_adjustment_for_pit_strategy(self, f1_dag):
        """Test adjustment set for pit strategy effect."""
        adjustment = f1_dag.get_adjustment_set("pit_stop_strategy", "race_position")
        # Should include some variables to adjust for
        assert isinstance(adjustment, list)

    def test_summary(self, f1_dag):
        """Test DAG summary generation."""
        summary = f1_dag.summary()
        assert "num_nodes" in summary
        assert "num_edges" in summary
        assert summary["num_nodes"] > 0
        assert summary["num_edges"] > 0

    def test_to_dict(self, f1_dag):
        """Test DAG serialization."""
        dag_dict = f1_dag.to_dict()
        assert "nodes" in dag_dict
        assert "edges" in dag_dict
        assert len(dag_dict["nodes"]) == len(f1_dag.nodes)
        assert len(dag_dict["edges"]) == len(f1_dag.edges)
