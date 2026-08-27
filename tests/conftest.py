from __future__ import annotations

from types import SimpleNamespace

import pytest
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_graph_store import GraphNode, NodeSwapResult

from kestrel_feature_skills import ProceduralSkillsFeature


class GraphStorage:
    def __init__(self) -> None:
        self.nodes = {}
        self.added = []
        self.deleted = []

    async def add_node(self, node) -> None:
        self.nodes[node.node_id] = node
        self.added.append(node)

    async def compare_and_swap_node(
        self, node_id, expected, new_node, allowed_node_types=None
    ):
        existing = self.nodes.get(node_id)
        if existing is None:
            if expected is not None:
                return NodeSwapResult.NOT_FOUND
            if (
                allowed_node_types is not None
                and new_node.node_type not in allowed_node_types
            ):
                return NodeSwapResult.TYPE_NOT_ALLOWED
            self.nodes[node_id] = new_node
            self.added.append(new_node)
            return NodeSwapResult.SWAPPED
        if (
            allowed_node_types is not None
            and existing.node_type not in allowed_node_types
        ):
            return NodeSwapResult.TYPE_NOT_ALLOWED
        if expected is None or existing.properties != expected:
            return NodeSwapResult.PREDICATE_FAILED
        persisted = GraphNode(
            node_id=node_id,
            node_type=existing.node_type,
            label=existing.label,
            properties=dict(new_node.properties),
        )
        self.nodes[node_id] = persisted
        self.added.append(persisted)
        return NodeSwapResult.SWAPPED

    async def get_node(self, node_id: str):
        return self.nodes.get(node_id)

    async def get_nodes_by_type(self, node_type: str):
        return [node for node in self.nodes.values() if node.node_type == node_type]

    async def delete_node(self, node_id: str) -> None:
        self.nodes.pop(node_id, None)
        self.deleted.append(node_id)


@pytest.fixture
async def feature(tmp_path, monkeypatch):
    monkeypatch.delenv("KESTREL_HOME", raising=False)
    monkeypatch.delenv("KESTREL_SHARED_SKILLS_DIR", raising=False)
    db = await AsyncDatabase.sqlite(str(tmp_path / "agent.db"))
    graph = GraphStorage()
    agent = SimpleNamespace(
        did="did:test:skills",
        agent_id="did:test:skills",
        procedural_skills_root=tmp_path / "skills",
        _raw_storage=SimpleNamespace(db=db),
        storage=graph,
    )
    value = ProceduralSkillsFeature(agent)
    await value.initialize()
    yield value
    await value.shutdown()
    await db.close()
