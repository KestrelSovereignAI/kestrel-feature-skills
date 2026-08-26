from __future__ import annotations

from types import SimpleNamespace

import pytest
from kestrel_sovereign.storage.async_database import AsyncDatabase

from kestrel_feature_skills import ProceduralSkillsFeature


class GraphStorage:
    def __init__(self) -> None:
        self.nodes = {}
        self.added = []
        self.deleted = []

    async def add_node(self, node) -> None:
        self.nodes[node.node_id] = node
        self.added.append(node)

    async def get_node(self, node_id: str):
        return self.nodes.get(node_id)

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
