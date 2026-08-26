from __future__ import annotations

import os
import uuid
from types import SimpleNamespace

import pytest
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore

from kestrel_feature_skills import ProceduralSkillsFeature
from kestrel_feature_skills.enablement import SkillEnablementStore
from kestrel_feature_skills.feature import PROCEDURAL_SKILL_NODE_TYPE
from kestrel_feature_skills.models import SkillState

POSTGRES_URL = os.environ.get("KESTREL_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    not POSTGRES_URL,
    reason="KESTREL_TEST_POSTGRES_URL is not configured",
)


@pytest.mark.asyncio
async def test_postgres_enablement_matches_sqlite_without_feature_tables():
    """Exercise the namespaced core-table contract on the second backend."""

    db = await AsyncDatabase.postgres(str(POSTGRES_URL))
    agent_id = f"did:test:skills-postgres:{uuid.uuid4()}"
    try:
        first = SkillEnablementStore(db, agent_id)
        second = SkillEnablementStore(db, f"{agent_id}:other")
        await first.set("same", enabled=True, priority=1)
        await second.set("same", enabled=False, priority=9)
        await first.set("same", enabled=False, priority=4)

        assert await first.load() == {"same": SkillState(False, 4)}
        assert await second.load() == {"same": SkillState(False, 9)}
        tables = await db.fetchall(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
        )
        assert not any(name.startswith("skill_") for (name,) in tables)

        await first.delete("same")
        assert await first.load() == {}
        assert await second.load() == {"same": SkillState(False, 9)}
    finally:
        await db.execute(
            "DELETE FROM bootstrap_config WHERE agent_id IN (?, ?)",
            (agent_id, f"{agent_id}:other"),
        )
        await db.close()


@pytest.mark.asyncio
async def test_postgres_feature_writes_and_removes_procedural_skill_node(tmp_path):
    db = await AsyncDatabase.postgres(str(POSTGRES_URL))
    agent_id = f"did:test:skills-postgres-graph:{uuid.uuid4()}"
    graph = AsyncGraphStore(db, agent_id=agent_id)
    agent = SimpleNamespace(
        did=agent_id,
        agent_id=agent_id,
        procedural_skills_root=tmp_path / "skills",
        _raw_storage=SimpleNamespace(db=db),
        storage=graph,
    )
    feature = ProceduralSkillsFeature(agent)
    try:
        await feature.initialize()
        created = await feature.create_skill(
            name="postgres-graph",
            description="PostgreSQL graph parity",
            body="Procedure.",
        )
        assert created["indexed"] is True
        nodes = await graph.get_nodes_by_type(PROCEDURAL_SKILL_NODE_TYPE)
        node = next(
            item for item in nodes if item.properties["name"] == "postgres-graph"
        )
        assert node.node_type == PROCEDURAL_SKILL_NODE_TYPE

        deleted = await feature.delete_skill(name="postgres-graph")
        assert deleted["graph_deleted"] is True
        assert await graph.get_node(node.node_id) is None
    finally:
        await feature.shutdown()
        await db.execute(
            "DELETE FROM graph_node_owners WHERE agent_id = ?",
            (agent_id,),
        )
        await db.execute(
            "DELETE FROM bootstrap_config WHERE agent_id = ?",
            (agent_id,),
        )
        await db.close()
