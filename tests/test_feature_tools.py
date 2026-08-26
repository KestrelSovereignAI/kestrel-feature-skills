from __future__ import annotations

import pytest
from kestrel_sdk.features.contributions import PermissionLevel
from kestrel_sdk.storage.database import DatabaseError
from kestrel_sdk.tools.result import ToolResultStatus

from kestrel_feature_skills.context import render_context_clause
from kestrel_feature_skills.enablement import MAX_PRIORITY
from kestrel_feature_skills.feature import PROCEDURAL_SKILL_NODE_TYPE
from kestrel_feature_skills.format import serialize_skill_markdown
from kestrel_feature_skills.git_source import GitCheckout
from kestrel_feature_skills.models import SkillDocument

EXPECTED_TOOLS = {
    "skill_list",
    "skill_read",
    "skill_search",
    "skill_create",
    "skill_edit",
    "skill_enable",
    "skill_disable",
    "skill_delete",
    "skill_install",
}


class UnexpectedGraphError(Exception):
    """A non-database graph failure used to prove best-effort containment."""


def test_feature_exposes_exact_tool_and_permission_contract(feature):
    tools = {tool.name for tool in feature.get_tools()}
    assert tools == EXPECTED_TOOLS
    permissions = feature.get_feature_permission_defaults()
    assert permissions.feature_default is PermissionLevel.ASK
    assert set(permissions.tool_overrides) == EXPECTED_TOOLS
    assert permissions.tool_overrides["skill_list"] is PermissionLevel.ALLOW
    assert permissions.tool_overrides["skill_read"] is PermissionLevel.ALLOW
    assert permissions.tool_overrides["skill_search"] is PermissionLevel.ALLOW
    assert permissions.tool_overrides["skill_delete"] is PermissionLevel.ALWAYS_ASK
    assert permissions.tool_overrides["skill_install"] is PermissionLevel.ALWAYS_ASK
    assert all(
        permissions.tool_overrides[name] is PermissionLevel.ASK
        for name in ("skill_create", "skill_edit", "skill_enable", "skill_disable")
    )


@pytest.mark.asyncio
async def test_progressive_disclosure_list_search_then_read(feature):
    secret = "BODY-ONLY-SENTINEL-9321"
    created = await feature.skill_create(
        "progressive", "Use for progressive tests", secret
    )
    assert created.status is ToolResultStatus.OK
    listed = await feature.skill_list()
    searched = await feature.skill_search("progressive")
    assert secret not in repr(listed)
    assert secret not in repr(searched)
    read = await feature.skill_read("progressive")
    assert read.status is ToolResultStatus.OK
    assert secret in read.confirmation
    assert read.data["body"] == secret


@pytest.mark.asyncio
async def test_enable_disable_updates_bootstrap_and_cached_context(feature):
    await feature.skill_create("toggle", "Enabled description", "body")
    assert feature.context_clause_text == ""
    enabled = await feature.skill_enable("toggle", priority=4)
    assert enabled.status is ToolResultStatus.OK
    assert "Enabled description" in feature.context_clause_text
    first = feature.context_clause_text.encode()
    await feature.refresh()
    assert feature.context_clause_text.encode() == first
    disabled = await feature.skill_disable("toggle")
    assert disabled.status is ToolResultStatus.OK
    assert feature.context_clause_text == ""


@pytest.mark.asyncio
async def test_create_writes_procedural_skill_graph_node(feature):
    result = await feature.skill_create("indexed", "Graph indexed", "body")
    assert result.status is ToolResultStatus.OK
    node = feature.agent.storage.added[-1]
    assert node.node_type == PROCEDURAL_SKILL_NODE_TYPE
    assert node.node_type != "skill"
    assert node.properties["name"] == "indexed"


@pytest.mark.asyncio
async def test_refresh_indexes_valid_folder_discovered_outside_the_tools(feature):
    folder = feature.agent.procedural_skills_root / "discovered-index"
    folder.mkdir()
    (folder / "SKILL.md").write_text(
        serialize_skill_markdown(
            SkillDocument("discovered-index", "Discovered graph index", "body")
        ),
        encoding="utf-8",
    )

    await feature.refresh()

    node = feature.agent.storage.added[-1]
    assert node.node_type == PROCEDURAL_SKILL_NODE_TYPE
    assert node.properties["name"] == "discovered-index"


@pytest.mark.asyncio
async def test_refresh_removes_index_for_authoritative_folder_removed_outside_tools(
    feature,
):
    await feature.skill_create("vanished-index", "Vanishing graph index", "body")
    node_id = feature._node_id("vanished-index")
    assert node_id in feature.agent.storage.nodes
    folder = feature.agent.procedural_skills_root / "vanished-index"
    for path in folder.iterdir():
        path.unlink()
    folder.rmdir()

    await feature.refresh()

    assert node_id not in feature.agent.storage.nodes
    assert node_id in feature.agent.storage.deleted


@pytest.mark.asyncio
async def test_initialize_removes_persisted_stale_index_from_previous_process(feature):
    await feature.skill_create("restart-stale", "Stale across restart", "body")
    node_id = feature._node_id("restart-stale")
    folder = feature.agent.procedural_skills_root / "restart-stale"
    for path in folder.iterdir():
        path.unlink()
    folder.rmdir()

    from kestrel_feature_skills import ProceduralSkillsFeature

    replacement = ProceduralSkillsFeature(feature.agent)
    await replacement.initialize()
    try:
        assert node_id not in feature.agent.storage.nodes
        assert node_id in feature.agent.storage.deleted
    finally:
        await replacement.shutdown()


@pytest.mark.asyncio
async def test_create_rolls_back_when_disabled_state_cannot_be_verified(
    feature, monkeypatch
):
    async def fail(*_args, **_kwargs):
        raise DatabaseError("database offline")

    monkeypatch.setattr(feature._enablement, "set", fail)
    monkeypatch.setattr(feature._enablement, "load", fail)

    result = await feature.skill_create("db-outage", "DB outage", "body")

    assert result.status is ToolResultStatus.ERROR
    assert not (feature.agent.procedural_skills_root / "db-outage").exists()
    assert "db-outage" not in feature.snapshot.by_name()
    assert "database offline" in feature.catalog_payload()["enablement_error"]


@pytest.mark.parametrize("priority", (MAX_PRIORITY + 1, 1.5, True))
@pytest.mark.asyncio
async def test_create_validates_priority_before_publishing_folder(feature, priority):
    folder = feature.agent.procedural_skills_root / "invalid-priority"

    with pytest.raises(ValueError, match="priority"):
        await feature.create_skill(
            name="invalid-priority",
            description="Invalid priority",
            body="body",
            priority=priority,
        )

    assert not folder.exists()
    assert "invalid-priority" not in feature.snapshot.by_name()


@pytest.mark.asyncio
async def test_failed_create_cannot_reenable_from_stale_persisted_state(
    feature, monkeypatch
):
    await feature.skill_create("stale-enabled", "Old local skill", "body")
    await feature.skill_enable("stale-enabled")
    folder = feature.agent.procedural_skills_root / "stale-enabled"
    for path in folder.iterdir():
        path.unlink()
    folder.rmdir()
    await feature.refresh()

    async def fail_state(*_args, **_kwargs):
        raise DatabaseError("database offline")

    feature._states.clear()
    monkeypatch.setattr(feature._enablement, "set", fail_state)
    monkeypatch.setattr(feature._enablement, "load", fail_state)

    with pytest.raises(DatabaseError, match="database offline"):
        await feature.create_skill(
            name="stale-enabled",
            description="New untrusted procedure",
            body="body",
        )

    assert not folder.exists()
    assert "stale-enabled" not in feature.snapshot.by_name()
    assert "New untrusted procedure" not in feature.context_clause_text


@pytest.mark.asyncio
async def test_refresh_retains_last_known_enablement_during_database_outage(
    feature, monkeypatch
):
    await feature.skill_create("last-known", "Retain enabled state", "body")
    await feature.skill_enable("last-known", priority=9)

    async def fail():
        raise DatabaseError("database offline")

    monkeypatch.setattr(feature._enablement, "load", fail)
    await feature.refresh()

    record = feature.snapshot.by_name()["last-known"]
    assert record.state.enabled is True
    assert record.state.priority == 9
    assert "Retain enabled state" in feature.context_clause_text


@pytest.mark.asyncio
async def test_graph_failure_is_recoverable_and_file_remains_authoritative(
    feature, monkeypatch
):
    async def fail(_node):
        raise UnexpectedGraphError("graph unavailable")

    monkeypatch.setattr(feature.agent.storage, "add_node", fail)
    result = await feature.skill_create("file-first", "File survives", "body")
    assert result.status is ToolResultStatus.OK
    assert result.data["indexed"] is False
    assert (feature.agent.procedural_skills_root / "file-first" / "SKILL.md").is_file()


@pytest.mark.asyncio
async def test_delete_reports_partial_after_authoritative_folder_removal(
    feature, monkeypatch
):
    await feature.skill_create("partial-delete", "Partial delete", "body")
    folder = feature.agent.procedural_skills_root / "partial-delete"

    async def fail(_node_id):
        raise UnexpectedGraphError("graph unavailable")

    monkeypatch.setattr(feature.agent.storage, "delete_node", fail)
    result = await feature.skill_delete("partial-delete")

    assert result.status is ToolResultStatus.PARTIAL
    assert not folder.exists()
    assert result.data["removed_file"] is True
    assert result.data["graph_deleted"] is False
    assert "graph index cleanup failed" in result.error


@pytest.mark.asyncio
async def test_invalid_frontmatter_edit_is_rejected_before_replace(feature):
    await feature.skill_create("edit-me", "Original", "body")
    path = feature.agent.procedural_skills_root / "edit-me" / "SKILL.md"
    original = path.read_text()
    result = await feature.skill_edit("edit-me", "---\nname: edit-me\n---\nbody")
    assert result.status is ToolResultStatus.ERROR
    assert path.read_text() == original


@pytest.mark.asyncio
async def test_delete_refuses_host_shared_skill(feature, tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    shared.mkdir()
    folder = shared / "shared-only"
    folder.mkdir()
    (folder / "SKILL.md").write_text(
        serialize_skill_markdown(SkillDocument("shared-only", "Shared", "body")),
        encoding="utf-8",
    )
    monkeypatch.setenv("KESTREL_SHARED_SKILLS_DIR", str(shared))
    # Reinitialize a new feature so the source list captures the env setting.
    from kestrel_feature_skills import ProceduralSkillsFeature

    other = ProceduralSkillsFeature(feature.agent)
    await other.initialize()
    try:
        result = await other.skill_delete("shared-only")
        assert result.status is ToolResultStatus.ERROR
        assert "local override" in result.error
        assert folder.is_dir()
    finally:
        await other.shutdown()


@pytest.mark.asyncio
async def test_git_install_records_revision_and_leaves_skill_disabled(
    feature, tmp_path, monkeypatch
):
    checkout_root = tmp_path / "checkout"
    source = checkout_root / "skills" / "remote"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text(
        serialize_skill_markdown(SkillDocument("remote", "Remote skill", "body")),
        encoding="utf-8",
    )

    def fake_checkout(self, *, url, ref, skill_name, target):
        assert url == "https://example.com/repo.git"
        assert ref == "main"
        assert skill_name == "remote"
        return GitCheckout(
            root=checkout_root,
            skill_folder=source,
            revision="b" * 40,
            remote_url=url,
            ref=ref,
        )

    monkeypatch.setattr(
        "kestrel_feature_skills.git_source.GitSkillSource.checkout", fake_checkout
    )
    result = await feature.skill_install(
        "https://example.com/repo.git", "remote", "main"
    )
    assert result.status is ToolResultStatus.OK
    assert result.data["revision"] == "b" * 40
    assert result.data["enabled"] is False
    await feature.refresh()
    record = feature.snapshot.by_name()["remote"]
    assert record.provenance.revision == "b" * 40
    assert record.provenance.remote_url == "https://example.com/repo.git"
    assert record.state.enabled is False


@pytest.mark.asyncio
async def test_git_install_is_not_published_when_disabled_state_cannot_persist(
    feature, tmp_path, monkeypatch
):
    await feature.skill_create("remote-fail", "Previously local", "body")
    await feature.skill_enable("remote-fail")
    folder = feature.agent.procedural_skills_root / "remote-fail"
    for path in folder.iterdir():
        path.unlink()
    folder.rmdir()
    await feature.refresh()

    checkout_root = tmp_path / "checkout-fail"
    source = checkout_root / "skills" / "remote-fail"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text(
        serialize_skill_markdown(
            SkillDocument("remote-fail", "Untrusted remote description", "body")
        ),
        encoding="utf-8",
    )

    def fake_checkout(self, *, url, ref, skill_name, target):
        return GitCheckout(
            root=checkout_root,
            skill_folder=source,
            revision="c" * 40,
            remote_url=url,
            ref=ref,
        )

    async def fail_state(*_args, **_kwargs):
        raise DatabaseError("database offline")

    monkeypatch.setattr(
        "kestrel_feature_skills.git_source.GitSkillSource.checkout", fake_checkout
    )
    monkeypatch.setattr(feature._enablement, "set", fail_state)

    result = await feature.skill_install(
        "https://example.com/repo.git", "remote-fail", "main"
    )

    assert result.status is ToolResultStatus.ERROR
    assert not folder.exists()
    assert "remote-fail" not in feature.snapshot.by_name()
    assert "Untrusted remote description" not in feature.context_clause_text


@pytest.mark.asyncio
async def test_python_editor_has_no_execution_effect(feature, tmp_path):
    await feature.skill_create("scripted", "Script editor", "body")
    marker = tmp_path / "marker"
    source = f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n"
    result = await feature.skill_edit("scripted", source, "scripts/tool.py")
    assert result.status is ToolResultStatus.OK
    assert not marker.exists()
    tree = feature.tree(name="scripted")
    python = next(item for item in tree if item["path"] == "scripts/tool.py")
    assert python["execution_risk"] is True


def test_context_renderer_is_not_registered_through_a_fake_legacy_hook(feature):
    assert feature.context_clause_text == render_context_clause(feature.snapshot).text
    assert not hasattr(feature, "get_context_clause_registrations")
    assert feature.get_hooks() == []
