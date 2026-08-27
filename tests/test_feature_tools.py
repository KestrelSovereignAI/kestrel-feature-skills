from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest
from kestrel_sdk.features.contributions import PermissionLevel
from kestrel_sdk.storage.database import DatabaseError
from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.privacy import PrivacyConfig

from kestrel_feature_skills import ProceduralSkillsFeature
from kestrel_feature_skills.context import render_context_clause
from kestrel_feature_skills.enablement import MAX_PRIORITY
from kestrel_feature_skills.errors import GitSourceError
from kestrel_feature_skills.feature import PROCEDURAL_SKILL_NODE_TYPE
from kestrel_feature_skills.format import serialize_skill_markdown
from kestrel_feature_skills.git_source import GitCheckout
from kestrel_feature_skills.models import SkillDocument, SkillState

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


@pytest.mark.asyncio
@pytest.mark.parametrize("storage", ("none", "temp", "deidentified"))
async def test_volatile_privacy_initialization_never_opens_persistent_skills(
    tmp_path, monkeypatch, storage
):
    import kestrel_feature_skills.feature as module

    root = tmp_path / "persistent-skills"
    touched = []

    def fail_store(_root):
        touched.append("store")
        raise AssertionError("persistent skill root opened")

    def fail_database(_agent):
        touched.append("database")
        raise AssertionError("raw database opened")

    monkeypatch.setattr(module, "SkillStore", fail_store)
    monkeypatch.setattr(module, "resolve_feature_database", fail_database)
    agent = SimpleNamespace(
        did="did:test:private-skills",
        procedural_skills_root=root,
        privacy_config=PrivacyConfig(storage=storage),
        _privacy_transition_lock=asyncio.Lock(),
    )
    private = ProceduralSkillsFeature(agent)

    await private.initialize()
    try:
        assert touched == []
        assert not root.exists()
        assert private.catalog_payload()["skills"] == []
        assert private.context_clause_text == ""
        created = await private.skill_create("private", "Private", "body")
        assert created.status is ToolResultStatus.ERROR
        assert "privacy mode" in created.error
        assert not root.exists()
    finally:
        await private.shutdown()


@pytest.mark.asyncio
async def test_transition_to_volatile_mode_blocks_every_persistent_mutation(
    feature, monkeypatch
):
    await feature.skill_create("private-guard", "Private guard", "Original body")
    await feature.skill_enable("private-guard", priority=7)
    folder = feature.agent.procedural_skills_root / "private-guard"
    original = (folder / "SKILL.md").read_bytes()
    feature.agent._privacy_transition_lock = asyncio.Lock()
    feature.agent.privacy_config = PrivacyConfig(storage="none")

    checkout_called = False

    def fail_checkout(*_args, **_kwargs):
        nonlocal checkout_called
        checkout_called = True
        raise AssertionError("git checkout started in volatile privacy mode")

    monkeypatch.setattr(
        "kestrel_feature_skills.git_source.GitSkillSource.checkout", fail_checkout
    )
    operations = (
        feature.skill_edit(
            "private-guard",
            serialize_skill_markdown(
                SkillDocument("private-guard", "Changed", "Changed body")
            ),
        ),
        feature.skill_disable("private-guard"),
        feature.skill_delete("private-guard"),
        feature.skill_create("new-private", "New private", "body"),
        feature.skill_install(
            "https://example.com/skills.git", "remote-private", "main"
        ),
    )
    results = [await operation for operation in operations]

    assert all(result.status is ToolResultStatus.ERROR for result in results)
    assert all("privacy mode" in result.error for result in results)
    assert checkout_called is False
    assert folder.is_dir()
    assert (folder / "SKILL.md").read_bytes() == original
    assert not (feature.agent.procedural_skills_root / "new-private").exists()
    assert not (feature.agent.procedural_skills_root / "remote-private").exists()
    assert feature.catalog_payload()["skills"] == []
    assert feature.context_clause_text == ""

    await feature.refresh()
    assert feature._store is None
    assert feature._db is None
    feature.agent.privacy_config = PrivacyConfig(storage="full")
    await feature.refresh()
    assert feature.snapshot.by_name()["private-guard"].state.enabled is True


@pytest.mark.asyncio
async def test_read_tools_rehydrate_after_returning_to_persistent_privacy(feature):
    await feature.skill_create("privacy-return", "Privacy return", "Persistent body")
    await feature.skill_enable("privacy-return", priority=7)
    feature.agent._privacy_transition_lock = asyncio.Lock()
    feature.agent.privacy_config = PrivacyConfig(storage="none")
    await feature.refresh()
    assert feature._store is None
    assert (await feature.skill_list()).data["count"] == 0

    feature.agent.privacy_config = PrivacyConfig(storage="full")

    listed = await feature.skill_list()
    read = await feature.skill_read("privacy-return")
    searched = await feature.skill_search("privacy return")
    assert listed.data["count"] == 1
    assert listed.data["skills"][0]["enabled"] is True
    assert read.status is ToolResultStatus.OK
    assert read.data["body"] == "Persistent body"
    assert searched.data["count"] == 1
    assert feature.context_clause_text


@pytest.mark.asyncio
async def test_mutation_rehydrates_after_returning_to_persistent_privacy(feature):
    await feature.skill_create("privacy-mutation", "Privacy mutation", "Body")
    await feature.skill_enable("privacy-mutation", priority=7)
    feature.agent._privacy_transition_lock = asyncio.Lock()
    feature.agent.privacy_config = PrivacyConfig(storage="none")
    await feature.refresh()
    feature.agent.privacy_config = PrivacyConfig(storage="full")

    disabled = await feature.skill_disable("privacy-mutation")

    assert disabled.status is ToolResultStatus.OK
    assert disabled.data["enabled"] is False


@pytest.mark.asyncio
async def test_privacy_transition_waits_for_in_flight_persistent_mutation(
    feature, monkeypatch
):
    await feature.skill_create("transition-lock", "Transition lock", "body")
    feature.agent._privacy_transition_lock = asyncio.Lock()
    feature.agent.privacy_config = PrivacyConfig(storage="full")
    entered_write = asyncio.Event()
    release_write = asyncio.Event()
    transition_entered = asyncio.Event()
    original_set = feature._enablement.set

    async def paused_set(*args, **kwargs):
        entered_write.set()
        await release_write.wait()
        return await original_set(*args, **kwargs)

    monkeypatch.setattr(feature._enablement, "set", paused_set)
    mutation = asyncio.create_task(
        feature.set_skill_state(name="transition-lock", enabled=True)
    )
    await asyncio.wait_for(entered_write.wait(), timeout=5)

    async def transition():
        async with feature.agent._privacy_transition_lock:
            transition_entered.set()
            feature.agent.privacy_config = PrivacyConfig(storage="none")

    privacy_change = asyncio.create_task(transition())
    await asyncio.sleep(0)
    assert not transition_entered.is_set()
    release_write.set()
    result = await mutation
    await privacy_change

    assert result["enabled"] is True
    assert transition_entered.is_set()
    assert feature.catalog_payload()["skills"] == []


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
async def test_skill_read_can_open_an_inventoried_resource(feature):
    created = await feature.skill_create(
        "resource-reader",
        "Read a bundled reference on demand",
        "Open the inventoried reference when needed.",
    )
    assert created.status is ToolResultStatus.OK
    edited = await feature.skill_edit(
        "resource-reader",
        "RESOURCE-ONLY-SENTINEL-3018\n",
        "references.md",
    )
    assert edited.status is ToolResultStatus.OK

    inventory = await feature.skill_read("resource-reader")
    assert any(item["path"] == "references.md" for item in inventory.data["resources"])

    opened = await feature.skill_read("resource-reader", "references.md")
    assert opened.status is ToolResultStatus.OK
    assert opened.data["path"] == "references.md"
    assert opened.data["content"] == "RESOURCE-ONLY-SENTINEL-3018\n"
    assert "RESOURCE-ONLY-SENTINEL-3018" in opened.confirmation


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
@pytest.mark.parametrize(
    "initial_enabled, requested_enabled",
    ((False, True), (True, False)),
)
async def test_ambiguous_state_write_reconciles_cached_context(
    feature, monkeypatch, initial_enabled, requested_enabled
):
    name = "ambiguous-state"
    await feature.skill_create(name, "Ambiguous state description", "body")
    if initial_enabled:
        await feature.skill_enable(name, priority=13)
    original_set = feature._enablement.set
    failure_injected = False

    async def commit_then_fail(*args, **kwargs):
        nonlocal failure_injected
        state = await original_set(*args, **kwargs)
        if not failure_injected:
            failure_injected = True
            raise DatabaseError("connection lost after state commit")
        return state

    monkeypatch.setattr(feature._enablement, "set", commit_then_fail)

    result = (
        await feature.skill_enable(name, priority=13)
        if requested_enabled
        else await feature.skill_disable(name)
    )

    assert result.status is ToolResultStatus.ERROR
    assert (await feature._enablement.load())[name].enabled is requested_enabled
    assert feature.snapshot.by_name()[name].state.enabled is requested_enabled
    assert (
        "Ambiguous state description" in feature.context_clause_text
    ) is requested_enabled


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
async def test_create_reports_unpersisted_custom_priority_without_database(
    feature, tmp_path
):
    agent = SimpleNamespace(
        did="did:test:no-skills-database",
        agent_id="did:test:no-skills-database",
        procedural_skills_root=tmp_path / "skills",
        storage=feature.agent.storage,
    )
    unavailable = ProceduralSkillsFeature(agent)
    await unavailable.initialize()
    try:
        result = await unavailable.skill_create(
            "unpersisted-priority",
            "Unpersisted priority",
            "Procedure.",
            enabled=False,
            priority=7,
        )

        assert result.status is ToolResultStatus.PARTIAL
        assert "priority" in result.error
        assert "database unavailable" in result.error
        assert result.data["enabled"] is False
        assert result.data["priority"] == 100
        assert (
            agent.procedural_skills_root / "unpersisted-priority" / "SKILL.md"
        ).is_file()
    finally:
        await unavailable.shutdown()


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
async def test_cancelled_create_never_publishes_against_stale_enabled_state(
    feature, monkeypatch
):
    name = "cancelled-replacement"
    await feature.skill_create(name, "Old local skill", "body")
    await feature.skill_enable(name)
    folder = feature.agent.procedural_skills_root / name
    for path in folder.iterdir():
        path.unlink()
    folder.rmdir()
    await feature.refresh()

    entered_state_write = asyncio.Event()

    async def paused_state_write(*_args, **_kwargs):
        entered_state_write.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(feature._enablement, "set", paused_state_write)
    creation = asyncio.create_task(
        feature.create_skill(
            name=name,
            description="Untrusted replacement",
            body="body",
        )
    )
    await asyncio.wait_for(entered_state_write.wait(), timeout=5)
    creation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await creation

    assert not folder.exists()
    await feature.refresh()
    assert "Untrusted replacement" not in feature.context_clause_text


@pytest.mark.asyncio
async def test_ambiguous_prepublication_disable_restores_prior_create_state(
    feature, monkeypatch
):
    name = "ambiguous-disable-create"
    prior_state = await feature._enablement.set(name, enabled=True, priority=17)
    feature._states[name] = prior_state
    original_set = feature._enablement.set
    failure_injected = False

    async def commit_then_fail(*args, **kwargs):
        nonlocal failure_injected
        state = await original_set(*args, **kwargs)
        if kwargs["enabled"] is False and not failure_injected:
            failure_injected = True
            raise DatabaseError("connection lost after disabling commit")
        return state

    monkeypatch.setattr(feature._enablement, "set", commit_then_fail)

    with pytest.raises(DatabaseError, match="connection lost"):
        await feature.create_skill(
            name=name,
            description="Replacement",
            body="body",
        )

    assert not (feature.agent.procedural_skills_root / name).exists()
    assert (await feature._enablement.load())[name] == SkillState(True, 17)


@pytest.mark.asyncio
@pytest.mark.parametrize("prior_state", (SkillState(True, 8), None))
async def test_ambiguous_enabled_create_failure_restores_prior_state(
    feature, monkeypatch, prior_state
):
    name = "ambiguous-create"
    if prior_state is not None:
        await feature.skill_create(name, "Previous skill", "body")
        await feature.skill_enable(name, priority=prior_state.priority)
        folder = feature.agent.procedural_skills_root / name
        for path in folder.iterdir():
            path.unlink()
        folder.rmdir()
        await feature.refresh()
    original_set = feature._enablement.set
    failure_injected = False

    async def commit_then_fail(*args, **kwargs):
        nonlocal failure_injected
        state = await original_set(*args, **kwargs)
        if kwargs["enabled"] and not failure_injected:
            failure_injected = True
            raise DatabaseError("connection lost after commit")
        return state

    monkeypatch.setattr(feature._enablement, "set", commit_then_fail)

    result = await feature.skill_create(
        name,
        "Ambiguous create",
        "body",
        enabled=True,
    )

    assert result.status is ToolResultStatus.ERROR
    assert not (feature.agent.procedural_skills_root / name).exists()
    persisted = await feature._enablement.load()
    if prior_state is None:
        assert name not in persisted
        assert name not in feature._states
    else:
        assert persisted[name] == prior_state
        assert feature._states[name] == prior_state


@pytest.mark.asyncio
@pytest.mark.parametrize("prior_state", (SkillState(True, 8), None))
async def test_failed_create_publication_restores_prior_enablement(
    feature, monkeypatch, prior_state
):
    name = "create-publication-failure"
    if prior_state is not None:
        await feature.skill_create(name, "Previous skill", "body")
        await feature.skill_enable(name, priority=prior_state.priority)
        folder = feature.agent.procedural_skills_root / name
        for path in folder.iterdir():
            path.unlink()
        folder.rmdir()
        await feature.refresh()

    def fail_publication(_document):
        raise OSError("disk publication failed")

    monkeypatch.setattr(feature._store, "create", fail_publication)

    result = await feature.skill_create(name, "Replacement", "body")

    assert result.status is ToolResultStatus.ERROR
    persisted = await feature._enablement.load()
    if prior_state is None:
        assert name not in persisted
        assert name not in feature._states
    else:
        assert persisted[name] == prior_state
        assert feature._states[name] == prior_state


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
async def test_cancelled_post_publication_refresh_reconciles_catalog(
    feature, monkeypatch
):
    name = "cancelled-edit-refresh"
    await feature.skill_create(name, "Original description", "Original body")
    await feature.skill_enable(name)
    original_refresh = feature._refresh_locked
    refresh_started = asyncio.Event()
    release_refresh = asyncio.Event()

    async def paused_refresh():
        refresh_started.set()
        await release_refresh.wait()
        return await original_refresh()

    monkeypatch.setattr(feature, "_refresh_locked", paused_refresh)
    replacement = serialize_skill_markdown(
        SkillDocument(name, "Replacement description", "Replacement body")
    )
    edit = asyncio.create_task(
        feature.edit_skill(
            name=name,
            relative_path="SKILL.md",
            content=replacement,
        )
    )
    await asyncio.wait_for(refresh_started.wait(), timeout=5)
    edit.cancel()
    await asyncio.sleep(0)
    try:
        assert not edit.done(), (
            "cancelled mutation returned before its committed filesystem state "
            "was reconciled"
        )
        edit.cancel()
        await asyncio.sleep(0)
        assert not edit.done(), "repeated cancellation skipped catalog reconciliation"
    finally:
        release_refresh.set()
    with pytest.raises(asyncio.CancelledError):
        await edit

    record = feature.snapshot.by_name()[name]
    assert record.document.description == "Replacement description"
    assert record.document.body == "Replacement body"
    assert "Replacement description" in feature.context_clause_text
    assert "Original description" not in feature.context_clause_text


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

    def fake_checkout(self, *, url, ref, skill_name, target, cancel_event=None):
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
async def test_cancelled_git_install_waits_for_worker_before_releasing(
    feature, monkeypatch
):
    worker_started = threading.Event()
    release_legacy_worker = threading.Event()
    worker_finished = threading.Event()

    def blocked_checkout(self, **kwargs):
        cancel_event = kwargs.get("cancel_event")
        worker_started.set()
        if cancel_event is None:
            release_legacy_worker.wait(timeout=5)
        else:
            cancel_event.wait(timeout=5)
        worker_finished.set()
        raise GitSourceError("git checkout cancelled")

    monkeypatch.setattr(
        "kestrel_feature_skills.git_source.GitSkillSource.checkout",
        blocked_checkout,
    )
    feature.agent._privacy_transition_lock = asyncio.Lock()
    install = asyncio.create_task(
        feature.install_skill(
            source_url="https://example.com/repo.git",
            skill_name="cancelled-install",
            ref="main",
        )
    )
    assert await asyncio.to_thread(worker_started.wait, 5)

    install.cancel()
    with pytest.raises(asyncio.CancelledError):
        await install
    finished_before_lock_release = worker_finished.is_set()
    release_legacy_worker.set()
    assert await asyncio.to_thread(worker_finished.wait, 5)

    assert finished_before_lock_release, (
        "install cancellation propagated and released the privacy lock while "
        "the Git worker was still running"
    )


@pytest.mark.asyncio
async def test_ambiguous_prepublication_disable_restores_prior_install_state(
    feature, tmp_path, monkeypatch
):
    name = "ambiguous-disable-install"
    prior_state = await feature._enablement.set(name, enabled=True, priority=19)
    feature._states[name] = prior_state
    checkout_root = tmp_path / "checkout-ambiguous-disable"
    source = checkout_root / "skills" / name
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text(
        serialize_skill_markdown(SkillDocument(name, "Remote", "body")),
        encoding="utf-8",
    )

    def fake_checkout(self, *, url, ref, skill_name, target, cancel_event=None):
        return GitCheckout(
            root=checkout_root,
            skill_folder=source,
            revision="e" * 40,
            remote_url=url,
            ref=ref,
        )

    original_set = feature._enablement.set
    failure_injected = False

    async def commit_then_fail(*args, **kwargs):
        nonlocal failure_injected
        state = await original_set(*args, **kwargs)
        if kwargs["enabled"] is False and not failure_injected:
            failure_injected = True
            raise DatabaseError("connection lost after disabling commit")
        return state

    monkeypatch.setattr(
        "kestrel_feature_skills.git_source.GitSkillSource.checkout", fake_checkout
    )
    monkeypatch.setattr(feature._enablement, "set", commit_then_fail)

    with pytest.raises(DatabaseError, match="connection lost"):
        await feature.install_skill(
            source_url="https://example.com/repo.git",
            skill_name=name,
            ref="main",
        )

    assert not (feature.agent.procedural_skills_root / name).exists()
    assert (await feature._enablement.load())[name] == SkillState(True, 19)


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

    def fake_checkout(self, *, url, ref, skill_name, target, cancel_event=None):
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
@pytest.mark.parametrize("prior_state", (SkillState(True, 7), None))
async def test_failed_git_install_restores_prior_enablement_state(
    feature, tmp_path, monkeypatch, prior_state
):
    name = "install-rollback"
    folder = feature.agent.procedural_skills_root / name
    if prior_state is not None:
        await feature.skill_create(name, "Previously local", "body")
        await feature.skill_enable(name, priority=prior_state.priority)
        (folder / "SKILL.md").write_text("", encoding="utf-8")
    else:
        folder.mkdir()
        (folder / "SKILL.md").write_text("", encoding="utf-8")
    await feature.refresh()
    assert name not in feature.snapshot.by_name()

    checkout_root = tmp_path / "checkout-rollback"
    source = checkout_root / "skills" / name
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text(
        serialize_skill_markdown(SkillDocument(name, "Remote", "body")),
        encoding="utf-8",
    )

    def fake_checkout(self, *, url, ref, skill_name, target, cancel_event=None):
        return GitCheckout(
            root=checkout_root,
            skill_folder=source,
            revision="d" * 40,
            remote_url=url,
            ref=ref,
        )

    monkeypatch.setattr(
        "kestrel_feature_skills.git_source.GitSkillSource.checkout", fake_checkout
    )

    result = await feature.skill_install("https://example.com/repo.git", name, "main")

    assert result.status is ToolResultStatus.ERROR
    persisted = await feature._enablement.load()
    if prior_state is None:
        assert name not in persisted
        assert name not in feature._states
    else:
        assert persisted[name] == prior_state
        assert feature._states[name] == prior_state


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


@pytest.mark.asyncio
@pytest.mark.parametrize("relative_path", ("notes.txt", "tool.py"))
async def test_read_file_editability_matches_write_policy(feature, relative_path):
    await feature.skill_create("read-policy", "Read policy", "body")
    folder = feature.agent.procedural_skills_root / "read-policy"
    (folder / relative_path).write_text("resource", encoding="utf-8")
    await feature.refresh()

    tree_entry = next(
        entry
        for entry in feature.tree(name="read-policy")
        if entry["path"] == relative_path
    )
    response = feature.read_file(name="read-policy", relative_path=relative_path)

    assert tree_entry["editable"] is False
    assert response["editable"] is False


def test_context_renderer_is_not_registered_through_a_fake_legacy_hook(feature):
    assert feature.context_clause_text == render_context_clause(feature.snapshot).text
    assert not hasattr(feature, "get_context_clause_registrations")
    assert feature.get_hooks() == []
