from __future__ import annotations

import asyncio
import threading
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from kestrel_sdk.features.contributions import PermissionLevel
from kestrel_sdk.storage.database import DatabaseError
from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.privacy import PrivacyConfig
from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore, GraphNode

import kestrel_feature_skills.store as store_module
from kestrel_feature_skills import ProceduralSkillsFeature
from kestrel_feature_skills.context import render_context_clause
from kestrel_feature_skills.enablement import DEFAULT_PRIORITY, MAX_PRIORITY
from kestrel_feature_skills.errors import (
    GitSourceError,
    SkillConflictError,
    SkillFormatError,
    SkillNotFoundError,
    SkillPathError,
)
from kestrel_feature_skills.feature import PROCEDURAL_SKILL_NODE_TYPE
from kestrel_feature_skills.format import serialize_skill_markdown
from kestrel_feature_skills.git_source import GitCheckout
from kestrel_feature_skills.models import SkillDocument, SkillProvenance, SkillState
from kestrel_feature_skills.sources import (
    AGENT_LOCAL_PRECEDENCE,
    HOST_SHARED_PRECEDENCE,
    PROVENANCE_FILENAME,
    DirectorySkillSource,
    SkillCatalog,
    serialize_provenance,
)

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


def delete_revision(feature, name: str) -> str:
    """Return the revision token exposed for a deletable local generation."""

    record = next(
        item for item in feature.catalog_payload()["skills"] if item["name"] == name
    )
    revision = record["delete_revision"]
    assert isinstance(revision, str)
    return revision


@pytest.mark.asyncio
async def test_catalog_refresh_scans_without_blocking_the_event_loop(
    feature, monkeypatch
):
    original_refresh = feature._catalog.refresh
    release = threading.Event()
    observed_release = []

    def pause_catalog(states):
        observed_release.append(release.wait(timeout=1))
        return original_refresh(states)

    monkeypatch.setattr(feature._catalog, "refresh", pause_catalog)
    asyncio.get_running_loop().call_later(0.01, release.set)

    await feature.refresh()

    assert observed_release == [True]


@pytest.mark.asyncio
async def test_cancelled_refresh_drains_catalog_worker_before_return(
    feature, monkeypatch
):
    original_refresh = feature._catalog.refresh
    started = threading.Event()
    release = threading.Event()

    def pause_catalog(states):
        started.set()
        assert release.wait(timeout=5)
        return original_refresh(states)

    monkeypatch.setattr(feature._catalog, "refresh", pause_catalog)
    refresh = asyncio.create_task(feature.refresh())
    assert await asyncio.to_thread(started.wait, 1)
    refresh.cancel()
    await asyncio.sleep(0.05)
    try:
        assert not refresh.done(), "cancellation escaped while catalog scan was active"
    finally:
        release.set()

    with pytest.raises(asyncio.CancelledError):
        await refresh


class UnexpectedGraphError(Exception):
    """A non-database graph failure used to prove best-effort containment."""


@pytest.mark.asyncio
async def test_same_name_creates_serialize_disabled_guard_with_publication(
    feature, monkeypatch
):
    name = "serialized-create-guard"
    prior = await feature._enablement.set(name, enabled=True, priority=17)
    feature._states[name] = prior
    rival = ProceduralSkillsFeature(feature.agent)
    await rival.initialize()
    guard_written = asyncio.Event()
    release_first = asyncio.Event()
    original_prepare = feature._prepare_disabled_state

    async def pause_after_guard(*args, **kwargs):
        prepared = await original_prepare(*args, **kwargs)
        guard_written.set()
        await release_first.wait()
        return prepared

    monkeypatch.setattr(feature, "_prepare_disabled_state", pause_after_guard)
    first = asyncio.create_task(
        feature.create_skill(
            name=name,
            description="First serialized creator",
            body="Procedure.",
            priority=7,
        )
    )
    await asyncio.wait_for(guard_written.wait(), timeout=5)
    second = asyncio.create_task(
        rival.create_skill(
            name=name,
            description="Second serialized creator",
            body="Procedure.",
        )
    )
    await asyncio.sleep(0.1)
    second_waited = not second.done()
    release_first.set()
    try:
        outcomes = await asyncio.gather(first, second, return_exceptions=True)
    finally:
        release_first.set()
        await rival.shutdown()

    assert second_waited, "a rival publication passed the first creator's state guard"
    assert sum(isinstance(outcome, dict) for outcome in outcomes) == 1
    assert sum(isinstance(outcome, SkillConflictError) for outcome in outcomes) == 1
    assert (await feature._enablement.load())[name] == SkillState(False, 7)
    assert feature.snapshot.by_name()[name].state == SkillState(False, 7)


@pytest.mark.asyncio
async def test_same_name_install_waits_for_create_guard_and_publication(
    feature, monkeypatch
):
    name = "serialized-install-guard"
    prior = await feature._enablement.set(name, enabled=True, priority=17)
    feature._states[name] = prior
    rival = ProceduralSkillsFeature(feature.agent)
    await rival.initialize()
    guard_written = asyncio.Event()
    release_first = asyncio.Event()
    original_prepare = feature._prepare_disabled_state

    async def pause_after_guard(*args, **kwargs):
        prepared = await original_prepare(*args, **kwargs)
        guard_written.set()
        await release_first.wait()
        return prepared

    async def fake_checkout(*, source_url, ref, skill_name, target):
        source = target / skill_name
        source.mkdir(parents=True)
        (source / "SKILL.md").write_text(
            serialize_skill_markdown(
                SkillDocument(skill_name, "Installing rival", "Procedure.")
            ),
            encoding="utf-8",
        )
        return GitCheckout(
            root=target,
            skill_folder=source,
            revision="a" * 40,
            remote_url=source_url,
            ref=ref,
        )

    monkeypatch.setattr(feature, "_prepare_disabled_state", pause_after_guard)
    monkeypatch.setattr(rival, "_checkout_git_until_stopped", fake_checkout)
    first = asyncio.create_task(
        feature.create_skill(
            name=name,
            description="First serialized creator",
            body="Procedure.",
            priority=7,
        )
    )
    await asyncio.wait_for(guard_written.wait(), timeout=5)
    second = asyncio.create_task(
        rival.install_skill(
            source_url="https://example.com/skills.git",
            skill_name=name,
            ref="main",
        )
    )
    await asyncio.sleep(0.1)
    second_waited = not second.done()
    release_first.set()
    try:
        outcomes = await asyncio.gather(first, second, return_exceptions=True)
    finally:
        release_first.set()
        await rival.shutdown()

    assert second_waited, "an install passed the first creator's state guard"
    assert sum(isinstance(outcome, dict) for outcome in outcomes) == 1
    assert sum(isinstance(outcome, SkillConflictError) for outcome in outcomes) == 1
    assert (await feature._enablement.load())[name] == SkillState(False, 7)


@pytest.mark.asyncio
async def test_enable_waits_for_same_name_local_override_publication(
    feature, tmp_path, monkeypatch
):
    name = "serialized-state-update"
    shared_root = tmp_path / "shared-skills"
    shared_folder = shared_root / name
    shared_folder.mkdir(parents=True)
    (shared_folder / "SKILL.md").write_text(
        serialize_skill_markdown(
            SkillDocument(name, "Host shared predecessor", "Procedure.")
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("KESTREL_SHARED_SKILLS_DIR", str(shared_root))
    creator = ProceduralSkillsFeature(feature.agent)
    enabler = ProceduralSkillsFeature(feature.agent)
    await creator.initialize()
    await enabler.initialize()
    guard_written = asyncio.Event()
    release_creator = asyncio.Event()
    original_prepare = creator._prepare_disabled_state

    async def pause_after_guard(*args, **kwargs):
        prepared = await original_prepare(*args, **kwargs)
        guard_written.set()
        await release_creator.wait()
        return prepared

    monkeypatch.setattr(creator, "_prepare_disabled_state", pause_after_guard)
    creation = asyncio.create_task(
        creator.create_skill(
            name=name,
            description="Agent-local replacement",
            body="Procedure.",
            enabled=False,
            priority=7,
        )
    )
    await asyncio.wait_for(guard_written.wait(), timeout=5)
    state_update = asyncio.create_task(enabler.set_skill_state(name=name, enabled=True))
    await asyncio.sleep(0.1)
    update_waited = not state_update.done()
    release_creator.set()
    try:
        created, updated = await asyncio.gather(creation, state_update)
    finally:
        release_creator.set()
        await creator.shutdown()
        await enabler.shutdown()

    assert update_waited, "state update passed an in-flight same-name publication"
    assert created["enabled"] is False
    assert updated["enabled"] is True
    assert (await feature._enablement.load())[name] == SkillState(True, 7)


@pytest.mark.asyncio
async def test_create_waits_for_same_name_delete_state_cleanup(feature, monkeypatch):
    name = "serialized-delete-cleanup"
    created = await feature.create_skill(
        name=name,
        description="Original local skill",
        body="Procedure.",
        enabled=True,
        priority=13,
    )
    assert created["enabled"] is True
    creator = ProceduralSkillsFeature(feature.agent)
    await creator.initialize()
    folder_removed = asyncio.Event()
    release_delete = asyncio.Event()
    original_delete_state = feature._enablement.delete

    async def pause_before_state_cleanup(skill_name):
        if skill_name == name:
            folder_removed.set()
            await release_delete.wait()
        return await original_delete_state(skill_name)

    monkeypatch.setattr(feature._enablement, "delete", pause_before_state_cleanup)
    deletion = asyncio.create_task(feature.delete_skill(name=name))
    await asyncio.wait_for(folder_removed.wait(), timeout=5)
    replacement = asyncio.create_task(
        creator.create_skill(
            name=name,
            description="Replacement local skill",
            body="Procedure.",
            priority=7,
        )
    )
    await asyncio.sleep(0.1)
    replacement_waited = not replacement.done()
    release_delete.set()
    try:
        deleted, recreated = await asyncio.gather(deletion, replacement)
    finally:
        release_delete.set()
        await creator.shutdown()

    assert replacement_waited, "publication passed same-name delete state cleanup"
    assert deleted["removed_file"] is True
    assert recreated["priority"] == 7
    assert (await feature._enablement.load())[name] == SkillState(False, 7)


@pytest.mark.asyncio
async def test_feature_catalog_rejects_replacement_of_pinned_local_root(
    feature, tmp_path
):
    await feature.skill_create("trusted-root", "Trusted original", "Procedure.")
    local_root = feature._store.local_root
    displaced = tmp_path / "displaced-skills-root"
    local_root.rename(displaced)
    local_root.mkdir()
    attacker = local_root / "attacker-root"
    attacker.mkdir()
    (attacker / "SKILL.md").write_text(
        serialize_skill_markdown(
            SkillDocument("attacker-root", "Untrusted replacement", "Procedure.")
        ),
        encoding="utf-8",
    )

    await feature.refresh()

    assert feature.snapshot.records == ()
    assert any(
        error.source_id == "agent-local" and "changed" in error.error
        for error in feature.snapshot.errors
    )


@pytest.mark.asyncio
async def test_unresolvable_optional_shared_root_does_not_abort_local_catalog(
    feature, tmp_path, monkeypatch
):
    await feature.skill_create("healthy-local", "Healthy local", "Procedure.")
    shared_root = tmp_path / "unresolvable-shared"
    shared_root.mkdir()
    monkeypatch.setenv("KESTREL_SHARED_SKILLS_DIR", str(shared_root))
    real_resolve = Path.resolve

    def fail_shared_resolution(path, *args, **kwargs):
        if path == shared_root:
            raise RuntimeError("symlink loop while resolving optional shared root")
        return real_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", fail_shared_resolution)
    isolated = ProceduralSkillsFeature(feature.agent)
    try:
        await isolated.initialize()
        assert "healthy-local" in isolated.snapshot.by_name()
        assert any(
            error.source_id == "host-shared" and "source root" in error.error
            for error in isolated.snapshot.errors
        )
    finally:
        await isolated.shutdown()


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
        feature.skill_delete("private-guard", "0" * 64),
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
async def test_skill_read_holds_privacy_lock_through_inventory_and_resource_read(
    feature, monkeypatch
):
    await feature.skill_create("privacy-read-lock", "Privacy read lock", "Body")
    await feature.skill_edit("privacy-read-lock", "Resource", relative_path="notes.md")
    feature.agent._privacy_transition_lock = asyncio.Lock()
    feature.agent.privacy_config = PrivacyConfig(storage="full")
    observed = []
    original_tree = feature._store.tree
    original_read_file = feature._store.read_file

    def observed_tree(record):
        observed.append(("tree", feature.agent._privacy_transition_lock.locked()))
        return original_tree(record)

    def observed_read_file(record, relative_path):
        observed.append(("file", feature.agent._privacy_transition_lock.locked()))
        return original_read_file(record, relative_path)

    monkeypatch.setattr(feature._store, "tree", observed_tree)
    monkeypatch.setattr(feature._store, "read_file", observed_read_file)

    result = await feature.skill_read("privacy-read-lock", "notes.md")

    assert result.status is ToolResultStatus.OK
    assert result.data["content"] == "Resource"
    assert observed == [("tree", True), ("file", True)]


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
    tools = {tool.name: tool for tool in feature.get_tools()}
    assert set(tools) == EXPECTED_TOOLS
    delete_parameters = {
        parameter.name: parameter
        for parameter in tools["skill_delete"].schema.parameters
    }
    assert delete_parameters["delete_revision"].required is True
    assert (
        "skill_list or skill_search" in delete_parameters["delete_revision"].description
    )
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
async def test_unknown_state_update_does_not_create_publication_claim(
    feature, monkeypatch
):
    attempted_claims = []
    real_acquire = feature._store.try_acquire_publication_state_claim

    def record_claim(name):
        attempted_claims.append(name)
        return real_acquire(name)

    monkeypatch.setattr(
        feature._store,
        "try_acquire_publication_state_claim",
        record_claim,
    )

    with pytest.raises(SkillNotFoundError, match="was not found"):
        await feature.set_skill_state(name="never-published", enabled=True)

    assert attempted_claims == []
    assert not list(
        feature._store._internal_root.glob(".publication-state-bucket-*.lock")
    )


@pytest.mark.asyncio
async def test_state_update_refreshes_external_publication_before_precheck(feature):
    name = "externally-published-state"
    folder = feature.agent.procedural_skills_root / name
    folder.mkdir()
    (folder / "SKILL.md").write_text(
        serialize_skill_markdown(
            SkillDocument(name, "External publication", "Procedure.")
        ),
        encoding="utf-8",
    )

    updated = await feature.set_skill_state(name=name, enabled=True, priority=19)

    assert updated["enabled"] is True
    assert updated["priority"] == 19
    assert feature.snapshot.by_name()[name].state == SkillState(True, 19)


@pytest.mark.asyncio
async def test_state_change_rolls_back_when_folder_changes_during_persistence(
    feature, monkeypatch
):
    name = "state-content-race"
    await feature.skill_create(name, "Original", "Original procedure.")
    record = feature.snapshot.by_name()[name]
    folder = feature.agent.procedural_skills_root / name
    external = serialize_skill_markdown(
        SkillDocument(name, "External", "EXTERNAL procedure.")
    )
    real_set = feature._enablement.set
    injected = False

    async def persist_then_mutate(*args, **kwargs):
        nonlocal injected
        state = await real_set(*args, **kwargs)
        if not injected:
            injected = True
            (folder / "SKILL.md").write_text(external, encoding="utf-8")
        return state

    monkeypatch.setattr(feature._enablement, "set", persist_then_mutate)

    with pytest.raises(SkillConflictError, match="changed after it was read"):
        await feature.set_skill_state(
            name=name,
            enabled=True,
            priority=19,
            expected_revision=record.revision,
        )

    assert injected
    assert (await feature._enablement.load())[name] == SkillState(False, 100)
    assert feature.snapshot.by_name()[name].state == SkillState(False, 100)
    assert (folder / "SKILL.md").read_text(encoding="utf-8") == external


@pytest.mark.asyncio
async def test_state_change_stays_locally_disabled_when_rollback_cannot_persist(
    feature, monkeypatch
):
    name = "state-rollback-failure"
    await feature.skill_create(name, "Original", "Original procedure.")
    record = feature.snapshot.by_name()[name]
    folder = feature.agent.procedural_skills_root / name
    replacement = serialize_skill_markdown(
        SkillDocument(name, "Unapproved replacement", "Replacement procedure.")
    )
    real_set = feature._enablement.set

    async def commit_enable_but_refuse_rollback(*args, **kwargs):
        if kwargs["enabled"] is False:
            raise DatabaseError("rollback database write failed")
        state = await real_set(*args, **kwargs)
        (folder / "SKILL.md").write_text(replacement, encoding="utf-8")
        return state

    monkeypatch.setattr(
        feature._enablement,
        "set",
        commit_enable_but_refuse_rollback,
    )

    with pytest.raises(SkillConflictError, match="changed after it was read"):
        await feature.set_skill_state(
            name=name,
            enabled=True,
            priority=19,
            expected_revision=record.revision,
        )

    assert (await feature._enablement.load())[name] == SkillState(True, 19)
    assert feature.snapshot.by_name()[name].state == SkillState(False, 100)
    assert "Unapproved replacement" not in feature.context_clause_text
    assert (
        "rollback database write failed"
        in feature.catalog_payload()["enablement_error"]
    )

    restarted = ProceduralSkillsFeature(feature.agent)
    await restarted.initialize()
    try:
        assert restarted.snapshot.by_name()[name].state == SkillState(False, 100)
        assert "Unapproved replacement" not in restarted.context_clause_text
    finally:
        await restarted.shutdown()

    monkeypatch.setattr(feature._enablement, "set", real_set)
    guarded_revision = feature.snapshot.by_name()[name].revision
    await feature.set_skill_state(
        name=name,
        enabled=False,
        priority=100,
        expected_revision=guarded_revision,
    )
    assert feature.catalog_payload()["enablement_error"] is None
    assert feature.snapshot.by_name()[name].state == SkillState(False, 100)


@pytest.mark.asyncio
async def test_disable_race_does_not_restore_enablement_to_replacement(
    feature, monkeypatch
):
    name = "disable-raced-replacement"
    await feature.skill_create(name, "Original", "Original procedure.")
    await feature.skill_enable(name, priority=23)
    record = feature.snapshot.by_name()[name]
    folder = feature.agent.procedural_skills_root / name
    replacement = serialize_skill_markdown(
        SkillDocument(name, "Unapproved replacement", "Replacement procedure.")
    )
    real_set = feature._enablement.set
    mutated = False

    async def persist_then_replace(*args, **kwargs):
        nonlocal mutated
        state = await real_set(*args, **kwargs)
        if not mutated:
            mutated = True
            (folder / "SKILL.md").write_text(replacement, encoding="utf-8")
        return state

    monkeypatch.setattr(feature._enablement, "set", persist_then_replace)

    with pytest.raises(SkillConflictError, match="changed after it was read"):
        await feature.set_skill_state(
            name=name,
            enabled=False,
            expected_revision=record.revision,
        )

    assert (await feature._enablement.load())[name] == SkillState(False, 23)
    assert feature.snapshot.by_name()[name].state == SkillState(False, 23)
    assert "Unapproved replacement" not in feature.context_clause_text


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ("replace", "remove"))
async def test_state_change_rolls_back_when_folder_changes_during_final_refresh(
    feature, monkeypatch, mutation
):
    name = f"state-final-refresh-{mutation}"
    await feature.skill_create(name, "Original", "Original procedure.")
    record = feature.snapshot.by_name()[name]
    folder = feature.agent.procedural_skills_root / name
    replacement = serialize_skill_markdown(
        SkillDocument(name, "Replacement", "Replacement procedure.")
    )
    real_refresh = feature._refresh_locked
    refresh_count = 0

    async def refresh_then_mutate():
        nonlocal refresh_count
        await real_refresh()
        refresh_count += 1
        if refresh_count != 3:
            return
        if mutation == "replace":
            (folder / "SKILL.md").write_text(replacement, encoding="utf-8")
        else:
            for child in folder.iterdir():
                child.unlink()
            folder.rmdir()

    monkeypatch.setattr(feature, "_refresh_locked", refresh_then_mutate)

    with pytest.raises((SkillConflictError, SkillPathError), match="changed"):
        await feature.set_skill_state(
            name=name,
            enabled=True,
            priority=19,
            expected_revision=record.revision,
        )

    assert refresh_count >= 4
    assert (await feature._enablement.load())[name] == SkillState(False, 100)
    if mutation == "replace":
        assert feature.snapshot.by_name()[name].document.description == "Replacement"
        assert feature.snapshot.by_name()[name].state == SkillState(False, 100)
    else:
        assert name not in feature.snapshot.by_name()
        folder.mkdir()
        (folder / "SKILL.md").write_text(replacement, encoding="utf-8")
        await feature.refresh()
        assert feature.snapshot.by_name()[name].state == SkillState(False, 100)


@pytest.mark.asyncio
async def test_cancelled_state_refresh_rolls_back_a_replaced_generation(
    feature, monkeypatch
):
    name = "cancelled-state-replacement"
    await feature.skill_create(name, "Original", "Original procedure.")
    record = feature.snapshot.by_name()[name]
    folder = feature.agent.procedural_skills_root / name
    replacement = serialize_skill_markdown(
        SkillDocument(name, "Unapproved replacement", "Replacement procedure.")
    )
    real_refresh = feature._refresh_locked
    final_refresh_started = asyncio.Event()
    release_final_refresh = asyncio.Event()
    rollback_started = asyncio.Event()
    release_rollback = asyncio.Event()
    refresh_calls = 0
    real_set = feature._enablement.set

    async def pause_rollback(*args, **kwargs):
        if kwargs["enabled"] is False:
            rollback_started.set()
            await release_rollback.wait()
        return await real_set(*args, **kwargs)

    async def pause_final_refresh():
        nonlocal refresh_calls
        refresh_calls += 1
        if refresh_calls == 3:
            final_refresh_started.set()
            await release_final_refresh.wait()
        return await real_refresh()

    monkeypatch.setattr(feature, "_refresh_locked", pause_final_refresh)
    monkeypatch.setattr(feature._enablement, "set", pause_rollback)
    update = asyncio.create_task(
        feature.set_skill_state(
            name=name,
            enabled=True,
            priority=19,
            expected_revision=record.revision,
        )
    )
    try:
        await asyncio.wait_for(final_refresh_started.wait(), timeout=5)
        (folder / "SKILL.md").write_text(replacement, encoding="utf-8")
        update.cancel()
        await asyncio.sleep(0)
        assert not update.done(), "cancellation skipped state consistency cleanup"
        release_final_refresh.set()
        await asyncio.wait_for(rollback_started.wait(), timeout=5)
        update.cancel()
        await asyncio.sleep(0)
        assert not update.done(), "repeated cancellation skipped state rollback"
        release_rollback.set()

        with pytest.raises(asyncio.CancelledError):
            await update

        assert (await feature._enablement.load())[name] == SkillState(False, 100)
        assert feature.snapshot.by_name()[name].state == SkillState(False, 100)
        assert "Unapproved replacement" not in feature.context_clause_text
    finally:
        release_final_refresh.set()
        release_rollback.set()
        if not update.done():
            update.cancel()
            await asyncio.wait_for(
                asyncio.gather(update, return_exceptions=True),
                timeout=5,
            )


@pytest.mark.asyncio
async def test_state_finalization_guard_is_visible_to_another_feature_instance(
    feature, monkeypatch
):
    name = "cross-instance-state-guard"
    await feature.skill_create(name, "Guarded", "Guarded procedure.")
    record = feature.snapshot.by_name()[name]
    observer = ProceduralSkillsFeature(feature.agent)
    await observer.initialize()
    real_refresh = feature._refresh_locked
    final_refresh_started = asyncio.Event()
    release_final_refresh = asyncio.Event()
    refresh_calls = 0

    async def pause_final_refresh(**kwargs):
        nonlocal refresh_calls
        refresh_calls += 1
        if refresh_calls == 3:
            final_refresh_started.set()
            await release_final_refresh.wait()
        return await real_refresh(**kwargs)

    monkeypatch.setattr(feature, "_refresh_locked", pause_final_refresh)
    update = asyncio.create_task(
        feature.set_skill_state(
            name=name,
            enabled=True,
            expected_revision=record.revision,
        )
    )
    try:
        await asyncio.wait_for(final_refresh_started.wait(), timeout=5)
        await observer.refresh()
        assert observer.snapshot.by_name()[name].state.enabled is False
        assert "Guarded" not in observer.context_clause_text

        release_final_refresh.set()
        result = await update
        assert result["enabled"] is True
        await observer.refresh()
        assert observer.snapshot.by_name()[name].state.enabled is True
        assert "Guarded" in observer.context_clause_text
    finally:
        release_final_refresh.set()
        if not update.done():
            update.cancel()
            with pytest.raises(asyncio.CancelledError):
                await update
        await observer.shutdown()


@pytest.mark.asyncio
async def test_state_change_rolls_back_when_final_refresh_resolves_a_new_override(
    feature, tmp_path, monkeypatch
):
    name = "state-final-refresh-override"
    shared_root = tmp_path / "shared"
    shared_folder = shared_root / name
    shared_folder.mkdir(parents=True)
    (shared_folder / "SKILL.md").write_text(
        serialize_skill_markdown(
            SkillDocument(name, "Shared generation", "Shared procedure.")
        ),
        encoding="utf-8",
    )
    feature._catalog = SkillCatalog(
        (
            DirectorySkillSource(
                root=feature.agent.procedural_skills_root,
                source_id="agent-local",
                kind="agent-local",
                precedence=AGENT_LOCAL_PRECEDENCE,
            ),
            DirectorySkillSource(
                root=shared_root,
                source_id="host-shared",
                kind="host-shared",
                precedence=HOST_SHARED_PRECEDENCE,
            ),
        )
    )
    await feature.refresh()
    record = feature.snapshot.by_name()[name]
    local_folder = feature.agent.procedural_skills_root / name
    real_refresh = feature._refresh_locked
    refresh_count = 0

    async def publish_override_then_refresh():
        nonlocal refresh_count
        refresh_count += 1
        if refresh_count == 3:
            local_folder.mkdir()
            (local_folder / "SKILL.md").write_text(
                serialize_skill_markdown(
                    SkillDocument(name, "Local replacement", "Local procedure.")
                ),
                encoding="utf-8",
            )
        await real_refresh()

    monkeypatch.setattr(feature, "_refresh_locked", publish_override_then_refresh)

    with pytest.raises(SkillConflictError, match="resolved source changed"):
        await feature.set_skill_state(
            name=name,
            enabled=True,
            priority=19,
            expected_revision=record.revision,
        )

    assert refresh_count >= 4
    assert name not in await feature._enablement.load()
    replacement = feature.snapshot.by_name()[name]
    assert replacement.source_kind == "agent-local"
    assert replacement.document.description == "Local replacement"
    assert replacement.state == SkillState(False, 100)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "initial_enabled, requested_enabled",
    ((False, True), (True, False)),
)
async def test_ambiguous_state_write_keeps_cached_context_fail_closed(
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
    assert feature.snapshot.by_name()[name].state.enabled is False
    assert "Ambiguous state description" not in feature.context_clause_text


@pytest.mark.asyncio
async def test_create_writes_procedural_skill_graph_node(feature):
    result = await feature.skill_create("indexed", "Graph indexed", "body")
    assert result.status is ToolResultStatus.OK
    node = feature.agent.storage.added[-1]
    assert node.node_type == PROCEDURAL_SKILL_NODE_TYPE
    assert node.node_type != "skill"
    assert node.properties["name"] == "indexed"
    assert datetime.fromisoformat(node.properties["created_at"]).tzinfo is not None


@pytest.mark.asyncio
async def test_graph_index_preserves_created_at_and_repairs_legacy_untimed_node(
    feature,
):
    name = "timed-index"
    await feature.skill_create(name, "Timed graph index", "body")
    node_id = feature._node_id(name)
    created_at = feature.agent.storage.nodes[node_id].properties["created_at"]

    await feature.skill_enable(name, priority=8)

    assert feature.agent.storage.nodes[node_id].properties["created_at"] == created_at

    feature.agent.storage.nodes[node_id].properties.pop("created_at")
    feature.agent.storage.added.clear()
    replacement = ProceduralSkillsFeature(feature.agent)
    await replacement.initialize()
    try:
        repaired = feature.agent.storage.nodes[node_id]
        assert len(feature.agent.storage.added) == 1
        assert (
            datetime.fromisoformat(repaired.properties["created_at"]).tzinfo is not None
        )
        assert repaired.properties["enabled"] is True
        assert repaired.properties["priority"] == 8
    finally:
        await replacement.shutdown()


@pytest.mark.asyncio
async def test_initialize_removes_legacy_untimed_stale_index(feature):
    name = "untimed-restart-stale"
    await feature.skill_create(name, "Untimed stale graph index", "body")
    node_id = feature._node_id(name)
    feature.agent.storage.nodes[node_id].properties.pop("created_at")
    folder = feature.agent.procedural_skills_root / name
    for path in folder.iterdir():
        path.unlink()
    folder.rmdir()

    replacement = ProceduralSkillsFeature(feature.agent)
    await replacement.initialize()
    try:
        assert node_id not in feature.agent.storage.nodes
        assert node_id in feature.agent.storage.deleted
    finally:
        await replacement.shutdown()


@pytest.mark.asyncio
async def test_indexing_preserves_non_skill_node_at_deterministic_id(feature):
    name = "graph-id-collision"
    node_id = feature._node_id(name)
    unrelated = GraphNode(
        node_id=node_id,
        node_type="episode",
        label="unrelated graph data",
        properties={"sentinel": "must survive"},
    )
    feature.agent.storage.nodes[node_id] = unrelated

    created = await feature.create_skill(
        name=name,
        description="Colliding procedural skill",
        body="Procedure.",
    )

    assert created["indexed"] is False
    assert feature.agent.storage.nodes[node_id] is unrelated
    assert feature.agent.storage.nodes[node_id].node_type == "episode"
    assert feature.agent.storage.nodes[node_id].properties == {
        "sentinel": "must survive"
    }
    assert name not in feature._indexed_names


@pytest.mark.asyncio
async def test_sqlite_graph_cas_preserves_non_skill_node_at_index_id(feature, tmp_path):
    name = "sqlite-graph-id-collision"
    agent_id = "did:test:sqlite-graph-id-collision"
    graph = AsyncGraphStore(feature.agent._raw_storage.db, agent_id=agent_id)
    agent = SimpleNamespace(
        did=agent_id,
        agent_id=agent_id,
        procedural_skills_root=tmp_path / "sqlite-graph-skills",
        _raw_storage=feature.agent._raw_storage,
        storage=graph,
    )
    indexed = ProceduralSkillsFeature(agent)
    node_id = indexed._node_id(name)
    unrelated = GraphNode(
        node_id=node_id,
        node_type="episode",
        label="unrelated SQLite graph data",
        properties={"agent_id": agent_id, "sentinel": "must survive"},
    )
    await graph.add_node(unrelated)
    await indexed.initialize()
    try:
        created = await indexed.create_skill(
            name=name,
            description="SQLite colliding procedural skill",
            body="Procedure.",
        )

        persisted = await graph.get_node(node_id)
        assert created["indexed"] is False
        assert persisted is not None
        assert persisted.node_type == "episode"
        assert persisted.label == "unrelated SQLite graph data"
        assert persisted.properties == {
            "agent_id": agent_id,
            "sentinel": "must survive",
        }
    finally:
        await indexed.shutdown()
        await graph.delete_node(node_id)


@pytest.mark.asyncio
async def test_graph_index_cas_does_not_overwrite_racing_identity_replacement(
    feature, monkeypatch
):
    name = "racing-index-identity"
    await feature.skill_create(name, "Original description", "body")
    node_id = feature._node_id(name)
    original_swap = feature.agent.storage.compare_and_swap_node
    replacement = GraphNode(
        node_id=node_id,
        node_type="episode",
        label="concurrent replacement",
        properties=dict(feature.agent.storage.nodes[node_id].properties),
    )

    async def replace_identity_before_swap(*args, **kwargs):
        feature.agent.storage.nodes[node_id] = replacement
        return await original_swap(*args, **kwargs)

    monkeypatch.setattr(
        feature.agent.storage,
        "compare_and_swap_node",
        replace_identity_before_swap,
    )
    folder = feature.agent.procedural_skills_root / name
    (folder / "SKILL.md").write_text(
        serialize_skill_markdown(SkillDocument(name, "Updated description", "body")),
        encoding="utf-8",
    )

    await feature.refresh()

    assert feature.agent.storage.nodes[node_id] is replacement
    assert replacement.node_type == "episode"
    assert replacement.label == "concurrent replacement"
    assert replacement.properties["description"] == "Original description"
    assert name not in feature._indexed_names


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
async def test_unchanged_refresh_does_not_reindex_graph_node(feature):
    await feature.skill_create("stable-index", "Stable graph index", "body")
    feature.agent.storage.added.clear()

    await feature.refresh()
    await feature.refresh()

    assert feature.agent.storage.added == []


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ("missing", "corrupt"))
async def test_unchanged_refresh_repairs_externally_damaged_graph_node(feature, damage):
    name = f"repair-{damage}-index"
    await feature.skill_create(name, "Repair graph index", "body")
    node_id = feature._node_id(name)
    feature.agent.storage.added.clear()
    if damage == "missing":
        feature.agent.storage.nodes.pop(node_id)
    else:
        feature.agent.storage.nodes[node_id].properties["description"] = "corrupt"

    await feature.refresh()

    assert len(feature.agent.storage.added) == 1
    repaired = feature.agent.storage.nodes[node_id]
    assert repaired.properties["description"] == "Repair graph index"
    assert name in feature._indexed_names


@pytest.mark.asyncio
async def test_unchanged_refresh_verifies_graph_nodes_in_one_batch(
    feature, monkeypatch
):
    for index in range(3):
        await feature.skill_create(
            f"batch-index-{index}",
            f"Batch graph index {index}",
            "body",
        )
    batch_calls = 0
    point_calls = 0
    original_batch = feature.agent.storage.get_nodes_by_type
    original_point = feature.agent.storage.get_node

    async def count_batch(node_type):
        nonlocal batch_calls
        batch_calls += 1
        return await original_batch(node_type)

    async def count_point(node_id):
        nonlocal point_calls
        point_calls += 1
        return await original_point(node_id)

    monkeypatch.setattr(feature.agent.storage, "get_nodes_by_type", count_batch)
    monkeypatch.setattr(feature.agent.storage, "get_node", count_point)

    await feature.refresh()

    assert batch_calls == 1
    assert point_calls == 0


@pytest.mark.asyncio
async def test_state_change_indexes_graph_once_despite_multiple_refreshes(feature):
    await feature.skill_create("changed-index", "Changed graph index", "body")
    feature.agent.storage.added.clear()

    result = await feature.skill_enable("changed-index", priority=8)

    assert result.status is ToolResultStatus.OK
    assert len(feature.agent.storage.added) == 1
    assert feature.agent.storage.added[0].properties["enabled"] is True
    assert feature.agent.storage.added[0].properties["priority"] == 8


@pytest.mark.asyncio
async def test_restart_uses_matching_persisted_graph_payload_without_upsert(feature):
    await feature.skill_create("restart-index", "Restart graph index", "body")
    feature.agent.storage.added.clear()

    replacement = ProceduralSkillsFeature(feature.agent)
    await replacement.initialize()
    try:
        assert feature.agent.storage.added == []
        assert "restart-index" in replacement._indexed_names
    finally:
        await replacement.shutdown()


@pytest.mark.asyncio
async def test_catalog_discovery_stays_pinned_when_root_ancestor_symlink_moves(
    feature, tmp_path
):
    first_parent = tmp_path / "first-parent"
    second_parent = tmp_path / "second-parent"
    first_parent.mkdir()
    second_parent.mkdir()
    current_parent = tmp_path / "current-parent"
    current_parent.symlink_to(first_parent, target_is_directory=True)
    agent = SimpleNamespace(
        did="did:test:pinned-skill-root",
        agent_id="did:test:pinned-skill-root",
        procedural_skills_root=current_parent / "skills",
        _raw_storage=feature.agent._raw_storage,
        storage=feature.agent.storage,
    )
    pinned = ProceduralSkillsFeature(agent)
    await pinned.initialize()
    try:
        created = await pinned.skill_create("from-first", "Pinned first root", "body")
        assert created.status is ToolResultStatus.OK

        (second_parent / "skills" / "from-second").mkdir(parents=True)
        (second_parent / "skills" / "from-second" / "SKILL.md").write_text(
            serialize_skill_markdown(
                SkillDocument("from-second", "Retargeted second root", "body")
            ),
            encoding="utf-8",
        )
        current_parent.unlink()
        current_parent.symlink_to(second_parent, target_is_directory=True)

        await pinned.refresh()

        assert "from-first" in pinned.snapshot.by_name()
        assert "from-second" not in pinned.snapshot.by_name()
        assert pinned._store.local_root == (first_parent / "skills").resolve()
    finally:
        await pinned.shutdown()


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
async def test_refresh_preserves_different_label_at_stale_skill_index_id(feature):
    name = "vanished-label-collision"
    await feature.skill_create(name, "Vanishing graph index", "body")
    node_id = feature._node_id(name)
    collision = GraphNode(
        node_id=node_id,
        node_type=PROCEDURAL_SKILL_NODE_TYPE,
        label="different-owner",
        properties={"name": name, "sentinel": "must survive"},
    )
    feature.agent.storage.nodes[node_id] = collision
    folder = feature.agent.procedural_skills_root / name
    for path in folder.iterdir():
        path.unlink()
    folder.rmdir()

    await feature.refresh()

    assert feature.agent.storage.nodes[node_id] is collision
    assert feature.agent.storage.nodes[node_id].properties == {
        "name": name,
        "sentinel": "must survive",
    }
    assert node_id not in feature.agent.storage.deleted


@pytest.mark.asyncio
async def test_refresh_does_not_delete_racing_identity_replacement(
    feature, monkeypatch
):
    name = "vanished-racing-replacement"
    await feature.skill_create(name, "Vanishing graph index", "body")
    node_id = feature._node_id(name)
    replacement = GraphNode(
        node_id=node_id,
        node_type="episode",
        label="concurrent refresh replacement",
        properties={"sentinel": "must survive refresh cleanup"},
    )
    original_delete = feature.agent.storage.delete_node
    original_compare_delete = feature.agent.storage.compare_and_delete_node

    async def race_unconditional_delete(candidate):
        feature.agent.storage.nodes[candidate] = replacement
        await original_delete(candidate)

    async def race_conditional_delete(*args, **kwargs):
        feature.agent.storage.nodes[node_id] = replacement
        return await original_compare_delete(*args, **kwargs)

    monkeypatch.setattr(feature.agent.storage, "delete_node", race_unconditional_delete)
    monkeypatch.setattr(
        feature.agent.storage,
        "compare_and_delete_node",
        race_conditional_delete,
    )
    folder = feature.agent.procedural_skills_root / name
    for path in folder.iterdir():
        path.unlink()
    folder.rmdir()

    await feature.refresh()

    assert feature.agent.storage.nodes[node_id] is replacement
    assert replacement.properties == {"sentinel": "must survive refresh cleanup"}
    assert node_id not in feature.agent.storage.deleted


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
async def test_create_enable_rejects_a_generation_changed_during_state_write(
    feature, monkeypatch
):
    name = "create-enable-generation-race"
    folder = feature.agent.procedural_skills_root / name
    replacement = serialize_skill_markdown(
        SkillDocument(
            name,
            "Unapproved replacement during create enable",
            "Unapproved replacement procedure.",
        )
    )
    original_set = feature._enablement.set
    replacement_published = False

    async def persist_then_replace(*args, **kwargs):
        nonlocal replacement_published
        state = await original_set(*args, **kwargs)
        if kwargs["enabled"] is True and not replacement_published:
            replacement_published = True
            (folder / "SKILL.md").write_text(replacement, encoding="utf-8")
        return state

    monkeypatch.setattr(feature._enablement, "set", persist_then_replace)

    with pytest.raises((SkillConflictError, SkillPathError), match="changed"):
        await feature.create_skill(
            name=name,
            description="Approved create generation",
            body="Approved procedure.",
            enabled=True,
            priority=19,
        )

    assert replacement_published is True
    assert (await feature._enablement.load())[name].enabled is False
    await feature.refresh()
    assert feature.snapshot.by_name()[name].state.enabled is False
    assert "Unapproved replacement during create enable" not in (
        feature.context_clause_text
    )


@pytest.mark.asyncio
async def test_create_enable_rejects_generation_marker_rotation_during_state_write(
    feature, monkeypatch
):
    name = "create-enable-marker-race"
    folder = feature.agent.procedural_skills_root / name
    original_set = feature._enablement.set
    marker_rotated = False

    async def persist_then_rotate_marker(*args, **kwargs):
        nonlocal marker_rotated
        state = await original_set(*args, **kwargs)
        if kwargs["enabled"] is True and not marker_rotated:
            marker_rotated = True
            (folder / store_module.GENERATION_FILENAME).write_text(
                f"kestrel-skill-generation-v1:{'0' * 64}\n",
                encoding="ascii",
            )
        return state

    monkeypatch.setattr(feature._enablement, "set", persist_then_rotate_marker)

    with pytest.raises(SkillPathError, match="generation changed"):
        await feature.create_skill(
            name=name,
            description="Approved marker generation",
            body="Approved procedure.",
            enabled=True,
        )

    assert marker_rotated is True
    assert (await feature._enablement.load())[name].enabled is False
    assert feature.snapshot.by_name()[name].state.enabled is False


@pytest.mark.asyncio
async def test_create_enable_guard_is_visible_to_another_feature_instance(
    feature, monkeypatch
):
    name = "create-enable-cross-instance-guard"
    final_refresh_started = asyncio.Event()
    release_final_refresh = asyncio.Event()
    original_refresh = feature._refresh_locked
    paused = False

    async def pause_final_refresh(**kwargs):
        nonlocal paused
        if not paused:
            paused = True
            final_refresh_started.set()
            await release_final_refresh.wait()
        return await original_refresh(**kwargs)

    monkeypatch.setattr(feature, "_refresh_locked", pause_final_refresh)
    creation = asyncio.create_task(
        feature.create_skill(
            name=name,
            description="Guarded create enable",
            body="Guarded create procedure.",
            enabled=True,
            priority=23,
        )
    )
    observer = ProceduralSkillsFeature(feature.agent)
    try:
        await asyncio.wait_for(final_refresh_started.wait(), timeout=5)
        assert (await feature._enablement.load())[name] == SkillState(True, 23)
        await observer.initialize()
        assert observer.snapshot.by_name()[name].state == SkillState(False, 23)
        assert "Guarded create enable" not in observer.context_clause_text

        release_final_refresh.set()
        created = await creation
        assert created["enabled"] is True
        await observer.refresh()
        assert observer.snapshot.by_name()[name].state == SkillState(True, 23)
        assert "Guarded create enable" in observer.context_clause_text
    finally:
        release_final_refresh.set()
        if not creation.done():
            creation.cancel()
            await asyncio.gather(creation, return_exceptions=True)
        await observer.shutdown()


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
    original_set = feature._enablement.set

    async def paused_state_write(*args, **kwargs):
        if kwargs["enabled"] is False:
            entered_state_write.set()
            await asyncio.Event().wait()
        return await original_set(*args, **kwargs)

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
async def test_duplicate_create_preserves_an_existing_durable_fail_closed_guard(
    feature,
):
    name = "duplicate-create-quarantined"
    await feature.skill_create(
        name,
        "Previously quarantined skill",
        "Quarantined procedure.",
        enabled=True,
        priority=17,
    )
    feature._retain_fail_closed_state(
        name,
        priority=17,
        error="prior state rollback remained uncertain",
    )
    await feature.refresh()
    assert feature.snapshot.by_name()[name].state == SkillState(False, 17)

    contender = ProceduralSkillsFeature(feature.agent)
    await contender.initialize()
    try:
        with pytest.raises(SkillConflictError, match="already exists"):
            await contender.create_skill(
                name=name,
                description="Duplicate must not clear quarantine",
                body="Duplicate procedure.",
            )

        assert (await contender._enablement.load())[name] == SkillState(True, 17)
        await contender.refresh()
        assert contender.snapshot.by_name()[name].state == SkillState(False, 17)
        assert "Previously quarantined skill" not in contender.context_clause_text
        assert name in contender._durable_fail_closed_names
    finally:
        await contender.shutdown()


@pytest.mark.asyncio
async def test_create_rollback_preserves_replacement_swapped_after_publication(
    feature, tmp_path, monkeypatch
):
    name = "create-rollback-replacement"
    published = feature.agent.procedural_skills_root / name
    displaced = tmp_path / "displaced-created-skill"
    marker = published / "replacement-must-survive.md"
    real_create = feature._store.create_pinned

    def create_then_swap(document):
        folder, identity = real_create(document)
        folder.rename(displaced)
        folder.mkdir()
        marker.write_text("replacement", encoding="utf-8")
        (folder / "SKILL.md").write_text(
            serialize_skill_markdown(
                SkillDocument(
                    name,
                    "Raced replacement must remain disabled",
                    "Unapproved procedure.",
                )
            ),
            encoding="utf-8",
        )
        return folder, identity

    original_set = feature._enablement.set

    async def commit_enable_then_fail(*args, **kwargs):
        state = await original_set(*args, **kwargs)
        if kwargs["enabled"] is True:
            raise DatabaseError("connection lost after enabling commit")
        return state

    monkeypatch.setattr(feature._store, "create_pinned", create_then_swap)
    monkeypatch.setattr(feature._enablement, "set", commit_enable_then_fail)

    result = await feature.skill_create(
        name,
        "Rollback must stay inode-pinned",
        "Procedure.",
        enabled=True,
    )

    assert result.status is ToolResultStatus.ERROR
    assert marker.read_text(encoding="utf-8") == "replacement"
    assert (displaced / "SKILL.md").is_file()
    assert (await feature._enablement.load())[name] == SkillState(
        False, DEFAULT_PRIORITY
    )
    assert feature.snapshot.by_name()[name].state.enabled is False
    assert "Raced replacement must remain disabled" not in feature.context_clause_text


@pytest.mark.asyncio
async def test_create_rollback_keeps_raced_replacement_disabled_over_prior_state(
    feature, tmp_path, monkeypatch
):
    name = "create-rollback-prior-enabled"
    prior_state = await feature._enablement.set(name, enabled=True, priority=17)
    feature._states[name] = prior_state
    published = feature.agent.procedural_skills_root / name
    displaced = tmp_path / "displaced-prior-enabled-skill"
    marker = published / "replacement-must-stay-disabled.md"
    real_create = feature._store.create_pinned

    def create_then_swap(document):
        folder, identity = real_create(document)
        folder.rename(displaced)
        folder.mkdir()
        marker.write_text("replacement", encoding="utf-8")
        (folder / "SKILL.md").write_text(
            serialize_skill_markdown(
                SkillDocument(
                    name,
                    "Raced replacement over prior enabled state",
                    "Unapproved procedure.",
                )
            ),
            encoding="utf-8",
        )
        return folder, identity

    original_set = feature._enablement.set
    enable_failure_injected = False

    async def commit_enable_then_fail_once(*args, **kwargs):
        nonlocal enable_failure_injected
        state = await original_set(*args, **kwargs)
        if kwargs["enabled"] is True and not enable_failure_injected:
            enable_failure_injected = True
            raise DatabaseError("connection lost after enabling commit")
        return state

    monkeypatch.setattr(feature._store, "create_pinned", create_then_swap)
    monkeypatch.setattr(feature._enablement, "set", commit_enable_then_fail_once)

    result = await feature.skill_create(
        name,
        "Rollback must not restore prior enablement",
        "Procedure.",
        enabled=True,
    )

    assert result.status is ToolResultStatus.ERROR
    assert marker.read_text(encoding="utf-8") == "replacement"
    assert (displaced / "SKILL.md").is_file()
    assert (await feature._enablement.load())[name] == SkillState(
        False, DEFAULT_PRIORITY
    )
    assert feature.snapshot.by_name()[name].state == SkillState(False, DEFAULT_PRIORITY)
    assert "Raced replacement over prior enabled state" not in (
        feature.context_clause_text
    )


@pytest.mark.asyncio
async def test_failed_create_publication_keeps_raced_replacement_disabled(
    feature, tmp_path, monkeypatch
):
    name = "create-publication-raced-replacement"
    prior_state = await feature._enablement.set(name, enabled=True, priority=17)
    feature._states[name] = prior_state
    published = feature.agent.procedural_skills_root / name
    displaced = tmp_path / "displaced-failed-create-publication"
    marker = published / "replacement-survived-failed-publication.md"

    def swap_then_fail(*_args, **_kwargs):
        candidates = list(
            feature._store._internal_root.glob(
                f"{store_module.SKILL_PUBLICATION_STAGING_PREFIX}create-*"
            )
        )
        assert len(candidates) == 1
        candidates[0].rename(displaced)
        published.mkdir()
        marker.write_text("replacement", encoding="utf-8")
        (published / "SKILL.md").write_text(
            serialize_skill_markdown(
                SkillDocument(
                    name,
                    "Unapproved replacement after failed create publication",
                    "Unapproved procedure.",
                )
            ),
            encoding="utf-8",
        )
        raise OSError("simulated create publication failure")

    monkeypatch.setattr(store_module, "_atomic_write_primary_at", swap_then_fail)

    result = await feature.skill_create(
        name,
        "Create publication must fail closed",
        "Procedure.",
        enabled=False,
    )

    assert result.status is ToolResultStatus.ERROR
    assert marker.read_text(encoding="utf-8") == "replacement"
    assert displaced.is_dir()
    assert (await feature._enablement.load())[name] == SkillState(
        False, DEFAULT_PRIORITY
    )
    assert feature.snapshot.by_name()[name].state == SkillState(False, DEFAULT_PRIORITY)
    assert "Unapproved replacement after failed create publication" not in (
        feature.context_clause_text
    )


@pytest.mark.asyncio
async def test_create_conflict_keeps_raced_replacement_disabled(feature, monkeypatch):
    name = "create-conflict-raced-replacement"
    prior_state = await feature._enablement.set(name, enabled=True, priority=17)
    feature._states[name] = prior_state
    published = feature.agent.procedural_skills_root / name

    def publish_replacement_then_conflict(_document):
        published.mkdir()
        (published / "SKILL.md").write_text(
            serialize_skill_markdown(
                SkillDocument(
                    name,
                    "Unapproved replacement after create conflict",
                    "Unapproved procedure.",
                )
            ),
            encoding="utf-8",
        )
        raise SkillConflictError("a non-cooperating writer won publication")

    monkeypatch.setattr(
        feature._store,
        "create_pinned",
        publish_replacement_then_conflict,
    )

    result = await feature.skill_create(name, "Requested skill", "Procedure.")

    assert result.status is ToolResultStatus.ERROR
    assert (await feature._enablement.load())[name] == SkillState(
        False, DEFAULT_PRIORITY
    )
    assert feature.snapshot.by_name()[name].state == SkillState(False, DEFAULT_PRIORITY)
    assert "Unapproved replacement after create conflict" not in (
        feature.context_clause_text
    )


@pytest.mark.asyncio
async def test_create_rollback_preserves_concurrent_same_inode_resource(
    feature, monkeypatch
):
    name = "create-rollback-concurrent-resource"
    enablement_started = asyncio.Event()
    release_enablement = asyncio.Event()
    original_set = feature._enablement.set

    async def pause_after_enable_commit(*args, **kwargs):
        state = await original_set(*args, **kwargs)
        if kwargs["enabled"] is True:
            enablement_started.set()
            await release_enablement.wait()
            raise DatabaseError("connection lost after enabling commit")
        return state

    monkeypatch.setattr(feature._enablement, "set", pause_after_enable_commit)
    creation = asyncio.create_task(
        feature.skill_create(
            name,
            "Concurrent resource must survive rollback",
            "Procedure.",
            enabled=True,
        )
    )
    await asyncio.wait_for(enablement_started.wait(), timeout=5)

    source = DirectorySkillSource(
        root=feature.agent.procedural_skills_root,
        source_id="agent-local",
        kind="agent-local",
        precedence=AGENT_LOCAL_PRECEDENCE,
    )
    records, errors = source.discover()
    assert errors == ()
    feature._store.write_file(records[0], "notes.md", "Concurrent resource.\n")
    release_enablement.set()

    result = await creation

    assert result.status is ToolResultStatus.ERROR
    folder = feature.agent.procedural_skills_root / name
    assert (folder / "notes.md").read_text(encoding="utf-8") == (
        "Concurrent resource.\n"
    )
    assert (await feature._enablement.load())[name] == SkillState(
        False, DEFAULT_PRIORITY
    )
    assert feature.snapshot.by_name()[name].state.enabled is False
    assert "Concurrent resource must survive" not in feature.context_clause_text


@pytest.mark.asyncio
async def test_cancelled_create_drains_enablement_restore_after_publication_failure(
    feature, monkeypatch
):
    name = "cancelled-create-restore"
    original_set = feature._enablement.set
    original_delete = feature._enablement.delete
    restoration_started = asyncio.Event()
    release_restoration = asyncio.Event()

    async def commit_enable_then_fail(*args, **kwargs):
        state = await original_set(*args, **kwargs)
        if kwargs["enabled"] is True:
            raise DatabaseError("connection lost after enabling commit")
        return state

    async def paused_restore(candidate):
        restoration_started.set()
        await release_restoration.wait()
        return await original_delete(candidate)

    monkeypatch.setattr(feature._enablement, "set", commit_enable_then_fail)
    monkeypatch.setattr(feature._enablement, "delete", paused_restore)
    creation = asyncio.create_task(
        feature.create_skill(
            name=name,
            description="Cancelled restoration",
            body="Procedure.",
            enabled=True,
        )
    )
    await asyncio.wait_for(restoration_started.wait(), timeout=5)
    creation.cancel()
    await asyncio.sleep(0)
    try:
        assert not creation.done(), "cancellation skipped owned enablement cleanup"
    finally:
        release_restoration.set()

    with pytest.raises(DatabaseError, match="connection lost"):
        await creation
    assert name not in await feature._enablement.load()
    assert not (feature.agent.procedural_skills_root / name).exists()
    assert name not in feature.snapshot.by_name()


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

    monkeypatch.setattr(feature._store, "create_pinned", fail_publication)

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
    async def fail(_node_id, _expected, _node, **_kwargs):
        raise UnexpectedGraphError("graph unavailable")

    monkeypatch.setattr(feature.agent.storage, "compare_and_swap_node", fail)
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
    approved_revision = delete_revision(feature, "partial-delete")

    async def fail(_node_id, **_kwargs):
        raise UnexpectedGraphError("graph unavailable")

    monkeypatch.setattr(feature.agent.storage, "compare_and_delete_node", fail)
    result = await feature.skill_delete("partial-delete", approved_revision)

    assert result.status is ToolResultStatus.PARTIAL
    assert not folder.exists()
    assert result.data["removed_file"] is True
    assert result.data["graph_deleted"] is False
    assert "graph index cleanup failed" in result.error


@pytest.mark.asyncio
async def test_delete_tool_rejects_a_replacement_created_after_approval(feature):
    name = "tool-stale-delete"
    await feature.skill_create(name, "Approved generation", "Approved procedure.")
    approved_revision = delete_revision(feature, name)
    await feature.delete_skill(name=name, expected_revision=approved_revision)
    await feature.skill_create(name, "Replacement generation", "Keep this procedure.")

    result = await feature.skill_delete(
        name,
        delete_revision=approved_revision,
    )

    assert result.status is ToolResultStatus.ERROR
    assert "changed before deletion" in result.error
    assert feature.snapshot.by_name()[name].document.description == (
        "Replacement generation"
    )


@pytest.mark.asyncio
async def test_delete_tool_rejects_non_ascii_revision_as_a_tool_error(feature):
    name = "tool-invalid-delete-revision"
    await feature.skill_create(name, "Keep this generation", "Keep this procedure.")

    result = await feature.skill_delete(name, delete_revision="é" * 64)

    assert result.status is ToolResultStatus.ERROR
    assert "revision" in result.error
    assert feature.snapshot.by_name()[name].document.description == (
        "Keep this generation"
    )


@pytest.mark.asyncio
async def test_delete_reconciles_graph_cleanup_completed_by_final_refresh(
    feature, monkeypatch
):
    name = "refresh-reconciled-graph-delete"
    await feature.skill_create(name, "Refresh reconciled graph delete", "body")
    approved_revision = delete_revision(feature, name)
    node_id = feature._node_id(name)
    original_delete = feature.agent.storage.compare_and_delete_node
    attempts = 0

    async def fail_once_then_delete(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise UnexpectedGraphError("graph unavailable once")
        return await original_delete(*args, **kwargs)

    monkeypatch.setattr(
        feature.agent.storage,
        "compare_and_delete_node",
        fail_once_then_delete,
    )

    result = await feature.skill_delete(name, approved_revision)

    assert attempts >= 2
    assert node_id not in feature.agent.storage.nodes
    assert result.status is ToolResultStatus.OK
    assert result.data["graph_deleted"] is True
    assert result.data["errors"] == []


@pytest.mark.asyncio
async def test_delete_preserves_fallback_graph_index_repaired_by_final_refresh(
    feature, tmp_path, monkeypatch
):
    name = "refresh-repaired-fallback"
    await feature.skill_create(name, "Local skill", "Local procedure.")
    shared_root = tmp_path / "shared"
    shared_folder = shared_root / name
    shared_folder.mkdir(parents=True)
    (shared_folder / "SKILL.md").write_text(
        serialize_skill_markdown(
            SkillDocument(name, "Shared fallback", "Shared procedure.")
        ),
        encoding="utf-8",
    )
    feature._catalog = SkillCatalog(
        (
            DirectorySkillSource(
                root=feature.agent.procedural_skills_root,
                source_id="agent-local",
                kind="agent-local",
                precedence=AGENT_LOCAL_PRECEDENCE,
            ),
            DirectorySkillSource(
                root=shared_root,
                source_id="host-shared",
                kind="host-shared",
                precedence=HOST_SHARED_PRECEDENCE,
            ),
        )
    )
    await feature.refresh()
    approved_revision = delete_revision(feature, name)
    original_delete = feature.agent.storage.compare_and_delete_node
    attempts = 0

    async def fail_once_then_delete(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise UnexpectedGraphError("graph unavailable once")
        return await original_delete(*args, **kwargs)

    monkeypatch.setattr(
        feature.agent.storage,
        "compare_and_delete_node",
        fail_once_then_delete,
    )

    result = await feature.skill_delete(name, approved_revision)

    fallback = feature.snapshot.by_name()[name]
    node = feature.agent.storage.nodes[feature._node_id(name)]
    assert fallback.source_kind == "host-shared"
    assert node.properties["source_id"] == "host-shared"
    assert name in feature._indexed_names
    assert result.status is ToolResultStatus.OK
    assert result.data["graph_deleted"] is False
    assert result.data["graph_retained"] is True
    assert result.data["errors"] == []


@pytest.mark.asyncio
async def test_delete_preserves_different_label_at_skill_index_id(feature):
    name = "delete-label-collision"
    await feature.skill_create(name, "Delete graph collision", "body")
    node_id = feature._node_id(name)
    collision = GraphNode(
        node_id=node_id,
        node_type=PROCEDURAL_SKILL_NODE_TYPE,
        label="different-owner",
        properties={"name": name, "sentinel": "must survive"},
    )
    feature.agent.storage.nodes[node_id] = collision
    approved_revision = delete_revision(feature, name)

    result = await feature.skill_delete(name, approved_revision)

    assert result.status is ToolResultStatus.OK
    assert result.data["graph_deleted"] is True
    assert feature.agent.storage.nodes[node_id] is collision
    assert feature.agent.storage.nodes[node_id].properties == {
        "name": name,
        "sentinel": "must survive",
    }
    assert node_id not in feature.agent.storage.deleted


@pytest.mark.asyncio
async def test_delete_does_not_remove_racing_identity_replacement(feature, monkeypatch):
    name = "delete-racing-replacement"
    await feature.skill_create(name, "Delete graph race", "body")
    node_id = feature._node_id(name)
    replacement = GraphNode(
        node_id=node_id,
        node_type="episode",
        label="concurrent delete replacement",
        properties={"sentinel": "must survive delete"},
    )
    original_delete = feature.agent.storage.delete_node
    original_compare_delete = feature.agent.storage.compare_and_delete_node

    async def race_unconditional_delete(candidate):
        feature.agent.storage.nodes[candidate] = replacement
        await original_delete(candidate)

    async def race_conditional_delete(*args, **kwargs):
        feature.agent.storage.nodes[node_id] = replacement
        return await original_compare_delete(*args, **kwargs)

    monkeypatch.setattr(feature.agent.storage, "delete_node", race_unconditional_delete)
    monkeypatch.setattr(
        feature.agent.storage,
        "compare_and_delete_node",
        race_conditional_delete,
    )
    approved_revision = delete_revision(feature, name)

    result = await feature.skill_delete(name, approved_revision)

    assert result.status is ToolResultStatus.OK
    assert result.data["graph_deleted"] is True
    assert feature.agent.storage.nodes[node_id] is replacement
    assert replacement.properties == {"sentinel": "must survive delete"}
    assert node_id not in feature.agent.storage.deleted


@pytest.mark.asyncio
async def test_delete_reconciles_enablement_cleanup_that_committed_before_error(
    feature, monkeypatch
):
    name = "ambiguous-delete"
    await feature.skill_create(name, "Ambiguous delete", "body", enabled=True)
    original_delete = feature._enablement.delete

    async def commit_then_disconnect(candidate):
        await original_delete(candidate)
        raise DatabaseError("connection lost after delete commit")

    monkeypatch.setattr(feature._enablement, "delete", commit_then_disconnect)
    approved_revision = delete_revision(feature, name)

    result = await feature.skill_delete(name, approved_revision)

    assert result.status is ToolResultStatus.OK
    assert result.data["config_deleted"] is True
    assert result.data["errors"] == []
    assert name not in await feature._enablement.load()


@pytest.mark.asyncio
async def test_delete_uses_final_refresh_to_reconcile_ambiguous_cleanup(
    feature, monkeypatch
):
    name = "refresh-reconciled-delete"
    await feature.skill_create(name, "Refresh reconciled delete", "body", enabled=True)
    original_delete = feature._enablement.delete
    original_load = feature._enablement.load
    post_commit = False
    refused_reconciliation = False

    async def commit_then_disconnect(candidate):
        nonlocal post_commit
        await original_delete(candidate)
        post_commit = True
        raise DatabaseError("connection lost after delete commit")

    async def fail_first_post_commit_load():
        nonlocal refused_reconciliation
        if post_commit and not refused_reconciliation:
            refused_reconciliation = True
            raise DatabaseError("read connection also reset once")
        return await original_load()

    monkeypatch.setattr(feature._enablement, "delete", commit_then_disconnect)
    monkeypatch.setattr(feature._enablement, "load", fail_first_post_commit_load)
    approved_revision = delete_revision(feature, name)

    result = await feature.skill_delete(name, approved_revision)

    assert refused_reconciliation is True
    assert result.status is ToolResultStatus.OK
    assert result.data["config_deleted"] is True
    assert result.data["errors"] == []
    assert name not in await original_load()


@pytest.mark.asyncio
async def test_delete_reports_enablement_cleanup_that_failed_before_commit(
    feature, monkeypatch
):
    name = "failed-delete-state"
    await feature.skill_create(name, "Failed state delete", "body", enabled=True)

    async def disconnect_before_commit(_candidate):
        raise DatabaseError("connection lost before delete commit")

    monkeypatch.setattr(feature._enablement, "delete", disconnect_before_commit)
    approved_revision = delete_revision(feature, name)

    result = await feature.skill_delete(name, approved_revision)

    assert result.status is ToolResultStatus.PARTIAL
    assert result.data["config_deleted"] is False
    assert "enablement row cleanup failed" in result.error
    assert name in await feature._enablement.load()


@pytest.mark.asyncio
async def test_delete_failure_cannot_enable_a_same_named_shared_fallback(
    feature, tmp_path, monkeypatch
):
    name = "shadowed-delete"
    shared_root = tmp_path / "shared"
    shared_folder = shared_root / name
    shared_folder.mkdir(parents=True)
    (shared_folder / "SKILL.md").write_text(
        serialize_skill_markdown(
            SkillDocument(name, "Unapproved shared fallback", "Shared procedure")
        ),
        encoding="utf-8",
    )
    feature._catalog = SkillCatalog(
        (
            DirectorySkillSource(
                root=feature.agent.procedural_skills_root,
                source_id="agent-local",
                kind="agent-local",
                precedence=AGENT_LOCAL_PRECEDENCE,
            ),
            DirectorySkillSource(
                root=shared_root,
                source_id="host-shared",
                kind="host-shared",
                precedence=HOST_SHARED_PRECEDENCE,
            ),
        )
    )
    await feature.skill_create(name, "Approved local skill", "Local procedure")
    await feature.skill_enable(name, priority=13)

    async def fail_delete(_name):
        raise DatabaseError("database cleanup offline")

    monkeypatch.setattr(feature._enablement, "delete", fail_delete)
    approved_revision = delete_revision(feature, name)
    result = await feature.skill_delete(name, approved_revision)

    assert result.status is ToolResultStatus.PARTIAL
    fallback = feature.snapshot.by_name()[name]
    assert fallback.source_kind == "host-shared"
    assert fallback.state.enabled is False
    assert "Unapproved shared fallback" not in feature.context_clause_text


@pytest.mark.asyncio
async def test_failed_folder_removal_retains_disabled_tombstone(feature, monkeypatch):
    name = "failed-folder-delete"
    await feature.skill_create(name, "Must not remain enabled", "Procedure")
    await feature.skill_enable(name, priority=23)

    def fail_removal(_record):
        raise OSError("recursive removal failed")

    monkeypatch.setattr(feature._store, "delete", fail_removal)
    approved_revision = delete_revision(feature, name)
    result = await feature.skill_delete(name, approved_revision)

    assert result.status is ToolResultStatus.ERROR
    assert "remains disabled" in result.error
    assert (await feature._enablement.load())[name] == SkillState(False, 23)
    assert feature.snapshot.by_name()[name].state == SkillState(False, 23)
    assert "Must not remain enabled" not in feature.context_clause_text


@pytest.mark.asyncio
async def test_cancelled_disabled_guard_restores_enabled_shared_state(
    feature, tmp_path, monkeypatch
):
    name = "cancelled-local-override"
    shared_root = tmp_path / "shared-cancelled"
    shared_folder = shared_root / name
    shared_folder.mkdir(parents=True)
    (shared_folder / "SKILL.md").write_text(
        serialize_skill_markdown(
            SkillDocument(name, "Approved shared skill", "Shared procedure")
        ),
        encoding="utf-8",
    )
    feature._catalog = SkillCatalog(
        (
            DirectorySkillSource(
                root=feature.agent.procedural_skills_root,
                source_id="agent-local",
                kind="agent-local",
                precedence=AGENT_LOCAL_PRECEDENCE,
            ),
            DirectorySkillSource(
                root=shared_root,
                source_id="host-shared",
                kind="host-shared",
                precedence=HOST_SHARED_PRECEDENCE,
            ),
        )
    )
    await feature._enablement.set(name, enabled=True, priority=17)
    await feature.refresh()
    original_set = feature._enablement.set
    guard_committed = asyncio.Event()
    release_guard = asyncio.Event()

    async def commit_guard_then_pause(*args, **kwargs):
        state = await original_set(*args, **kwargs)
        if kwargs["enabled"] is False:
            guard_committed.set()
            await release_guard.wait()
        return state

    monkeypatch.setattr(feature._enablement, "set", commit_guard_then_pause)
    creation = asyncio.create_task(
        feature.create_skill(
            name=name,
            description="Cancelled local replacement",
            body="Local procedure",
        )
    )
    await asyncio.wait_for(guard_committed.wait(), timeout=5)
    creation.cancel()
    release_guard.set()
    with pytest.raises(asyncio.CancelledError):
        await creation

    persisted = (await feature._enablement.load())[name]
    assert persisted == SkillState(True, 17)
    assert not (feature.agent.procedural_skills_root / name).exists()
    resolved = feature.snapshot.by_name()[name]
    assert resolved.source_kind == "host-shared"
    assert resolved.state == SkillState(True, 17)
    assert "Approved shared skill" in feature.context_clause_text


@pytest.mark.asyncio
async def test_invalid_frontmatter_edit_is_rejected_before_replace(feature):
    await feature.skill_create("edit-me", "Original", "body")
    path = feature.agent.procedural_skills_root / "edit-me" / "SKILL.md"
    original = path.read_text()
    result = await feature.skill_edit("edit-me", "---\nname: edit-me\n---\nbody")
    assert result.status is ToolResultStatus.ERROR
    assert path.read_text() == original


@pytest.mark.asyncio
async def test_edit_rejects_stale_git_local_record_when_host_source_wins(
    feature, tmp_path, monkeypatch
):
    name = "host-wins-before-edit"
    shared_root = tmp_path / "shared-edit-race"
    shared_root.mkdir()
    monkeypatch.setenv("KESTREL_SHARED_SKILLS_DIR", str(shared_root))
    editor = ProceduralSkillsFeature(feature.agent)
    await editor.initialize()
    try:
        await editor.create_skill(
            name=name,
            description="Git-local original",
            body="Original local procedure.",
        )
        local_folder = feature.agent.procedural_skills_root / name
        (local_folder / PROVENANCE_FILENAME).write_bytes(
            serialize_provenance(
                SkillProvenance(
                    kind="git",
                    source_id="https://example.com/origin.git",
                    locator=f"main:{name}",
                    revision="a" * 40,
                    remote_url="https://example.com/origin.git",
                )
            )
        )
        await editor.refresh()
        original = (local_folder / "SKILL.md").read_text(encoding="utf-8")
        assert editor.snapshot.by_name()[name].provenance.kind == "git"

        shared_folder = shared_root / name
        shared_folder.mkdir()
        (shared_folder / "SKILL.md").write_text(
            serialize_skill_markdown(
                SkillDocument(name, "New host winner", "Host procedure.")
            ),
            encoding="utf-8",
        )
        replacement = serialize_skill_markdown(
            SkillDocument(name, "Stale local edit", "Edited local procedure.")
        )

        with pytest.raises(SkillConflictError, match="resolved source changed"):
            await editor.edit_skill(
                name=name,
                relative_path="SKILL.md",
                content=replacement,
            )

        assert (local_folder / "SKILL.md").read_text(encoding="utf-8") == original
        assert editor.snapshot.by_name()[name].document.description == "New host winner"
    finally:
        await editor.shutdown()


@pytest.mark.asyncio
async def test_revision_guarded_edit_accepts_skill_published_by_another_instance(
    feature,
):
    editor = ProceduralSkillsFeature(feature.agent)
    await editor.initialize()
    try:
        created = await feature.create_skill(
            name="cross-instance-new",
            description="Published after the other instance initialized",
            body="Original body.",
        )

        edited = await editor.edit_skill(
            name="cross-instance-new",
            relative_path="SKILL.md",
            content=serialize_skill_markdown(
                SkillDocument(
                    "cross-instance-new",
                    "Edited through the refreshed revision",
                    "Edited body.",
                )
            ),
            expected_revision=created["revision"],
        )

        assert edited["revision"] != created["revision"]
        assert (
            editor.snapshot.by_name()["cross-instance-new"].document.description
            == "Edited through the refreshed revision"
        )
    finally:
        await editor.shutdown()


@pytest.mark.asyncio
async def test_revision_guarded_edit_accepts_new_revision_from_another_instance(
    feature,
):
    created = await feature.create_skill(
        name="cross-instance-update",
        description="Original description",
        body="Original body.",
    )
    editor = ProceduralSkillsFeature(feature.agent)
    await editor.initialize()
    try:
        updated = await feature.edit_skill(
            name="cross-instance-update",
            relative_path="SKILL.md",
            content=serialize_skill_markdown(
                SkillDocument(
                    "cross-instance-update",
                    "Updated by the first instance",
                    "First update.",
                )
            ),
            expected_revision=created["revision"],
        )

        edited = await editor.edit_skill(
            name="cross-instance-update",
            relative_path="SKILL.md",
            content=serialize_skill_markdown(
                SkillDocument(
                    "cross-instance-update",
                    "Edited by the stale instance",
                    "Second update.",
                )
            ),
            expected_revision=updated["revision"],
        )

        assert edited["revision"] != updated["revision"]
        assert (
            editor.snapshot.by_name()["cross-instance-update"].document.description
            == "Edited by the stale instance"
        )
    finally:
        await editor.shutdown()


@pytest.mark.asyncio
async def test_edit_reports_cached_skill_that_became_invalid(feature):
    name = "invalid-before-edit"
    await feature.create_skill(
        name=name,
        description="Valid before external growth",
        body="Procedure.",
    )
    folder = feature.agent.procedural_skills_root / name
    for index in range(256):
        (folder / f"external-{index:03}.md").write_text("x", encoding="utf-8")

    with pytest.raises(SkillFormatError, match="changed or became invalid"):
        await feature.edit_skill(
            name=name,
            relative_path="notes.md",
            content="must not publish",
        )

    assert not (folder / "notes.md").exists()


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
    refresh_calls = 0

    async def paused_refresh():
        nonlocal refresh_calls
        refresh_calls += 1
        if refresh_calls == 1:
            return await original_refresh()
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
        result = await other.skill_delete("shared-only", "0" * 64)
        assert result.status is ToolResultStatus.ERROR
        assert "local override" in result.error
        assert folder.is_dir()
    finally:
        await other.shutdown()


@pytest.mark.asyncio
async def test_delete_removes_shadowed_local_git_installation(
    feature, tmp_path, monkeypatch
):
    name = "shadowed-git-delete"
    local = feature.agent.procedural_skills_root / name
    local.mkdir()
    (local / "SKILL.md").write_text(
        serialize_skill_markdown(
            SkillDocument(name, "Hidden Git installation", "Local procedure.")
        ),
        encoding="utf-8",
    )
    (local / PROVENANCE_FILENAME).write_bytes(
        serialize_provenance(
            SkillProvenance(
                kind="git",
                source_id="https://example.com/skills.git",
                locator=f"main:{name}",
                revision="a" * 40,
                remote_url="https://example.com/skills.git",
            )
        )
    )
    shared = tmp_path / "shared-shadow"
    shared_skill = shared / name
    shared_skill.mkdir(parents=True)
    (shared_skill / "SKILL.md").write_text(
        serialize_skill_markdown(
            SkillDocument(name, "Visible host skill", "Host procedure.")
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("KESTREL_SHARED_SKILLS_DIR", str(shared))
    other = ProceduralSkillsFeature(feature.agent)
    await other.initialize()
    try:
        assert other.snapshot.by_name()[name].source_kind == "host-shared"
        catalog_record = next(
            item for item in other.catalog_payload()["skills"] if item["name"] == name
        )
        assert catalog_record["editable"] is False
        assert catalog_record["deletable"] is True
        enabled = await other.skill_enable(name, priority=17)
        assert enabled.status is ToolResultStatus.OK
        other.agent.storage.added.clear()
        other.agent.storage.deleted.clear()
        approved_revision = delete_revision(other, name)

        result = await other.skill_delete(name, approved_revision)

        assert result.status is ToolResultStatus.OK
        assert "host-shared source remains resolved" in result.confirmation
        assert not local.exists()
        assert shared_skill.is_dir()
        remaining = other.snapshot.by_name()[name]
        assert remaining.source_kind == "host-shared"
        assert remaining.state == SkillState(True, 17)
        assert other.agent.storage.added == []
        assert other.agent.storage.deleted == []
    finally:
        await other.shutdown()


@pytest.mark.asyncio
async def test_shadowed_delete_succeeds_when_host_winner_disappears(
    feature, tmp_path, monkeypatch
):
    name = "shadowed-delete-host-race"
    local = feature.agent.procedural_skills_root / name
    local.mkdir()
    (local / "SKILL.md").write_text(
        serialize_skill_markdown(
            SkillDocument(name, "Hidden Git installation", "Local procedure.")
        ),
        encoding="utf-8",
    )
    (local / PROVENANCE_FILENAME).write_bytes(
        serialize_provenance(
            SkillProvenance(
                kind="git",
                source_id="https://example.com/skills.git",
                locator=f"main:{name}",
                revision="a" * 40,
                remote_url="https://example.com/skills.git",
            )
        )
    )
    shared = tmp_path / "shared-delete-race"
    shared_skill = shared / name
    shared_skill.mkdir(parents=True)
    (shared_skill / "SKILL.md").write_text(
        serialize_skill_markdown(
            SkillDocument(name, "Transient host winner", "Host procedure.")
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("KESTREL_SHARED_SKILLS_DIR", str(shared))
    other = ProceduralSkillsFeature(feature.agent)
    await other.initialize()
    real_delete = other._store.delete

    def delete_local_as_host_disappears(record):
        real_delete(record)
        (shared_skill / "SKILL.md").unlink()
        shared_skill.rmdir()

    monkeypatch.setattr(other._store, "delete", delete_local_as_host_disappears)
    try:
        approved_revision = delete_revision(other, name)
        result = await other.skill_delete(name, approved_revision)

        assert result.status is ToolResultStatus.OK
        assert result.data["removed_file"] is True
        assert result.data["resolved_skill_retained"] is False
        assert result.data["remaining_source_kind"] is None
        assert not local.exists()
        assert not shared_skill.exists()
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
    assert result.data["source_revision"] == "b" * 40
    assert len(result.data["revision"]) == 64
    assert result.data["enabled"] is False
    await feature.refresh()
    record = feature.snapshot.by_name()["remote"]
    assert result.data["revision"] == record.revision
    assert record.provenance.revision == "b" * 40
    assert record.provenance.remote_url == "https://example.com/repo.git"
    assert record.state.enabled is False
    edited = await feature.edit_skill(
        name="remote",
        relative_path="SKILL.md",
        content=serialize_skill_markdown(
            SkillDocument("remote", "Edited after install", "body")
        ),
        expected_revision=result.data["revision"],
    )
    assert edited["revision"] != result.data["revision"]
    assert feature.snapshot.by_name()["remote"].document.description == (
        "Edited after install"
    )


@pytest.mark.asyncio
async def test_git_install_uses_recoverable_internal_checkout_workspace(
    feature, monkeypatch
):
    observed_workspaces = []

    async def fake_checkout(*, source_url, ref, skill_name, target):
        workspace = target.parent
        observed_workspaces.append(workspace)
        assert workspace.parent == feature._store._internal_root
        source = target / skill_name
        source.mkdir(parents=True)
        (source / "SKILL.md").write_text(
            serialize_skill_markdown(
                SkillDocument(skill_name, "Recoverable checkout", "Procedure.")
            ),
            encoding="utf-8",
        )
        return GitCheckout(
            root=target,
            skill_folder=source,
            revision="a" * 40,
            remote_url=source_url,
            ref=ref,
        )

    monkeypatch.setattr(feature, "_checkout_git_until_stopped", fake_checkout)

    installed = await feature.install_skill(
        source_url="https://example.com/repo.git",
        skill_name="recoverable-checkout",
        ref="main",
    )

    assert installed["enabled"] is False
    assert len(observed_workspaces) == 1
    assert not observed_workspaces[0].exists()


@pytest.mark.asyncio
async def test_git_install_compensates_when_checkout_workspace_cleanup_fails(
    feature, monkeypatch
):
    name = "cleanup-failed-install"
    observed_workspaces = []

    def cleanup_fails_after_publication(_name, *, expected):
        assert expected
        raise OSError("simulated checkout workspace cleanup failure")

    async def fake_checkout(*, source_url, ref, skill_name, target):
        observed_workspaces.append(target.parent)
        source = target / skill_name
        source.mkdir(parents=True)
        (source / "SKILL.md").write_text(
            serialize_skill_markdown(
                SkillDocument(skill_name, "Cleanup failure", "Procedure.")
            ),
            encoding="utf-8",
        )
        return GitCheckout(
            root=target,
            skill_folder=source,
            revision="a" * 40,
            remote_url=source_url,
            ref=ref,
        )

    monkeypatch.setattr(
        feature._store,
        "_remove_git_checkout_workspace",
        cleanup_fails_after_publication,
    )
    monkeypatch.setattr(feature, "_checkout_git_until_stopped", fake_checkout)

    with pytest.raises(OSError, match="checkout workspace cleanup failure"):
        await feature.install_skill(
            source_url="https://example.com/repo.git",
            skill_name=name,
            ref="main",
        )

    assert not (feature.agent.procedural_skills_root / name).exists()
    assert name not in await feature._enablement.load()
    assert name not in feature.snapshot.by_name()
    assert len(observed_workspaces) == 1
    assert observed_workspaces[0].is_dir()

    store_module.SkillStore(feature.agent.procedural_skills_root)

    assert not observed_workspaces[0].exists()


@pytest.mark.asyncio
async def test_git_install_holds_name_claim_through_cleanup_compensation(
    feature, monkeypatch
):
    name = "cleanup-claim-install"
    rollback_started = asyncio.Event()
    release_rollback = asyncio.Event()
    rival = ProceduralSkillsFeature(feature.agent)
    await rival.initialize()
    original_rollback = feature._rollback_installed_publication

    def cleanup_fails_after_publication(_name, *, expected):
        assert expected
        raise OSError("simulated checkout workspace cleanup failure")

    async def pause_before_rollback(**kwargs):
        rollback_started.set()
        await release_rollback.wait()
        return await original_rollback(**kwargs)

    async def fake_checkout(*, source_url, ref, skill_name, target):
        source = target / skill_name
        source.mkdir(parents=True)
        (source / "SKILL.md").write_text(
            serialize_skill_markdown(
                SkillDocument(skill_name, "Cleanup claim", "Procedure.")
            ),
            encoding="utf-8",
        )
        return GitCheckout(
            root=target,
            skill_folder=source,
            revision="a" * 40,
            remote_url=source_url,
            ref=ref,
        )

    monkeypatch.setattr(
        feature._store,
        "_remove_git_checkout_workspace",
        cleanup_fails_after_publication,
    )
    monkeypatch.setattr(feature, "_checkout_git_until_stopped", fake_checkout)
    monkeypatch.setattr(
        feature, "_rollback_installed_publication", pause_before_rollback
    )

    installation = asyncio.create_task(
        feature.install_skill(
            source_url="https://example.com/repo.git",
            skill_name=name,
            ref="main",
        )
    )
    await asyncio.wait_for(rollback_started.wait(), timeout=5)
    state_update = asyncio.create_task(
        rival.set_skill_state(name=name, enabled=True, priority=23)
    )
    await asyncio.sleep(0.1)
    state_update_waited = not state_update.done()
    release_rollback.set()
    try:
        with pytest.raises(OSError, match="checkout workspace cleanup failure"):
            await installation
        if state_update_waited:
            with pytest.raises(SkillNotFoundError):
                await state_update
        else:
            await state_update
    finally:
        release_rollback.set()
        await asyncio.gather(installation, state_update, return_exceptions=True)
        await rival.shutdown()
        store_module.SkillStore(feature.agent.procedural_skills_root)

    assert state_update_waited, "state update passed install cleanup compensation"
    assert not (feature.agent.procedural_skills_root / name).exists()
    assert name not in await feature._enablement.load()


@pytest.mark.asyncio
async def test_git_install_refreshes_a_stale_conflict_before_checkout(
    feature, monkeypatch
):
    name = "stale-install-conflict"
    await feature.skill_create(name, "Removed local skill", "Old procedure.")
    stale_folder = feature.agent.procedural_skills_root / name
    for path in stale_folder.iterdir():
        path.unlink()
    stale_folder.rmdir()

    async def fake_checkout(*, source_url, ref, skill_name, target):
        source = target / skill_name
        source.mkdir(parents=True)
        (source / "SKILL.md").write_text(
            serialize_skill_markdown(
                SkillDocument(skill_name, "Fresh remote skill", "New procedure.")
            ),
            encoding="utf-8",
        )
        return GitCheckout(
            root=target,
            skill_folder=source,
            revision="a" * 40,
            remote_url=source_url,
            ref=ref,
        )

    monkeypatch.setattr(feature, "_checkout_git_until_stopped", fake_checkout)

    installed = await feature.install_skill(
        source_url="https://example.com/repo.git",
        skill_name=name,
        ref="main",
    )

    assert installed["source_revision"] == "a" * 40
    assert installed["revision"] == feature.snapshot.by_name()[name].revision
    assert installed["enabled"] is False
    assert feature.snapshot.by_name()[name].document.description == "Fresh remote skill"


@pytest.mark.asyncio
async def test_git_install_rechecks_host_sources_after_checkout(
    feature, tmp_path, monkeypatch
):
    name = "host-arrived-during-checkout"
    shared_root = tmp_path / "shared-skills"
    shared_root.mkdir()
    monkeypatch.setenv("KESTREL_SHARED_SKILLS_DIR", str(shared_root))
    installer = ProceduralSkillsFeature(feature.agent)
    await installer.initialize()

    async def checkout_while_host_skill_appears(*, source_url, ref, skill_name, target):
        shared_folder = shared_root / skill_name
        shared_folder.mkdir()
        (shared_folder / "SKILL.md").write_text(
            serialize_skill_markdown(
                SkillDocument(skill_name, "New host skill", "Host procedure.")
            ),
            encoding="utf-8",
        )
        source = target / skill_name
        source.mkdir(parents=True)
        (source / "SKILL.md").write_text(
            serialize_skill_markdown(
                SkillDocument(skill_name, "Remote skill", "Remote procedure.")
            ),
            encoding="utf-8",
        )
        return GitCheckout(
            root=target,
            skill_folder=source,
            revision="f" * 40,
            remote_url=source_url,
            ref=ref,
        )

    monkeypatch.setattr(
        installer, "_checkout_git_until_stopped", checkout_while_host_skill_appears
    )
    try:
        with pytest.raises(SkillConflictError, match="resolved catalog"):
            await installer.install_skill(
                source_url="https://example.com/repo.git",
                skill_name=name,
                ref="main",
            )
        assert not (feature.agent.procedural_skills_root / name).exists()
    finally:
        await installer.shutdown()


@pytest.mark.asyncio
async def test_git_install_rolls_back_when_host_source_wins_during_publication(
    feature, tmp_path, monkeypatch
):
    name = "host-wins-during-install"
    shared_root = tmp_path / "shared-install-race"
    shared_root.mkdir()
    monkeypatch.setenv("KESTREL_SHARED_SKILLS_DIR", str(shared_root))
    installer = ProceduralSkillsFeature(feature.agent)
    await installer.initialize()
    checkout_root = tmp_path / "checkout-host-install-race"
    source = checkout_root / "skills" / name
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text(
        serialize_skill_markdown(
            SkillDocument(name, "Remote loser", "Remote procedure.")
        ),
        encoding="utf-8",
    )

    async def fake_checkout(*, source_url, ref, skill_name, target):
        return GitCheckout(
            root=checkout_root,
            skill_folder=source,
            revision="b" * 40,
            remote_url=source_url,
            ref=ref,
        )

    original_publish = installer._publish_installed_skill_with_state_claim

    async def publish_then_host_wins(**kwargs):
        operation = await original_publish(**kwargs)
        shared_folder = shared_root / name
        shared_folder.mkdir()
        (shared_folder / "SKILL.md").write_text(
            serialize_skill_markdown(
                SkillDocument(name, "Host winner", "Host procedure.")
            ),
            encoding="utf-8",
        )
        return operation

    monkeypatch.setattr(installer, "_checkout_git_until_stopped", fake_checkout)
    monkeypatch.setattr(
        installer,
        "_publish_installed_skill_with_state_claim",
        publish_then_host_wins,
    )
    try:
        with pytest.raises(SkillConflictError, match="resolved source changed"):
            await installer.install_skill(
                source_url="https://example.com/repo.git",
                skill_name=name,
                ref="main",
            )

        assert not (feature.agent.procedural_skills_root / name).exists()
        assert name not in await installer._enablement.load()
        assert installer.snapshot.by_name()[name].document.description == "Host winner"
    finally:
        await installer.shutdown()


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
async def test_failed_install_publication_keeps_raced_replacement_disabled(
    feature, tmp_path, monkeypatch
):
    name = "install-publication-raced-replacement"
    prior_state = await feature._enablement.set(name, enabled=True, priority=19)
    feature._states[name] = prior_state
    published = feature.agent.procedural_skills_root / name
    displaced = tmp_path / "displaced-failed-install-publication"
    marker = published / "replacement-survived-failed-install.md"
    checkout_root = tmp_path / "checkout-failed-install-publication"
    source = checkout_root / "skills" / name
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text(
        serialize_skill_markdown(SkillDocument(name, "Remote", "Procedure.")),
        encoding="utf-8",
    )

    def fake_checkout(self, *, url, ref, skill_name, target, cancel_event=None):
        return GitCheckout(
            root=checkout_root,
            skill_folder=source,
            revision="f" * 40,
            remote_url=url,
            ref=ref,
        )

    def swap_then_fail(*_args, **_kwargs):
        candidates = list(
            feature._store._internal_root.glob(
                f"{store_module.SKILL_PUBLICATION_STAGING_PREFIX}install-*"
            )
        )
        assert len(candidates) == 1
        candidates[0].rename(displaced)
        published.mkdir()
        marker.write_text("replacement", encoding="utf-8")
        (published / "SKILL.md").write_text(
            serialize_skill_markdown(
                SkillDocument(
                    name,
                    "Unapproved replacement after failed install publication",
                    "Unapproved procedure.",
                )
            ),
            encoding="utf-8",
        )
        raise OSError("simulated install publication failure")

    monkeypatch.setattr(
        "kestrel_feature_skills.git_source.GitSkillSource.checkout", fake_checkout
    )
    monkeypatch.setattr(store_module, "_atomic_replace_file_at", swap_then_fail)

    result = await feature.skill_install("https://example.com/repo.git", name, "main")

    assert result.status is ToolResultStatus.ERROR
    assert marker.read_text(encoding="utf-8") == "replacement"
    assert displaced.is_dir()
    assert (await feature._enablement.load())[name] == SkillState(
        False, DEFAULT_PRIORITY
    )
    assert feature.snapshot.by_name()[name].state == SkillState(False, DEFAULT_PRIORITY)
    assert "Unapproved replacement after failed install publication" not in (
        feature.context_clause_text
    )


@pytest.mark.asyncio
async def test_install_conflict_keeps_raced_replacement_disabled(
    feature, tmp_path, monkeypatch
):
    name = "install-conflict-raced-replacement"
    prior_state = await feature._enablement.set(name, enabled=True, priority=19)
    feature._states[name] = prior_state
    published = feature.agent.procedural_skills_root / name
    checkout_root = tmp_path / "checkout-install-conflict"
    source = checkout_root / "skills" / name
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text(
        serialize_skill_markdown(SkillDocument(name, "Remote", "Procedure.")),
        encoding="utf-8",
    )

    def fake_checkout(self, *, url, ref, skill_name, target, cancel_event=None):
        return GitCheckout(
            root=checkout_root,
            skill_folder=source,
            revision="a" * 40,
            remote_url=url,
            ref=ref,
        )

    def publish_replacement_then_conflict(*_args, **_kwargs):
        published.mkdir()
        (published / "SKILL.md").write_text(
            serialize_skill_markdown(
                SkillDocument(
                    name,
                    "Unapproved replacement after install conflict",
                    "Unapproved procedure.",
                )
            ),
            encoding="utf-8",
        )
        raise SkillConflictError("a non-cooperating writer won installation")

    monkeypatch.setattr(
        "kestrel_feature_skills.git_source.GitSkillSource.checkout",
        fake_checkout,
    )
    monkeypatch.setattr(
        feature._store,
        "install_folder_pinned",
        publish_replacement_then_conflict,
    )

    result = await feature.skill_install("https://example.com/repo.git", name, "main")

    assert result.status is ToolResultStatus.ERROR
    assert (await feature._enablement.load())[name] == SkillState(
        False, DEFAULT_PRIORITY
    )
    assert feature.snapshot.by_name()[name].state == SkillState(False, DEFAULT_PRIORITY)
    assert "Unapproved replacement after install conflict" not in (
        feature.context_clause_text
    )


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
async def test_mixed_case_python_suffix_keeps_execution_risk_warning(feature):
    await feature.skill_create("mixed-python", "Mixed Python", "body")
    folder = feature.agent.procedural_skills_root / "mixed-python" / "scripts"
    folder.mkdir()
    (folder / "check.PY").write_text("print('still executable')\n", encoding="utf-8")
    await feature.refresh()

    tree = feature.tree(name="mixed-python")
    python = next(item for item in tree if item["path"] == "scripts/check.PY")
    response = feature.read_file(
        name="mixed-python",
        relative_path="scripts/check.PY",
    )

    assert python["execution_risk"] is True
    assert response["language"] == "python"
    assert response["execution_risk"] is True


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


@pytest.mark.asyncio
async def test_nested_metadata_like_resources_remain_in_inventory(feature):
    name = "nested-metadata-resources"
    await feature.skill_create(name, "Nested metadata resources", "body")
    folder = feature.agent.procedural_skills_root / name
    resources = folder / "docs"
    resources.mkdir()
    expected = {
        f"docs/{PROVENANCE_FILENAME}": "nested provenance notes",
        "docs/.SKILL.md.tmp.notes.md": "nested temporary-looking notes",
    }
    for relative_path, content in expected.items():
        (folder / relative_path).write_text(content, encoding="utf-8")
    await feature.refresh()

    tree_paths = {entry["path"] for entry in feature.tree(name=name)}
    assert expected.keys() <= tree_paths
    for relative_path, content in expected.items():
        result = await feature.skill_read(name, relative_path)
        assert result.status is ToolResultStatus.OK
        assert result.data["content"] == content


def test_context_renderer_is_registered_through_sdk_contract(feature):
    assert feature.context_clause_text == render_context_clause(feature.snapshot).text
    registrations = feature.get_context_clause_registrations()
    assert len(registrations) == 1
    registration = registrations[0]
    assert registration.owner == feature.contribution_owner
    assert registration.name == "procedural-skills"
    assert registration.renderer() == feature.context_clause_text
    assert feature.get_hooks() == []


@pytest.mark.asyncio
async def test_enablement_transition_publishes_fresh_core_owned_clause(feature):
    published = []
    feature.agent.feature_contribution_runtime = SimpleNamespace(
        is_active=lambda candidate: candidate is feature
    )
    feature.agent.refresh_feature_context_clauses = lambda candidate: published.append(
        candidate.context_clause_text
    )
    await feature.skill_create(
        "published-context",
        "Published description only",
        "PROCEDURE-BODY-MUST-STAY-OUT",
    )

    await feature.skill_enable("published-context", priority=7)
    await feature.skill_disable("published-context")

    assert len(published) == 2
    assert "Published description only" in published[0]
    assert "PROCEDURE-BODY-MUST-STAY-OUT" not in published[0]
    assert published[1] == ""


@pytest.mark.asyncio
async def test_failed_context_publication_remains_retryable(feature):
    created = await feature.skill_create(
        "retry-context",
        "Retry this description",
        "Procedure remains private",
    )
    assert created.status is ToolResultStatus.OK
    await feature._enablement.set("retry-context", enabled=True, priority=7)
    feature.agent.feature_contribution_runtime = SimpleNamespace(
        is_active=lambda candidate: candidate is feature
    )

    def fail_refresh(_candidate):
        raise RuntimeError("simulated core publication failure")

    feature.agent.refresh_feature_context_clauses = fail_refresh
    with pytest.raises(RuntimeError, match="simulated core publication failure"):
        await feature.refresh()
    assert feature.context_clause_text == ""

    published = []
    feature.agent.refresh_feature_context_clauses = lambda candidate: published.append(
        candidate.context_clause_text
    )
    await feature.refresh()

    assert len(published) == 1
    assert "Retry this description" in published[0]
