from __future__ import annotations

import json

import pytest

import kestrel_feature_skills.git_source as git_source_module
from kestrel_feature_skills.errors import GitSourceError
from kestrel_feature_skills.format import serialize_skill_markdown
from kestrel_feature_skills.git_source import GitSkillSource, validate_remote_url
from kestrel_feature_skills.models import SkillDocument, SkillProvenance, SkillState
from kestrel_feature_skills.sources import (
    AGENT_LOCAL_PRECEDENCE,
    HOST_SHARED_PRECEDENCE,
    PROVENANCE_FILENAME,
    REMOTE_PRECEDENCE,
    DirectorySkillSource,
    SkillCatalog,
    serialize_provenance,
)


def make_skill(root, name, description):
    folder = root / name
    folder.mkdir(parents=True)
    document = SkillDocument(name, description, "# Procedure\n\nDo it.")
    (folder / "SKILL.md").write_text(
        serialize_skill_markdown(document), encoding="utf-8"
    )
    return folder


def catalog(local, shared):
    return SkillCatalog(
        (
            DirectorySkillSource(
                root=local,
                source_id="agent-local",
                kind="agent-local",
                precedence=AGENT_LOCAL_PRECEDENCE,
            ),
            DirectorySkillSource(
                root=shared,
                source_id="host-shared",
                kind="host-shared",
                precedence=HOST_SHARED_PRECEDENCE,
            ),
        )
    )


def test_agent_local_shadows_host_shared_deterministically(tmp_path):
    local = tmp_path / "local"
    shared = tmp_path / "shared"
    local.mkdir()
    shared.mkdir()
    make_skill(local, "overlap", "local wins")
    make_skill(shared, "overlap", "shared loses")
    snapshot = catalog(local, shared).refresh()
    assert snapshot.records[0].document.description == "local wins"
    assert snapshot.records[0].source_id == "agent-local"
    assert snapshot.shadowed["overlap"][0].source_id == "host-shared"


def test_enablement_and_priority_survive_catalog_reload(tmp_path):
    local = tmp_path / "local"
    shared = tmp_path / "shared"
    local.mkdir()
    shared.mkdir()
    make_skill(local, "later", "later")
    make_skill(shared, "first", "first")
    states = {
        "later": SkillState(True, 200),
        "first": SkillState(True, -5),
    }
    first = catalog(local, shared).refresh(states)
    second = catalog(local, shared).refresh(states)
    assert [(r.name, r.state.priority) for r in first.records] == [
        ("first", -5),
        ("later", 200),
    ]
    assert first == second


def test_git_provenance_sidecar_survives_reload(tmp_path):
    local = tmp_path / "local"
    shared = tmp_path / "shared"
    local.mkdir()
    shared.mkdir()
    folder = make_skill(local, "installed", "installed")
    provenance = SkillProvenance(
        kind="git",
        source_id="https://example.com/skills.git",
        locator="main:installed",
        revision="a" * 40,
        remote_url="https://example.com/skills.git",
    )
    (folder / PROVENANCE_FILENAME).write_bytes(serialize_provenance(provenance))
    record = catalog(local, shared).refresh().records[0]
    assert record.provenance == provenance
    assert record.source_kind == "agent-local"
    assert record.source_id == "https://example.com/skills.git"
    assert record.precedence == REMOTE_PRECEDENCE
    assert record.editable is True


def test_host_shared_skill_shadows_git_installed_origin(tmp_path):
    local = tmp_path / "local"
    shared = tmp_path / "shared"
    local.mkdir()
    shared.mkdir()
    remote = make_skill(local, "overlap", "remote loses")
    make_skill(shared, "overlap", "host wins")
    provenance = SkillProvenance(
        kind="git",
        source_id="https://example.com/skills.git",
        locator="main:overlap",
        revision="a" * 40,
        remote_url="https://example.com/skills.git",
    )
    (remote / PROVENANCE_FILENAME).write_bytes(serialize_provenance(provenance))

    snapshot = catalog(local, shared).refresh()

    assert snapshot.records[0].document.description == "host wins"
    assert snapshot.records[0].source_id == "host-shared"
    assert snapshot.shadowed["overlap"] == (provenance,)


def test_malformed_folder_is_reported_not_loaded(tmp_path):
    local = tmp_path / "local"
    shared = tmp_path / "shared"
    local.mkdir()
    shared.mkdir()
    broken = local / "broken"
    broken.mkdir()
    (broken / "SKILL.md").write_text("", encoding="utf-8")
    snapshot = catalog(local, shared).refresh()
    assert snapshot.records == ()
    assert len(snapshot.errors) == 1
    assert "empty" in snapshot.errors[0].error


def test_unknown_provenance_metadata_is_visible_error(tmp_path):
    local = tmp_path / "local"
    shared = tmp_path / "shared"
    local.mkdir()
    shared.mkdir()
    folder = make_skill(local, "bad-meta", "bad metadata")
    (folder / PROVENANCE_FILENAME).write_text(
        json.dumps(
            {"version": 1, "kind": "git", "source_id": "x", "locator": "y", "extra": 1}
        ),
        encoding="utf-8",
    )
    snapshot = catalog(local, shared).refresh()
    assert not snapshot.records
    assert "unsupported field" in snapshot.errors[0].error


@pytest.mark.parametrize(
    "url",
    (
        "http://example.com/skills.git",
        "https://user:secret@example.com/skills.git",
        "https://example.com/skills.git?token=secret",
        "file:///tmp/skills.git",
    ),
)
def test_git_source_rejects_non_https_or_credential_bearing_urls(url):
    with pytest.raises(GitSourceError):
        validate_remote_url(url)


def test_git_source_detects_when_recorded_commit_changes(monkeypatch):
    revisions = iter(("a" * 40, "b" * 40))

    def fake_git(argv, *, timeout=120):
        assert argv[:2] == ["ls-remote", "--exit-code"]
        assert timeout == 60
        return f"{next(revisions)}\trefs/heads/main"

    monkeypatch.setattr(git_source_module, "_run_git", fake_git)
    source = GitSkillSource()
    assert not source.has_changed(
        url="https://example.com/skills.git",
        ref="main",
        installed_revision="a" * 40,
    )
    assert source.has_changed(
        url="https://example.com/skills.git",
        ref="main",
        installed_revision="a" * 40,
    )
