from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

import kestrel_feature_skills.git_source as git_source_module
import kestrel_feature_skills.sources as sources_module
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


def test_discovery_rejects_folder_moved_outside_root_then_replaced_by_symlink(
    tmp_path, monkeypatch
):
    local = tmp_path / "local"
    shared = tmp_path / "shared"
    outside = tmp_path / "outside"
    local.mkdir()
    shared.mkdir()
    outside.mkdir()
    folder = make_skill(local, "raced", "Race containment")
    moved = outside / folder.name
    real_validate = sources_module.validate_skill_folder
    swapped = False

    def validate_then_swap(candidate, *, source_root):
        nonlocal swapped
        document = real_validate(candidate, source_root=source_root)
        if candidate == folder and not swapped:
            candidate.rename(moved)
            candidate.symlink_to(moved, target_is_directory=True)
            swapped = True
        return document

    monkeypatch.setattr(sources_module, "validate_skill_folder", validate_then_swap)

    snapshot = catalog(local, shared).refresh()

    assert snapshot.records == ()
    assert len(snapshot.errors) == 1
    assert any(
        word in snapshot.errors[0].error for word in ("changed", "escape", "symlink")
    )


def test_source_root_disappearing_during_resolution_is_a_visible_error(
    tmp_path, monkeypatch
):
    local = tmp_path / "local"
    local.mkdir()
    real_resolve = Path.resolve

    def fail_root_resolution(path, *args, **kwargs):
        if path == local:
            raise FileNotFoundError("source root moved")
        return real_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", fail_root_resolution)
    source = DirectorySkillSource(
        root=local,
        source_id="agent-local",
        kind="agent-local",
        precedence=AGENT_LOCAL_PRECEDENCE,
    )

    records, errors = source.discover()

    assert records == ()
    assert len(errors) == 1
    assert "source root" in errors[0].error


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


def test_json_escaped_surrogates_are_rejected_in_documents_and_provenance(tmp_path):
    local = tmp_path / "local"
    shared = tmp_path / "shared"
    local.mkdir()
    shared.mkdir()
    broken = local / "broken-text"
    broken.mkdir()
    (broken / "SKILL.md").write_text(
        '---\nname: "broken-text"\ndescription: "\\ud800"\n---\n\nbody\n',
        encoding="ascii",
    )
    folder = make_skill(local, "bad-origin", "bad origin")
    (folder / PROVENANCE_FILENAME).write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "git",
                "source_id": "\ud800",
                "locator": "main:bad-origin",
                "revision": None,
                "remote_url": None,
            }
        ),
        encoding="ascii",
    )

    snapshot = catalog(local, shared).refresh()

    assert snapshot.records == ()
    assert len(snapshot.errors) == 2
    assert all("UTF-8" in error.error for error in snapshot.errors)


@pytest.mark.parametrize(
    "url",
    (
        "http://example.com/skills.git",
        "https://user:secret@example.com/skills.git",
        "https://example.com/skills.git?token=secret",
        "https://[",
        "file:///tmp/skills.git",
    ),
)
def test_git_source_rejects_non_https_or_credential_bearing_urls(url):
    with pytest.raises(GitSourceError):
        validate_remote_url(url)


@pytest.mark.parametrize(
    "url",
    (
        "https://example.com/skills.git\x00",
        "https://example.com/skills.git\n--upload-pack=evil",
        "https://example.com/skills-\ud800.git",
    ),
)
def test_git_source_rejects_control_or_unencodable_url_text(url):
    with pytest.raises(GitSourceError, match="URL"):
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


def test_git_checkout_uses_remote_default_branch_for_head(tmp_path, monkeypatch):
    commands = []
    target = tmp_path / "checkout"

    def fake_git(argv, *, timeout=120, size_limit_root=None, max_bytes=None):
        commands.append((argv, size_limit_root, max_bytes))
        if argv[0] == "clone":
            make_skill(target / "skills", "remote", "Remote default branch")
            return ""
        return "a" * 40

    monkeypatch.setattr(git_source_module, "_run_git", fake_git)

    checkout = GitSkillSource().checkout(
        url="https://example.com/skills.git",
        ref="HEAD",
        skill_name="remote",
        target=target,
    )

    clone, size_limit_root, max_bytes = commands[0]
    assert clone[:3] == ["clone", "--depth", "1"]
    assert "--filter=blob:none" in clone
    assert "--sparse" in clone
    assert "--no-checkout" in clone
    assert "--branch" not in clone
    assert "--single-branch" not in clone
    assert size_limit_root == target
    assert max_bytes is not None and 0 < max_bytes <= 64 * 1024 * 1024
    sparse, sparse_root, sparse_max = commands[1]
    assert sparse[:5] == [
        "-C",
        str(target),
        "sparse-checkout",
        "set",
        "--no-cone",
    ]
    assert sparse[-2:] == ["/remote/", "/skills/remote/"]
    assert (sparse_root, sparse_max) == (size_limit_root, max_bytes)
    checkout_command, checkout_root, checkout_max = commands[2]
    assert checkout_command == [
        "-C",
        str(target),
        "checkout",
        "--detach",
        "HEAD",
    ]
    assert (checkout_root, checkout_max) == (size_limit_root, max_bytes)
    assert checkout.ref == "HEAD"


def test_git_runner_rejects_a_checkout_that_crosses_its_disk_bound(tmp_path):
    target = tmp_path / "oversized-git-data"

    with pytest.raises(GitSourceError, match="transfer limit"):
        git_source_module._run_git(
            ["init", str(target)],
            size_limit_root=target,
            max_bytes=1,
        )


def test_git_runner_rejects_host_transport_rewrites(tmp_path, monkeypatch):
    origin = tmp_path / "repository.git"
    git_source_module._run_git(["init", "--bare", str(origin)])
    host_config = tmp_path / "host.gitconfig"
    host_config.write_text(
        '[url "file://' + origin.parent.as_posix() + '/"]\n'
        "\tinsteadOf = https://127.0.0.1:1/\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(host_config))
    target = tmp_path / "rewritten-checkout"

    with pytest.raises(GitSourceError):
        git_source_module._run_git(
            ["clone", "--", "https://127.0.0.1:1/repository.git", str(target)],
            timeout=5,
        )


def test_git_runner_uses_sanitized_config_and_https_only_protocols(
    tmp_path, monkeypatch
):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "print(json.dumps({\n"
        "    'argv': sys.argv[1:],\n"
        "    'global': os.environ.get('GIT_CONFIG_GLOBAL'),\n"
        "    'nosystem': os.environ.get('GIT_CONFIG_NOSYSTEM'),\n"
        "}))\n",
        encoding="utf-8",
    )
    fake_git.chmod(0o700)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ['PATH']}")

    payload = json.loads(git_source_module._run_git(["probe"]))

    assert payload["global"] == os.devnull
    assert payload["nosystem"] == "1"
    assert payload["argv"][:6] == [
        "-c",
        "protocol.allow=never",
        "-c",
        "protocol.https.allow=always",
        "-c",
        "protocol.file.allow=never",
    ]


def test_git_runner_caps_subprocess_output(tmp_path, monkeypatch):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "sys.stderr.write('x' * 1_048_577)\n"
        "sys.stderr.flush()\n",
        encoding="utf-8",
    )
    fake_git.chmod(0o700)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ['PATH']}")

    with pytest.raises(GitSourceError, match="output limit"):
        git_source_module._run_git(["probe"], timeout=5)


def test_git_checkout_materializes_only_the_requested_folder(tmp_path, monkeypatch):
    origin = tmp_path / "origin"
    origin.mkdir()
    git_source_module._run_git(["init", "--initial-branch=main", str(origin)])
    git_source_module._run_git(
        ["-C", str(origin), "config", "uploadpack.allowFilter", "true"]
    )
    make_skill(origin / "skills", "remote", "Sparse checkout")
    unrelated = origin / "unrelated"
    unrelated.mkdir()
    (unrelated / "large.bin").write_bytes(b"x" * (1024 * 1024))
    git_source_module._run_git(["-C", str(origin), "add", "."])
    git_source_module._run_git(
        [
            "-C",
            str(origin),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "fixture",
        ]
    )
    source_url = origin.as_uri()
    monkeypatch.setattr(
        git_source_module, "validate_remote_url", lambda _url: source_url
    )
    monkeypatch.setattr(
        git_source_module,
        "_GIT_CONFIG_PREFIX",
        (
            "-c",
            "protocol.allow=never",
            "-c",
            "protocol.https.allow=never",
            "-c",
            "protocol.file.allow=always",
        ),
    )
    target = tmp_path / "checkout"

    checkout = GitSkillSource().checkout(
        url=source_url,
        ref="HEAD",
        skill_name="remote",
        target=target,
    )

    assert checkout.skill_folder == target / "skills" / "remote"
    assert (checkout.skill_folder / "SKILL.md").is_file()
    assert not (target / "unrelated").exists()
