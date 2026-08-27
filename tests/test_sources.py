from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

import kestrel_feature_skills.git_source as git_source_module
import kestrel_feature_skills.sources as sources_module
from kestrel_feature_skills.errors import GitSourceError
from kestrel_feature_skills.format import MAX_SKILL_FILE_BYTES, serialize_skill_markdown
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


@pytest.mark.parametrize(
    "payload, message",
    (
        (b"x" * (MAX_SKILL_FILE_BYTES + 1), "exceeds"),
        (b"text-prefix\xff", "UTF-8"),
    ),
)
def test_discovery_rejects_resources_that_skill_read_cannot_open(
    tmp_path, payload, message
):
    local = tmp_path / "local"
    shared = tmp_path / "shared"
    local.mkdir()
    shared.mkdir()
    folder = make_skill(local, "unreadable-resource", "Invalid resource")
    (folder / "reference.txt").write_bytes(payload)

    snapshot = catalog(local, shared).refresh()

    assert snapshot.records == ()
    assert len(snapshot.errors) == 1
    assert message in snapshot.errors[0].error


@pytest.mark.parametrize("filename", (r"notes\draft.md", "C:notes.md"))
def test_discovery_rejects_resource_paths_that_readers_cannot_address(
    tmp_path, filename
):
    local = tmp_path / "local"
    shared = tmp_path / "shared"
    local.mkdir()
    shared.mkdir()
    folder = make_skill(local, "unaddressable-resource", "Invalid resource path")
    (folder / filename).write_text("notes", encoding="utf-8")

    snapshot = catalog(local, shared).refresh()

    assert snapshot.records == ()
    assert len(snapshot.errors) == 1
    assert "absolute and backslash" in snapshot.errors[0].error


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


def test_provenance_reader_uses_nonblocking_open_across_fifo_swap(
    tmp_path, monkeypatch
):
    local = tmp_path / "local"
    local.mkdir()
    folder = make_skill(local, "swapped-provenance", "Swapped provenance")
    provenance_path = folder / PROVENANCE_FILENAME
    provenance_path.write_bytes(
        serialize_provenance(
            SkillProvenance(
                kind="git",
                source_id="https://example.com/skills.git",
                locator="main:swapped-provenance",
                revision="a" * 40,
                remote_url="https://example.com/skills.git",
            )
        )
    )
    real_open = os.open
    folder_fd = real_open(
        folder,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    source = DirectorySkillSource(
        root=local,
        source_id="agent-local",
        kind="agent-local",
        precedence=AGENT_LOCAL_PRECEDENCE,
    )
    swapped = False
    completed = threading.Event()
    errors = []

    def swap_during_open(name, flags, *args, dir_fd=None, **kwargs):
        nonlocal swapped
        if name == PROVENANCE_FILENAME and dir_fd is not None and not swapped:
            provenance_path.unlink()
            os.mkfifo(provenance_path)
            swapped = True
        return real_open(name, flags, *args, dir_fd=dir_fd, **kwargs)

    monkeypatch.setattr(sources_module.os, "open", swap_during_open)

    def read_swapped_provenance():
        try:
            sources_module._load_provenance_at(
                source,
                folder_fd,
                folder.name,
            )
        except Exception as exc:  # noqa: BLE001 - asserted below
            errors.append(exc)
        finally:
            completed.set()

    reader = threading.Thread(target=read_swapped_provenance)
    reader.start()
    finished_without_writer = completed.wait(timeout=0.5)
    if not finished_without_writer:
        writer = real_open(provenance_path, os.O_WRONLY | os.O_NONBLOCK)
        os.close(writer)
    reader.join(timeout=5)
    os.close(folder_fd)

    assert swapped
    assert finished_without_writer, "provenance read blocked while opening a FIFO"
    assert not reader.is_alive()
    assert len(errors) == 1
    assert "changed during discovery" in str(errors[0])


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


def test_malformed_link_url_is_quarantined_without_aborting_catalog(tmp_path):
    local = tmp_path / "local"
    shared = tmp_path / "shared"
    local.mkdir()
    shared.mkdir()
    make_skill(local, "healthy", "Healthy skill")
    broken = local / "broken-url"
    broken.mkdir()
    (broken / "SKILL.md").write_text(
        serialize_skill_markdown(
            SkillDocument("broken-url", "Broken URL", "[broken](//[invalid)")
        ),
        encoding="utf-8",
    )

    snapshot = catalog(local, shared).refresh()

    assert [record.name for record in snapshot.records] == ["healthy"]
    assert len(snapshot.errors) == 1
    assert "malformed link URL" in snapshot.errors[0].error


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
    real_validate = sources_module.validate_skill_folder_descriptor
    swapped = False

    def validate_then_swap(descriptor, *, folder_name):
        nonlocal swapped
        document = real_validate(descriptor, folder_name=folder_name)
        if folder_name == folder.name and not swapped:
            candidate = local / folder_name
            candidate.rename(moved)
            candidate.symlink_to(moved, target_is_directory=True)
            swapped = True
        return document

    monkeypatch.setattr(
        sources_module,
        "validate_skill_folder_descriptor",
        validate_then_swap,
    )

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


def test_dangling_configured_source_root_is_a_visible_error(tmp_path):
    missing = tmp_path / "missing-source"
    configured = tmp_path / "configured-source"
    configured.symlink_to(missing, target_is_directory=True)
    source = DirectorySkillSource(
        root=configured,
        source_id="host-shared",
        kind="host-shared",
        precedence=HOST_SHARED_PRECEDENCE,
    )

    records, errors = source.discover()

    assert records == ()
    assert len(errors) == 1
    assert "source root" in errors[0].error


def test_discovery_does_not_follow_a_source_root_path_replacement(
    tmp_path, monkeypatch
):
    local = tmp_path / "local"
    displaced = tmp_path / "displaced-local"
    local.mkdir()
    make_skill(local, "trusted", "Trusted original")
    real_iterdir = Path.iterdir
    real_listdir = os.listdir
    swapped = False

    def swap_root():
        nonlocal swapped
        if not swapped:
            swapped = True
            local.rename(displaced)
            local.mkdir()
            make_skill(local, "attacker", "Untrusted replacement")

    def swap_before_path_enumeration(path):
        if path == local:
            swap_root()
        return real_iterdir(path)

    def swap_before_descriptor_enumeration(path):
        if isinstance(path, int):
            swap_root()
        return real_listdir(path)

    monkeypatch.setattr(Path, "iterdir", swap_before_path_enumeration)
    monkeypatch.setattr(
        sources_module.os, "listdir", swap_before_descriptor_enumeration
    )
    source = DirectorySkillSource(
        root=local,
        source_id="agent-local",
        kind="agent-local",
        precedence=AGENT_LOCAL_PRECEDENCE,
    )

    records, errors = source.discover()

    assert swapped
    assert records == ()
    assert len(errors) == 1
    assert "source" in errors[0].error and "changed" in errors[0].error


def test_discovery_error_with_non_utf8_locator_remains_json_serializable(
    tmp_path, monkeypatch
):
    local = tmp_path / "local"
    local.mkdir()
    real_listdir = os.listdir
    injected = False

    def fake_listdir(path):
        nonlocal injected
        if isinstance(path, int) and not injected:
            injected = True
            return ["bad-\udcff"]
        return real_listdir(path)

    monkeypatch.setattr(sources_module.os, "listdir", fake_listdir)
    source = DirectorySkillSource(
        root=local,
        source_id="agent-local",
        kind="agent-local",
        precedence=AGENT_LOCAL_PRECEDENCE,
    )

    records, errors = source.discover()
    payload = {"errors": [error.to_dict() for error in errors]}

    assert records == ()
    assert len(errors) == 1
    assert errors[0].locator == "bad-\\udcff"
    json.dumps(payload, ensure_ascii=False).encode("utf-8")


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


def test_provenance_decoder_value_error_is_quarantined_without_aborting_catalog(
    tmp_path,
):
    local = tmp_path / "local"
    shared = tmp_path / "shared"
    local.mkdir()
    shared.mkdir()
    make_skill(local, "healthy", "healthy skill")
    broken = make_skill(local, "bad-provenance-number", "bad provenance")
    (broken / PROVENANCE_FILENAME).write_text(
        '{"version":' + "9" * 5000 + "}",
        encoding="ascii",
    )

    snapshot = catalog(local, shared).refresh()

    assert [record.name for record in snapshot.records] == ["healthy"]
    assert len(snapshot.errors) == 1
    assert snapshot.errors[0].locator == "bad-provenance-number"
    assert f"invalid {PROVENANCE_FILENAME}" in snapshot.errors[0].error


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
        "https://example.com:not-a-port/skills.git",
        "https://example.com:99999/skills.git",
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


def test_git_source_compares_annotated_tag_peeled_commit(monkeypatch):
    tag_object = "a" * 40
    peeled_commit = "b" * 40

    def fake_git(argv, *, timeout=120):
        assert argv[:2] == ["ls-remote", "--exit-code"]
        assert timeout == 60
        return f"{tag_object}\trefs/tags/v1.0.0\n{peeled_commit}\trefs/tags/v1.0.0^{{}}"

    monkeypatch.setattr(git_source_module, "_run_git", fake_git)

    assert not GitSkillSource().has_changed(
        url="https://example.com/skills.git",
        ref="v1.0.0",
        installed_revision=peeled_commit,
    )


def test_git_checkout_uses_remote_default_branch_for_head(tmp_path, monkeypatch):
    commands = []
    target = tmp_path / "checkout"

    def fake_git(
        argv,
        *,
        timeout=120,
        size_limit_root=None,
        max_bytes=None,
        max_entries=None,
        cancel_event=None,
    ):
        commands.append((argv, size_limit_root, max_bytes, max_entries))
        if argv[0] == "clone":
            make_skill(target / "skills", "remote", "Remote default branch")
            return ""
        if "ls-tree" in argv:
            return f"100644 blob {'b' * 40} 128\tskills/remote/SKILL.md\x00"
        return "a" * 40

    monkeypatch.setattr(git_source_module, "_run_git", fake_git)

    checkout = GitSkillSource().checkout(
        url="https://example.com/skills.git",
        ref="HEAD",
        skill_name="remote",
        target=target,
    )

    clone, size_limit_root, max_bytes, max_entries = commands[0]
    assert clone[:3] == ["clone", "--depth", "1"]
    assert "--filter=blob:none" in clone
    assert "--sparse" in clone
    assert "--no-checkout" in clone
    assert "--branch" not in clone
    assert "--single-branch" not in clone
    assert size_limit_root == target
    assert max_bytes is not None and 0 < max_bytes <= 64 * 1024 * 1024
    assert max_entries is not None and 512 < max_entries <= 8192
    preflight, preflight_root, preflight_max, preflight_entries = commands[1]
    assert preflight[:4] == ["-C", str(target), "ls-tree", "-r"]
    assert (preflight_root, preflight_max, preflight_entries) == (
        size_limit_root,
        max_bytes,
        max_entries,
    )
    sparse, sparse_root, sparse_max, sparse_entries = commands[2]
    assert sparse[:5] == [
        "-C",
        str(target),
        "sparse-checkout",
        "set",
        "--no-cone",
    ]
    assert sparse[-2:] == ["/remote/", "/skills/remote/"]
    assert (sparse_root, sparse_max, sparse_entries) == (
        size_limit_root,
        max_bytes,
        max_entries,
    )
    checkout_command, checkout_root, checkout_max, checkout_entries = commands[3]
    assert checkout_command == [
        "-C",
        str(target),
        "checkout",
        "--detach",
        "HEAD",
    ]
    assert (checkout_root, checkout_max, checkout_entries) == (
        size_limit_root,
        max_bytes,
        max_entries,
    )
    assert checkout.ref == "HEAD"


def test_git_checkout_rejects_oversized_sparse_blob_before_materialization(
    tmp_path, monkeypatch
):
    commands = []

    def fake_git(argv, **_kwargs):
        commands.append(argv)
        if "ls-tree" in argv:
            return (
                "100644 blob "
                f"{'a' * 40} {git_source_module.MAX_GIT_TRANSFER_BYTES + 1}"
                "\tremote/huge.bin\x00"
            )
        if "rev-parse" in argv:
            return "b" * 40
        return ""

    monkeypatch.setattr(git_source_module, "_run_git", fake_git)

    with pytest.raises(GitSourceError, match="transfer limit"):
        GitSkillSource().checkout(
            url="https://example.com/skills.git",
            ref="main",
            skill_name="remote",
            target=tmp_path / "checkout",
        )

    assert not any(
        "checkout" in command and "--detach" in command for command in commands
    )


def test_git_checkout_rejects_submodule_before_materialization(tmp_path, monkeypatch):
    commands = []

    def fake_git(argv, **_kwargs):
        commands.append(argv)
        if "ls-tree" in argv:
            return f"160000 commit {'a' * 40} -\tremote/nested\x00"
        return ""

    monkeypatch.setattr(git_source_module, "_run_git", fake_git)

    with pytest.raises(GitSourceError, match="submodule"):
        GitSkillSource().checkout(
            url="https://example.com/skills.git",
            ref="main",
            skill_name="remote",
            target=tmp_path / "checkout",
        )

    assert not any(
        "checkout" in command and "--detach" in command for command in commands
    )


def test_git_runner_rejects_a_checkout_that_crosses_its_disk_bound(tmp_path):
    target = tmp_path / "oversized-git-data"

    with pytest.raises(GitSourceError, match="transfer limit"):
        git_source_module._run_git(
            ["init", str(target)],
            size_limit_root=target,
            max_bytes=1,
            max_entries=4096,
        )


def test_git_runner_rejects_checkout_with_too_many_zero_byte_entries(
    tmp_path, monkeypatch
):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, sys, time\n"
        "root = pathlib.Path(sys.argv[-1])\n"
        "root.mkdir()\n"
        "for index in range(17):\n"
        "    (root / f'empty-{index}').touch()\n"
        "time.sleep(5)\n",
        encoding="utf-8",
    )
    fake_git.chmod(0o700)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ['PATH']}")
    target = tmp_path / "entry-heavy-checkout"

    with pytest.raises(GitSourceError, match="entry limit"):
        git_source_module._run_git(
            ["probe", str(target)],
            timeout=5,
            size_limit_root=target,
            max_bytes=1024,
            max_entries=16,
        )


def test_git_runner_reaps_process_when_checkout_measurement_fails(
    tmp_path, monkeypatch
):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text(
        f"#!{sys.executable}\nimport time\ntime.sleep(30)\n",
        encoding="utf-8",
    )
    fake_git.chmod(0o700)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ['PATH']}")
    processes = []
    real_popen = git_source_module.subprocess.Popen

    def tracking_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        processes.append(process)
        return process

    def fail_measurement(*_args, **_kwargs):
        raise GitSourceError("could not measure bounded git checkout data")

    monkeypatch.setattr(git_source_module.subprocess, "Popen", tracking_popen)
    monkeypatch.setattr(git_source_module, "_tree_limit_error", fail_measurement)

    with pytest.raises(GitSourceError, match="could not measure"):
        git_source_module._run_git(
            ["probe"],
            size_limit_root=tmp_path,
            max_bytes=1024,
            max_entries=16,
        )

    assert len(processes) == 1
    process = processes[0]
    alive_after_return = process.poll() is None
    if alive_after_return:
        git_source_module._kill_process_group(process)
        process.wait()
    assert not alive_after_return
    assert process.poll() is not None


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


def test_git_runner_uses_sanitized_config_https_only_protocols_and_no_redirects(
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
    assert payload["argv"][:8] == [
        "-c",
        "protocol.allow=never",
        "-c",
        "protocol.https.allow=always",
        "-c",
        "protocol.file.allow=never",
        "-c",
        "http.followRedirects=false",
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


def test_git_runner_terminates_subprocess_when_cancelled(tmp_path, monkeypatch):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    marker = tmp_path / "git-started"
    fake_git = fake_bin / "git"
    fake_git.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, time\n"
        f"pathlib.Path({str(marker)!r}).write_text('started')\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    fake_git.chmod(0o700)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ['PATH']}")
    cancel_event = threading.Event()
    errors = []

    def run_git():
        try:
            git_source_module._run_git(["probe"], timeout=1, cancel_event=cancel_event)
        except Exception as exc:  # noqa: BLE001 - asserted below
            errors.append(exc)

    worker = threading.Thread(target=run_git)
    worker.start()
    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists()
        cancel_event.set()
        worker.join(timeout=5)
    finally:
        cancel_event.set()
        worker.join(timeout=5)

    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], GitSourceError)
    assert "cancel" in str(errors[0])


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
