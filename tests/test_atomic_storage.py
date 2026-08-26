from __future__ import annotations

import os
import shutil
import threading
import time
from pathlib import Path

import pytest

from kestrel_feature_skills.errors import (
    SkillConflictError,
    SkillFormatError,
    SkillPathError,
)
from kestrel_feature_skills.format import (
    MAX_FOLDER_BYTES,
    MAX_FOLDER_FILES,
    serialize_skill_markdown,
    validate_skill_folder,
)
from kestrel_feature_skills.models import SkillDocument, SkillProvenance, SkillRecord
from kestrel_feature_skills.store import (
    CLAIM_STALENESS_SECONDS,
    SkillStore,
    atomic_write_primary,
)


def payload(name="atomic"):
    return serialize_skill_markdown(
        SkillDocument(name, "Atomic write", "# Procedure\n\nWrite once.")
    ).encode("utf-8")


def test_atomic_create_leaves_no_temporary_or_claim_files(tmp_path):
    root = tmp_path / "skills"
    store = SkillStore(root)
    folder = store.create(SkillDocument("atomic", "Atomic write", "Do it."))
    assert (folder / "SKILL.md").is_file()
    assert not list(folder.glob(".SKILL.md.tmp.*"))
    assert not (folder / ".SKILL.md.claim").exists()


def test_collision_refuses_to_overwrite_existing_primary(tmp_path):
    folder = tmp_path / "atomic"
    folder.mkdir()
    original = payload()
    (folder / "SKILL.md").write_bytes(original)
    with pytest.raises(SkillConflictError, match="already exists"):
        atomic_write_primary(folder, payload("other"), overwrite=False)
    assert (folder / "SKILL.md").read_bytes() == original


def test_crash_during_non_overwriting_finalization_cleans_owned_files(
    tmp_path, monkeypatch
):
    folder = tmp_path / "atomic"
    folder.mkdir()
    import kestrel_feature_skills.store as module

    real_link = os.link
    calls = 0

    def fail_second_link(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated finalization crash")
        return real_link(source, destination)

    monkeypatch.setattr(module.os, "link", fail_second_link)
    with pytest.raises(OSError, match="simulated"):
        atomic_write_primary(folder, payload(), overwrite=False)
    assert not (folder / "SKILL.md").exists()
    assert not list(folder.glob(".SKILL.md.tmp.*"))
    assert not (folder / ".SKILL.md.claim").exists()


def test_concurrent_writers_exactly_one_wins(tmp_path):
    folder = tmp_path / "atomic"
    folder.mkdir()
    barrier = threading.Barrier(2, timeout=5)
    results = []

    def writer(index):
        barrier.wait()
        try:
            atomic_write_primary(folder, payload(), overwrite=False)
            results.append((index, "won"))
        except SkillConflictError:
            results.append((index, "lost"))

    threads = [threading.Thread(target=writer, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()
    assert sorted(result for _, result in results) == ["lost", "won"]
    assert validate_skill_folder(folder, source_root=tmp_path).name == "atomic"
    assert not (folder / ".SKILL.md.claim").exists()
    assert not list(folder.glob(".SKILL.md.tmp.*"))


def test_fresh_claim_blocks_writer_without_deleting_claim(tmp_path):
    folder = tmp_path / "atomic"
    folder.mkdir()
    claim = folder / ".SKILL.md.claim"
    claim.write_text("winner", encoding="utf-8")
    with pytest.raises(SkillConflictError, match="concurrent"):
        atomic_write_primary(folder, payload(), overwrite=False)
    assert claim.read_text(encoding="utf-8") == "winner"


def test_stale_claim_is_reclaimed(tmp_path):
    folder = tmp_path / "atomic"
    folder.mkdir()
    claim = folder / ".SKILL.md.claim"
    claim.write_text("orphan", encoding="utf-8")
    stale = time.time() - CLAIM_STALENESS_SECONDS - 5
    os.utime(claim, (stale, stale))
    atomic_write_primary(folder, payload(), overwrite=False)
    assert (folder / "SKILL.md").is_file()
    assert not claim.exists()


def test_two_edits_serialize_on_same_claim(tmp_path):
    folder = tmp_path / "atomic"
    folder.mkdir()
    atomic_write_primary(folder, payload(), overwrite=False)
    barrier = threading.Barrier(2, timeout=5)
    results = []

    def writer(index):
        barrier.wait()
        try:
            value = serialize_skill_markdown(
                SkillDocument("atomic", f"writer {index}", f"body {index}")
            ).encode()
            atomic_write_primary(folder, value, overwrite=True)
            results.append("won")
        except SkillConflictError:
            results.append("lost")

    threads = [threading.Thread(target=writer, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert sorted(results) == ["lost", "won"]
    assert validate_skill_folder(folder, source_root=tmp_path).description.startswith(
        "writer"
    )


def test_reclaimed_claim_fences_the_expired_writer(tmp_path, monkeypatch):
    folder = tmp_path / "atomic"
    folder.mkdir()
    atomic_write_primary(folder, payload(), overwrite=False)
    first_claimed = threading.Event()
    second_claimed = threading.Event()
    allow_second_finalize = threading.Event()
    real_link = os.link
    real_replace = os.replace
    results = {}

    def coordinated_link(source, destination):
        result = real_link(source, destination)
        if Path(destination).name != ".SKILL.md.claim":
            return result
        if threading.current_thread().name == "expired-writer":
            stale = time.time() - CLAIM_STALENESS_SECONDS - 5
            os.utime(destination, (stale, stale))
            first_claimed.set()
            assert second_claimed.wait(timeout=5)
        else:
            second_claimed.set()
        return result

    def coordinated_replace(source, destination):
        if threading.current_thread().name == "replacement-writer":
            assert allow_second_finalize.wait(timeout=5)
        return real_replace(source, destination)

    monkeypatch.setattr(os, "link", coordinated_link)
    monkeypatch.setattr(os, "replace", coordinated_replace)

    def write(label, description):
        try:
            atomic_write_primary(
                folder,
                serialize_skill_markdown(
                    SkillDocument("atomic", description, "body")
                ).encode(),
                overwrite=True,
            )
            results[label] = "won"
        except (SkillConflictError, FileNotFoundError):
            results[label] = "lost"
        finally:
            if label == "expired":
                allow_second_finalize.set()

    first = threading.Thread(
        target=write,
        args=("expired", "expired payload"),
        name="expired-writer",
    )
    second = threading.Thread(
        target=write,
        args=("replacement", "replacement payload"),
        name="replacement-writer",
    )
    first.start()
    assert first_claimed.wait(timeout=5)
    second.start()
    first.join(timeout=10)
    second.join(timeout=10)

    assert results == {"expired": "lost", "replacement": "won"}
    assert (
        validate_skill_folder(folder, source_root=tmp_path).description
        == "replacement payload"
    )


def test_limit_crossing_resource_edit_is_not_published(tmp_path):
    store = SkillStore(tmp_path / "skills")
    folder = store.create(SkillDocument("bounded", "Bounded", "Procedure."))
    record = SkillRecord(
        document=SkillDocument("bounded", "Bounded", "Procedure."),
        folder=folder,
        source_id="agent-local",
        source_kind="agent-local",
        precedence=0,
        provenance=SkillProvenance("agent-local", "agent-local", "bounded"),
    )
    resources = folder / "resources"
    resources.mkdir()
    for index in range(MAX_FOLDER_FILES - 1):
        (resources / f"{index:03}.md").write_text("x", encoding="utf-8")
    rejected = resources / "overflow.md"

    with pytest.raises(SkillFormatError, match="exceeds"):
        store.write_file(record, "resources/overflow.md", "overflow")

    assert not rejected.exists()
    assert validate_skill_folder(folder, source_root=store.local_root).name == "bounded"


def test_limit_crossing_primary_edit_preserves_the_original(tmp_path):
    store = SkillStore(tmp_path / "skills")
    folder = store.create(SkillDocument("bounded", "Bounded", "Procedure."))
    record = SkillRecord(
        document=SkillDocument("bounded", "Bounded", "Procedure."),
        folder=folder,
        source_id="agent-local",
        source_kind="agent-local",
        precedence=0,
        provenance=SkillProvenance("agent-local", "agent-local", "bounded"),
    )
    primary = folder / "SKILL.md"
    original = primary.read_bytes()
    resources = folder / "resources"
    resources.mkdir()
    (resources / "padding.md").write_bytes(
        b"x" * (MAX_FOLDER_BYTES - len(original) - 16)
    )
    validate_skill_folder(folder, source_root=store.local_root)
    replacement = serialize_skill_markdown(
        SkillDocument("bounded", "Bounded", "Procedure.\n\n" + ("y" * 128))
    )

    with pytest.raises(SkillFormatError, match="exceeds"):
        store.edit_primary(record, replacement)

    assert primary.read_bytes() == original
    assert validate_skill_folder(folder, source_root=store.local_root).name == "bounded"


def test_primary_path_alias_uses_the_serialized_primary_writer(tmp_path, monkeypatch):
    store = SkillStore(tmp_path / "skills")
    folder = store.create(SkillDocument("aliased", "Original", "Procedure."))
    record = SkillRecord(
        document=SkillDocument("aliased", "Original", "Procedure."),
        folder=folder,
        source_id="agent-local",
        source_kind="agent-local",
        precedence=0,
        provenance=SkillProvenance("agent-local", "agent-local", "aliased"),
    )
    replacement = serialize_skill_markdown(
        SkillDocument("aliased", "Replacement", "Changed procedure.")
    )
    calls = []

    def serialized_edit(selected, content):
        calls.append((selected, content))
        return selected.document

    monkeypatch.setattr(store, "edit_primary", serialized_edit)

    store.write_file(record, "./SKILL.md", replacement)

    assert calls == [(record, replacement)]


def test_install_copies_resources_then_publishes_primary(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = source_root / "remote"
    source.mkdir()
    (source / "scripts").mkdir()
    (source / "scripts" / "helper.py").write_text(
        "print('not executed')\n", encoding="utf-8"
    )
    document = SkillDocument("remote", "Remote", "See [helper](scripts/helper.py).")
    (source / "SKILL.md").write_text(
        serialize_skill_markdown(document), encoding="utf-8"
    )
    store = SkillStore(tmp_path / "local")
    folder = store.install_folder(
        source,
        provenance=SkillProvenance(
            kind="git",
            source_id="https://example.com/repo.git",
            locator="main:remote",
            revision="a" * 40,
            remote_url="https://example.com/repo.git",
        ),
    )
    assert (folder / "scripts" / "helper.py").read_text() == "print('not executed')\n"
    assert validate_skill_folder(folder, source_root=store.local_root) == document


def test_python_edit_is_scoped_to_scripts_and_never_runs(tmp_path):
    store = SkillStore(tmp_path / "skills")
    folder = store.create(SkillDocument("edit", "Edit", "Procedure."))
    record = SkillRecord(
        document=SkillDocument("edit", "Edit", "Procedure."),
        folder=folder,
        source_id="agent-local",
        source_kind="agent-local",
        precedence=0,
        provenance=SkillProvenance("agent-local", "agent-local", "edit"),
    )
    marker = tmp_path / "executed"
    source = f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n"
    store.write_file(record, "scripts/danger.py", source)
    assert not marker.exists()
    assert (folder / "scripts" / "danger.py").read_text() == source
    with pytest.raises(SkillPathError, match="only below scripts"):
        store.write_file(record, "danger.py", source)


def test_mutation_rejects_folder_replaced_by_symlink_after_discovery(tmp_path):
    store = SkillStore(tmp_path / "skills")
    folder = store.create(SkillDocument("swapped", "Swap guard", "Procedure."))
    record = SkillRecord(
        document=SkillDocument("swapped", "Swap guard", "Procedure."),
        folder=folder,
        source_id="agent-local",
        source_kind="agent-local",
        precedence=0,
        provenance=SkillProvenance("agent-local", "agent-local", "swapped"),
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "keep.md"
    marker.write_text("must survive", encoding="utf-8")
    shutil.rmtree(folder)
    folder.symlink_to(outside, target_is_directory=True)

    with pytest.raises(SkillPathError, match="changed after discovery"):
        store.write_file(record, "scripts/danger.py", "raise RuntimeError\n")
    with pytest.raises(SkillPathError, match="changed after discovery"):
        store.delete(record)
    assert marker.read_text(encoding="utf-8") == "must survive"
