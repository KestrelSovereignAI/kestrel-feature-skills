from __future__ import annotations

import errno
import fcntl
import multiprocessing
import os
import shutil
import stat
import threading
import time
from pathlib import Path

import pytest

import kestrel_feature_skills.sources as sources_module
import kestrel_feature_skills.store as store_module
from kestrel_feature_skills.errors import (
    SkillConflictError,
    SkillFormatError,
    SkillPathError,
    SkillPublicationCleanupError,
)
from kestrel_feature_skills.format import (
    MAX_FOLDER_BYTES,
    MAX_FOLDER_FILES,
    MAX_RESOURCE_PATH_BYTES,
    MAX_SKILL_FILE_BYTES,
    inspect_skill_folder,
    serialize_skill_markdown,
    validate_skill_folder,
)
from kestrel_feature_skills.models import SkillDocument, SkillProvenance, SkillRecord
from kestrel_feature_skills.sources import DirectorySkillSource
from kestrel_feature_skills.store import (
    CLAIM_STALENESS_SECONDS,
    INTERNAL_DIRECTORY,
    SkillStore,
    atomic_write_primary,
)


def payload(name="atomic"):
    return serialize_skill_markdown(
        SkillDocument(name, "Atomic write", "# Procedure\n\nWrite once.")
    ).encode("utf-8")


def git_provenance(name: str) -> SkillProvenance:
    """Return complete installed-source metadata for storage-focused tests."""

    remote_url = "https://example.com/repo.git"
    return SkillProvenance(
        kind="git",
        source_id=remote_url,
        locator=f"main:{name}",
        revision="a" * 40,
        remote_url=remote_url,
    )


def write_bounded_text_padding(root: Path, total_bytes: int) -> None:
    """Fill a folder without violating the per-resource read contract."""

    remaining = total_bytes
    index = 0
    while remaining:
        chunk = min(remaining, MAX_SKILL_FILE_BYTES)
        (root / f"padding-{index:02}.md").write_bytes(b"x" * chunk)
        remaining -= chunk
        index += 1


def _write_resource_from_separate_process(root, lock_state, result):
    """Discover and mutate a just-published skill from another process."""

    try:
        lock_path = (
            Path(root)
            / INTERNAL_DIRECTORY
            / store_module._mutation_lock_name(
                Path(root).resolve(),
                "cross-process-create",
            )
        )
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock_state.put("blocked")
            else:
                lock_state.put("acquired")
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
        store = SkillStore(Path(root))
        source = DirectorySkillSource(
            root=store.local_root,
            source_id="agent-local",
            kind="agent-local",
            precedence=0,
        )
        deadline = time.monotonic() + 5
        records, errors = source.discover()
        while not records and not errors and time.monotonic() < deadline:
            time.sleep(0.01)
            records, errors = source.discover()
        if errors or len(records) != 1:
            raise AssertionError(f"unexpected discovery result: {records=}, {errors=}")
        store.write_file(records[0], "notes.md", "Concurrent resource.\n")
    except Exception as exc:  # noqa: BLE001 - report child failures to parent
        result.put(("error", f"{type(exc).__name__}: {exc}"))
    else:
        result.put(("ok", ""))


def _try_publication_state_claim_from_separate_process(root, name, result):
    """Report whether another process can acquire a publication-state claim."""

    try:
        store = SkillStore(Path(root))
        claim = store.try_acquire_publication_state_claim(name)
        result.put(claim is not None)
        if claim is not None:
            claim.release()
    except Exception as exc:  # noqa: BLE001 - report child failures to parent
        result.put(f"{type(exc).__name__}: {exc}")


def _hold_git_checkout_workspace_from_separate_process(root, result):
    """Hold a checkout owner lock until the parent deliberately kills us."""

    try:
        store = SkillStore(Path(root))
        with store.git_checkout_workspace() as workspace:
            (workspace / "partial.pack").write_bytes(b"active checkout")
            result.put(("ready", str(workspace)))
            time.sleep(60)
    except Exception as exc:  # noqa: BLE001 - report child failures to parent
        result.put(("error", f"{type(exc).__name__}: {exc}"))


def _crash_after_delete_detach(root):
    """Exit immediately after a delete moves its expected public folder."""

    store = SkillStore(Path(root))
    record = DirectorySkillSource(
        root=store.local_root,
        source_id="agent-local",
        kind="agent-local",
        precedence=0,
    ).discover()[0][0]
    real_quarantine = store_module._quarantine_directory_at

    def detach_then_exit(*args, **kwargs):
        real_quarantine(*args, **kwargs)
        os._exit(86)

    store_module._quarantine_directory_at = detach_then_exit
    store.delete(record)
    os._exit(87)


def _crash_after_publication_rollback_detach(root, operation):
    """Exit after one create/install rollback detaches its public generation."""

    root = Path(root)
    store = SkillStore(root)
    name = f"crash-{operation.replace('_', '-')}"
    source = root.parent / "rollback-source" / name
    publication = None
    folder = None
    if operation in {"install_failure", "rollback_installed"}:
        source.mkdir(parents=True)
        (source / "SKILL.md").write_text(
            serialize_skill_markdown(
                SkillDocument(name, "Crash install rollback", "Unique procedure.")
            ),
            encoding="utf-8",
        )
    if operation == "rollback_created":
        folder, publication = store.create_pinned(
            SkillDocument(name, "Crash create rollback", "Unique procedure.")
        )
    elif operation == "rollback_installed":
        folder, publication = store.install_folder_pinned(
            source,
            provenance=git_provenance(name),
        )

    if operation in {"create_failure", "install_failure"}:
        real_fsync_pair = store_module._fsync_directory_pair

        def fail_post_publication_sync(first_fd, second_fd):
            if (root / name).is_dir():
                raise OSError(errno.EIO, "post-publication sync failed")
            return real_fsync_pair(first_fd, second_fd)

        store_module._fsync_directory_pair = fail_post_publication_sync

    real_quarantine = store_module._quarantine_directory_at

    def detach_then_exit(*args, **kwargs):
        real_quarantine(*args, **kwargs)
        os._exit(86)

    store_module._quarantine_directory_at = detach_then_exit
    if operation == "create_failure":
        store.create_pinned(
            SkillDocument(name, "Crash create cleanup", "Unique procedure.")
        )
    elif operation == "install_failure":
        store.install_folder_pinned(source, provenance=git_provenance(name))
    elif operation == "rollback_created":
        assert folder is not None and publication is not None
        store.rollback_created(folder, identity=publication)
    else:
        assert operation == "rollback_installed"
        assert folder is not None and publication is not None
        store.rollback_installed(folder, identity=publication)
    os._exit(87)


def test_atomic_create_leaves_no_temporary_or_claim_files(tmp_path):
    root = tmp_path / "skills"
    store = SkillStore(root)
    folder = store.create(SkillDocument("atomic", "Atomic write", "Do it."))
    assert (folder / "SKILL.md").is_file()
    assert not list(folder.glob(".SKILL.md.tmp.*"))
    assert not (folder / ".SKILL.md.claim").exists()


def test_create_rejects_staging_directory_fsync_failure(tmp_path, monkeypatch):
    store = SkillStore(tmp_path / "skills")
    real_fsync = os.fsync
    failed = False

    def fail_first_directory_fsync(descriptor):
        nonlocal failed
        if not failed and stat.S_ISDIR(os.fstat(descriptor).st_mode):
            failed = True
            raise OSError(errno.EIO, "staging directory fsync failed")
        return real_fsync(descriptor)

    monkeypatch.setattr(store_module.os, "fsync", fail_first_directory_fsync)

    with pytest.raises(OSError, match="staging directory fsync failed"):
        store.create(SkillDocument("fsync-stage-failure", "Durability", "Procedure."))

    assert failed
    assert not (store.local_root / "fsync-stage-failure").exists()


def test_create_rejects_publication_directory_fsync_failure(tmp_path, monkeypatch):
    store = SkillStore(tmp_path / "skills")
    real_fsync_pair = store_module._fsync_directory_pair
    failed = False

    def fail_first_pair(first_fd, second_fd):
        nonlocal failed
        if not failed:
            failed = True
            raise OSError(errno.ENOSPC, "publication directory fsync failed")
        return real_fsync_pair(first_fd, second_fd)

    monkeypatch.setattr(store_module, "_fsync_directory_pair", fail_first_pair)

    with pytest.raises(OSError, match="publication directory fsync failed"):
        store.create(
            SkillDocument("fsync-publication-failure", "Durability", "Procedure.")
        )

    assert failed
    assert not (store.local_root / "fsync-publication-failure").exists()


def test_create_publication_failure_preserves_directly_modified_folder(
    tmp_path, monkeypatch
):
    store = SkillStore(tmp_path / "skills")
    name = "create-publication-direct-edit"
    folder = store.local_root / name
    note = folder / "operator-note.md"
    real_fsync_pair = store_module._fsync_directory_pair
    failed = False

    def edit_after_publication_then_fail(first_fd, second_fd):
        nonlocal failed
        if not failed and folder.is_dir():
            note.write_text("must survive publication rollback", encoding="utf-8")
            failed = True
            raise OSError(errno.EIO, "publication sync failed after direct edit")
        return real_fsync_pair(first_fd, second_fd)

    monkeypatch.setattr(
        store_module,
        "_fsync_directory_pair",
        edit_after_publication_then_fail,
    )

    with pytest.raises(
        SkillPublicationCleanupError,
        match="modified folder was preserved",
    ):
        store.create(SkillDocument(name, "Direct edit", "Procedure."))

    assert failed
    assert note.read_text(encoding="utf-8") == "must survive publication rollback"


def test_publication_state_claim_is_exclusive_across_processes(tmp_path):
    root = tmp_path / "skills"
    store = SkillStore(root)
    claim = store.try_acquire_publication_state_claim("cross-process-state")
    assert claim is not None
    context = multiprocessing.get_context("spawn")

    blocked_result = context.Queue()
    blocked = context.Process(
        target=_try_publication_state_claim_from_separate_process,
        args=(str(root), "cross-process-state", blocked_result),
    )
    blocked.start()
    blocked.join(timeout=10)
    assert not blocked.is_alive()
    assert blocked.exitcode == 0
    assert blocked_result.get(timeout=10) is False

    claim.release()
    acquired_result = context.Queue()
    acquired = context.Process(
        target=_try_publication_state_claim_from_separate_process,
        args=(str(root), "cross-process-state", acquired_result),
    )
    acquired.start()
    acquired.join(timeout=10)
    assert not acquired.is_alive()
    assert acquired.exitcode == 0
    assert acquired_result.get(timeout=10) is True


def test_create_serializes_post_publication_validation_across_processes(
    tmp_path, monkeypatch
):
    root = tmp_path / "skills"
    store = SkillStore(root)
    published = threading.Event()
    release_publication = threading.Event()
    creation_errors = []
    real_publish = store_module._atomic_write_primary_at

    def publish_then_pause(*args, **kwargs):
        real_publish(*args, **kwargs)
        published.set()
        assert release_publication.wait(timeout=10)

    def create_skill():
        try:
            store.create(
                SkillDocument(
                    "cross-process-create",
                    "Cross-process creation",
                    "Procedure.",
                )
            )
        except Exception as exc:  # noqa: BLE001 - asserted in the parent
            creation_errors.append(exc)

    monkeypatch.setattr(
        store_module,
        "_atomic_write_primary_at",
        publish_then_pause,
    )
    creator = threading.Thread(target=create_skill)
    creator.start()
    assert published.wait(timeout=10)

    context = multiprocessing.get_context("spawn")
    lock_state = context.Queue()
    result = context.Queue()
    mutator = context.Process(
        target=_write_resource_from_separate_process,
        args=(str(root), lock_state, result),
    )
    mutator.start()
    observed_lock_state = lock_state.get(timeout=10)

    release_publication.set()
    creator.join(timeout=10)
    mutator.join(timeout=10)
    if mutator.is_alive():  # pragma: no cover - prevents a leaked test process
        mutator.terminate()
        mutator.join(timeout=5)
        pytest.fail("concurrent mutator did not finish")

    assert observed_lock_state == "blocked"
    assert creation_errors == []
    assert result.get(timeout=2) == ("ok", "")
    folder = root / "cross-process-create"
    assert (folder / "SKILL.md").is_file()
    assert (folder / "notes.md").read_text(encoding="utf-8") == (
        "Concurrent resource.\n"
    )


def test_create_cleans_folder_after_malformed_link_url(tmp_path):
    store = SkillStore(tmp_path / "skills")

    with pytest.raises(SkillPathError, match="malformed link URL"):
        store.create(SkillDocument("bad-url", "Bad URL", "[broken](//[invalid)"))

    assert not (store.local_root / "bad-url").exists()


def test_create_rejects_dangling_direct_child_symlink_without_orphan(tmp_path):
    store = SkillStore(tmp_path / "skills")
    alternate = store.local_root / "alternate-target"
    lexical = store.local_root / "dangling"
    lexical.symlink_to(alternate, target_is_directory=True)

    with pytest.raises(SkillPathError, match="symlink"):
        store.create(SkillDocument("dangling", "Dangling", "Procedure."))

    assert lexical.is_symlink()
    assert not alternate.exists()


def test_create_refuses_replaced_local_root_without_writing_through_symlink(tmp_path):
    root = tmp_path / "skills"
    store = SkillStore(root)
    displaced = tmp_path / "displaced-skills"
    outside = tmp_path / "outside"
    outside.mkdir()
    root.rename(displaced)
    root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(SkillPathError, match="real directory|directory changed|root"):
        store.create(SkillDocument("root-race", "Root race", "Procedure."))

    assert not (outside / "root-race").exists()
    assert not (displaced / "root-race").exists()


def test_constructor_refuses_local_root_symlink_swap_during_pinning(
    tmp_path, monkeypatch
):
    root = tmp_path / "skills"
    root.mkdir()
    displaced = tmp_path / "displaced-skills-at-init"
    outside = tmp_path / "outside-at-init"
    outside.mkdir()
    real_is_symlink = Path.is_symlink
    swapped = False

    def swap_after_symlink_check(path):
        nonlocal swapped
        result = real_is_symlink(path)
        if path == root and not swapped:
            swapped = True
            root.rename(displaced)
            root.symlink_to(outside, target_is_directory=True)
        return result

    monkeypatch.setattr(Path, "is_symlink", swap_after_symlink_check)

    with pytest.raises(
        SkillPathError,
        match="root|real directory|directory changed|symlink",
    ):
        SkillStore(root)

    assert root.is_symlink()
    assert displaced.is_dir()
    assert outside.is_dir()


def test_constructor_refuses_local_root_directory_swap_during_pinning(
    tmp_path, monkeypatch
):
    root = tmp_path / "skills"
    root.mkdir()
    displaced = tmp_path / "displaced-skills-directory-swap"
    replacement_marker = root / "replacement-must-survive.md"
    real_is_symlink = Path.is_symlink
    swapped = False

    def swap_after_symlink_check(path):
        nonlocal swapped
        result = real_is_symlink(path)
        if path == root and not swapped:
            swapped = True
            root.rename(displaced)
            root.mkdir()
            replacement_marker.write_text("replacement", encoding="utf-8")
        return result

    monkeypatch.setattr(Path, "is_symlink", swap_after_symlink_check)

    with pytest.raises(SkillPathError, match="directory changed|root"):
        SkillStore(root)

    assert displaced.is_dir()
    assert replacement_marker.read_text(encoding="utf-8") == "replacement"


def test_constructor_refuses_internal_directory_symlink(tmp_path):
    root = tmp_path / "skills"
    outside = tmp_path / "outside-internal"
    root.mkdir()
    outside.mkdir()
    (root / INTERNAL_DIRECTORY).symlink_to(outside, target_is_directory=True)

    with pytest.raises(SkillPathError, match="internal"):
        SkillStore(root)


def test_mutation_refuses_replaced_internal_directory(tmp_path):
    store = SkillStore(tmp_path / "skills")
    folder = store.create(SkillDocument("internal-root-race", "Original", "Procedure."))
    record = DirectorySkillSource(
        root=store.local_root,
        source_id="agent-local",
        kind="agent-local",
        precedence=0,
    ).discover()[0][0]
    original = (folder / "SKILL.md").read_bytes()
    displaced = tmp_path / "displaced-internal"
    store._internal_root.rename(displaced)
    replacement = store._internal_root
    replacement.mkdir()

    with pytest.raises(SkillPathError):
        store.edit_primary(
            record,
            serialize_skill_markdown(
                SkillDocument("internal-root-race", "Edited", "Changed.")
            ),
        )

    assert (folder / "SKILL.md").read_bytes() == original
    assert list(replacement.iterdir()) == []


def test_create_pins_publication_if_local_root_is_replaced_mid_write(
    tmp_path, monkeypatch
):
    root = tmp_path / "skills"
    store = SkillStore(root)
    displaced = tmp_path / "displaced-skills-mid-write"
    outside = tmp_path / "outside-mid-write"
    outside.mkdir()
    outside_decoy = outside / "root-race-mid-write"
    outside_decoy.mkdir()
    (outside_decoy / "marker.md").write_text("outside", encoding="utf-8")
    real_publish = store_module._atomic_write_primary_at
    swapped = False

    def swap_root_then_publish(*args, **kwargs):
        nonlocal swapped
        if not swapped:
            swapped = True
            root.rename(displaced)
            root.symlink_to(outside, target_is_directory=True)
        return real_publish(*args, **kwargs)

    monkeypatch.setattr(
        store_module,
        "_atomic_write_primary_at",
        swap_root_then_publish,
    )

    with pytest.raises(SkillPathError, match="real directory|directory changed"):
        store.create(SkillDocument("root-race-mid-write", "Root race", "Procedure."))

    assert not (outside_decoy / "SKILL.md").exists()
    assert (outside_decoy / "marker.md").read_text(encoding="utf-8") == "outside"
    assert not (displaced / "root-race-mid-write").exists()


def test_create_failure_cleanup_preserves_raced_replacement(tmp_path, monkeypatch):
    store = SkillStore(tmp_path / "skills")
    original = store.local_root / "create-cleanup-race"
    displaced = tmp_path / "displaced-create"
    replacement = None
    replacement_marker = None

    def swap_then_fail(*args, **kwargs):
        nonlocal replacement, replacement_marker
        candidates = list(
            store._internal_root.glob(
                f"{store_module.SKILL_PUBLICATION_STAGING_PREFIX}create-*"
            )
        )
        assert len(candidates) == 1
        replacement = candidates[0]
        replacement.rename(displaced)
        replacement.mkdir()
        replacement_marker = replacement / "replacement-must-survive.md"
        replacement_marker.write_text("replacement", encoding="utf-8")
        raise SkillPathError("simulated validation failure")

    monkeypatch.setattr(store_module, "_atomic_write_primary_at", swap_then_fail)

    with pytest.raises(SkillPathError, match="cleanup could not confirm removal"):
        store.create(SkillDocument("create-cleanup-race", "Cleanup race", "Procedure."))

    assert replacement_marker is not None
    assert replacement_marker.read_text(encoding="utf-8") == "replacement"
    assert replacement is not None and replacement.is_dir()
    assert not original.exists()
    assert not list(store.local_root.glob(".create-cleanup-race.delete.*"))
    assert displaced.is_dir()


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

    def fail_second_link(source, destination, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated finalization crash")
        return real_link(source, destination, **kwargs)

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


def test_stale_claim_reaps_the_crashed_writers_temporary_hardlink(tmp_path):
    folder = tmp_path / "atomic"
    folder.mkdir()
    atomic_write_primary(folder, payload(), overwrite=False)
    crashed_temporary = folder / ".SKILL.md.tmp.crashed-writer"
    crashed_temporary.write_bytes(payload("crashed"))
    claim = folder / ".SKILL.md.claim"
    os.link(crashed_temporary, claim)
    stale = time.time() - CLAIM_STALENESS_SECONDS - 5
    os.utime(claim, (stale, stale))

    atomic_write_primary(folder, payload(), overwrite=True)

    assert not claim.exists()
    assert not crashed_temporary.exists()
    assert not list(folder.glob(".SKILL.md.tmp.*"))
    assert validate_skill_folder(folder, source_root=tmp_path).name == "atomic"


def test_two_edits_serialize_on_same_claim(tmp_path, monkeypatch):
    folder = tmp_path / "atomic"
    folder.mkdir()
    atomic_write_primary(folder, payload(), overwrite=False)
    import kestrel_feature_skills.store as module

    claim_created = threading.Event()
    release_claim = threading.Event()
    results = []
    real_link = os.link

    def hold_first_claim(source, destination, **kwargs):
        result = real_link(source, destination, **kwargs)
        if Path(destination).name == ".SKILL.md.claim":
            claim_created.set()
            assert release_claim.wait(timeout=5)
        return result

    monkeypatch.setattr(module.os, "link", hold_first_claim)

    def writer(index):
        try:
            value = serialize_skill_markdown(
                SkillDocument("atomic", f"writer {index}", f"body {index}")
            ).encode()
            atomic_write_primary(folder, value, overwrite=True)
            results.append("won")
        except SkillConflictError:
            results.append("lost")

    winner = threading.Thread(target=writer, args=(0,))
    loser = threading.Thread(target=writer, args=(1,))
    winner.start()
    try:
        assert claim_created.wait(timeout=5)
        loser.start()
        loser.join(timeout=5)
        assert not loser.is_alive()
        assert results == ["lost"]
    finally:
        release_claim.set()
        winner.join(timeout=10)
        if loser.ident is not None:
            loser.join(timeout=10)
    assert not winner.is_alive()
    assert not loser.is_alive()
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

    def coordinated_link(source, destination, **kwargs):
        result = real_link(source, destination, **kwargs)
        if Path(destination).name != ".SKILL.md.claim":
            return result
        if threading.current_thread().name == "expired-writer":
            stale = time.time() - CLAIM_STALENESS_SECONDS - 5
            os.utime(
                destination,
                (stale, stale),
                dir_fd=kwargs.get("dst_dir_fd"),
                follow_symlinks=False,
            )
            first_claimed.set()
            assert second_claimed.wait(timeout=5)
        else:
            second_claimed.set()
        return result

    def coordinated_replace(source, destination, **kwargs):
        if threading.current_thread().name == "replacement-writer":
            assert allow_second_finalize.wait(timeout=5)
        return real_replace(source, destination, **kwargs)

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


def test_unexpected_claim_inode_is_rejected_before_locking(tmp_path, monkeypatch):
    claim = tmp_path / ".SKILL.md.claim"
    claim.write_text("replacement", encoding="utf-8")
    current = claim.stat()
    expected = (current.st_dev, current.st_ino + 1)
    lock_operations = []
    real_flock = store_module.fcntl.flock

    def tracked_flock(descriptor, operation):
        lock_operations.append(operation)
        return real_flock(descriptor, operation)

    monkeypatch.setattr(store_module.fcntl, "flock", tracked_flock)

    directory_fd = store_module._open_directory(tmp_path)
    try:
        assert (
            store_module._lock_claim_at(directory_fd, claim.name, expected=expected)
            is None
        )
    finally:
        os.close(directory_fd)
    assert lock_operations == []


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO requires POSIX")
def test_stale_fifo_claim_is_rejected_without_waiting_for_a_writer(
    tmp_path, monkeypatch
):
    claim = tmp_path / ".SKILL.md.claim"
    os.mkfifo(claim)
    stale = time.time() - CLAIM_STALENESS_SECONDS - 1
    os.utime(claim, (stale, stale))
    claim_stat = claim.stat()
    expected = (claim_stat.st_dev, claim_stat.st_ino)
    completed = threading.Event()
    result = {}
    lock_operations = []
    real_flock = store_module.fcntl.flock

    def tracked_flock(descriptor, operation):
        lock_operations.append(operation)
        return real_flock(descriptor, operation)

    monkeypatch.setattr(store_module.fcntl, "flock", tracked_flock)

    directory_fd = store_module._open_directory(tmp_path)

    def lock_stale_claim():
        try:
            result["descriptor"] = store_module._lock_claim_at(
                directory_fd,
                claim.name,
                expected=expected,
            )
        finally:
            completed.set()

    worker = threading.Thread(target=lock_stale_claim)
    worker.start()
    completed_without_writer = completed.wait(timeout=0.25)
    if not completed_without_writer:
        # Release the buggy blocking open so this regression never strands a
        # test worker while demonstrating the missing O_NONBLOCK/type guard.
        unblocker = os.open(claim, os.O_RDWR | getattr(os, "O_NONBLOCK", 0))
        os.close(unblocker)
    worker.join(timeout=5)
    os.close(directory_fd)

    assert not worker.is_alive()
    assert completed_without_writer, "stale FIFO claim blocked waiting for a writer"
    assert result["descriptor"] is None
    assert lock_operations == []


def test_primary_write_failure_does_not_leave_public_artifact_or_quarantine(
    tmp_path, monkeypatch
):
    store = SkillStore(tmp_path / "skills")
    folder = store.create(
        SkillDocument("primary-write-failure", "Original", "Procedure.")
    )
    record = DirectorySkillSource(
        root=store.local_root,
        source_id="agent-local",
        kind="agent-local",
        precedence=0,
    ).discover()[0][0]
    original = (folder / "SKILL.md").read_bytes()
    real_validate = store_module.validate_skill_folder
    injected = False

    def fail_staged_validation(candidate, *, source_root):
        nonlocal injected
        if candidate != folder:
            injected = True
            raise OSError("simulated primary write failure")
        return real_validate(candidate, source_root=source_root)

    monkeypatch.setattr(store_module, "validate_skill_folder", fail_staged_validation)

    with pytest.raises(OSError, match="primary write failure"):
        store.edit_primary(
            record,
            serialize_skill_markdown(
                SkillDocument("primary-write-failure", "Edited", "Changed.")
            ),
        )

    assert injected
    assert (folder / "SKILL.md").read_bytes() == original
    assert list(folder.glob(".SKILL.md.tmp.*")) == []
    assert list(store._internal_root.glob(".kestrel-skill-git-*")) == []
    records, errors = DirectorySkillSource(
        root=store.local_root,
        source_id="agent-local",
        kind="agent-local",
        precedence=0,
    ).discover()
    assert [item.name for item in records] == ["primary-write-failure"]
    assert errors == ()


def test_active_writer_claim_cannot_be_stolen_during_final_publication(
    tmp_path, monkeypatch
):
    folder = tmp_path / "atomic"
    folder.mkdir()
    atomic_write_primary(folder, payload(), overwrite=False)
    first_at_publish = threading.Event()
    allow_first_publish = threading.Event()
    real_replace = os.replace
    results = {}

    def pause_first_publish(source, destination, **kwargs):
        if (
            threading.current_thread().name == "first-writer"
            and Path(destination).name == "SKILL.md"
        ):
            stale = time.time() - CLAIM_STALENESS_SECONDS - 5
            os.utime(folder / ".SKILL.md.claim", (stale, stale))
            first_at_publish.set()
            assert allow_first_publish.wait(timeout=5)
        return real_replace(source, destination, **kwargs)

    monkeypatch.setattr(os, "replace", pause_first_publish)

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
        except SkillConflictError:
            results[label] = "lost"
        finally:
            if label == "second":
                allow_first_publish.set()

    first = threading.Thread(
        target=write,
        args=("first", "first payload"),
        name="first-writer",
    )
    second = threading.Thread(
        target=write,
        args=("second", "second payload"),
        name="second-writer",
    )
    first.start()
    assert first_at_publish.wait(timeout=5)
    second.start()
    second.join(timeout=10)
    first.join(timeout=10)

    assert not first.is_alive()
    assert not second.is_alive()
    assert results == {"second": "lost", "first": "won"}
    assert (
        validate_skill_folder(folder, source_root=tmp_path).description
        == "first payload"
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


def test_resource_write_failure_does_not_inventory_public_temporary(
    tmp_path, monkeypatch
):
    store = SkillStore(tmp_path / "skills")
    folder = store.create(
        SkillDocument("resource-write-failure", "Original", "Procedure.")
    )
    record = DirectorySkillSource(
        root=store.local_root,
        source_id="agent-local",
        kind="agent-local",
        precedence=0,
    ).discover()[0][0]
    real_validate = store_module.validate_skill_folder
    injected = False

    def fail_staged_validation(candidate, *, source_root):
        nonlocal injected
        if candidate != folder:
            injected = True
            raise OSError("simulated resource write failure")
        return real_validate(candidate, source_root=source_root)

    monkeypatch.setattr(store_module, "validate_skill_folder", fail_staged_validation)

    with pytest.raises(OSError, match="resource write failure"):
        store.write_file(record, "notes.md", "Replacement.")

    assert injected
    assert not (folder / "notes.md").exists()
    assert list(folder.glob(".tmp.*")) == []
    assert list(store._internal_root.glob(".kestrel-skill-git-*")) == []
    records, errors = DirectorySkillSource(
        root=store.local_root,
        source_id="agent-local",
        kind="agent-local",
        precedence=0,
    ).discover()
    assert [item.name for item in records] == ["resource-write-failure"]
    assert errors == ()


def test_nested_resource_write_failure_removes_new_empty_parent_directories(
    tmp_path, monkeypatch
):
    store = SkillStore(tmp_path / "skills")
    folder = store.create(
        SkillDocument("nested-write-failure", "Original", "Procedure.")
    )
    record = DirectorySkillSource(
        root=store.local_root,
        source_id="agent-local",
        kind="agent-local",
        precedence=0,
    ).discover()[0][0]

    def fail_publication(*args, **kwargs):
        raise OSError("simulated nested resource publication failure")

    monkeypatch.setattr(store_module, "_exchange_directories_at", fail_publication)

    with pytest.raises(OSError, match="nested resource publication failure"):
        store.write_file(record, "docs/new/note.md", "replacement")

    assert not (folder / "docs").exists()
    records, errors = DirectorySkillSource(
        root=store.local_root,
        source_id="agent-local",
        kind="agent-local",
        precedence=0,
    ).discover()
    assert [item.name for item in records] == ["nested-write-failure"]
    assert [entry["path"] for entry in store.tree(records[0])] == ["SKILL.md"]
    assert errors == ()
    recovery = tuple(
        store._internal_root.glob(f"{store_module.SKILL_RECOVERY_PREFIX}*")
    )
    assert len(recovery) == 1
    assert (
        recovery[0] / "nested-write-failure" / "docs" / "new" / "note.md"
    ).read_text(encoding="utf-8") == "replacement"


def test_nested_resource_write_failure_preserves_preexisting_parent(
    tmp_path, monkeypatch
):
    store = SkillStore(tmp_path / "skills")
    folder = store.create(
        SkillDocument("existing-parent-failure", "Original", "Procedure.")
    )
    (folder / "docs").mkdir()
    record = DirectorySkillSource(
        root=store.local_root,
        source_id="agent-local",
        kind="agent-local",
        precedence=0,
    ).discover()[0][0]

    def fail_publication(*args, **kwargs):
        raise OSError("simulated nested resource publication failure")

    monkeypatch.setattr(store_module, "_exchange_directories_at", fail_publication)

    with pytest.raises(OSError, match="nested resource publication failure"):
        store.write_file(record, "docs/new/note.md", "replacement")

    assert (folder / "docs").is_dir()
    assert list((folder / "docs").iterdir()) == []


def test_nested_resource_write_rollback_preserves_replaced_parent(
    tmp_path, monkeypatch
):
    store = SkillStore(tmp_path / "skills")
    folder = store.create(
        SkillDocument("replaced-parent-failure", "Original", "Procedure.")
    )
    record = DirectorySkillSource(
        root=store.local_root,
        source_id="agent-local",
        kind="agent-local",
        precedence=0,
    ).discover()[0][0]
    marker = folder / "docs" / "new" / "replacement-survived.md"
    real_exchange = store_module._exchange_directories_at
    injected = False

    def publish_after_external_parent_appears(*args, **kwargs):
        nonlocal injected
        if not injected:
            injected = True
            marker.parent.mkdir(parents=True)
            marker.write_text("replacement", encoding="utf-8")
        return real_exchange(*args, **kwargs)

    monkeypatch.setattr(
        store_module,
        "_exchange_directories_at",
        publish_after_external_parent_appears,
    )

    with pytest.raises(SkillConflictError, match="changed after it was read"):
        store.write_file(record, "docs/new/note.md", "replacement")

    assert injected
    assert marker.read_text(encoding="utf-8") == "replacement"


def test_second_exchange_preserves_late_public_change(tmp_path, monkeypatch):
    store = SkillStore(tmp_path / "skills")
    name = "second-exchange-race"
    folder = store.create(SkillDocument(name, "Original", "Original procedure."))
    record = DirectorySkillSource(
        root=store.local_root,
        source_id="agent-local",
        kind="agent-local",
        precedence=0,
    ).discover()[0][0]
    old_external = serialize_skill_markdown(
        SkillDocument(name, "Old external", "OLD-EXTERNAL procedure.")
    )
    late_public = serialize_skill_markdown(
        SkillDocument(name, "Late public", "LATE-PUBLIC procedure.")
    )
    desired = serialize_skill_markdown(
        SkillDocument(name, "Desired", "Desired procedure.")
    )
    real_exchange = store_module._exchange_directories_at
    exchange_count = 0

    def mutate_each_exchange(*args, **kwargs):
        nonlocal exchange_count
        exchange_count += 1
        if exchange_count == 1:
            (folder / "SKILL.md").write_text(old_external, encoding="utf-8")
        elif exchange_count == 2:
            (folder / "SKILL.md").write_text(late_public, encoding="utf-8")
        return real_exchange(*args, **kwargs)

    monkeypatch.setattr(
        store_module,
        "_exchange_directories_at",
        mutate_each_exchange,
    )

    with pytest.raises(SkillConflictError, match="changed after it was read"):
        store.edit_primary(record, desired)

    assert exchange_count == 2
    assert (folder / "SKILL.md").read_text(encoding="utf-8") == old_external
    preserved = list(
        (store.local_root / INTERNAL_DIRECTORY).glob(".kestrel-skill-recovery-*")
    )
    assert len(preserved) == 1
    assert (preserved[0] / "SKILL.md").read_text(encoding="utf-8") == late_public


def test_atomic_exchange_fsyncs_both_directory_entries(tmp_path, monkeypatch):
    store = SkillStore(tmp_path / "skills")
    name = "durable-exchange"
    store.create(SkillDocument(name, "Original", "Original procedure."))
    record = DirectorySkillSource(
        root=store.local_root,
        source_id="agent-local",
        kind="agent-local",
        precedence=0,
    ).discover()[0][0]
    real_fsync_pair = store_module._fsync_directory_pair
    synced_pairs = []

    def record_fsync_pair(first_fd, second_fd):
        synced_pairs.append((os.fstat(first_fd).st_ino, os.fstat(second_fd).st_ino))
        return real_fsync_pair(first_fd, second_fd)

    monkeypatch.setattr(store_module, "_fsync_directory_pair", record_fsync_pair)

    store.edit_primary(
        record,
        serialize_skill_markdown(SkillDocument(name, "Edited", "Edited procedure.")),
    )

    assert len(synced_pairs) == 1
    assert synced_pairs[0][0] != synced_pairs[0][1]


def test_validated_tree_fsync_visits_nested_resource_directories(tmp_path, monkeypatch):
    folder = tmp_path / "durable-tree"
    nested = folder / "resources" / "nested"
    nested.mkdir(parents=True)
    (folder / "SKILL.md").write_text(
        serialize_skill_markdown(
            SkillDocument("durable-tree", "Durable tree", "Procedure.")
        ),
        encoding="utf-8",
    )
    (nested / "notes.md").write_text("notes", encoding="utf-8")
    snapshot = inspect_skill_folder(folder, source_root=tmp_path)
    expected = {
        folder.stat().st_ino,
        (folder / "resources").stat().st_ino,
        nested.stat().st_ino,
    }
    synced = set()
    real_fsync = os.fsync

    def record_fsync(descriptor):
        synced.add(os.fstat(descriptor).st_ino)
        return real_fsync(descriptor)

    monkeypatch.setattr(store_module.os, "fsync", record_fsync)
    folder_fd = os.open(
        folder,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        store_module._fsync_validated_directories_at(folder_fd, snapshot)
    finally:
        os.close(folder_fd)

    assert expected <= synced


def test_edit_wires_nested_tree_fsync_before_exchange(tmp_path, monkeypatch):
    store = SkillStore(tmp_path / "skills")
    name = "durable-edit-tree"
    folder = store.create(SkillDocument(name, "Original", "Original procedure."))
    nested = folder / "resources" / "nested"
    nested.mkdir(parents=True)
    (nested / "notes.md").write_text("notes", encoding="utf-8")
    record = DirectorySkillSource(
        root=store.local_root,
        source_id="agent-local",
        kind="agent-local",
        precedence=0,
    ).discover()[0][0]
    real_fsync_tree = store_module._fsync_validated_directories_at
    calls = []

    def record_fsync_tree(folder_fd, snapshot):
        calls.append({entry.path for entry in snapshot.entries if entry.is_directory})
        return real_fsync_tree(folder_fd, snapshot)

    monkeypatch.setattr(
        store_module,
        "_fsync_validated_directories_at",
        record_fsync_tree,
    )

    store.edit_primary(
        record,
        serialize_skill_markdown(SkillDocument(name, "Edited", "Edited procedure.")),
    )

    assert calls == [{"resources", "resources/nested"}]


def test_edit_workspace_cleanup_preserves_raced_path_replacement(tmp_path, monkeypatch):
    store = SkillStore(tmp_path / "skills")
    name = "workspace-cleanup-race"
    store.create(SkillDocument(name, "Original", "Original procedure."))
    record = DirectorySkillSource(
        root=store.local_root,
        source_id="agent-local",
        kind="agent-local",
        precedence=0,
    ).discover()[0][0]
    victim = tmp_path / "victim"
    victim.mkdir()
    marker = victim / "DO-NOT-DELETE"
    marker.write_text("survive", encoding="utf-8")
    displaced_workspace = tmp_path / "displaced-edit-workspace"
    real_publish = store._publish_replacement_folder
    replacement_path = None

    def replace_workspace_after_publication(**kwargs):
        nonlocal replacement_path
        result = real_publish(**kwargs)
        staged_root = kwargs["staged_root"]
        staged_root.rename(displaced_workspace)
        victim.rename(staged_root)
        replacement_path = staged_root
        return result

    monkeypatch.setattr(
        store,
        "_publish_replacement_folder",
        replace_workspace_after_publication,
    )
    try:
        edited = store.edit_primary(
            record,
            serialize_skill_markdown(
                SkillDocument(name, "Edited", "Edited procedure.")
            ),
        )

        assert edited.description == "Edited"
        assert "Edited procedure." in (store.local_root / name / "SKILL.md").read_text(
            encoding="utf-8"
        )
        assert replacement_path is not None
        assert (replacement_path / marker.name).read_text(encoding="utf-8") == "survive"
    finally:
        shutil.rmtree(displaced_workspace, ignore_errors=True)
        if replacement_path is not None:
            shutil.rmtree(replacement_path, ignore_errors=True)


def test_resource_edit_does_not_report_failure_after_cleanup_refusal(
    tmp_path, monkeypatch
):
    store = SkillStore(tmp_path / "skills")
    name = "resource-cleanup-failure"
    store.create(SkillDocument(name, "Original", "Original procedure."))
    record = DirectorySkillSource(
        root=store.local_root,
        source_id="agent-local",
        kind="agent-local",
        precedence=0,
    ).discover()[0][0]

    def refuse_cleanup(_name, *, expected):
        del expected
        raise SkillPathError("workspace changed after validation")

    monkeypatch.setattr(store, "_remove_git_checkout_workspace", refuse_cleanup)

    store.write_file(record, "notes.md", "committed notes")

    assert (store.local_root / name / "notes.md").read_text(encoding="utf-8") == (
        "committed notes"
    )


def test_normal_lock_accumulation_does_not_consume_source_entry_budget(tmp_path):
    store = SkillStore(tmp_path / "skills")
    survivor = store.create(
        SkillDocument("lock-budget-survivor", "Survivor", "Procedure.")
    )
    for index in range(5_000):
        name = f"cycled-{index:04}"
        claim = store.try_acquire_publication_state_claim(name)
        assert claim is not None
        claim.release()
        with store_module._serialized_skill_mutation(
            store.local_root,
            store._internal_root,
            name,
            root_identity=store.local_root_identity,
            internal_root_identity=store._internal_root_identity,
        ):
            pass

    records, errors = DirectorySkillSource(
        root=store.local_root,
        source_id="agent-local",
        kind="agent-local",
        precedence=0,
    ).discover()

    assert survivor.is_dir()
    assert list(store.local_root.glob(".*.mutation.lock")) == []
    assert list(store.local_root.glob(".*.publication-state.lock")) == []
    assert len(list(store._internal_root.glob(".mutation-bucket-*.lock"))) <= 256
    assert (
        len(list(store._internal_root.glob(".publication-state-bucket-*.lock"))) <= 256
    )
    assert len(store_module._MUTATION_LOCKS) == 256
    assert len(store_module._PUBLICATION_STATE_LOCKS) == 256
    assert [item.name for item in records] == ["lock-budget-survivor"]
    assert errors == ()


def test_source_publication_locks_are_bounded_across_agent_roots(tmp_path):
    internal = tmp_path / "internal"
    internal.mkdir()
    internal_fd = os.open(
        internal,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        for index in range(1_000):
            with store_module._serialized_source_publication(
                tmp_path / f"agent-{index:04}" / "skills",
                internal_fd,
                enabled=True,
            ):
                pass
    finally:
        os.close(internal_fd)

    assert isinstance(store_module._SOURCE_PUBLICATION_LOCKS, tuple)
    assert len(store_module._SOURCE_PUBLICATION_LOCKS) == store_module.NAME_LOCK_BUCKETS


def test_crash_orphaned_edit_artifacts_are_outside_skill_discovery(tmp_path):
    store = SkillStore(tmp_path / "skills")
    survivor = store.create(
        SkillDocument("crash-artifact-survivor", "Survivor", "Procedure.")
    )
    (store._internal_root / ".crash-artifact-survivor.primary.tmp.crashed").write_text(
        "incomplete primary",
        encoding="utf-8",
    )
    (store._internal_root / ".crash-artifact-survivor.resource.tmp.crashed").write_text(
        "incomplete resource", encoding="utf-8"
    )

    records, errors = DirectorySkillSource(
        root=store.local_root,
        source_id="agent-local",
        kind="agent-local",
        precedence=0,
    ).discover()

    assert survivor.is_dir()
    assert [item.name for item in records] == ["crash-artifact-survivor"]
    assert errors == ()


def test_store_reaps_crash_orphaned_generation_temporaries(tmp_path, monkeypatch):
    store = SkillStore(tmp_path / "skills")
    survivor = store.create(
        SkillDocument("generation-survivor", "Survivor", "Procedure.")
    )
    crash_artifacts = tuple(
        store.local_root / f".kestrel-generation.tmp.crashed-{index}"
        for index in range(3)
    )
    for artifact in crash_artifacts:
        artifact.write_text("incomplete generation", encoding="utf-8")
    interrupted_reaper = (
        store._internal_root / ".kestrel-generation.reap.crashed-cleanup"
    )
    interrupted_reaper.write_text("retired generation", encoding="utf-8")

    reopened = SkillStore(store.local_root)
    monkeypatch.setattr(sources_module, "MAX_SOURCE_ENTRIES", 3)
    records, errors = DirectorySkillSource(
        root=reopened.local_root,
        source_id="agent-local",
        kind="agent-local",
        precedence=0,
        expected_root_identity=reopened.local_root_identity,
    ).discover()

    assert survivor.is_dir()
    assert not any(artifact.exists() for artifact in crash_artifacts)
    assert not interrupted_reaper.exists()
    assert [item.name for item in records] == ["generation-survivor"]
    assert errors == ()


def test_store_waits_for_live_generation_writer_before_reaping(tmp_path, monkeypatch):
    local = tmp_path / "skills"
    folder = local / "live-generation-writer"
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(
        serialize_skill_markdown(
            SkillDocument("live-generation-writer", "Live writer", "Procedure.")
        ),
        encoding="utf-8",
    )
    real_payload = sources_module._new_generation_payload
    writer_paused = threading.Event()
    release_writer = threading.Event()
    reaper_started = threading.Event()
    reaper_done = threading.Event()
    writer_results = []
    errors = []

    def pause_after_temporary_open():
        writer_paused.set()
        assert release_writer.wait(timeout=5)
        return real_payload()

    def discover():
        try:
            writer_results.append(
                DirectorySkillSource(
                    root=local,
                    source_id="agent-local",
                    kind="agent-local",
                    precedence=0,
                ).discover()
            )
        except Exception as exc:  # noqa: BLE001 - asserted below
            errors.append(exc)

    def reopen_store():
        reaper_started.set()
        try:
            SkillStore(local)
        except Exception as exc:  # noqa: BLE001 - asserted below
            errors.append(exc)
        finally:
            reaper_done.set()

    monkeypatch.setattr(
        sources_module,
        "_new_generation_payload",
        pause_after_temporary_open,
    )
    writer = threading.Thread(target=discover)
    writer.start()
    assert writer_paused.wait(timeout=5)
    assert any(
        path.name.startswith(".kestrel-generation.tmp.")
        for path in (local / INTERNAL_DIRECTORY).iterdir()
    )

    reaper = threading.Thread(target=reopen_store)
    reaper.start()
    assert reaper_started.wait(timeout=5)
    try:
        assert not reaper_done.wait(timeout=0.1), (
            "startup reaper did not wait for the live marker writer"
        )
    finally:
        release_writer.set()
    writer.join(timeout=5)
    reaper.join(timeout=5)

    assert not writer.is_alive()
    assert not reaper.is_alive()
    assert errors == []
    assert len(writer_results) == 1
    records, discovery_errors = writer_results[0]
    assert [record.name for record in records] == ["live-generation-writer"]
    assert discovery_errors == ()
    assert (folder / ".kestrel-generation").is_file()
    assert not any(
        path.name.startswith(".kestrel-generation.tmp.")
        for path in (local / INTERNAL_DIRECTORY).iterdir()
    )


@pytest.mark.parametrize("operation", ("create", "install"))
def test_publication_refuses_full_source_without_leaving_skill(
    tmp_path, monkeypatch, operation
):
    store = SkillStore(tmp_path / "skills")
    monkeypatch.setattr(sources_module, "MAX_SOURCE_ENTRIES", 3)
    monkeypatch.setattr(store_module, "MAX_SOURCE_ENTRIES", 3, raising=False)
    (store.local_root / "ordinary-one.txt").write_text("one", encoding="utf-8")
    (store.local_root / "ordinary-two.txt").write_text("two", encoding="utf-8")
    name = f"full-source-{operation}"

    with pytest.raises(SkillFormatError, match="source.*capacity"):
        if operation == "create":
            store.create(SkillDocument(name, "Full source", "Procedure."))
        else:
            source = tmp_path / "source" / name
            source.mkdir(parents=True)
            (source / "SKILL.md").write_text(
                serialize_skill_markdown(
                    SkillDocument(name, "Full source", "Procedure.")
                ),
                encoding="utf-8",
            )
            store.install_folder(
                source,
                provenance=git_provenance(name),
            )

    assert not (store.local_root / name).exists()
    assert {path.name for path in store.local_root.iterdir()} == {
        INTERNAL_DIRECTORY,
        "ordinary-one.txt",
        "ordinary-two.txt",
    }


def test_concurrent_publications_share_one_source_capacity_reservation(
    tmp_path, monkeypatch
):
    store = SkillStore(tmp_path / "skills")
    monkeypatch.setattr(store_module, "MAX_SOURCE_ENTRIES", 2)
    real_capacity_check = store_module._require_source_publication_capacity_at
    first_checked = threading.Event()
    release_first = threading.Event()
    results = []
    calls = 0

    def pause_first_capacity_check(root_fd):
        nonlocal calls
        real_capacity_check(root_fd)
        calls += 1
        if calls == 1:
            first_checked.set()
            assert release_first.wait(timeout=5)

    def create(name):
        try:
            store.create(SkillDocument(name, "Capacity race", "Procedure."))
        except Exception as exc:  # noqa: BLE001 - asserted below
            results.append((name, exc))
        else:
            results.append((name, None))

    monkeypatch.setattr(
        store_module,
        "_require_source_publication_capacity_at",
        pause_first_capacity_check,
    )
    first = threading.Thread(target=create, args=("capacity-race-one",))
    second = threading.Thread(target=create, args=("capacity-race-two",))
    first.start()
    assert first_checked.wait(timeout=5)
    second.start()
    try:
        second.join(timeout=0.1)
        assert second.is_alive()
        assert calls == 1
    finally:
        release_first.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert len(results) == 2
    assert sum(error is None for _name, error in results) == 1
    failures = [error for _name, error in results if error is not None]
    assert len(failures) == 1
    assert isinstance(failures[0], SkillFormatError)
    assert (
        len(
            [
                path
                for path in store.local_root.iterdir()
                if not path.name.startswith(".")
            ]
        )
        == 1
    )


def test_source_publication_rejects_nonregular_lock(tmp_path):
    store = SkillStore(tmp_path / "skills")
    lock = store._internal_root / ".source-publication.lock"
    os.mkfifo(lock)

    with pytest.raises(SkillPathError, match="source publication lock.*regular"):
        store.create(SkillDocument("nonregular-source-lock", "Lock type", "Procedure."))

    assert not (store.local_root / "nonregular-source-lock").exists()


def test_source_publication_rejects_replaced_lock(tmp_path, monkeypatch):
    store = SkillStore(tmp_path / "skills")
    lock = store._internal_root / ".source-publication.lock"
    real_identity = store_module._identity_at
    swapped = False

    def replace_before_identity_check(directory_fd, name):
        nonlocal swapped
        if name == lock.name and not swapped:
            swapped = True
            lock.unlink()
            lock.write_text("replacement", encoding="utf-8")
        return real_identity(directory_fd, name)

    monkeypatch.setattr(
        store_module,
        "_identity_at",
        replace_before_identity_check,
    )

    with pytest.raises(SkillPathError, match="source publication lock changed"):
        store.create(
            SkillDocument("replaced-source-lock", "Lock identity", "Procedure.")
        )

    assert swapped
    assert lock.read_text(encoding="utf-8") == "replacement"
    assert not (store.local_root / "replaced-source-lock").exists()


def test_concurrent_resource_edits_revalidate_the_serialized_folder(
    tmp_path, monkeypatch
):
    store = SkillStore(tmp_path / "skills")
    folder = store.create(SkillDocument("concurrent", "Concurrent", "Procedure."))
    record = SkillRecord(
        document=SkillDocument("concurrent", "Concurrent", "Procedure."),
        folder=folder,
        source_id="agent-local",
        source_kind="agent-local",
        precedence=0,
        provenance=SkillProvenance("agent-local", "agent-local", "concurrent"),
    )
    resources = folder / "resources"
    resources.mkdir()
    for index in range(MAX_FOLDER_FILES - 2):
        (resources / f"{index:03}.md").write_text("x", encoding="utf-8")

    import kestrel_feature_skills.store as module

    real_validate = module.validate_skill_folder
    first_validated = threading.Event()
    second_validated = threading.Event()
    release_first = threading.Event()
    validation_count = 0
    validation_guard = threading.Lock()

    def pause_first_staged_validation(candidate, *, source_root):
        nonlocal validation_count
        result = real_validate(candidate, source_root=source_root)
        if candidate != folder and candidate.name == record.name:
            with validation_guard:
                validation_count += 1
                position = validation_count
            if position == 1:
                first_validated.set()
                assert release_first.wait(timeout=5)
            else:
                second_validated.set()
        return result

    monkeypatch.setattr(module, "validate_skill_folder", pause_first_staged_validation)
    results = []

    def write(path):
        try:
            store.write_file(record, path, "new")
            results.append("published")
        except SkillFormatError:
            results.append("rejected")

    first = threading.Thread(target=write, args=("resources/first.md",))
    second = threading.Thread(target=write, args=("resources/second.md",))
    first.start()
    assert first_validated.wait(timeout=5)
    second.start()
    second_validated.wait(timeout=0.5)
    release_first.set()
    first.join(timeout=10)
    second.join(timeout=10)

    assert not first.is_alive() and not second.is_alive()
    assert sorted(results) == ["published", "rejected"]
    assert (
        validate_skill_folder(folder, source_root=store.local_root).name == "concurrent"
    )


def test_resource_edit_rejects_symlink_swap_to_primary_after_staging(
    tmp_path, monkeypatch
):
    store = SkillStore(tmp_path / "skills")
    folder = store.create(SkillDocument("swapped", "Original", "Procedure."))
    notes = folder / "notes.md"
    notes.write_text("original notes", encoding="utf-8")
    primary = folder / "SKILL.md"
    original_primary = primary.read_bytes()
    record = SkillRecord(
        document=SkillDocument("swapped", "Original", "Procedure."),
        folder=folder,
        source_id="agent-local",
        source_kind="agent-local",
        precedence=0,
        provenance=SkillProvenance("agent-local", "agent-local", "swapped"),
    )
    import kestrel_feature_skills.store as module

    real_validate = module.validate_skill_folder
    swapped = False

    def swap_after_staged_validation(candidate, *, source_root):
        nonlocal swapped
        result = real_validate(candidate, source_root=source_root)
        if candidate != folder and not swapped:
            notes.unlink()
            notes.symlink_to(primary.name)
            swapped = True
        return result

    monkeypatch.setattr(module, "validate_skill_folder", swap_after_staged_validation)

    with pytest.raises(SkillPathError, match="symlinks"):
        store.write_file(record, "notes.md", "not valid SKILL.md frontmatter")

    assert primary.read_bytes() == original_primary
    assert notes.is_symlink()


def test_read_pins_intermediate_directories_against_symlink_swap(tmp_path, monkeypatch):
    store = SkillStore(tmp_path / "skills")
    folder = store.create(SkillDocument("read-race", "Read race", "Procedure."))
    references = folder / "references"
    references.mkdir()
    (references / "notes.md").write_text("inside", encoding="utf-8")
    source = DirectorySkillSource(
        root=store.local_root,
        source_id="agent-local",
        kind="agent-local",
        precedence=0,
    )
    record = source.discover()[0][0]
    outside = tmp_path / "outside-read"
    outside.mkdir()
    (outside / "notes.md").write_text("OUTSIDE-SECRET", encoding="utf-8")
    real_inspect = store_module.inspect_skill_folder_descriptor
    swapped = False

    def swap_before_snapshot(directory_fd, *, folder_name):
        nonlocal swapped
        if not swapped:
            shutil.rmtree(references)
            references.symlink_to(outside, target_is_directory=True)
            swapped = True
        return real_inspect(directory_fd, folder_name=folder_name)

    monkeypatch.setattr(
        store_module,
        "inspect_skill_folder_descriptor",
        swap_before_snapshot,
    )

    with pytest.raises(SkillPathError, match="symlink|directory"):
        store.read_file(record, "references/notes.md")

    assert (outside / "notes.md").read_text(encoding="utf-8") == "OUTSIDE-SECRET"


def test_read_rejects_replaced_fifo_without_waiting_for_a_writer(tmp_path):
    store = SkillStore(tmp_path / "skills")
    folder = store.create(SkillDocument("fifo-read", "FIFO read", "Procedure."))
    resource = folder / "notes.md"
    resource.write_text("ordinary resource", encoding="utf-8")
    record = DirectorySkillSource(
        root=store.local_root,
        source_id="agent-local",
        kind="agent-local",
        precedence=0,
    ).discover()[0][0]
    resource.unlink()
    os.mkfifo(resource)
    completed = threading.Event()
    errors = []

    def read_replaced_resource():
        try:
            store.read_file(record, "notes.md")
        except Exception as exc:  # noqa: BLE001 - asserted below
            errors.append(exc)
        finally:
            completed.set()

    reader = threading.Thread(target=read_replaced_resource)
    reader.start()
    finished_without_writer = completed.wait(timeout=0.5)
    if not finished_without_writer:
        writer = os.open(resource, os.O_WRONLY | os.O_NONBLOCK)
        os.close(writer)
    reader.join(timeout=5)

    assert finished_without_writer, "resource read blocked while opening a FIFO"
    assert not reader.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], SkillPathError)
    assert "regular file" in str(errors[0])


def test_read_keeps_max_length_resource_traversal_descriptor_relative(
    tmp_path, monkeypatch
):
    store = SkillStore(tmp_path / "skills")
    folder = store.create(
        SkillDocument("long-read", "Long descriptor read", "Procedure.")
    )
    parts = tuple(character * 250 for character in "abcd")
    filename = "notes.md"
    relative_path = "/".join((*parts, filename))
    assert len(relative_path.encode("utf-8")) <= MAX_RESOURCE_PATH_BYTES

    descriptor = os.open(folder, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        for part in parts:
            os.mkdir(part, mode=0o700, dir_fd=descriptor)
            child = os.open(
                part,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
        resource = os.open(
            filename,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=descriptor,
        )
        try:
            os.write(resource, b"descriptor-relative resource")
        finally:
            os.close(resource)
    finally:
        os.close(descriptor)

    source = DirectorySkillSource(
        root=store.local_root,
        source_id="agent-local",
        kind="agent-local",
        precedence=0,
    )
    records, errors = source.discover()
    assert errors == ()
    record = records[0]
    assert relative_path in {item["path"] for item in store.tree(record)}

    def absolute_resolution_would_overflow(*_args, **_kwargs):
        raise OSError(errno.ENAMETOOLONG, "simulated macOS PATH_MAX")

    monkeypatch.setattr(
        store_module,
        "lexical_contained_path",
        absolute_resolution_would_overflow,
    )

    assert store.read_file(record, relative_path) == "descriptor-relative resource"


def test_write_pins_intermediate_directories_against_symlink_swap(
    tmp_path, monkeypatch
):
    store = SkillStore(tmp_path / "skills")
    folder = store.create(SkillDocument("write-race", "Write race", "Procedure."))
    references = folder / "references"
    references.mkdir()
    (references / "notes.md").write_text("inside", encoding="utf-8")
    source = DirectorySkillSource(
        root=store.local_root,
        source_id="agent-local",
        kind="agent-local",
        precedence=0,
    )
    record = source.discover()[0][0]
    outside = tmp_path / "outside-write"
    outside.mkdir()
    outside_notes = outside / "notes.md"
    outside_notes.write_text("OUTSIDE-ORIGINAL", encoding="utf-8")
    displaced = tmp_path / "displaced-references"
    real_exchange = store_module._exchange_directories_at
    swapped = False

    def swap_at_exchange(*args, **kwargs):
        nonlocal swapped
        if not swapped:
            swapped = True
            references.rename(displaced)
            references.symlink_to(outside, target_is_directory=True)
        return real_exchange(*args, **kwargs)

    monkeypatch.setattr(store_module, "_exchange_directories_at", swap_at_exchange)

    with pytest.raises(SkillPathError, match="symlink|directory"):
        store.write_file(record, "references/notes.md", "replacement")

    assert swapped
    assert outside_notes.read_text(encoding="utf-8") == "OUTSIDE-ORIGINAL"


def test_primary_edit_pins_skill_folder_against_replacement(tmp_path, monkeypatch):
    store = SkillStore(tmp_path / "skills")
    folder = store.create(
        SkillDocument("primary-race", "Primary race", "Original procedure.")
    )
    source = DirectorySkillSource(
        root=store.local_root,
        source_id="agent-local",
        kind="agent-local",
        precedence=0,
    )
    record = source.discover()[0][0]
    outside = tmp_path / "outside-primary"
    outside.mkdir()
    outside_primary = outside / "SKILL.md"
    outside_primary.write_text("OUTSIDE-PRIMARY", encoding="utf-8")
    displaced = tmp_path / "displaced-primary"
    real_open_directory_at = store_module._open_directory_at
    swapped = False

    def swap_before_pinned_open(parent_fd, name, *, expected=None):
        nonlocal swapped
        if name == record.name and not swapped:
            folder.rename(displaced)
            outside.rename(folder)
            swapped = True
        return real_open_directory_at(parent_fd, name, expected=expected)

    monkeypatch.setattr(store_module, "_open_directory_at", swap_before_pinned_open)
    replacement = serialize_skill_markdown(
        SkillDocument("primary-race", "Replacement", "Changed procedure.")
    )

    with pytest.raises(SkillPathError, match="directory"):
        store.edit_primary(record, replacement)

    assert (folder / "SKILL.md").read_text(encoding="utf-8") == "OUTSIDE-PRIMARY"


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
    write_bounded_text_padding(
        resources,
        MAX_FOLDER_BYTES - len(original) - 16,
    )
    validate_skill_folder(folder, source_root=store.local_root)
    replacement = serialize_skill_markdown(
        SkillDocument("bounded", "Bounded", "Procedure.\n\n" + ("y" * 128))
    )

    with pytest.raises(SkillFormatError, match="exceeds"):
        store.edit_primary(record, replacement)

    assert primary.read_bytes() == original
    assert validate_skill_folder(folder, source_root=store.local_root).name == "bounded"


def test_primary_path_alias_is_rejected_before_reaching_a_writer(tmp_path, monkeypatch):
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

    with pytest.raises(SkillPathError, match="canonical"):
        store.write_file(record, "./SKILL.md", replacement)

    assert calls == []


def test_resource_edit_supports_maximum_length_filename(tmp_path):
    store = SkillStore(tmp_path / "skills")
    document = SkillDocument("long-resource", "Long resource", "Procedure.")
    folder = store.create(document)
    folder_stat = folder.stat()
    record = SkillRecord(
        document=document,
        folder=folder,
        source_id="agent-local",
        source_kind="agent-local",
        precedence=0,
        provenance=SkillProvenance("agent-local", "agent-local", document.name),
        folder_identity=(folder_stat.st_dev, folder_stat.st_ino),
    )
    filename = f"{'a' * 252}.md"
    assert len(filename.encode("utf-8")) == 255

    store.write_file(record, filename, "replacement")

    assert (folder / filename).read_text(encoding="utf-8") == "replacement"
    assert not tuple(folder.glob(".tmp.*"))
    assert not tuple(folder.glob(".*.tmp.*"))


def test_install_copies_resources_then_fsyncs_tree_and_publishes_primary(
    tmp_path,
    monkeypatch,
):
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = source_root / "remote"
    source.mkdir()
    (source / "scripts" / "nested").mkdir(parents=True)
    (source / "scripts" / "nested" / "helper.py").write_text(
        "print('not executed')\n", encoding="utf-8"
    )
    document = SkillDocument(
        "remote",
        "Remote",
        "See [helper](scripts/nested/helper.py).",
    )
    (source / "SKILL.md").write_text(
        serialize_skill_markdown(document), encoding="utf-8"
    )
    store = SkillStore(tmp_path / "local")
    real_fsync_tree = store_module._fsync_validated_directories_at
    calls = []

    def record_fsync_tree(folder_fd, snapshot):
        calls.append({entry.path for entry in snapshot.entries if entry.is_directory})
        return real_fsync_tree(folder_fd, snapshot)

    monkeypatch.setattr(
        store_module,
        "_fsync_validated_directories_at",
        record_fsync_tree,
    )
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
    assert (folder / "scripts" / "nested" / "helper.py").read_text() == (
        "print('not executed')\n"
    )
    assert calls == [{"scripts", "scripts/nested"}]
    assert validate_skill_folder(folder, source_root=store.local_root) == document


def test_install_rejects_publication_directory_fsync_failure(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    source = source_root / "remote-fsync-failure"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text(
        serialize_skill_markdown(
            SkillDocument("remote-fsync-failure", "Remote", "Procedure.")
        ),
        encoding="utf-8",
    )
    store = SkillStore(tmp_path / "local")
    real_fsync_pair = store_module._fsync_directory_pair
    failed = False

    def fail_first_pair(first_fd, second_fd):
        nonlocal failed
        if not failed:
            failed = True
            raise OSError(errno.EIO, "install publication fsync failed")
        return real_fsync_pair(first_fd, second_fd)

    monkeypatch.setattr(store_module, "_fsync_directory_pair", fail_first_pair)

    with pytest.raises(OSError, match="install publication fsync failed"):
        store.install_folder(
            source,
            provenance=git_provenance("remote-fsync-failure"),
        )

    assert failed
    assert not (store.local_root / "remote-fsync-failure").exists()


def test_install_publication_failure_preserves_directly_modified_folder(
    tmp_path, monkeypatch
):
    source = tmp_path / "source" / "install-publication-direct-edit"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text(
        serialize_skill_markdown(
            SkillDocument(source.name, "Direct edit", "Procedure.")
        ),
        encoding="utf-8",
    )
    store = SkillStore(tmp_path / "local")
    folder = store.local_root / source.name
    note = folder / "operator-note.md"
    real_fsync_pair = store_module._fsync_directory_pair
    failed = False

    def edit_after_publication_then_fail(first_fd, second_fd):
        nonlocal failed
        if not failed and folder.is_dir():
            note.write_text("must survive publication rollback", encoding="utf-8")
            failed = True
            raise OSError(errno.EIO, "publication sync failed after direct edit")
        return real_fsync_pair(first_fd, second_fd)

    monkeypatch.setattr(
        store_module,
        "_fsync_directory_pair",
        edit_after_publication_then_fail,
    )

    with pytest.raises(
        SkillPublicationCleanupError,
        match="modified folder was preserved",
    ):
        store.install_folder(source, provenance=git_provenance(source.name))

    assert failed
    assert note.read_text(encoding="utf-8") == "must survive publication rollback"


def test_install_rollback_preserves_a_changed_publication(tmp_path):
    source_root = tmp_path / "rollback-source"
    source = source_root / "rollback-install"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text(
        serialize_skill_markdown(
            SkillDocument("rollback-install", "Remote", "Procedure.")
        ),
        encoding="utf-8",
    )
    (source / "notes.md").write_text("published", encoding="utf-8")
    store = SkillStore(tmp_path / "rollback-local")
    folder, publication = store.install_folder_pinned(
        source,
        provenance=git_provenance("rollback-install"),
    )
    (folder / "notes.md").write_text("changed after publication", encoding="utf-8")

    with pytest.raises(SkillPathError, match="contents changed"):
        store.rollback_installed(folder, identity=publication)

    assert (folder / "notes.md").read_text(encoding="utf-8") == (
        "changed after publication"
    )
    assert not tuple(store.local_root.glob(".rollback-install.delete.*"))


def test_install_rollback_quarantines_before_comparing_publication(
    tmp_path, monkeypatch
):
    source_root = tmp_path / "rollback-race-source"
    source = source_root / "rollback-race-install"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text(
        serialize_skill_markdown(
            SkillDocument("rollback-race-install", "Remote", "Procedure.")
        ),
        encoding="utf-8",
    )
    (source / "notes.md").write_text("published", encoding="utf-8")
    store = SkillStore(tmp_path / "rollback-race-local")
    folder, publication = store.install_folder_pinned(
        source,
        provenance=git_provenance("rollback-race-install"),
    )
    inspected = False
    real_inspect = store_module._inspect_child_at
    real_quarantine = store_module._quarantine_directory_at

    def observe_inspection(*args, **kwargs):
        nonlocal inspected
        inspected = True
        return real_inspect(*args, **kwargs)

    def require_detached_before_inspection(*args, **kwargs):
        assert not inspected, (
            "rollback compared the live publication before detaching it, leaving "
            "a time-of-check/time-of-delete window for direct filesystem edits"
        )
        return real_quarantine(*args, **kwargs)

    monkeypatch.setattr(store_module, "_inspect_child_at", observe_inspection)
    monkeypatch.setattr(
        store_module, "_quarantine_directory_at", require_detached_before_inspection
    )

    store.rollback_installed(folder, identity=publication)

    assert not folder.exists()


def test_install_rollback_preserves_changed_quarantine_and_raced_replacement(
    tmp_path, monkeypatch
):
    name = "rollback-restore-race"
    source = tmp_path / "rollback-restore-source" / name
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text(
        serialize_skill_markdown(SkillDocument(name, "Remote", "Procedure.")),
        encoding="utf-8",
    )
    (source / "notes.md").write_text("published", encoding="utf-8")
    store = SkillStore(tmp_path / "rollback-restore-local")
    folder, publication = store.install_folder_pinned(
        source,
        provenance=git_provenance(name),
    )
    real_inspect = store_module._inspect_child_at
    replacement_marker = folder / "replacement.md"

    def change_quarantine_and_republish(root_fd, child_name, **kwargs):
        quarantined = store._internal_root / child_name
        (quarantined / "notes.md").write_text(
            "changed while detached", encoding="utf-8"
        )
        folder.mkdir()
        replacement_marker.write_text("replacement", encoding="utf-8")
        (folder / "SKILL.md").write_text(
            serialize_skill_markdown(
                SkillDocument(name, "Raced replacement", "Different procedure.")
            ),
            encoding="utf-8",
        )
        return real_inspect(root_fd, child_name, **kwargs)

    monkeypatch.setattr(
        store_module,
        "_inspect_child_at",
        change_quarantine_and_republish,
    )

    with pytest.raises(SkillPathError, match="contents changed"):
        store.rollback_installed(folder, identity=publication)

    assert replacement_marker.read_text(encoding="utf-8") == "replacement"
    quarantines = list(
        store._internal_root.glob(f"{store_module.SKILL_RECOVERY_PREFIX}*")
    )
    assert len(quarantines) == 1
    assert (quarantines[0] / "notes.md").read_text(encoding="utf-8") == (
        "changed while detached"
    )


def test_install_rejects_generated_provenance_crossing_file_limit(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = source_root / "file-boundary"
    source.mkdir()
    document = SkillDocument("file-boundary", "Boundary", "Procedure.")
    (source / "SKILL.md").write_text(
        serialize_skill_markdown(document), encoding="utf-8"
    )
    for index in range(MAX_FOLDER_FILES - 1):
        (source / f"{index:03}.md").write_bytes(b"")
    store = SkillStore(tmp_path / "local")

    with pytest.raises(SkillFormatError, match="exceeds.*files"):
        store.install_folder(
            source,
            provenance=git_provenance(document.name),
        )

    assert not (store.local_root / document.name).exists()


def test_created_publication_rollback_removes_only_unchanged_contents(tmp_path):
    store = SkillStore(tmp_path / "created-rollback-local")
    document = SkillDocument("created-rollback", "Created", "Procedure.")
    folder, publication = store.create_pinned(document)

    store.rollback_created(folder, identity=publication)

    assert not folder.exists()


def test_created_publication_rollback_restores_changed_contents(tmp_path):
    store = SkillStore(tmp_path / "changed-created-rollback-local")
    document = SkillDocument(
        "changed-created-rollback",
        "Created",
        "Procedure.",
    )
    folder, publication = store.create_pinned(document)
    marker = folder / "operator-note.md"
    marker.write_text("must survive", encoding="utf-8")

    with pytest.raises(SkillPathError, match="contents changed"):
        store.rollback_created(folder, identity=publication)

    assert marker.read_text(encoding="utf-8") == "must survive"
    assert not tuple(store.local_root.glob(".changed-created-rollback.delete.*"))


def test_install_rejects_generated_provenance_crossing_byte_limit(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = source_root / "byte-boundary"
    source.mkdir()
    document = SkillDocument("byte-boundary", "Boundary", "Procedure.")
    primary = serialize_skill_markdown(document).encode("utf-8")
    (source / "SKILL.md").write_bytes(primary)
    write_bounded_text_padding(source, MAX_FOLDER_BYTES - len(primary))
    store = SkillStore(tmp_path / "local")

    with pytest.raises(SkillFormatError, match="exceeds.*bytes"):
        store.install_folder(
            source,
            provenance=git_provenance(document.name),
        )

    assert not (store.local_root / document.name).exists()


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

    with pytest.raises(SkillFormatError, match="UTF-8"):
        store.write_file(record, "notes.md", "bad\ud800")
    assert not (folder / "notes.md").exists()


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


@pytest.mark.parametrize("operation", ("edit", "delete"))
def test_mutation_lock_refuses_replaced_local_root_without_outside_write(
    tmp_path, operation
):
    root = tmp_path / "skills"
    store = SkillStore(root)
    store.create(SkillDocument("root-lock-race", "Root lock race", "Procedure."))
    source = DirectorySkillSource(
        root=store.local_root,
        source_id="agent-local",
        kind="agent-local",
        precedence=0,
    )
    record = source.discover()[0][0]
    displaced = tmp_path / "displaced-root-lock"
    outside = tmp_path / "outside-root-lock"
    outside.mkdir()
    root.rename(displaced)
    root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(SkillPathError):
        if operation == "edit":
            store.edit_primary(
                record,
                serialize_skill_markdown(
                    SkillDocument(
                        "root-lock-race",
                        "Changed",
                        "Changed procedure.",
                    )
                ),
            )
        else:
            store.delete(record)

    assert not (outside / ".root-lock-race.mutation.lock").exists()
    assert (displaced / "root-lock-race" / "SKILL.md").is_file()


@pytest.mark.parametrize("operation", ("edit", "delete"))
def test_mutation_rejects_real_folder_replacement_after_discovery(tmp_path, operation):
    store = SkillStore(tmp_path / "skills")
    original = store.create(SkillDocument("replaced", "Original", "Procedure."))
    source = DirectorySkillSource(
        root=store.local_root,
        source_id="agent-local",
        kind="agent-local",
        precedence=0,
    )
    record = source.discover()[0][0]
    original.rename(tmp_path / "old-replaced")
    replacement = store.local_root / "replaced"
    replacement.mkdir()
    marker = replacement / "marker.md"
    marker.write_text("replacement must survive", encoding="utf-8")
    (replacement / "SKILL.md").write_text(
        serialize_skill_markdown(
            SkillDocument("replaced", "Replacement", "Different procedure.")
        ),
        encoding="utf-8",
    )

    with pytest.raises(SkillPathError, match="changed after discovery"):
        if operation == "edit":
            store.write_file(record, "notes.md", "must not publish")
        else:
            store.delete(record)

    assert marker.read_text(encoding="utf-8") == "replacement must survive"


def test_rollback_created_preserves_replacement_swapped_after_identity_check(
    tmp_path, monkeypatch
):
    store = SkillStore(tmp_path / "skills")
    folder = store.create(SkillDocument("rollback-race", "Rollback", "Procedure."))
    identity = (folder.stat().st_dev, folder.stat().st_ino)
    displaced = tmp_path / "displaced-rollback"
    marker = folder / "replacement-must-survive.md"
    real_rename_no_replace = store_module._rename_directory_no_replace_at
    swapped = False

    def swap_before_removal(
        directory_fd,
        source_name,
        destination_name,
        *,
        destination_directory_fd=None,
    ):
        nonlocal swapped
        if source_name == folder.name and not swapped:
            swapped = True
            folder.rename(displaced)
            folder.mkdir()
            marker.write_text("replacement", encoding="utf-8")
        return real_rename_no_replace(
            directory_fd,
            source_name,
            destination_name,
            destination_directory_fd=destination_directory_fd,
        )

    monkeypatch.setattr(
        store_module, "_rename_directory_no_replace_at", swap_before_removal
    )

    with pytest.raises(SkillPathError, match="changed|preserved"):
        store.rollback_created(folder, identity=identity)

    quarantines = list(
        store._internal_root.glob(f"{store_module.SKILL_RECOVERY_PREFIX}*")
    )
    assert len(quarantines) == 1
    assert (quarantines[0] / marker.name).read_text(encoding="utf-8") == "replacement"
    assert (displaced / "SKILL.md").is_file()


def test_delete_refuses_replacement_swapped_in_after_identity_verification(
    tmp_path, monkeypatch
):
    store = SkillStore(tmp_path / "skills")
    original = store.create(SkillDocument("delete-race", "Original", "Procedure."))
    source = DirectorySkillSource(
        root=store.local_root,
        source_id="agent-local",
        kind="agent-local",
        precedence=0,
    )
    record = source.discover()[0][0]
    displaced = tmp_path / "displaced-original"
    replacement_marker = original / "replacement-must-survive.md"
    real_require = store._require_local

    def swap_after_verification(candidate):
        verified = real_require(candidate)
        verified.rename(displaced)
        verified.mkdir()
        replacement_marker.write_text("replacement", encoding="utf-8")
        return verified

    monkeypatch.setattr(store, "_require_local", swap_after_verification)

    with pytest.raises(SkillPathError, match="changed during deletion"):
        store.delete(record)

    assert replacement_marker.read_text(encoding="utf-8") == "replacement"
    assert (displaced / "SKILL.md").is_file()


def test_install_refuses_replaced_local_root_without_writing_through_symlink(tmp_path):
    root = tmp_path / "skills"
    store = SkillStore(root)
    source_root = tmp_path / "source"
    source = source_root / "remote-root-race"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text(
        serialize_skill_markdown(
            SkillDocument("remote-root-race", "Remote root race", "Procedure.")
        ),
        encoding="utf-8",
    )
    displaced = tmp_path / "displaced-install-root"
    outside = tmp_path / "outside-install"
    outside.mkdir()
    root.rename(displaced)
    root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(SkillPathError, match="real directory|directory changed|root"):
        store.install_folder(
            source,
            provenance=SkillProvenance(
                "git",
                "https://example.com/repo.git",
                "main:remote-root-race",
                revision="a" * 40,
                remote_url="https://example.com/repo.git",
            ),
        )

    assert not (outside / "remote-root-race").exists()
    assert not (displaced / "remote-root-race").exists()


def test_install_pins_publication_if_local_root_is_replaced_mid_write(
    tmp_path, monkeypatch
):
    root = tmp_path / "skills"
    store = SkillStore(root)
    source_root = tmp_path / "source-mid-write"
    source = source_root / "install-root-race-mid-write"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text(
        serialize_skill_markdown(
            SkillDocument(
                "install-root-race-mid-write",
                "Install root race",
                "Procedure.",
            )
        ),
        encoding="utf-8",
    )
    displaced = tmp_path / "displaced-install-mid-write"
    outside = tmp_path / "outside-install-mid-write"
    outside.mkdir()
    outside_decoy = outside / "install-root-race-mid-write"
    outside_decoy.mkdir()
    (outside_decoy / "marker.md").write_text("outside", encoding="utf-8")
    real_publish = store_module._atomic_replace_file_at
    swapped = False

    def swap_root_then_publish(*args, **kwargs):
        nonlocal swapped
        if not swapped:
            swapped = True
            root.rename(displaced)
            root.symlink_to(outside, target_is_directory=True)
        return real_publish(*args, **kwargs)

    monkeypatch.setattr(
        store_module,
        "_atomic_replace_file_at",
        swap_root_then_publish,
    )

    with pytest.raises(SkillPathError, match="real directory|directory changed"):
        store.install_folder(
            source,
            provenance=SkillProvenance(
                "git",
                "https://example.com/repo.git",
                "main:install-root-race-mid-write",
                revision="c" * 40,
                remote_url="https://example.com/repo.git",
            ),
        )

    assert not (outside_decoy / ".kestrel-provenance.json").exists()
    assert not (outside_decoy / "SKILL.md").exists()
    assert (outside_decoy / "marker.md").read_text(encoding="utf-8") == "outside"
    assert not (displaced / "install-root-race-mid-write").exists()


def test_install_failure_cleanup_preserves_raced_replacement(tmp_path, monkeypatch):
    store = SkillStore(tmp_path / "skills")
    source_root = tmp_path / "source"
    source = source_root / "install-cleanup-race"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text(
        serialize_skill_markdown(
            SkillDocument("install-cleanup-race", "Install cleanup", "Procedure.")
        ),
        encoding="utf-8",
    )
    target = store.local_root / "install-cleanup-race"
    displaced = tmp_path / "displaced-install"
    replacement = None
    marker = None

    def swap_then_fail(*_args, **_kwargs):
        nonlocal marker, replacement
        candidates = list(
            store._internal_root.glob(
                f"{store_module.SKILL_PUBLICATION_STAGING_PREFIX}install-*"
            )
        )
        assert len(candidates) == 1
        replacement = candidates[0]
        replacement.rename(displaced)
        replacement.mkdir()
        marker = replacement / "replacement-must-survive.md"
        marker.write_text("replacement", encoding="utf-8")
        raise OSError("simulated provenance failure")

    monkeypatch.setattr(store_module, "_atomic_replace_file_at", swap_then_fail)

    with pytest.raises(SkillPathError, match="cleanup could not confirm removal"):
        store.install_folder(
            source,
            provenance=SkillProvenance(
                "git",
                "https://example.com/repo.git",
                "main:install-cleanup-race",
                revision="b" * 40,
                remote_url="https://example.com/repo.git",
            ),
        )

    assert marker is not None
    assert marker.read_text(encoding="utf-8") == "replacement"
    assert replacement is not None and replacement.is_dir()
    assert not target.exists()
    assert not list(store.local_root.glob(".install-cleanup-race.delete.*"))
    assert displaced.is_dir()


def test_delete_preserves_replacement_swapped_during_atomic_quarantine(
    tmp_path, monkeypatch
):
    store = SkillStore(tmp_path / "skills")
    original = store.create(
        SkillDocument("delete-quarantine-race", "Original", "Procedure.")
    )
    source = DirectorySkillSource(
        root=store.local_root,
        source_id="agent-local",
        kind="agent-local",
        precedence=0,
    )
    record = source.discover()[0][0]
    displaced = tmp_path / "displaced-quarantine-original"
    replacement_marker = original / "replacement-must-return.md"
    real_rename_no_replace = store_module._rename_directory_no_replace_at
    swapped = False

    def swap_as_quarantine_starts(
        directory_fd,
        source_name,
        destination_name,
        *,
        destination_directory_fd=None,
    ):
        nonlocal swapped
        if (
            source_name == record.name
            and destination_name.startswith(store_module.SKILL_RECOVERY_PREFIX)
            and not swapped
        ):
            swapped = True
            original.rename(displaced)
            original.mkdir()
            replacement_marker.write_text("replacement", encoding="utf-8")
        return real_rename_no_replace(
            directory_fd,
            source_name,
            destination_name,
            destination_directory_fd=destination_directory_fd,
        )

    monkeypatch.setattr(
        store_module,
        "_rename_directory_no_replace_at",
        swap_as_quarantine_starts,
    )

    with pytest.raises(SkillPathError, match="changed during deletion"):
        store.delete(record)

    quarantines = list(
        store._internal_root.glob(f"{store_module.SKILL_RECOVERY_PREFIX}*")
    )
    assert len(quarantines) == 1
    assert (quarantines[0] / replacement_marker.name).read_text(
        encoding="utf-8"
    ) == "replacement"
    assert (displaced / "SKILL.md").is_file()
    assert not original.exists()


def test_delete_pins_private_quarantine_during_validation(tmp_path, monkeypatch):
    store = SkillStore(tmp_path / "skills")
    folder = store.create(
        SkillDocument("delete-recursive-race", "Original", "Procedure.")
    )
    (folder / "resources").mkdir()
    (folder / "resources" / "notes.md").write_text("notes", encoding="utf-8")
    source = DirectorySkillSource(
        root=store.local_root,
        source_id="agent-local",
        kind="agent-local",
        precedence=0,
    )
    record = source.discover()[0][0]
    displaced = tmp_path / "displaced-recursive-original"
    replacement_marker = None
    real_open_directory_at = store_module._open_directory_at
    swapped = False

    def swap_quarantine_before_recursive_removal(
        parent_fd,
        name,
        *,
        expected=None,
    ):
        nonlocal replacement_marker, swapped
        if name.startswith(store_module.SKILL_RECOVERY_PREFIX) and not swapped:
            swapped = True
            quarantines = list(
                store._internal_root.glob(f"{store_module.SKILL_RECOVERY_PREFIX}*")
            )
            assert len(quarantines) == 1
            quarantine = quarantines[0]
            quarantine.rename(displaced)
            quarantine.mkdir()
            replacement_marker = quarantine / "replacement-must-survive.md"
            replacement_marker.write_text("replacement", encoding="utf-8")
        return real_open_directory_at(parent_fd, name, expected=expected)

    monkeypatch.setattr(
        store_module,
        "_open_directory_at",
        swap_quarantine_before_recursive_removal,
    )
    try:
        with pytest.raises(SkillPathError, match="changed after validation"):
            store.delete(record)

        assert replacement_marker is not None
        assert replacement_marker.read_text(encoding="utf-8") == "replacement"
        assert displaced.is_dir()
    finally:
        shutil.rmtree(displaced, ignore_errors=True)
        for quarantine in store._internal_root.glob(
            f"{store_module.SKILL_RECOVERY_PREFIX}*"
        ):
            shutil.rmtree(quarantine, ignore_errors=True)


def test_delete_crash_does_not_strand_a_public_quarantine(tmp_path):
    root = tmp_path / "skills"
    store = SkillStore(root)
    name = "crash-delete-recovery"
    store.create(SkillDocument(name, "Crash delete", "Procedure."))
    context = multiprocessing.get_context("spawn")
    deleter = context.Process(
        target=_crash_after_delete_detach,
        args=(str(root),),
    )

    deleter.start()
    deleter.join(timeout=10)
    if deleter.is_alive():  # pragma: no cover - prevents a leaked test process
        deleter.terminate()
        deleter.join(timeout=5)
        pytest.fail("crashing deleter did not finish")

    assert deleter.exitcode == 86
    SkillStore(root)
    assert not tuple(root.glob(f".{name}.delete.*"))
    recovery = tuple(
        (root / INTERNAL_DIRECTORY).glob(f"{store_module.SKILL_RECOVERY_PREFIX}*")
    )
    assert len(recovery) == 1
    assert (recovery[0] / "SKILL.md").is_file()


@pytest.mark.parametrize(
    "operation",
    (
        "create_failure",
        "install_failure",
        "rollback_created",
        "rollback_installed",
    ),
)
def test_publication_rollback_crash_does_not_strand_public_quarantine(
    tmp_path, operation
):
    root = tmp_path / "skills"
    SkillStore(root)
    name = f"crash-{operation.replace('_', '-')}"
    context = multiprocessing.get_context("spawn")
    worker = context.Process(
        target=_crash_after_publication_rollback_detach,
        args=(str(root), operation),
    )

    worker.start()
    worker.join(timeout=10)
    if worker.is_alive():  # pragma: no cover - prevents a leaked test process
        worker.terminate()
        worker.join(timeout=5)
        pytest.fail("crashing publication rollback did not finish")

    assert worker.exitcode == 86
    assert not (root / name).exists()
    assert not tuple(root.glob(f".{name}.delete.*"))
    SkillStore(root)
    recovery = tuple(
        (root / INTERNAL_DIRECTORY).glob(f"{store_module.SKILL_RECOVERY_PREFIX}*")
    )
    assert len(recovery) == 1
    assert (recovery[0] / "SKILL.md").is_file()


def test_delete_retires_whole_generation_without_unlinking_nested_entries(
    tmp_path,
    monkeypatch,
):
    store = SkillStore(tmp_path / "skills")
    folder = store.create(SkillDocument("delete-nested-race", "Original", "Procedure."))
    resources = folder / "resources"
    resources.mkdir()
    (resources / "original.md").write_text("original", encoding="utf-8")
    record = DirectorySkillSource(
        root=store.local_root,
        source_id="agent-local",
        kind="agent-local",
        precedence=0,
    ).discover()[0][0]
    real_purge = store_module._purge_internal_directory_at
    observed_before_purge = []

    def observe_whole_retired_generation(internal_fd, retired_name, *, expected):
        retired = store.local_root / INTERNAL_DIRECTORY / retired_name
        observed_before_purge.append(
            (retired / "resources" / "original.md").read_text(encoding="utf-8")
        )
        return real_purge(internal_fd, retired_name, expected=expected)

    monkeypatch.setattr(
        store_module,
        "_purge_internal_directory_at",
        observe_whole_retired_generation,
    )
    store.delete(record)

    assert not folder.exists()
    assert observed_before_purge == ["original"]
    assert not tuple(
        (store.local_root / INTERNAL_DIRECTORY).glob(
            f"{store_module.SKILL_TRASH_PREFIX}*"
        )
    )


def test_delete_never_unlinks_replacement_at_recovery_visible_name(
    tmp_path,
    monkeypatch,
):
    store = SkillStore(tmp_path / "skills")
    name = "delete-retirement-race"
    folder = store.create(SkillDocument(name, "Original", "Procedure."))
    original_marker = folder / "ORIGINAL-DATA"
    original_marker.write_text("original", encoding="utf-8")
    record = DirectorySkillSource(
        root=store.local_root,
        source_id="agent-local",
        kind="agent-local",
        precedence=0,
    ).discover()[0][0]
    real_rename_no_replace = store_module._rename_directory_no_replace_at
    raced_name = None

    def replace_source_after_retirement(
        directory_fd,
        source_name,
        destination_name,
        *,
        destination_directory_fd=None,
    ):
        nonlocal raced_name
        result = real_rename_no_replace(
            directory_fd,
            source_name,
            destination_name,
            destination_directory_fd=destination_directory_fd,
        )
        if source_name.startswith(
            store_module.SKILL_RECOVERY_PREFIX
        ) and destination_name.startswith(store_module.SKILL_TRASH_PREFIX):
            raced_name = source_name
            replacement = store._internal_root / source_name
            replacement.mkdir()
            (replacement / "REPLACEMENT-DATA").write_text(
                "replacement",
                encoding="utf-8",
            )
        return result

    monkeypatch.setattr(
        store_module,
        "_rename_directory_no_replace_at",
        replace_source_after_retirement,
    )

    store.delete(record)

    assert raced_name is not None
    assert (store._internal_root / raced_name / "REPLACEMENT-DATA").read_text(
        encoding="utf-8"
    ) == "replacement"
    assert not tuple(
        (store.local_root / INTERNAL_DIRECTORY).glob(
            f"{store_module.SKILL_TRASH_PREFIX}*"
        )
    )


def test_delete_preserves_retired_generation_when_private_purge_fails(
    tmp_path,
    monkeypatch,
):
    store = SkillStore(tmp_path / "skills")
    name = "delete-purge-retry"
    folder = store.create(SkillDocument(name, "Original", "Procedure."))
    marker = folder / "RECOVERY-DATA"
    marker.write_text("recoverable", encoding="utf-8")
    record = DirectorySkillSource(
        root=store.local_root,
        source_id="agent-local",
        kind="agent-local",
        precedence=0,
    ).discover()[0][0]
    real_purge = store_module._purge_internal_directory_at

    def fail_private_purge(*_args, **_kwargs):
        raise OSError("simulated private trash purge failure")

    monkeypatch.setattr(
        store_module,
        "_purge_internal_directory_at",
        fail_private_purge,
    )

    store.delete(record)

    retired = list(
        (store.local_root / INTERNAL_DIRECTORY).glob(
            f"{store_module.SKILL_TRASH_PREFIX}*"
        )
    )
    assert len(retired) == 1
    assert (retired[0] / marker.name).read_text(encoding="utf-8") == "recoverable"

    monkeypatch.setattr(store_module, "_purge_internal_directory_at", real_purge)
    SkillStore(store.local_root)

    assert not tuple(
        (store.local_root / INTERNAL_DIRECTORY).glob(
            f"{store_module.SKILL_TRASH_PREFIX}*"
        )
    )


def test_repeated_create_delete_does_not_accumulate_private_trash(tmp_path):
    store = SkillStore(tmp_path / "skills")
    source = DirectorySkillSource(
        root=store.local_root,
        source_id="agent-local",
        kind="agent-local",
        precedence=0,
    )

    for index in range(20):
        name = f"repeat-delete-{index}"
        store.create(SkillDocument(name, "Temporary", "Procedure."))
        record = source.discover()[0][0]
        store.delete(record)

    assert not tuple(
        (store.local_root / INTERNAL_DIRECTORY).glob(
            f"{store_module.SKILL_TRASH_PREFIX}*"
        )
    )


def test_delete_preserves_quarantined_folder_if_retirement_fails(tmp_path, monkeypatch):
    store = SkillStore(tmp_path / "skills")
    original = store.create(
        SkillDocument("delete-restore", "Restore on failure", "Procedure.")
    )
    (original / "resources").mkdir()
    (original / "resources" / "notes.md").write_text("notes", encoding="utf-8")
    source = DirectorySkillSource(
        root=store.local_root,
        source_id="agent-local",
        kind="agent-local",
        precedence=0,
    )
    record = source.discover()[0][0]

    real_rename_no_replace = store_module._rename_directory_no_replace_at

    def fail_retirement(
        directory_fd,
        source_name,
        destination_name,
        *,
        destination_directory_fd=None,
    ):
        if source_name.startswith(
            store_module.SKILL_RECOVERY_PREFIX
        ) and destination_name.startswith(store_module.SKILL_TRASH_PREFIX):
            raise OSError("simulated retirement failure")
        return real_rename_no_replace(
            directory_fd,
            source_name,
            destination_name,
            destination_directory_fd=destination_directory_fd,
        )

    monkeypatch.setattr(
        store_module,
        "_rename_directory_no_replace_at",
        fail_retirement,
    )

    with pytest.raises(OSError, match="simulated retirement failure"):
        store.delete(record)

    quarantines = list(
        store._internal_root.glob(f"{store_module.SKILL_RECOVERY_PREFIX}*")
    )
    assert len(quarantines) == 1
    assert (quarantines[0] / "SKILL.md").is_file()
    assert not original.exists()


def test_delete_accepts_a_maximum_length_resource_component(tmp_path):
    store = SkillStore(tmp_path / "skills")
    folder = store.create(
        SkillDocument("delete-long-resource", "Long resource", "Procedure.")
    )
    resource = folder / f"{'a' * 252}.md"
    assert len(resource.name.encode()) == 255
    resource.write_text("long resource", encoding="utf-8")
    record = DirectorySkillSource(
        root=store.local_root,
        source_id="agent-local",
        kind="agent-local",
        precedence=0,
    ).discover()[0][0]

    store.delete(record)

    assert not folder.exists()


@pytest.mark.parametrize("operation", ("primary", "resource"))
def test_edit_bounds_external_folder_before_staging_copy(
    tmp_path, monkeypatch, operation
):
    store = SkillStore(tmp_path / "skills")
    folder = store.create(
        SkillDocument("bounded-edit", "Bound before copy", "Procedure.")
    )
    record = DirectorySkillSource(
        root=store.local_root,
        source_id="agent-local",
        kind="agent-local",
        precedence=0,
    ).discover()[0][0]
    for index in range(512):
        (folder / f"external-{index:03}.md").write_text("x", encoding="utf-8")

    def reject_unbounded_copy(*_args, **_kwargs):
        raise AssertionError("unbounded path copy ran before folder limits")

    monkeypatch.setattr(store_module.shutil, "copytree", reject_unbounded_copy)

    with pytest.raises(SkillFormatError, match="exceeds.*(?:files|entries)"):
        if operation == "primary":
            store.edit_primary(
                record,
                serialize_skill_markdown(
                    SkillDocument("bounded-edit", "Edited", "Procedure.")
                ),
            )
        else:
            store.write_file(record, "notes.md", "edited")


def test_tree_inventory_does_not_reopen_folder_after_descriptor_validation(
    tmp_path, monkeypatch
):
    store = SkillStore(tmp_path / "skills")
    folder = store.create(SkillDocument("pinned-tree", "Pinned tree", "Procedure."))
    (folder / "inside.md").write_text("inside", encoding="utf-8")
    record = DirectorySkillSource(
        root=store.local_root,
        source_id="agent-local",
        kind="agent-local",
        precedence=0,
    ).discover()[0][0]
    displaced = tmp_path / "pinned-tree-original"
    outside = tmp_path / "outside-tree"
    outside.mkdir()
    (outside / "outside-secret.md").write_text("secret", encoding="utf-8")
    real_validate = store_module.validate_skill_folder

    def replace_after_validation(candidate, *, source_root):
        document = real_validate(candidate, source_root=source_root)
        candidate.rename(displaced)
        candidate.symlink_to(outside, target_is_directory=True)
        return document

    monkeypatch.setattr(
        store_module,
        "validate_skill_folder",
        replace_after_validation,
    )

    entries = store.tree(record)

    assert {entry["path"] for entry in entries} == {"SKILL.md", "inside.md"}


def test_install_keeps_target_hidden_until_complete(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    source = source_root / "hidden-install"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text(
        serialize_skill_markdown(
            SkillDocument("hidden-install", "Hidden install", "Procedure.")
        ),
        encoding="utf-8",
    )
    (source / "notes.md").write_text("resource", encoding="utf-8")
    store = SkillStore(tmp_path / "skills")
    target = store.local_root / "hidden-install"
    real_copy = store_module._copy_regular_file_at
    observed_copy = False

    def assert_target_hidden(*args, **kwargs):
        nonlocal observed_copy
        observed_copy = True
        assert not target.exists(), "public target appeared before install completed"
        return real_copy(*args, **kwargs)

    monkeypatch.setattr(store_module, "_copy_regular_file_at", assert_target_hidden)

    installed = store.install_folder(
        source,
        provenance=git_provenance("hidden-install"),
    )

    assert observed_copy
    assert installed == target
    assert (target / "SKILL.md").is_file()


def test_create_keeps_target_hidden_until_complete(tmp_path, monkeypatch):
    store = SkillStore(tmp_path / "skills")
    target = store.local_root / "hidden-create"
    real_write = store_module._atomic_write_primary_at
    observed_write = False

    def assert_target_hidden(*args, **kwargs):
        nonlocal observed_write
        observed_write = True
        assert not target.exists(), "public target appeared before create completed"
        return real_write(*args, **kwargs)

    monkeypatch.setattr(store_module, "_atomic_write_primary_at", assert_target_hidden)

    created = store.create(
        SkillDocument("hidden-create", "Hidden create", "Procedure.")
    )

    assert observed_write
    assert created == target
    assert (target / "SKILL.md").is_file()


@pytest.mark.parametrize("operation", ("create", "install"))
def test_publication_staging_uses_private_recovery_domain(
    tmp_path, monkeypatch, operation
):
    name = f"private-staging-{operation}"
    store = SkillStore(tmp_path / "skills")
    real_write = store_module._atomic_write_primary_at
    observed_staging = False

    def assert_private_staging(*args, **kwargs):
        nonlocal observed_staging
        observed_staging = True
        assert not tuple(store.local_root.glob(f".{name}.{operation}.*"))
        assert tuple(
            store._internal_root.glob(
                f"{store_module.SKILL_PUBLICATION_STAGING_PREFIX}{operation}-*"
            )
        )
        return real_write(*args, **kwargs)

    monkeypatch.setattr(
        store_module,
        "_atomic_write_primary_at",
        assert_private_staging,
    )

    if operation == "create":
        published = store.create(SkillDocument(name, "Private staging", "Procedure."))
    else:
        source = tmp_path / "source" / name
        source.mkdir(parents=True)
        (source / "SKILL.md").write_text(
            serialize_skill_markdown(
                SkillDocument(name, "Private staging", "Procedure.")
            ),
            encoding="utf-8",
        )
        published = store.install_folder(source, provenance=git_provenance(name))

    assert observed_staging
    assert published == store.local_root / name
    assert not tuple(
        store._internal_root.glob(
            f"{store_module.SKILL_PUBLICATION_STAGING_PREFIX}{operation}-*"
        )
    )


def test_store_reaps_crash_orphaned_publication_staging(tmp_path):
    root = tmp_path / "skills"
    store = SkillStore(root)
    orphan = (
        store._internal_root
        / f"{store_module.SKILL_PUBLICATION_STAGING_PREFIX}create-crashed"
    )
    orphan.mkdir()
    (orphan / "partial.md").write_text("partial publication", encoding="utf-8")

    SkillStore(root)

    assert not orphan.exists()


def test_store_does_not_reap_live_publication_staging(tmp_path, monkeypatch):
    root = tmp_path / "skills"
    store = SkillStore(root)
    publication_started = threading.Event()
    release_publication = threading.Event()
    constructor_finished = threading.Event()
    errors = []
    real_write = store_module._atomic_write_primary_at

    def pause_live_publication(*args, **kwargs):
        publication_started.set()
        assert release_publication.wait(timeout=5)
        return real_write(*args, **kwargs)

    def publish():
        try:
            store.create(SkillDocument("live-staging", "Live staging", "Procedure."))
        except Exception as exc:  # noqa: BLE001 - asserted below
            errors.append(exc)

    def reopen():
        try:
            SkillStore(root)
        except Exception as exc:  # noqa: BLE001 - asserted below
            errors.append(exc)
        finally:
            constructor_finished.set()

    monkeypatch.setattr(
        store_module,
        "_atomic_write_primary_at",
        pause_live_publication,
    )
    publisher = threading.Thread(target=publish)
    publisher.start()
    assert publication_started.wait(timeout=5)
    staging = tuple(
        store._internal_root.glob(f"{store_module.SKILL_PUBLICATION_STAGING_PREFIX}*")
    )
    assert len(staging) == 1

    constructor = threading.Thread(target=reopen)
    constructor.start()
    assert not constructor_finished.wait(timeout=0.1)
    assert staging[0].is_dir(), "a live publication staging folder was reaped"
    release_publication.set()
    publisher.join(timeout=5)
    constructor.join(timeout=5)

    assert not publisher.is_alive()
    assert not constructor.is_alive()
    assert errors == []
    assert (store.local_root / "live-staging" / "SKILL.md").is_file()


@pytest.mark.parametrize("operation", ("create", "install"))
def test_publication_does_not_replace_raced_empty_destination(
    tmp_path, monkeypatch, operation
):
    name = f"no-replace-{operation}"
    store = SkillStore(tmp_path / "skills")
    target = store.local_root / name
    real_identity = store_module._identity_at
    target_checks = 0
    raced_identity = None

    def race_after_absence_check(directory_fd, candidate):
        nonlocal raced_identity, target_checks
        identity = real_identity(directory_fd, candidate)
        if candidate == name and identity is None:
            target_checks += 1
            if target_checks == 2:
                os.mkdir(name, mode=0o700, dir_fd=directory_fd)
                raced = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                raced_identity = (raced.st_dev, raced.st_ino)
                return None
        return identity

    monkeypatch.setattr(store_module, "_identity_at", race_after_absence_check)

    with pytest.raises(SkillConflictError, match="already exists"):
        if operation == "create":
            store.create(SkillDocument(name, "No replace create", "Procedure."))
        else:
            source_root = tmp_path / "source"
            source = source_root / name
            source.mkdir(parents=True)
            (source / "SKILL.md").write_text(
                serialize_skill_markdown(
                    SkillDocument(name, "No replace install", "Procedure.")
                ),
                encoding="utf-8",
            )
            store.install_folder(
                source,
                provenance=git_provenance(name),
            )

    assert raced_identity is not None
    target_stat = target.stat()
    assert (target_stat.st_dev, target_stat.st_ino) == raced_identity
    assert list(target.iterdir()) == []


def test_hidden_install_orphan_does_not_block_retry(tmp_path):
    source_root = tmp_path / "source"
    source = source_root / "retry-install"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text(
        serialize_skill_markdown(
            SkillDocument("retry-install", "Retry install", "Procedure.")
        ),
        encoding="utf-8",
    )
    store = SkillStore(tmp_path / "skills")
    orphan = store.local_root / ".retry-install.install.crashed"
    orphan.mkdir()
    (orphan / "partial.md").write_text("partial", encoding="utf-8")

    installed = store.install_folder(
        source,
        provenance=git_provenance("retry-install"),
    )

    assert (installed / "SKILL.md").is_file()
    assert orphan.is_dir()
    records, errors = DirectorySkillSource(
        root=store.local_root,
        source_id="agent-local",
        kind="agent-local",
        precedence=0,
    ).discover()
    assert [record.name for record in records] == ["retry-install"]
    assert errors == ()


def test_store_reaps_only_unlocked_git_checkout_workspaces(tmp_path):
    root = tmp_path / "skills"
    store = SkillStore(root)
    orphan = store._internal_root / ".kestrel-skill-git-crashed"
    orphan.mkdir()
    (orphan / "partial.pack").write_bytes(b"partial checkout")

    reopened = SkillStore(root)

    assert not orphan.exists()
    with reopened.git_checkout_workspace() as active:
        (active / "partial.pack").write_bytes(b"active checkout")
        concurrent = SkillStore(root)
        assert concurrent.local_root == reopened.local_root
        assert active.is_dir()
    assert not active.exists()


def test_store_reaps_git_checkout_after_owner_process_is_killed(tmp_path):
    root = tmp_path / "skills"
    SkillStore(root)
    context = multiprocessing.get_context("spawn")
    result = context.Queue()
    owner = context.Process(
        target=_hold_git_checkout_workspace_from_separate_process,
        args=(str(root), result),
    )
    owner.start()
    state, detail = result.get(timeout=10)
    try:
        assert state == "ready", detail
        workspace = Path(detail)
        SkillStore(root)
        assert workspace.is_dir(), "a live checkout owner was reaped"
    finally:
        owner.terminate()
        owner.join(timeout=10)
    assert not owner.is_alive()

    SkillStore(root)

    assert not workspace.exists()


def test_failed_edit_workspace_survives_cleanup_and_startup_reaper(
    tmp_path, monkeypatch
):
    root = tmp_path / "skills"
    store = SkillStore(root)
    real_preserve = store_module._preserve_recovery_directory_at
    refused_retirement = False

    def fail_first_retirement(*args, **kwargs):
        nonlocal refused_retirement
        if not refused_retirement:
            refused_retirement = True
            raise OSError(errno.EIO, "failed workspace retirement")
        return real_preserve(*args, **kwargs)

    monkeypatch.setattr(
        store_module,
        "_preserve_recovery_directory_at",
        fail_first_retirement,
    )

    with (
        pytest.raises(RuntimeError, match="simulated failed edit"),
        store.git_checkout_workspace(cleanup_errors_fatal=False) as workspace,
        store.preserve_git_checkout_on_failure(workspace),
    ):
        preserved = workspace / "failed-edit" / "SKILL.md"
        preserved.parent.mkdir()
        preserved.write_text("recover me", encoding="utf-8")
        raise RuntimeError("simulated failed edit")

    assert refused_retirement
    stranded = tuple(
        (root / INTERNAL_DIRECTORY).glob(f"{store_module.GIT_CHECKOUT_PREFIX}*")
    )
    assert len(stranded) == 1
    assert (stranded[0] / store_module.GIT_CHECKOUT_RECOVERY_MARKER).is_file()
    assert (stranded[0] / "failed-edit" / "SKILL.md").read_text(
        encoding="utf-8"
    ) == "recover me"

    monkeypatch.setattr(
        store_module,
        "_preserve_recovery_directory_at",
        real_preserve,
    )
    SkillStore(root)

    assert not tuple(
        (root / INTERNAL_DIRECTORY).glob(f"{store_module.GIT_CHECKOUT_PREFIX}*")
    )
    recovery = tuple(
        (root / INTERNAL_DIRECTORY).glob(f"{store_module.SKILL_RECOVERY_PREFIX}*")
    )
    assert len(recovery) == 1
    assert (recovery[0] / "failed-edit" / "SKILL.md").read_text(
        encoding="utf-8"
    ) == "recover me"


@pytest.mark.parametrize("operation", ("primary", "resource"))
def test_validation_failures_do_not_accumulate_recovery_workspaces(tmp_path, operation):
    store = SkillStore(tmp_path / "skills")
    name = f"invalid-{operation}-retry"
    store.create(SkillDocument(name, "Original", "Original procedure."))
    record = DirectorySkillSource(
        root=store.local_root,
        source_id="agent-local",
        kind="agent-local",
        precedence=0,
    ).discover()[0][0]
    invalid = serialize_skill_markdown(
        SkillDocument(name, "Invalid link", "[missing](missing.md)")
    )

    for _ in range(5):
        with pytest.raises(
            (SkillPathError, SkillFormatError),
            match="does not exist|editor file exceeds",
        ):
            if operation == "primary":
                store.edit_primary(record, invalid)
            else:
                store.write_file(
                    record,
                    "notes.md",
                    "x" * (store_module.MAX_EDITOR_FILE_BYTES + 1),
                )

    internal = store.local_root / INTERNAL_DIRECTORY
    assert not tuple(internal.glob(f"{store_module.SKILL_RECOVERY_PREFIX}*"))
    assert not tuple(internal.glob(f"{store_module.GIT_CHECKOUT_PREFIX}*"))


def test_validated_primary_edit_survives_publication_preflight_failure(
    tmp_path, monkeypatch
):
    store = SkillStore(tmp_path / "skills")
    name = "validated-edit-recovery"
    folder = store.create(SkillDocument(name, "Original", "Original procedure."))
    record = DirectorySkillSource(
        root=store.local_root,
        source_id="agent-local",
        kind="agent-local",
        precedence=0,
    ).discover()[0][0]
    desired = serialize_skill_markdown(
        SkillDocument(name, "Desired", "Validated desired procedure.")
    )

    def fail_publication_fsync(*_args, **_kwargs):
        raise OSError(errno.EIO, "publication preflight fsync failed")

    monkeypatch.setattr(
        store_module,
        "_fsync_validated_directories_at",
        fail_publication_fsync,
    )

    with pytest.raises(OSError, match="publication preflight fsync failed"):
        store.edit_primary(record, desired)

    assert "Original procedure." in (folder / "SKILL.md").read_text(encoding="utf-8")
    recovery = tuple(
        (store.local_root / INTERNAL_DIRECTORY).glob(
            f"{store_module.SKILL_RECOVERY_PREFIX}*"
        )
    )
    assert len(recovery) == 1
    assert (recovery[0] / name / "SKILL.md").read_text(encoding="utf-8") == desired


def test_clearing_absent_fail_closed_marker_fsyncs_retry_proof(tmp_path, monkeypatch):
    store = SkillStore(tmp_path / "skills")
    internal_inode = store._internal_root.stat().st_ino
    synced = []
    real_fsync = os.fsync

    def record_fsync(descriptor):
        synced.append(os.fstat(descriptor).st_ino)
        return real_fsync(descriptor)

    monkeypatch.setattr(store_module.os, "fsync", record_fsync)

    store.clear_fail_closed_state("already-absent")

    assert synced == [internal_inode]


def test_fail_closed_error_is_bounded_after_ascii_json_escaping(tmp_path):
    store = SkillStore(tmp_path / "skills")
    name = "bounded-fail-closed-error"
    error = "uncertain rollback: " + ("\U0001fab6" * 2_048)

    store.retain_fail_closed_state(name, priority=17, error=error)

    marker = store._internal_root / store_module._fail_closed_state_name(name)
    assert len(marker.read_bytes()) <= store_module.MAX_FAIL_CLOSED_ERROR_BYTES
    states, errors = store.load_fail_closed_states()
    assert states[name].enabled is False
    assert states[name].priority == 17
    assert errors[name].startswith("uncertain rollback: ")
    assert len(errors[name]) < len(error)


def test_recursive_cleanup_handles_more_than_python_recursion_limit(tmp_path):
    store = SkillStore(tmp_path / "skills")
    folder = store.create(SkillDocument("deep-cleanup", "Deep cleanup", "Procedure."))
    identity = (folder.stat().st_dev, folder.stat().st_ino)
    descriptor = os.open(
        folder,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        for _ in range(1_100):
            os.mkdir("d", mode=0o700, dir_fd=descriptor)
            child = os.open(
                "d",
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
    finally:
        os.close(descriptor)

    root_fd = os.open(
        store.local_root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        store_module._remove_expected_directory_at(
            root_fd,
            folder.name,
            expected=identity,
        )
    finally:
        os.close(root_fd)

    assert not folder.exists()
    retired_root = store.local_root / INTERNAL_DIRECTORY
    assert not tuple(retired_root.glob(f"{store_module.SKILL_TRASH_PREFIX}*"))


def test_primary_edit_rechecks_original_bytes_at_atomic_exchange(tmp_path, monkeypatch):
    store = SkillStore(tmp_path / "skills")
    name = "exchange-precondition"
    folder = store.create(SkillDocument(name, "Original", "Original procedure."))
    record = DirectorySkillSource(
        root=store.local_root,
        source_id="agent-local",
        kind="agent-local",
        precedence=0,
    ).discover()[0][0]
    external = serialize_skill_markdown(
        SkillDocument(name, "External", "EXTERNAL procedure.")
    )
    desired = serialize_skill_markdown(
        SkillDocument(name, "Desired", "Desired procedure.")
    )
    real_exchange = store_module._exchange_directories_at
    injected = False

    def mutate_at_exchange(source_fd, source_name, destination_fd, destination_name):
        nonlocal injected
        if not injected and destination_name == name:
            injected = True
            (folder / "SKILL.md").write_text(external, encoding="utf-8")
        return real_exchange(source_fd, source_name, destination_fd, destination_name)

    monkeypatch.setattr(
        store_module,
        "_exchange_directories_at",
        mutate_at_exchange,
    )

    with pytest.raises(SkillConflictError, match="changed after it was read"):
        store.edit_primary(record, desired)

    assert injected
    assert (folder / "SKILL.md").read_text(encoding="utf-8") == external
    preserved = list(
        (store.local_root / INTERNAL_DIRECTORY).glob(".kestrel-skill-recovery-*")
    )
    assert len(preserved) == 1
    assert (preserved[0] / "SKILL.md").read_text(encoding="utf-8") == desired


def test_edit_preserves_public_change_detected_after_exchange(tmp_path, monkeypatch):
    store = SkillStore(tmp_path / "skills")
    name = "exchange-postcondition"
    folder = store.create(SkillDocument(name, "Original", "Original procedure."))
    record = DirectorySkillSource(
        root=store.local_root,
        source_id="agent-local",
        kind="agent-local",
        precedence=0,
    ).discover()[0][0]
    external = serialize_skill_markdown(
        SkillDocument(name, "External", "EXTERNAL procedure.")
    )
    desired = serialize_skill_markdown(
        SkillDocument(name, "Desired", "Desired procedure.")
    )
    root_identity = (store.local_root.stat().st_dev, store.local_root.stat().st_ino)
    real_inspect = store_module._inspect_child_at
    injected = False

    def mutate_public_candidate(parent_fd, child_name, *, expected, folder_name=None):
        nonlocal injected
        parent = os.fstat(parent_fd)
        if (
            not injected
            and child_name == name
            and (parent.st_dev, parent.st_ino) == root_identity
            and expected != record.folder_identity
        ):
            injected = True
            (folder / "SKILL.md").write_text(external, encoding="utf-8")
        return real_inspect(
            parent_fd,
            child_name,
            expected=expected,
            folder_name=folder_name,
        )

    monkeypatch.setattr(store_module, "_inspect_child_at", mutate_public_candidate)

    with pytest.raises(
        SkillPublicationCleanupError,
        match="both generations were preserved",
    ):
        store.edit_primary(record, desired)

    assert injected
    assert (folder / "SKILL.md").read_text(encoding="utf-8") == external
    preserved = list(
        (store.local_root / INTERNAL_DIRECTORY).glob(".kestrel-skill-recovery-*")
    )
    assert len(preserved) == 1
    assert "Original procedure." in (preserved[0] / "SKILL.md").read_text(
        encoding="utf-8"
    )


def test_edit_preserves_prior_generation_when_public_recovery_move_fails(
    tmp_path, monkeypatch
):
    store = SkillStore(tmp_path / "skills")
    name = "failed-public-recovery-move"
    folder = store.create(SkillDocument(name, "Original", "Original procedure."))
    record = DirectorySkillSource(
        root=store.local_root,
        source_id="agent-local",
        kind="agent-local",
        precedence=0,
    ).discover()[0][0]
    desired = serialize_skill_markdown(
        SkillDocument(name, "Desired", "Desired procedure.")
    )
    external = serialize_skill_markdown(
        SkillDocument(name, "External", "EXTERNAL procedure.")
    )
    root_identity = (store.local_root.stat().st_dev, store.local_root.stat().st_ino)
    real_inspect = store_module._inspect_child_at
    real_rename = store_module._rename_directory_no_replace_at
    injected = False
    refused_public_recovery = False

    def mutate_public_candidate(parent_fd, child_name, *, expected, folder_name=None):
        nonlocal injected
        parent = os.fstat(parent_fd)
        if (
            not injected
            and child_name == name
            and (parent.st_dev, parent.st_ino) == root_identity
            and expected != record.folder_identity
        ):
            injected = True
            (folder / "SKILL.md").write_text(external, encoding="utf-8")
        return real_inspect(
            parent_fd,
            child_name,
            expected=expected,
            folder_name=folder_name,
        )

    def fail_public_recovery_move(
        source_fd,
        source_name,
        destination_name,
        *,
        destination_directory_fd=None,
    ):
        nonlocal refused_public_recovery
        if destination_name.startswith(".kestrel-edit-conflict-"):
            refused_public_recovery = True
            raise OSError(errno.ENOSPC, "public recovery move refused")
        return real_rename(
            source_fd,
            source_name,
            destination_name,
            destination_directory_fd=destination_directory_fd,
        )

    monkeypatch.setattr(store_module, "_inspect_child_at", mutate_public_candidate)
    monkeypatch.setattr(
        store_module,
        "_rename_directory_no_replace_at",
        fail_public_recovery_move,
    )

    with pytest.raises(SkillPublicationCleanupError):
        store.edit_primary(record, desired)

    assert injected
    assert not refused_public_recovery
    assert (folder / "SKILL.md").read_text(encoding="utf-8") == external
    recovery = tuple(
        (store.local_root / INTERNAL_DIRECTORY).glob(".kestrel-skill-recovery-*")
    )
    assert len(recovery) == 1
    assert "Original procedure." in (recovery[0] / "SKILL.md").read_text(
        encoding="utf-8"
    )


def test_edit_retires_marked_workspace_when_private_recovery_move_fails(
    tmp_path, monkeypatch
):
    store = SkillStore(tmp_path / "skills")
    name = "failed-private-recovery-move"
    folder = store.create(SkillDocument(name, "Original", "Original procedure."))
    record = DirectorySkillSource(
        root=store.local_root,
        source_id="agent-local",
        kind="agent-local",
        precedence=0,
    ).discover()[0][0]
    desired = serialize_skill_markdown(
        SkillDocument(name, "Desired", "Desired procedure.")
    )
    external = serialize_skill_markdown(
        SkillDocument(name, "External", "EXTERNAL procedure.")
    )
    root_identity = (store.local_root.stat().st_dev, store.local_root.stat().st_ino)
    real_inspect = store_module._inspect_child_at
    injected = False

    def mutate_public_candidate(parent_fd, child_name, *, expected, folder_name=None):
        nonlocal injected
        parent = os.fstat(parent_fd)
        if (
            not injected
            and child_name == name
            and (parent.st_dev, parent.st_ino) == root_identity
            and expected != record.folder_identity
        ):
            injected = True
            (folder / "SKILL.md").write_text(external, encoding="utf-8")
        return real_inspect(
            parent_fd,
            child_name,
            expected=expected,
            folder_name=folder_name,
        )

    def fail_private_recovery(*_args, **_kwargs):
        raise OSError(errno.EIO, "private recovery move failed")

    monkeypatch.setattr(store_module, "_inspect_child_at", mutate_public_candidate)
    monkeypatch.setattr(store, "_preserve_edit_recovery", fail_private_recovery)

    with pytest.raises(SkillPublicationCleanupError):
        store.edit_primary(record, desired)

    assert injected
    assert (folder / "SKILL.md").read_text(encoding="utf-8") == external
    recovery = tuple(
        (store.local_root / INTERNAL_DIRECTORY).glob(".kestrel-skill-recovery-*")
    )
    assert len(recovery) == 1
    preserved_primaries = tuple(recovery[0].rglob("SKILL.md"))
    assert len(preserved_primaries) == 1
    assert "Original procedure." in preserved_primaries[0].read_text(encoding="utf-8")


def test_edit_conflict_recovery_does_not_exceed_source_entry_limit(
    tmp_path, monkeypatch
):
    store = SkillStore(tmp_path / "skills")
    monkeypatch.setattr(sources_module, "MAX_SOURCE_ENTRIES", 3)
    name = "bounded-edit-recovery"
    folder = store.create(SkillDocument(name, "Original", "Original procedure."))
    (store.local_root / "ordinary.txt").write_text("ordinary", encoding="utf-8")
    record = DirectorySkillSource(
        root=store.local_root,
        source_id="agent-local",
        kind="agent-local",
        precedence=0,
    ).discover()[0][0]
    desired = serialize_skill_markdown(
        SkillDocument(name, "Desired", "Desired procedure.")
    )
    external = serialize_skill_markdown(
        SkillDocument(name, "External", "EXTERNAL procedure.")
    )
    root_identity = (store.local_root.stat().st_dev, store.local_root.stat().st_ino)
    real_inspect = store_module._inspect_child_at
    injected = False

    def mutate_public_candidate(parent_fd, child_name, *, expected, folder_name=None):
        nonlocal injected
        parent = os.fstat(parent_fd)
        if (
            not injected
            and child_name == name
            and (parent.st_dev, parent.st_ino) == root_identity
            and expected != record.folder_identity
        ):
            injected = True
            (folder / "SKILL.md").write_text(external, encoding="utf-8")
        return real_inspect(
            parent_fd,
            child_name,
            expected=expected,
            folder_name=folder_name,
        )

    monkeypatch.setattr(store_module, "_inspect_child_at", mutate_public_candidate)

    with pytest.raises(SkillPublicationCleanupError):
        store.edit_primary(record, desired)

    records, errors = DirectorySkillSource(
        root=store.local_root,
        source_id="agent-local",
        kind="agent-local",
        precedence=0,
    ).discover()

    assert injected
    assert errors == ()
    assert [item.name for item in records] == [name]
    assert len(tuple(store.local_root.iterdir())) == 3


def test_delete_restores_same_inode_change_made_at_quarantine(tmp_path, monkeypatch):
    store = SkillStore(tmp_path / "skills")
    name = "delete-content-race"
    folder = store.create(SkillDocument(name, "Original", "Original procedure."))
    record = DirectorySkillSource(
        root=store.local_root,
        source_id="agent-local",
        kind="agent-local",
        precedence=0,
    ).discover()[0][0]
    external = serialize_skill_markdown(
        SkillDocument(name, "External", "EXTERNAL procedure.")
    )
    real_quarantine = store_module._quarantine_directory_at
    injected = False

    def mutate_then_quarantine(root_fd, child_name, *, expected, internal_fd=None):
        nonlocal injected
        if not injected and child_name == name:
            injected = True
            (folder / "SKILL.md").write_text(external, encoding="utf-8")
        return real_quarantine(
            root_fd,
            child_name,
            expected=expected,
            internal_fd=internal_fd,
        )

    monkeypatch.setattr(
        store_module,
        "_quarantine_directory_at",
        mutate_then_quarantine,
    )

    with pytest.raises(SkillConflictError, match="changed after it was read"):
        store.delete(record)

    assert injected
    assert (folder / "SKILL.md").read_text(encoding="utf-8") == external


def test_recursive_fail_closed_json_is_treated_as_malformed(tmp_path, monkeypatch):
    store = SkillStore(tmp_path / "skills")
    name = "recursive-quarantine"
    marker = store._internal_root / store_module._fail_closed_state_name(name)
    marker.write_text("{}", encoding="ascii")

    def recursive_decoder_failure(_payload):
        raise RecursionError("maximum recursion depth exceeded while decoding JSON")

    monkeypatch.setattr(store_module.json, "loads", recursive_decoder_failure)

    states, errors = store.load_fail_closed_states()

    assert states[name].enabled is False
    assert states[name].priority == 100
    assert "metadata is malformed" in errors[name]
