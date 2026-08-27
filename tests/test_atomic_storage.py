from __future__ import annotations

import fcntl
import multiprocessing
import os
import shutil
import threading
import time
from pathlib import Path

import pytest

import kestrel_feature_skills.store as store_module
from kestrel_feature_skills.errors import (
    SkillConflictError,
    SkillFormatError,
    SkillPathError,
)
from kestrel_feature_skills.format import (
    MAX_FOLDER_BYTES,
    MAX_FOLDER_FILES,
    MAX_SKILL_FILE_BYTES,
    serialize_skill_markdown,
    validate_skill_folder,
)
from kestrel_feature_skills.models import SkillDocument, SkillProvenance, SkillRecord
from kestrel_feature_skills.sources import DirectorySkillSource
from kestrel_feature_skills.store import (
    CLAIM_STALENESS_SECONDS,
    SkillStore,
    atomic_write_primary,
)


def payload(name="atomic"):
    return serialize_skill_markdown(
        SkillDocument(name, "Atomic write", "# Procedure\n\nWrite once.")
    ).encode("utf-8")


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
        lock_path = Path(root) / ".cross-process-create.mutation.lock"
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


def test_atomic_create_leaves_no_temporary_or_claim_files(tmp_path):
    root = tmp_path / "skills"
    store = SkillStore(root)
    folder = store.create(SkillDocument("atomic", "Atomic write", "Do it."))
    assert (folder / "SKILL.md").is_file()
    assert not list(folder.glob(".SKILL.md.tmp.*"))
    assert not (folder / ".SKILL.md.claim").exists()


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
        candidates = list(store.local_root.glob(".create-cleanup-race.create.*"))
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
    real_reject = store_module.reject_symlink_chain
    swapped = False

    def swap_after_check(root, path):
        nonlocal swapped
        real_reject(root, path)
        if root == folder and path == references / "notes.md" and not swapped:
            shutil.rmtree(references)
            references.symlink_to(outside, target_is_directory=True)
            swapped = True

    monkeypatch.setattr(store_module, "reject_symlink_chain", swap_after_check)

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
    real_reject = store_module.reject_symlink_chain
    checked = 0

    def swap_after_final_check(root, path):
        nonlocal checked
        real_reject(root, path)
        if root == folder and path == references / "notes.md":
            checked += 1
            if checked == 3:
                references.rename(displaced)
                references.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(store_module, "reject_symlink_chain", swap_after_final_check)

    with pytest.raises(SkillPathError, match="symlink|directory"):
        store.write_file(record, "references/notes.md", "replacement")

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
            provenance=SkillProvenance("git", "https://example.com/repo.git", "main"),
        )

    assert not (store.local_root / document.name).exists()


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
            provenance=SkillProvenance("git", "https://example.com/repo.git", "main"),
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
    real_rename = os.rename
    swapped = False

    def swap_before_removal(source_name, destination_name, **kwargs):
        nonlocal swapped
        if source_name == folder.name and not swapped:
            swapped = True
            real_rename(folder, displaced)
            folder.mkdir()
            marker.write_text("replacement", encoding="utf-8")
        return real_rename(source_name, destination_name, **kwargs)

    monkeypatch.setattr(store_module.os, "rename", swap_before_removal)

    with pytest.raises(SkillPathError, match="changed|preserved"):
        store.rollback_created(folder, identity=identity)

    quarantines = list(store.local_root.glob(".rollback-race.delete.*"))
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
        candidates = list(store.local_root.glob(".install-cleanup-race.install.*"))
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
    real_rename = os.rename
    swapped = False

    def swap_as_quarantine_starts(source_name, destination_name, **kwargs):
        nonlocal swapped
        if source_name == record.name and not swapped:
            swapped = True
            real_rename(original, displaced)
            original.mkdir()
            replacement_marker.write_text("replacement", encoding="utf-8")
        return real_rename(source_name, destination_name, **kwargs)

    monkeypatch.setattr(store_module.os, "rename", swap_as_quarantine_starts)

    with pytest.raises(SkillPathError, match="changed during deletion"):
        store.delete(record)

    quarantines = list(store.local_root.glob(".delete-quarantine-race.delete.*"))
    assert len(quarantines) == 1
    assert (quarantines[0] / replacement_marker.name).read_text(
        encoding="utf-8"
    ) == "replacement"
    assert (displaced / "SKILL.md").is_file()
    assert not original.exists()


def test_delete_pins_quarantined_folder_during_recursive_removal(tmp_path, monkeypatch):
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
    real_rmtree = shutil.rmtree
    swapped = False

    def swap_quarantine_before_recursive_removal(path, *args, **kwargs):
        nonlocal replacement_marker, swapped
        if not swapped:
            swapped = True
            quarantines = list(store.local_root.glob(".delete-recursive-race.delete.*"))
            assert len(quarantines) == 1
            quarantine = quarantines[0]
            quarantine.rename(displaced)
            quarantine.mkdir()
            replacement_marker = quarantine / "replacement-must-survive.md"
            replacement_marker.write_text("replacement", encoding="utf-8")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(
        store_module.shutil,
        "rmtree",
        swap_quarantine_before_recursive_removal,
    )
    try:
        with pytest.raises(SkillPathError, match="changed during deletion"):
            store.delete(record)

        assert replacement_marker is not None
        assert replacement_marker.read_text(encoding="utf-8") == "replacement"
        assert displaced.is_dir()
    finally:
        real_rmtree(displaced, ignore_errors=True)
        for quarantine in store.local_root.glob(".delete-recursive-race.delete.*"):
            real_rmtree(quarantine, ignore_errors=True)


def test_delete_preserves_quarantined_folder_if_recursive_removal_fails(
    tmp_path, monkeypatch
):
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

    def fail_removal(*_args, **_kwargs):
        raise OSError("simulated recursive removal failure")

    monkeypatch.setattr(store_module.shutil, "rmtree", fail_removal)

    with pytest.raises(OSError, match="simulated recursive removal failure") as caught:
        store.delete(record)

    quarantines = list(store.local_root.glob(".delete-restore.delete.*"))
    assert len(quarantines) == 1
    assert (quarantines[0] / "SKILL.md").is_file()
    assert not original.exists()
    assert any(
        "preserved as .delete-restore.delete." in note
        for note in caught.value.__notes__
    )


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
        provenance=SkillProvenance(
            "git", "https://example.com/repo.git", "main:hidden-install"
        ),
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
                provenance=SkillProvenance(
                    "git", "https://example.com/repo.git", f"main:{name}"
                ),
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
        provenance=SkillProvenance(
            "git", "https://example.com/repo.git", "main:retry-install"
        ),
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
