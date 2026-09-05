"""File-authoritative skill storage with atomic primary-record writes."""

from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import hmac
import os
import shutil
import stat
import tempfile
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .errors import (
    SkillConflictError,
    SkillFormatError,
    SkillNotFoundError,
    SkillPathError,
    SkillPublicationCleanupError,
    SkillReadOnlyError,
)
from .format import (
    MAX_SKILL_FILE_BYTES,
    PRIMARY_WRITER_CLAIM,
    PRIMARY_WRITER_TEMP_PREFIX,
    SKILL_FILENAME,
    ValidatedSkillFolder,
    inspect_skill_folder,
    inspect_skill_folder_descriptor,
    parse_skill_markdown,
    serialize_skill_markdown,
    validate_resource_path,
    validate_skill_folder,
    validate_skill_folder_descriptor,
    validate_skill_name,
)
from .models import CatalogSnapshot, SkillDocument, SkillProvenance, SkillRecord
from .paths import (
    direct_child,
    lexical_contained_path,
    reject_symlink_chain,
)
from .sources import (
    MAX_SOURCE_ENTRIES,
    PROVENANCE_FILENAME,
    _folder_revision,
    serialize_provenance,
)

CLAIM_STALENESS_SECONDS = 60
INTERNAL_DIRECTORY = ".kestrel-internal"
GIT_CHECKOUT_PREFIX = ".kestrel-skill-git-"
GIT_CHECKOUT_LOCK = ".checkout-owner.lock"
SKILL_TRASH_PREFIX = ".kestrel-skill-trash-"
SKILL_TRASH_LOCK = ".skill-trash.lock"
MAX_EDITOR_FILE_BYTES = 262_144
NAME_LOCK_BUCKETS = 256
_MUTATION_LOCKS = tuple(threading.RLock() for _ in range(NAME_LOCK_BUCKETS))
_SOURCE_PUBLICATION_LOCKS = tuple(threading.RLock() for _ in range(NAME_LOCK_BUCKETS))
_PUBLICATION_STATE_LOCKS = tuple(threading.Lock() for _ in range(NAME_LOCK_BUCKETS))
_TRASH_THREAD_LOCK = threading.RLock()
_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)


def has_python_execution_risk(relative_path: str) -> bool:
    """Return whether a resource suffix can conventionally denote Python code."""

    return Path(relative_path).suffix.casefold() == ".py"


def _name_lock_bucket(root: Path, name: str, *, domain: str) -> int:
    material = f"{domain}\0{root}\0{name}".encode()
    return (
        int.from_bytes(hashlib.sha256(material).digest()[:2], "big") % NAME_LOCK_BUCKETS
    )


def _mutation_lock_name(root: Path, name: str) -> str:
    bucket = _name_lock_bucket(root, name, domain="mutation")
    return f".mutation-bucket-{bucket:03d}.lock"


def _publication_state_lock_name(root: Path, name: str) -> str:
    bucket = _name_lock_bucket(root, name, domain="publication-state")
    return f".publication-state-bucket-{bucket:03d}.lock"


@dataclass(frozen=True, slots=True)
class CreatedSkillPublication:
    """Inode evidence proving a new folder still has its published contents."""

    folder_identity: tuple[int, int]
    primary_identity: tuple[int, int]
    primary_size: int
    primary_mtime_ns: int
    primary_ctime_ns: int


@dataclass(frozen=True, slots=True)
class InstalledSkillPublication:
    """Inode and content evidence for one newly installed skill folder."""

    folder_identity: tuple[int, int]
    snapshot: ValidatedSkillFolder


@dataclass(frozen=True, slots=True)
class CreatedParentDirectory:
    """A pinned parent/name pair for one directory created during an edit."""

    parent_descriptor: int
    name: str
    identity: tuple[int, int]


@dataclass(slots=True)
class _RemovalFrame:
    """One directory in the implementation-owned trash deletion walk."""

    parent_fd: int
    name: str
    identity: tuple[int, int]
    descriptor: int
    entries: Iterator[str]


@dataclass(slots=True)
class PublicationStateClaim:
    """An exclusive same-name claim spanning database guard and publication."""

    descriptor: int
    thread_lock: threading.Lock
    released: bool = False

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        try:
            try:
                fcntl.flock(self.descriptor, fcntl.LOCK_UN)
            finally:
                os.close(self.descriptor)
        finally:
            self.thread_lock.release()


@contextmanager
def _serialized_source_publication(
    root: Path,
    internal_fd: int,
    *,
    enabled: bool,
):
    """Serialize source-capacity reservations across processes when requested."""

    if not enabled:
        yield
        return
    bucket = _name_lock_bucket(root, "", domain="source-publication")
    thread_lock = _SOURCE_PUBLICATION_LOCKS[bucket]
    with thread_lock:
        lock_name = ".source-publication.lock"
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            descriptor = os.open(lock_name, flags, 0o600, dir_fd=internal_fd)
        except OSError as exc:
            raise SkillPathError("could not open the source publication lock") from exc
        locked = False
        try:
            value = os.fstat(descriptor)
            if not stat.S_ISREG(value.st_mode):
                raise SkillPathError("source publication lock must be a regular file")
            identity = (value.st_dev, value.st_ino)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            locked = True
            if _identity_at(internal_fd, lock_name) != identity:
                raise SkillPathError(
                    "source publication lock changed while acquiring it"
                )
            yield
        finally:
            try:
                if locked:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


@contextmanager
def _serialized_skill_trash(internal_fd: int):
    """Serialize durable retirement and private-trash reaping across processes."""

    with _TRASH_THREAD_LOCK:
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            descriptor = os.open(
                SKILL_TRASH_LOCK,
                flags,
                0o600,
                dir_fd=internal_fd,
            )
        except OSError as exc:
            raise SkillPathError("could not open the skill trash lock") from exc
        locked = False
        try:
            value = os.fstat(descriptor)
            if not stat.S_ISREG(value.st_mode):
                raise SkillPathError("skill trash lock must be a regular file")
            identity = (value.st_dev, value.st_ino)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            locked = True
            if _identity_at(internal_fd, SKILL_TRASH_LOCK) != identity:
                raise SkillPathError("skill trash lock changed while acquiring it")
            yield
        finally:
            try:
                if locked:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


@contextmanager
def _serialized_skill_mutation(
    root: Path,
    internal_root: Path,
    name: str,
    *,
    root_identity: tuple[int, int],
    internal_root_identity: tuple[int, int],
    source_publication: bool = False,
):
    """Serialize validation and publication across threads and processes."""

    name = validate_skill_name(name)
    bucket = _name_lock_bucket(root, name, domain="mutation")
    thread_lock = _MUTATION_LOCKS[bucket]
    with thread_lock:
        root_fd = _open_directory(root, expected=root_identity)
        try:
            internal_fd = _open_directory(
                internal_root,
                expected=internal_root_identity,
            )
            try:
                lock_name = _mutation_lock_name(root, name)
                flags = (
                    os.O_RDWR
                    | os.O_CREAT
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0)
                )
                try:
                    descriptor = os.open(lock_name, flags, 0o600, dir_fd=internal_fd)
                except OSError as exc:
                    raise SkillPathError(
                        "could not open the skill mutation lock"
                    ) from exc
                locked = False
                try:
                    value = os.fstat(descriptor)
                    if not stat.S_ISREG(value.st_mode):
                        raise SkillPathError(
                            "skill mutation lock must be a regular file"
                        )
                    lock_identity = (value.st_dev, value.st_ino)
                    fcntl.flock(descriptor, fcntl.LOCK_EX)
                    locked = True
                    if _identity_at(internal_fd, lock_name) != lock_identity:
                        raise SkillPathError(
                            "skill mutation lock changed while acquiring it"
                        )
                    with _serialized_source_publication(
                        root,
                        internal_fd,
                        enabled=source_publication,
                    ):
                        yield root_fd, internal_fd
                finally:
                    try:
                        if locked:
                            fcntl.flock(descriptor, fcntl.LOCK_UN)
                    finally:
                        os.close(descriptor)
            finally:
                os.close(internal_fd)
        finally:
            os.close(root_fd)


def _unlock_claim(descriptor: int | None) -> None:
    if descriptor is None:
        return
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _write_tmp(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    created = os.fstat(descriptor)
    created_identity = (created.st_dev, created.st_ino)
    completed = False
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        completed = True
    finally:
        os.close(descriptor)
        if not completed:
            try:
                current = path.lstat()
            except FileNotFoundError:
                pass
            else:
                if (current.st_dev, current.st_ino) == created_identity:
                    path.unlink(missing_ok=True)


def _write_tmp_at(directory_fd: int, name: str, payload: bytes) -> None:
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
        dir_fd=directory_fd,
    )
    created = os.fstat(descriptor)
    created_identity = (created.st_dev, created.st_ino)
    completed = False
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        completed = True
    finally:
        os.close(descriptor)
        if not completed and _identity_at(directory_fd, name) == created_identity:
            _unlink_at(directory_fd, name)


def _identity_at(directory_fd: int, name: str) -> tuple[int, int] | None:
    try:
        value = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    return value.st_dev, value.st_ino


def _require_source_publication_capacity_at(root_fd: int) -> None:
    """Fail before staging when one more immediate source entry cannot fit."""

    entries = 0
    with os.scandir(root_fd) as candidates:
        for _candidate in candidates:
            entries += 1
            if entries >= MAX_SOURCE_ENTRIES:
                raise SkillFormatError(
                    "agent-local skill source is at source entry capacity "
                    f"{MAX_SOURCE_ENTRIES}"
                )


def _claim_is_stale_at(directory_fd: int, name: str) -> bool:
    try:
        value = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError:
        return False
    return time.time() - value.st_mtime > CLAIM_STALENESS_SECONDS


def _lock_claim_at(
    directory_fd: int, name: str, *, expected: tuple[int, int] | None
) -> int | None:
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory_fd,
        )
    except OSError:
        return None
    try:
        value = os.fstat(descriptor)
        if not stat.S_ISREG(value.st_mode):
            os.close(descriptor)
            return None
        identity = (value.st_dev, value.st_ino)
        if expected is not None and identity != expected:
            os.close(descriptor)
            return None
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if _identity_at(directory_fd, name) != identity:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
            return None
    except (BlockingIOError, OSError):
        os.close(descriptor)
        return None
    return descriptor


def _rename_directory_no_replace_at(
    directory_fd: int,
    source_name: str,
    destination_name: str,
    *,
    destination_directory_fd: int | None = None,
) -> None:
    """Atomically publish a directory only when its destination is absent."""

    destination_fd = (
        directory_fd if destination_directory_fd is None else destination_directory_fd
    )
    source = os.fsencode(source_name)
    destination = os.fsencode(destination_name)
    libc = ctypes.CDLL(None, use_errno=True)

    if hasattr(libc, "renameatx_np"):
        rename = libc.renameatx_np
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        arguments = (directory_fd, source, destination_fd, destination, 0x00000004)
    elif hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        arguments = (directory_fd, source, destination_fd, destination, 0x00000001)
    else:
        raise SkillPathError(
            "atomic no-replace directory publication is unavailable on this platform"
        )

    ctypes.set_errno(0)
    if rename(*arguments) == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            os.strerror(error_number),
            destination_name,
        )
    raise OSError(error_number, os.strerror(error_number), destination_name)


def _exchange_directories_at(
    source_directory_fd: int,
    source_name: str,
    destination_directory_fd: int,
    destination_name: str,
) -> None:
    """Atomically exchange two directory entries without an absent-name window."""

    source = os.fsencode(source_name)
    destination = os.fsencode(destination_name)
    libc = ctypes.CDLL(None, use_errno=True)
    if hasattr(libc, "renameatx_np"):
        rename = libc.renameatx_np
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        arguments = (
            source_directory_fd,
            source,
            destination_directory_fd,
            destination,
            0x00000002,
        )
    elif hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        arguments = (
            source_directory_fd,
            source,
            destination_directory_fd,
            destination,
            0x00000002,
        )
    else:
        raise SkillPathError(
            "atomic directory exchange is unavailable on this platform"
        )

    ctypes.set_errno(0)
    if rename(*arguments) == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {
        errno.ENOSYS,
        errno.EINVAL,
        errno.ENOTSUP,
        errno.EOPNOTSUPP,
    }:
        raise SkillPathError(
            "atomic directory exchange is unavailable on this filesystem"
        )
    raise OSError(error_number, os.strerror(error_number), destination_name)


def _fsync_directory_pair(first_fd: int, second_fd: int) -> None:
    """Durably order a cross-directory exchange on both directory entries."""

    os.fsync(first_fd)
    if second_fd != first_fd:
        os.fsync(second_fd)


def _open_relative_directory_at(root_fd: int, parts: tuple[str, ...]) -> int:
    """Open one already-validated relative directory without lexical traversal."""

    descriptor = os.dup(root_fd)
    try:
        for part in parts:
            child = _open_directory_at(descriptor, part)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _fsync_validated_directories_at(
    folder_fd: int,
    snapshot: ValidatedSkillFolder,
) -> None:
    """Persist every captured directory entry bottom-up before publication."""

    directories: set[tuple[str, ...]] = {()}
    for entry in snapshot.entries:
        parts = Path(entry.path).parts
        parent_depth = len(parts) if entry.is_directory else len(parts) - 1
        for depth in range(1, parent_depth + 1):
            directories.add(parts[:depth])
    for parts in sorted(directories, key=lambda value: (-len(value), value)):
        descriptor = _open_relative_directory_at(folder_fd, parts)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _unlink_at(directory_fd: int, name: str) -> None:
    try:
        os.unlink(name, dir_fd=directory_fd)
    except FileNotFoundError:
        pass


def _reap_temporary_hardlinks_at(
    directory_fd: int,
    *,
    expected: tuple[int, int],
    temporary_prefix: str = ".SKILL.md.tmp.",
) -> None:
    """Remove only crashed-writer temporaries hardlinked to a stale claim."""

    for name in os.listdir(directory_fd):
        if not name.startswith(temporary_prefix):
            continue
        if _identity_at(directory_fd, name) != expected:
            continue
        # Same-name feature writers are serialized while recovery holds the
        # mutation lock, and UUID temporary names are never reused. Repeat the
        # identity check immediately before unlinking so an unrelated orphan
        # or current writer's temporary is always preserved.
        if _identity_at(directory_fd, name) == expected:
            _unlink_at(directory_fd, name)


def _atomic_write_primary_at(
    directory_fd: int,
    folder_name: str,
    payload: bytes,
    *,
    overwrite: bool,
    artifact_directory_fd: int | None = None,
) -> None:
    """Apply the primary hardlink-claim protocol using private artifacts."""

    if len(payload) > MAX_SKILL_FILE_BYTES:
        raise SkillFormatError(f"{SKILL_FILENAME} exceeds {MAX_SKILL_FILE_BYTES} bytes")
    if not overwrite and _identity_at(directory_fd, SKILL_FILENAME) is not None:
        raise SkillConflictError(f"skill already exists: {folder_name}")
    artifact_fd = (
        artifact_directory_fd if artifact_directory_fd is not None else directory_fd
    )
    if artifact_directory_fd is None:
        temporary_prefix = ".SKILL.md.tmp."
        claim = ".SKILL.md.claim"
    else:
        temporary_prefix = f".{folder_name}.primary.tmp."
        claim = f".{folder_name}.primary.claim"
    temporary = f"{temporary_prefix}{uuid.uuid4().hex}"
    owned_claim: tuple[int, int] | None = None
    claim_lock: int | None = None
    _write_tmp_at(artifact_fd, temporary, payload)
    temporary_identity = _identity_at(artifact_fd, temporary)
    if temporary_identity is None:  # pragma: no cover - fsync'd file vanished
        _unlink_at(artifact_fd, temporary)
        raise OSError("skill writer temporary file vanished before claiming")
    try:
        try:
            os.link(
                temporary,
                claim,
                src_dir_fd=artifact_fd,
                dst_dir_fd=artifact_fd,
                follow_symlinks=False,
            )
            owned_claim = temporary_identity
            claim_lock = _lock_claim_at(artifact_fd, claim, expected=owned_claim)
            if claim_lock is None:
                raise SkillConflictError(
                    f"skill write could not lock its claim for {folder_name}"
                )
        except FileExistsError:
            if not _claim_is_stale_at(artifact_fd, claim):
                raise SkillConflictError(
                    f"concurrent skill write is already in progress for {folder_name}"
                ) from None
            stale_identity = _identity_at(artifact_fd, claim)
            stale_lock = _lock_claim_at(artifact_fd, claim, expected=stale_identity)
            if stale_lock is None:
                raise SkillConflictError(
                    f"active skill writer still owns the stale claim for {folder_name}"
                ) from None
            try:
                if (
                    stale_identity is None
                    or _identity_at(artifact_fd, claim) != stale_identity
                    or not _claim_is_stale_at(artifact_fd, claim)
                ):
                    raise SkillConflictError(
                        f"stale skill claim changed during recovery for {folder_name}"
                    )
                _reap_temporary_hardlinks_at(
                    artifact_fd,
                    expected=stale_identity,
                    temporary_prefix=temporary_prefix,
                )
                os.unlink(claim, dir_fd=artifact_fd)
                try:
                    os.link(
                        temporary,
                        claim,
                        src_dir_fd=artifact_fd,
                        dst_dir_fd=artifact_fd,
                        follow_symlinks=False,
                    )
                    owned_claim = temporary_identity
                except FileExistsError as exc:
                    raise SkillConflictError(
                        f"concurrent skill write won stale-claim recovery for {folder_name}"
                    ) from exc
                claim_lock = _lock_claim_at(artifact_fd, claim, expected=owned_claim)
                if claim_lock is None:
                    raise SkillConflictError(
                        f"skill write could not lock its reclaimed claim for {folder_name}"
                    )
            finally:
                _unlock_claim(stale_lock)
        if owned_claim is None or _identity_at(artifact_fd, claim) != owned_claim:
            raise SkillConflictError(
                f"skill write lost its reclaimed claim for {folder_name}"
            )
        if overwrite:
            os.replace(
                temporary,
                SKILL_FILENAME,
                src_dir_fd=artifact_fd,
                dst_dir_fd=directory_fd,
            )
        else:
            try:
                os.link(
                    temporary,
                    SKILL_FILENAME,
                    src_dir_fd=artifact_fd,
                    dst_dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except FileExistsError as exc:
                raise SkillConflictError(
                    f"skill already exists: {folder_name}"
                ) from exc
    finally:
        _unlink_at(artifact_fd, temporary)
        if owned_claim is not None and _identity_at(artifact_fd, claim) == owned_claim:
            _unlink_at(artifact_fd, claim)
        _unlock_claim(claim_lock)


def _open_directory(path: Path, *, expected: tuple[int, int] | None = None) -> int:
    try:
        descriptor = os.open(path, _DIRECTORY_FLAGS)
    except OSError as exc:
        raise SkillPathError("skill directory must be a real directory") from exc
    value = os.fstat(descriptor)
    identity = (value.st_dev, value.st_ino)
    if not stat.S_ISDIR(value.st_mode) or (
        expected is not None and identity != expected
    ):
        os.close(descriptor)
        raise SkillPathError("skill directory changed after validation")
    return descriptor


def _open_directory_at(
    parent_fd: int,
    name: str,
    *,
    expected: tuple[int, int] | None = None,
) -> int:
    try:
        descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    except OSError as exc:
        raise SkillPathError("skill directory must be a real directory") from exc
    value = os.fstat(descriptor)
    identity = (value.st_dev, value.st_ino)
    if not stat.S_ISDIR(value.st_mode) or (
        expected is not None and identity != expected
    ):
        os.close(descriptor)
        raise SkillPathError("skill directory changed after validation")
    return descriptor


def _quarantine_directory_at(
    root_fd: int,
    name: str,
    *,
    expected: tuple[int, int],
) -> str:
    """Atomically detach one expected directory before durable retirement."""

    try:
        before = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
    except OSError as exc:
        raise SkillPathError("local skill folder changed during deletion") from exc
    if not stat.S_ISDIR(before.st_mode) or (before.st_dev, before.st_ino) != expected:
        raise SkillPathError("local skill folder changed during deletion")

    quarantine = f".{name}.delete.{uuid.uuid4().hex}"
    try:
        os.rename(
            name,
            quarantine,
            src_dir_fd=root_fd,
            dst_dir_fd=root_fd,
        )
    except OSError as exc:
        raise SkillPathError("local skill folder changed during deletion") from exc

    moved = _identity_at(root_fd, quarantine)
    if moved == expected:
        return quarantine

    # A filesystem actor won the race between the identity check and rename.
    # Never delete the moved replacement, and do not attempt a check-then-rename
    # restoration that could itself overwrite a newly raced destination.
    raise SkillPathError(
        "local skill folder changed during deletion; "
        f"unexpected folder preserved as {quarantine}"
    )


def _remove_expected_directory_at(
    root_fd: int,
    name: str,
    *,
    expected: tuple[int, int],
) -> None:
    """Remove exactly one pinned child, preserving any raced replacement."""

    quarantine = _quarantine_directory_at(root_fd, name, expected=expected)
    _remove_quarantined_directory_at(
        root_fd,
        quarantine,
        expected=expected,
    )


def _remove_quarantined_directory_at(
    root_fd: int,
    quarantine: str,
    *,
    expected: tuple[int, int],
) -> None:
    """Move an owned detached generation into durable implementation trash.

    Public and recovery-visible names are never recursively unlinked. Physical
    cleanup happens only under ``.kestrel-internal``, whose contents are an
    implementation-owned domain and are not a supported direct-edit surface.
    """

    internal_fd = _open_directory_at(root_fd, INTERNAL_DIRECTORY)
    try:
        with _serialized_skill_trash(internal_fd):
            if _identity_at(root_fd, quarantine) != expected:
                raise SkillPathError(
                    "quarantined skill directory changed before retirement; "
                    f"its replacement was preserved as {quarantine}"
                )
            retired = f"{SKILL_TRASH_PREFIX}{uuid.uuid4().hex}"
            _rename_directory_no_replace_at(
                root_fd,
                quarantine,
                retired,
                destination_directory_fd=internal_fd,
            )
            if _identity_at(internal_fd, retired) != expected:
                _fsync_directory_pair(root_fd, internal_fd)
                raise SkillPathError(
                    "quarantined skill directory changed during retirement; "
                    f"the unexpected generation was preserved as {retired}"
                )
            _fsync_directory_pair(root_fd, internal_fd)
            try:
                _purge_internal_directory_at(
                    internal_fd,
                    retired,
                    expected=expected,
                )
            except (OSError, SkillPathError):
                # Retirement already made the user-visible delete durable.
                # Preserve partial/private recovery state for startup retry.
                pass
    finally:
        os.close(internal_fd)


def _purge_internal_directory_at(
    internal_fd: int,
    name: str,
    *,
    expected: tuple[int, int],
) -> None:
    """Recursively purge one pinned implementation-owned private directory."""

    frames: list[_RemovalFrame] = []
    try:
        descriptor = _open_directory_at(internal_fd, name, expected=expected)
        frames.append(
            _RemovalFrame(
                parent_fd=internal_fd,
                name=name,
                identity=expected,
                descriptor=descriptor,
                entries=iter(os.listdir(descriptor)),
            )
        )
        while frames:
            frame = frames[-1]
            try:
                entry = next(frame.entries)
            except StopIteration:
                if _identity_at(frame.parent_fd, frame.name) != frame.identity:
                    raise SkillPathError(
                        "quarantined skill directory changed during deletion; "
                        "its replacement was preserved"
                    )
                os.close(frame.descriptor)
                frame.descriptor = -1
                os.rmdir(frame.name, dir_fd=frame.parent_fd)
                frames.pop()
                continue

            value = os.stat(entry, dir_fd=frame.descriptor, follow_symlinks=False)
            entry_identity = (value.st_dev, value.st_ino)
            # A valid resource component can already occupy NAME_MAX bytes.
            # Keep recovery names independent of attacker-controlled length.
            detached = f".kestrel-delete-{uuid.uuid4().hex}"
            _rename_directory_no_replace_at(frame.descriptor, entry, detached)
            if _identity_at(frame.descriptor, detached) != entry_identity:
                raise SkillPathError(
                    "nested skill entry changed during deletion; "
                    f"its replacement was preserved as {detached}"
                )
            if stat.S_ISDIR(value.st_mode):
                child_descriptor = _open_directory_at(
                    frame.descriptor,
                    detached,
                    expected=entry_identity,
                )
                frames.append(
                    _RemovalFrame(
                        parent_fd=frame.descriptor,
                        name=detached,
                        identity=entry_identity,
                        descriptor=child_descriptor,
                        entries=iter(os.listdir(child_descriptor)),
                    )
                )
            else:
                if _identity_at(frame.descriptor, detached) != entry_identity:
                    raise SkillPathError(
                        "nested skill file changed during deletion; "
                        f"its replacement was preserved as {detached}"
                    )
                os.unlink(detached, dir_fd=frame.descriptor)
    except BaseException as exc:
        exc.add_note(
            "private cleanup did not complete; remaining implementation-owned "
            f"data, if any, is preserved as {name}"
        )
        raise
    finally:
        for frame in reversed(frames):
            if frame.descriptor >= 0:
                os.close(frame.descriptor)


def _restore_quarantined_directory_at(
    root_fd: int,
    quarantine: str,
    name: str,
    *,
    expected: tuple[int, int],
) -> None:
    """Restore preserved rollback data without overwriting a raced replacement."""

    if _identity_at(root_fd, quarantine) != expected:
        raise SkillPathError(
            "quarantined skill folder changed while rollback inspected it; "
            f"remaining data is preserved as {quarantine}"
        )
    try:
        _rename_directory_no_replace_at(
            root_fd,
            quarantine,
            name,
        )
    except FileExistsError as exc:
        raise SkillPathError(
            "skill name was republished while rollback inspected its prior data; "
            f"the prior data is preserved as {quarantine}"
        ) from exc
    if _identity_at(root_fd, name) != expected:
        raise SkillPathError(
            "restored skill folder changed identity after rollback inspection"
        )


def _open_parent_at(
    root_fd: int,
    relative: Path,
    *,
    create: bool = False,
    created_parents: list[CreatedParentDirectory] | None = None,
) -> tuple[int, str]:
    """Traverse a relative parent chain without following mutable symlinks."""

    descriptor = os.dup(root_fd)
    try:
        for part in relative.parts[:-1]:
            created_identity: tuple[int, int] | None = None
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                else:
                    try:
                        created = os.stat(
                            part, dir_fd=descriptor, follow_symlinks=False
                        )
                    except OSError as exc:
                        raise SkillPathError(
                            "new skill path parent could not be inspected"
                        ) from exc
                    if not stat.S_ISDIR(created.st_mode):
                        raise SkillPathError("new skill path parent is not a directory")
                    created_identity = (created.st_dev, created.st_ino)
                    if created_parents is not None:
                        created_parents.append(
                            CreatedParentDirectory(
                                parent_descriptor=os.dup(descriptor),
                                name=part,
                                identity=created_identity,
                            )
                        )
            try:
                child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
            except OSError as exc:
                raise SkillPathError(
                    "skill path parents must be real directories without symlinks"
                ) from exc
            value = os.fstat(child)
            if not stat.S_ISDIR(value.st_mode):
                os.close(child)
                raise SkillPathError("skill path parent is not a directory")
            if (
                created_identity is not None
                and (
                    value.st_dev,
                    value.st_ino,
                )
                != created_identity
            ):
                os.close(child)
                raise SkillPathError("new skill path parent changed during creation")
            os.close(descriptor)
            descriptor = child
        return descriptor, relative.name
    except BaseException:
        os.close(descriptor)
        raise


def _release_created_parent_directories(
    claims: list[CreatedParentDirectory],
    *,
    rollback: bool,
) -> None:
    """Close creation claims, removing only unchanged empty parents on rollback."""

    cleanup_error: OSError | None = None
    for claim in reversed(claims):
        try:
            if rollback and _identity_at(claim.parent_descriptor, claim.name) == (
                claim.identity
            ):
                try:
                    os.rmdir(claim.name, dir_fd=claim.parent_descriptor)
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    # A raced writer may have populated the new directory. Keep
                    # that content instead of turning rollback into deletion.
                    if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                        cleanup_error = cleanup_error or exc
        finally:
            os.close(claim.parent_descriptor)
    claims.clear()
    if cleanup_error is not None:
        raise SkillPublicationCleanupError(
            "failed resource edit left a newly created parent directory"
        ) from cleanup_error


def _atomic_replace_file_at(
    directory_fd: int,
    name: str,
    payload: bytes,
    *,
    artifact_directory_fd: int | None = None,
    artifact_label: str | None = None,
) -> None:
    if len(payload) > MAX_EDITOR_FILE_BYTES:
        raise SkillFormatError(f"editor file exceeds {MAX_EDITOR_FILE_BYTES} bytes")
    artifact_fd = (
        artifact_directory_fd if artifact_directory_fd is not None else directory_fd
    )
    temporary = (
        f".{validate_skill_name(artifact_label)}.resource.tmp.{uuid.uuid4().hex}"
        if artifact_label is not None
        else f".tmp.{uuid.uuid4().hex}"
    )
    _write_tmp_at(artifact_fd, temporary, payload)
    try:
        os.replace(
            temporary,
            name,
            src_dir_fd=artifact_fd,
            dst_dir_fd=directory_fd,
        )
    finally:
        _unlink_at(artifact_fd, temporary)


def _copy_regular_file_at(source: Path, directory_fd: int, name: str) -> None:
    """Copy one staged regular file into a pinned destination directory."""

    source_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        source_fd = os.open(source, source_flags)
    except OSError as exc:
        raise SkillPathError("staged skill resource must be a regular file") from exc
    destination_fd: int | None = None
    try:
        if not stat.S_ISREG(os.fstat(source_fd).st_mode):
            raise SkillPathError("staged skill resource must be a regular file")
        destination_fd = os.open(
            name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=directory_fd,
        )
        with (
            os.fdopen(source_fd, "rb", closefd=False) as source_handle,
            os.fdopen(destination_fd, "wb", closefd=False) as destination_handle,
        ):
            shutil.copyfileobj(source_handle, destination_handle)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
    finally:
        os.close(source_fd)
        if destination_fd is not None:
            os.close(destination_fd)


def atomic_write_primary(folder: Path, payload: bytes, *, overwrite: bool) -> None:
    """Write ``SKILL.md`` with the inherited hardlink claim protocol.

    A per-writer temporary file is hardlinked to one shared claim. The first
    writer wins; a second writer cannot remove the winner's claim because
    cleanup is conditional on this writer having created the link. Creates use
    a second non-overwriting hardlink for finalization. Intentional edits use
    ``os.replace`` only after obtaining the same exclusive claim.
    """

    descriptor = _open_directory(folder)
    try:
        _atomic_write_primary_at(
            descriptor,
            folder.name,
            payload,
            overwrite=overwrite,
        )
    finally:
        os.close(descriptor)


def atomic_replace_file(path: Path, payload: bytes) -> None:
    """Atomically replace an operator-edited resource after containment checks."""

    if len(payload) > MAX_EDITOR_FILE_BYTES:
        raise SkillFormatError(f"editor file exceeds {MAX_EDITOR_FILE_BYTES} bytes")
    tmp = path.with_name(f".tmp.{uuid.uuid4().hex}")
    _write_tmp(tmp, payload)
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _materialize_validated_folder(
    snapshot: ValidatedSkillFolder,
    destination: Path,
) -> None:
    """Rebuild one bounded descriptor snapshot in a private staging folder."""

    destination.mkdir(mode=0o700)
    for entry in snapshot.entries:
        target = destination.joinpath(*Path(entry.path).parts)
        if entry.is_directory:
            target.mkdir(mode=0o700)
            continue
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        assert entry.payload is not None
        _write_tmp(target, entry.payload)


def _inspect_child_at(
    root_fd: int,
    name: str,
    *,
    expected: tuple[int, int] | None,
    folder_name: str | None = None,
) -> ValidatedSkillFolder:
    """Inspect one child without reopening its mutable lexical path."""

    descriptor = _open_directory_at(root_fd, name, expected=expected)
    try:
        return inspect_skill_folder_descriptor(
            descriptor,
            folder_name=folder_name or name,
        )
    finally:
        os.close(descriptor)


def _require_record_snapshot(
    record: SkillRecord,
    snapshot: ValidatedSkillFolder,
) -> None:
    """Require captured folder bytes to match the record being mutated/read."""

    if not record.content_revision:
        return
    if record.folder_identity is None:
        raise SkillConflictError(
            f"skill {record.name!r} has no folder generation; reload and retry"
        )
    current = _folder_revision(snapshot, record.folder_identity)
    if not hmac.compare_digest(current, record.content_revision):
        raise SkillConflictError(
            f"skill {record.name!r} changed after it was read; reload and retry"
        )


class SkillStore:
    """Mutate only the agent-local root; read from the resolved catalog."""

    def __init__(self, local_root: Path):
        local_root.mkdir(parents=True, exist_ok=True)
        try:
            root_stat = local_root.lstat()
        except OSError as exc:
            raise SkillPathError(
                "agent-local skill root must be a real directory"
            ) from exc
        root_identity = (root_stat.st_dev, root_stat.st_ino)
        if not stat.S_ISDIR(root_stat.st_mode) or local_root.is_symlink():
            raise SkillPathError("agent-local skill root must not be a symlink")
        # Open the lexical root without following its final component before
        # resolving it.  Otherwise a rename-to-symlink race between the check
        # above and ``resolve()`` can make another agent's directory the
        # authoritative local store.
        descriptor = _open_directory(local_root, expected=root_identity)
        try:
            try:
                os.mkdir(INTERNAL_DIRECTORY, mode=0o700, dir_fd=descriptor)
            except FileExistsError:
                pass
            try:
                internal_stat = os.stat(
                    INTERNAL_DIRECTORY,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise SkillPathError(
                    "skill internal directory could not be inspected"
                ) from exc
            if not stat.S_ISDIR(internal_stat.st_mode):
                raise SkillPathError("skill internal path must be a real directory")
            internal_identity = (internal_stat.st_dev, internal_stat.st_ino)
            internal_descriptor = _open_directory_at(
                descriptor,
                INTERNAL_DIRECTORY,
                expected=internal_identity,
            )
            os.close(internal_descriptor)
            try:
                resolved_root = local_root.resolve(strict=True)
            except OSError as exc:
                raise SkillPathError(
                    "agent-local skill root changed while resolving it"
                ) from exc
            verification = _open_directory(resolved_root, expected=root_identity)
            os.close(verification)
            try:
                lexical_stat = local_root.lstat()
            except OSError as exc:
                raise SkillPathError(
                    "agent-local skill root changed while resolving it"
                ) from exc
            if (
                not stat.S_ISDIR(lexical_stat.st_mode)
                or (lexical_stat.st_dev, lexical_stat.st_ino) != root_identity
            ):
                raise SkillPathError(
                    "agent-local skill root changed while resolving it"
                )
            lexical_verification = _open_directory(
                local_root,
                expected=root_identity,
            )
            os.close(lexical_verification)
            internal_root = resolved_root / INTERNAL_DIRECTORY
            internal_verification = _open_directory(
                internal_root,
                expected=internal_identity,
            )
            os.close(internal_verification)
        finally:
            os.close(descriptor)
        self.local_root = resolved_root
        self._local_root_identity = root_identity
        self._internal_root = internal_root
        self._internal_root_identity = internal_identity
        self._reap_retired_skill_folders()
        self._reap_git_checkout_workspaces()

    @staticmethod
    def _lock_git_checkout_workspace(workspace_fd: int) -> int | None:
        """Lock one checkout workspace, returning ``None`` when it is active."""

        flags = (
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            descriptor = os.open(
                GIT_CHECKOUT_LOCK,
                flags,
                0o600,
                dir_fd=workspace_fd,
            )
        except OSError as exc:
            raise SkillPathError("could not open the Git checkout owner lock") from exc
        try:
            value = os.fstat(descriptor)
            if not stat.S_ISREG(value.st_mode):
                raise SkillPathError("Git checkout owner lock must be a regular file")
            identity = (value.st_dev, value.st_ino)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(descriptor)
                return None
        except BaseException:
            os.close(descriptor)
            raise
        try:
            if _identity_at(workspace_fd, GIT_CHECKOUT_LOCK) != identity:
                raise SkillPathError(
                    "Git checkout owner lock changed while acquiring it"
                )
            return descriptor
        except BaseException:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
            raise

    def _git_checkout_workspace_names(self) -> tuple[str, ...]:
        internal_fd = _open_directory(
            self._internal_root,
            expected=self._internal_root_identity,
        )
        try:
            with os.scandir(internal_fd) as entries:
                return tuple(
                    entry.name
                    for entry in entries
                    if entry.name.startswith(GIT_CHECKOUT_PREFIX)
                )
        finally:
            os.close(internal_fd)

    def _reap_retired_skill_folders(self) -> None:
        """Physically clean only generations already in the private trash domain."""

        internal_fd = _open_directory(
            self._internal_root,
            expected=self._internal_root_identity,
        )
        try:
            with _serialized_skill_trash(internal_fd):
                with os.scandir(internal_fd) as entries:
                    candidates = tuple(
                        (entry.name, entry.stat(follow_symlinks=False))
                        for entry in entries
                        if entry.name.startswith(SKILL_TRASH_PREFIX)
                    )
                for name, value in candidates:
                    if not stat.S_ISDIR(value.st_mode):
                        continue
                    identity = (value.st_dev, value.st_ino)
                    try:
                        _purge_internal_directory_at(
                            internal_fd,
                            name,
                            expected=identity,
                        )
                    except (OSError, SkillPathError):
                        # The private domain is best-effort retention cleanup.
                        # Preserve anything that changed and retry next startup.
                        continue
        finally:
            os.close(internal_fd)

    def _reap_git_checkout_workspaces(self) -> None:
        """Remove crash-orphaned checkouts while preserving live processes."""

        # Avoid creating the shared publication lock in an otherwise pristine
        # store. If a candidate appears after this advisory scan, its owner lock
        # still protects it and a later initialization will reconcile it.
        if not self._git_checkout_workspace_names():
            return
        internal_fd = _open_directory(
            self._internal_root,
            expected=self._internal_root_identity,
        )
        try:
            with _serialized_source_publication(
                self.local_root,
                internal_fd,
                enabled=True,
            ):
                with os.scandir(internal_fd) as entries:
                    candidates = tuple(
                        (entry.name, entry.stat(follow_symlinks=False))
                        for entry in entries
                        if entry.name.startswith(GIT_CHECKOUT_PREFIX)
                    )
                for name, value in candidates:
                    if not stat.S_ISDIR(value.st_mode):
                        continue
                    identity = (value.st_dev, value.st_ino)
                    workspace_fd = _open_directory_at(
                        internal_fd,
                        name,
                        expected=identity,
                    )
                    try:
                        owner_lock = self._lock_git_checkout_workspace(workspace_fd)
                    finally:
                        os.close(workspace_fd)
                    if owner_lock is None:
                        continue
                    try:
                        _purge_internal_directory_at(
                            internal_fd,
                            name,
                            expected=identity,
                        )
                    finally:
                        fcntl.flock(owner_lock, fcntl.LOCK_UN)
                        os.close(owner_lock)
        finally:
            os.close(internal_fd)

    def _remove_git_checkout_workspace(
        self,
        name: str,
        *,
        expected: tuple[int, int],
    ) -> None:
        """Remove one owned checkout workspace under the cross-process lock."""

        internal_fd = _open_directory(
            self._internal_root,
            expected=self._internal_root_identity,
        )
        try:
            with _serialized_source_publication(
                self.local_root,
                internal_fd,
                enabled=True,
            ):
                if _identity_at(internal_fd, name) is not None:
                    # The private workspace name is stable recovery state, not
                    # a public record. Remove it in place so a kill during
                    # cleanup leaves the same prefix for the next reaper pass.
                    _purge_internal_directory_at(
                        internal_fd,
                        name,
                        expected=expected,
                    )
        finally:
            os.close(internal_fd)

    @contextmanager
    def git_checkout_workspace(self):
        """Yield a crash-recoverable, exclusively owned Git staging folder."""

        name = f"{GIT_CHECKOUT_PREFIX}{uuid.uuid4().hex}"
        identity: tuple[int, int] | None = None
        owner_lock: int | None = None
        owner_lock_attempted = False
        try:
            internal_fd = _open_directory(
                self._internal_root,
                expected=self._internal_root_identity,
            )
            try:
                with _serialized_source_publication(
                    self.local_root,
                    internal_fd,
                    enabled=True,
                ):
                    os.mkdir(name, mode=0o700, dir_fd=internal_fd)
                    identity = _identity_at(internal_fd, name)
                    if identity is None:
                        raise SkillPathError("Git checkout workspace was not created")
                    workspace_fd = _open_directory_at(
                        internal_fd,
                        name,
                        expected=identity,
                    )
                    try:
                        owner_lock_attempted = True
                        owner_lock = self._lock_git_checkout_workspace(workspace_fd)
                    finally:
                        os.close(workspace_fd)
                    if owner_lock is None:
                        raise SkillPathError(
                            "new Git checkout workspace could not be owned"
                        )
            finally:
                os.close(internal_fd)
        except BaseException:
            try:
                if identity is not None and (
                    not owner_lock_attempted or owner_lock is not None
                ):
                    self._remove_git_checkout_workspace(name, expected=identity)
            finally:
                if owner_lock is not None:
                    fcntl.flock(owner_lock, fcntl.LOCK_UN)
                    os.close(owner_lock)
            raise

        workspace = self._internal_root / name
        try:
            yield workspace
        finally:
            try:
                assert identity is not None
                self._remove_git_checkout_workspace(name, expected=identity)
            finally:
                if owner_lock is not None:
                    fcntl.flock(owner_lock, fcntl.LOCK_UN)
                    os.close(owner_lock)

    @property
    def local_root_identity(self) -> tuple[int, int]:
        """Return the inode identity pinned when this store was constructed."""

        return self._local_root_identity

    def try_acquire_publication_state_claim(
        self, name: str
    ) -> PublicationStateClaim | None:
        """Try to claim one name across state guards and filesystem publication."""

        name = validate_skill_name(name)
        bucket = _name_lock_bucket(
            self.local_root,
            name,
            domain="publication-state",
        )
        thread_lock = _PUBLICATION_STATE_LOCKS[bucket]
        if not thread_lock.acquire(blocking=False):
            return None

        root_fd: int | None = None
        internal_fd: int | None = None
        descriptor: int | None = None
        locked = False
        claimed = False
        try:
            root_fd = _open_directory(
                self.local_root,
                expected=self._local_root_identity,
            )
            internal_fd = _open_directory(
                self._internal_root,
                expected=self._internal_root_identity,
            )
            lock_name = _publication_state_lock_name(self.local_root, name)
            flags = (
                os.O_RDWR
                | os.O_CREAT
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0)
            )
            try:
                descriptor = os.open(lock_name, flags, 0o600, dir_fd=internal_fd)
            except OSError as exc:
                raise SkillPathError(
                    "could not open the skill publication-state lock"
                ) from exc
            value = os.fstat(descriptor)
            if not stat.S_ISREG(value.st_mode):
                raise SkillPathError(
                    "skill publication-state lock must be a regular file"
                )
            lock_identity = (value.st_dev, value.st_ino)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return None
            locked = True
            if _identity_at(internal_fd, lock_name) != lock_identity:
                raise SkillPathError(
                    "skill publication-state lock changed while acquiring it"
                )
            claimed = True
            return PublicationStateClaim(descriptor, thread_lock)
        finally:
            if internal_fd is not None:
                os.close(internal_fd)
            if root_fd is not None:
                os.close(root_fd)
            if not claimed:
                if descriptor is not None:
                    try:
                        if locked:
                            fcntl.flock(descriptor, fcntl.LOCK_UN)
                    finally:
                        os.close(descriptor)
                thread_lock.release()

    @staticmethod
    def get(snapshot: CatalogSnapshot, name: str) -> SkillRecord:
        name = validate_skill_name(name)
        record = snapshot.by_name().get(name)
        if record is None:
            raise SkillNotFoundError(f"skill {name!r} was not found")
        return record

    def create(self, document: SkillDocument) -> Path:
        """Create a skill and return its public folder path."""

        folder, _identity = self.create_pinned(document)
        return folder

    def create_pinned(
        self, document: SkillDocument
    ) -> tuple[Path, CreatedSkillPublication]:
        """Create a skill and retain the inode identity captured at publication."""

        name = validate_skill_name(document.name)
        payload = serialize_skill_markdown(document).encode("utf-8")
        with tempfile.TemporaryDirectory(prefix=".kestrel-skill-create-") as temporary:
            staged_root = Path(temporary).resolve(strict=True)
            staged_folder = staged_root / name
            staged_folder.mkdir(mode=0o700)
            (staged_folder / SKILL_FILENAME).write_bytes(payload)
            validate_skill_folder(staged_folder, source_root=staged_root)

        with _serialized_skill_mutation(
            self.local_root,
            self._internal_root,
            name,
            root_identity=self._local_root_identity,
            internal_root_identity=self._internal_root_identity,
            source_publication=True,
        ) as (root_fd, _artifact_fd):
            _require_source_publication_capacity_at(root_fd)
            folder = direct_child(
                self.local_root,
                name,
                root_identity=self._local_root_identity,
            )
            if _identity_at(root_fd, name) is not None:
                raise SkillConflictError(f"skill already exists: {name}")
            staging_name = f".{name}.create.{uuid.uuid4().hex}"
            created_identity: tuple[int, int] | None = None
            cleanup_name = staging_name
            publication_collision = False
            try:
                try:
                    os.mkdir(staging_name, mode=0o700, dir_fd=root_fd)
                except FileExistsError as exc:
                    raise SkillPathError(
                        "create staging name unexpectedly collided"
                    ) from exc
                created_identity = _identity_at(root_fd, staging_name)
                if created_identity is None:
                    raise SkillPathError(
                        "created skill staging folder vanished before publication"
                    )
                folder_fd = _open_directory_at(
                    root_fd,
                    staging_name,
                    expected=created_identity,
                )
                try:
                    _atomic_write_primary_at(
                        folder_fd,
                        name,
                        payload,
                        overwrite=False,
                    )
                    if set(os.listdir(folder_fd)) != {SKILL_FILENAME}:
                        raise SkillConflictError(
                            "created skill staging folder changed during publication"
                        )
                    validate_skill_folder_descriptor(
                        folder_fd,
                        folder_name=name,
                    )
                    primary = os.stat(
                        SKILL_FILENAME,
                        dir_fd=folder_fd,
                        follow_symlinks=False,
                    )
                    published = CreatedSkillPublication(
                        folder_identity=created_identity,
                        primary_identity=(primary.st_dev, primary.st_ino),
                        primary_size=primary.st_size,
                        primary_mtime_ns=primary.st_mtime_ns,
                        primary_ctime_ns=primary.st_ctime_ns,
                    )
                    try:
                        os.fsync(folder_fd)
                    except OSError:
                        pass
                finally:
                    os.close(folder_fd)
                if _identity_at(root_fd, staging_name) != created_identity:
                    raise SkillPathError(
                        "created skill staging folder changed during publication"
                    )
                verification_fd = _open_directory(
                    self.local_root,
                    expected=self._local_root_identity,
                )
                os.close(verification_fd)
                if _identity_at(root_fd, name) is not None:
                    raise SkillConflictError(f"skill already exists: {name}")
                try:
                    _rename_directory_no_replace_at(root_fd, staging_name, name)
                except FileExistsError as exc:
                    publication_collision = True
                    raise SkillConflictError(f"skill already exists: {name}") from exc
                cleanup_name = name
                if _identity_at(root_fd, name) != created_identity:
                    raise SkillPathError(
                        "created skill folder changed during publication"
                    )
                try:
                    os.fsync(root_fd)
                except OSError:
                    pass
            except BaseException as publication_error:
                if (
                    created_identity is not None
                    and _identity_at(root_fd, cleanup_name) is not None
                ):
                    try:
                        _remove_expected_directory_at(
                            root_fd,
                            cleanup_name,
                            expected=created_identity,
                        )
                    except BaseException as cleanup_error:
                        cleanup_error.add_note(
                            f"skill creation originally failed: {publication_error}"
                        )
                        raise SkillPublicationCleanupError(
                            "skill creation cleanup could not confirm removal: "
                            f"{cleanup_error}"
                        ) from cleanup_error
                if (
                    cleanup_name != name
                    and _identity_at(root_fd, name) is not None
                    and not publication_collision
                ):
                    raise SkillPublicationCleanupError(
                        "skill creation failed while an unowned same-named folder "
                        "appeared; its removal was not attempted"
                    ) from publication_error
                raise
        return folder, published

    @staticmethod
    def _same_snapshot(
        left: ValidatedSkillFolder,
        right: ValidatedSkillFolder,
    ) -> bool:
        return left.document == right.document and tuple(
            sorted(left.entries, key=lambda item: item.path)
        ) == tuple(sorted(right.entries, key=lambda item: item.path))

    def _publish_replacement_folder(
        self,
        *,
        root_fd: int,
        staged_root: Path,
        record: SkillRecord,
        candidate: ValidatedSkillFolder,
    ) -> None:
        """Exchange a complete folder and atomically compensate stale writers."""

        staged_fd = _open_directory(staged_root)
        exchanged = False
        expected_identity = record.folder_identity or _identity_at(root_fd, record.name)
        try:
            staged_value = os.stat(
                record.name,
                dir_fd=staged_fd,
                follow_symlinks=False,
            )
            candidate_identity = (staged_value.st_dev, staged_value.st_ino)
            if not stat.S_ISDIR(staged_value.st_mode):
                raise SkillPathError("validated edit staging folder changed")
            candidate_fd = _open_directory_at(
                staged_fd,
                record.name,
                expected=candidate_identity,
            )
            try:
                _fsync_validated_directories_at(candidate_fd, candidate)
            finally:
                os.close(candidate_fd)
            if expected_identity is None:
                raise SkillConflictError(
                    f"skill {record.name!r} has no folder generation; reload and retry"
                )

            current = _inspect_child_at(
                root_fd,
                record.name,
                expected=expected_identity,
            )
            _require_record_snapshot(record, current)
            _exchange_directories_at(
                staged_fd,
                record.name,
                root_fd,
                record.name,
            )
            exchanged = True
            _fsync_directory_pair(staged_fd, root_fd)
            if (
                _identity_at(root_fd, record.name) != candidate_identity
                or _identity_at(staged_fd, record.name) != expected_identity
            ):
                raise SkillConflictError(
                    f"skill {record.name!r} changed during publication; retry"
                )

            detached = _inspect_child_at(
                staged_fd,
                record.name,
                expected=expected_identity,
                folder_name=record.name,
            )
            _require_record_snapshot(record, detached)
            published = _inspect_child_at(
                root_fd,
                record.name,
                expected=candidate_identity,
            )
            if not self._same_snapshot(published, candidate):
                raise SkillConflictError(
                    f"edited skill {record.name!r} changed during publication; retry"
                )
        except BaseException as publication_error:
            if exchanged:
                rollback_safe = False
                try:
                    if (
                        _identity_at(root_fd, record.name) == candidate_identity
                        and _identity_at(staged_fd, record.name) == expected_identity
                    ):
                        current_candidate = _inspect_child_at(
                            root_fd,
                            record.name,
                            expected=candidate_identity,
                        )
                        rollback_safe = self._same_snapshot(
                            current_candidate,
                            candidate,
                        )
                except (SkillPathError, SkillFormatError, OSError):
                    rollback_safe = False
                if rollback_safe:
                    try:
                        _exchange_directories_at(
                            staged_fd,
                            record.name,
                            root_fd,
                            record.name,
                        )
                        preserved = f".kestrel-edit-conflict-{uuid.uuid4().hex}"
                        _rename_directory_no_replace_at(
                            staged_fd,
                            record.name,
                            preserved,
                            destination_directory_fd=root_fd,
                        )
                        _fsync_directory_pair(staged_fd, root_fd)
                        if _identity_at(root_fd, record.name) != expected_identity:
                            raise SkillPathError(
                                "prior skill generation was not restored after edit conflict"
                            )
                        publication_error.add_note(
                            "the displaced candidate generation was preserved for "
                            f"recovery as {preserved}"
                        )
                    except BaseException as rollback_error:
                        rollback_error.add_note(
                            f"skill edit originally failed: {publication_error}"
                        )
                        raise SkillPublicationCleanupError(
                            "skill edit rollback could not restore the prior generation: "
                            f"{rollback_error}"
                        ) from rollback_error
                else:
                    preserved = f".kestrel-edit-conflict-{uuid.uuid4().hex}"
                    try:
                        if _identity_at(staged_fd, record.name) is not None:
                            _rename_directory_no_replace_at(
                                staged_fd,
                                record.name,
                                preserved,
                                destination_directory_fd=root_fd,
                            )
                            _fsync_directory_pair(staged_fd, root_fd)
                    except BaseException as preservation_error:
                        preservation_error.add_note(
                            f"skill edit originally failed: {publication_error}"
                        )
                        raise SkillPublicationCleanupError(
                            "skill edit conflict generations could not both be preserved: "
                            f"{preservation_error}"
                        ) from preservation_error
                    publication_error.add_note(
                        "the prior generation was preserved for recovery as "
                        f"{preserved}; the concurrently changed public generation was "
                        "left untouched"
                    )
                    raise SkillPublicationCleanupError(
                        "skill edit detected a concurrent change after atomic publication; "
                        "both generations were preserved"
                    ) from publication_error
            raise
        finally:
            os.close(staged_fd)

    def edit_primary(self, record: SkillRecord, content: str) -> SkillDocument:
        document = parse_skill_markdown(
            content, source=f"{record.name}/{SKILL_FILENAME}"
        )
        if document.name != record.name:
            raise SkillFormatError("frontmatter name cannot rename a skill folder")
        try:
            payload = content.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise SkillFormatError("editor content must be valid UTF-8 text") from exc
        with _serialized_skill_mutation(
            self.local_root,
            self._internal_root,
            record.name,
            root_identity=self._local_root_identity,
            internal_root_identity=self._internal_root_identity,
        ) as (root_fd, _artifact_fd):
            self._require_local(record)
            snapshot = _inspect_child_at(
                root_fd,
                record.name,
                expected=record.folder_identity,
            )
            _require_record_snapshot(record, snapshot)
            with self.git_checkout_workspace() as workspace:
                staged_root = workspace.resolve(strict=True)
                staged_folder = staged_root / record.name
                _materialize_validated_folder(snapshot, staged_folder)
                atomic_write_primary(staged_folder, payload, overwrite=True)
                validate_skill_folder(staged_folder, source_root=staged_root)
                candidate = inspect_skill_folder(staged_folder, source_root=staged_root)
                self._publish_replacement_folder(
                    root_fd=root_fd,
                    staged_root=staged_root,
                    record=record,
                    candidate=candidate,
                )
        return candidate.document

    def rollback_created(
        self,
        folder: Path,
        *,
        identity: CreatedSkillPublication | tuple[int, int],
    ) -> None:
        """Remove this operation's new folder without following a replacement."""

        name = validate_skill_name(folder.name)
        expected = self.local_root / name
        if expected != folder:
            raise SkillPathError(
                "created skill folder changed before persistence rollback"
            )
        folder_identity = (
            identity.folder_identity
            if isinstance(identity, CreatedSkillPublication)
            else identity
        )
        with _serialized_skill_mutation(
            self.local_root,
            self._internal_root,
            name,
            root_identity=self._local_root_identity,
            internal_root_identity=self._internal_root_identity,
        ) as (root_fd, _artifact_fd):
            if _identity_at(root_fd, name) is None:
                return
            quarantine = _quarantine_directory_at(
                root_fd,
                name,
                expected=folder_identity,
            )
            if isinstance(identity, CreatedSkillPublication):
                try:
                    descriptor = _open_directory_at(
                        root_fd,
                        quarantine,
                        expected=folder_identity,
                    )
                    try:
                        if set(os.listdir(descriptor)) != {SKILL_FILENAME}:
                            raise SkillPathError(
                                "created skill contents changed; rollback preserved them"
                            )
                        primary = os.stat(
                            SKILL_FILENAME,
                            dir_fd=descriptor,
                            follow_symlinks=False,
                        )
                        observed_primary = (
                            primary.st_dev,
                            primary.st_ino,
                            primary.st_size,
                            primary.st_mtime_ns,
                            primary.st_ctime_ns,
                        )
                        expected_primary = (
                            *identity.primary_identity,
                            identity.primary_size,
                            identity.primary_mtime_ns,
                            identity.primary_ctime_ns,
                        )
                        if observed_primary != expected_primary:
                            raise SkillPathError(
                                "created skill contents changed; rollback preserved them"
                            )
                    finally:
                        os.close(descriptor)
                except BaseException as inspection_error:
                    try:
                        _restore_quarantined_directory_at(
                            root_fd,
                            quarantine,
                            name,
                            expected=folder_identity,
                        )
                    except Exception as restoration_error:  # noqa: BLE001
                        inspection_error.add_note(str(restoration_error))
                    raise
            _remove_quarantined_directory_at(
                root_fd,
                quarantine,
                expected=folder_identity,
            )

    def rollback_installed(
        self,
        folder: Path,
        *,
        identity: InstalledSkillPublication,
    ) -> None:
        """Remove an unchanged install without following or deleting a replacement."""

        name = validate_skill_name(folder.name)
        expected = self.local_root / name
        if expected != folder:
            raise SkillPathError(
                "installed skill folder changed before publication rollback"
            )
        with _serialized_skill_mutation(
            self.local_root,
            self._internal_root,
            name,
            root_identity=self._local_root_identity,
            internal_root_identity=self._internal_root_identity,
        ) as (root_fd, _artifact_fd):
            if _identity_at(root_fd, name) is None:
                return
            quarantine = _quarantine_directory_at(
                root_fd,
                name,
                expected=identity.folder_identity,
            )
            try:
                current = _inspect_child_at(
                    root_fd,
                    quarantine,
                    expected=identity.folder_identity,
                    folder_name=name,
                )
                current_entries = tuple(
                    sorted(current.entries, key=lambda item: item.path)
                )
                published_entries = tuple(
                    sorted(identity.snapshot.entries, key=lambda item: item.path)
                )
                if (
                    current.document != identity.snapshot.document
                    or current_entries != published_entries
                ):
                    raise SkillPathError(
                        "installed skill contents changed; rollback preserved them"
                    )
            except BaseException as inspection_error:
                try:
                    _restore_quarantined_directory_at(
                        root_fd,
                        quarantine,
                        name,
                        expected=identity.folder_identity,
                    )
                except Exception as restoration_error:  # noqa: BLE001
                    inspection_error.add_note(str(restoration_error))
                raise
            _remove_quarantined_directory_at(
                root_fd,
                quarantine,
                expected=identity.folder_identity,
            )

    def read_file(self, record: SkillRecord, relative_path: str) -> str:
        validate_resource_path(relative_path)
        folder = self._require_real_folder(record)
        folder_fd = _open_directory(folder, expected=record.folder_identity)
        try:
            snapshot = inspect_skill_folder_descriptor(
                folder_fd,
                folder_name=record.name,
            )
        finally:
            os.close(folder_fd)
        _require_record_snapshot(record, snapshot)
        entry = next(
            (item for item in snapshot.entries if item.path == relative_path),
            None,
        )
        if entry is None or entry.payload is None:
            raise SkillPathError(
                "requested skill path must be an inventoried regular file"
            )
        payload = entry.payload
        try:
            return payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SkillFormatError("the editor only opens UTF-8 text files") from exc

    def write_file(self, record: SkillRecord, relative_path: str, content: str) -> None:
        validate_resource_path(relative_path)
        folder = self._require_local(record)
        if not isinstance(content, str):
            raise SkillFormatError("editor content must be text")
        path = lexical_contained_path(folder, relative_path, must_exist=False)
        relative = path.relative_to(folder)
        if relative.as_posix() == SKILL_FILENAME:
            self.edit_primary(record, content)
            return
        reject_symlink_chain(folder, path)
        if path.suffix == ".py":
            if not relative.parts or relative.parts[0] != "scripts":
                raise SkillPathError("Python files are allowed only below scripts/")
        elif path.suffix != ".md":
            raise SkillPathError(
                "the editor writes only .md resources and scripts/*.py"
            )
        try:
            payload = content.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise SkillFormatError("editor content must be valid UTF-8 text") from exc
        with _serialized_skill_mutation(
            self.local_root,
            self._internal_root,
            record.name,
            root_identity=self._local_root_identity,
            internal_root_identity=self._internal_root_identity,
        ) as (root_fd, _artifact_fd):
            folder = self._require_local(record)
            path = lexical_contained_path(folder, relative.as_posix(), must_exist=False)
            reject_symlink_chain(folder, path)
            snapshot = _inspect_child_at(
                root_fd,
                record.name,
                expected=record.folder_identity,
            )
            _require_record_snapshot(record, snapshot)
            with self.git_checkout_workspace() as workspace:
                staged_root = workspace.resolve(strict=True)
                staged_folder = staged_root / record.name
                _materialize_validated_folder(snapshot, staged_folder)
                staged_path = lexical_contained_path(
                    staged_folder, relative.as_posix(), must_exist=False
                )
                reject_symlink_chain(staged_folder, staged_path)
                staged_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                reject_symlink_chain(staged_folder, staged_path)
                atomic_replace_file(staged_path, payload)
                validate_skill_folder(staged_folder, source_root=staged_root)
                candidate = inspect_skill_folder(
                    staged_folder,
                    source_root=staged_root,
                )
                self._publish_replacement_folder(
                    root_fd=root_fd,
                    staged_root=staged_root,
                    record=record,
                    candidate=candidate,
                )

    @staticmethod
    def file_is_editable(record: SkillRecord, relative_path: str) -> bool:
        """Return whether ``write_file`` accepts this existing file path."""

        path = Path(relative_path)
        return bool(
            record.editable
            and (
                relative_path == SKILL_FILENAME
                or path.suffix == ".md"
                or (
                    path.suffix == ".py"
                    and bool(path.parts)
                    and path.parts[0] == "scripts"
                )
            )
        )

    def tree(self, record: SkillRecord) -> tuple[dict[str, object], ...]:
        folder = self._require_real_folder(record)
        folder_fd = _open_directory(folder, expected=record.folder_identity)
        try:
            snapshot = inspect_skill_folder_descriptor(
                folder_fd,
                folder_name=record.name,
            )
        finally:
            os.close(folder_fd)
        _require_record_snapshot(record, snapshot)
        entries: list[dict[str, object]] = []
        for entry in sorted(snapshot.entries, key=lambda item: item.path):
            path = Path(entry.path)
            if len(path.parts) == 1 and (
                path.name == PROVENANCE_FILENAME
                or path.name == PRIMARY_WRITER_CLAIM
                or path.name.startswith(PRIMARY_WRITER_TEMP_PREFIX)
            ):
                continue
            is_file = not entry.is_directory
            editable = bool(is_file and self.file_is_editable(record, entry.path))
            entries.append(
                {
                    "path": entry.path,
                    "type": "file" if is_file else "directory",
                    "bytes": len(entry.payload) if entry.payload is not None else None,
                    "editable": editable,
                    "execution_risk": bool(
                        is_file and has_python_execution_risk(entry.path)
                    ),
                }
            )
        return tuple(entries)

    def install_folder(
        self,
        source_folder: Path,
        *,
        provenance: SkillProvenance,
    ) -> Path:
        """Install a validated folder and return its local path."""

        target, _publication = self.install_folder_pinned(
            source_folder,
            provenance=provenance,
        )
        return target

    def install_folder_pinned(
        self,
        source_folder: Path,
        *,
        provenance: SkillProvenance,
    ) -> tuple[Path, InstalledSkillPublication]:
        """Install a folder and return evidence for safe compensating rollback."""

        source_snapshot = inspect_skill_folder(
            source_folder, source_root=source_folder.parent
        )
        document = source_snapshot.document
        with tempfile.TemporaryDirectory(prefix=".kestrel-skill-install-") as temporary:
            staged_root = Path(temporary).resolve(strict=True)
            staged_folder = staged_root / document.name
            _materialize_validated_folder(source_snapshot, staged_folder)
            provenance_payload = serialize_provenance(provenance)
            atomic_replace_file(
                staged_folder / PROVENANCE_FILENAME,
                provenance_payload,
            )
            document = validate_skill_folder(staged_folder, source_root=staged_root)
            primary_payload = (staged_folder / SKILL_FILENAME).read_bytes()

            with _serialized_skill_mutation(
                self.local_root,
                self._internal_root,
                document.name,
                root_identity=self._local_root_identity,
                internal_root_identity=self._internal_root_identity,
                source_publication=True,
            ) as (root_fd, _artifact_fd):
                _require_source_publication_capacity_at(root_fd)
                target = direct_child(
                    self.local_root,
                    document.name,
                    root_identity=self._local_root_identity,
                )
                if _identity_at(root_fd, document.name) is not None:
                    raise SkillConflictError(f"skill already exists: {document.name}")
                staging_name = f".{document.name}.install.{uuid.uuid4().hex}"
                created_identity: tuple[int, int] | None = None
                publication: InstalledSkillPublication | None = None
                cleanup_name = staging_name
                publication_collision = False
                try:
                    os.mkdir(staging_name, mode=0o700, dir_fd=root_fd)
                except FileExistsError as exc:
                    raise SkillPathError(
                        "install staging name unexpectedly collided"
                    ) from exc
                created_identity = _identity_at(root_fd, staging_name)
                try:
                    if created_identity is None:
                        raise SkillPathError(
                            "installed skill staging folder vanished before publication"
                        )
                    target_fd = _open_directory_at(
                        root_fd,
                        staging_name,
                        expected=created_identity,
                    )
                    try:
                        for path in sorted(
                            staged_folder.rglob("*"), key=lambda item: item.as_posix()
                        ):
                            relative = path.relative_to(staged_folder)
                            if relative.as_posix() in {
                                SKILL_FILENAME,
                                PROVENANCE_FILENAME,
                            }:
                                continue
                            parent_fd, filename = _open_parent_at(
                                target_fd,
                                relative,
                                create=True,
                            )
                            try:
                                if path.is_dir():
                                    try:
                                        os.mkdir(filename, mode=0o700, dir_fd=parent_fd)
                                    except FileExistsError:
                                        pass
                                elif path.is_file():
                                    _copy_regular_file_at(path, parent_fd, filename)
                                else:
                                    raise SkillPathError(
                                        "unsupported remote resource: "
                                        f"{relative.as_posix()}"
                                    )
                            finally:
                                os.close(parent_fd)
                        _atomic_replace_file_at(
                            target_fd,
                            PROVENANCE_FILENAME,
                            provenance_payload,
                        )
                        _atomic_write_primary_at(
                            target_fd,
                            document.name,
                            primary_payload,
                            overwrite=False,
                        )
                        published_snapshot = inspect_skill_folder_descriptor(
                            target_fd,
                            folder_name=document.name,
                        )
                        publication = InstalledSkillPublication(
                            folder_identity=created_identity,
                            snapshot=published_snapshot,
                        )
                        _fsync_validated_directories_at(
                            target_fd,
                            published_snapshot,
                        )
                    finally:
                        os.close(target_fd)
                    if _identity_at(root_fd, staging_name) != created_identity:
                        raise SkillPathError(
                            "installed skill staging folder changed during publication"
                        )
                    verification_fd = _open_directory(
                        self.local_root,
                        expected=self._local_root_identity,
                    )
                    os.close(verification_fd)
                    if _identity_at(root_fd, document.name) is not None:
                        raise SkillConflictError(
                            f"skill already exists: {document.name}"
                        )
                    try:
                        _rename_directory_no_replace_at(
                            root_fd,
                            staging_name,
                            document.name,
                        )
                    except FileExistsError as exc:
                        publication_collision = True
                        raise SkillConflictError(
                            f"skill already exists: {document.name}"
                        ) from exc
                    cleanup_name = document.name
                    if _identity_at(root_fd, document.name) != created_identity:
                        raise SkillPathError(
                            "installed skill folder changed during publication"
                        )
                    try:
                        os.fsync(root_fd)
                    except OSError:
                        pass
                except BaseException as publication_error:
                    if (
                        created_identity is not None
                        and _identity_at(root_fd, cleanup_name) is not None
                    ):
                        try:
                            _remove_expected_directory_at(
                                root_fd,
                                cleanup_name,
                                expected=created_identity,
                            )
                        except BaseException as cleanup_error:
                            cleanup_error.add_note(
                                "skill installation originally failed: "
                                f"{publication_error}"
                            )
                            raise SkillPublicationCleanupError(
                                "skill installation cleanup could not confirm removal: "
                                f"{cleanup_error}"
                            ) from cleanup_error
                    if (
                        cleanup_name != document.name
                        and _identity_at(root_fd, document.name) is not None
                        and not publication_collision
                    ):
                        raise SkillPublicationCleanupError(
                            "skill installation failed while an unowned same-named "
                            "folder appeared; its removal was not attempted"
                        ) from publication_error
                    raise
        if publication is None:
            raise SkillPublicationCleanupError(
                "skill installation completed without rollback evidence"
            )
        return target, publication

    def delete(self, record: SkillRecord) -> None:
        with _serialized_skill_mutation(
            self.local_root,
            self._internal_root,
            record.name,
            root_identity=self._local_root_identity,
            internal_root_identity=self._internal_root_identity,
        ) as (root_fd, _artifact_fd):
            expected = self._require_local(record)
            try:
                current = expected.lstat()
            except OSError as exc:
                raise SkillPathError(
                    "local skill folder changed during deletion"
                ) from exc
            expected_identity = record.folder_identity or (
                current.st_dev,
                current.st_ino,
            )
            quarantine = _quarantine_directory_at(
                root_fd,
                record.name,
                expected=expected_identity,
            )
            try:
                snapshot = _inspect_child_at(
                    root_fd,
                    quarantine,
                    expected=expected_identity,
                    folder_name=record.name,
                )
                _require_record_snapshot(record, snapshot)
            except BaseException as inspection_error:
                try:
                    _restore_quarantined_directory_at(
                        root_fd,
                        quarantine,
                        record.name,
                        expected=expected_identity,
                    )
                except Exception as restoration_error:  # noqa: BLE001
                    inspection_error.add_note(str(restoration_error))
                raise
            _remove_quarantined_directory_at(
                root_fd,
                quarantine,
                expected=expected_identity,
            )

    def assert_current(self, record: SkillRecord) -> None:
        """Check one folder generation under the cooperative mutation lock."""

        if record.folder != self.local_root / record.name:
            folder = self._require_real_folder(record)
            folder_fd = _open_directory(folder, expected=record.folder_identity)
            try:
                snapshot = inspect_skill_folder_descriptor(
                    folder_fd,
                    folder_name=record.name,
                )
            finally:
                os.close(folder_fd)
            _require_record_snapshot(record, snapshot)
            return

        with _serialized_skill_mutation(
            self.local_root,
            self._internal_root,
            record.name,
            root_identity=self._local_root_identity,
            internal_root_identity=self._internal_root_identity,
        ) as (root_fd, _artifact_fd):
            self._require_local(record)
            snapshot = _inspect_child_at(
                root_fd,
                record.name,
                expected=record.folder_identity,
            )
            _require_record_snapshot(record, snapshot)

    def require_local_record(self, record: SkillRecord) -> None:
        """Fail before persistent side effects when a record is not mutable here."""

        self._require_local(record)

    @staticmethod
    def search(snapshot: CatalogSnapshot, query: str) -> tuple[SkillRecord, ...]:
        if not isinstance(query, str) or not query.strip():
            raise SkillFormatError("search query must not be empty")
        needle = query.casefold().strip()
        return tuple(
            record
            for record in snapshot.records
            if needle in record.name.casefold()
            or needle in record.document.description.casefold()
        )

    @staticmethod
    def _require_real_folder(record: SkillRecord) -> Path:
        try:
            current = record.folder.lstat()
        except OSError as exc:
            raise SkillPathError(
                "skill folder changed after discovery; reload before accessing it"
            ) from exc
        identity = (current.st_dev, current.st_ino)
        if (
            record.folder.is_symlink()
            or not record.folder.is_dir()
            or (
                record.folder_identity is not None
                and identity != record.folder_identity
            )
        ):
            raise SkillPathError(
                "skill folder changed after discovery; reload before accessing it"
            )
        return record.folder

    def _require_local(self, record: SkillRecord) -> Path:
        if not record.editable:
            raise SkillReadOnlyError(
                f"skill {record.name!r} comes from {record.source_id!r}; create a local override to edit it"
            )
        expected = self.local_root / record.name
        try:
            current = expected.lstat()
        except OSError as exc:
            raise SkillPathError(
                "local skill folder changed after discovery; reload before mutating it"
            ) from exc
        identity = (current.st_dev, current.st_ino)
        if (
            record.folder != expected
            or expected.is_symlink()
            or not expected.is_dir()
            or (
                record.folder_identity is not None
                and identity != record.folder_identity
            )
        ):
            raise SkillPathError(
                "local skill folder changed after discovery; reload before mutating it"
            )
        return expected


__all__ = [
    "CLAIM_STALENESS_SECONDS",
    "INTERNAL_DIRECTORY",
    "MAX_EDITOR_FILE_BYTES",
    "NAME_LOCK_BUCKETS",
    "SkillStore",
    "atomic_replace_file",
    "atomic_write_primary",
]
