"""File-authoritative skill storage with atomic primary-record writes."""

from __future__ import annotations

import fcntl
import os
import shutil
import stat
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .errors import (
    SkillConflictError,
    SkillFormatError,
    SkillNotFoundError,
    SkillPathError,
    SkillReadOnlyError,
)
from .format import (
    MAX_SKILL_FILE_BYTES,
    SKILL_FILENAME,
    parse_skill_markdown,
    serialize_skill_markdown,
    validate_resource_path,
    validate_skill_folder,
    validate_skill_name,
)
from .models import CatalogSnapshot, SkillDocument, SkillProvenance, SkillRecord
from .paths import (
    direct_child,
    lexical_contained_path,
    reject_symlink_chain,
)
from .sources import PROVENANCE_FILENAME, serialize_provenance

CLAIM_STALENESS_SECONDS = 60
MAX_EDITOR_FILE_BYTES = 262_144
_INTERNAL_PREFIXES = (".SKILL.md.tmp.", ".SKILL.md.claim")
_MUTATION_LOCKS: dict[tuple[str, str], threading.RLock] = {}
_MUTATION_LOCKS_GUARD = threading.Lock()
_PUBLICATION_STATE_LOCKS: dict[tuple[str, str], threading.Lock] = {}
_PUBLICATION_STATE_LOCKS_GUARD = threading.Lock()
_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)


@dataclass(frozen=True, slots=True)
class CreatedSkillPublication:
    """Inode evidence proving a new folder still has its published contents."""

    folder_identity: tuple[int, int]
    primary_identity: tuple[int, int]
    primary_size: int
    primary_mtime_ns: int
    primary_ctime_ns: int


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
def _serialized_skill_mutation(
    root: Path,
    name: str,
    *,
    root_identity: tuple[int, int],
):
    """Serialize validation and publication across threads and processes."""

    key = (str(root), validate_skill_name(name))
    with _MUTATION_LOCKS_GUARD:
        thread_lock = _MUTATION_LOCKS.setdefault(key, threading.RLock())
    with thread_lock:
        root_fd = _open_directory(root, expected=root_identity)
        lock_name = f".{name}.mutation.lock"
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            try:
                descriptor = os.open(lock_name, flags, 0o600, dir_fd=root_fd)
            except OSError as exc:
                raise SkillPathError("could not open the skill mutation lock") from exc
            locked = False
            try:
                value = os.fstat(descriptor)
                if not stat.S_ISREG(value.st_mode):
                    raise SkillPathError("skill mutation lock must be a regular file")
                lock_identity = (value.st_dev, value.st_ino)
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                locked = True
                if _identity_at(root_fd, lock_name) != lock_identity:
                    raise SkillPathError(
                        "skill mutation lock changed while acquiring it"
                    )
                yield root_fd
            finally:
                try:
                    if locked:
                        fcntl.flock(descriptor, fcntl.LOCK_UN)
                finally:
                    os.close(descriptor)
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
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _write_tmp_at(directory_fd: int, name: str, payload: bytes) -> None:
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
        dir_fd=directory_fd,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _identity_at(directory_fd: int, name: str) -> tuple[int, int] | None:
    try:
        value = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    return value.st_dev, value.st_ino


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
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory_fd,
        )
    except OSError:
        return None
    try:
        value = os.fstat(descriptor)
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


def _unlink_at(directory_fd: int, name: str) -> None:
    try:
        os.unlink(name, dir_fd=directory_fd)
    except FileNotFoundError:
        pass


def _atomic_write_primary_at(
    directory_fd: int, folder_name: str, payload: bytes, *, overwrite: bool
) -> None:
    """Apply the primary hardlink-claim protocol inside one pinned directory."""

    if len(payload) > MAX_SKILL_FILE_BYTES:
        raise SkillFormatError(f"{SKILL_FILENAME} exceeds {MAX_SKILL_FILE_BYTES} bytes")
    if not overwrite and _identity_at(directory_fd, SKILL_FILENAME) is not None:
        raise SkillConflictError(f"skill already exists: {folder_name}")
    temporary = f".SKILL.md.tmp.{uuid.uuid4().hex}"
    claim = ".SKILL.md.claim"
    owned_claim: tuple[int, int] | None = None
    claim_lock: int | None = None
    _write_tmp_at(directory_fd, temporary, payload)
    temporary_identity = _identity_at(directory_fd, temporary)
    if temporary_identity is None:  # pragma: no cover - fsync'd file vanished
        _unlink_at(directory_fd, temporary)
        raise OSError("skill writer temporary file vanished before claiming")
    try:
        try:
            os.link(
                temporary,
                claim,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
            owned_claim = temporary_identity
            claim_lock = _lock_claim_at(directory_fd, claim, expected=owned_claim)
            if claim_lock is None:
                raise SkillConflictError(
                    f"skill write could not lock its claim for {folder_name}"
                )
        except FileExistsError:
            if not _claim_is_stale_at(directory_fd, claim):
                raise SkillConflictError(
                    f"concurrent skill write is already in progress for {folder_name}"
                ) from None
            stale_identity = _identity_at(directory_fd, claim)
            stale_lock = _lock_claim_at(directory_fd, claim, expected=stale_identity)
            if stale_lock is None:
                raise SkillConflictError(
                    f"active skill writer still owns the stale claim for {folder_name}"
                ) from None
            try:
                if (
                    stale_identity is None
                    or _identity_at(directory_fd, claim) != stale_identity
                    or not _claim_is_stale_at(directory_fd, claim)
                ):
                    raise SkillConflictError(
                        f"stale skill claim changed during recovery for {folder_name}"
                    )
                os.unlink(claim, dir_fd=directory_fd)
                try:
                    os.link(
                        temporary,
                        claim,
                        src_dir_fd=directory_fd,
                        dst_dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                    owned_claim = temporary_identity
                except FileExistsError as exc:
                    raise SkillConflictError(
                        f"concurrent skill write won stale-claim recovery for {folder_name}"
                    ) from exc
                claim_lock = _lock_claim_at(directory_fd, claim, expected=owned_claim)
                if claim_lock is None:
                    raise SkillConflictError(
                        f"skill write could not lock its reclaimed claim for {folder_name}"
                    )
            finally:
                _unlock_claim(stale_lock)
        if owned_claim is None or _identity_at(directory_fd, claim) != owned_claim:
            raise SkillConflictError(
                f"skill write lost its reclaimed claim for {folder_name}"
            )
        if overwrite:
            os.replace(
                temporary,
                SKILL_FILENAME,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
        else:
            try:
                os.link(
                    temporary,
                    SKILL_FILENAME,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except FileExistsError as exc:
                raise SkillConflictError(
                    f"skill already exists: {folder_name}"
                ) from exc
    finally:
        _unlink_at(directory_fd, temporary)
        if owned_claim is not None and _identity_at(directory_fd, claim) == owned_claim:
            _unlink_at(directory_fd, claim)
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
    """Atomically detach one expected directory before recursive deletion."""

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
    descriptor: int | None = None
    try:
        # Pin the quarantined inode before traversing it. Resolving the
        # quarantine name again through ``shutil.rmtree`` would let a raced
        # replacement become the recursive-deletion target.
        descriptor = _open_directory_at(root_fd, quarantine, expected=expected)
        for entry in os.listdir(descriptor):
            value = os.stat(entry, dir_fd=descriptor, follow_symlinks=False)
            if stat.S_ISDIR(value.st_mode):
                shutil.rmtree(entry, dir_fd=descriptor)
            else:
                os.unlink(entry, dir_fd=descriptor)
        if _identity_at(root_fd, quarantine) != expected:
            raise SkillPathError(
                "quarantined skill folder changed during deletion; "
                "its replacement was preserved"
            )
        os.rmdir(quarantine, dir_fd=root_fd)
    except BaseException as exc:
        exc.add_note(
            "skill removal did not complete; remaining data, if any, "
            f"is preserved as {quarantine}"
        )
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _open_parent_at(
    root_fd: int, relative: Path, *, create: bool = False
) -> tuple[int, str]:
    """Traverse a relative parent chain without following mutable symlinks."""

    descriptor = os.dup(root_fd)
    try:
        for part in relative.parts[:-1]:
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
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
            os.close(descriptor)
            descriptor = child
        return descriptor, relative.name
    except BaseException:
        os.close(descriptor)
        raise


def _atomic_replace_file_at(directory_fd: int, name: str, payload: bytes) -> None:
    if len(payload) > MAX_EDITOR_FILE_BYTES:
        raise SkillFormatError(f"editor file exceeds {MAX_EDITOR_FILE_BYTES} bytes")
    temporary = f".tmp.{uuid.uuid4().hex}"
    _write_tmp_at(directory_fd, temporary, payload)
    try:
        os.replace(
            temporary,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
    finally:
        _unlink_at(directory_fd, temporary)


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
        finally:
            os.close(descriptor)
        self.local_root = resolved_root
        self._local_root_identity = root_identity

    @property
    def local_root_identity(self) -> tuple[int, int]:
        """Return the inode identity pinned when this store was constructed."""

        return self._local_root_identity

    def try_acquire_publication_state_claim(
        self, name: str
    ) -> PublicationStateClaim | None:
        """Try to claim one name across state guards and filesystem publication."""

        name = validate_skill_name(name)
        key = (str(self.local_root), name)
        with _PUBLICATION_STATE_LOCKS_GUARD:
            thread_lock = _PUBLICATION_STATE_LOCKS.setdefault(key, threading.Lock())
        if not thread_lock.acquire(blocking=False):
            return None

        root_fd: int | None = None
        descriptor: int | None = None
        locked = False
        claimed = False
        try:
            root_fd = _open_directory(
                self.local_root,
                expected=self._local_root_identity,
            )
            lock_name = f".{name}.publication-state.lock"
            flags = (
                os.O_RDWR
                | os.O_CREAT
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0)
            )
            try:
                descriptor = os.open(lock_name, flags, 0o600, dir_fd=root_fd)
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
            if _identity_at(root_fd, lock_name) != lock_identity:
                raise SkillPathError(
                    "skill publication-state lock changed while acquiring it"
                )
            claimed = True
            return PublicationStateClaim(descriptor, thread_lock)
        finally:
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

        created_identity: tuple[int, int] | None = None
        with _serialized_skill_mutation(
            self.local_root,
            name,
            root_identity=self._local_root_identity,
        ) as root_fd:
            try:
                folder = direct_child(
                    self.local_root,
                    name,
                    root_identity=self._local_root_identity,
                )
                try:
                    os.mkdir(name, mode=0o700, dir_fd=root_fd)
                except FileExistsError as exc:
                    raise SkillConflictError(f"skill already exists: {name}") from exc
                created_identity = _identity_at(root_fd, name)
                if created_identity is None:
                    raise SkillPathError(
                        "created skill folder vanished before publication"
                    )
                folder_fd = _open_directory_at(
                    root_fd,
                    name,
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
                            "created skill folder changed during publication"
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
                finally:
                    os.close(folder_fd)
                if _identity_at(root_fd, name) != created_identity:
                    raise SkillPathError(
                        "created skill folder changed during publication"
                    )
                verification_fd = _open_directory(
                    self.local_root,
                    expected=self._local_root_identity,
                )
                os.close(verification_fd)
            except BaseException as publication_error:
                if created_identity is not None:
                    try:
                        _remove_expected_directory_at(
                            root_fd,
                            name,
                            expected=created_identity,
                        )
                    except BaseException as cleanup_error:
                        cleanup_error.add_note(
                            f"skill creation originally failed: {publication_error}"
                        )
                        raise cleanup_error from publication_error
                raise
        return folder, published

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
            record.name,
            root_identity=self._local_root_identity,
        ) as root_fd:
            folder = self._require_local(record)
            with tempfile.TemporaryDirectory(
                prefix=".kestrel-skill-edit-"
            ) as temporary:
                staged_root = Path(temporary).resolve(strict=True)
                staged_folder = staged_root / record.name
                shutil.copytree(folder, staged_folder, symlinks=True)
                atomic_write_primary(staged_folder, payload, overwrite=True)
                candidate = validate_skill_folder(
                    staged_folder, source_root=staged_root
                )

            # Validation and publication share the per-skill lock, so every
            # candidate includes the preceding resource/primary mutation.
            folder = self._require_local(record)
            descriptor = _open_directory_at(
                root_fd,
                record.name,
                expected=record.folder_identity,
            )
            try:
                _atomic_write_primary_at(
                    descriptor,
                    record.name,
                    payload,
                    overwrite=True,
                )
            finally:
                os.close(descriptor)
        return candidate

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
            name,
            root_identity=self._local_root_identity,
        ) as root_fd:
            if _identity_at(root_fd, name) is None:
                return
            if isinstance(identity, CreatedSkillPublication):
                descriptor = _open_directory_at(
                    root_fd,
                    name,
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
            _remove_expected_directory_at(
                root_fd,
                name,
                expected=folder_identity,
            )

    def read_file(self, record: SkillRecord, relative_path: str) -> str:
        validate_resource_path(relative_path)
        folder = self._require_real_folder(record)
        path = lexical_contained_path(folder, relative_path, must_exist=True)
        reject_symlink_chain(folder, path)
        folder_fd = _open_directory(folder, expected=record.folder_identity)
        try:
            parent_fd, filename = _open_parent_at(folder_fd, path.relative_to(folder))
            try:
                flags = (
                    os.O_RDONLY
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NONBLOCK", 0)
                )
                try:
                    descriptor = os.open(filename, flags, dir_fd=parent_fd)
                except OSError as exc:
                    raise SkillPathError(
                        "requested skill path must be a regular file without symlinks"
                    ) from exc
                try:
                    value = os.fstat(descriptor)
                    if not stat.S_ISREG(value.st_mode):
                        raise SkillPathError(
                            "requested skill path must be a regular file"
                        )
                    if value.st_size > MAX_EDITOR_FILE_BYTES:
                        raise SkillFormatError(
                            f"file exceeds {MAX_EDITOR_FILE_BYTES} bytes"
                        )
                    with os.fdopen(descriptor, "rb", closefd=False) as handle:
                        payload = handle.read(MAX_EDITOR_FILE_BYTES + 1)
                    if len(payload) > MAX_EDITOR_FILE_BYTES:
                        raise SkillFormatError(
                            f"file exceeds {MAX_EDITOR_FILE_BYTES} bytes"
                        )
                finally:
                    os.close(descriptor)
            finally:
                os.close(parent_fd)
        finally:
            os.close(folder_fd)
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
            record.name,
            root_identity=self._local_root_identity,
        ) as root_fd:
            folder = self._require_local(record)
            path = lexical_contained_path(folder, relative.as_posix(), must_exist=False)
            reject_symlink_chain(folder, path)
            with tempfile.TemporaryDirectory(
                prefix=".kestrel-skill-edit-"
            ) as temporary:
                staged_root = Path(temporary).resolve(strict=True)
                staged_folder = staged_root / record.name
                shutil.copytree(folder, staged_folder, symlinks=True)
                staged_path = lexical_contained_path(
                    staged_folder, relative.as_posix(), must_exist=False
                )
                reject_symlink_chain(staged_folder, staged_path)
                staged_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                reject_symlink_chain(staged_folder, staged_path)
                atomic_replace_file(staged_path, payload)
                validate_skill_folder(staged_folder, source_root=staged_root)

            # The complete candidate was valid. The held lock makes publication
            # part of the same serial transaction as complete-folder validation.
            folder = self._require_local(record)
            path = lexical_contained_path(folder, relative.as_posix(), must_exist=False)
            reject_symlink_chain(folder, path)
            folder_fd = _open_directory_at(
                root_fd,
                record.name,
                expected=record.folder_identity,
            )
            try:
                parent_fd, filename = _open_parent_at(
                    folder_fd,
                    path.relative_to(folder),
                    create=True,
                )
                try:
                    _atomic_replace_file_at(parent_fd, filename, payload)
                finally:
                    os.close(parent_fd)
            finally:
                os.close(folder_fd)

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
        validate_skill_folder(folder, source_root=folder.parent)
        entries: list[dict[str, object]] = []
        for path in sorted(folder.rglob("*"), key=lambda item: item.as_posix()):
            relative = path.relative_to(folder).as_posix()
            if path.parent == folder and (
                path.name == PROVENANCE_FILENAME
                or path.name.startswith(_INTERNAL_PREFIXES)
            ):
                continue
            if path.is_symlink():
                raise SkillPathError(f"symlink appeared during tree read: {relative}")
            is_file = path.is_file()
            editable = bool(is_file and self.file_is_editable(record, relative))
            entries.append(
                {
                    "path": relative,
                    "type": "file" if is_file else "directory",
                    "bytes": path.stat().st_size if is_file else None,
                    "editable": editable,
                    "execution_risk": bool(is_file and path.suffix == ".py"),
                }
            )
        return tuple(entries)

    def install_folder(
        self,
        source_folder: Path,
        *,
        provenance: SkillProvenance,
    ) -> Path:
        document = validate_skill_folder(
            source_folder, source_root=source_folder.parent
        )
        with tempfile.TemporaryDirectory(prefix=".kestrel-skill-install-") as temporary:
            staged_root = Path(temporary).resolve(strict=True)
            staged_folder = staged_root / document.name
            shutil.copytree(source_folder, staged_folder, symlinks=True)
            provenance_payload = serialize_provenance(provenance)
            atomic_replace_file(
                staged_folder / PROVENANCE_FILENAME,
                provenance_payload,
            )
            document = validate_skill_folder(staged_folder, source_root=staged_root)
            primary_payload = (staged_folder / SKILL_FILENAME).read_bytes()

            root_fd = _open_directory(
                self.local_root,
                expected=self._local_root_identity,
            )
            created_identity: tuple[int, int] | None = None
            try:
                target = direct_child(
                    self.local_root,
                    document.name,
                    root_identity=self._local_root_identity,
                )
                try:
                    os.mkdir(document.name, mode=0o700, dir_fd=root_fd)
                except FileExistsError as exc:
                    raise SkillConflictError(
                        f"skill already exists: {document.name}"
                    ) from exc
                created_identity = _identity_at(root_fd, document.name)
                if created_identity is None:
                    raise SkillPathError(
                        "installed skill folder vanished before publication"
                    )
                target_fd = _open_directory_at(
                    root_fd,
                    document.name,
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
                                    f"unsupported remote resource: {relative.as_posix()}"
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
                finally:
                    os.close(target_fd)
                if _identity_at(root_fd, document.name) != created_identity:
                    raise SkillPathError(
                        "installed skill folder changed during publication"
                    )
                verification_fd = _open_directory(
                    self.local_root,
                    expected=self._local_root_identity,
                )
                os.close(verification_fd)
            except BaseException as publication_error:
                if created_identity is not None:
                    try:
                        _remove_expected_directory_at(
                            root_fd,
                            document.name,
                            expected=created_identity,
                        )
                    except BaseException as cleanup_error:
                        cleanup_error.add_note(
                            f"skill installation originally failed: {publication_error}"
                        )
                        raise cleanup_error from publication_error
                raise
            finally:
                os.close(root_fd)
        return target

    def delete(self, record: SkillRecord) -> None:
        with _serialized_skill_mutation(
            self.local_root,
            record.name,
            root_identity=self._local_root_identity,
        ) as root_fd:
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
            _remove_expected_directory_at(
                root_fd,
                record.name,
                expected=expected_identity,
            )

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
    "MAX_EDITOR_FILE_BYTES",
    "SkillStore",
    "atomic_replace_file",
    "atomic_write_primary",
]
