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
    validate_document_references,
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
_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)


@contextmanager
def _serialized_skill_mutation(root: Path, name: str):
    """Serialize validation and publication across threads and processes."""

    key = (str(root), validate_skill_name(name))
    with _MUTATION_LOCKS_GUARD:
        thread_lock = _MUTATION_LOCKS.setdefault(key, threading.RLock())
    with thread_lock:
        lock_path = root / f".{name}.mutation.lock"
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(lock_path, flags, 0o600)
        except OSError as exc:
            raise SkillPathError("could not open the skill mutation lock") from exc
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise SkillPathError("skill mutation lock must be a regular file")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


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
    temporary = f".{name}.tmp.{uuid.uuid4().hex}"
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
    tmp = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex}")
    _write_tmp(tmp, payload)
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


class SkillStore:
    """Mutate only the agent-local root; read from the resolved catalog."""

    def __init__(self, local_root: Path):
        local_root.mkdir(parents=True, exist_ok=True)
        if local_root.is_symlink():
            raise SkillPathError("agent-local skill root must not be a symlink")
        self.local_root = local_root.resolve(strict=True)
        root_stat = self.local_root.stat()
        self._local_root_identity = (root_stat.st_dev, root_stat.st_ino)

    @staticmethod
    def get(snapshot: CatalogSnapshot, name: str) -> SkillRecord:
        name = validate_skill_name(name)
        record = snapshot.by_name().get(name)
        if record is None:
            raise SkillNotFoundError(f"skill {name!r} was not found")
        return record

    def create(self, document: SkillDocument) -> Path:
        name = validate_skill_name(document.name)
        folder = direct_child(self.local_root, name)
        try:
            folder.mkdir(mode=0o700)
        except FileExistsError as exc:
            raise SkillConflictError(f"skill already exists: {name}") from exc
        try:
            validate_document_references(document, folder)
            atomic_write_primary(
                folder,
                serialize_skill_markdown(document).encode("utf-8"),
                overwrite=False,
            )
            validate_skill_folder(folder, source_root=self.local_root)
        except (SkillFormatError, SkillPathError, SkillConflictError, OSError):
            if folder.exists() and not any(folder.iterdir()):
                folder.rmdir()
            elif folder.exists() and not (folder / SKILL_FILENAME).exists():
                shutil.rmtree(folder)
            raise
        return folder

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
        with _serialized_skill_mutation(self.local_root, record.name):
            folder = self._require_local(record)
            with tempfile.TemporaryDirectory(
                prefix=".kestrel-skill-edit-", dir=self.local_root
            ) as temporary:
                staged_root = Path(temporary)
                staged_folder = staged_root / record.name
                shutil.copytree(folder, staged_folder, symlinks=True)
                atomic_write_primary(staged_folder, payload, overwrite=True)
                candidate = validate_skill_folder(
                    staged_folder, source_root=staged_root
                )

            # Validation and publication share the per-skill lock, so every
            # candidate includes the preceding resource/primary mutation.
            folder = self._require_local(record)
            descriptor = _open_directory(
                folder,
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

    def rollback_created(self, folder: Path, *, identity: tuple[int, int]) -> None:
        """Remove this operation's new folder without following a replacement."""

        expected = direct_child(self.local_root, folder.name)
        try:
            current = expected.lstat()
        except FileNotFoundError:
            return
        if (
            expected != folder
            or expected.is_symlink()
            or not expected.is_dir()
            or (current.st_dev, current.st_ino) != identity
        ):
            raise SkillPathError(
                "created skill folder changed before persistence rollback"
            )
        shutil.rmtree(expected)

    def read_file(self, record: SkillRecord, relative_path: str) -> str:
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
        with _serialized_skill_mutation(self.local_root, record.name):
            folder = self._require_local(record)
            path = lexical_contained_path(folder, relative.as_posix(), must_exist=False)
            reject_symlink_chain(folder, path)
            with tempfile.TemporaryDirectory(
                prefix=".kestrel-skill-edit-", dir=self.local_root
            ) as temporary:
                staged_root = Path(temporary)
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
            reject_symlink_chain(folder, path)
            folder_fd = _open_directory(folder, expected=record.folder_identity)
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
            if path.name == PROVENANCE_FILENAME or path.name.startswith(
                _INTERNAL_PREFIXES
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
        target = direct_child(self.local_root, document.name)
        try:
            target.mkdir(mode=0o700)
        except FileExistsError as exc:
            raise SkillConflictError(f"skill already exists: {document.name}") from exc
        try:
            for path in sorted(
                source_folder.rglob("*"), key=lambda item: item.as_posix()
            ):
                relative = path.relative_to(source_folder)
                if relative.as_posix() in {SKILL_FILENAME, PROVENANCE_FILENAME}:
                    continue
                destination = target / relative
                if path.is_dir():
                    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
                elif path.is_file():
                    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                    shutil.copyfile(path, destination, follow_symlinks=False)
                else:
                    raise SkillPathError(
                        f"unsupported remote resource: {relative.as_posix()}"
                    )
            atomic_replace_file(
                target / PROVENANCE_FILENAME, serialize_provenance(provenance)
            )
            atomic_write_primary(
                target,
                serialize_skill_markdown(document).encode("utf-8"),
                overwrite=False,
            )
            validate_skill_folder(target, source_root=self.local_root)
        except (SkillFormatError, SkillPathError, SkillConflictError, OSError):
            if target.exists():
                shutil.rmtree(target)
            raise
        return target

    def delete(self, record: SkillRecord) -> None:
        with _serialized_skill_mutation(self.local_root, record.name):
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
            root_fd = _open_directory(
                self.local_root,
                expected=self._local_root_identity,
            )
            quarantine: str | None = None
            try:
                quarantine = _quarantine_directory_at(
                    root_fd,
                    record.name,
                    expected=expected_identity,
                )
                shutil.rmtree(quarantine, dir_fd=root_fd)
            except BaseException as exc:
                if quarantine is not None:
                    exc.add_note(
                        "skill deletion did not complete; remaining data, if any, "
                        f"is preserved as {quarantine}"
                    )
                raise
            finally:
                os.close(root_fd)

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
