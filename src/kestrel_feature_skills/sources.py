"""Ordered skill sources and deterministic precedence resolution."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import secrets
import stat
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

from .errors import SkillError, SkillFormatError, SkillPathError
from .format import (
    GENERATION_FILENAME,
    ValidatedSkillFolder,
    inspect_skill_folder_descriptor,
    validate_generation_payload,
    validate_skill_name,
)
from .git_source import is_full_object_id, validate_ref, validate_remote_url
from .models import (
    CatalogSnapshot,
    DiscoveryError,
    SkillProvenance,
    SkillRecord,
    SkillState,
)

PROVENANCE_FILENAME = ".kestrel-provenance.json"
PROVENANCE_VERSION = 1
INTERNAL_DIRECTORY = ".kestrel-internal"
AGENT_LOCAL_PRECEDENCE = 0
HOST_SHARED_PRECEDENCE = 100
REMOTE_PRECEDENCE = 200
MAX_SOURCE_ENTRIES = 4096
_PROVENANCE_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_PROVENANCE_BOUNDS = {
    "kind": 64,
    "source_id": 2048,
    "locator": 1024,
    "revision": 200,
    "remote_url": 2048,
}
_GENERATION_TEMP_PREFIX = ".kestrel-generation.tmp."
_GENERATION_LOCK_FILENAME = ".generation-marker.lock"
_GENERATION_THREAD_LOCK = threading.RLock()


def _json_safe_text(value: object) -> str:
    """Preserve readable text while escaping filesystem surrogate code points."""

    return str(value).encode("utf-8", errors="backslashreplace").decode("utf-8")


def _new_generation_payload() -> bytes:
    """Return an unguessable, non-reusable local-folder generation witness."""

    return f"kestrel-skill-generation-v1:{secrets.token_hex(32)}\n".encode("ascii")


@contextmanager
def _generation_marker_workspace(root_fd: int) -> Iterator[int]:
    """Pin and lock the private workspace shared by marker writers/reapers."""

    with _GENERATION_THREAD_LOCK:
        created_internal = False
        try:
            os.mkdir(INTERNAL_DIRECTORY, mode=0o700, dir_fd=root_fd)
            created_internal = True
        except FileExistsError:
            pass
        internal_stat = os.stat(
            INTERNAL_DIRECTORY,
            dir_fd=root_fd,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(internal_stat.st_mode):
            raise SkillPathError("skill internal path must be a real directory")
        internal_identity = (internal_stat.st_dev, internal_stat.st_ino)
        internal_fd = os.open(INTERNAL_DIRECTORY, _DIRECTORY_FLAGS, dir_fd=root_fd)
        lock_fd: int | None = None
        locked = False
        try:
            opened_internal = os.fstat(internal_fd)
            if (opened_internal.st_dev, opened_internal.st_ino) != internal_identity:
                raise SkillPathError("skill internal path changed while opening it")
            lock_fd = os.open(
                _GENERATION_LOCK_FILENAME,
                os.O_RDWR
                | os.O_CREAT
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=internal_fd,
            )
            lock_stat = os.fstat(lock_fd)
            if not stat.S_ISREG(lock_stat.st_mode):
                raise SkillPathError("generation marker lock must be a regular file")
            lock_identity = (lock_stat.st_dev, lock_stat.st_ino)
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            locked = True
            lexical_lock = os.stat(
                _GENERATION_LOCK_FILENAME,
                dir_fd=internal_fd,
                follow_symlinks=False,
            )
            if (lexical_lock.st_dev, lexical_lock.st_ino) != lock_identity:
                raise SkillPathError(
                    "generation marker lock changed while acquiring it"
                )
            if created_internal:
                os.fsync(root_fd)
            os.fsync(internal_fd)
            yield internal_fd
        finally:
            if lock_fd is not None:
                try:
                    if locked:
                        fcntl.flock(lock_fd, fcntl.LOCK_UN)
                finally:
                    os.close(lock_fd)
            os.close(internal_fd)


def _read_generation_marker(folder_fd: int) -> bytes:
    """Read and validate the pinned local generation marker."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(GENERATION_FILENAME, flags, dir_fd=folder_fd)
    try:
        value = os.fstat(descriptor)
        if not stat.S_ISREG(value.st_mode):
            raise SkillPathError(f"{GENERATION_FILENAME} must be a regular file")
        payload = os.read(descriptor, 256)
        if os.read(descriptor, 1):
            raise SkillFormatError(f"{GENERATION_FILENAME} exceeds its fixed size")
        validate_generation_payload(payload)
        lexical = os.stat(
            GENERATION_FILENAME,
            dir_fd=folder_fd,
            follow_symlinks=False,
        )
        if (lexical.st_dev, lexical.st_ino) != (value.st_dev, value.st_ino):
            raise SkillPathError(f"{GENERATION_FILENAME} changed during discovery")
        return payload
    finally:
        os.close(descriptor)


def _ensure_local_generation_marker(root_fd: int, folder_fd: int) -> None:
    """Create missing local generation metadata without replacing an owner value."""

    try:
        _read_generation_marker(folder_fd)
        return
    except FileNotFoundError:
        pass

    # Private staging keeps a crashed writer outside the bounded public source.
    # The shared lock also lets startup distinguish orphaned temporaries from a
    # marker publication that is still live in another process.
    with _generation_marker_workspace(root_fd) as artifact_fd:
        try:
            _read_generation_marker(folder_fd)
            return
        except FileNotFoundError:
            pass
        temporary = f"{_GENERATION_TEMP_PREFIX}{secrets.token_hex(16)}"
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        descriptor = os.open(temporary, flags, 0o600, dir_fd=artifact_fd)
        created = os.fstat(descriptor)
        created_identity = (created.st_dev, created.st_ino)
        completed = False
        try:
            payload = _new_generation_payload()
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            completed = True
        finally:
            os.close(descriptor)
            if not completed:
                try:
                    current = os.stat(
                        temporary,
                        dir_fd=artifact_fd,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    pass
                else:
                    if (current.st_dev, current.st_ino) == created_identity:
                        os.unlink(temporary, dir_fd=artifact_fd)
        try:
            try:
                os.link(
                    temporary,
                    GENERATION_FILENAME,
                    src_dir_fd=artifact_fd,
                    dst_dir_fd=folder_fd,
                    follow_symlinks=False,
                )
            except FileExistsError:
                # Another feature instance won publication. Its completed
                # hardlink is the sole generation owner for this folder.
                _read_generation_marker(folder_fd)
            else:
                os.fsync(folder_fd)
                _read_generation_marker(folder_fd)
        finally:
            try:
                current = os.stat(
                    temporary,
                    dir_fd=artifact_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                if (current.st_dev, current.st_ino) == created_identity:
                    os.unlink(temporary, dir_fd=artifact_fd)
            os.fsync(artifact_fd)
        os.fsync(folder_fd)


def _folder_revision(
    snapshot: ValidatedSkillFolder,
    folder_identity: tuple[int, int],
) -> str:
    """Bind an opaque revision to one folder generation and all validated bytes."""

    digest = hashlib.sha256(b"kestrel-skill-folder-revision-v1\0")
    for value in folder_identity:
        encoded = str(value).encode("ascii")
        digest.update(len(encoded).to_bytes(2, "big"))
        digest.update(encoded)
    for entry in sorted(snapshot.entries, key=lambda item: item.path):
        path = entry.path.encode("utf-8")
        digest.update(len(path).to_bytes(4, "big"))
        digest.update(path)
        if entry.payload is None:
            digest.update(b"D")
            continue
        digest.update(b"F")
        digest.update(len(entry.payload).to_bytes(8, "big"))
        digest.update(entry.payload)
    return digest.hexdigest()


def _configured_revision(content_revision: str, state: SkillState) -> str:
    """Include mutable enablement state in the operator-visible revision."""

    digest = hashlib.sha256(b"kestrel-skill-record-revision-v1\0")
    digest.update(content_revision.encode("ascii"))
    digest.update(b"\0enabled=" + (b"1" if state.enabled else b"0"))
    digest.update(b"\0priority=" + str(state.priority).encode("ascii"))
    return digest.hexdigest()


class SkillSource(ABC):
    """A deterministic provider of validated skill folders."""

    source_id: str
    kind: str
    precedence: int

    @abstractmethod
    def discover(self) -> tuple[tuple[SkillRecord, ...], tuple[DiscoveryError, ...]]:
        """Return valid records and visible rejections."""


_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)


def _bounded_source_names(root_fd: int) -> list[str]:
    """Enumerate one source root without materializing unbounded attacker work."""

    names: list[str] = []
    with os.scandir(root_fd) as entries:
        for entry in entries:
            if len(names) >= MAX_SOURCE_ENTRIES:
                raise SkillFormatError(
                    f"skill source exceeds source entry limit {MAX_SOURCE_ENTRIES}"
                )
            names.append(entry.name)
    names.sort()
    return names


def _default_provenance(
    source: DirectorySkillSource, folder_name: str
) -> SkillProvenance:
    return SkillProvenance(
        kind=source.kind,
        source_id=source.source_id,
        locator=folder_name,
    )


def _provenance_text(
    value: object, *, field: str, optional: bool = False
) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value:
        suffix = " or null" if optional else ""
        raise SkillFormatError(
            f"{PROVENANCE_FILENAME}.{field} must be a non-empty string{suffix}"
        )
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise SkillFormatError(
            f"{PROVENANCE_FILENAME}.{field} must be valid UTF-8 text"
        ) from exc
    if len(encoded) > _PROVENANCE_BOUNDS[field]:
        raise SkillFormatError(
            f"{PROVENANCE_FILENAME}.{field} exceeds "
            f"{_PROVENANCE_BOUNDS[field]} UTF-8 bytes"
        )
    if _PROVENANCE_CONTROL_RE.search(value):
        raise SkillFormatError(
            f"{PROVENANCE_FILENAME}.{field} contains control characters"
        )
    return value


def _validated_provenance(provenance: SkillProvenance) -> SkillProvenance:
    validated = SkillProvenance(
        kind=_provenance_text(provenance.kind, field="kind"),
        source_id=_provenance_text(provenance.source_id, field="source_id"),
        locator=_provenance_text(provenance.locator, field="locator"),
        revision=_provenance_text(provenance.revision, field="revision", optional=True),
        remote_url=_provenance_text(
            provenance.remote_url, field="remote_url", optional=True
        ),
    )
    if validated.kind != "git":
        return validated
    if not is_full_object_id(validated.revision):
        raise SkillFormatError(
            "Git provenance revision must be a full lowercase commit hash"
        )
    if validated.remote_url is None:
        raise SkillFormatError("Git provenance remote_url must be present")
    try:
        remote_url = validate_remote_url(validated.remote_url)
    except SkillError as exc:
        raise SkillFormatError("Git provenance remote_url is invalid") from exc
    if validated.source_id != remote_url:
        raise SkillFormatError(
            "Git provenance source_id must match its validated remote_url"
        )
    ref, separator, skill_name = validated.locator.rpartition(":")
    if not separator:
        raise SkillFormatError("Git provenance locator must identify ref:skill-name")
    try:
        validate_ref(ref)
        validate_skill_name(skill_name)
    except SkillError as exc:
        raise SkillFormatError(
            "Git provenance locator must identify a valid ref and skill name"
        ) from exc
    return validated


def _parse_provenance_payload(
    source: DirectorySkillSource,
    folder_name: str,
    payload_bytes: bytes,
) -> SkillProvenance:
    """Parse provenance from the same captured bytes used by record revision."""

    if len(payload_bytes) > 16_384:
        raise SkillFormatError(f"{PROVENANCE_FILENAME} exceeds 16384 bytes")
    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise SkillFormatError(f"invalid {PROVENANCE_FILENAME}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("version") != PROVENANCE_VERSION:
        raise SkillFormatError(f"{PROVENANCE_FILENAME} has an unsupported version")
    allowed = {"version", "kind", "source_id", "locator", "revision", "remote_url"}
    unknown = set(payload) - allowed
    if unknown:
        raise SkillFormatError(
            f"{PROVENANCE_FILENAME} has unsupported field(s): {', '.join(sorted(unknown))}"
        )
    provenance = _validated_provenance(
        SkillProvenance(
            kind=payload.get("kind"),
            source_id=payload.get("source_id"),
            locator=payload.get("locator"),
            revision=payload.get("revision"),
            remote_url=payload.get("remote_url"),
        )
    )
    if (
        provenance.kind == "git"
        and provenance.locator.rpartition(":")[2] != folder_name
    ):
        raise SkillFormatError(
            "Git provenance locator skill name must match its containing folder"
        )
    return provenance


def serialize_provenance(provenance: SkillProvenance) -> bytes:
    provenance = _validated_provenance(provenance)
    payload = {"version": PROVENANCE_VERSION, **provenance.to_dict()}
    return (
        json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")


class DirectorySkillSource(SkillSource):
    """An immediate-child folder source with explicit precedence."""

    def __init__(
        self,
        *,
        root: Path,
        source_id: str,
        kind: str,
        precedence: int,
        expected_root_identity: tuple[int, int] | None = None,
    ):
        self.root = root
        self.source_id = source_id
        self.kind = kind
        self.precedence = precedence
        self.expected_root_identity = expected_root_identity

    def discover(self) -> tuple[tuple[SkillRecord, ...], tuple[DiscoveryError, ...]]:
        try:
            root_before = self.root.lstat()
        except FileNotFoundError as exc:
            if self.expected_root_identity is None:
                return (), ()
            error = DiscoveryError(
                source_id=_json_safe_text(self.source_id),
                locator=_json_safe_text(self.root),
                error=_json_safe_text(
                    f"could not resolve skill source root: pinned root disappeared: {exc}"
                ),
            )
            return (), (error,)
        except OSError as exc:
            error = DiscoveryError(
                source_id=_json_safe_text(self.source_id),
                locator=_json_safe_text(self.root),
                error=_json_safe_text(f"could not resolve skill source root: {exc}"),
            )
            return (), (error,)
        root_fd: int | None = None
        try:
            observed_identity = (root_before.st_dev, root_before.st_ino)
            root_identity = self.expected_root_identity or observed_identity
            if observed_identity != root_identity:
                raise SkillPathError("skill source root changed after configuration")
            if stat.S_ISLNK(root_before.st_mode) or not stat.S_ISDIR(
                root_before.st_mode
            ):
                raise SkillPathError(
                    "skill source root must be a real directory, not a symlink"
                )
            root_resolved = self.root.resolve(strict=True)
            root_fd = os.open(self.root, _DIRECTORY_FLAGS)
            opened_root = os.fstat(root_fd)
            root_after = self.root.lstat()
            if (
                stat.S_ISLNK(root_after.st_mode)
                or not stat.S_ISDIR(root_after.st_mode)
                or (root_after.st_dev, root_after.st_ino) != root_identity
                or (opened_root.st_dev, opened_root.st_ino) != root_identity
            ):
                raise SkillPathError("skill source root changed during discovery")
        except (SkillError, OSError, RuntimeError) as exc:
            if root_fd is not None:
                os.close(root_fd)
            error = DiscoveryError(
                source_id=_json_safe_text(self.source_id),
                locator=_json_safe_text(self.root),
                error=_json_safe_text(f"could not resolve skill source root: {exc}"),
            )
            return (), (error,)
        records: list[SkillRecord] = []
        errors: list[DiscoveryError] = []
        try:
            candidate_names = _bounded_source_names(root_fd)
        except (SkillError, OSError) as exc:
            os.close(root_fd)
            return (), (
                DiscoveryError(
                    source_id=_json_safe_text(self.source_id),
                    locator=_json_safe_text(self.root),
                    error=_json_safe_text(f"could not enumerate source: {exc}"),
                ),
            )
        try:
            for folder_name in candidate_names:
                if folder_name.startswith("."):
                    continue
                folder_fd: int | None = None
                try:
                    before = os.stat(
                        folder_name,
                        dir_fd=root_fd,
                        follow_symlinks=False,
                    )
                    if not stat.S_ISDIR(before.st_mode) and not stat.S_ISLNK(
                        before.st_mode
                    ):
                        continue
                    before_identity = (before.st_dev, before.st_ino)
                    folder_fd = os.open(folder_name, _DIRECTORY_FLAGS, dir_fd=root_fd)
                    opened = os.fstat(folder_fd)
                    folder_identity = (opened.st_dev, opened.st_ino)
                    if (
                        not stat.S_ISDIR(opened.st_mode)
                        or folder_identity != before_identity
                    ):
                        raise SkillPathError(
                            "skill folder changed identity during discovery"
                        )
                    if self.kind == "agent-local":
                        _ensure_local_generation_marker(root_fd, folder_fd)
                    snapshot = inspect_skill_folder_descriptor(
                        folder_fd,
                        folder_name=folder_name,
                    )
                    document = snapshot.document
                    provenance_entry = next(
                        (
                            entry
                            for entry in snapshot.entries
                            if entry.path == PROVENANCE_FILENAME
                        ),
                        None,
                    )
                    if provenance_entry is None:
                        provenance = _default_provenance(self, folder_name)
                    elif provenance_entry.payload is None:
                        raise SkillFormatError(
                            f"{PROVENANCE_FILENAME} must be a regular file"
                        )
                    else:
                        provenance = _parse_provenance_payload(
                            self,
                            folder_name,
                            provenance_entry.payload,
                        )
                    lexical_after = os.stat(
                        folder_name,
                        dir_fd=root_fd,
                        follow_symlinks=False,
                    )
                    root_after = self.root.lstat()
                    if (
                        stat.S_ISLNK(lexical_after.st_mode)
                        or (lexical_after.st_dev, lexical_after.st_ino)
                        != folder_identity
                        or stat.S_ISLNK(root_after.st_mode)
                        or (root_after.st_dev, root_after.st_ino) != root_identity
                    ):
                        raise SkillPathError(
                            "skill source or folder changed identity during discovery"
                        )
                    resolved_folder = root_resolved / folder_name
                except (SkillError, OSError) as exc:
                    errors.append(
                        DiscoveryError(
                            source_id=_json_safe_text(self.source_id),
                            locator=_json_safe_text(folder_name),
                            error=_json_safe_text(exc),
                        )
                    )
                    continue
                finally:
                    if folder_fd is not None:
                        os.close(folder_fd)
                content_revision = _folder_revision(snapshot, folder_identity)
                records.append(
                    SkillRecord(
                        document=document,
                        folder=resolved_folder,
                        source_id=(
                            provenance.source_id
                            if provenance.kind == "git"
                            else self.source_id
                        ),
                        source_kind=self.kind,
                        precedence=(
                            REMOTE_PRECEDENCE
                            if provenance.kind == "git" and self.kind == "agent-local"
                            else self.precedence
                        ),
                        provenance=provenance,
                        folder_identity=folder_identity,
                        content_revision=content_revision,
                        revision=content_revision,
                    )
                )
        finally:
            os.close(root_fd)
        return tuple(records), tuple(errors)


class SkillCatalog:
    """Resolve an ordered source set into one record per skill name."""

    def __init__(self, sources: tuple[SkillSource, ...]):
        identities = [source.source_id for source in sources]
        if len(set(identities)) != len(identities):
            raise ValueError("skill source_id values must be unique")
        self.sources = tuple(
            sorted(sources, key=lambda source: (source.precedence, source.source_id))
        )

    def refresh(
        self, states: Mapping[str, SkillState] | None = None
    ) -> CatalogSnapshot:
        states = states or {}
        candidates: list[SkillRecord] = []
        errors: list[DiscoveryError] = []
        shadowed: dict[str, list[SkillProvenance]] = {}
        shadowed_records: dict[str, list[SkillRecord]] = {}
        for source in self.sources:
            records, source_errors = source.discover()
            errors.extend(source_errors)
            candidates.extend(records)
        resolved: dict[str, SkillRecord] = {}
        for record in sorted(
            candidates,
            key=lambda item: (item.precedence, item.name, item.source_id),
        ):
            state = states.get(record.name, SkillState())
            content_revision = record.content_revision or record.revision
            configured = replace(
                record,
                state=state,
                content_revision=content_revision,
                revision=_configured_revision(content_revision, state),
            )
            if record.name not in resolved:
                resolved[record.name] = configured
            else:
                shadowed.setdefault(record.name, []).append(record.provenance)
                shadowed_records.setdefault(record.name, []).append(configured)
        ordered = tuple(
            sorted(
                resolved.values(),
                key=lambda item: (item.state.priority, item.name, item.source_id),
            )
        )
        immutable_shadowed = MappingProxyType(
            {name: tuple(values) for name, values in sorted(shadowed.items())}
        )
        immutable_shadowed_records = MappingProxyType(
            {name: tuple(values) for name, values in sorted(shadowed_records.items())}
        )
        return CatalogSnapshot(
            records=ordered,
            errors=tuple(errors),
            shadowed=immutable_shadowed,
            shadowed_records=immutable_shadowed_records,
        )


__all__ = [
    "AGENT_LOCAL_PRECEDENCE",
    "HOST_SHARED_PRECEDENCE",
    "INTERNAL_DIRECTORY",
    "MAX_SOURCE_ENTRIES",
    "PROVENANCE_FILENAME",
    "REMOTE_PRECEDENCE",
    "DirectorySkillSource",
    "SkillCatalog",
    "SkillSource",
    "serialize_provenance",
]
