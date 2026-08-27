"""Ordered skill sources and deterministic precedence resolution."""

from __future__ import annotations

import json
import os
import re
import stat
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

from .errors import SkillError, SkillFormatError, SkillPathError
from .format import validate_skill_folder_descriptor
from .models import (
    CatalogSnapshot,
    DiscoveryError,
    SkillProvenance,
    SkillRecord,
    SkillState,
)

PROVENANCE_FILENAME = ".kestrel-provenance.json"
PROVENANCE_VERSION = 1
AGENT_LOCAL_PRECEDENCE = 0
HOST_SHARED_PRECEDENCE = 100
REMOTE_PRECEDENCE = 200
_PROVENANCE_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_PROVENANCE_BOUNDS = {
    "kind": 64,
    "source_id": 2048,
    "locator": 1024,
    "revision": 200,
    "remote_url": 2048,
}


def _json_safe_text(value: object) -> str:
    """Preserve readable text while escaping filesystem surrogate code points."""

    return str(value).encode("utf-8", errors="backslashreplace").decode("utf-8")


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
_READ_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)


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
    return SkillProvenance(
        kind=_provenance_text(provenance.kind, field="kind"),
        source_id=_provenance_text(provenance.source_id, field="source_id"),
        locator=_provenance_text(provenance.locator, field="locator"),
        revision=_provenance_text(provenance.revision, field="revision", optional=True),
        remote_url=_provenance_text(
            provenance.remote_url, field="remote_url", optional=True
        ),
    )


def _load_provenance_at(
    source: DirectorySkillSource,
    folder_fd: int,
    folder_name: str,
) -> SkillProvenance:
    try:
        before = os.stat(
            PROVENANCE_FILENAME,
            dir_fd=folder_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return _default_provenance(source, folder_name)
    except OSError as exc:
        raise SkillFormatError(f"could not inspect {PROVENANCE_FILENAME}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise SkillFormatError(f"{PROVENANCE_FILENAME} must be a regular file")
    if before.st_size > 16_384:
        raise SkillFormatError(f"{PROVENANCE_FILENAME} exceeds 16384 bytes")
    try:
        descriptor = os.open(PROVENANCE_FILENAME, _READ_FLAGS, dir_fd=folder_fd)
    except OSError as exc:
        raise SkillFormatError(f"could not read {PROVENANCE_FILENAME}") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
            before.st_dev,
            before.st_ino,
        ):
            raise SkillFormatError(f"{PROVENANCE_FILENAME} changed during discovery")
        if opened.st_size > 16_384:
            raise SkillFormatError(f"{PROVENANCE_FILENAME} exceeds 16384 bytes")
        payload_bytes = bytearray()
        while len(payload_bytes) <= 16_384:
            chunk = os.read(descriptor, min(16_385 - len(payload_bytes), 16_384))
            if not chunk:
                break
            payload_bytes.extend(chunk)
        if len(payload_bytes) > 16_384:
            raise SkillFormatError(f"{PROVENANCE_FILENAME} exceeds 16384 bytes")
        after = os.fstat(descriptor)
        if (after.st_dev, after.st_ino) != (opened.st_dev, opened.st_ino):
            raise SkillFormatError(f"{PROVENANCE_FILENAME} changed during discovery")
        payload = json.loads(bytes(payload_bytes).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SkillFormatError(f"invalid {PROVENANCE_FILENAME}: {exc}") from exc
    finally:
        os.close(descriptor)
    if not isinstance(payload, dict) or payload.get("version") != PROVENANCE_VERSION:
        raise SkillFormatError(f"{PROVENANCE_FILENAME} has an unsupported version")
    allowed = {"version", "kind", "source_id", "locator", "revision", "remote_url"}
    unknown = set(payload) - allowed
    if unknown:
        raise SkillFormatError(
            f"{PROVENANCE_FILENAME} has unsupported field(s): {', '.join(sorted(unknown))}"
        )
    return _validated_provenance(
        SkillProvenance(
            kind=payload.get("kind"),
            source_id=payload.get("source_id"),
            locator=payload.get("locator"),
            revision=payload.get("revision"),
            remote_url=payload.get("remote_url"),
        )
    )


def serialize_provenance(provenance: SkillProvenance) -> bytes:
    provenance = _validated_provenance(provenance)
    payload = {"version": PROVENANCE_VERSION, **provenance.to_dict()}
    return (
        json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")


class DirectorySkillSource(SkillSource):
    """An immediate-child folder source with explicit precedence."""

    def __init__(self, *, root: Path, source_id: str, kind: str, precedence: int):
        self.root = root
        self.source_id = source_id
        self.kind = kind
        self.precedence = precedence

    def discover(self) -> tuple[tuple[SkillRecord, ...], tuple[DiscoveryError, ...]]:
        if not self.root.exists():
            return (), ()
        root_fd: int | None = None
        try:
            root_before = self.root.lstat()
            root_identity = (root_before.st_dev, root_before.st_ino)
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
        except (SkillError, OSError) as exc:
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
            candidate_names = sorted(os.listdir(root_fd))
        except OSError as exc:
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
                    document = validate_skill_folder_descriptor(
                        folder_fd,
                        folder_name=folder_name,
                    )
                    provenance = _load_provenance_at(
                        self,
                        folder_fd,
                        folder_name,
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
                            if provenance.kind == "git"
                            else self.precedence
                        ),
                        provenance=provenance,
                        folder_identity=folder_identity,
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
            configured = replace(record, state=state)
            if record.name not in resolved:
                resolved[record.name] = configured
            else:
                shadowed.setdefault(record.name, []).append(record.provenance)
        ordered = tuple(
            sorted(
                resolved.values(),
                key=lambda item: (item.state.priority, item.name, item.source_id),
            )
        )
        immutable_shadowed = MappingProxyType(
            {name: tuple(values) for name, values in sorted(shadowed.items())}
        )
        return CatalogSnapshot(
            records=ordered,
            errors=tuple(errors),
            shadowed=immutable_shadowed,
        )


__all__ = [
    "AGENT_LOCAL_PRECEDENCE",
    "HOST_SHARED_PRECEDENCE",
    "PROVENANCE_FILENAME",
    "REMOTE_PRECEDENCE",
    "DirectorySkillSource",
    "SkillCatalog",
    "SkillSource",
    "serialize_provenance",
]
