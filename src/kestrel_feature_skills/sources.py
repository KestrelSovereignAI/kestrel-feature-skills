"""Ordered skill sources and deterministic precedence resolution."""

from __future__ import annotations

import json
import re
import stat
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

from .errors import SkillError, SkillFormatError, SkillPathError
from .format import validate_skill_folder
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


def _default_provenance(source: DirectorySkillSource, folder: Path) -> SkillProvenance:
    return SkillProvenance(
        kind=source.kind,
        source_id=source.source_id,
        locator=folder.name,
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


def _load_provenance(source: DirectorySkillSource, folder: Path) -> SkillProvenance:
    metadata = folder / PROVENANCE_FILENAME
    if not metadata.exists():
        return _default_provenance(source, folder)
    if metadata.is_symlink() or not metadata.is_file():
        raise SkillFormatError(f"{PROVENANCE_FILENAME} must be a regular file")
    if metadata.stat().st_size > 16_384:
        raise SkillFormatError(f"{PROVENANCE_FILENAME} exceeds 16384 bytes")
    try:
        payload = json.loads(metadata.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SkillFormatError(f"invalid {PROVENANCE_FILENAME}: {exc}") from exc
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
            root_after = self.root.lstat()
            if (
                stat.S_ISLNK(root_after.st_mode)
                or not stat.S_ISDIR(root_after.st_mode)
                or (root_after.st_dev, root_after.st_ino) != root_identity
            ):
                raise SkillPathError("skill source root changed during discovery")
        except (SkillError, OSError) as exc:
            error = DiscoveryError(
                source_id=_json_safe_text(self.source_id),
                locator=_json_safe_text(self.root),
                error=_json_safe_text(f"could not resolve skill source root: {exc}"),
            )
            return (), (error,)
        records: list[SkillRecord] = []
        errors: list[DiscoveryError] = []
        try:
            candidates = sorted(self.root.iterdir(), key=lambda path: path.name)
        except OSError as exc:
            return (), (
                DiscoveryError(
                    source_id=_json_safe_text(self.source_id),
                    locator=_json_safe_text(self.root),
                    error=_json_safe_text(f"could not enumerate source: {exc}"),
                ),
            )
        for folder in candidates:
            if folder.name.startswith("."):
                continue
            if not folder.is_dir() and not folder.is_symlink():
                continue
            try:
                before = folder.lstat()
                before_identity = (before.st_dev, before.st_ino)
                document = validate_skill_folder(folder, source_root=self.root)
                provenance = _load_provenance(self, folder)
                resolved_folder = folder.resolve(strict=True)
                lexical_after = folder.lstat()
                lexical_identity = (lexical_after.st_dev, lexical_after.st_ino)
                if (
                    stat.S_ISLNK(lexical_after.st_mode)
                    or lexical_identity != before_identity
                ):
                    raise SkillPathError(
                        "skill folder changed identity during discovery"
                    )
                try:
                    resolved_folder.relative_to(root_resolved)
                except ValueError as exc:
                    raise SkillPathError(
                        "skill folder escaped its configured source during discovery"
                    ) from exc
                after = resolved_folder.lstat()
                folder_identity = (after.st_dev, after.st_ino)
                if folder_identity != before_identity:
                    raise SkillFormatError(
                        "skill folder changed identity during discovery"
                    )
            except (SkillError, OSError) as exc:
                errors.append(
                    DiscoveryError(
                        source_id=_json_safe_text(self.source_id),
                        locator=_json_safe_text(folder.name),
                        error=_json_safe_text(exc),
                    )
                )
                continue
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
