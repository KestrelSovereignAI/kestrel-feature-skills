"""Ordered skill sources and deterministic precedence resolution."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

from .errors import SkillError, SkillFormatError
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
    required = ("kind", "source_id", "locator")
    if any(
        not isinstance(payload.get(key), str) or not payload[key] for key in required
    ):
        raise SkillFormatError(
            f"{PROVENANCE_FILENAME} requires non-empty origin strings"
        )
    for optional in ("revision", "remote_url"):
        if payload.get(optional) is not None and not isinstance(payload[optional], str):
            raise SkillFormatError(
                f"{PROVENANCE_FILENAME}.{optional} must be a string or null"
            )
    return SkillProvenance(
        kind=payload["kind"],
        source_id=payload["source_id"],
        locator=payload["locator"],
        revision=payload.get("revision"),
        remote_url=payload.get("remote_url"),
    )


def serialize_provenance(provenance: SkillProvenance) -> bytes:
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
        if self.root.is_symlink() or not self.root.is_dir():
            error = DiscoveryError(
                source_id=self.source_id,
                locator=str(self.root),
                error="skill source root must be a real directory, not a symlink",
            )
            return (), (error,)
        records: list[SkillRecord] = []
        errors: list[DiscoveryError] = []
        try:
            candidates = sorted(self.root.iterdir(), key=lambda path: path.name)
        except OSError as exc:
            return (), (
                DiscoveryError(
                    source_id=self.source_id,
                    locator=str(self.root),
                    error=f"could not enumerate source: {exc}",
                ),
            )
        for folder in candidates:
            if folder.name.startswith("."):
                continue
            if not folder.is_dir() and not folder.is_symlink():
                continue
            try:
                document = validate_skill_folder(folder, source_root=self.root)
                provenance = _load_provenance(self, folder)
            except (SkillError, OSError) as exc:
                errors.append(
                    DiscoveryError(
                        source_id=self.source_id,
                        locator=folder.name,
                        error=str(exc),
                    )
                )
                continue
            records.append(
                SkillRecord(
                    document=document,
                    folder=folder.resolve(),
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
