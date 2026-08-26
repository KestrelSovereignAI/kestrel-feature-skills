"""Data-only models for skills, provenance, and discovery."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SkillDocument:
    """The canonical contents of one ``SKILL.md``."""

    name: str
    description: str
    body: str


@dataclass(frozen=True, slots=True)
class SkillProvenance:
    """Where one resolved skill came from."""

    kind: str
    source_id: str
    locator: str
    revision: str | None = None
    remote_url: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "kind": self.kind,
            "source_id": self.source_id,
            "locator": self.locator,
            "revision": self.revision,
            "remote_url": self.remote_url,
        }


@dataclass(frozen=True, slots=True)
class SkillState:
    """Per-agent enablement state stored on the bootstrap substrate."""

    enabled: bool = False
    priority: int = 100


@dataclass(frozen=True, slots=True)
class SkillRecord:
    """One validated skill as resolved from a source."""

    document: SkillDocument
    folder: Path
    source_id: str
    source_kind: str
    precedence: int
    provenance: SkillProvenance
    state: SkillState = SkillState()

    @property
    def name(self) -> str:
        return self.document.name

    @property
    def editable(self) -> bool:
        return self.source_kind == "agent-local"


@dataclass(frozen=True, slots=True)
class DiscoveryError:
    """A malformed candidate that was rejected rather than half-loaded."""

    source_id: str
    locator: str
    error: str

    def to_dict(self) -> dict[str, str]:
        return {
            "source_id": self.source_id,
            "locator": self.locator,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class CatalogSnapshot:
    """An immutable, deterministic resolution of every configured source."""

    records: tuple[SkillRecord, ...] = ()
    errors: tuple[DiscoveryError, ...] = ()
    shadowed: Mapping[str, tuple[SkillProvenance, ...]] = field(default_factory=dict)

    def by_name(self) -> dict[str, SkillRecord]:
        return {record.name: record for record in self.records}


@dataclass(frozen=True, slots=True)
class ContextRender:
    """Deterministic prompt text plus its accounting evidence."""

    text: str
    included: tuple[str, ...]
    dropped: tuple[str, ...]
    token_costs: Mapping[str, int]


__all__ = [
    "CatalogSnapshot",
    "ContextRender",
    "DiscoveryError",
    "SkillDocument",
    "SkillProvenance",
    "SkillRecord",
    "SkillState",
]
