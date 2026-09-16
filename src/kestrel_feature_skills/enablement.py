"""Per-agent skill enablement on core's existing bootstrap configuration table."""

from __future__ import annotations

import uuid
from typing import Any

from .errors import EnablementUnavailableError
from .format import validate_skill_name
from .models import SkillState

CONFIG_PREFIX = "skill:"
STATE_AGENT_PREFIX = "procedural-skill-state:"
DEFAULT_PRIORITY = 100
MIN_PRIORITY = -100_000
MAX_PRIORITY = 100_000


def validate_priority(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("priority must be an integer")  # noqa: TRY004
    priority = value
    if not MIN_PRIORITY <= priority <= MAX_PRIORITY:
        raise ValueError(f"priority must be between {MIN_PRIORITY} and {MAX_PRIORITY}")
    return priority


def _decode_enabled(value: object) -> bool:
    """Accept only database-native boolean values or SQLite's exact 0/1."""

    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return value == 1
    raise ValueError("persisted enabled value must be boolean or integer 0/1")


class SkillEnablementStore:
    """Read and write isolated rows without creating a feature-owned table.

    Core's bootstrap loader treats every enabled row for the literal agent DID
    as a filename to inject in full.  Skill state therefore uses a disjoint
    logical-agent namespace on the same ``bootstrap_config`` substrate.  This
    keeps full procedure bodies behind ``skill_read`` even on core versions
    that do not yet understand the ``skill:`` filename namespace.
    """

    def __init__(self, db: Any | None, agent_id: str):
        self.db = db
        self.agent_id = agent_id

    @property
    def storage_agent_id(self) -> str:
        return f"{STATE_AGENT_PREFIX}{self.agent_id}"

    @property
    def available(self) -> bool:
        return self.db is not None and bool(self.agent_id)

    def _require(self) -> Any:
        if not self.available:
            raise EnablementUnavailableError(
                "skill enablement requires an initialized agent database and identity"
            )
        return self.db

    async def load(self) -> dict[str, SkillState]:
        if not self.available:
            return {}
        rows = await self.db.fetchall(
            """
            SELECT file_name, enabled, priority
            FROM bootstrap_config
            WHERE agent_id = ? AND file_name LIKE ?
            ORDER BY priority ASC, file_name ASC
            """,
            (self.storage_agent_id, f"{CONFIG_PREFIX}%"),
        )
        states: dict[str, SkillState] = {}
        for file_name, enabled, priority in rows:
            if not isinstance(file_name, str) or not file_name.startswith(
                CONFIG_PREFIX
            ):
                continue
            name = file_name[len(CONFIG_PREFIX) :]
            try:
                validate_skill_name(name)
                normalized_enabled = _decode_enabled(enabled)
                normalized_priority = validate_priority(priority)
            except ValueError:
                continue
            states[name] = SkillState(normalized_enabled, normalized_priority)
        return states

    async def set(self, name: str, *, enabled: bool, priority: int) -> SkillState:
        db = self._require()
        name = validate_skill_name(name)
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a boolean")  # noqa: TRY004
        priority = validate_priority(priority)
        file_name = f"{CONFIG_PREFIX}{name}"
        row_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"kestrel-skills:{self.storage_agent_id}:{name}",
            )
        )
        await db.execute(
            """
            INSERT INTO bootstrap_config
                (id, agent_id, file_name, file_path, enabled, priority, max_size_bytes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (agent_id, file_name) DO UPDATE SET
                file_path = excluded.file_path,
                enabled = excluded.enabled,
                priority = excluded.priority,
                max_size_bytes = excluded.max_size_bytes
            """,
            (
                row_id,
                self.storage_agent_id,
                file_name,
                f"skill://{name}",
                int(enabled),
                priority,
                262_144,
            ),
        )
        return SkillState(enabled, priority)

    async def delete(self, name: str) -> None:
        db = self._require()
        name = validate_skill_name(name)
        await db.execute(
            "DELETE FROM bootstrap_config WHERE agent_id = ? AND file_name = ?",
            (self.storage_agent_id, f"{CONFIG_PREFIX}{name}"),
        )


__all__ = [
    "CONFIG_PREFIX",
    "DEFAULT_PRIORITY",
    "MAX_PRIORITY",
    "MIN_PRIORITY",
    "STATE_AGENT_PREFIX",
    "SkillEnablementStore",
    "validate_priority",
]
