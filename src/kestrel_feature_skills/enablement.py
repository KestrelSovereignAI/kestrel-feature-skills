"""Per-agent skill enablement on core's existing bootstrap configuration table."""

from __future__ import annotations

import uuid
from typing import Any

from .errors import EnablementUnavailableError
from .format import validate_skill_name
from .models import SkillState

CONFIG_PREFIX = "skill:"
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


class SkillEnablementStore:
    """Read and write namespaced rows without creating a feature-owned table."""

    def __init__(self, db: Any | None, agent_id: str):
        self.db = db
        self.agent_id = agent_id

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
            (self.agent_id, f"{CONFIG_PREFIX}%"),
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
                normalized_priority = validate_priority(priority)
            except ValueError:
                continue
            states[name] = SkillState(bool(enabled), normalized_priority)
        return states

    async def set(self, name: str, *, enabled: bool, priority: int) -> SkillState:
        db = self._require()
        name = validate_skill_name(name)
        priority = validate_priority(priority)
        file_name = f"{CONFIG_PREFIX}{name}"
        row_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"kestrel-skills:{self.agent_id}:{name}")
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
                self.agent_id,
                file_name,
                f"skill://{name}",
                int(bool(enabled)),
                priority,
                262_144,
            ),
        )
        return SkillState(bool(enabled), priority)

    async def delete(self, name: str) -> None:
        db = self._require()
        name = validate_skill_name(name)
        await db.execute(
            "DELETE FROM bootstrap_config WHERE agent_id = ? AND file_name = ?",
            (self.agent_id, f"{CONFIG_PREFIX}{name}"),
        )


__all__ = [
    "CONFIG_PREFIX",
    "DEFAULT_PRIORITY",
    "MAX_PRIORITY",
    "MIN_PRIORITY",
    "SkillEnablementStore",
    "validate_priority",
]
