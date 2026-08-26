"""Kestrel feature entry point for folder-shaped procedural skills."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from kestrel_sdk.features.base import Feature, tool
from kestrel_sdk.features.contributions import (
    FeaturePermissionDefaults,
    PermissionLevel,
)
from kestrel_sdk.features.ui import UIContributions
from kestrel_sdk.storage.database import DatabaseError
from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult
from kestrel_sovereign.features.storage_access import (
    hides_persisted_user_content,
    resolve_feature_database,
)
from kestrel_sovereign.storage.async_graph_store import GraphNode
from kestrel_sovereign.storage.privacy_wrapper import optional_transition_lock

from .context import (
    DEFAULT_CONTEXT_BUDGET_BYTES,
    estimate_skill_token_cost,
    render_context_clause,
)
from .enablement import DEFAULT_PRIORITY, SkillEnablementStore, validate_priority
from .errors import SkillConflictError, SkillError, SkillPrivacyError
from .format import SKILL_FILENAME, validate_skill_name
from .git_source import GitSkillSource
from .models import (
    CatalogSnapshot,
    SkillDocument,
    SkillProvenance,
    SkillRecord,
    SkillState,
)
from .sources import (
    AGENT_LOCAL_PRECEDENCE,
    HOST_SHARED_PRECEDENCE,
    DirectorySkillSource,
    SkillCatalog,
)
from .store import SkillStore

logger = logging.getLogger(__name__)

PROCEDURAL_SKILL_NODE_TYPE = "procedural_skill"
SKILLS_CAPABILITY = "procedural-skills"
_STATIC_DIR = Path(__file__).resolve().parent / "static"


def _agent_id(agent: object) -> str:
    value = getattr(agent, "did", None) or getattr(agent, "agent_id", None)
    return value if isinstance(value, str) else ""


def _agent_local_root(agent: object) -> Path:
    explicit = getattr(agent, "procedural_skills_root", None)
    if explicit:
        return Path(explicit)
    for attribute in ("bootstrap_service", "context_builder"):
        service = getattr(agent, attribute, None)
        base = (
            getattr(service, "agent_data_path", None) if service is not None else None
        )
        if base:
            return Path(base) / "skills"
    storage_path = getattr(agent, "storage_path", None)
    if storage_path:
        return Path(storage_path).parent / "skills"
    raise RuntimeError(
        "ProceduralSkillsFeature requires an agent data path or procedural_skills_root"
    )


def _host_shared_root() -> Path | None:
    explicit = os.environ.get("KESTREL_SHARED_SKILLS_DIR")
    if explicit:
        return Path(explicit)
    kestrel_home = os.environ.get("KESTREL_HOME")
    return Path(kestrel_home) / "skills" if kestrel_home else None


def _privacy_transition_lock(agent: object):
    getter = getattr(agent, "_get_privacy_transition_lock", None)
    if callable(getter):
        try:
            candidate = getter()
        except Exception:  # noqa: BLE001 - missing lock must not break a feature
            candidate = None
        if hasattr(candidate, "__aenter__"):
            return candidate
    candidate = getattr(agent, "_privacy_transition_lock", None)
    return candidate if hasattr(candidate, "__aenter__") else None


class ProceduralSkillsFeature(Feature):
    """Author, discover, enable, and progressively disclose procedural skills."""

    def __init__(self, agent: object):
        super().__init__(agent)
        self._db: Any | None = None
        self._enablement: SkillEnablementStore | None = None
        self._store: SkillStore | None = None
        self._catalog: SkillCatalog | None = None
        self._snapshot = CatalogSnapshot()
        self._context_render = render_context_clause(self._snapshot)
        self._states: dict[str, SkillState] = {}
        self._enablement_error: str | None = None
        self._indexed_names: frozenset[str] = frozenset()
        self._router = None

    @property
    def tool_description(self) -> str:
        return (
            "Manage folder-shaped procedural skills with descriptions in context and "
            "full procedures disclosed only through skill_read"
        )

    async def initialize(self) -> None:
        from .api import build_router

        self._router = build_router(self)
        async with optional_transition_lock(_privacy_transition_lock(self.agent)):
            if self._privacy_hidden():
                self._hide_persistent_state()
            else:
                self._ensure_persistent_services()
                await self._refresh_locked()
        logger.info(
            "ProceduralSkillsFeature initialized (persistent_access=%s, skills=%d, errors=%d)",
            not self._privacy_hidden(),
            len(self._snapshot.records),
            len(self._snapshot.errors),
        )

    async def shutdown(self) -> None:
        self._router = None

    def get_router(self):
        return self._router

    def get_ui_contributions(self) -> UIContributions:
        return UIContributions(
            static_dir=str(_STATIC_DIR),
            modules=["skills.js"],
            css=["skills.css"],
            capability=SKILLS_CAPABILITY,
        )

    def get_feature_permission_defaults(self) -> FeaturePermissionDefaults:
        return FeaturePermissionDefaults(
            feature_default=PermissionLevel.ASK,
            tool_overrides={
                "skill_list": PermissionLevel.ALLOW,
                "skill_read": PermissionLevel.ALLOW,
                "skill_search": PermissionLevel.ALLOW,
                "skill_create": PermissionLevel.ASK,
                "skill_edit": PermissionLevel.ASK,
                "skill_enable": PermissionLevel.ASK,
                "skill_disable": PermissionLevel.ASK,
                "skill_delete": PermissionLevel.ALWAYS_ASK,
                "skill_install": PermissionLevel.ALWAYS_ASK,
            },
        )

    @property
    def snapshot(self) -> CatalogSnapshot:
        return CatalogSnapshot() if self._privacy_hidden() else self._snapshot

    @property
    def context_clause_text(self) -> str:
        """The memoized, byte-stable text awaiting the SDK contribution seam."""

        return "" if self._privacy_hidden() else self._context_render.text

    @property
    def enablement_available(self) -> bool:
        return bool(
            not self._privacy_hidden()
            and self._enablement
            and self._enablement.available
        )

    async def refresh(self) -> CatalogSnapshot:
        async with optional_transition_lock(_privacy_transition_lock(self.agent)):
            if self._privacy_hidden():
                self._hide_persistent_state()
                return self._snapshot
            self._ensure_persistent_services()
            return await self._refresh_locked()

    async def _refresh_locked(self) -> CatalogSnapshot:
        if self._catalog is None or self._enablement is None:
            raise RuntimeError("ProceduralSkillsFeature is not initialized")
        try:
            states = await self._enablement.load()
        except DatabaseError as exc:
            self._enablement_error = str(exc)
            states = self._states
            logger.warning(
                "Could not reload procedural skill enablement; retaining the last known state: %s",
                exc,
            )
        else:
            self._states = dict(states)
            self._enablement_error = None
        self._snapshot = self._catalog.refresh(states)
        self._context_render = render_context_clause(
            self._snapshot,
            max_bytes=DEFAULT_CONTEXT_BUDGET_BYTES,
        )
        current_names = {record.name for record in self._snapshot.records}
        indexed = set(self._indexed_names)
        indexed.update(await self._persisted_index_names())
        for stale_name in sorted(indexed - current_names):
            if await self._delete_index_node(stale_name):
                indexed.discard(stale_name)
        for record in self._snapshot.records:
            if await self._index_record(record):
                indexed.add(record.name)
        self._indexed_names = frozenset(indexed)
        return self._snapshot

    def catalog_payload(self) -> dict[str, object]:
        if self._privacy_hidden():
            return self._empty_catalog_payload()
        included = set(self._context_render.included)
        records = [
            self._record_payload(record, included) for record in self._snapshot.records
        ]
        return {
            "skills": records,
            "count": len(records),
            "errors": [error.to_dict() for error in self._snapshot.errors],
            "enablement_available": self.enablement_available,
            "enablement_error": self._enablement_error,
            "context": {
                "text": self._context_render.text,
                "included": list(self._context_render.included),
                "dropped": list(self._context_render.dropped),
                "bytes": len(self._context_render.text.encode("utf-8")),
            },
        }

    def _empty_catalog_payload(self) -> dict[str, object]:
        return {
            "skills": [],
            "count": 0,
            "errors": [],
            "enablement_available": False,
            "enablement_error": (
                "persistent procedural skills are unavailable in the current "
                "privacy mode"
            ),
            "context": {
                "text": "",
                "included": [],
                "dropped": [],
                "bytes": 0,
            },
        }

    def _privacy_hidden(self) -> bool:
        return hides_persisted_user_content(self.agent)

    def _require_persistent_access(self) -> None:
        if self._privacy_hidden():
            raise SkillPrivacyError(
                "persistent procedural skills are unavailable in the current privacy mode"
            )

    def _hide_persistent_state(self) -> None:
        self._db = None
        self._enablement = None
        self._store = None
        self._catalog = None
        self._snapshot = CatalogSnapshot()
        self._context_render = render_context_clause(self._snapshot)
        self._states = {}
        self._enablement_error = (
            "persistent procedural skills are unavailable in the current privacy mode"
        )
        self._indexed_names = frozenset()

    def _ensure_persistent_services(self) -> None:
        if all(
            service is not None
            for service in (self._store, self._catalog, self._enablement)
        ):
            return
        local_root = _agent_local_root(self.agent)
        store = SkillStore(local_root)
        sources = [
            DirectorySkillSource(
                root=local_root,
                source_id="agent-local",
                kind="agent-local",
                precedence=AGENT_LOCAL_PRECEDENCE,
            )
        ]
        shared_root = _host_shared_root()
        if (
            shared_root is not None
            and shared_root.resolve(strict=False) != local_root.resolve()
        ):
            sources.append(
                DirectorySkillSource(
                    root=shared_root,
                    source_id="host-shared",
                    kind="host-shared",
                    precedence=HOST_SHARED_PRECEDENCE,
                )
            )
        database = resolve_feature_database(self.agent)
        self._store = store
        self._catalog = SkillCatalog(tuple(sources))
        self._db = database
        self._enablement = SkillEnablementStore(database, _agent_id(self.agent))

    @asynccontextmanager
    async def _persistent_mutation(self):
        async with optional_transition_lock(_privacy_transition_lock(self.agent)):
            self._require_persistent_access()
            self._ensure_persistent_services()
            yield

    def _record_payload(
        self, record: SkillRecord, included: set[str]
    ) -> dict[str, object]:
        shadows = self._snapshot.shadowed.get(record.name, ())
        return {
            "name": record.name,
            "description": record.document.description,
            "enabled": record.state.enabled,
            "priority": record.state.priority,
            "token_cost": estimate_skill_token_cost(
                record.name,
                record.document.description,
            ),
            "context_included": record.name in included,
            "source_id": record.source_id,
            "source_kind": record.source_kind,
            "editable": record.editable,
            "provenance": record.provenance.to_dict(),
            "shadowed": [provenance.to_dict() for provenance in shadows],
        }

    async def create_skill(
        self,
        *,
        name: str,
        description: str,
        body: str,
        enabled: bool = False,
        priority: int = DEFAULT_PRIORITY,
    ) -> dict[str, object]:
        async with self._persistent_mutation():
            return await self._create_skill_locked(
                name=name,
                description=description,
                body=body,
                enabled=enabled,
                priority=priority,
            )

    async def _create_skill_locked(
        self,
        *,
        name: str,
        description: str,
        body: str,
        enabled: bool = False,
        priority: int = DEFAULT_PRIORITY,
    ) -> dict[str, object]:
        store, enablement = self._require_services()
        document = SkillDocument(validate_skill_name(name), description, body)
        # Serialization performs complete format validation before mkdir/write.
        from .format import serialize_skill_markdown

        serialize_skill_markdown(document)
        resolved_priority = validate_priority(priority)
        previous_state = self._states.get(document.name)
        folder = store.create(document)
        created_stat = folder.stat()
        created_identity = (created_stat.st_dev, created_stat.st_ino)
        state_error: str | None = None
        if enablement.available:
            try:
                state = await enablement.set(
                    name, enabled=enabled, priority=resolved_priority
                )
                self._states[name] = state
            except DatabaseError as exc:
                try:
                    persisted_states = await enablement.load()
                except DatabaseError:
                    store.rollback_created(folder, identity=created_identity)
                    if previous_state is None:
                        self._states.pop(name, None)
                    else:
                        self._states[name] = previous_state
                    await self._refresh_locked()
                    raise
                observed_state = persisted_states.get(name)
                if observed_state and observed_state.enabled:
                    store.rollback_created(folder, identity=created_identity)
                    self._states = dict(persisted_states)
                    await self._refresh_locked()
                    raise
                self._states = dict(persisted_states)
                self._states[name] = observed_state or SkillState(
                    False, resolved_priority
                )
                state_error = str(exc)
        elif enabled:
            state_error = "agent database unavailable; the new skill remains disabled"
        await self._refresh_locked()
        record = SkillStore.get(self._snapshot, name)
        return {
            "name": name,
            "folder": str(folder),
            "enabled": record.state.enabled,
            "priority": record.state.priority,
            "indexed": name in self._indexed_names,
            "state_error": state_error,
        }

    async def edit_skill(
        self, *, name: str, relative_path: str, content: str
    ) -> dict[str, object]:
        async with self._persistent_mutation():
            return await self._edit_skill_locked(
                name=name, relative_path=relative_path, content=content
            )

    async def _edit_skill_locked(
        self, *, name: str, relative_path: str, content: str
    ) -> dict[str, object]:
        store, _ = self._require_services()
        record = SkillStore.get(self._snapshot, name)
        store.write_file(record, relative_path, content)
        await self._refresh_locked()
        SkillStore.get(self._snapshot, name)
        return {
            "name": name,
            "path": relative_path,
            "indexed": name in self._indexed_names,
        }

    async def set_skill_state(
        self,
        *,
        name: str,
        enabled: bool,
        priority: int | None = None,
    ) -> dict[str, object]:
        async with self._persistent_mutation():
            return await self._set_skill_state_locked(
                name=name, enabled=enabled, priority=priority
            )

    async def _set_skill_state_locked(
        self,
        *,
        name: str,
        enabled: bool,
        priority: int | None = None,
    ) -> dict[str, object]:
        _, enablement = self._require_services()
        record = SkillStore.get(self._snapshot, name)
        resolved_priority = (
            record.state.priority if priority is None else validate_priority(priority)
        )
        state = await enablement.set(name, enabled=enabled, priority=resolved_priority)
        self._states[name] = state
        await self._refresh_locked()
        refreshed = SkillStore.get(self._snapshot, name)
        return {
            "name": name,
            "enabled": refreshed.state.enabled,
            "priority": refreshed.state.priority,
            "indexed": name in self._indexed_names,
            "context_bytes": len(self._context_render.text.encode("utf-8")),
        }

    async def delete_skill(self, *, name: str) -> dict[str, object]:
        async with self._persistent_mutation():
            return await self._delete_skill_locked(name=name)

    async def _delete_skill_locked(self, *, name: str) -> dict[str, object]:
        store, enablement = self._require_services()
        record = SkillStore.get(self._snapshot, name)
        node_id = self._node_id(name)
        store.delete(record)
        config_deleted = True
        graph_deleted = True
        errors: list[str] = []
        if enablement.available:
            try:
                await enablement.delete(name)
                self._states.pop(name, None)
            except DatabaseError as exc:
                config_deleted = False
                errors.append(f"enablement row cleanup failed: {exc}")
        storage = getattr(self.agent, "storage", None)
        if (
            storage is not None
            and hasattr(storage, "get_node")
            and hasattr(storage, "delete_node")
        ):
            try:
                node = await storage.get_node(node_id)
                if (
                    node is not None
                    and getattr(node, "node_type", None) == PROCEDURAL_SKILL_NODE_TYPE
                ):
                    await storage.delete_node(node_id)
            except Exception as exc:  # noqa: BLE001 - graph is a recoverable index
                graph_deleted = False
                errors.append(f"graph index cleanup failed: {exc}")
        await self._refresh_locked()
        return {
            "name": name,
            "removed_file": True,
            "config_deleted": config_deleted,
            "graph_deleted": graph_deleted,
            "errors": errors,
        }

    async def install_skill(
        self,
        *,
        source_url: str,
        skill_name: str,
        ref: str = "HEAD",
    ) -> dict[str, object]:
        async with self._persistent_mutation():
            return await self._install_skill_locked(
                source_url=source_url, skill_name=skill_name, ref=ref
            )

    async def _install_skill_locked(
        self,
        *,
        source_url: str,
        skill_name: str,
        ref: str = "HEAD",
    ) -> dict[str, object]:
        store, enablement = self._require_services()
        skill_name = validate_skill_name(skill_name)
        if skill_name in self._snapshot.by_name():
            raise SkillConflictError(
                f"skill already exists in the resolved catalog: {skill_name}"
            )
        with tempfile.TemporaryDirectory(
            prefix=".kestrel-skill-git-",
            dir=store.local_root.parent,
        ) as temporary:
            target = Path(temporary) / "checkout"
            checkout = await asyncio.to_thread(
                GitSkillSource().checkout,
                url=source_url,
                ref=ref,
                skill_name=skill_name,
                target=target,
            )
            provenance = SkillProvenance(
                kind="git",
                source_id=checkout.remote_url,
                locator=f"{checkout.ref}:{skill_name}",
                revision=checkout.revision,
                remote_url=checkout.remote_url,
            )
            previous_state: SkillState | None = None
            state_was_persisted = False
            if enablement.available:
                previous_state = (await enablement.load()).get(skill_name)
                state = await enablement.set(
                    skill_name, enabled=False, priority=DEFAULT_PRIORITY
                )
                self._states[skill_name] = state
                state_was_persisted = True
            try:
                folder = store.install_folder(
                    checkout.skill_folder, provenance=provenance
                )
            except Exception as publication_error:
                if state_was_persisted:
                    try:
                        if previous_state is None:
                            await enablement.delete(skill_name)
                            self._states.pop(skill_name, None)
                        else:
                            restored = await enablement.set(
                                skill_name,
                                enabled=previous_state.enabled,
                                priority=previous_state.priority,
                            )
                            self._states[skill_name] = restored
                    except DatabaseError as rollback_error:
                        message = (
                            f"skill install publication failed ({publication_error}); "
                            f"enablement rollback also failed ({rollback_error})"
                        )
                        self._enablement_error = message
                        raise DatabaseError(message) from rollback_error
                raise
        await self._refresh_locked()
        record = SkillStore.get(self._snapshot, skill_name)
        return {
            "name": skill_name,
            "folder": str(folder),
            "revision": checkout.revision,
            "remote_url": checkout.remote_url,
            "enabled": record.state.enabled,
            "indexed": skill_name in self._indexed_names,
            "state_error": None,
        }

    def read_skill(self, *, name: str) -> dict[str, object]:
        store, _ = self._require_services()
        record = SkillStore.get(self._snapshot, name)
        tree = store.tree(record)
        resources = [entry for entry in tree if entry["path"] != SKILL_FILENAME]
        return {
            "name": name,
            "description": record.document.description,
            "body": record.document.body,
            "resources": resources,
            "provenance": record.provenance.to_dict(),
        }

    def read_file(self, *, name: str, relative_path: str) -> dict[str, object]:
        store, _ = self._require_services()
        record = SkillStore.get(self._snapshot, name)
        content = store.read_file(record, relative_path)
        return {
            "name": name,
            "path": relative_path,
            "content": content,
            "editable": record.editable,
            "language": "python" if relative_path.endswith(".py") else "markdown",
            "execution_risk": relative_path.endswith(".py"),
        }

    def tree(self, *, name: str) -> tuple[dict[str, object], ...]:
        store, _ = self._require_services()
        return store.tree(SkillStore.get(self._snapshot, name))

    async def _index_record(self, record: SkillRecord) -> bool:
        storage = getattr(self.agent, "storage", None)
        if storage is None or not hasattr(storage, "add_node"):
            return False
        node = GraphNode(
            node_id=self._node_id(record.name),
            node_type=PROCEDURAL_SKILL_NODE_TYPE,
            label=record.name,
            properties={
                "agent_id": _agent_id(self.agent),
                "name": record.name,
                "description": record.document.description,
                "source_id": record.source_id,
                "enabled": record.state.enabled,
                "priority": record.state.priority,
            },
        )
        try:
            await storage.add_node(node)
        except Exception as exc:  # noqa: BLE001 - graph is a recoverable index
            logger.warning(
                "Could not update procedural_skill index for %s: %s", record.name, exc
            )
            return False
        return True

    async def _delete_index_node(self, name: str) -> bool:
        """Best-effort removal for a catalog entry no longer on disk."""

        storage = getattr(self.agent, "storage", None)
        if (
            storage is None
            or not hasattr(storage, "get_node")
            or not hasattr(storage, "delete_node")
        ):
            return False
        node_id = self._node_id(name)
        try:
            node = await storage.get_node(node_id)
            if node is None:
                return True
            if getattr(node, "node_type", None) != PROCEDURAL_SKILL_NODE_TYPE:
                logger.warning(
                    "Refusing to delete non-procedural node at expected skill index id %s",
                    node_id,
                )
                return True
            await storage.delete_node(node_id)
        except Exception as exc:  # noqa: BLE001 - graph is a recoverable index
            logger.warning(
                "Could not remove stale procedural_skill index for %s: %s", name, exc
            )
            return False
        return True

    async def _persisted_index_names(self) -> set[str]:
        """Discover this agent's prior index nodes so restart can reconcile them."""

        storage = getattr(self.agent, "storage", None)
        if storage is None or not hasattr(storage, "get_nodes_by_type"):
            return set()
        try:
            nodes = await storage.get_nodes_by_type(PROCEDURAL_SKILL_NODE_TYPE)
        except Exception as exc:  # noqa: BLE001 - graph is a recoverable index
            logger.warning("Could not enumerate procedural_skill index nodes: %s", exc)
            return set()
        names: set[str] = set()
        for node in nodes:
            properties = getattr(node, "properties", None)
            name = properties.get("name") if isinstance(properties, dict) else None
            try:
                name = validate_skill_name(name)
            except SkillError:
                continue
            if getattr(node, "node_id", None) == self._node_id(name):
                names.add(name)
        return names

    def _node_id(self, name: str) -> str:
        material = f"{_agent_id(self.agent)}\x00{name}".encode()
        return f"procedural-skill:{hashlib.sha256(material).hexdigest()[:32]}"

    def _require_services(self) -> tuple[SkillStore, SkillEnablementStore]:
        self._require_persistent_access()
        if self._store is None or self._enablement is None:
            raise RuntimeError("ProceduralSkillsFeature is not initialized")
        return self._store, self._enablement

    @tool(
        "skill_list",
        "List procedural skills without disclosing their procedure bodies",
        category=ToolCategory.UTILITY,
        command_prefix="!skill list",
    )
    async def skill_list(self) -> ToolResult:
        payload = self.catalog_payload()
        names = [item["name"] for item in payload["skills"]]  # type: ignore[index]
        confirmation = (
            f"{len(names)} procedural skill(s): {', '.join(names)}"
            if names
            else "No procedural skills were discovered."
        )
        return ToolResult.ok(confirmation, data=payload)

    @tool(
        "skill_read",
        "Read one procedural skill body and its resource inventory",
        category=ToolCategory.UTILITY,
        command_prefix="!skill read",
    )
    async def skill_read(self, name: str) -> ToolResult:
        try:
            payload = self.read_skill(name=name)
        except SkillError as exc:
            return ToolResult.failed(str(exc))
        return ToolResult.ok(
            f"Read procedural skill {name}:\n{payload['body']}",
            data=payload,
        )

    @tool(
        "skill_search",
        "Search procedural skill names and descriptions without disclosing bodies",
        category=ToolCategory.UTILITY,
        command_prefix="!skill search",
    )
    async def skill_search(self, query: str) -> ToolResult:
        try:
            matches = SkillStore.search(self.snapshot, query)
        except SkillError as exc:
            return ToolResult.failed(str(exc))
        rows = [
            self._record_payload(record, set(self._context_render.included))
            for record in matches
        ]
        return ToolResult.ok(
            f"Found {len(rows)} procedural skill(s) matching {query!r}.",
            data={"skills": rows, "count": len(rows)},
        )

    @tool(
        "skill_create",
        "Create a local procedural skill folder; it is disabled unless explicitly enabled",
        category=ToolCategory.UTILITY,
        command_prefix="!skill create",
    )
    async def skill_create(
        self,
        name: str,
        description: str,
        body: str,
        enabled: bool = False,
        priority: int = DEFAULT_PRIORITY,
    ) -> ToolResult:
        try:
            payload = await self.create_skill(
                name=name,
                description=description,
                body=body,
                enabled=enabled,
                priority=priority,
            )
        except (SkillError, OSError, DatabaseError, RuntimeError, ValueError) as exc:
            return ToolResult.failed(str(exc))
        if payload["state_error"]:
            return ToolResult.partial(
                f"Created procedural skill {name} on disk.",
                str(payload["state_error"]),
                data=payload,
            )
        return ToolResult.ok(f"Created procedural skill {name}.", data=payload)

    @tool(
        "skill_edit",
        "Edit SKILL.md, a Markdown resource, or a scripts/*.py file without executing it",
        category=ToolCategory.UTILITY,
        command_prefix="!skill edit",
    )
    async def skill_edit(
        self,
        name: str,
        content: str,
        relative_path: str = SKILL_FILENAME,
    ) -> ToolResult:
        try:
            payload = await self.edit_skill(
                name=name,
                relative_path=relative_path,
                content=content,
            )
        except (SkillError, OSError, DatabaseError, RuntimeError) as exc:
            return ToolResult.failed(str(exc))
        return ToolResult.ok(
            f"Edited {name}/{relative_path} without executing it.", data=payload
        )

    @tool(
        "skill_enable",
        "Enable a procedural skill description in future prompt context",
        category=ToolCategory.UTILITY,
        command_prefix="!skill enable",
    )
    async def skill_enable(self, name: str, priority: int | None = None) -> ToolResult:
        try:
            payload = await self.set_skill_state(
                name=name, enabled=True, priority=priority
            )
        except (SkillError, DatabaseError, RuntimeError, ValueError) as exc:
            return ToolResult.failed(str(exc))
        return ToolResult.ok(f"Enabled procedural skill {name}.", data=payload)

    @tool(
        "skill_disable",
        "Disable a procedural skill description in future prompt context",
        category=ToolCategory.UTILITY,
        command_prefix="!skill disable",
    )
    async def skill_disable(self, name: str) -> ToolResult:
        try:
            payload = await self.set_skill_state(name=name, enabled=False)
        except (SkillError, DatabaseError, RuntimeError, ValueError) as exc:
            return ToolResult.failed(str(exc))
        return ToolResult.ok(f"Disabled procedural skill {name}.", data=payload)

    @tool(
        "skill_delete",
        "Permanently delete a local procedural skill and its secondary index",
        category=ToolCategory.UTILITY,
        command_prefix="!skill delete",
    )
    async def skill_delete(self, name: str) -> ToolResult:
        try:
            payload = await self.delete_skill(name=name)
        except (SkillError, OSError, DatabaseError, RuntimeError) as exc:
            return ToolResult.failed(str(exc))
        errors = payload["errors"]
        if errors:
            return ToolResult.partial(
                f"Deleted the authoritative skill folder for {name}.",
                "; ".join(errors),
                data=payload,
            )
        return ToolResult.ok(f"Deleted procedural skill {name}.", data=payload)

    @tool(
        "skill_install",
        "Install one procedural skill from a bounded HTTPS git source without executing code",
        category=ToolCategory.UTILITY,
        command_prefix="!skill install",
    )
    async def skill_install(
        self,
        source_url: str,
        skill_name: str,
        ref: str = "HEAD",
    ) -> ToolResult:
        try:
            payload = await self.install_skill(
                source_url=source_url,
                skill_name=skill_name,
                ref=ref,
            )
        except (SkillError, OSError, DatabaseError, RuntimeError) as exc:
            return ToolResult.failed(str(exc))
        if payload["state_error"]:
            return ToolResult.partial(
                f"Installed procedural skill {skill_name} at {payload['revision']} and left it disabled.",
                str(payload["state_error"]),
                data=payload,
            )
        return ToolResult.ok(
            f"Installed procedural skill {skill_name} at {payload['revision']} and left it disabled.",
            data=payload,
        )


__all__ = [
    "PROCEDURAL_SKILL_NODE_TYPE",
    "SKILLS_CAPABILITY",
    "ProceduralSkillsFeature",
]
