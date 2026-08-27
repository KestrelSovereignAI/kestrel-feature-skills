"""Kestrel feature entry point for folder-shaped procedural skills."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import tempfile
import threading
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
from .errors import SkillConflictError, SkillError, SkillPathError, SkillPrivacyError
from .format import SKILL_FILENAME, validate_skill_name
from .git_source import GitCheckout, GitSkillSource
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
from .store import CreatedSkillPublication, SkillStore

logger = logging.getLogger(__name__)

PROCEDURAL_SKILL_NODE_TYPE = "procedural_skill"
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
            # Core 0.53.x cannot derive a feature capability for an extracted
            # class absent from its static registry. The module therefore
            # probes its live, agent-scoped route and gates its own panel.
            capability=None,
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
        async with self.persistent_read(refresh=True):
            return self._snapshot

    @asynccontextmanager
    async def persistent_read(self, *, refresh: bool = False):
        """Hold the privacy-transition lock through one persisted read scope."""

        async with optional_transition_lock(_privacy_transition_lock(self.agent)):
            if self._privacy_hidden():
                self._hide_persistent_state()
                yield False
                return
            needs_refresh = any(
                service is None
                for service in (self._store, self._catalog, self._enablement)
            )
            self._ensure_persistent_services()
            if refresh or needs_refresh:
                await self._refresh_locked()
            yield True

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
        return await self._rebuild_snapshot_locked(states)

    async def _rebuild_snapshot_locked(
        self, states: dict[str, SkillState]
    ) -> CatalogSnapshot:
        """Rebuild catalog, prompt clause, and graph index from observed state."""

        if self._catalog is None:
            raise RuntimeError("ProceduralSkillsFeature is not initialized")
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
                # SkillStore pins the resolved directory and its inode.  Use
                # that same path for discovery so retargeting a symlinked
                # ancestor cannot split reads from mutations.
                root=store.local_root,
                source_id="agent-local",
                kind="agent-local",
                precedence=AGENT_LOCAL_PRECEDENCE,
                expected_root_identity=store.local_root_identity,
            )
        ]
        shared_root = _host_shared_root()
        include_shared = shared_root is not None
        if shared_root is not None:
            try:
                include_shared = not shared_root.samefile(store.local_root)
            except (OSError, RuntimeError):
                # An optional malformed source belongs in catalog errors; it
                # must not abort construction of the healthy local source.
                include_shared = True
        if shared_root is not None and include_shared:
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
            needs_refresh = any(
                service is None
                for service in (self._store, self._catalog, self._enablement)
            )
            self._ensure_persistent_services()
            if needs_refresh:
                await self._refresh_locked()
            try:
                yield
            except BaseException as operation_error:
                # A cancellation or failure can arrive after a filesystem or
                # database mutation committed but before the ordinary refresh.
                # Reconcile while the privacy-transition lock is still held so
                # subsequent reads never retain the pre-mutation snapshot.
                await self._reconcile_interrupted_mutation(operation_error)
                raise

    @asynccontextmanager
    async def _publication_state_claim(self, store: SkillStore, name: str):
        """Hold a cross-process name claim without blocking the event loop."""

        claim = None
        while claim is None:
            claim = store.try_acquire_publication_state_claim(name)
            if claim is None:
                await asyncio.sleep(0.025)
        try:
            yield
        finally:
            claim.release()

    async def _reconcile_interrupted_mutation(
        self, operation_error: BaseException
    ) -> None:
        """Finish one refresh despite repeated cancellation, then propagate."""

        worker = asyncio.create_task(self._refresh_locked())
        while True:
            try:
                await asyncio.shield(worker)
                return
            except asyncio.CancelledError:
                if not worker.done():
                    # A repeated cancellation belongs to the interrupted
                    # caller. Keep draining the independently owned refresh.
                    continue
                try:
                    worker.result()
                except BaseException as refresh_error:
                    refresh_error.add_note(
                        "procedural skill mutation was already interrupted: "
                        f"{operation_error}"
                    )
                    raise refresh_error from operation_error
                raise
            except BaseException as refresh_error:
                refresh_error.add_note(
                    "procedural skill mutation was already interrupted: "
                    f"{operation_error}"
                )
                raise refresh_error from operation_error

    @staticmethod
    async def _drain_shielded_task(worker: asyncio.Task):
        """Wait for owned cleanup work even if the caller is cancelled again."""

        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
        return worker.result()

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
        async with self._publication_state_claim(store, document.name):
            return await self._create_skill_with_publication_claim(
                store=store,
                enablement=enablement,
                document=document,
                enabled=enabled,
                resolved_priority=resolved_priority,
            )

    async def _create_skill_with_publication_claim(
        self,
        *,
        store: SkillStore,
        enablement: SkillEnablementStore,
        document: SkillDocument,
        enabled: bool,
        resolved_priority: int,
    ) -> dict[str, object]:
        name = document.name
        previous_state: SkillState | None = None
        state_was_persisted = False
        if enablement.available:
            previous_state, state = await self._prepare_disabled_state(
                enablement,
                document.name,
                priority=resolved_priority,
                operation="skill create disabled-state preparation",
            )
            self._states[document.name] = state
            state_was_persisted = True
        try:
            folder, created_identity = store.create_pinned(document)
        except Exception as publication_error:
            if state_was_persisted:
                await self._restore_enablement_after_publication_failure(
                    enablement,
                    document.name,
                    previous_state,
                    publication_error,
                    operation="skill create",
                )
            raise
        state_error: str | None = None
        if enablement.available and enabled:
            try:
                state = await enablement.set(
                    name, enabled=True, priority=resolved_priority
                )
                self._states[name] = state
            except DatabaseError as exc:
                try:
                    persisted_states = await enablement.load()
                except DatabaseError as load_error:
                    load_error.add_note(
                        f"skill enablement update originally failed: {exc}"
                    )
                    await self._rollback_created_publication(
                        store,
                        enablement,
                        folder,
                        identity=created_identity,
                        previous_state=previous_state,
                        publication_error=load_error,
                    )
                    await self._refresh_locked()
                    raise
                observed_state = persisted_states.get(name)
                if observed_state and observed_state.enabled:
                    await self._rollback_created_publication(
                        store,
                        enablement,
                        folder,
                        identity=created_identity,
                        previous_state=previous_state,
                        publication_error=exc,
                    )
                    await self._refresh_locked()
                    raise
                self._states = dict(persisted_states)
                self._states[name] = observed_state or SkillState(
                    False, resolved_priority
                )
                state_error = str(exc)
        elif enabled and resolved_priority != DEFAULT_PRIORITY:
            state_error = (
                "agent database unavailable; the new skill remains disabled and uses "
                f"default priority {DEFAULT_PRIORITY} instead of requested priority "
                f"{resolved_priority}"
            )
        elif enabled:
            state_error = "agent database unavailable; the new skill remains disabled"
        elif resolved_priority != DEFAULT_PRIORITY:
            state_error = (
                "agent database unavailable; requested priority "
                f"{resolved_priority} could not be persisted; the new skill uses "
                f"default priority {DEFAULT_PRIORITY}"
            )
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

    async def _rollback_created_publication(
        self,
        store: SkillStore,
        enablement: SkillEnablementStore,
        folder: Path,
        *,
        identity: CreatedSkillPublication,
        previous_state: SkillState | None,
        publication_error: BaseException,
    ) -> None:
        """Restore enablement even when inode-pinned filesystem rollback refuses."""

        rollback_error: BaseException | None = None
        try:
            store.rollback_created(folder, identity=identity)
        except BaseException as exc:  # noqa: BLE001 - both cleanup rails must run
            rollback_error = exc
            exc.add_note(
                f"skill create state update originally failed: {publication_error}"
            )
        restoration = asyncio.create_task(
            self._restore_enablement_after_publication_failure(
                enablement,
                folder.name,
                previous_state,
                publication_error,
                operation="skill create state update",
            )
        )
        try:
            await self._drain_shielded_task(restoration)
        except BaseException as state_error:
            if rollback_error is not None:
                state_error.add_note(
                    f"filesystem rollback also reported: {rollback_error}"
                )
            raise
        if rollback_error is not None:
            raise rollback_error from publication_error

    async def _restore_enablement_after_publication_failure(
        self,
        enablement: SkillEnablementStore,
        name: str,
        previous_state: SkillState | None,
        publication_error: BaseException,
        *,
        operation: str,
    ) -> None:
        try:
            if previous_state is None:
                await enablement.delete(name)
                self._states.pop(name, None)
            else:
                restored = await enablement.set(
                    name,
                    enabled=previous_state.enabled,
                    priority=previous_state.priority,
                )
                self._states[name] = restored
        except DatabaseError as rollback_error:
            message = (
                f"{operation} failed ({publication_error}); "
                f"enablement rollback also failed ({rollback_error})"
            )
            self._enablement_error = message
            raise DatabaseError(message) from rollback_error

    async def _prepare_disabled_state(
        self,
        enablement: SkillEnablementStore,
        name: str,
        *,
        priority: int,
        operation: str,
    ) -> tuple[SkillState | None, SkillState]:
        """Persist a disabled publication guard without losing prior state."""

        try:
            previous_state = (await enablement.load()).get(name)
        except DatabaseError as exc:
            self._enablement_error = str(exc)
            raise
        guard = asyncio.create_task(
            enablement.set(name, enabled=False, priority=priority)
        )
        try:
            state = await asyncio.shield(guard)
        except BaseException as guard_error:
            if isinstance(guard_error, DatabaseError):
                self._enablement_error = str(guard_error)
            # The write may have committed immediately before an exception or
            # caller cancellation. Drain it, then restore the prior row through
            # independently owned cleanup before propagating the original error.
            if not guard.done():
                guard.cancel()
            try:
                await self._drain_shielded_task(guard)
            except BaseException as drain_error:  # noqa: BLE001 - cleanup owns task
                guard_error.add_note(
                    f"disabled-state guard completion also reported: {drain_error}"
                )
            restoration = asyncio.create_task(
                self._restore_enablement_after_publication_failure(
                    enablement,
                    name,
                    previous_state,
                    guard_error,
                    operation=operation,
                )
            )
            await self._drain_shielded_task(restoration)
            raise
        return previous_state, state

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
        store, enablement = self._require_services()
        name = validate_skill_name(name)
        async with self._publication_state_claim(store, name):
            # A publication or deletion in another feature instance may have
            # completed while this caller waited. Resolve the state mutation
            # against the catalog version protected by the same name claim.
            await self._refresh_locked()
            return await self._set_skill_state_with_publication_claim(
                enablement=enablement,
                name=name,
                enabled=enabled,
                priority=priority,
            )

    async def _set_skill_state_with_publication_claim(
        self,
        *,
        enablement: SkillEnablementStore,
        name: str,
        enabled: bool,
        priority: int | None,
    ) -> dict[str, object]:
        record = SkillStore.get(self._snapshot, name)
        resolved_priority = (
            record.state.priority if priority is None else validate_priority(priority)
        )
        try:
            state = await enablement.set(
                name, enabled=enabled, priority=resolved_priority
            )
        except DatabaseError as exc:
            self._enablement_error = str(exc)
            observed_states = await enablement.load()
            self._states = dict(observed_states)
            await self._rebuild_snapshot_locked(self._states)
            raise
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
        name = validate_skill_name(name)
        async with self._publication_state_claim(store, name):
            # Re-resolve after waiting so a stale feature instance cannot
            # delete a replacement that was published before it won the claim.
            await self._refresh_locked()
            return await self._delete_skill_with_publication_claim(
                store=store,
                enablement=enablement,
                name=name,
            )

    async def _delete_skill_with_publication_claim(
        self,
        *,
        store: SkillStore,
        enablement: SkillEnablementStore,
        name: str,
    ) -> dict[str, object]:
        record = SkillStore.get(self._snapshot, name)
        node_id = self._node_id(name)
        store.require_local_record(record)
        if enablement.available:
            _previous_state, disabled_state = await self._prepare_disabled_state(
                enablement,
                name,
                priority=record.state.priority,
                operation="skill delete disabled-state preparation",
            )
            self._states[name] = disabled_state
        try:
            store.delete(record)
        except BaseException as deletion_error:
            # A failed recursive removal can leave the original only under a
            # quarantine name. Retaining the disabled tombstone is the only
            # fail-safe result when folder cleanup cannot be confirmed.
            if enablement.available:
                deletion_error.add_note(
                    "the skill was disabled before deletion and remains disabled"
                )
            raise
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
            checkout = await self._checkout_git_until_stopped(
                source_url=source_url,
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
            async with self._publication_state_claim(store, skill_name):
                # Checkout can be slow enough for a host-shared skill with the
                # same name to appear after the initial catalog check. Refresh
                # while holding the publication claim so that local install
                # cannot silently shadow the newly resolved source.
                await self._refresh_locked()
                if skill_name in self._snapshot.by_name():
                    raise SkillConflictError(
                        f"skill already exists in the resolved catalog: {skill_name}"
                    )
                folder = await self._publish_installed_skill_with_state_claim(
                    store=store,
                    enablement=enablement,
                    source_folder=checkout.skill_folder,
                    provenance=provenance,
                    skill_name=skill_name,
                )
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

    async def _publish_installed_skill_with_state_claim(
        self,
        *,
        store: SkillStore,
        enablement: SkillEnablementStore,
        source_folder: Path,
        provenance: SkillProvenance,
        skill_name: str,
    ) -> Path:
        previous_state: SkillState | None = None
        state_was_persisted = False
        if enablement.available:
            previous_state, state = await self._prepare_disabled_state(
                enablement,
                skill_name,
                priority=DEFAULT_PRIORITY,
                operation="skill install disabled-state preparation",
            )
            self._states[skill_name] = state
            state_was_persisted = True
        try:
            return store.install_folder(source_folder, provenance=provenance)
        except Exception as publication_error:
            if state_was_persisted:
                await self._restore_enablement_after_publication_failure(
                    enablement,
                    skill_name,
                    previous_state,
                    publication_error,
                    operation="skill install publication",
                )
            raise

    @staticmethod
    async def _checkout_git_until_stopped(
        *, source_url: str, ref: str, skill_name: str, target: Path
    ) -> GitCheckout:
        """Keep cancellation inside the privacy lock until Git has stopped."""

        cancel_event = threading.Event()
        worker = asyncio.create_task(
            asyncio.to_thread(
                GitSkillSource().checkout,
                url=source_url,
                ref=ref,
                skill_name=skill_name,
                target=target,
                cancel_event=cancel_event,
            )
        )
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            cancel_event.set()
            while True:
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    # Repeated cancellation must not release the privacy lock
                    # while the same checkout thread is still active.
                    cancel_event.set()
                    continue
                except Exception:  # noqa: BLE001 - preserve caller cancellation
                    break
                else:
                    break
            raise

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
            "editable": store.file_is_editable(record, relative_path),
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
        try:
            async with self.persistent_read():
                payload = self.catalog_payload()
        except (SkillError, OSError, DatabaseError, RuntimeError) as exc:
            return ToolResult.failed(str(exc))
        names = [item["name"] for item in payload["skills"]]  # type: ignore[index]
        confirmation = (
            f"{len(names)} procedural skill(s): {', '.join(names)}"
            if names
            else "No procedural skills were discovered."
        )
        return ToolResult.ok(confirmation, data=payload)

    @tool(
        "skill_read",
        "Read one procedural skill body or one inventoried bundled resource",
        category=ToolCategory.UTILITY,
        command_prefix="!skill read",
    )
    async def skill_read(self, name: str, path: str = SKILL_FILENAME) -> ToolResult:
        try:
            async with self.persistent_read():
                inventory = self.read_skill(name=name)
                if path == SKILL_FILENAME:
                    payload = inventory
                    confirmation = f"Read procedural skill {name}:\n{payload['body']}"
                else:
                    inventoried_files = {
                        str(item["path"])
                        for item in inventory["resources"]
                        if item.get("type") == "file"
                    }
                    if path not in inventoried_files:
                        raise SkillPathError(
                            f"resource is not in the skill inventory: {path}"
                        )
                    payload = self.read_file(name=name, relative_path=path)
                    confirmation = (
                        f"Read procedural skill resource {name}/{path} as text; "
                        f"no code was executed:\n{payload['content']}"
                    )
        except (SkillError, OSError, DatabaseError, RuntimeError) as exc:
            return ToolResult.failed(str(exc))
        return ToolResult.ok(confirmation, data=payload)

    @tool(
        "skill_search",
        "Search procedural skill names and descriptions without disclosing bodies",
        category=ToolCategory.UTILITY,
        command_prefix="!skill search",
    )
    async def skill_search(self, query: str) -> ToolResult:
        try:
            async with self.persistent_read():
                matches = SkillStore.search(self.snapshot, query)
                rows = [
                    self._record_payload(record, set(self._context_render.included))
                    for record in matches
                ]
        except (SkillError, OSError, DatabaseError, RuntimeError) as exc:
            return ToolResult.failed(str(exc))
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
    "ProceduralSkillsFeature",
]
