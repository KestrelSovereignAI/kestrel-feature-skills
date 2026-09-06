"""Kestrel feature entry point for folder-shaped procedural skills."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kestrel_sdk.features.base import Feature, tool
from kestrel_sdk.features.contributions import (
    ContextClauseRegistration,
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
from kestrel_sovereign.storage.async_graph_store import (
    GraphNode,
    NodeDeleteResult,
    NodeSwapResult,
)
from kestrel_sovereign.storage.privacy_wrapper import optional_transition_lock

from .context import (
    DEFAULT_CONTEXT_BUDGET_BYTES,
    estimate_skill_token_cost,
    render_context_clause,
)
from .enablement import DEFAULT_PRIORITY, SkillEnablementStore, validate_priority
from .errors import (
    SkillConflictError,
    SkillDeletionError,
    SkillError,
    SkillFormatError,
    SkillPathError,
    SkillPrivacyError,
    SkillPublicationCleanupError,
)
from .format import SKILL_FILENAME, validate_skill_name
from .git_source import GitCheckout, GitSkillSource
from .models import (
    CatalogSnapshot,
    ContextRender,
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
from .store import (
    CreatedSkillPublication,
    InstalledSkillPublication,
    SkillStore,
    has_python_execution_risk,
)

logger = logging.getLogger(__name__)

PROCEDURAL_SKILL_NODE_TYPE = "procedural_skill"
_STATIC_DIR = Path(__file__).resolve().parent / "static"


class _ContextPublicationError(RuntimeError):
    """A catalog refresh committed locally but failed to publish prompt bytes."""


@dataclass(frozen=True, slots=True)
class _InstalledSkillOperation:
    """One installed publication plus the state needed to compensate safely."""

    folder: Path
    publication: InstalledSkillPublication
    previous_state: SkillState | None
    disabled_state: SkillState | None
    state_was_persisted: bool
    preserve_fail_closed: bool


def _same_resolved_folder(left: SkillRecord, right: SkillRecord) -> bool:
    """Return whether two snapshots prove the same resolved folder identity."""

    return bool(
        left.folder == right.folder
        and left.folder_identity is not None
        and left.folder_identity == right.folder_identity
    )


def _deletion_record(
    snapshot: CatalogSnapshot,
    resolved_record: SkillRecord,
) -> SkillRecord:
    """Return the exact local record a delete action would remove."""

    if resolved_record.editable:
        return resolved_record
    return next(
        (
            candidate
            for candidate in snapshot.shadowed_records.get(resolved_record.name, ())
            if candidate.source_kind == "agent-local"
        ),
        resolved_record,
    )


def _require_expected_revision(
    record: SkillRecord,
    expected_revision: str | None,
    *,
    operation: str,
) -> None:
    """Reject an operator mutation against content/state it did not observe."""

    if expected_revision is None:
        return
    if (
        not isinstance(expected_revision, str)
        or len(expected_revision) != 64
        or any(character not in "0123456789abcdef" for character in expected_revision)
    ):
        raise SkillConflictError(
            f"skill {record.name!r} received an invalid revision token for {operation}; "
            "reload and retry"
        )
    if not hmac.compare_digest(record.revision, expected_revision):
        raise SkillConflictError(
            f"skill {record.name!r} changed before {operation}; reload and retry"
        )


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
        self._context_publication_uncertain = False
        self._context_clause_registration = ContextClauseRegistration(
            owner=self.contribution_owner,
            name="procedural-skills",
            priority=100,
            renderer=self._render_contributed_context,
        )
        self._states: dict[str, SkillState] = {}
        self._enablement_error: str | None = None
        self._durable_fail_closed_load_error: str | None = None
        self._fail_closed_states: dict[str, SkillState] = {}
        self._fail_closed_errors: dict[str, str] = {}
        self._durable_fail_closed_names: set[str] = set()
        self._releasing_fail_closed_names: ContextVar[frozenset[str]] = ContextVar(
            f"skills-releasing-fail-closed-{id(self)}",
            default=frozenset(),
        )
        self._indexed_names: frozenset[str] = frozenset()
        self._indexed_payloads: dict[str, str] | None = None
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
            capability="procedural-skills",
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

    def get_context_clause_registrations(
        self,
    ) -> tuple[ContextClauseRegistration, ...]:
        return (self._context_clause_registration,)

    def _render_contributed_context(self) -> str:
        """Resolve bytes only when core performs a lifecycle transition."""

        return self.context_clause_text

    def _publish_context_transition(self) -> None:
        """Refresh core's immutable cache after persisted config changes."""

        runtime = getattr(self.agent, "feature_contribution_runtime", None)
        is_active = getattr(runtime, "is_active", None)
        if not callable(is_active) or not is_active(self):
            return
        refresh = getattr(self.agent, "refresh_feature_context_clauses", None)
        if not callable(refresh):
            raise RuntimeError(  # noqa: TRY004 - missing host capability, not bad input
                "active procedural skills require core context-clause refresh support"
            )
        refresh(self)

    def _replace_context_render(self, replacement: ContextRender) -> None:
        """Publish new bytes atomically enough for a later retry."""

        previous = self._context_render
        self._context_render = replacement
        if (
            replacement.text == previous.text
            and not self._context_publication_uncertain
        ):
            return
        try:
            # A prior host-owned batch may have prepared these exact local
            # bytes and then aborted because another feature failed. Publish
            # even when the local render is unchanged so the next deliberate
            # Skills refresh repairs that possible registry divergence.
            self._publish_context_transition()
        except asyncio.CancelledError:
            self._context_render = previous
            raise
        except Exception as exc:
            # The core registry still owns ``previous``. Retain that exact
            # local value so a subsequent refresh sees a difference and retries
            # publication instead of treating the failed update as committed.
            self._context_render = previous
            raise _ContextPublicationError(
                f"procedural skill context publication failure: {exc}"
            ) from exc
        except BaseException:
            self._context_render = previous
            raise
        self._context_publication_uncertain = False

    @property
    def snapshot(self) -> CatalogSnapshot:
        return CatalogSnapshot() if self._privacy_hidden() else self._snapshot

    @property
    def context_clause_text(self) -> str:
        """The memoized, byte-stable text published through the SDK seam."""

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

    async def prepare_context_clause_refresh(self) -> None:
        """Rehydrate policy-dependent state before Core renders a new batch.

        Core owns publication of the complete context-clause batch.  This hook
        therefore refreshes the local immutable render without calling the
        synchronous per-feature publication seam.
        """

        async with optional_transition_lock(_privacy_transition_lock(self.agent)):
            if self._privacy_hidden():
                self._hide_persistent_state(publish_context=False)
                return
            self._ensure_persistent_services()
            await self._refresh_locked(publish_context=False)

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

    async def _refresh_locked(
        self,
        *,
        publish_context: bool = True,
        initial_states: dict[str, SkillState] | None = None,
        primary_enablement_error: str | None = None,
    ) -> CatalogSnapshot:
        if self._catalog is None or self._enablement is None:
            raise RuntimeError("ProceduralSkillsFeature is not initialized")
        states = (
            await self._load_refresh_states(
                primary_error=primary_enablement_error,
            )
            if initial_states is None
            else initial_states
        )
        while True:
            snapshot = await self._scan_catalog_until_stopped(
                self._catalog,
                states,
            )
            verified_states = await self._load_refresh_states(
                primary_error=primary_enablement_error,
                database_failure_is_unsafe=True,
            )
            if verified_states == states:
                break
            # A sibling process changed a durable guard or database row while
            # the folder scan was in flight. Discard that mixed-time snapshot
            # before it can reach the prompt or graph, and scan again using the
            # newly observed state. Continuous churn can delay a refresh, but
            # it can never publish an unapproved folder generation as enabled.
            states = verified_states
        return await self._publish_snapshot_locked(
            snapshot,
            publish_context=publish_context,
        )

    async def _load_refresh_states(
        self,
        *,
        primary_error: str | None = None,
        database_failure_is_unsafe: bool = False,
    ) -> dict[str, SkillState]:
        """Read database and durable guards for one side of a catalog scan."""

        if self._enablement is None:
            raise RuntimeError("ProceduralSkillsFeature is not initialized")
        durable_state_available = self._load_durable_fail_closed_states()
        try:
            states = await self._enablement.load()
        except DatabaseError as exc:
            states = dict(self._states)
            if database_failure_is_unsafe or not durable_state_available:
                states = {
                    name: SkillState(False, state.priority)
                    for name, state in states.items()
                }
            states = self._with_fail_closed_states(
                states,
                exclude=self._releasing_fail_closed_names.get(),
            )
            database_error = str(exc)
            if primary_error and primary_error != database_error:
                database_error = f"{primary_error}; {database_error}"
            self._enablement_error = self._combined_enablement_error(database_error)
            logger.warning(
                "Could not reload procedural skill enablement; retaining the last known state: %s",
                exc,
            )
        else:
            if durable_state_available and self._enablement.available:
                await self._reap_orphaned_fail_closed_states(states)
            if not durable_state_available:
                states = {
                    name: SkillState(False, state.priority)
                    for name, state in states.items()
                }
            states = self._with_fail_closed_states(
                states,
                exclude=self._releasing_fail_closed_names.get(),
            )
            self._states = dict(states)
            self._enablement_error = self._combined_enablement_error(primary_error)
        return states

    async def _reap_orphaned_fail_closed_states(
        self,
        observed_states: dict[str, SkillState],
    ) -> None:
        """Retire guards whose folder and database row are provably absent."""

        store, enablement = self._require_services()
        candidates = self._durable_fail_closed_names - observed_states.keys()
        for name in sorted(candidates):
            try:
                if store.local_entry_identity(name) is not None:
                    continue
                claim = store.try_acquire_publication_state_claim(name)
            except (OSError, SkillError) as exc:
                logger.warning(
                    "Could not inspect orphaned procedural skill guard %s: %s",
                    name,
                    exc,
                )
                continue
            # A mutation holding this claim may be between its filesystem and
            # database commit points. Defer to a later refresh instead of
            # blocking (or trying to reacquire our own current claim).
            if claim is None:
                continue
            try:
                if store.local_entry_identity(name) is not None:
                    continue
                try:
                    verified_states = await enablement.load()
                except DatabaseError as exc:
                    logger.warning(
                        "Could not verify orphaned procedural skill guard %s: %s",
                        name,
                        exc,
                    )
                    continue
                if name in verified_states:
                    continue
                try:
                    self._clear_fail_closed_state(name)
                except (OSError, SkillError) as exc:
                    # The guard stays fail-closed and another refresh retries.
                    logger.warning(
                        "Could not retire orphaned procedural skill guard %s: %s",
                        name,
                        exc,
                    )
            finally:
                claim.release()

    async def _refresh_committed_mutation(self) -> str | None:
        """Refresh a committed mutation, preserving a publication error as data."""

        try:
            await self._refresh_locked()
        except _ContextPublicationError as exc:
            # Graph reconciliation follows context publication. Its cached
            # success set therefore describes the prior snapshot when this
            # exception occurs and must not be reported for the new revision.
            self._indexed_payloads = None
            self._indexed_names = frozenset()
            return str(exc)
        return None

    def _combined_enablement_error(self, primary: str | None = None) -> str | None:
        messages = [
            message
            for message in (primary, self._durable_fail_closed_load_error)
            if message
        ] + list(self._fail_closed_errors.values())
        return "; ".join(messages) or None

    def _with_fail_closed_states(
        self,
        states: dict[str, SkillState],
        *,
        exclude: frozenset[str] = frozenset(),
    ) -> dict[str, SkillState]:
        guarded = dict(states)
        guarded.update(
            (name, state)
            for name, state in self._fail_closed_states.items()
            if name not in exclude
        )
        return guarded

    def _load_durable_fail_closed_states(self) -> bool:
        """Merge crash-safe quarantine state before interpreting database rows."""

        if self._store is None:
            return False
        try:
            states, errors = self._store.load_fail_closed_states()
        except (OSError, SkillError) as exc:
            message = f"could not load durable skill quarantine state: {exc}"
            self._durable_fail_closed_load_error = message
            self._enablement_error = self._combined_enablement_error()
            logger.error(message)
            return False
        self._durable_fail_closed_load_error = None
        for name in self._durable_fail_closed_names - states.keys():
            self._fail_closed_states.pop(name, None)
            self._fail_closed_errors.pop(name, None)
        self._fail_closed_states.update(states)
        self._fail_closed_errors.update(errors)
        self._durable_fail_closed_names = set(states)
        return True

    def _retain_fail_closed_state(
        self,
        name: str,
        *,
        priority: int,
        error: str,
    ) -> None:
        state = SkillState(False, priority)
        self._fail_closed_states[name] = state
        self._fail_closed_errors[name] = error
        self._states[name] = state
        self._enablement_error = self._combined_enablement_error()
        if self._store is None:
            raise RuntimeError("ProceduralSkillsFeature is not initialized")
        try:
            self._store.retain_fail_closed_state(
                name,
                priority=priority,
                error=error,
            )
            self._durable_fail_closed_names.add(name)
        except (OSError, SkillError) as marker_error:
            message = (
                f"{error}; durable fail-closed marker could not be persisted "
                f"({marker_error})"
            )
            self._fail_closed_errors[name] = message
            self._enablement_error = self._combined_enablement_error()
            raise SkillPublicationCleanupError(message) from marker_error

    def _clear_fail_closed_state(self, name: str) -> None:
        if self._store is not None:
            self._store.clear_fail_closed_state(name)
        self._durable_fail_closed_names.discard(name)
        self._fail_closed_states.pop(name, None)
        self._fail_closed_errors.pop(name, None)
        self._enablement_error = self._combined_enablement_error()

    @staticmethod
    async def _scan_catalog_until_stopped(
        catalog: SkillCatalog,
        states: dict[str, SkillState],
    ) -> CatalogSnapshot:
        """Keep cancellation scoped until the owned catalog scan has stopped."""

        worker = asyncio.create_task(asyncio.to_thread(catalog.refresh, states))
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            while True:
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    continue
                except Exception:  # noqa: BLE001 - retrieve worker failure first
                    break
                else:
                    break
            raise

    async def _publish_snapshot_locked(
        self,
        snapshot: CatalogSnapshot,
        *,
        publish_context: bool = True,
    ) -> CatalogSnapshot:
        """Publish one state-stable catalog to the prompt and graph index."""

        self._snapshot = snapshot
        replacement = render_context_clause(
            self._snapshot,
            max_bytes=DEFAULT_CONTEXT_BUDGET_BYTES,
        )
        if publish_context:
            self._replace_context_render(replacement)
        else:
            self._context_render = replacement
            self._context_publication_uncertain = True
        records_by_name = {record.name: record for record in self._snapshot.records}
        desired_payloads = {
            name: self._index_payload(self._index_node(record))
            for name, record in records_by_name.items()
        }
        # The graph is a recoverable secondary index and may be repaired by a
        # different process. Reconcile against one fresh, agent-scoped batch
        # instead of trusting the in-memory cache or issuing one query per skill.
        indexed_payloads = await self._persisted_index_payloads()
        for stale_name in sorted(indexed_payloads.keys() - records_by_name.keys()):
            if await self._delete_index_node(stale_name):
                indexed_payloads.pop(stale_name, None)
        for record in self._snapshot.records:
            desired_payload = desired_payloads[record.name]
            if indexed_payloads.get(record.name) == desired_payload:
                continue
            if await self._index_record(record):
                indexed_payloads[record.name] = desired_payload
        self._indexed_payloads = indexed_payloads
        self._indexed_names = frozenset(
            name
            for name, payload in indexed_payloads.items()
            if desired_payloads.get(name) == payload
        )
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

    def _hide_persistent_state(self, *, publish_context: bool = True) -> None:
        self._db = None
        self._enablement = None
        self._store = None
        self._catalog = None
        self._snapshot = CatalogSnapshot()
        replacement = render_context_clause(self._snapshot)
        if publish_context:
            self._replace_context_render(replacement)
        else:
            self._context_render = replacement
            self._context_publication_uncertain = True
        self._states = {}
        self._durable_fail_closed_load_error = None
        self._enablement_error = (
            "persistent procedural skills are unavailable in the current privacy mode"
        )
        self._indexed_names = frozenset()
        self._indexed_payloads = None

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
        deletion_record = _deletion_record(self._snapshot, record)
        deletable = deletion_record.editable
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
            "deletable": deletable,
            "revision": record.revision,
            "delete_revision": deletion_record.revision if deletable else None,
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
        self._load_durable_fail_closed_states()
        preserve_fail_closed = name in self._fail_closed_states
        previous_state: SkillState | None = None
        disabled_state: SkillState | None = None
        state_was_persisted = False
        prior_local_identity = store.local_entry_identity(name)
        if prior_local_identity is not None:
            raise SkillConflictError(f"skill already exists: {name}")
        if not enablement.available:
            if not preserve_fail_closed:
                self._retain_fail_closed_state(
                    name,
                    priority=DEFAULT_PRIORITY,
                    error=(
                        "agent database unavailable; persisted skill state could not "
                        "be verified, so the new skill remains disabled until an "
                        "explicit state update succeeds"
                    ),
                )
            preserve_fail_closed = True
        if enablement.available:
            previous_state, disabled_state = await self._prepare_disabled_state(
                enablement,
                document.name,
                priority=resolved_priority,
                operation="skill create disabled-state preparation",
            )
            self._states[document.name] = disabled_state
            state_was_persisted = True
        try:
            folder, created_identity = store.create_pinned(document)
        except Exception as publication_error:
            if state_was_persisted:
                restore_state = (
                    disabled_state
                    if self._publication_failure_requires_disabled_state(
                        store,
                        name,
                        publication_error,
                        prior_local_identity=prior_local_identity,
                    )
                    else previous_state
                )
                await self._restore_enablement_after_publication_failure(
                    enablement,
                    document.name,
                    restore_state,
                    publication_error,
                    operation="skill create",
                    preserve_fail_closed=preserve_fail_closed,
                )
            raise
        state_error: str | None = None
        if enablement.available and enabled:
            if not preserve_fail_closed:
                self._retain_fail_closed_state(
                    name,
                    priority=resolved_priority,
                    error=(
                        "skill create enablement is being finalized; the skill remains "
                        "disabled until its published generation is revalidated"
                    ),
                )
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
                        disabled_state=SkillState(False, resolved_priority),
                        publication_error=load_error,
                        preserve_fail_closed=preserve_fail_closed,
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
                        disabled_state=SkillState(False, resolved_priority),
                        publication_error=exc,
                        preserve_fail_closed=preserve_fail_closed,
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
        elif not enablement.available:
            state_error = (
                "agent database unavailable; persisted skill state could not be "
                "verified; the new skill remains disabled until an explicit state "
                "update succeeds"
            )
        retain_existing_fail_closed = preserve_fail_closed and not state_was_persisted
        try:
            releasing_names = (
                frozenset() if retain_existing_fail_closed else frozenset((name,))
            )
            release_token = self._releasing_fail_closed_names.set(releasing_names)
            try:
                await self._refresh_locked()
            finally:
                self._releasing_fail_closed_names.reset(release_token)
            record = SkillStore.get(self._snapshot, name)
            store.assert_created_current(folder, identity=created_identity)
            if not retain_existing_fail_closed:
                self._clear_fail_closed_state(name)
        except BaseException as consistency_error:
            await self._rollback_created_publication(
                store,
                enablement,
                folder,
                identity=created_identity,
                previous_state=previous_state,
                disabled_state=SkillState(False, resolved_priority),
                publication_error=consistency_error,
                preserve_fail_closed=preserve_fail_closed,
            )
            await self._refresh_locked()
            raise
        return {
            "name": name,
            "folder": str(folder),
            "enabled": record.state.enabled,
            "priority": record.state.priority,
            "indexed": name in self._indexed_names,
            "state_error": state_error,
            "revision": record.revision,
        }

    @staticmethod
    def _publication_failure_requires_disabled_state(
        store: SkillStore,
        name: str,
        publication_error: BaseException,
        *,
        prior_local_identity: tuple[int, int] | None,
    ) -> bool:
        """Retain the guard whenever failed publication may have a replacement."""

        if isinstance(publication_error, SkillPublicationCleanupError):
            return True
        try:
            current_identity = store.local_entry_identity(name)
        except (OSError, SkillError) as inspection_error:
            publication_error.add_note(
                "the same-name local entry could not be inspected after publication "
                f"failed: {inspection_error}"
            )
            return True
        return current_identity is not None and current_identity != prior_local_identity

    async def _rollback_created_publication(
        self,
        store: SkillStore,
        enablement: SkillEnablementStore,
        folder: Path,
        *,
        identity: CreatedSkillPublication,
        previous_state: SkillState | None,
        disabled_state: SkillState,
        publication_error: BaseException,
        preserve_fail_closed: bool = False,
    ) -> None:
        """Restore prior state only after inode-pinned filesystem rollback succeeds."""

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
                disabled_state if rollback_error is not None else previous_state,
                publication_error,
                operation="skill create state update",
                preserve_fail_closed=preserve_fail_closed,
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
        preserve_fail_closed: bool = False,
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
            current = self._states.get(name)
            priority = (
                previous_state.priority
                if previous_state is not None
                else (current.priority if current is not None else DEFAULT_PRIORITY)
            )
            self._retain_fail_closed_state(
                name,
                priority=priority,
                error=message,
            )
            raise DatabaseError(message) from rollback_error
        if not preserve_fail_closed:
            self._clear_fail_closed_state(name)

    async def _prepare_disabled_state(
        self,
        enablement: SkillEnablementStore,
        name: str,
        *,
        priority: int,
        operation: str,
    ) -> tuple[SkillState | None, SkillState]:
        """Persist a disabled publication guard without losing prior state."""

        preserve_fail_closed = name in self._fail_closed_states
        if not preserve_fail_closed:
            # This synchronous durable marker is the first side effect under the
            # same-name claim. Database reads and writes await external work, while
            # direct filesystem publishers do not honor that claim; sibling
            # feature instances must therefore see the name disabled before the
            # first state await can yield to such a replacement.
            self._retain_fail_closed_state(
                name,
                priority=priority,
                error=(
                    f"{operation} is being finalized; the skill remains disabled "
                    "until its filesystem generation and database row are verified"
                ),
            )
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
                    # Even a successful row restoration cannot prove that a
                    # non-cooperating filesystem writer did not replace this name
                    # while either state await was in flight. Keep the durable
                    # fence until an explicit successful publication/state
                    # operation validates the current generation.
                    preserve_fail_closed=True,
                )
            )
            await self._drain_shielded_task(restoration)
            raise
        return previous_state, state

    async def edit_skill(
        self,
        *,
        name: str,
        relative_path: str,
        content: str,
        expected_revision: str | None = None,
    ) -> dict[str, object]:
        async with self._persistent_mutation():
            return await self._edit_skill_locked(
                name=name,
                relative_path=relative_path,
                content=content,
                expected_revision=expected_revision,
            )

    async def _edit_skill_locked(
        self,
        *,
        name: str,
        relative_path: str,
        content: str,
        expected_revision: str | None = None,
    ) -> dict[str, object]:
        store, _ = self._require_services()
        name = validate_skill_name(name)
        cached = (
            SkillStore.get(self._snapshot, name) if expected_revision is None else None
        )
        async with self._publication_state_claim(store, name):
            await self._refresh_locked()
            record = self._snapshot.by_name().get(name)
            if record is None:
                raise SkillFormatError(
                    f"skill {name!r} changed or became invalid before editing; "
                    "reload and repair its folder"
                )
            if cached is not None and (
                not _same_resolved_folder(cached, record)
                or cached.revision != record.revision
            ):
                raise SkillConflictError(
                    f"resolved source changed before editing {name}; reload and retry"
                )
            _require_expected_revision(
                record,
                expected_revision,
                operation="editing",
            )
            await self._write_skill_file_until_stopped(
                store,
                record,
                relative_path,
                content,
            )
            refresh_error = await self._refresh_committed_mutation()
            resolved = self._snapshot.by_name().get(name)
            if resolved is None:
                raise SkillFormatError(
                    f"skill {name!r} changed or became invalid after editing; "
                    "reload and repair its folder"
                )
            if resolved.folder != record.folder or not resolved.editable:
                raise SkillConflictError(
                    f"resolved source changed after editing {name}; reload before "
                    "making another change"
                )
        return {
            "name": name,
            "path": relative_path,
            "indexed": name in self._indexed_names,
            "revision": resolved.revision,
            "refresh_error": refresh_error,
        }

    async def _write_skill_file_until_stopped(
        self,
        store: SkillStore,
        record: SkillRecord,
        relative_path: str,
        content: str,
    ) -> None:
        """Keep claims and the privacy lock until blocking storage work stops."""

        worker = asyncio.create_task(
            asyncio.to_thread(
                store.write_file,
                record,
                relative_path,
                content,
            )
        )
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            while True:
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    continue
                except Exception:  # noqa: BLE001 - preserve caller cancellation
                    break
                else:
                    break
            raise

    async def set_skill_state(
        self,
        *,
        name: str,
        enabled: bool,
        priority: int | None = None,
        expected_revision: str | None = None,
    ) -> dict[str, object]:
        async with self._persistent_mutation():
            return await self._set_skill_state_locked(
                name=name,
                enabled=enabled,
                priority=priority,
                expected_revision=expected_revision,
            )

    async def _set_skill_state_locked(
        self,
        *,
        name: str,
        enabled: bool,
        priority: int | None = None,
        expected_revision: str | None = None,
    ) -> dict[str, object]:
        store, enablement = self._require_services()
        name = validate_skill_name(name)
        # Reject arbitrary names before the filesystem-backed claim allocator.
        # Refresh first so skills published by another process remain eligible;
        # the second refresh under the claim still closes publication races.
        await self._refresh_locked()
        SkillStore.get(self._snapshot, name)
        async with self._publication_state_claim(store, name):
            # A publication or deletion in another feature instance may have
            # completed while this caller waited. Resolve the state mutation
            # against the catalog version protected by the same name claim.
            await self._refresh_locked()
            return await self._set_skill_state_with_publication_claim(
                store=store,
                enablement=enablement,
                name=name,
                enabled=enabled,
                priority=priority,
                expected_revision=expected_revision,
            )

    async def _set_skill_state_with_publication_claim(
        self,
        *,
        store: SkillStore,
        enablement: SkillEnablementStore,
        name: str,
        enabled: bool,
        priority: int | None,
        expected_revision: str | None,
    ) -> dict[str, object]:
        record = SkillStore.get(self._snapshot, name)
        _require_expected_revision(
            record,
            expected_revision,
            operation="changing state",
        )
        resolved_priority = (
            record.state.priority if priority is None else validate_priority(priority)
        )
        store.assert_current(record)
        previous_state_present = name in self._states
        previous_state = self._states.get(name, record.state)
        self._retain_fail_closed_state(
            name,
            priority=previous_state.priority,
            error=(
                "skill state update is being finalized; the skill remains disabled "
                "until its folder generation is revalidated"
            ),
        )
        try:
            state = await enablement.set(
                name, enabled=enabled, priority=resolved_priority
            )
        except DatabaseError as exc:
            self._enablement_error = str(exc)
            observed_states = await enablement.load()
            observed = observed_states.get(name)
            if observed is None or not observed.enabled:
                self._clear_fail_closed_state(name)
            self._states = self._with_fail_closed_states(observed_states)
            self._enablement_error = self._combined_enablement_error(str(exc))
            await self._refresh_locked(
                initial_states=self._states,
                primary_enablement_error=str(exc),
            )
            raise
        try:
            store.assert_current(record)
        except BaseException as consistency_error:
            await self._rollback_state_after_consistency_error(
                enablement=enablement,
                name=name,
                previous_state=(previous_state if previous_state_present else None),
                consistency_error=consistency_error,
            )
            raise
        self._states[name] = state
        finalization = asyncio.create_task(
            self._finalize_skill_state_change(
                store=store,
                enablement=enablement,
                name=name,
                record=record,
                previous_state=(previous_state if previous_state_present else None),
            )
        )
        try:
            refreshed = await asyncio.shield(finalization)
        except asyncio.CancelledError as cancellation:
            # State persistence already committed. Keep the independently owned
            # generation check and any required rollback inside the name claim,
            # even when cancellation lands on the final refresh await.
            try:
                await self._drain_shielded_task(finalization)
            except BaseException as finalization_error:  # noqa: BLE001 - drain outcome
                cancellation.add_note(
                    "cancelled skill state finalization also reported: "
                    f"{finalization_error}"
                )
            raise
        return {
            "name": name,
            "enabled": refreshed.state.enabled,
            "priority": refreshed.state.priority,
            "indexed": name in self._indexed_names,
            "context_bytes": len(self._context_render.text.encode("utf-8")),
            "revision": refreshed.revision,
        }

    async def _finalize_skill_state_change(
        self,
        *,
        store: SkillStore,
        enablement: SkillEnablementStore,
        name: str,
        record: SkillRecord,
        previous_state: SkillState | None,
    ) -> SkillRecord:
        """Refresh and validate one committed state write before releasing its claim."""

        try:
            release_token = self._releasing_fail_closed_names.set(frozenset((name,)))
            try:
                await self._refresh_locked()
            finally:
                self._releasing_fail_closed_names.reset(release_token)
            refreshed = SkillStore.get(self._snapshot, name)
            if not _same_resolved_folder(record, refreshed):
                raise SkillConflictError(
                    f"resolved source changed after changing state for {name}; "
                    "reload and retry"
                )
            # The final refresh awaits database and catalog work. Revalidate
            # the original filesystem snapshot after that await so an edit or
            # removal during the refresh cannot leave the requested state on a
            # different or future same-named generation.
            store.assert_current(record)
            # The database row, refreshed catalog, and original folder are now
            # mutually consistent. This synchronous unlink is the commit point
            # that lets sibling processes observe the approved enabled state.
            self._clear_fail_closed_state(name)
        except BaseException as consistency_error:
            await self._rollback_state_after_consistency_error(
                enablement=enablement,
                name=name,
                previous_state=previous_state,
                consistency_error=consistency_error,
            )
            raise
        return refreshed

    async def _rollback_state_after_consistency_error(
        self,
        *,
        enablement: SkillEnablementStore,
        name: str,
        previous_state: SkillState | None,
        consistency_error: BaseException,
    ) -> None:
        """Restore only a fail-closed prior row after a generation race."""

        safe_state = (
            None
            if previous_state is None
            else SkillState(False, previous_state.priority)
        )
        try:
            await self._restore_enablement_after_publication_failure(
                enablement,
                name,
                safe_state,
                consistency_error,
                operation="skill state update",
            )
        except DatabaseError as rollback_error:
            consistency_error.add_note(
                "state rollback after a concurrent skill-folder change failed: "
                f"{rollback_error}"
            )
        await self._refresh_locked()

    async def delete_skill(
        self,
        *,
        name: str,
        expected_revision: str | None = None,
    ) -> dict[str, object]:
        async with self._persistent_mutation():
            return await self._delete_skill_locked(
                name=name,
                expected_revision=expected_revision,
            )

    async def _delete_skill_locked(
        self,
        *,
        name: str,
        expected_revision: str | None = None,
    ) -> dict[str, object]:
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
                expected_revision=expected_revision,
            )

    async def _delete_skill_with_publication_claim(
        self,
        *,
        store: SkillStore,
        enablement: SkillEnablementStore,
        name: str,
        expected_revision: str | None,
    ) -> dict[str, object]:
        resolved_record = SkillStore.get(self._snapshot, name)
        record = _deletion_record(self._snapshot, resolved_record)
        deleting_shadowed_local = record is not resolved_record
        store.require_local_record(record)
        _require_expected_revision(
            record,
            expected_revision,
            operation="deletion",
        )
        if deleting_shadowed_local:
            store.delete(record)
            refresh_error = await self._refresh_committed_mutation()
            remaining = self._snapshot.by_name().get(name)
            graph_storage = getattr(self.agent, "storage", None)
            graph_deleted = graph_storage is None or remaining is None
            graph_retained = (
                graph_storage is not None
                and remaining is not None
                and name in self._indexed_names
            )
            errors: list[str] = []
            if refresh_error is not None:
                errors.append(refresh_error)
            if remaining is None and graph_storage is not None:
                # Refresh already attempted the idempotent stale-node removal;
                # retry once so its recoverable failure is reflected in this
                # mutation result just as it is for an unshadowed deletion.
                graph_deleted = await self._delete_index_node(name)
            if not graph_deleted and not graph_retained:
                errors.append("graph index cleanup failed")
            return {
                "name": name,
                "removed_file": True,
                "config_deleted": False,
                "config_retained": True,
                "graph_deleted": graph_deleted,
                "graph_retained": graph_retained,
                "errors": errors,
                "deleted_source_kind": record.source_kind,
                "resolved_skill_retained": remaining is not None,
                "remaining_source_kind": (
                    remaining.source_kind if remaining is not None else None
                ),
            }
        guard_retained = False
        if enablement.available:
            _previous_state, disabled_state = await self._prepare_disabled_state(
                enablement,
                name,
                priority=record.state.priority,
                operation="skill delete disabled-state preparation",
            )
            self._states[name] = disabled_state
            guard_retained = True
        else:
            if name not in self._fail_closed_states:
                self._retain_fail_closed_state(
                    name,
                    priority=record.state.priority,
                    error=(
                        "agent database unavailable during skill deletion; "
                        "enablement configuration was retained and the name remains "
                        "disabled until an explicit state update succeeds"
                    ),
                )
            guard_retained = True
        try:
            store.delete(record)
        except BaseException as deletion_error:
            # A failed recursive removal can leave the original only under a
            # quarantine name. Retaining the disabled tombstone is the only
            # fail-safe result when folder cleanup cannot be confirmed.
            if guard_retained:
                caveat = "the skill was disabled before deletion and remains disabled"
                deletion_error.add_note(caveat)
                if isinstance(deletion_error, Exception):
                    raise SkillDeletionError(
                        f"skill folder deletion failed ({deletion_error}); {caveat}"
                    ) from deletion_error
            raise
        config_deleted = False
        graph_deleted = True
        errors: list[str] = []
        enablement_cleanup_error: str | None = None
        enablement_cleanup_observed_absent = False
        if enablement.available:
            try:
                await enablement.delete(name)
                self._states.pop(name, None)
                config_deleted = True
            except DatabaseError as exc:
                try:
                    observed_states = await enablement.load()
                except DatabaseError as reconciliation_error:
                    enablement_cleanup_error = (
                        "enablement row cleanup failed and could not be reconciled: "
                        f"{exc}; reconciliation failed: {reconciliation_error}"
                    )
                else:
                    self._states = dict(observed_states)
                    if name in observed_states:
                        enablement_cleanup_error = (
                            f"enablement row cleanup failed: {exc}"
                        )
                    else:
                        enablement_cleanup_observed_absent = True
                        config_deleted = True
                        self._enablement_error = self._combined_enablement_error()
        else:
            enablement_cleanup_error = (
                "agent database unavailable; enablement configuration was retained"
            )
        if getattr(self.agent, "storage", None) is not None:
            graph_deleted = await self._delete_index_node(name)
        guard_cleanup_pending = False
        if config_deleted:
            # The authoritative folder and database row are already absent.
            # Release the durable guard before context publication so a host
            # refresh failure cannot strand a tombstone with no retry target.
            try:
                self._clear_fail_closed_state(name)
            except (OSError, SkillError):
                # Continue through reconciliation and retry below. The delete
                # is already committed and must never be reported as uncommitted.
                guard_cleanup_pending = True
        refresh_error = await self._refresh_committed_mutation()
        if refresh_error is not None:
            errors.append(refresh_error)
        remaining = self._snapshot.by_name().get(name)
        graph_retained = remaining is not None and name in self._indexed_names
        if (
            not graph_deleted
            and remaining is None
            and getattr(self.agent, "storage", None) is not None
        ):
            # The refresh performs the same idempotent stale-index cleanup. Ask
            # once more after it completes so a transient first failure cannot
            # leave the result claiming cleanup is incomplete after it succeeded.
            graph_deleted = await self._delete_index_node(name)
        if not graph_deleted and not graph_retained:
            errors.append("graph index cleanup failed")
        if enablement_cleanup_error is not None:
            # The final refresh also loads durable quarantine markers, so its
            # aggregate enablement error/state cannot distinguish a healthy
            # absent database row from the intentional fail-closed tombstone.
            # Re-read the raw row while the same-name claim is still held and
            # release the marker only when absence is directly observed.
            final_refresh_observed_absent = False
            if enablement.available:
                try:
                    final_observed_states = await enablement.load()
                except DatabaseError:
                    pass
                else:
                    final_refresh_observed_absent = name not in final_observed_states
            if not (
                enablement_cleanup_observed_absent or final_refresh_observed_absent
            ):
                config_deleted = False
                errors.insert(0, enablement_cleanup_error)
            else:
                config_deleted = True
        if config_deleted and (
            guard_cleanup_pending or name in self._fail_closed_states
        ):
            try:
                self._clear_fail_closed_state(name)
            except (OSError, SkillError):
                errors.append("durable fail-closed marker cleanup remains incomplete")
        return {
            "name": name,
            "removed_file": True,
            "config_deleted": config_deleted,
            "config_retained": not config_deleted,
            "graph_deleted": graph_deleted,
            "graph_retained": graph_retained,
            "errors": errors,
            "deleted_source_kind": record.source_kind,
            "resolved_skill_retained": remaining is not None,
            "remaining_source_kind": (
                remaining.source_kind if remaining is not None else None
            ),
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
        # The cached snapshot may still contain a folder that was removed by an
        # external actor. Refresh before the fast conflict check so an available
        # name does not require a separate user-triggered reload.
        await self._refresh_locked()
        if skill_name in self._snapshot.by_name():
            raise SkillConflictError(
                f"skill already exists in the resolved catalog: {skill_name}"
            )
        operation: _InstalledSkillOperation | None = None
        async with self._publication_state_claim(store, skill_name):
            try:
                with store.git_checkout_workspace() as temporary:
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
                    # Checkout can be slow enough for a host-shared skill with the
                    # same name to appear after the initial catalog check. Refresh
                    # while holding the publication claim so that local install
                    # cannot silently shadow the newly resolved source.
                    await self._refresh_locked()
                    if skill_name in self._snapshot.by_name():
                        raise SkillConflictError(
                            f"skill already exists in the resolved catalog: {skill_name}"
                        )
                    operation = await self._publish_installed_skill_with_state_claim(
                        store=store,
                        enablement=enablement,
                        source_folder=checkout.skill_folder,
                        provenance=provenance,
                        skill_name=skill_name,
                    )
                    await self._refresh_locked()
                    record = SkillStore.get(self._snapshot, skill_name)
                    if not (
                        record.folder == operation.folder
                        and record.folder_identity
                        == operation.publication.folder_identity
                        and record.provenance == provenance
                    ):
                        raise SkillConflictError(
                            "resolved source changed during installation of "
                            f"{skill_name}; the shadowed local publication was "
                            "rolled back"
                        )
                    if not (
                        operation.preserve_fail_closed
                        and not operation.state_was_persisted
                    ):
                        self._clear_fail_closed_state(skill_name)
            except BaseException as installation_error:
                # Context-manager exit is part of installation finalization. Keep
                # the same-name claim through compensation so another feature
                # instance cannot mutate the publication that is being rolled back.
                if operation is not None:
                    await self._rollback_installed_publication(
                        store=store,
                        enablement=enablement,
                        operation=operation,
                        publication_error=installation_error,
                    )
                raise
        assert operation is not None
        return {
            "name": skill_name,
            "folder": str(operation.folder),
            "revision": record.revision,
            "source_revision": checkout.revision,
            "remote_url": checkout.remote_url,
            "enabled": record.state.enabled,
            "indexed": skill_name in self._indexed_names,
            "state_error": (
                None
                if operation.state_was_persisted
                else (
                    "agent database unavailable; persisted skill state could not be "
                    "verified; the installed skill remains disabled until an explicit "
                    "state update succeeds"
                )
            ),
        }

    async def _publish_installed_skill_with_state_claim(
        self,
        *,
        store: SkillStore,
        enablement: SkillEnablementStore,
        source_folder: Path,
        provenance: SkillProvenance,
        skill_name: str,
    ) -> _InstalledSkillOperation:
        self._load_durable_fail_closed_states()
        preserve_fail_closed = skill_name in self._fail_closed_states
        previous_state: SkillState | None = None
        disabled_state: SkillState | None = None
        state_was_persisted = False
        prior_local_identity = store.local_entry_identity(skill_name)
        if prior_local_identity is not None:
            raise SkillConflictError(f"skill already exists: {skill_name}")
        if not enablement.available:
            if not preserve_fail_closed:
                self._retain_fail_closed_state(
                    skill_name,
                    priority=DEFAULT_PRIORITY,
                    error=(
                        "agent database unavailable; persisted skill state could not "
                        "be verified, so the installed skill remains disabled until "
                        "an explicit state update succeeds"
                    ),
                )
            preserve_fail_closed = True
        if enablement.available:
            previous_state, disabled_state = await self._prepare_disabled_state(
                enablement,
                skill_name,
                priority=DEFAULT_PRIORITY,
                operation="skill install disabled-state preparation",
            )
            self._states[skill_name] = disabled_state
            state_was_persisted = True
        try:
            folder, publication = store.install_folder_pinned(
                source_folder,
                provenance=provenance,
            )
        except Exception as publication_error:
            if state_was_persisted:
                restore_state = (
                    disabled_state
                    if self._publication_failure_requires_disabled_state(
                        store,
                        skill_name,
                        publication_error,
                        prior_local_identity=prior_local_identity,
                    )
                    else previous_state
                )
                await self._restore_enablement_after_publication_failure(
                    enablement,
                    skill_name,
                    restore_state,
                    publication_error,
                    operation="skill install publication",
                    preserve_fail_closed=preserve_fail_closed,
                )
            raise
        return _InstalledSkillOperation(
            folder=folder,
            publication=publication,
            previous_state=previous_state,
            disabled_state=disabled_state,
            state_was_persisted=state_was_persisted,
            preserve_fail_closed=preserve_fail_closed,
        )

    async def _rollback_installed_publication(
        self,
        *,
        store: SkillStore,
        enablement: SkillEnablementStore,
        operation: _InstalledSkillOperation,
        publication_error: BaseException,
    ) -> None:
        """Remove an unchanged shadowed install and restore its prior state."""

        rollback_error: BaseException | None = None
        try:
            store.rollback_installed(
                operation.folder,
                identity=operation.publication,
            )
        except BaseException as exc:  # noqa: BLE001 - state must remain disabled
            rollback_error = exc
            exc.add_note(
                f"skill install finalization originally failed: {publication_error}"
            )
        if operation.state_was_persisted:
            restoration = asyncio.create_task(
                self._restore_enablement_after_publication_failure(
                    enablement,
                    operation.folder.name,
                    (
                        operation.disabled_state
                        if rollback_error is not None
                        else operation.previous_state
                    ),
                    publication_error,
                    operation="skill install finalization",
                    preserve_fail_closed=operation.preserve_fail_closed,
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
            raise SkillPublicationCleanupError(
                "skill install finalization cleanup could not confirm removal: "
                f"{rollback_error}"
            ) from rollback_error

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
            "revision": record.revision,
        }

    def read_file(self, *, name: str, relative_path: str) -> dict[str, object]:
        store, _ = self._require_services()
        record = SkillStore.get(self._snapshot, name)
        content = store.read_file(record, relative_path)
        execution_risk = has_python_execution_risk(relative_path)
        return {
            "name": name,
            "path": relative_path,
            "content": content,
            "editable": store.file_is_editable(record, relative_path),
            "language": "python" if execution_risk else "markdown",
            "execution_risk": execution_risk,
            "revision": record.revision,
        }

    def tree(self, *, name: str) -> tuple[dict[str, object], ...]:
        store, _ = self._require_services()
        return store.tree(SkillStore.get(self._snapshot, name))

    async def _index_record(self, record: SkillRecord) -> bool:
        storage = getattr(self.agent, "storage", None)
        if (
            storage is None
            or not hasattr(storage, "get_node")
            or not hasattr(storage, "compare_and_swap_node")
        ):
            return False
        node = self._index_node(record)
        try:
            existing = await storage.get_node(node.node_id)
            if existing is not None and (
                getattr(existing, "node_type", None) != PROCEDURAL_SKILL_NODE_TYPE
                or getattr(existing, "label", None) != record.name
            ):
                logger.warning(
                    "Refusing to overwrite non-matching graph node at expected "
                    "skill index id %s",
                    node.node_id,
                )
                return False
            if existing is not None and self._valid_index_created_at(
                existing.properties
            ):
                node = self._index_node(
                    record,
                    created_at=existing.properties["created_at"],
                )
            expected = (
                dict(existing.properties)
                if existing is not None and isinstance(existing.properties, dict)
                else None
            )
            outcome = await storage.compare_and_swap_node(
                node.node_id,
                expected,
                node,
                expected_node_type=PROCEDURAL_SKILL_NODE_TYPE,
                expected_label=record.name,
            )
            if outcome != NodeSwapResult.SWAPPED:
                logger.warning(
                    "Could not conditionally update procedural_skill index for %s: %s",
                    record.name,
                    outcome,
                )
                return False
            persisted = await storage.get_node(node.node_id)
            if persisted is None or self._index_payload(
                persisted
            ) != self._index_payload(node):
                logger.warning(
                    "Procedural_skill index for %s changed during verification",
                    record.name,
                )
                return False
        except Exception as exc:  # noqa: BLE001 - graph is a recoverable index
            logger.warning(
                "Could not update procedural_skill index for %s: %s", record.name, exc
            )
            return False
        return True

    def _index_node(
        self,
        record: SkillRecord,
        *,
        created_at: str | None = None,
    ) -> GraphNode:
        return GraphNode(
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
                # Core's scoped EPHEMERAL leak purge requires every owned graph
                # node to carry a timezone-qualified creation time. Preserve
                # this value on later index refreshes so ordinary catalog/state
                # changes cannot make a pre-stint node look newly created.
                "created_at": created_at or datetime.now(UTC).isoformat(),
            },
        )

    @staticmethod
    def _valid_index_created_at(properties: object) -> bool:
        if not isinstance(properties, dict):
            return False
        value = properties.get("created_at")
        if not isinstance(value, str) or len(value) > 64:
            return False
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return False
        return parsed.tzinfo is not None

    @staticmethod
    def _index_payload(node: GraphNode) -> str:
        """Return a type-preserving canonical payload for graph change detection."""

        properties = dict(node.properties)
        # Creation provenance is deliberately stable metadata rather than
        # catalog content. Ignoring its exact value here lets desired payloads
        # be computed without inventing a new timestamp on every refresh; the
        # persisted-node scan separately rejects missing/invalid timestamps so
        # legacy untimed rows are still repaired.
        properties.pop("created_at", None)

        return json.dumps(
            {
                "node_id": node.node_id,
                "node_type": node.node_type,
                "label": node.label,
                "properties": properties,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    async def _delete_index_node(self, name: str) -> bool:
        """Best-effort removal for a catalog entry no longer on disk."""

        storage = getattr(self.agent, "storage", None)
        if (
            storage is None
            or not hasattr(storage, "get_node")
            or not hasattr(storage, "compare_and_delete_node")
        ):
            return False
        node_id = self._node_id(name)
        try:
            node = await storage.get_node(node_id)
            if node is None:
                return True
            if (
                getattr(node, "node_type", None) != PROCEDURAL_SKILL_NODE_TYPE
                or getattr(node, "label", None) != name
            ):
                logger.warning(
                    "Refusing to delete non-matching node at expected skill index id %s",
                    node_id,
                )
                return True
            outcome = await storage.compare_and_delete_node(
                node_id,
                expected_node_type=PROCEDURAL_SKILL_NODE_TYPE,
                expected_label=name,
            )
            if outcome == NodeDeleteResult.PREDICATE_FAILED:
                logger.warning(
                    "Refusing to delete graph node whose identity changed at "
                    "expected skill index id %s",
                    node_id,
                )
                return True
            if outcome not in {
                NodeDeleteResult.DELETED,
                NodeDeleteResult.NOT_FOUND,
            }:
                logger.warning(
                    "Could not conditionally remove procedural_skill index for %s: %s",
                    name,
                    outcome,
                )
                return False
        except Exception as exc:  # noqa: BLE001 - graph is a recoverable index
            logger.warning(
                "Could not remove stale procedural_skill index for %s: %s", name, exc
            )
            return False
        return True

    async def _persisted_index_payloads(self) -> dict[str, str]:
        """Load this agent's prior graph payloads so restart skips unchanged nodes."""

        storage = getattr(self.agent, "storage", None)
        if storage is None or not hasattr(storage, "get_nodes_by_type"):
            return {}
        try:
            nodes = await storage.get_nodes_by_type(PROCEDURAL_SKILL_NODE_TYPE)
        except Exception as exc:  # noqa: BLE001 - graph is a recoverable index
            logger.warning("Could not enumerate procedural_skill index nodes: %s", exc)
            return {}
        payloads: dict[str, str] = {}
        for node in nodes:
            properties = getattr(node, "properties", None)
            name = properties.get("name") if isinstance(properties, dict) else None
            try:
                name = validate_skill_name(name)
            except SkillError:
                continue
            if getattr(node, "node_id", None) == self._node_id(name):
                if not self._valid_index_created_at(properties):
                    # Retain the validated ownership key while forcing a
                    # mismatch. A live legacy row is repaired below; a stale
                    # row whose folder disappeared is routed through guarded
                    # deletion instead of surviving every restart.
                    payloads[name] = ""
                    continue
                try:
                    payloads[name] = self._index_payload(node)
                except (AttributeError, TypeError, ValueError):
                    continue
        return payloads

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
            async with self.persistent_read(refresh=True):
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
        refresh_error = payload["refresh_error"]
        if refresh_error:
            return ToolResult.partial(
                f"Edited {name}/{relative_path} on disk.",
                str(refresh_error),
                data=payload,
            )
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
        except (SkillError, OSError, DatabaseError, RuntimeError, ValueError) as exc:
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
        except (SkillError, OSError, DatabaseError, RuntimeError, ValueError) as exc:
            return ToolResult.failed(str(exc))
        return ToolResult.ok(f"Disabled procedural skill {name}.", data=payload)

    @tool(
        "skill_delete",
        "Permanently delete the exact local skill generation identified by a "
        "delete_revision from skill_list or skill_search",
        category=ToolCategory.UTILITY,
        command_prefix="!skill delete",
    )
    async def skill_delete(self, name: str, delete_revision: str) -> ToolResult:
        """Delete one approved skill generation.

        Args:
            name: Name of the local skill to delete.
            delete_revision: Exact deletion token returned by skill_list or
                skill_search before approval was requested.
        """

        try:
            payload = await self.delete_skill(
                name=name,
                expected_revision=delete_revision,
            )
        except (SkillError, OSError, DatabaseError, RuntimeError) as exc:
            return ToolResult.failed(str(exc))
        errors = payload["errors"]
        if errors:
            return ToolResult.partial(
                f"Deleted the authoritative skill folder for {name}.",
                "; ".join(errors),
                data=payload,
            )
        if payload["resolved_skill_retained"]:
            return ToolResult.ok(
                f"Deleted the local procedural skill for {name}; the "
                f"{payload['remaining_source_kind']} source remains resolved.",
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
                f"Installed procedural skill {skill_name} at {payload['source_revision']} and left it disabled.",
                str(payload["state_error"]),
                data=payload,
            )
        return ToolResult.ok(
            f"Installed procedural skill {skill_name} at {payload['source_revision']} and left it disabled.",
            data=payload,
        )


__all__ = [
    "PROCEDURAL_SKILL_NODE_TYPE",
    "ProceduralSkillsFeature",
]
