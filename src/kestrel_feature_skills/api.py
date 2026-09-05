"""Operator HTTP surface, scoped to validated skill roots."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from kestrel_sdk.storage.database import DatabaseError
from kestrel_sovereign.security.demo_isolation import enforce_destructive_op
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from .enablement import DEFAULT_PRIORITY
from .errors import (
    EnablementUnavailableError,
    GitInputError,
    GitSourceError,
    SkillConflictError,
    SkillFormatError,
    SkillNotFoundError,
    SkillPathError,
    SkillPrivacyError,
    SkillReadOnlyError,
)
from .format import MAX_RESOURCE_PATH_BYTES

_REVISION_PATTERN = r"^[0-9a-f]{64}$"

if TYPE_CHECKING:
    from .feature import ProceduralSkillsFeature


class _StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateSkillRequest(_StrictRequest):
    name: str
    description: str
    body: str
    enabled: bool = False
    priority: StrictInt = DEFAULT_PRIORITY


class EditFileRequest(_StrictRequest):
    path: str
    content: str


class SkillStateRequest(_StrictRequest):
    enabled: bool
    priority: StrictInt | None = None


class InstallSkillRequest(_StrictRequest):
    source_url: str
    skill_name: str
    ref: str = Field(default="HEAD", max_length=200)


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, SkillNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, SkillConflictError):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, (SkillReadOnlyError, SkillPrivacyError)):
        return HTTPException(status_code=403, detail=str(exc))
    if isinstance(exc, EnablementUnavailableError):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, GitInputError):
        return HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, GitSourceError):
        return HTTPException(status_code=502, detail=str(exc))
    if isinstance(exc, (SkillFormatError, SkillPathError, ValueError)):
        return HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, DatabaseError):
        return HTTPException(
            status_code=503, detail="skill configuration database is unavailable"
        )
    if isinstance(exc, OSError):
        return HTTPException(
            status_code=500, detail="skill filesystem operation failed"
        )
    return HTTPException(status_code=500, detail="procedural skill operation failed")


def build_router(feature: ProceduralSkillsFeature) -> APIRouter:
    """Build one feature-instance router; no route accepts an arbitrary root."""

    router = APIRouter(prefix="/api/procedural-skills", tags=["procedural-skills"])

    @router.get("")
    async def catalog() -> dict[str, object]:
        try:
            async with feature.persistent_read():
                return feature.catalog_payload()
        except (SkillPathError, DatabaseError, OSError, RuntimeError) as exc:
            raise _http_error(exc) from exc

    @router.post("/reload")
    async def reload_catalog() -> dict[str, object]:
        try:
            async with feature.persistent_read(refresh=True):
                return feature.catalog_payload()
        except (SkillPathError, DatabaseError, OSError, RuntimeError) as exc:
            raise _http_error(exc) from exc

    @router.post("")
    async def create_skill(request: CreateSkillRequest) -> dict[str, object]:
        try:
            return await feature.create_skill(
                name=request.name,
                description=request.description,
                body=request.body,
                enabled=request.enabled,
                priority=request.priority,
            )
        except (
            SkillFormatError,
            SkillPathError,
            SkillPrivacyError,
            SkillConflictError,
            DatabaseError,
            OSError,
            RuntimeError,
            ValueError,
        ) as exc:
            raise _http_error(exc) from exc

    @router.post("/install")
    async def install_skill(request: InstallSkillRequest) -> dict[str, object]:
        try:
            return await feature.install_skill(
                source_url=request.source_url,
                skill_name=request.skill_name,
                ref=request.ref,
            )
        except (
            GitSourceError,
            SkillFormatError,
            SkillPathError,
            SkillPrivacyError,
            SkillConflictError,
            SkillNotFoundError,
            DatabaseError,
            OSError,
            RuntimeError,
            ValueError,
        ) as exc:
            raise _http_error(exc) from exc

    @router.get("/{name}/tree")
    async def tree(name: str) -> dict[str, object]:
        try:
            async with feature.persistent_read(refresh=True):
                return {"name": name, "entries": list(feature.tree(name=name))}
        except (
            SkillNotFoundError,
            SkillFormatError,
            SkillPathError,
            SkillPrivacyError,
            SkillConflictError,
            DatabaseError,
            OSError,
            RuntimeError,
        ) as exc:
            raise _http_error(exc) from exc

    @router.get("/{name}/file")
    async def read_file(
        name: str,
        path: str = Query(..., min_length=1, max_length=MAX_RESOURCE_PATH_BYTES),
    ) -> dict[str, object]:
        try:
            async with feature.persistent_read(refresh=True):
                return feature.read_file(name=name, relative_path=path)
        except (
            SkillNotFoundError,
            SkillFormatError,
            SkillPathError,
            SkillPrivacyError,
            SkillConflictError,
            DatabaseError,
            OSError,
            RuntimeError,
        ) as exc:
            raise _http_error(exc) from exc

    @router.put("/{name}/file")
    async def write_file(
        name: str,
        request: EditFileRequest,
        expected_revision: str = Header(
            ...,
            alias="If-Match",
            min_length=64,
            max_length=64,
            pattern=_REVISION_PATTERN,
        ),
    ) -> dict[str, object]:
        try:
            return await feature.edit_skill(
                name=name,
                relative_path=request.path,
                content=request.content,
                expected_revision=expected_revision,
            )
        except (
            SkillNotFoundError,
            SkillReadOnlyError,
            SkillConflictError,
            SkillFormatError,
            SkillPathError,
            SkillPrivacyError,
            DatabaseError,
            OSError,
            RuntimeError,
        ) as exc:
            raise _http_error(exc) from exc

    @router.patch("/{name}/state")
    async def set_state(
        name: str,
        request: SkillStateRequest,
        expected_revision: str = Header(
            ...,
            alias="If-Match",
            min_length=64,
            max_length=64,
            pattern=_REVISION_PATTERN,
        ),
    ) -> dict[str, object]:
        try:
            return await feature.set_skill_state(
                name=name,
                enabled=request.enabled,
                priority=request.priority,
                expected_revision=expected_revision,
            )
        except (
            SkillNotFoundError,
            SkillConflictError,
            SkillFormatError,
            SkillPathError,
            EnablementUnavailableError,
            SkillPrivacyError,
            DatabaseError,
            RuntimeError,
            ValueError,
        ) as exc:
            raise _http_error(exc) from exc

    @router.delete("/{name}", dependencies=[Depends(enforce_destructive_op)])
    async def delete_skill(
        name: str,
        expected_revision: str = Header(
            ...,
            alias="If-Match",
            min_length=64,
            max_length=64,
            pattern=_REVISION_PATTERN,
        ),
    ) -> dict[str, object]:
        # Core's server-side destructive rail is load-bearing. The package UI
        # attaches its audited opt-in header only after explicit confirmation;
        # agent-initiated deletion separately uses the ALWAYS_ASK tool rail.
        try:
            return await feature.delete_skill(
                name=name,
                expected_revision=expected_revision,
            )
        except (
            SkillNotFoundError,
            SkillReadOnlyError,
            SkillConflictError,
            SkillFormatError,
            SkillPathError,
            SkillPrivacyError,
            DatabaseError,
            OSError,
            RuntimeError,
        ) as exc:
            raise _http_error(exc) from exc

    return router


__all__ = ["build_router"]
