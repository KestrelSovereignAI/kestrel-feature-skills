"""Operator HTTP surface, scoped to validated skill roots."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Query
from kestrel_sdk.storage.database import DatabaseError
from pydantic import BaseModel, ConfigDict, Field

from .enablement import DEFAULT_PRIORITY
from .errors import (
    EnablementUnavailableError,
    GitSourceError,
    SkillConflictError,
    SkillFormatError,
    SkillNotFoundError,
    SkillPathError,
    SkillReadOnlyError,
)

if TYPE_CHECKING:
    from .feature import ProceduralSkillsFeature


class _StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateSkillRequest(_StrictRequest):
    name: str
    description: str
    body: str
    enabled: bool = False
    priority: int = DEFAULT_PRIORITY


class EditFileRequest(_StrictRequest):
    path: str
    content: str


class SkillStateRequest(_StrictRequest):
    enabled: bool
    priority: int | None = None


class InstallSkillRequest(_StrictRequest):
    source_url: str
    skill_name: str
    ref: str = Field(default="HEAD", max_length=200)


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, SkillNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, SkillConflictError):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, SkillReadOnlyError):
        return HTTPException(status_code=403, detail=str(exc))
    if isinstance(exc, EnablementUnavailableError):
        return HTTPException(status_code=503, detail=str(exc))
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
        return feature.catalog_payload()

    @router.post("/reload")
    async def reload_catalog() -> dict[str, object]:
        try:
            await feature.refresh()
            return feature.catalog_payload()
        except (DatabaseError, OSError, RuntimeError) as exc:
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
            return {"name": name, "entries": list(feature.tree(name=name))}
        except (SkillNotFoundError, SkillFormatError, SkillPathError, OSError) as exc:
            raise _http_error(exc) from exc

    @router.get("/{name}/file")
    async def read_file(
        name: str,
        path: str = Query(..., min_length=1, max_length=1024),
    ) -> dict[str, object]:
        try:
            return feature.read_file(name=name, relative_path=path)
        except (
            SkillNotFoundError,
            SkillFormatError,
            SkillPathError,
            OSError,
        ) as exc:
            raise _http_error(exc) from exc

    @router.put("/{name}/file")
    async def write_file(name: str, request: EditFileRequest) -> dict[str, object]:
        try:
            return await feature.edit_skill(
                name=name,
                relative_path=request.path,
                content=request.content,
            )
        except (
            SkillNotFoundError,
            SkillReadOnlyError,
            SkillFormatError,
            SkillPathError,
            DatabaseError,
            OSError,
            RuntimeError,
        ) as exc:
            raise _http_error(exc) from exc

    @router.patch("/{name}/state")
    async def set_state(name: str, request: SkillStateRequest) -> dict[str, object]:
        try:
            return await feature.set_skill_state(
                name=name,
                enabled=request.enabled,
                priority=request.priority,
            )
        except (
            SkillNotFoundError,
            EnablementUnavailableError,
            DatabaseError,
            RuntimeError,
            ValueError,
        ) as exc:
            raise _http_error(exc) from exc

    @router.delete("/{name}")
    async def delete_skill(name: str) -> dict[str, object]:
        # This authenticated operator route is called only after the package UI's
        # explicit destructive confirmation. Agent-initiated deletion goes
        # through the separately declared ALWAYS_ASK tool permission rail.
        try:
            return await feature.delete_skill(name=name)
        except (
            SkillNotFoundError,
            SkillReadOnlyError,
            SkillPathError,
            DatabaseError,
            OSError,
            RuntimeError,
        ) as exc:
            raise _http_error(exc) from exc

    return router


__all__ = ["build_router"]
