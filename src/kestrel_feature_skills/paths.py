"""Path-containment helpers shared by storage, sources, and HTTP routes."""

from __future__ import annotations

import os
import re
from pathlib import Path, PurePosixPath

from .errors import SkillPathError

_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")


def direct_child(
    root: Path,
    name: str,
    *,
    root_identity: tuple[int, int] | None = None,
) -> Path:
    """Return a direct child without permitting alternate path spellings."""

    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise SkillPathError("skill name must identify one direct child folder")
    try:
        root_value = root.lstat()
    except OSError as exc:
        raise SkillPathError("skill root must be a real directory") from exc
    identity = (root_value.st_dev, root_value.st_ino)
    if (
        root.is_symlink()
        or not root.is_dir()
        or (root_identity is not None and identity != root_identity)
    ):
        raise SkillPathError("skill root directory changed after validation")
    candidate = root / name
    if candidate.is_symlink():
        raise SkillPathError("skill direct child must not be a symlink")
    return candidate


def contained_path(root: Path, relative: str, *, must_exist: bool = True) -> Path:
    """Resolve a POSIX-style relative path and prove it remains below ``root``.

    Backslashes are rejected on every platform so a value cannot be harmless on
    POSIX and become traversal when the same skill is installed on Windows.
    Existing symlinks are resolved before the containment comparison.
    """

    if not isinstance(relative, str) or not relative or "\x00" in relative:
        raise SkillPathError("path must be a non-empty string without NUL bytes")
    if "\\" in relative or relative.startswith("/") or _WINDOWS_DRIVE.match(relative):
        raise SkillPathError("absolute and backslash paths are not allowed")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise SkillPathError("path traversal is not allowed")

    root_resolved = root.resolve(strict=True)
    candidate = root.joinpath(*pure.parts)
    try:
        resolved = candidate.resolve(strict=must_exist)
    except FileNotFoundError as exc:
        raise SkillPathError(f"path does not exist: {relative}") from exc
    try:
        common = Path(os.path.commonpath((root_resolved, resolved)))
    except ValueError as exc:
        raise SkillPathError("path is on a different filesystem root") from exc
    if common != root_resolved:
        raise SkillPathError("path escapes the skill folder")
    return resolved


def lexical_contained_path(
    root: Path, relative: str, *, must_exist: bool = True
) -> Path:
    """Validate containment while retaining the caller's lexical path.

    Mutation code must inspect the spelling requested by the caller rather
    than publishing to the resolved symlink target returned by
    :func:`contained_path`. This helper performs the same containment proof,
    then returns a normalized lexical child below the resolved root.
    """

    contained_path(root, relative, must_exist=must_exist)
    pure = PurePosixPath(relative)
    return root.resolve(strict=True).joinpath(*pure.parts)


def reject_symlink_chain(root: Path, path: Path) -> None:
    """Reject any existing symlink from ``root`` through ``path``."""

    root_resolved = root.resolve(strict=True)
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise SkillPathError("path is not below the expected root") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise SkillPathError(f"symlinks are not allowed in skill paths: {part}")
    existing_parent = path.parent
    while not existing_parent.exists() and existing_parent != root:
        existing_parent = existing_parent.parent
    resolved_parent = existing_parent.resolve(strict=True)
    if Path(os.path.commonpath((root_resolved, resolved_parent))) != root_resolved:
        raise SkillPathError("path parent escapes the skill folder")


__all__ = [
    "contained_path",
    "direct_child",
    "lexical_contained_path",
    "reject_symlink_chain",
]
