"""File-authoritative skill storage with atomic primary-record writes."""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

from .errors import (
    SkillConflictError,
    SkillFormatError,
    SkillNotFoundError,
    SkillPathError,
    SkillReadOnlyError,
)
from .format import (
    MAX_SKILL_FILE_BYTES,
    SKILL_FILENAME,
    parse_skill_markdown,
    serialize_skill_markdown,
    validate_document_references,
    validate_skill_folder,
    validate_skill_name,
)
from .models import CatalogSnapshot, SkillDocument, SkillProvenance, SkillRecord
from .paths import contained_path, direct_child, reject_symlink_chain
from .sources import PROVENANCE_FILENAME, serialize_provenance

CLAIM_STALENESS_SECONDS = 60
MAX_EDITOR_FILE_BYTES = 262_144
_INTERNAL_PREFIXES = (".SKILL.md.tmp.", ".SKILL.md.claim")


def _claim_is_stale(path: Path) -> bool:
    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        return False
    return age > CLAIM_STALENESS_SECONDS


def _claim_identity(path: Path) -> tuple[int, int] | None:
    try:
        value = path.stat()
    except OSError:
        return None
    return value.st_dev, value.st_ino


def _write_tmp(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def atomic_write_primary(folder: Path, payload: bytes, *, overwrite: bool) -> None:
    """Write ``SKILL.md`` with the inherited hardlink claim protocol.

    A per-writer temporary file is hardlinked to one shared claim. The first
    writer wins; a second writer cannot remove the winner's claim because
    cleanup is conditional on this writer having created the link. Creates use
    a second non-overwriting hardlink for finalization. Intentional edits use
    ``os.replace`` only after obtaining the same exclusive claim.
    """

    if len(payload) > MAX_SKILL_FILE_BYTES:
        raise SkillFormatError(f"{SKILL_FILENAME} exceeds {MAX_SKILL_FILE_BYTES} bytes")
    target = folder / SKILL_FILENAME
    if not overwrite and target.exists():
        raise SkillConflictError(f"skill already exists: {folder.name}")
    tmp = folder / f".SKILL.md.tmp.{uuid.uuid4().hex}"
    claim = folder / ".SKILL.md.claim"
    owned_claim: tuple[int, int] | None = None
    _write_tmp(tmp, payload)
    tmp_identity = _claim_identity(tmp)
    if tmp_identity is None:  # pragma: no cover - fsync'd file vanished externally
        tmp.unlink(missing_ok=True)
        raise OSError("skill writer temporary file vanished before claiming")
    try:
        try:
            os.link(tmp, claim)
            owned_claim = tmp_identity
        except FileExistsError:
            if not _claim_is_stale(claim):
                raise SkillConflictError(
                    f"concurrent skill write is already in progress for {folder.name}"
                ) from None
            try:
                claim.unlink()
            except FileNotFoundError:
                pass
            try:
                os.link(tmp, claim)
                owned_claim = tmp_identity
            except FileExistsError as exc:
                raise SkillConflictError(
                    f"concurrent skill write won stale-claim recovery for {folder.name}"
                ) from exc
        if owned_claim is None or _claim_identity(claim) != owned_claim:
            raise SkillConflictError(
                f"skill write lost its reclaimed claim for {folder.name}"
            )
        if overwrite:
            os.replace(tmp, target)
        else:
            try:
                os.link(tmp, target)
            except FileExistsError as exc:
                raise SkillConflictError(
                    f"skill already exists: {folder.name}"
                ) from exc
    finally:
        tmp.unlink(missing_ok=True)
        if owned_claim is not None and _claim_identity(claim) == owned_claim:
            claim.unlink(missing_ok=True)


def atomic_replace_file(path: Path, payload: bytes) -> None:
    """Atomically replace an operator-edited resource after containment checks."""

    if len(payload) > MAX_EDITOR_FILE_BYTES:
        raise SkillFormatError(f"editor file exceeds {MAX_EDITOR_FILE_BYTES} bytes")
    tmp = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex}")
    _write_tmp(tmp, payload)
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


class SkillStore:
    """Mutate only the agent-local root; read from the resolved catalog."""

    def __init__(self, local_root: Path):
        local_root.mkdir(parents=True, exist_ok=True)
        if local_root.is_symlink():
            raise SkillPathError("agent-local skill root must not be a symlink")
        self.local_root = local_root.resolve(strict=True)

    @staticmethod
    def get(snapshot: CatalogSnapshot, name: str) -> SkillRecord:
        name = validate_skill_name(name)
        record = snapshot.by_name().get(name)
        if record is None:
            raise SkillNotFoundError(f"skill {name!r} was not found")
        return record

    def create(self, document: SkillDocument) -> Path:
        name = validate_skill_name(document.name)
        folder = direct_child(self.local_root, name)
        try:
            folder.mkdir(mode=0o700)
        except FileExistsError as exc:
            raise SkillConflictError(f"skill already exists: {name}") from exc
        try:
            validate_document_references(document, folder)
            atomic_write_primary(
                folder,
                serialize_skill_markdown(document).encode("utf-8"),
                overwrite=False,
            )
            validate_skill_folder(folder, source_root=self.local_root)
        except (SkillFormatError, SkillPathError, SkillConflictError, OSError):
            if folder.exists() and not any(folder.iterdir()):
                folder.rmdir()
            elif folder.exists() and not (folder / SKILL_FILENAME).exists():
                shutil.rmtree(folder)
            raise
        return folder

    def edit_primary(self, record: SkillRecord, content: str) -> SkillDocument:
        folder = self._require_local(record)
        document = parse_skill_markdown(
            content, source=f"{record.name}/{SKILL_FILENAME}"
        )
        if document.name != record.name:
            raise SkillFormatError("frontmatter name cannot rename a skill folder")
        validate_document_references(document, folder)
        atomic_write_primary(folder, content.encode("utf-8"), overwrite=True)
        return validate_skill_folder(folder, source_root=self.local_root)

    def read_file(self, record: SkillRecord, relative_path: str) -> str:
        folder = self._require_real_folder(record)
        path = contained_path(folder, relative_path, must_exist=True)
        if path.is_symlink() or not path.is_file():
            raise SkillPathError("requested skill path must be a regular file")
        if path.stat().st_size > MAX_EDITOR_FILE_BYTES:
            raise SkillFormatError(f"file exceeds {MAX_EDITOR_FILE_BYTES} bytes")
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise SkillFormatError("the editor only opens UTF-8 text files") from exc

    def write_file(self, record: SkillRecord, relative_path: str, content: str) -> None:
        folder = self._require_local(record)
        if not isinstance(content, str):
            raise SkillFormatError("editor content must be text")
        if relative_path == SKILL_FILENAME:
            self.edit_primary(record, content)
            return
        path = contained_path(folder, relative_path, must_exist=False)
        reject_symlink_chain(folder, path)
        relative = path.relative_to(folder)
        if path.suffix == ".py":
            if not relative.parts or relative.parts[0] != "scripts":
                raise SkillPathError("Python files are allowed only below scripts/")
        elif path.suffix != ".md":
            raise SkillPathError(
                "the editor writes only .md resources and scripts/*.py"
            )
        payload = content.encode("utf-8")
        with tempfile.TemporaryDirectory(
            prefix=".kestrel-skill-edit-", dir=self.local_root
        ) as temporary:
            staged_root = Path(temporary)
            staged_folder = staged_root / record.name
            shutil.copytree(folder, staged_folder, symlinks=True)
            staged_path = contained_path(
                staged_folder, relative.as_posix(), must_exist=False
            )
            reject_symlink_chain(staged_folder, staged_path)
            staged_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            reject_symlink_chain(staged_folder, staged_path)
            atomic_replace_file(staged_path, payload)
            validate_skill_folder(staged_folder, source_root=staged_root)

        # The complete candidate was valid. Re-resolve the real folder and path
        # immediately before the single-file atomic publication so a folder or
        # ancestor swapped for a symlink cannot redirect the write.
        folder = self._require_local(record)
        path = contained_path(folder, relative.as_posix(), must_exist=False)
        reject_symlink_chain(folder, path)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        reject_symlink_chain(folder, path)
        atomic_replace_file(path, payload)

    def tree(self, record: SkillRecord) -> tuple[dict[str, object], ...]:
        folder = self._require_real_folder(record)
        validate_skill_folder(folder, source_root=folder.parent)
        entries: list[dict[str, object]] = []
        for path in sorted(folder.rglob("*"), key=lambda item: item.as_posix()):
            relative = path.relative_to(folder).as_posix()
            if path.name == PROVENANCE_FILENAME or path.name.startswith(
                _INTERNAL_PREFIXES
            ):
                continue
            if path.is_symlink():
                raise SkillPathError(f"symlink appeared during tree read: {relative}")
            is_file = path.is_file()
            editable = bool(
                record.editable
                and is_file
                and (
                    relative == SKILL_FILENAME
                    or path.suffix == ".md"
                    or (path.suffix == ".py" and relative.startswith("scripts/"))
                )
            )
            entries.append(
                {
                    "path": relative,
                    "type": "file" if is_file else "directory",
                    "bytes": path.stat().st_size if is_file else None,
                    "editable": editable,
                    "execution_risk": bool(is_file and path.suffix == ".py"),
                }
            )
        return tuple(entries)

    def install_folder(
        self,
        source_folder: Path,
        *,
        provenance: SkillProvenance,
    ) -> Path:
        document = validate_skill_folder(
            source_folder, source_root=source_folder.parent
        )
        target = direct_child(self.local_root, document.name)
        try:
            target.mkdir(mode=0o700)
        except FileExistsError as exc:
            raise SkillConflictError(f"skill already exists: {document.name}") from exc
        try:
            for path in sorted(
                source_folder.rglob("*"), key=lambda item: item.as_posix()
            ):
                relative = path.relative_to(source_folder)
                if relative.as_posix() in {SKILL_FILENAME, PROVENANCE_FILENAME}:
                    continue
                destination = target / relative
                if path.is_dir():
                    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
                elif path.is_file():
                    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                    shutil.copyfile(path, destination, follow_symlinks=False)
                else:
                    raise SkillPathError(
                        f"unsupported remote resource: {relative.as_posix()}"
                    )
            atomic_replace_file(
                target / PROVENANCE_FILENAME, serialize_provenance(provenance)
            )
            atomic_write_primary(
                target,
                serialize_skill_markdown(document).encode("utf-8"),
                overwrite=False,
            )
            validate_skill_folder(target, source_root=self.local_root)
        except (SkillFormatError, SkillPathError, SkillConflictError, OSError):
            if target.exists():
                shutil.rmtree(target)
            raise
        return target

    def delete(self, record: SkillRecord) -> None:
        expected = self._require_local(record)
        shutil.rmtree(expected)

    @staticmethod
    def search(snapshot: CatalogSnapshot, query: str) -> tuple[SkillRecord, ...]:
        if not isinstance(query, str) or not query.strip():
            raise SkillFormatError("search query must not be empty")
        needle = query.casefold().strip()
        return tuple(
            record
            for record in snapshot.records
            if needle in record.name.casefold()
            or needle in record.document.description.casefold()
        )

    @staticmethod
    def _require_real_folder(record: SkillRecord) -> Path:
        if record.folder.is_symlink() or not record.folder.is_dir():
            raise SkillPathError(
                "skill folder changed after discovery; reload before accessing it"
            )
        return record.folder

    def _require_local(self, record: SkillRecord) -> Path:
        if not record.editable:
            raise SkillReadOnlyError(
                f"skill {record.name!r} comes from {record.source_id!r}; create a local override to edit it"
            )
        expected = self.local_root / record.name
        if record.folder != expected or expected.is_symlink() or not expected.is_dir():
            raise SkillPathError(
                "local skill folder changed after discovery; reload before mutating it"
            )
        return expected


__all__ = [
    "CLAIM_STALENESS_SECONDS",
    "MAX_EDITOR_FILE_BYTES",
    "SkillStore",
    "atomic_replace_file",
    "atomic_write_primary",
]
