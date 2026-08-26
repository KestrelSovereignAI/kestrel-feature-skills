"""Strict parsing and validation for folder-shaped procedural skills."""

from __future__ import annotations

import json
import os
import re
import stat
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .errors import SkillFormatError, SkillPathError
from .models import SkillDocument
from .paths import contained_path

SKILL_FILENAME = "SKILL.md"
MAX_SKILL_FILE_BYTES = 262_144
MAX_DESCRIPTION_BYTES = 512
MAX_FOLDER_FILES = 256
MAX_FOLDER_BYTES = 2_097_152
SKILL_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9_-]{0,62}[a-z0-9])?$")
_FRONTMATTER_KEYS = frozenset({"name", "description"})
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_UNICODE_LINE_SEPARATOR = re.compile(r"[\x85\u2028\u2029]")
_MARKDOWN_DESTINATION = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
_MARKDOWN_REFERENCE_DEFINITION = re.compile(
    r"(?m)^[ \t]{0,3}\[[^\]\r\n]+\]:[ \t]*"
    r"(?:\r?\n[ \t]{0,3})?(?:<([^>\r\n]+)>|(\S+))"
)
_REMOTE_SCHEMES = frozenset({"http", "https", "mailto"})


def _utf8_bytes(value: str, *, label: str) -> bytes:
    try:
        return value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise SkillFormatError(f"{label} must be valid UTF-8 text") from exc


def validate_skill_name(name: object) -> str:
    if not isinstance(name, str) or not SKILL_NAME_RE.fullmatch(name):
        raise SkillFormatError(
            "skill name must be 1-64 lowercase letters, numbers, hyphens, or underscores"
        )
    return name


def _parse_scalar(raw: str, *, key: str) -> str:
    value = raw.strip()
    if not value:
        raise SkillFormatError(f"frontmatter field {key!r} must not be empty")
    if value.startswith('"'):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise SkillFormatError(
                f"invalid quoted scalar for {key!r}: {exc.msg}"
            ) from exc
        if not isinstance(decoded, str):
            raise SkillFormatError(f"frontmatter field {key!r} must be a string")
        return decoded
    if value.startswith("'"):
        if len(value) < 2 or not value.endswith("'"):
            raise SkillFormatError(f"unterminated single-quoted scalar for {key!r}")
        return value[1:-1].replace("''", "'")
    if value[0] in "!&*[{>|%@`" or value in {"null", "true", "false", "~"}:
        raise SkillFormatError(
            f"frontmatter field {key!r} uses unsupported YAML syntax; quote it"
        )
    return value


def parse_skill_markdown(
    content: str | bytes, *, source: str = SKILL_FILENAME
) -> SkillDocument:
    """Parse the deliberately small, safe YAML-frontmatter subset."""

    if isinstance(content, bytes):
        if len(content) > MAX_SKILL_FILE_BYTES:
            raise SkillFormatError(f"{source} exceeds {MAX_SKILL_FILE_BYTES} bytes")
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SkillFormatError(f"{source} must be UTF-8") from exc
    elif isinstance(content, str):
        text = content
        if len(_utf8_bytes(text, label=source)) > MAX_SKILL_FILE_BYTES:
            raise SkillFormatError(f"{source} exceeds {MAX_SKILL_FILE_BYTES} bytes")
    else:
        raise SkillFormatError(f"{source} must be text")

    if not text or not text.strip():
        raise SkillFormatError(f"{source} is empty")
    normalized = text.replace("\r\n", "\n")
    if normalized.startswith("\ufeff"):
        raise SkillFormatError(f"{source} must not start with a byte-order mark")
    lines = normalized.splitlines()
    if not lines or lines[0] != "---":
        raise SkillFormatError(
            f"{source} must start with an exact '---' frontmatter fence"
        )
    try:
        closing = lines.index("---", 1)
    except ValueError as exc:
        raise SkillFormatError(
            f"{source} is missing its closing frontmatter fence"
        ) from exc

    fields: dict[str, str] = {}
    for line_number, line in enumerate(lines[1:closing], start=2):
        if not line.strip():
            continue
        if line[:1].isspace() or ":" not in line:
            raise SkillFormatError(
                f"{source}:{line_number} must be a top-level 'key: value' scalar"
            )
        key, raw = line.split(":", 1)
        if key not in _FRONTMATTER_KEYS:
            raise SkillFormatError(
                f"{source}:{line_number} has unsupported field {key!r}"
            )
        if key in fields:
            raise SkillFormatError(f"{source}:{line_number} duplicates field {key!r}")
        fields[key] = _parse_scalar(raw, key=key)

    missing = _FRONTMATTER_KEYS - fields.keys()
    if missing:
        raise SkillFormatError(
            f"{source} is missing required field(s): {', '.join(sorted(missing))}"
        )
    name = validate_skill_name(fields["name"])
    description = fields["description"].strip()
    if not description:
        raise SkillFormatError("description must not be empty")
    if (
        "\n" in description
        or "\r" in description
        or _CONTROL.search(description)
        or _UNICODE_LINE_SEPARATOR.search(description)
    ):
        raise SkillFormatError("description must be one printable line")
    if len(_utf8_bytes(description, label="description")) > MAX_DESCRIPTION_BYTES:
        raise SkillFormatError(
            f"description exceeds {MAX_DESCRIPTION_BYTES} UTF-8 bytes"
        )
    body = "\n".join(lines[closing + 1 :]).strip()
    if not body:
        raise SkillFormatError("SKILL.md procedure body must not be empty")
    if _CONTROL.search(body):
        raise SkillFormatError("SKILL.md body contains control characters")
    return SkillDocument(name=name, description=description, body=body)


def serialize_skill_markdown(document: SkillDocument) -> str:
    """Serialize with JSON-quoted scalars, a strict subset valid as YAML."""

    name = validate_skill_name(document.name)
    reparsed = parse_skill_markdown(
        "---\n"
        f"name: {json.dumps(name, ensure_ascii=False)}\n"
        f"description: {json.dumps(document.description, ensure_ascii=False)}\n"
        "---\n\n"
        f"{document.body.strip()}\n"
    )
    return (
        "---\n"
        f"name: {json.dumps(reparsed.name, ensure_ascii=False)}\n"
        f"description: {json.dumps(reparsed.description, ensure_ascii=False)}\n"
        "---\n\n"
        f"{reparsed.body}\n"
    )


def _local_markdown_destinations(body: str) -> tuple[str, ...]:
    destinations: list[str] = []
    raw_destinations = [
        match.group(1) for match in _MARKDOWN_DESTINATION.finditer(body)
    ]
    raw_destinations.extend(
        match.group(1) or match.group(2)
        for match in _MARKDOWN_REFERENCE_DEFINITION.finditer(body)
    )
    for candidate in raw_destinations:
        raw = candidate.strip()
        if raw.startswith("<") and ">" in raw:
            raw = raw[1 : raw.index(">")]
        else:
            raw = raw.split(maxsplit=1)[0]
        if not raw or raw.startswith("#"):
            continue
        decoded = unquote(raw)
        split = urlsplit(decoded)
        if split.scheme:
            if split.scheme.lower() not in _REMOTE_SCHEMES:
                raise SkillPathError(
                    f"unsupported link scheme in SKILL.md: {split.scheme}"
                )
            continue
        destinations.append(decoded.split("#", 1)[0].split("?", 1)[0])
    return tuple(destinations)


def validate_document_references(document: SkillDocument, folder: Path) -> None:
    """Require every bundled local link to resolve inside ``folder``."""

    for destination in _local_markdown_destinations(document.body):
        contained_path(folder, destination, must_exist=True)


def validate_skill_folder(folder: Path, *, source_root: Path) -> SkillDocument:
    """Validate the complete folder, including links and every symlink escape."""

    if folder.is_symlink():
        raise SkillPathError("skill folders must not be symlinks")
    if not folder.is_dir():
        raise SkillFormatError("skill candidate is not a directory")
    contained_path(source_root, folder.name)

    file_count = 0
    byte_count = 0

    def reject_walk_error(error: OSError) -> None:
        raise SkillFormatError("could not scan the complete skill folder") from error

    for current, directories, files in os.walk(
        folder,
        followlinks=False,
        onerror=reject_walk_error,
    ):
        current_path = Path(current)
        for entry in (*directories, *files):
            path = current_path / entry
            relative = path.relative_to(folder).as_posix()
            _utf8_bytes(relative, label="skill resource path")
            if path.is_symlink():
                raise SkillPathError(
                    f"symlinks are not allowed in skill folders: {path.name}"
                )
        for filename in files:
            path = current_path / filename
            mode = path.stat().st_mode
            if not stat.S_ISREG(mode):
                raise SkillPathError(
                    f"skill resources must be regular files: {filename}"
                )
            contained_path(folder, path.relative_to(folder).as_posix())
            file_count += 1
            byte_count += path.stat().st_size
            if file_count > MAX_FOLDER_FILES:
                raise SkillFormatError(f"skill folder exceeds {MAX_FOLDER_FILES} files")
            if byte_count > MAX_FOLDER_BYTES:
                raise SkillFormatError(f"skill folder exceeds {MAX_FOLDER_BYTES} bytes")

    primary = folder / SKILL_FILENAME
    if not primary.is_file() or primary.is_symlink():
        raise SkillFormatError(f"skill folder requires a regular {SKILL_FILENAME}")
    document = parse_skill_markdown(primary.read_bytes(), source=str(primary))
    if document.name != folder.name:
        raise SkillFormatError(
            f"frontmatter name {document.name!r} must match folder name {folder.name!r}"
        )
    validate_document_references(document, folder)
    return document


__all__ = [
    "MAX_DESCRIPTION_BYTES",
    "MAX_FOLDER_BYTES",
    "MAX_FOLDER_FILES",
    "MAX_SKILL_FILE_BYTES",
    "SKILL_FILENAME",
    "parse_skill_markdown",
    "serialize_skill_markdown",
    "validate_document_references",
    "validate_skill_folder",
    "validate_skill_name",
]
