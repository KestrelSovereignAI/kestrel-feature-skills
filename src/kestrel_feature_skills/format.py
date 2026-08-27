"""Strict parsing and validation for folder-shaped procedural skills."""

from __future__ import annotations

import html
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

from .errors import SkillFormatError, SkillPathError
from .models import SkillDocument
from .paths import contained_path

SKILL_FILENAME = "SKILL.md"
MAX_SKILL_FILE_BYTES = 262_144
MAX_DESCRIPTION_BYTES = 512
MAX_FOLDER_FILES = 256
MAX_FOLDER_ENTRIES = 512
MAX_FOLDER_DEPTH = 32
MAX_FOLDER_BYTES = 2_097_152
SKILL_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9_-]{0,62}[a-z0-9])?$")
_FRONTMATTER_KEYS = frozenset({"name", "description"})
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_UNICODE_LINE_SEPARATOR = re.compile(r"[\x85\u2028\u2029]")
_MARKDOWN_REFERENCE_DEFINITION = re.compile(
    r"(?m)^[ \t]{0,3}\[(?:\\[^\r\n]|[^\]\\\r\n])+\]:[ \t]*"
    r"(?:\r?\n[ \t]{0,3})?(?:<([^>\r\n]+)>|(\S+))"
)
_MARKDOWN_BACKSLASH_ESCAPE = re.compile(r"\\([!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~])")
_MARKDOWN_AUTOLINK = re.compile(r"<([A-Za-z][A-Za-z0-9+.-]{1,31}:[^<>\x00-\x20]*)>")
_MARKDOWN_FENCE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")
_MARKDOWN_BLOCKQUOTE_PREFIX = re.compile(r"^[ \t]{0,3}>[ \t]?")
_MARKDOWN_LIST_PREFIX = re.compile(r"^([ ]{0,3})((?:[-+*]|\d{1,9}[.)]))([ \t]+)")
_MARKDOWN_NONPARAGRAPH_BLOCK = re.compile(
    r"^(?:#{1,6}(?:[ \t]+|$)|(?:[*_-][ \t]*){3,}$|<[!/?A-Za-z])"
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
    if _CONTROL.search(normalized) or _UNICODE_LINE_SEPARATOR.search(normalized):
        raise SkillFormatError(
            f"{source} contains control or Unicode line-separator characters"
        )
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


def _indent_prefix(line: str, required_columns: int) -> int | None:
    """Return the character offset covering a CommonMark indentation width."""

    columns = 0
    for index, character in enumerate(line):
        if character == " ":
            columns += 1
        elif character == "\t":
            columns += 4 - (columns % 4)
        else:
            break
        if columns >= required_columns:
            return index + 1
    return None


def _list_marker_prefix(line: str) -> tuple[int, int] | None:
    """Return the consumed characters and continuation columns for a list marker."""

    match = _MARKDOWN_LIST_PREFIX.match(line)
    if match is None:
        return None
    marker_end = len(match.group(1)) + len(match.group(2))
    whitespace = match.group(3)
    whitespace_columns = 0
    consumed_whitespace = 0
    for character in whitespace:
        width = 1 if character == " " else 4 - ((marker_end + whitespace_columns) % 4)
        if whitespace_columns + width > 4:
            break
        whitespace_columns += width
        consumed_whitespace += 1
    if not whitespace_columns:
        return None
    # Five or more columns after a marker mean one separating column followed
    # by indented content; leave the excess visible for code-block detection.
    if whitespace_columns == 4 and consumed_whitespace < len(whitespace):
        whitespace_columns = 1
        consumed_whitespace = 1
    consumed = marker_end + consumed_whitespace
    return consumed, marker_end + whitespace_columns


def _markdown_container_lines(
    body: str,
) -> tuple[
    tuple[str, str, tuple[int, tuple[int, ...]], tuple[int, tuple[int, ...]]],
    ...,
]:
    """Normalize explicit containers and list continuations line by line.

    The container key lets fenced-code masking notice when an unclosed fence's
    blockquote or list item has ended. List levels are retained across blank
    lines so continuation-indented reference definitions receive the same view
    that CommonMark gives its inline/reference parser.
    """

    rows: list[
        tuple[str, str, tuple[int, tuple[int, ...]], tuple[int, tuple[int, ...]]]
    ] = []
    list_levels: list[tuple[int, int]] = []
    list_quote_depth = 0
    next_list_id = 0
    for line in body.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        ending = line[len(content) :]
        quote_depth = 0
        working = content
        while match := _MARKDOWN_BLOCKQUOTE_PREFIX.match(working):
            quote_depth += 1
            working = working[match.end() :]

        if quote_depth != list_quote_depth:
            list_levels = []
        list_quote_depth = quote_depth
        active_level: int | None = None
        retained_levels: list[tuple[int, int]] = []
        if not working.strip():
            retained_levels = list_levels
            active_level = list_levels[-1][0] if list_levels else None
            continued_ids = (
                tuple(item_id for _indent, item_id in retained_levels)
                if active_level is not None
                and _indent_prefix(working, active_level) is not None
                else ()
            )
            working = ""
        else:
            for level_index in range(len(list_levels) - 1, -1, -1):
                prefix = _indent_prefix(working, list_levels[level_index][0])
                if prefix is None:
                    continue
                active_level = list_levels[level_index][0]
                retained_levels = list_levels[: level_index + 1]
                working = working[prefix:]
                break

            continued_ids = tuple(item_id for _indent, item_id in retained_levels)
            absolute_indent = active_level or 0
            while marker := _list_marker_prefix(working):
                consumed, continuation_columns = marker
                absolute_indent += continuation_columns
                next_list_id += 1
                retained_levels.append((absolute_indent, next_list_id))
                active_level = absolute_indent
                working = working[consumed:]

        list_levels = retained_levels
        final_ids = tuple(item_id for _indent, item_id in retained_levels)
        rows.append(
            (
                working,
                ending,
                (quote_depth, final_ids),
                (quote_depth, continued_ids),
            )
        )
    return tuple(rows)


def _mask_markdown_code(body: str) -> str:
    """Normalize containers and mask code while preserving line boundaries."""

    rows = _markdown_container_lines(body)
    normalized = "".join(
        f"{content}{ending}" for content, ending, _key, _continued in rows
    )
    masked = list(normalized)

    def mask(start: int, end: int) -> None:
        for position in range(start, end):
            if masked[position] not in "\r\n":
                masked[position] = " "

    offset = 0
    fence_character: str | None = None
    fence_length = 0
    fence_container: tuple[int, tuple[int, ...]] | None = None
    indented_container: tuple[int, tuple[int, ...]] | None = None
    in_indented_code = False
    paragraph_container: tuple[int, tuple[int, ...]] | None = None
    paragraph_open = False
    for content, ending, container, continued_container in rows:
        line_length = len(content) + len(ending)
        if fence_character is not None and fence_container is not None:
            opening_quote_depth, opening_list_ids = fence_container
            current_quote_depth, continued_list_ids = continued_container
            continues_container = (
                current_quote_depth >= opening_quote_depth
                and continued_list_ids[: len(opening_list_ids)] == opening_list_ids
            )
            current_list_ids = container[1]
            starts_outside_block = bool(
                not content.strip()
                or current_quote_depth > opening_quote_depth
                or current_list_ids
                or _MARKDOWN_NONPARAGRAPH_BLOCK.match(content.lstrip(" \t"))
                or _MARKDOWN_REFERENCE_DEFINITION.match(content)
            )
            if not continues_container and starts_outside_block:
                fence_character = None
                fence_length = 0
                fence_container = None
        match = _MARKDOWN_FENCE.match(content)
        if fence_character is None:
            if match:
                run = match.group(1)
                tail = content[match.end() :]
                if run[0] != "`" or "`" not in tail:
                    fence_character = run[0]
                    fence_length = len(run)
                    fence_container = container
                    in_indented_code = False
                    paragraph_open = False
                    mask(offset, offset + line_length)
                    offset += line_length
                    continue
        else:
            mask(offset, offset + line_length)
            if match:
                run = match.group(1)
                if run[0] == fence_character and len(run) >= fence_length:
                    tail = content[match.end() :]
                    if not tail.strip():
                        fence_character = None
                        fence_length = 0
                        fence_container = None
            offset += line_length
            continue

        if container != paragraph_container:
            paragraph_container = container
            paragraph_open = False
        if container != indented_container:
            indented_container = container
            in_indented_code = False
        if not content.strip():
            paragraph_open = False
        else:
            indented = _indent_prefix(content, 4) is not None
            if indented and (in_indented_code or not paragraph_open):
                mask(offset, offset + line_length)
                in_indented_code = True
                paragraph_open = False
            else:
                in_indented_code = False
                paragraph_open = not (
                    _MARKDOWN_NONPARAGRAPH_BLOCK.match(content.lstrip(" \t"))
                    or _MARKDOWN_REFERENCE_DEFINITION.match(content)
                )
        offset += line_length

    visible = "".join(masked)

    def escaped(position: int) -> bool:
        backslashes = 0
        position -= 1
        while position >= 0 and visible[position] == "\\":
            backslashes += 1
            position -= 1
        return backslashes % 2 == 1

    runs = tuple(
        match for match in re.finditer(r"`+", visible) if not escaped(match.start())
    )
    position = 0
    while position < len(runs):
        opening = runs[position]
        closing_index = position + 1
        while closing_index < len(runs):
            closing = runs[closing_index]
            if len(closing.group(0)) == len(opening.group(0)):
                mask(opening.start(), closing.end())
                position = closing_index + 1
                break
            closing_index += 1
        else:
            position += 1
    return "".join(masked)


def _inline_markdown_destinations(body: str) -> tuple[str, ...]:
    """Extract inline-link targets with balanced, escape-aware label parsing."""

    destinations: list[str] = []
    bracket_depth = 0
    index = 0
    while index < len(body):
        character = body[index]
        if character == "\\":
            index += 2
            continue
        if character == "[":
            bracket_depth += 1
            index += 1
            continue
        if character != "]" or bracket_depth < 1:
            index += 1
            continue
        bracket_depth -= 1
        if index + 1 >= len(body) or body[index + 1] != "(":
            index += 1
            continue

        cursor = index + 2
        while cursor < len(body) and body[cursor].isspace():
            cursor += 1
        if cursor >= len(body):
            break
        if body[cursor] == ")":
            index = cursor + 1
            continue
        if body[cursor] == "<":
            start = cursor
            cursor += 1
            while cursor < len(body):
                if body[cursor] == "\\" and cursor + 1 < len(body):
                    cursor += 2
                    continue
                if body[cursor] in "\r\n<":
                    break
                if body[cursor] == ">":
                    destinations.append(body[start : cursor + 1])
                    cursor += 1
                    break
                cursor += 1
            index = cursor
            continue

        start = cursor
        parenthesis_depth = 0
        while cursor < len(body):
            if body[cursor] == "\\" and cursor + 1 < len(body):
                cursor += 2
                continue
            if body[cursor] == "(":
                parenthesis_depth += 1
                cursor += 1
                continue
            if body[cursor] == ")":
                if parenthesis_depth == 0:
                    destinations.append(body[start:cursor])
                    cursor += 1
                    break
                parenthesis_depth -= 1
                cursor += 1
                continue
            if body[cursor].isspace() and parenthesis_depth == 0:
                destinations.append(body[start:cursor])
                break
            cursor += 1
        index = cursor
    return tuple(destinations)


def _local_markdown_destinations(body: str) -> tuple[str, ...]:
    destinations: list[str] = []
    visible_body = _mask_markdown_code(body)
    raw_destinations = list(_inline_markdown_destinations(visible_body))
    raw_destinations.extend(
        match.group(1) or match.group(2)
        for match in _MARKDOWN_REFERENCE_DEFINITION.finditer(visible_body)
    )
    raw_destinations.extend(
        match.group(1) for match in _MARKDOWN_AUTOLINK.finditer(visible_body)
    )
    for candidate in raw_destinations:
        raw = candidate.strip()
        if raw.startswith("<") and ">" in raw:
            raw = raw[1 : raw.index(">")]
        else:
            raw = raw.split(maxsplit=1)[0]
        raw = _MARKDOWN_BACKSLASH_ESCAPE.sub(r"\1", raw)
        if not raw or raw.startswith("#"):
            continue
        # CommonMark resolves HTML character references before URL parsing, but
        # URL schemes are classified before percent-decoding. Keep those stages
        # separate so ``https%3A/...`` remains a local path subject to the same
        # containment checks as every other bundled reference.
        rendered = html.unescape(raw)
        try:
            split = urlsplit(rendered)
        except ValueError as exc:
            raise SkillPathError("malformed link URL in SKILL.md") from exc
        if split.scheme:
            if split.scheme.lower() not in _REMOTE_SCHEMES:
                raise SkillPathError(
                    f"unsupported link scheme in SKILL.md: {split.scheme}"
                )
            continue
        if split.netloc:
            raise SkillPathError("network-path links are not bundled skill resources")
        destinations.append(unquote(split.path))
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

    entry_count = 0
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
            entry_count += 1
            if entry_count > MAX_FOLDER_ENTRIES:
                raise SkillFormatError(
                    f"skill folder exceeds {MAX_FOLDER_ENTRIES} filesystem entries"
                )
            if len(PurePosixPath(relative).parts) > MAX_FOLDER_DEPTH:
                raise SkillFormatError(
                    f"skill folder exceeds maximum depth {MAX_FOLDER_DEPTH}"
                )
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
    "MAX_FOLDER_DEPTH",
    "MAX_FOLDER_ENTRIES",
    "MAX_FOLDER_FILES",
    "MAX_SKILL_FILE_BYTES",
    "SKILL_FILENAME",
    "parse_skill_markdown",
    "serialize_skill_markdown",
    "validate_document_references",
    "validate_skill_folder",
    "validate_skill_name",
]
