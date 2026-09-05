"""Strict parsing and validation for folder-shaped procedural skills."""

from __future__ import annotations

import html
import json
import os
import re
import stat
from collections.abc import Callable
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

from .errors import SkillFormatError, SkillPathError
from .models import SkillDocument
from .paths import contained_path

SKILL_FILENAME = "SKILL.md"
GENERATION_FILENAME = ".kestrel-generation"
_GENERATION_PAYLOAD_RE = re.compile(rb"kestrel-skill-generation-v1:([0-9a-f]{64})\n\Z")
MAX_SKILL_FILE_BYTES = 262_144
MAX_DESCRIPTION_BYTES = 512
MAX_FOLDER_FILES = 256
MAX_FOLDER_ENTRIES = 512
MAX_FOLDER_DEPTH = 32
MAX_FOLDER_BYTES = 2_097_152
MAX_RESOURCE_PATH_BYTES = 1024
MAX_MARKDOWN_CONTAINER_DEPTH = 4096
PRIMARY_WRITER_CLAIM = ".SKILL.md.claim"
PRIMARY_WRITER_TEMP_PREFIX = ".SKILL.md.tmp."
SKILL_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9_-]{0,62}[a-z0-9])?$")
_FRONTMATTER_KEYS = frozenset({"name", "description"})
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_UNICODE_LINE_SEPARATOR = re.compile(r"[\x85\u2028\u2029]")
_YAML_PLAIN_SYNTAX = re.compile(r"(?:^[-?:](?:[ \t]|$)|:[ \t]|:$|[ \t]#)")
_YAML_IMPLICIT_WORD = re.compile(
    r"(?:null|true|false|yes|no|on|off|~|\.inf|[-+]?\.inf|\.nan)",
    re.IGNORECASE,
)
_YAML_TIMESTAMP = re.compile(r"\d{4}-\d{1,2}-\d{1,2}(?:[Tt]|[ \t]+|$)")
_YAML_SEXAGESIMAL = re.compile(r"[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+(?:\.[0-9_]*)?")
_MARKDOWN_REFERENCE_DEFINITION = re.compile(
    r"(?m)^[ \t]{0,3}\[(?:\\[^\r\n]|[^\]\\]){1,999}\]:[ \t]*"
    r"(?:\r?\n[ \t]{0,3})?"
    r"(?:<((?:\\[^\r\n]|[^<>\\\r\n])+)>|(\S+))"
)
_MARKDOWN_REFERENCE_DEFINITION_START = re.compile(
    r"^[ \t]{0,3}\[(?:\\[^\r\n]|[^\]\\]){1,999}\]:"
)
_MARKDOWN_REFERENCE_LABEL_PREFIX = re.compile(r"^[ \t]{0,3}\[")
_MARKDOWN_BACKSLASH_ESCAPE = re.compile(r"\\([!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~])")
_MARKDOWN_UNESCAPE_TOKEN = re.compile(
    r"\\([!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~])"
    r"|(&(?:#[xX][0-9A-Fa-f]{1,6}|#[0-9]{1,7}|[A-Za-z][A-Za-z0-9]{0,30});)"
)
_MARKDOWN_AUTOLINK = re.compile(r"<([A-Za-z][A-Za-z0-9+.-]{1,31}:[^<>\x00-\x20]*)>")
# A CommonMark fence may be indented by at most three *columns*. Any tab in
# the leading whitespace reaches at least column four, so it starts indented
# code rather than a fence. Matching tabs here would mask live Markdown on the
# following line when the pseudo-fence is left open.
_MARKDOWN_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_MARKDOWN_LIST_PREFIX = re.compile(r"([ ]{0,3})((?:[-+*]|\d{1,9}[.)]))([ \t]+)")
_MARKDOWN_SETEXT_UNDERLINE = re.compile(r"^ {0,3}(?:=+|-+)[ \t]*$")
_MARKDOWN_NONPARAGRAPH_BLOCK = re.compile(
    r"^(?:#{1,6}(?:[ \t]+|$)|(?:[*_-][ \t]*){3,}$)"
)
_MARKDOWN_RAW_HTML_TAG = re.compile(
    r"^<(script|pre|style|textarea)(?:[ \t]|>|$)", re.IGNORECASE
)
_MARKDOWN_BLOCK_HTML_TAG = re.compile(
    r"^</?(?:address|article|aside|base|basefont|blockquote|body|caption|center|"
    r"col|colgroup|dd|details|dialog|dir|div|dl|dt|fieldset|figcaption|figure|"
    r"footer|form|frame|frameset|h[1-6]|head|header|hr|html|iframe|legend|li|"
    r"link|main|menu|menuitem|nav|noframes|ol|optgroup|option|p|param|search|"
    r"section|summary|table|tbody|td|tfoot|th|thead|title|tr|track|ul)"
    r"(?:[ \t]|/?>|$)",
    re.IGNORECASE,
)
_MARKDOWN_COMPLETE_HTML_TAG = re.compile(
    r"^(?:"
    r"<[A-Za-z][A-Za-z0-9-]*"
    r"(?:[ \t]+[A-Za-z_:][A-Za-z0-9_.:-]*"
    r"(?:[ \t]*=[ \t]*(?:[^\"'=<>`\x00-\x20]+|'[^']*'|\"[^\"]*\"))?)*"
    r"[ \t]*/?>"
    r"|</[A-Za-z][A-Za-z0-9-]*[ \t]*>"
    r")[ \t]*$"
)
_REMOTE_SCHEMES = frozenset({"http", "https", "mailto"})
_HTML_URL_ATTRIBUTES = frozenset(
    {
        "action",
        "background",
        "cite",
        "classid",
        "codebase",
        "data",
        "formaction",
        "href",
        "icon",
        "longdesc",
        "manifest",
        "poster",
        "profile",
        "src",
        "usemap",
        "xlink:href",
    }
)
_HTML_UNSUPPORTED_URL_ATTRIBUTES = frozenset(
    {"archive", "imagesrcset", "ping", "srcdoc", "srcset", "style"}
)
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")
_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_READ_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_NONBLOCK", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)


@dataclass(frozen=True, slots=True)
class ValidatedSkillFolderEntry:
    """One descriptor-pinned entry captured during bounded validation."""

    path: str
    payload: bytes | None

    @property
    def is_directory(self) -> bool:
        return self.payload is None


@dataclass(frozen=True, slots=True)
class ValidatedSkillFolder:
    """A complete bounded snapshot of one validated skill folder."""

    document: SkillDocument
    entries: tuple[ValidatedSkillFolderEntry, ...]


def _utf8_bytes(value: str, *, label: str) -> bytes:
    try:
        return value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise SkillFormatError(f"{label} must be valid UTF-8 text") from exc


def _validate_resource_payload(payload: bytes, *, source: str) -> None:
    """Keep every inventoried file within the bounded UTF-8 read contract."""

    if len(payload) > MAX_SKILL_FILE_BYTES:
        raise SkillFormatError(f"{source} exceeds {MAX_SKILL_FILE_BYTES} bytes")
    try:
        payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SkillFormatError(f"{source} must be UTF-8 text") from exc


def validate_generation_payload(payload: bytes) -> None:
    """Require the fixed, bounded syntax of internal generation metadata."""

    if _GENERATION_PAYLOAD_RE.fullmatch(payload) is None:
        raise SkillFormatError(
            f"{GENERATION_FILENAME} is not valid Kestrel generation metadata"
        )


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
        inner = value[1:-1]
        offset = 0
        while offset < len(inner):
            if inner[offset] != "'":
                offset += 1
                continue
            if offset + 1 >= len(inner) or inner[offset + 1] != "'":
                raise SkillFormatError(
                    f"invalid single-quoted scalar for {key!r}; quote apostrophes twice"
                )
            offset += 2
        return inner.replace("''", "'")
    normalized_number = value.replace("_", "")
    is_implicit_number = False
    try:
        float(normalized_number)
    except ValueError:
        try:
            int(normalized_number, 0)
        except ValueError:
            pass
        else:
            is_implicit_number = True
    else:
        is_implicit_number = True
    if (
        value[0] in "!,&*#[{]}>|%@`"
        or _YAML_PLAIN_SYNTAX.search(value)
        or _YAML_IMPLICIT_WORD.fullmatch(value)
        or _YAML_TIMESTAMP.match(value)
        or _YAML_SEXAGESIMAL.fullmatch(value)
        or is_implicit_number
    ):
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
    body_lines = lines[closing + 1 :]
    first_content = next(
        (index for index, line in enumerate(body_lines) if line.strip()),
        len(body_lines),
    )
    last_content = next(
        (
            index
            for index in range(len(body_lines) - 1, first_content - 1, -1)
            if body_lines[index].strip()
        ),
        first_content - 1,
    )
    body = "\n".join(body_lines[first_content : last_content + 1])
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
        f"{document.body}\n"
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


def _leading_indent_columns(line: str) -> int:
    """Measure leading CommonMark indentation once for container matching."""

    columns = 0
    for character in line:
        if character == " ":
            columns += 1
        elif character == "\t":
            columns += 4 - (columns % 4)
        else:
            break
    return columns


def _list_marker_prefix(line: str, start: int = 0) -> tuple[int, int, str] | None:
    """Return consumed characters, continuation columns, and marker text."""

    match = _MARKDOWN_LIST_PREFIX.match(line, start)
    if match is None:
        return None
    marker_end = len(match.group(1)) + len(match.group(2))
    whitespace = match.group(3)
    if not line[match.end() :].strip():
        # For an empty list item CommonMark ignores the amount of whitespace
        # after the marker: subsequent blocks are indented by marker width +
        # one column. Treating every trailing space as content indentation can
        # turn a live reference definition into apparent indented code and let
        # it bypass containment validation.
        return len(line) - start, marker_end + 1, match.group(2)
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
    return consumed, marker_end + whitespace_columns, match.group(2)


def _strip_blockquote_prefix(line: str) -> str | None:
    """Strip one CommonMark blockquote marker without losing tab columns."""

    index = 0
    while index < len(line) and index < 3 and line[index] == " ":
        index += 1
    if index >= len(line) or line[index] != ">":
        return None

    column = index + 1
    index += 1
    retained_indent = 0
    if index < len(line) and line[index] == " ":
        column += 1
        index += 1
    elif index < len(line) and line[index] == "\t":
        width = 4 - (column % 4)
        column += width
        index += 1
        # A tab expands before block parsing. Only its first column is the
        # optional marker delimiter; the remaining columns indent the content.
        retained_indent = width - 1

    while index < len(line) and line[index] in " \t":
        if line[index] == " ":
            width = 1
        else:
            width = 4 - (column % 4)
        retained_indent += width
        column += width
        index += 1
    return f"{' ' * retained_indent}{line[index:]}"


def _html_block_terminator(line: str) -> tuple[str, bool, bool] | None:
    """Return marker, case handling, and paragraph-interruption behavior."""

    if line.startswith("<!--"):
        return "-->", False, True
    if line.startswith("<?"):
        return "?>", False, True
    if line.startswith("<![CDATA["):
        return "]]>", False, True
    if re.match(r"^<![A-Z]", line):
        return ">", False, True
    raw_tag = _MARKDOWN_RAW_HTML_TAG.match(line)
    if raw_tag is not None:
        return f"</{raw_tag.group(1)}>", True, True
    if _MARKDOWN_BLOCK_HTML_TAG.match(line):
        return "", False, True
    if _MARKDOWN_COMPLETE_HTML_TAG.match(line):
        # Type-7 complete tags end at a blank line but cannot interrupt an
        # existing paragraph.
        return "", False, False
    return None


def _html_block_ends(line: str, terminator: tuple[str, bool, bool]) -> bool:
    marker, case_insensitive, _can_interrupt_paragraph = terminator
    if not marker:
        return not line.strip()
    if case_insensitive:
        return marker.casefold() in line.casefold()
    return marker in line


def _reference_continuation_content(
    line: str,
    *,
    quote_depth: int,
    list_levels: list[tuple[int, int]],
) -> str | None:
    """Normalize one look-ahead line within an existing container."""

    content = line.rstrip("\r\n")
    for _ in range(quote_depth):
        stripped = _strip_blockquote_prefix(content)
        if stripped is None:
            # A reference destination on the next line is paragraph-like and
            # may lazily omit one or more enclosing blockquote markers.
            break
        content = stripped
    if list_levels:
        prefix = _indent_prefix(content, list_levels[-1][0])
        if prefix is not None:
            content = content[prefix:]
    return content


def _container_ids(
    levels: list[tuple[int, int]], *, blank_line: bool
) -> tuple[int, ...]:
    """Materialize list identities only for lines whose block state needs them."""

    if blank_line:
        return ()
    return tuple(item_id for _indent, item_id in levels)


def _reference_definition_from_lines(
    content: str,
    *,
    line_index: int,
    lines: list[str],
    quote_depth: int,
    list_levels: list[tuple[int, int]],
    claim_scan_work: Callable[[int], None],
) -> tuple[re.Match[str] | None, tuple[int, ...]]:
    """Resolve a bounded CommonMark reference definition across source lines."""

    claim_scan_work(max(1, len(content)) * 2)
    match = _MARKDOWN_REFERENCE_DEFINITION.match(content)
    if match is not None:
        return match, ()
    if _MARKDOWN_REFERENCE_LABEL_PREFIX.match(content) is None:
        return None, ()

    candidate = content
    continuation_lines: list[int] = []
    definition_started = (
        _MARKDOWN_REFERENCE_DEFINITION_START.match(candidate) is not None
    )
    # CommonMark caps labels at 999 characters. The extra allowance covers the
    # opening indentation, brackets, colon, and one bounded destination line.
    limit = 999 + MAX_RESOURCE_PATH_BYTES
    for continuation_index in range(line_index + 1, len(lines)):
        continuation = _reference_continuation_content(
            lines[continuation_index],
            quote_depth=quote_depth,
            list_levels=list_levels,
        )
        if continuation is None or not continuation.strip():
            break
        continuation_lines.append(continuation_index)
        if definition_started:
            # CommonMark accepts arbitrary additional indentation when the
            # destination starts on the line after a complete ``[label]:``.
            continuation = continuation.lstrip(" \t")
        claim_scan_work(len(continuation) + 1)
        candidate = f"{candidate}\n{continuation}"
        # A definition cannot become complete before a literal closing ``]:``
        # appears. Avoid rerunning the bounded-but-nonconstant regex against a
        # growing candidate for every line of an incomplete label.
        if definition_started or "]:" in continuation:
            claim_scan_work(len(candidate))
            match = _MARKDOWN_REFERENCE_DEFINITION.match(candidate)
            if match is not None:
                return match, tuple(continuation_lines)
        if len(candidate) > limit:
            break
        if not definition_started and "]:" in continuation:
            claim_scan_work(len(candidate))
            definition_started = (
                _MARKDOWN_REFERENCE_DEFINITION_START.match(candidate) is not None
            )
        if definition_started:
            # Once the label and colon are complete, only one destination line
            # can remain in a CommonMark reference definition.
            continue
        if "]" in continuation:
            break
    return None, ()


def _analyze_markdown(
    body: str,
) -> tuple[
    str,
    tuple[str, ...],
    tuple[tuple[int, int], ...],
    tuple[str, ...],
]:
    """Normalize containers, mask code, and find block-valid definitions."""

    masked: list[str] = []

    def mask(start: int, end: int) -> None:
        for position in range(start, end):
            if masked[position] not in "\r\n":
                masked[position] = " "

    list_levels: list[tuple[int, int]] = []
    list_quote_depth = 0
    next_list_id = 0
    fence_character: str | None = None
    fence_length = 0
    fence_container: tuple[int, tuple[int, ...]] | None = None
    indented_container: tuple[int, tuple[int, ...]] | None = None
    in_indented_code = False
    paragraph_container: tuple[int, tuple[int, ...]] | None = None
    paragraph_open = False
    inline_blocks: list[list[int]] = []
    active_inline_block: int | None = None
    html_block_terminator: tuple[str, bool, bool] | None = None
    html_block_container: tuple[int, tuple[int, ...]] | None = None
    reference_destinations: list[str] = []
    reference_continuation_lines: set[int] = set()
    html_blocks: list[list[str]] = []
    lines = body.splitlines(keepends=True)
    reference_scan_work = 0
    max_reference_scan_work = max(8192, len(body) * 8)

    def claim_reference_scan_work(amount: int) -> None:
        nonlocal reference_scan_work
        reference_scan_work += amount
        if reference_scan_work > max_reference_scan_work:
            raise SkillFormatError(
                "reference definition structure exceeds validation complexity limit"
            )

    for line_index, line in enumerate(lines):
        raw_content = line.rstrip("\r\n")
        ending = line[len(raw_content) :]
        quote_depth = 0
        content = raw_content
        while (stripped := _strip_blockquote_prefix(content)) is not None:
            quote_depth += 1
            if quote_depth > MAX_MARKDOWN_CONTAINER_DEPTH:
                raise SkillFormatError(
                    "Markdown container nesting exceeds validation complexity limit"
                )
            content = stripped
        previous_list_levels = list_levels
        previous_list_quote_depth = list_quote_depth

        if quote_depth != list_quote_depth:
            list_levels = []
        list_quote_depth = quote_depth
        active_level: int | None = None
        retained_levels: list[tuple[int, int]] = []
        blank_line = not content.strip()
        if blank_line:
            retained_levels = list_levels
            active_level = list_levels[-1][0] if list_levels else None
            continued_ids = ()
            content = ""
        else:
            available_indent = _leading_indent_columns(content)
            for level_index in range(len(list_levels) - 1, -1, -1):
                required_indent = list_levels[level_index][0]
                if required_indent > available_indent:
                    continue
                prefix = _indent_prefix(content, required_indent)
                assert prefix is not None
                active_level = required_indent
                retained_levels = list_levels[: level_index + 1]
                content_offset = prefix
                break
            else:
                content_offset = 0

            continued_ids = tuple(item_id for _indent, item_id in retained_levels)
            absolute_indent = active_level or 0
            marker_cursor = content_offset
            paragraph_may_interrupt = paragraph_open
            while True:
                while (
                    stripped := _strip_blockquote_prefix(content[marker_cursor:])
                ) is not None:
                    quote_depth += 1
                    if (
                        quote_depth + len(retained_levels)
                        > MAX_MARKDOWN_CONTAINER_DEPTH
                    ):
                        raise SkillFormatError(
                            "Markdown container nesting exceeds validation complexity limit"
                        )
                    content = content[:marker_cursor] + stripped
                marker = _list_marker_prefix(content, marker_cursor)
                if marker is None:
                    break
                consumed, continuation_columns, marker_text = marker
                interrupts_paragraph = False
                if paragraph_may_interrupt:
                    parent_ids = tuple(item_id for _indent, item_id in retained_levels)
                    parent_container = (quote_depth, parent_ids)
                    interrupts_paragraph = paragraph_container == parent_container
                ordered_start = (
                    int(marker_text[:-1]) if marker_text[0].isdigit() else None
                )
                remaining_start = marker_cursor + consumed
                if interrupts_paragraph and (
                    not content[remaining_start:].strip()
                    or (ordered_start is not None and ordered_start != 1)
                ):
                    break
                if quote_depth + len(retained_levels) >= MAX_MARKDOWN_CONTAINER_DEPTH:
                    raise SkillFormatError(
                        "Markdown container nesting exceeds validation complexity limit"
                    )
                absolute_indent += continuation_columns
                next_list_id += 1
                retained_levels.append((absolute_indent, next_list_id))
                active_level = absolute_indent
                marker_cursor = remaining_start
                paragraph_may_interrupt = False
            content = content[marker_cursor:]

        list_levels = retained_levels
        final_ids = _container_ids(retained_levels, blank_line=blank_line)
        container = (quote_depth, final_ids)
        continued_container = (quote_depth, continued_ids)
        # Missing quote/list markers may lazily continue an open paragraph,
        # including when only an outer prefix is repeated on the new line.
        # Resolve that before inline-block segmentation or indentation can
        # hide a multiline link. Explicit new containers and block constructs
        # that can interrupt a paragraph never use this exception.
        previous_quote_depth, previous_list_ids = paragraph_container or (0, ())
        current_quote_depth, current_list_ids = container
        compatible_container_prefix = (
            current_quote_depth <= previous_quote_depth
            and current_list_ids == previous_list_ids[: len(current_list_ids)]
        )
        boundary_indent = _leading_indent_columns(content)
        is_block_boundary = False
        if boundary_indent <= 3:
            stripped_for_boundary = content.lstrip(" \t")
            opened_boundary_html = _html_block_terminator(stripped_for_boundary)
            is_block_boundary = bool(
                _MARKDOWN_FENCE.match(content)
                or _MARKDOWN_NONPARAGRAPH_BLOCK.match(stripped_for_boundary)
                or (opened_boundary_html is not None and opened_boundary_html[2])
            )
        if (
            paragraph_open
            and paragraph_container is not None
            and container != paragraph_container
            and compatible_container_prefix
            and content.strip()
            and not is_block_boundary
        ):
            container = paragraph_container
            continued_container = paragraph_container
            list_levels = previous_list_levels
            list_quote_depth = previous_list_quote_depth
        offset = len(masked)
        masked.extend(f"{content}{ending}")
        line_length = len(content) + len(ending)
        if fence_character is not None and fence_container is not None:
            opening_quote_depth, opening_list_ids = fence_container
            current_quote_depth, continued_list_ids = continued_container
            continues_container = (
                current_quote_depth >= opening_quote_depth
                and continued_list_ids[: len(opening_list_ids)] == opening_list_ids
            )
            # Fenced code blocks cannot use lazy continuation lines to escape
            # their blockquote or list item in CommonMark. Some renderers keep
            # outdented prose inside such a fence, but containment validation
            # follows the stricter reference grammar and treats the line as
            # live Markdown as soon as the opening container ends.
            if not continues_container:
                fence_character = None
                fence_length = 0
                fence_container = None
        if (
            fence_character is None
            and html_block_terminator is not None
            and html_block_container is not None
        ):
            opening_quote_depth, opening_list_ids = html_block_container
            current_quote_depth, continued_list_ids = continued_container
            continues_container = (
                current_quote_depth >= opening_quote_depth
                and continued_list_ids[: len(opening_list_ids)] == opening_list_ids
            )
            if continues_container:
                in_indented_code = False
                paragraph_open = False
                active_inline_block = None
                html_blocks[-1].append(content)
                mask(offset, offset + line_length)
                marker = html_block_terminator[0]
                terminator_line = (
                    raw_content
                    if marker or html_block_container == (0, ())
                    else content
                )
                if _html_block_ends(terminator_line, html_block_terminator):
                    html_block_terminator = None
                    html_block_container = None
                continue
            html_block_terminator = None
            html_block_container = None
        if line_index in reference_continuation_lines:
            in_indented_code = False
            paragraph_container = container
            paragraph_open = False
            active_inline_block = None
            mask(offset, offset + line_length)
            continue
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
                    active_inline_block = None
                    mask(offset, offset + line_length)
                    continue
        else:
            active_inline_block = None
            mask(offset, offset + line_length)
            if match:
                run = match.group(1)
                if run[0] == fence_character and len(run) >= fence_length:
                    tail = content[match.end() :]
                    if not tail.strip():
                        fence_character = None
                        fence_length = 0
                        fence_container = None
            continue

        if container != paragraph_container:
            paragraph_container = container
            paragraph_open = False
            active_inline_block = None
        if container != indented_container:
            indented_container = container
            in_indented_code = False
        if not content.strip():
            paragraph_open = False
            active_inline_block = None
        else:
            indented = _indent_prefix(content, 4) is not None
            if indented and (in_indented_code or not paragraph_open):
                mask(offset, offset + line_length)
                in_indented_code = True
                paragraph_open = False
                active_inline_block = None
            else:
                in_indented_code = False
                stripped_content = content.lstrip(" \t")
                if not paragraph_open:
                    reference_definition, continuation_indexes = (
                        _reference_definition_from_lines(
                            content,
                            line_index=line_index,
                            lines=lines,
                            quote_depth=quote_depth,
                            list_levels=retained_levels,
                            claim_scan_work=claim_reference_scan_work,
                        )
                    )
                    reference_continuation_lines.update(continuation_indexes)
                else:
                    reference_definition = None
                can_start_block = _leading_indent_columns(content) <= 3
                nonparagraph_block = (
                    _MARKDOWN_NONPARAGRAPH_BLOCK.match(stripped_content)
                    if can_start_block
                    else None
                )
                setext_underline = bool(
                    paragraph_open and _MARKDOWN_SETEXT_UNDERLINE.match(content)
                )
                opened_html_block = (
                    _html_block_terminator(stripped_content)
                    if can_start_block
                    else None
                )
                if setext_underline:
                    paragraph_open = False
                    active_inline_block = None
                elif opened_html_block is not None and (
                    not paragraph_open or opened_html_block[2]
                ):
                    html_blocks.append([content])
                    mask(offset, offset + line_length)
                    if not _html_block_ends(content, opened_html_block):
                        html_block_terminator = opened_html_block
                        html_block_container = container
                    paragraph_open = False
                    active_inline_block = None
                elif reference_definition is not None:
                    reference_destinations.append(
                        reference_definition.group(1) or reference_definition.group(2)
                    )
                    mask(offset, offset + line_length)
                    paragraph_open = False
                    active_inline_block = None
                elif nonparagraph_block is not None:
                    # ATX headings contain inline Markdown, but thematic breaks
                    # and HTML blocks do not. They also cannot continue a span
                    # into a neighboring block.
                    if stripped_content.startswith("#"):
                        inline_blocks.append([offset, offset + line_length])
                    paragraph_open = False
                    active_inline_block = None
                elif paragraph_open and active_inline_block is not None:
                    inline_blocks[active_inline_block][1] = offset + line_length
                else:
                    inline_blocks.append([offset, offset + line_length])
                    active_inline_block = len(inline_blocks) - 1
                    paragraph_open = True
    visible = "".join(masked)

    def escaped(position: int, lower_bound: int) -> bool:
        backslashes = 0
        position -= 1
        while position >= lower_bound and visible[position] == "\\":
            backslashes += 1
            position -= 1
        return backslashes % 2 == 1

    for block_start, block_end in inline_blocks:
        runs = tuple(
            match
            for match in re.finditer(r"`+", visible[block_start:block_end])
            if not escaped(block_start + match.start(), block_start)
        )
        position = 0
        while position < len(runs):
            opening = runs[position]
            closing_index = position + 1
            while closing_index < len(runs):
                closing = runs[closing_index]
                if len(closing.group(0)) == len(opening.group(0)):
                    mask(
                        block_start + opening.start(),
                        block_start + closing.end(),
                    )
                    position = closing_index + 1
                    break
                closing_index += 1
            else:
                position += 1
    return (
        "".join(masked),
        tuple(reference_destinations),
        tuple((start, end) for start, end in inline_blocks),
        tuple("\n".join(block) for block in html_blocks),
    )


def _mask_markdown_code(body: str) -> str:
    """Normalize containers and mask code while preserving line boundaries."""

    return _analyze_markdown(body)[0]


def _inline_link_suffix(body: str, cursor: int) -> tuple[int | None, int]:
    """Return a complete link suffix end and the characters examined."""

    start = cursor
    if cursor >= len(body):
        return None, 0
    if body[cursor] == ")":
        return cursor + 1, 1
    if not body[cursor].isspace():
        return None, 1
    while cursor < len(body) and body[cursor].isspace():
        cursor += 1
    if cursor >= len(body):
        return None, cursor - start
    if body[cursor] == ")":
        return cursor + 1, cursor - start + 1
    opening = body[cursor]
    if opening not in {'"', "'", "("}:
        return None, cursor - start + 1
    closing = ")" if opening == "(" else opening
    cursor += 1
    while cursor < len(body):
        character = body[cursor]
        if character == "\\" and cursor + 1 < len(body):
            cursor += 2
            continue
        if character == closing:
            cursor += 1
            break
        if opening == "(" and character == "(":
            return None, cursor - start + 1
        cursor += 1
    else:
        return None, cursor - start
    while cursor < len(body) and body[cursor].isspace():
        cursor += 1
    end = cursor + 1 if cursor < len(body) and body[cursor] == ")" else None
    return end, cursor - start + (1 if cursor < len(body) else 0)


def _inline_markdown_destinations(body: str) -> tuple[str, ...]:
    """Extract inline-link targets with balanced, escape-aware label parsing."""

    destinations: list[str] = []
    scan_work = 0
    max_scan_work = max(1, len(body)) * 4

    def claim_scan_work(amount: int) -> None:
        nonlocal scan_work
        scan_work += amount
        if scan_work > max_scan_work:
            raise SkillFormatError(
                "inline link structure exceeds validation complexity limit"
            )

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
        claim_scan_work(cursor - (index + 2))
        if cursor >= len(body):
            break
        if body[cursor] == ")":
            index = cursor + 1
            continue
        if body[cursor] == "<":
            start = cursor
            cursor += 1
            completed = False
            destination_work = 1
            while cursor < len(body):
                if body[cursor] == "\\" and cursor + 1 < len(body):
                    destination_work += 2
                    cursor += 2
                    continue
                destination_work += 1
                if body[cursor] in "\r\n<":
                    break
                if body[cursor] == ">":
                    suffix_end, suffix_work = _inline_link_suffix(body, cursor + 1)
                    claim_scan_work(suffix_work)
                    if suffix_end is not None:
                        destinations.append(body[start : cursor + 1])
                        cursor = suffix_end
                        completed = True
                    break
                cursor += 1
            claim_scan_work(destination_work)
            # An invalid outer destination is literal CommonMark, but content
            # inside it can still begin another live link. Resume immediately
            # after the rejected ``<`` instead of skipping nested markup.
            index = cursor if completed else start + 1
            continue

        start = cursor
        parenthesis_depth = 0
        completed = False
        destination_work = 0
        while cursor < len(body):
            if body[cursor] == "\\" and cursor + 1 < len(body):
                destination_work += 2
                cursor += 2
                continue
            destination_work += 1
            if body[cursor] == "(":
                parenthesis_depth += 1
                cursor += 1
                continue
            if body[cursor] == ")":
                if parenthesis_depth == 0:
                    destinations.append(body[start:cursor])
                    cursor += 1
                    completed = True
                    break
                parenthesis_depth -= 1
                cursor += 1
                continue
            if body[cursor].isspace() and parenthesis_depth == 0:
                suffix_end, suffix_work = _inline_link_suffix(body, cursor)
                claim_scan_work(suffix_work)
                if suffix_end is not None:
                    destinations.append(body[start:cursor])
                    cursor = suffix_end
                    completed = True
                break
            cursor += 1
        claim_scan_work(destination_work)
        # A destination with unmatched parentheses is literal CommonMark, so a
        # nested label inside it can still form a live link. Re-enter the
        # rejected destination instead of skipping every nested candidate.
        index = cursor if completed else start + 1
    return tuple(destinations)


def _escaped_at(body: str, position: int, *, lower_bound: int = 0) -> bool:
    backslashes = 0
    position -= 1
    while position >= lower_bound and body[position] == "\\":
        backslashes += 1
        position -= 1
    return backslashes % 2 == 1


def _angle_destination(candidate: str) -> str | None:
    """Strip an angle destination using its first unescaped closing delimiter."""

    if not candidate.startswith("<"):
        return None
    cursor = 1
    while cursor < len(candidate):
        if candidate[cursor] == "\\" and cursor + 1 < len(candidate):
            cursor += 2
            continue
        if candidate[cursor] == ">":
            return candidate[1:cursor]
        cursor += 1
    return None


class _RawHTMLDestinationParser(HTMLParser):
    """Collect single-URL HTML attributes and reject ambiguous URL carriers."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.destinations: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        normalized: dict[str, str | None] = {}
        for raw_name, value in attrs:
            name = raw_name.casefold()
            normalized.setdefault(name, value)
            if name in _HTML_UNSUPPORTED_URL_ATTRIBUTES:
                raise SkillPathError(
                    f"unsupported URL-bearing raw HTML attribute in SKILL.md: {name}"
                )
            if name in _HTML_URL_ATTRIBUTES and value:
                self.destinations.append(value)
        if (
            tag.casefold() == "meta"
            and (normalized.get("http-equiv") or "").casefold() == "refresh"
        ):
            raise SkillPathError("HTML meta refresh is not supported in SKILL.md")


def _commonmark_html_space_end(body: str, cursor: int) -> int | None:
    """Consume CommonMark HTML spacing with at most one line ending."""

    line_ending_seen = False
    while cursor < len(body) and body[cursor] in " \t\n":
        if body[cursor] == "\n":
            if line_ending_seen:
                return None
            line_ending_seen = True
        cursor += 1
    return cursor


def _commonmark_html_tag_end(body: str, start: int) -> int | None:
    """Return the exclusive end of one complete CommonMark open/close tag."""

    cursor = start + 1
    closing = cursor < len(body) and body[cursor] == "/"
    if closing:
        cursor += 1
    if cursor >= len(body) or not body[cursor].isascii() or not body[cursor].isalpha():
        return None
    cursor += 1
    while cursor < len(body) and (
        (body[cursor].isascii() and body[cursor].isalnum()) or body[cursor] == "-"
    ):
        cursor += 1

    if closing:
        cursor = _commonmark_html_space_end(body, cursor)
        if cursor is None or cursor >= len(body) or body[cursor] != ">":
            return None
        return cursor + 1

    separator_available = False
    while True:
        if separator_available:
            separator_seen = True
            separator_available = False
        else:
            separator_start = cursor
            cursor = _commonmark_html_space_end(body, cursor)
            if cursor is None:
                return None
            separator_seen = cursor != separator_start
        if cursor >= len(body):
            return None
        if body[cursor] == ">":
            return cursor + 1
        if body.startswith("/>", cursor):
            return cursor + 2
        if not separator_seen:
            return None
        if body[cursor] not in "_:" and not (
            body[cursor].isascii() and body[cursor].isalpha()
        ):
            return None
        cursor += 1
        while cursor < len(body) and (
            body[cursor] in "_.:-"
            or (body[cursor].isascii() and body[cursor].isalnum())
        ):
            cursor += 1
        after_name = _commonmark_html_space_end(body, cursor)
        if after_name is None:
            return None
        if after_name >= len(body) or body[after_name] != "=":
            separator_available = after_name > cursor
            cursor = after_name
            continue
        cursor = _commonmark_html_space_end(body, after_name + 1)
        if cursor is None or cursor >= len(body):
            return None
        quote = body[cursor] if body[cursor] in "\"'" else None
        if quote is not None:
            closing_quote = body.find(quote, cursor + 1)
            if closing_quote < 0:
                return None
            cursor = closing_quote + 1
            continue
        value_start = cursor
        while cursor < len(body) and body[cursor] not in " \t\n\"'=<>`":
            cursor += 1
        if cursor == value_start:
            return None


def _commonmark_special_terminator_end(
    body: str,
    cursor: int,
    terminator: str,
    missing_terminators: set[str],
) -> int | None:
    """Find a special token end without repeating a known-empty suffix scan."""

    if terminator in missing_terminators:
        return None
    ending = body.find(terminator, cursor)
    if ending < 0:
        # Every later opener has a strict suffix of the range just searched.
        # Remember the miss so adversarial repeated openers remain linear.
        missing_terminators.add(terminator)
        return None
    return ending + len(terminator)


def _commonmark_special_html_end(
    body: str,
    start: int,
    *,
    missing_terminators: set[str],
) -> int | None:
    """Return the end of a complete comment, PI, declaration, or CDATA token."""

    if body.startswith("<!-->", start):
        return start + len("<!-->")
    if body.startswith("<!--->", start):
        return start + len("<!--->")
    if body.startswith("<!--", start):
        # CommonMark 0.31.2 permits bare ``--`` inside comment text; the first
        # complete ``-->`` sequence is the only delimiter that matters here.
        return _commonmark_special_terminator_end(
            body,
            start + len("<!--"),
            "-->",
            missing_terminators,
        )
    if body.startswith("<?", start):
        return _commonmark_special_terminator_end(
            body,
            start + len("<?"),
            "?>",
            missing_terminators,
        )
    if body.startswith("<![CDATA[", start):
        return _commonmark_special_terminator_end(
            body,
            start + len("<![CDATA["),
            "]]>",
            missing_terminators,
        )
    if not body.startswith("<!", start):
        return None
    cursor = start + len("<!")
    if cursor >= len(body) or not body[cursor].isascii() or not body[cursor].isalpha():
        return None
    return _commonmark_special_terminator_end(
        body,
        cursor + 1,
        ">",
        missing_terminators,
    )


def _mask_non_commonmark_html_openers(body: str) -> str:
    """Neutralize text that Python's HTML parser could mistake for live HTML."""

    masked = list(body)
    cursor = 0
    missing_special_terminators: set[str] = set()
    while True:
        position = body.find("<", cursor)
        if position < 0:
            break
        if _escaped_at(body, position):
            masked[position] = " "
            cursor = position + 1
            continue
        if body.startswith(("<!", "<?"), position):
            ending = _commonmark_special_html_end(
                body,
                position,
                missing_terminators=missing_special_terminators,
            )
            if ending is None:
                # An incomplete declaration is literal CommonMark. Mask only
                # its opener so a later complete URL-bearing tag remains live.
                masked[position] = " "
                cursor = position + 1
            else:
                # Complete comments/instructions/declarations/CDATA cannot
                # carry live HTML attributes; remove the whole token so nested
                # tag-shaped text remains literal to HTMLParser as well.
                masked[position:ending] = " " * (ending - position)
                cursor = ending
            continue
        if _commonmark_html_tag_end(body, position) is None:
            # Incomplete open/close tags are literal CommonMark and must not
            # hold HTMLParser in a quoted attribute across a later live tag.
            masked[position] = " "
        cursor = position + 1
    return "".join(masked)


def _raw_html_destinations(fragments: tuple[str, ...]) -> tuple[str, ...]:
    destinations: list[str] = []
    for fragment in fragments:
        # Parser state never crosses CommonMark block boundaries. Otherwise an
        # incomplete literal in one paragraph can conceal a tag in the next.
        parser = _RawHTMLDestinationParser()
        parser.feed(fragment)
        parser.close()
        destinations.extend(parser.destinations)
    return tuple(destinations)


def _unescape_commonmark_destination(value: str) -> str:
    """Resolve exactly one CommonMark escape or character-reference token."""

    def replace(match: re.Match[str]) -> str:
        escaped = match.group(1)
        if escaped is not None:
            return escaped
        entity = match.group(2)
        assert entity is not None
        if entity.startswith("&#"):
            return html.unescape(entity)
        return html.entities.html5.get(entity[1:], entity)

    return _MARKDOWN_UNESCAPE_TOKEN.sub(replace, value)


def _local_markdown_destinations(body: str) -> tuple[str, ...]:
    destinations: list[str] = []
    visible_body, reference_destinations, inline_blocks, html_blocks = (
        _analyze_markdown(body)
    )
    raw_destinations: list[tuple[str, bool]] = []
    inline_html_fragments: list[str] = []
    for block_start, block_end in inline_blocks:
        inline_body = visible_body[block_start:block_end]
        raw_destinations.extend(
            (destination, False)
            for destination in _inline_markdown_destinations(inline_body)
        )
        raw_destinations.extend(
            (match.group(1), False)
            for match in _MARKDOWN_AUTOLINK.finditer(inline_body)
            if not _escaped_at(inline_body, match.start())
        )
        inline_html_fragments.append(_mask_non_commonmark_html_openers(inline_body))
    raw_destinations.extend(
        (destination, False) for destination in reference_destinations
    )
    raw_destinations.extend(
        (destination, True)
        for destination in _raw_html_destinations(
            tuple(inline_html_fragments) + html_blocks
        )
    )
    for candidate, character_references_decoded in raw_destinations:
        raw = candidate.strip()
        angle_destination = _angle_destination(raw)
        if angle_destination is not None:
            raw = angle_destination
        elif not raw.startswith("<"):
            raw = raw.split(maxsplit=1)[0]
        if character_references_decoded:
            raw = _MARKDOWN_BACKSLASH_ESCAPE.sub(r"\1", raw)
        else:
            raw = _unescape_commonmark_destination(raw)
        if not raw or raw.startswith("#"):
            continue
        # HTMLParser has already resolved character references in raw HTML
        # attributes. Markdown escapes and semicolon-terminated character
        # references were resolved together above so replacement text cannot
        # be decoded a second time. URL schemes are classified before percent-
        # decoding so ``https%3A/...`` remains a local path subject to checks.
        rendered = raw
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
        if not split.path:
            continue
        destinations.append(unquote(split.path))
    return tuple(destinations)


def validate_document_references(document: SkillDocument, folder: Path) -> None:
    """Require every bundled local link to resolve inside ``folder``."""

    for destination in _local_markdown_destinations(document.body):
        contained_path(folder, destination, must_exist=True)


def _direct_relative_parts(relative: object) -> tuple[str, ...]:
    if not isinstance(relative, str) or not relative or "\x00" in relative:
        raise SkillPathError("path must be a non-empty string without NUL bytes")
    if "\\" in relative or relative.startswith("/") or _WINDOWS_DRIVE.match(relative):
        raise SkillPathError("absolute and backslash paths are not allowed")
    pure = PurePosixPath(relative)
    if (
        not pure.parts
        or pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise SkillPathError("path traversal is not allowed")
    if pure.as_posix() != relative:
        raise SkillPathError("path must use its canonical POSIX spelling")
    return pure.parts


def validate_resource_path(relative: object) -> str:
    """Validate one portable, HTTP-addressable path inside a skill folder."""

    _direct_relative_parts(relative)
    assert isinstance(relative, str)  # established above
    if (
        len(_utf8_bytes(relative, label="skill resource path"))
        > MAX_RESOURCE_PATH_BYTES
    ):
        raise SkillFormatError(
            f"skill resource path exceeds {MAX_RESOURCE_PATH_BYTES} UTF-8 bytes"
        )
    return relative


def _open_pinned_directory_at(
    parent_fd: int,
    name: str,
    *,
    expected: tuple[int, int] | None = None,
) -> int:
    try:
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    except OSError as exc:
        raise SkillPathError("skill directory changed during validation") from exc
    opened = os.fstat(descriptor)
    identity = (opened.st_dev, opened.st_ino)
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISDIR(before.st_mode)
        or not stat.S_ISDIR(opened.st_mode)
        or identity != (before.st_dev, before.st_ino)
        or (expected is not None and identity != expected)
    ):
        os.close(descriptor)
        raise SkillPathError("skill directory changed during validation")
    return descriptor


def _read_regular_file_at(directory_fd: int, name: str, *, max_bytes: int) -> bytes:
    try:
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as exc:
        raise SkillFormatError(f"could not read regular file: {name}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise SkillPathError(f"skill resources must be regular files: {name}")
    try:
        descriptor = os.open(name, _READ_FLAGS, dir_fd=directory_fd)
    except OSError as exc:
        raise SkillFormatError(f"could not read regular file: {name}") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
            before.st_dev,
            before.st_ino,
        ):
            raise SkillPathError(f"skill resources must be regular files: {name}")
        opened_snapshot = (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        payload = bytearray()
        while len(payload) <= max_bytes:
            chunk = os.read(descriptor, min(65_536, max_bytes + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
        if len(payload) > max_bytes:
            raise SkillFormatError(f"{name} exceeds {max_bytes} bytes")
        after = os.fstat(descriptor)
        if (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) != opened_snapshot:
            raise SkillPathError(f"skill resource changed during validation: {name}")
        return bytes(payload)
    finally:
        os.close(descriptor)


def _validate_reference_in_snapshot(
    relative: str,
    captured: dict[str, ValidatedSkillFolderEntry],
) -> None:
    """Resolve one bundled link against the bytes and names already captured."""

    parts = _direct_relative_parts(relative)
    normalized = PurePosixPath(*parts).as_posix()
    if normalized == GENERATION_FILENAME:
        raise SkillPathError("internal skill generation metadata is not a resource")
    if normalized not in captured:
        raise SkillPathError(f"path does not exist: {relative}")
    for index in range(1, len(parts)):
        parent = PurePosixPath(*parts[:index]).as_posix()
        entry = captured.get(parent)
        if entry is None or not entry.is_directory:
            raise SkillPathError(f"path does not exist: {relative}")


def inspect_skill_folder_descriptor(
    folder_fd: int,
    *,
    folder_name: str,
) -> ValidatedSkillFolder:
    """Validate and capture a skill through one pinned, bounded traversal."""

    entry_count = 0
    file_count = 0
    byte_count = 0
    exact_primary_seen = False
    captured: list[ValidatedSkillFolderEntry] = []

    def scan(directory_fd: int, parents: tuple[str, ...]) -> None:
        nonlocal entry_count, file_count, byte_count, exact_primary_seen
        try:
            entries = os.scandir(directory_fd)
        except OSError as exc:
            raise SkillFormatError("could not scan the complete skill folder") from exc
        try:
            for entry in entries:
                name = entry.name
                relative_parts = (*parents, name)
                relative = PurePosixPath(*relative_parts).as_posix()
                internal_generation = not parents and name == GENERATION_FILENAME
                if not parents and name == SKILL_FILENAME:
                    exact_primary_seen = True
                entry_count += 0 if internal_generation else 1
                if entry_count > MAX_FOLDER_ENTRIES:
                    raise SkillFormatError(
                        f"skill folder exceeds {MAX_FOLDER_ENTRIES} filesystem entries"
                    )
                if len(relative_parts) > MAX_FOLDER_DEPTH:
                    raise SkillFormatError(
                        f"skill folder exceeds maximum depth {MAX_FOLDER_DEPTH}"
                    )
                validate_resource_path(relative)
                if not parents and (
                    name == PRIMARY_WRITER_CLAIM
                    or name.startswith(PRIMARY_WRITER_TEMP_PREFIX)
                ):
                    raise SkillFormatError(
                        f"top-level resource name is reserved by the skill writer: {name}"
                    )
                try:
                    value = os.stat(
                        name,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                except OSError as exc:
                    raise SkillFormatError(
                        "could not scan the complete skill folder"
                    ) from exc
                if stat.S_ISLNK(value.st_mode):
                    raise SkillPathError(
                        f"symlinks are not allowed in skill folders: {name}"
                    )
                if stat.S_ISDIR(value.st_mode):
                    captured.append(ValidatedSkillFolderEntry(relative, None))
                    child = _open_pinned_directory_at(
                        directory_fd,
                        name,
                        expected=(value.st_dev, value.st_ino),
                    )
                    try:
                        scan(child, relative_parts)
                    finally:
                        os.close(child)
                    continue
                if not stat.S_ISREG(value.st_mode):
                    raise SkillPathError(
                        f"skill resources must be regular files: {name}"
                    )
                payload = _read_regular_file_at(
                    directory_fd,
                    name,
                    max_bytes=MAX_SKILL_FILE_BYTES,
                )
                _validate_resource_payload(payload, source=relative)
                if internal_generation:
                    validate_generation_payload(payload)
                file_count += 0 if internal_generation else 1
                byte_count += 0 if internal_generation else len(payload)
                if file_count > MAX_FOLDER_FILES:
                    raise SkillFormatError(
                        f"skill folder exceeds {MAX_FOLDER_FILES} files"
                    )
                if byte_count > MAX_FOLDER_BYTES:
                    raise SkillFormatError(
                        f"skill folder exceeds {MAX_FOLDER_BYTES} bytes"
                    )
                captured.append(ValidatedSkillFolderEntry(relative, payload))
        except OSError as exc:
            raise SkillFormatError("could not scan the complete skill folder") from exc
        finally:
            entries.close()

    scan(folder_fd, ())
    if not exact_primary_seen:
        raise SkillFormatError("skill folder must contain exact-case SKILL.md")
    captured_entries = tuple(captured)
    captured_by_path = {entry.path: entry for entry in captured_entries}
    primary_entry = captured_by_path.get(SKILL_FILENAME)
    if primary_entry is None or primary_entry.payload is None:
        raise SkillPathError(f"skill resources must be regular files: {SKILL_FILENAME}")
    document = parse_skill_markdown(
        primary_entry.payload,
        source=f"{folder_name}/{SKILL_FILENAME}",
    )
    if document.name != folder_name:
        raise SkillFormatError(
            f"frontmatter name {document.name!r} must match folder name {folder_name!r}"
        )
    for destination in _local_markdown_destinations(document.body):
        _validate_reference_in_snapshot(destination, captured_by_path)
    return ValidatedSkillFolder(document, captured_entries)


def validate_skill_folder_descriptor(
    folder_fd: int,
    *,
    folder_name: str,
) -> SkillDocument:
    """Validate a skill entirely through an already-pinned folder descriptor."""

    return inspect_skill_folder_descriptor(
        folder_fd,
        folder_name=folder_name,
    ).document


def inspect_skill_folder(folder: Path, *, source_root: Path) -> ValidatedSkillFolder:
    """Capture a complete skill through a pinned and bounded path traversal."""

    if folder.is_symlink():
        raise SkillPathError("skill folders must not be symlinks")
    if not folder.is_dir():
        raise SkillFormatError("skill candidate is not a directory")
    contained_path(source_root, folder.name)
    try:
        identity = folder.lstat()
        root_fd = os.open(source_root, _DIRECTORY_FLAGS)
    except OSError as exc:
        raise SkillPathError("skill source changed during validation") from exc
    try:
        folder_fd = _open_pinned_directory_at(
            root_fd,
            folder.name,
            expected=(identity.st_dev, identity.st_ino),
        )
        try:
            return inspect_skill_folder_descriptor(
                folder_fd,
                folder_name=folder.name,
            )
        finally:
            os.close(folder_fd)
    finally:
        os.close(root_fd)


def validate_skill_folder(folder: Path, *, source_root: Path) -> SkillDocument:
    """Validate the complete folder, including links and every symlink escape."""

    return inspect_skill_folder(folder, source_root=source_root).document


__all__ = [
    "MAX_DESCRIPTION_BYTES",
    "MAX_FOLDER_BYTES",
    "MAX_FOLDER_DEPTH",
    "MAX_FOLDER_ENTRIES",
    "MAX_FOLDER_FILES",
    "MAX_RESOURCE_PATH_BYTES",
    "MAX_SKILL_FILE_BYTES",
    "PRIMARY_WRITER_CLAIM",
    "PRIMARY_WRITER_TEMP_PREFIX",
    "SKILL_FILENAME",
    "ValidatedSkillFolder",
    "ValidatedSkillFolderEntry",
    "inspect_skill_folder",
    "inspect_skill_folder_descriptor",
    "parse_skill_markdown",
    "serialize_skill_markdown",
    "validate_document_references",
    "validate_resource_path",
    "validate_skill_folder",
    "validate_skill_folder_descriptor",
    "validate_skill_name",
]
