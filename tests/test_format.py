from __future__ import annotations

import os
import threading
from types import SimpleNamespace

import pytest

import kestrel_feature_skills.format as format_module
from kestrel_feature_skills.errors import SkillFormatError, SkillPathError
from kestrel_feature_skills.format import (
    MAX_DESCRIPTION_BYTES,
    MAX_FOLDER_DEPTH,
    MAX_FOLDER_ENTRIES,
    MAX_SKILL_FILE_BYTES,
    parse_skill_markdown,
    serialize_skill_markdown,
    validate_skill_folder,
)
from kestrel_feature_skills.models import SkillDocument


def document(name="safe-skill", description="Use when a colon: or apostrophe's needed"):
    return SkillDocument(name, description, "# Procedure\n\n1. Do the safe thing.")


def write_skill(root, value=None):
    value = value or document()
    folder = root / value.name
    folder.mkdir()
    (folder / "SKILL.md").write_text(serialize_skill_markdown(value), encoding="utf-8")
    return folder


def test_round_trip_quotes_unicode_and_colons():
    original = document(description="Use “carefully”: don't lose {braces}.")
    encoded = serialize_skill_markdown(original)
    assert parse_skill_markdown(encoded) == original
    assert 'description: "' in encoded


@pytest.mark.parametrize(
    "content, needle",
    [
        ("", "empty"),
        ("name: x", "start"),
        ("---\nname: x\n---\nbody", "description"),
        ("---\nname: x\nname: y\ndescription: z\n---\nbody", "duplicates"),
        ("---\nname: x\ndescription: z\ntags: nope\n---\nbody", "unsupported"),
        ("---\nname: x\ndescription: !tag bad\n---\nbody", "quote"),
        ("---\nname: X Upper\ndescription: z\n---\nbody", "skill name"),
        ("---\nname: x\ndescription: z\n---\n", "body"),
        ("\ufeff---\nname: x\ndescription: z\n---\nbody", "byte-order"),
    ],
)
def test_malformed_documents_fail_visibly(content, needle):
    with pytest.raises(SkillFormatError, match=needle):
        parse_skill_markdown(content)


def test_oversized_document_and_description_rejected():
    with pytest.raises(SkillFormatError, match="exceeds"):
        parse_skill_markdown(b"x" * (MAX_SKILL_FILE_BYTES + 1))
    too_long = "x" * (MAX_DESCRIPTION_BYTES + 1)
    with pytest.raises(SkillFormatError, match="description exceeds"):
        serialize_skill_markdown(document(description=too_long))


def test_json_escaped_lone_surrogate_is_a_visible_format_error():
    content = '---\nname: "surrogate"\ndescription: "\\ud800"\n---\n\nProcedure.\n'

    with pytest.raises(SkillFormatError, match="UTF-8"):
        parse_skill_markdown(content)


@pytest.mark.parametrize(
    "scalar",
    (
        "Use when: task arrives",
        "hello # comment",
        "'foo'bar'",
        "TRUE",
        "2026-09-04",
        "42",
        "1:20",
        ",flow-indicator",
    ),
)
def test_plain_or_single_quoted_scalars_cannot_change_yaml_meaning(scalar):
    content = f"---\nname: safe-skill\ndescription: {scalar}\n---\n\nProcedure.\n"

    with pytest.raises(SkillFormatError, match="quote|single-quoted"):
        parse_skill_markdown(content)


def test_single_quoted_scalar_accepts_only_yaml_doubled_quotes():
    content = (
        "---\nname: safe-skill\n"
        "description: 'Don''t lose the apostrophe'\n"
        "---\n\nProcedure.\n"
    )

    assert parse_skill_markdown(content).description == "Don't lose the apostrophe"


@pytest.mark.parametrize("separator", ("\\u0085", "\\u2028", "\\u2029"))
def test_json_escaped_unicode_line_separators_are_rejected(separator):
    content = (
        '---\nname: "line-break"\ndescription: "safe'
        f"{separator}"
        'injected"\n---\n\nProcedure.\n'
    )

    with pytest.raises(SkillFormatError, match="one printable line"):
        parse_skill_markdown(content)


@pytest.mark.parametrize("codepoint", (0x80, 0x9B, 0x9F))
@pytest.mark.parametrize("field", ("description", "body"))
def test_json_escaped_c1_controls_are_rejected(codepoint, field):
    control = f"\\u{codepoint:04x}"
    description = f'"safe{control}unsafe"' if field == "description" else '"safe"'
    decoded_control = chr(codepoint)
    body = f"safe{decoded_control}unsafe" if field == "body" else "Procedure."
    content = f'---\nname: "c1-control"\ndescription: {description}\n---\n\n{body}\n'

    with pytest.raises(SkillFormatError, match="control|printable"):
        parse_skill_markdown(content)


@pytest.mark.parametrize(
    "separator", ("\x0b", "\x0c", "\x1c", "\x1e", "\x85", "\u2028", "\u2029")
)
def test_literal_control_and_unicode_separators_cannot_be_normalized_away(separator):
    content = (
        '---\nname: "literal-separator"\ndescription: "Safe"\n---\n\n'
        f"before{separator}after\n"
    )

    with pytest.raises(SkillFormatError, match="control|line-separator"):
        parse_skill_markdown(content)


def test_serializer_rejects_lone_surrogate_body_as_a_format_error():
    with pytest.raises(SkillFormatError, match="UTF-8"):
        serialize_skill_markdown(
            SkillDocument("safe-skill", "Safe description", "bad\ud800")
        )


def test_folder_name_must_match_frontmatter(tmp_path):
    folder = tmp_path / "other-name"
    folder.mkdir()
    (folder / "SKILL.md").write_text(
        serialize_skill_markdown(document()), encoding="utf-8"
    )
    with pytest.raises(SkillFormatError, match="must match"):
        validate_skill_folder(folder, source_root=tmp_path)


def test_folder_requires_exact_case_primary_filename_on_case_insensitive_lookup(
    tmp_path, monkeypatch
):
    value = document(name="exact-primary")
    folder = tmp_path / value.name
    folder.mkdir()
    (folder / "skill.md").write_text(
        serialize_skill_markdown(value),
        encoding="utf-8",
    )
    real_read = format_module._read_regular_file_at

    def case_insensitive_read(directory_fd, name, *, max_bytes):
        if name == "SKILL.md":
            name = "skill.md"
        return real_read(directory_fd, name, max_bytes=max_bytes)

    monkeypatch.setattr(format_module, "_read_regular_file_at", case_insensitive_read)

    with pytest.raises(SkillFormatError, match="exact-case SKILL.md"):
        validate_skill_folder(folder, source_root=tmp_path)


def test_folder_scan_error_rejects_candidate(tmp_path, monkeypatch):
    folder = write_skill(tmp_path)

    def failing_scan(_directory_fd):
        raise PermissionError("subdirectory became unreadable")

    monkeypatch.setattr(format_module.os, "scandir", failing_scan)

    with pytest.raises(SkillFormatError, match="scan"):
        validate_skill_folder(folder, source_root=tmp_path)


def test_folder_rejects_resource_path_that_is_not_strict_utf8(tmp_path, monkeypatch):
    folder = write_skill(tmp_path)

    class SurrogateEntries:
        def __init__(self):
            self.remaining = [SimpleNamespace(name="resource-\udcff")]

        def __iter__(self):
            return iter(self.remaining)

        def close(self):
            return None

    monkeypatch.setattr(format_module.os, "scandir", lambda _fd: SurrogateEntries())

    with pytest.raises(SkillFormatError, match="UTF-8"):
        validate_skill_folder(folder, source_root=tmp_path)


def test_folder_rejects_path_longer_than_http_read_contract(tmp_path, monkeypatch):
    folder = write_skill(tmp_path)
    part = "a" * 200

    class OneDirectory:
        def __iter__(self):
            return iter((SimpleNamespace(name=part),))

        def close(self):
            return None

    next_descriptor = 10_000
    real_close = os.close

    def open_synthetic_child(*_args, **_kwargs):
        nonlocal next_descriptor
        next_descriptor += 1
        return next_descriptor

    def close_synthetic_child(descriptor):
        if descriptor < 10_000:
            real_close(descriptor)

    monkeypatch.setattr(format_module.os, "scandir", lambda _fd: OneDirectory())
    monkeypatch.setattr(
        format_module.os,
        "stat",
        lambda *_args, **_kwargs: SimpleNamespace(
            st_mode=format_module.stat.S_IFDIR | 0o700,
            st_dev=1,
            st_ino=1,
        ),
    )
    monkeypatch.setattr(
        format_module,
        "_open_pinned_directory_at",
        open_synthetic_child,
    )
    monkeypatch.setattr(format_module.os, "close", close_synthetic_child)
    directory_fd = os.open(
        folder,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        with pytest.raises(SkillFormatError, match="path exceeds 1024 UTF-8 bytes"):
            format_module.validate_skill_folder_descriptor(
                directory_fd,
                folder_name=folder.name,
            )
    finally:
        real_close(directory_fd)


@pytest.mark.parametrize(
    "payload, message",
    (
        (b"x" * (MAX_SKILL_FILE_BYTES + 1), "exceeds"),
        (b"text-prefix\xff", "UTF-8"),
    ),
)
def test_folder_rejects_resources_that_skill_read_cannot_open(
    tmp_path, payload, message
):
    folder = write_skill(tmp_path)
    (folder / "reference.txt").write_bytes(payload)

    with pytest.raises(SkillFormatError, match=message):
        validate_skill_folder(folder, source_root=tmp_path)


@pytest.mark.parametrize(
    "destination",
    ["../secret.md", "/etc/passwd", "..%2fsecret.md", "scripts\\evil.py"],
)
def test_markdown_reference_escape_rejected(tmp_path, destination):
    value = SkillDocument("escape", "Escape test", f"[outside]({destination})")
    folder = tmp_path / "escape"
    folder.mkdir()
    (folder / "SKILL.md").write_text(serialize_skill_markdown(value), encoding="utf-8")
    with pytest.raises(SkillPathError):
        validate_skill_folder(folder, source_root=tmp_path)


@pytest.mark.parametrize(
    "body",
    (
        "[outer [inner]](../secret.md)",
        r"[escaped \]](../secret.md)",
        r"[complex \]]: ../secret.md",
    ),
)
def test_markdown_reference_escape_with_complex_label_is_rejected(tmp_path, body):
    value = SkillDocument("complex-label", "Complex label", body)
    folder = write_skill(tmp_path, value)

    with pytest.raises(SkillPathError):
        validate_skill_folder(folder, source_root=tmp_path)


@pytest.mark.parametrize(
    "body, resource",
    (
        ("[outer [inner]](notes.md)", "notes.md"),
        (r"[escaped \]](notes\(one\).md)", "notes(one).md"),
        (r"[complex \]]: notes.md", "notes.md"),
    ),
)
def test_markdown_complex_label_and_destination_resolve_local_resource(
    tmp_path, body, resource
):
    value = SkillDocument("complex-local", "Complex local link", body)
    folder = write_skill(tmp_path, value)
    (folder / resource).write_text("notes", encoding="utf-8")

    assert validate_skill_folder(folder, source_root=tmp_path) == value


def test_local_resource_reference_must_exist_and_stay_inside(tmp_path):
    value = SkillDocument("linked", "Linked resource", "Read [notes](notes.md).")
    folder = write_skill(tmp_path, value)
    with pytest.raises(SkillPathError, match="does not exist"):
        validate_skill_folder(folder, source_root=tmp_path)
    (folder / "notes.md").write_text("notes", encoding="utf-8")
    assert validate_skill_folder(folder, source_root=tmp_path) == value


def test_reference_style_markdown_escape_is_rejected(tmp_path):
    value = SkillDocument(
        "reference-link",
        "Reference link",
        "Read [outside][secret].\n\n[secret]: ../secret.md",
    )
    folder = write_skill(tmp_path, value)
    with pytest.raises(SkillPathError):
        validate_skill_folder(folder, source_root=tmp_path)


def test_multiline_reference_style_markdown_escape_is_rejected(tmp_path):
    (tmp_path / "outside.md").write_text("outside", encoding="utf-8")
    value = SkillDocument(
        "multiline-reference",
        "Multiline reference link",
        "Read [outside][secret].\n\n[secret]:\n  ../outside.md",
    )
    folder = write_skill(tmp_path, value)

    with pytest.raises(SkillPathError):
        validate_skill_folder(folder, source_root=tmp_path)


def test_reference_definition_after_multiline_definition_is_validated(tmp_path):
    value = SkillDocument(
        "consecutive-references",
        "Consecutive reference definitions",
        "[notes]:\n  notes.md\n[outside]: ../secret.md\n\nRead [outside][outside].",
    )
    folder = write_skill(tmp_path, value)
    (folder / "notes.md").write_text("notes", encoding="utf-8")

    with pytest.raises(SkillPathError, match="traversal"):
        validate_skill_folder(folder, source_root=tmp_path)


def test_reference_style_markdown_local_resource_is_accepted(tmp_path):
    value = SkillDocument(
        "reference-local",
        "Reference link",
        'Read [notes][local].\n\n[local]: <notes.md> "Operator notes"',
    )
    folder = write_skill(tmp_path, value)
    (folder / "notes.md").write_text("notes", encoding="utf-8")
    assert validate_skill_folder(folder, source_root=tmp_path) == value


@pytest.mark.parametrize(
    "body",
    (
        "> [bad]: javascript:alert(1)\n>\n> [click][bad]",
        "- [bad]: javascript:alert(1)\n\n  [click][bad]",
    ),
)
def test_reference_definitions_inside_markdown_containers_are_validated(tmp_path, body):
    value = SkillDocument("container-link", "Container link", body)
    folder = write_skill(tmp_path, value)

    with pytest.raises(SkillPathError, match="unsupported link scheme"):
        validate_skill_folder(folder, source_root=tmp_path)


def test_reference_definitions_inside_container_fences_remain_literal(tmp_path):
    value = SkillDocument(
        "container-code",
        "Container code",
        "> ```markdown\n> [bad]: javascript:alert(1)\n> [click][bad]\n> ````",
    )
    folder = write_skill(tmp_path, value)

    assert validate_skill_folder(folder, source_root=tmp_path) == value


def test_live_link_after_unclosed_blockquote_fence_is_validated(tmp_path):
    value = SkillDocument(
        "quote-fence-exit",
        "Blockquote fence exit",
        "> ```markdown\n> literal example\n\n[outside](../secret.md)",
    )
    folder = write_skill(tmp_path, value)

    with pytest.raises(SkillPathError, match="traversal"):
        validate_skill_folder(folder, source_root=tmp_path)


def test_live_link_after_unclosed_list_fence_is_validated(tmp_path):
    value = SkillDocument(
        "list-fence-exit",
        "List fence exit",
        "- ```markdown\n  literal example\n\n[outside](../secret.md)",
    )
    folder = write_skill(tmp_path, value)

    with pytest.raises(SkillPathError, match="traversal"):
        validate_skill_folder(folder, source_root=tmp_path)


def test_non_one_ordered_marker_cannot_interrupt_paragraph_to_hide_live_link(
    tmp_path,
):
    value = SkillDocument(
        "ordered-interruption",
        "Ordered paragraph interruption",
        "Paragraph text\n2. ```markdown\n   [outside](../secret.md)\n   ```",
    )
    folder = write_skill(tmp_path, value)

    with pytest.raises(SkillPathError, match="traversal"):
        validate_skill_folder(folder, source_root=tmp_path)


def test_one_ordered_marker_can_interrupt_paragraph_with_literal_fence(tmp_path):
    value = SkillDocument(
        "ordered-one-interruption",
        "Ordered one paragraph interruption",
        "Paragraph text\n1. ```markdown\n   [literal](../example.md)\n   ```",
    )
    folder = write_skill(tmp_path, value)

    assert validate_skill_folder(folder, source_root=tmp_path) == value


def test_non_one_ordered_marker_can_start_list_after_blank_line(tmp_path):
    value = SkillDocument(
        "ordered-after-blank",
        "Ordered list after blank",
        "Paragraph text\n\n2. ```markdown\n   [literal](../example.md)\n   ```",
    )
    folder = write_skill(tmp_path, value)

    assert validate_skill_folder(folder, source_root=tmp_path) == value


def test_blank_list_item_cannot_interrupt_paragraph_to_hide_live_link(tmp_path):
    value = SkillDocument(
        "blank-list-interruption",
        "Blank list paragraph interruption",
        "Paragraph text\n*   \n    ~~~markdown\n    [outside](../secret.md)\n    ~~~",
    )
    folder = write_skill(tmp_path, value)

    with pytest.raises(SkillPathError, match="traversal"):
        validate_skill_folder(folder, source_root=tmp_path)


def test_tab_indented_pseudo_fence_does_not_hide_following_live_link(tmp_path):
    value = SkillDocument(
        "tab-pseudo-fence",
        "Tab-indented pseudo-fence",
        "Example:\n\n\t```markdown\n[outside](../secret.md)",
    )
    folder = write_skill(tmp_path, value)

    with pytest.raises(SkillPathError, match="traversal"):
        validate_skill_folder(folder, source_root=tmp_path)


@pytest.mark.parametrize(
    "body",
    (
        "- ```markdown\n  literal example\n  - [literal](../example.md)",
        "> ```markdown\n> literal example\n> > [literal](../example.md)",
    ),
)
def test_nested_container_lines_inside_unclosed_fence_remain_literal(tmp_path, body):
    value = SkillDocument("nested-fence", "Nested fence continuation", body)
    folder = write_skill(tmp_path, value)

    assert validate_skill_folder(folder, source_root=tmp_path) == value


@pytest.mark.parametrize(
    "body",
    (
        "> ```markdown\n> literal example\nSee [outside](../secret.md)",
        "- ```markdown\n  literal example\nSee [outside](../secret.md)",
        "> ```markdown\n> literal example\n[outside](../secret.md)",
        "- ```markdown\n  literal example\n[outside](../secret.md)",
    ),
)
def test_outdented_line_ends_container_fence_and_validates_live_link(tmp_path, body):
    value = SkillDocument("outdented-fence", "Outdented fence exit", body)
    folder = write_skill(tmp_path, value)

    with pytest.raises(SkillPathError, match="traversal"):
        validate_skill_folder(folder, source_root=tmp_path)


def test_new_list_item_ends_unclosed_fence_and_validates_live_link(tmp_path):
    value = SkillDocument(
        "sibling-fence",
        "Sibling list fence",
        "- ```markdown\n  literal example\n- [outside](../secret.md)",
    )
    folder = write_skill(tmp_path, value)

    with pytest.raises(SkillPathError, match="traversal"):
        validate_skill_folder(folder, source_root=tmp_path)


def test_reference_definition_in_ordered_list_continuation_is_validated(tmp_path):
    value = SkillDocument(
        "list-reference",
        "Ordered list reference",
        "10. [outside][target]\n\n    [target]: ../secret.md",
    )
    folder = write_skill(tmp_path, value)

    with pytest.raises(SkillPathError, match="traversal"):
        validate_skill_folder(folder, source_root=tmp_path)


def test_link_example_inside_indented_code_block_is_literal(tmp_path):
    value = SkillDocument(
        "indented-code",
        "Indented code link",
        "Example:\n\n    [literal](../example.md)",
    )
    folder = write_skill(tmp_path, value)

    assert validate_skill_folder(folder, source_root=tmp_path) == value


def test_link_example_inside_list_indented_code_block_is_literal(tmp_path):
    value = SkillDocument(
        "list-indented-code",
        "List indented code link",
        "- Example:\n\n      [literal](../example.md)",
    )
    folder = write_skill(tmp_path, value)

    assert validate_skill_folder(folder, source_root=tmp_path) == value


def test_symlink_file_escape_rejected(tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_text("secret", encoding="utf-8")
    value = SkillDocument("linked", "Linked resource", "Read [notes](notes.md).")
    folder = write_skill(tmp_path, value)
    os.symlink(outside, folder / "notes.md")
    with pytest.raises(SkillPathError, match="symlinks"):
        validate_skill_folder(folder, source_root=tmp_path)


def test_symlink_skill_folder_rejected(tmp_path):
    real_root = tmp_path / "real"
    real_root.mkdir()
    real = write_skill(real_root)
    source = tmp_path / "source"
    source.mkdir()
    os.symlink(real, source / "safe-skill")
    with pytest.raises(SkillPathError, match="symlink"):
        validate_skill_folder(source / "safe-skill", source_root=source)


def test_remote_https_link_is_not_treated_as_bundled_path(tmp_path):
    value = SkillDocument(
        "remote-link", "Remote reference", "See [docs](https://example.com/a)."
    )
    folder = write_skill(tmp_path, value)
    assert validate_skill_folder(folder, source_root=tmp_path) == value


def test_query_only_same_document_link_is_accepted(tmp_path):
    value = SkillDocument(
        "query-link", "Query-only reference", "Switch to [full view](?mode=full)."
    )
    folder = write_skill(tmp_path, value)

    assert validate_skill_folder(folder, source_root=tmp_path) == value


def test_inline_link_label_cannot_cross_commonmark_block_boundary(tmp_path):
    value = SkillDocument(
        "cross-block-label",
        "Cross-block label",
        "stray [\n\n# heading](../secret.md)",
    )
    folder = write_skill(tmp_path, value)

    assert validate_skill_folder(folder, source_root=tmp_path) == value


@pytest.mark.parametrize(
    "body",
    (
        "[x](../missing.md bad title)",
        '[x](../missing.md "unterminated"',
    ),
)
def test_incomplete_inline_link_with_whitespace_is_literal(tmp_path, body):
    value = SkillDocument("incomplete-link", "Incomplete inline link", body)
    folder = write_skill(tmp_path, value)

    assert validate_skill_folder(folder, source_root=tmp_path) == value


@pytest.mark.parametrize(
    "body, message",
    (
        ("[outer](<broken [escape](../secret.md))", "traversal"),
        (
            "[outer](<broken [execute](javascript:alert(1)))",
            "unsupported link scheme",
        ),
        ('[outer](<broken> "unterminated [escape](../secret.md))', "traversal"),
        ("[outer](<broken\n[escape](../secret.md))", "traversal"),
    ),
)
def test_malformed_angle_destination_cannot_hide_nested_live_link(
    tmp_path, body, message
):
    value = SkillDocument(
        "nested-angle-escape",
        "Nested angle escape",
        body,
    )
    folder = write_skill(tmp_path, value)

    with pytest.raises(SkillPathError, match=message):
        validate_skill_folder(folder, source_root=tmp_path)


@pytest.mark.parametrize(
    "body, message",
    (
        ("[outer]((broken [escape](../secret.md))", "traversal"),
        (
            "[outer]((broken [execute](javascript:alert(1)))",
            "unsupported link scheme",
        ),
    ),
)
def test_malformed_parenthesized_destination_cannot_hide_nested_live_link(
    tmp_path, body, message
):
    value = SkillDocument(
        "nested-parenthesis-escape",
        "Nested parenthesis escape",
        body,
    )
    folder = write_skill(tmp_path, value)

    with pytest.raises(SkillPathError, match=message):
        validate_skill_folder(folder, source_root=tmp_path)


def test_many_incomplete_inline_links_fail_with_bounded_validation_work(tmp_path):
    value = SkillDocument(
        "bounded-links",
        "Bounded malformed links",
        "[x](" * 1024,
    )
    folder = write_skill(tmp_path, value)

    with pytest.raises(SkillFormatError, match="complexity limit"):
        validate_skill_folder(folder, source_root=tmp_path)


def test_deep_list_continuation_bounds_indent_prefix_work(tmp_path, monkeypatch):
    marker_count = 1024
    value = SkillDocument(
        "bounded-list-indent",
        "Bounded nested list indentation",
        "- " * marker_count + "item\n" + " " * marker_count + "continuation\n",
    )
    folder = write_skill(tmp_path, value)
    real_indent_prefix = format_module._indent_prefix
    real_list_marker_prefix = format_module._list_marker_prefix
    indent_calls = 0
    marker_offsets = []

    def counted_indent_prefix(line, required_columns):
        nonlocal indent_calls
        indent_calls += 1
        return real_indent_prefix(line, required_columns)

    def recorded_list_marker_prefix(line, start=0):
        marker_offsets.append(start)
        return real_list_marker_prefix(line, start)

    monkeypatch.setattr(format_module, "_indent_prefix", counted_indent_prefix)
    monkeypatch.setattr(
        format_module,
        "_list_marker_prefix",
        recorded_list_marker_prefix,
    )

    validated = validate_skill_folder(folder, source_root=tmp_path)
    assert validated == SkillDocument(
        value.name,
        value.description,
        value.body.strip(),
    )
    assert indent_calls <= 4, (
        "list continuation rescanned indentation for each nesting level"
    )
    assert max(marker_offsets) >= 2 * (marker_count - 1), (
        "nested list parsing copied progressively shorter line suffixes"
    )


@pytest.mark.parametrize("marker", ("- ", "> "))
def test_markdown_container_depth_has_explicit_complexity_limit(marker):
    body = marker * (format_module.MAX_MARKDOWN_CONTAINER_DEPTH + 1) + "item"

    with pytest.raises(SkillFormatError, match="complexity limit"):
        format_module._mask_markdown_code(body)


def test_regular_file_reader_rejects_fifo_before_open(tmp_path, monkeypatch):
    resource = tmp_path / "resource"
    os.mkfifo(resource)
    directory_fd = os.open(
        tmp_path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    opened = False

    def unexpected_open(*_args, **_kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("FIFO must be rejected before open")

    monkeypatch.setattr(format_module.os, "open", unexpected_open)
    try:
        with pytest.raises(SkillPathError, match="regular file"):
            format_module._read_regular_file_at(
                directory_fd,
                resource.name,
                max_bytes=1024,
            )
    finally:
        os.close(directory_fd)

    assert not opened


def test_regular_file_reader_uses_nonblocking_open_across_fifo_swap(
    tmp_path, monkeypatch
):
    resource = tmp_path / "resource"
    resource.write_text("ordinary resource", encoding="utf-8")
    directory_fd = os.open(
        tmp_path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    real_open = os.open
    swapped = False
    completed = threading.Event()
    errors = []

    def swap_during_open(name, flags, *, dir_fd=None):
        nonlocal swapped
        if name == resource.name and dir_fd == directory_fd and not swapped:
            resource.unlink()
            os.mkfifo(resource)
            swapped = True
        return real_open(name, flags, dir_fd=dir_fd)

    monkeypatch.setattr(format_module.os, "open", swap_during_open)

    def read_swapped_resource():
        try:
            format_module._read_regular_file_at(
                directory_fd,
                resource.name,
                max_bytes=1024,
            )
        except Exception as exc:  # noqa: BLE001 - asserted below
            errors.append(exc)
        finally:
            completed.set()

    reader = threading.Thread(target=read_swapped_resource)
    reader.start()
    finished_without_writer = completed.wait(timeout=0.5)
    if not finished_without_writer:
        writer = real_open(resource, os.O_WRONLY | os.O_NONBLOCK)
        os.close(writer)
    reader.join(timeout=5)
    os.close(directory_fd)

    assert finished_without_writer, "resource read blocked while opening a FIFO"
    assert not reader.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], SkillPathError)
    assert "regular file" in str(errors[0])


def test_escaped_angle_destination_terminator_is_part_of_path(tmp_path):
    value = SkillDocument(
        "escaped-angle", "Escaped angle destination", r"[notes](<foo\>bar.md>)"
    )
    folder = write_skill(tmp_path, value)
    (folder / "foo>bar.md").write_text("notes", encoding="utf-8")

    assert validate_skill_folder(folder, source_root=tmp_path) == value


def test_escaped_angle_reference_destination_is_part_of_path(tmp_path):
    value = SkillDocument(
        "escaped-angle-reference",
        "Escaped angle reference destination",
        "[notes][target]\n\n[target]: <foo\\>bar.md>",
    )
    folder = write_skill(tmp_path, value)
    (folder / "foo>bar.md").write_text("notes", encoding="utf-8")

    assert validate_skill_folder(folder, source_root=tmp_path) == value


@pytest.mark.parametrize(
    "body",
    (
        '[notes](notes.md "double-quoted title")',
        "[notes](notes.md 'single-quoted title')",
        "[notes](notes.md (parenthesized title))",
        '[notes](<notes.md> "angle title")',
    ),
)
def test_complete_inline_link_titles_remain_validated(tmp_path, body):
    value = SkillDocument("titled-link", "Titled inline link", body)
    folder = write_skill(tmp_path, value)

    with pytest.raises(SkillPathError, match="does not exist"):
        validate_skill_folder(folder, source_root=tmp_path)


def test_escaped_commonmark_autolink_is_literal(tmp_path):
    value = SkillDocument(
        "escaped-autolink",
        "Escaped autolink",
        r"Do not interpret \<javascript:alert(1)> as a link.",
    )
    folder = write_skill(tmp_path, value)

    assert validate_skill_folder(folder, source_root=tmp_path) == value


def test_even_backslash_parity_does_not_escape_commonmark_autolink(tmp_path):
    value = SkillDocument(
        "live-autolink-parity",
        "Live autolink after escaped backslash",
        r"A literal backslash \\<javascript:alert(1)> precedes a live autolink.",
    )
    folder = write_skill(tmp_path, value)

    with pytest.raises(SkillPathError, match="unsupported link scheme"):
        validate_skill_folder(folder, source_root=tmp_path)


def test_executable_link_scheme_rejected(tmp_path):
    value = SkillDocument("bad-link", "Bad link", "[click](javascript:alert(1))")
    folder = write_skill(tmp_path, value)
    with pytest.raises(SkillPathError, match="unsupported link scheme"):
        validate_skill_folder(folder, source_root=tmp_path)


@pytest.mark.parametrize(
    "body, bundled_name",
    (
        ("[outside](..&sol;outside.md)", "..&sol;outside.md"),
        ("[click](javascript&colon;alert(1))", None),
        ("[click](javascript&#58;alert(1))", None),
        ("[click](javascript&#x3a;alert(1))", None),
    ),
)
def test_commonmark_entities_cannot_hide_escape_or_executable_scheme(
    tmp_path, body, bundled_name
):
    value = SkillDocument("entity-link", "Entity link", body)
    folder = write_skill(tmp_path, value)
    if bundled_name is not None:
        (folder / bundled_name).write_text("decoy", encoding="utf-8")

    with pytest.raises(SkillPathError, match="traversal|unsupported link scheme"):
        validate_skill_folder(folder, source_root=tmp_path)


def test_percent_encoded_colon_cannot_turn_a_local_escape_into_remote_url(tmp_path):
    value = SkillDocument(
        "encoded-scheme",
        "Encoded scheme delimiter",
        "[outside](https%3A/../../outside.md)",
    )
    folder = write_skill(tmp_path, value)

    with pytest.raises(SkillPathError, match="traversal"):
        validate_skill_folder(folder, source_root=tmp_path)


def test_malformed_link_url_is_a_visible_path_error(tmp_path):
    value = SkillDocument("bad-url", "Bad URL", "[broken](//[invalid)")
    folder = write_skill(tmp_path, value)

    with pytest.raises(SkillPathError, match="malformed link URL"):
        validate_skill_folder(folder, source_root=tmp_path)


def test_executable_commonmark_autolink_scheme_rejected(tmp_path):
    value = SkillDocument(
        "bad-autolink", "Bad autolink", "Do not open <javascript:alert(1)>."
    )
    folder = write_skill(tmp_path, value)

    with pytest.raises(SkillPathError, match="unsupported link scheme"):
        validate_skill_folder(folder, source_root=tmp_path)


def test_https_commonmark_autolink_is_accepted(tmp_path):
    value = SkillDocument(
        "safe-autolink", "Safe autolink", "Read <https://example.com/docs>."
    )
    folder = write_skill(tmp_path, value)

    assert validate_skill_folder(folder, source_root=tmp_path) == value


def test_markdown_links_inside_code_spans_and_fences_are_not_validated(tmp_path):
    value = SkillDocument(
        "code-links",
        "Code link examples",
        "Use `[inline](../example.md)` as a literal.\n\n"
        "```markdown\n[fenced](../example.md)\n```",
    )
    folder = write_skill(tmp_path, value)

    assert validate_skill_folder(folder, source_root=tmp_path) == value


def test_code_span_delimiters_cannot_pair_across_paragraphs(tmp_path):
    value = SkillDocument(
        "cross-paragraph-code",
        "Cross-paragraph code delimiters",
        "`unclosed\n\n[outside](../secret.md)\n\n`",
    )
    folder = write_skill(tmp_path, value)

    with pytest.raises(SkillPathError, match="traversal"):
        validate_skill_folder(folder, source_root=tmp_path)


def test_code_span_delimiters_cannot_pair_across_setext_heading(tmp_path):
    value = SkillDocument(
        "setext-code-boundary",
        "Setext heading code boundary",
        "Heading `\n===\n[outside](../secret.md) `",
    )
    folder = write_skill(tmp_path, value)

    with pytest.raises(SkillPathError, match="traversal"):
        validate_skill_folder(folder, source_root=tmp_path)


@pytest.mark.parametrize(
    "body",
    (
        "<!--\n`\n-->\n[outside](../secret.md) `",
        "<script>\n`\ncontent </script>\n[outside](../secret.md) `",
        "<?instruction\n`\n?>\n[outside](../secret.md) `",
        "<!DECLARATION\n`\n>\n[outside](../secret.md) `",
        "<![CDATA[\n`\n]]>\n[outside](../secret.md) `",
    ),
)
def test_code_span_delimiters_cannot_escape_multiline_html_block(tmp_path, body):
    value = SkillDocument(
        "html-code-boundary",
        "HTML block code boundary",
        body,
    )
    folder = write_skill(tmp_path, value)

    with pytest.raises(SkillPathError, match="traversal"):
        validate_skill_folder(folder, source_root=tmp_path)


@pytest.mark.parametrize(
    "body",
    (
        "> <!--\n> literal HTML\n[outside](../secret.md)",
        "- <div>\n  literal HTML\n[outside](../secret.md)",
    ),
)
def test_live_link_after_unclosed_container_html_block_is_validated(tmp_path, body):
    value = SkillDocument("html-container-exit", "HTML container exit", body)
    folder = write_skill(tmp_path, value)

    with pytest.raises(SkillPathError, match="traversal"):
        validate_skill_folder(folder, source_root=tmp_path)


def test_multiline_code_span_within_one_paragraph_remains_literal(tmp_path):
    value = SkillDocument(
        "multiline-code",
        "Multiline code span",
        "Use `[literal]\n(../example.md)` as a wrapped example.",
    )
    folder = write_skill(tmp_path, value)

    assert validate_skill_folder(folder, source_root=tmp_path) == value


def test_type_seven_html_tag_does_not_interrupt_multiline_code_span(tmp_path):
    value = SkillDocument(
        "inline-html-code",
        "Inline HTML inside code span",
        "Use `[example](../example.md)\n<a>\nend` as a wrapped example.",
    )
    folder = write_skill(tmp_path, value)

    assert validate_skill_folder(folder, source_root=tmp_path) == value


@pytest.mark.parametrize(
    "body",
    (
        "<!--\n[example](../example.md)\n-->",
        "<div>\n[example](../example.md)\n\nAfter the block.",
        "<a>\n[example](../example.md)\n\nAfter the block.",
    ),
)
def test_markdown_links_inside_html_blocks_remain_literal(tmp_path, body):
    value = SkillDocument("html-literal", "Literal HTML contents", body)
    folder = write_skill(tmp_path, value)

    assert validate_skill_folder(folder, source_root=tmp_path) == value


def test_reference_definition_shape_cannot_interrupt_open_paragraph(tmp_path):
    value = SkillDocument(
        "paragraph-reference-shape",
        "Paragraph reference shape",
        "Introductory paragraph\n[example]: ../missing.md",
    )
    folder = write_skill(tmp_path, value)

    assert validate_skill_folder(folder, source_root=tmp_path) == value


def test_code_spans_within_setext_heading_and_after_html_remain_literal(tmp_path):
    value = SkillDocument(
        "block-code-controls",
        "Block code span controls",
        "Heading `[heading](../example.md)`\n===\n\n"
        "<!-- comment -->\n"
        "Use `[paragraph](../example.md)` as a literal.",
    )
    folder = write_skill(tmp_path, value)

    assert validate_skill_folder(folder, source_root=tmp_path) == value


def test_blockquote_tab_stop_does_not_hide_live_link(tmp_path):
    value = SkillDocument(
        "quote-tab-link",
        "Blockquote tab stop",
        "> \t[outside](../secret.md)",
    )
    folder = write_skill(tmp_path, value)

    with pytest.raises(SkillPathError, match="traversal"):
        validate_skill_folder(folder, source_root=tmp_path)


def test_blockquote_tabs_with_four_content_columns_remain_literal_code(tmp_path):
    value = SkillDocument(
        "quote-tab-code",
        "Blockquote indented code",
        ">\t\t[literal](../example.md)",
    )
    folder = write_skill(tmp_path, value)

    assert validate_skill_folder(folder, source_root=tmp_path) == value


def test_indented_live_autolink_is_not_mistaken_for_a_code_block(tmp_path):
    value = SkillDocument(
        "indented-autolink",
        "Indented autolink",
        "Paragraph continuation.\n    <javascript:alert(1)>",
    )
    folder = write_skill(tmp_path, value)

    with pytest.raises(SkillPathError, match="unsupported link scheme"):
        validate_skill_folder(folder, source_root=tmp_path)


def test_escaped_backtick_does_not_hide_a_live_markdown_link(tmp_path):
    value = SkillDocument(
        "escaped-code",
        "Escaped code delimiter",
        r"\` [outside](../example.md) `",
    )
    folder = write_skill(tmp_path, value)

    with pytest.raises(SkillPathError, match="traversal"):
        validate_skill_folder(folder, source_root=tmp_path)


def test_directory_only_tree_is_bounded_by_total_entries(tmp_path):
    folder = write_skill(tmp_path)
    for index in range(MAX_FOLDER_ENTRIES):
        (folder / f"empty-{index:03}").mkdir()

    with pytest.raises(SkillFormatError, match="filesystem entries"):
        validate_skill_folder(folder, source_root=tmp_path)


def test_deep_directory_only_tree_is_bounded(tmp_path):
    folder = write_skill(tmp_path)
    current = folder
    for _ in range(MAX_FOLDER_DEPTH + 1):
        current = current / "d"
        current.mkdir()

    with pytest.raises(SkillFormatError, match="depth"):
        validate_skill_folder(folder, source_root=tmp_path)


def test_descriptor_scan_stops_lazy_enumeration_at_total_entry_limit(
    tmp_path, monkeypatch
):
    folder = write_skill(tmp_path)
    directory_fd = os.open(
        folder,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )

    class LazyEntries:
        def __init__(self):
            self.consumed = 0
            self.closed = False

        def __iter__(self):
            return self

        def __next__(self):
            self.consumed += 1
            if self.consumed > MAX_FOLDER_ENTRIES + 1:
                raise AssertionError("entry cap did not stop lazy enumeration")
            return SimpleNamespace(name=f"resource-{self.consumed:04}")

        def close(self):
            self.closed = True

    entries = LazyEntries()
    monkeypatch.setattr(format_module.os, "scandir", lambda _fd: entries)
    monkeypatch.setattr(
        format_module.os,
        "stat",
        lambda *_args, **_kwargs: SimpleNamespace(
            st_mode=format_module.stat.S_IFREG | 0o600,
            st_dev=1,
            st_ino=1,
        ),
    )
    monkeypatch.setattr(
        format_module,
        "_read_regular_file_at",
        lambda *_args, **_kwargs: b"x",
    )
    monkeypatch.setattr(
        format_module,
        "MAX_FOLDER_FILES",
        MAX_FOLDER_ENTRIES + 1,
    )
    try:
        with pytest.raises(SkillFormatError, match="filesystem entries"):
            format_module.validate_skill_folder_descriptor(
                directory_fd,
                folder_name=folder.name,
            )
    finally:
        os.close(directory_fd)

    assert entries.consumed == MAX_FOLDER_ENTRIES + 1
    assert entries.closed


def test_descriptor_scan_never_materializes_all_directory_names(tmp_path, monkeypatch):
    value = document()
    folder = write_skill(tmp_path, value)
    directory_fd = os.open(
        folder,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )

    def unexpected_listdir(*_args, **_kwargs):
        raise AssertionError("descriptor validation must enumerate lazily")

    monkeypatch.setattr(format_module.os, "listdir", unexpected_listdir)
    try:
        assert (
            format_module.validate_skill_folder_descriptor(
                directory_fd,
                folder_name=folder.name,
            )
            == value
        )
    finally:
        os.close(directory_fd)


@pytest.mark.parametrize(
    "reserved_name",
    (".SKILL.md.claim", ".SKILL.md.tmp.authored"),
)
def test_validation_rejects_primary_writer_reserved_resources(tmp_path, reserved_name):
    folder = write_skill(tmp_path)
    (folder / reserved_name).write_text("authored resource", encoding="utf-8")

    with pytest.raises(SkillFormatError, match="reserved"):
        validate_skill_folder(folder, source_root=tmp_path)
