from __future__ import annotations

import os

import pytest

from kestrel_feature_skills.errors import SkillFormatError, SkillPathError
from kestrel_feature_skills.format import (
    MAX_DESCRIPTION_BYTES,
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


def test_folder_scan_error_rejects_candidate(tmp_path, monkeypatch):
    folder = write_skill(tmp_path)

    def failing_walk(root, *, followlinks, onerror=None):
        assert root == folder
        assert followlinks is False
        if onerror is not None:
            onerror(PermissionError("subdirectory became unreadable"))
        return iter(())

    monkeypatch.setattr("kestrel_feature_skills.format.os.walk", failing_walk)

    with pytest.raises(SkillFormatError, match="scan"):
        validate_skill_folder(folder, source_root=tmp_path)


def test_folder_rejects_resource_path_that_is_not_strict_utf8(tmp_path, monkeypatch):
    folder = write_skill(tmp_path)

    def walk_with_surrogate(root, *, followlinks, onerror=None):
        assert root == folder
        assert followlinks is False
        return iter(((str(folder), ["resource-\udcff"], ["SKILL.md"]),))

    monkeypatch.setattr("kestrel_feature_skills.format.os.walk", walk_with_surrogate)

    with pytest.raises(SkillFormatError, match="UTF-8"):
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


def test_reference_style_markdown_local_resource_is_accepted(tmp_path):
    value = SkillDocument(
        "reference-local",
        "Reference link",
        'Read [notes][local].\n\n[local]: <notes.md> "Operator notes"',
    )
    folder = write_skill(tmp_path, value)
    (folder / "notes.md").write_text("notes", encoding="utf-8")
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


def test_executable_link_scheme_rejected(tmp_path):
    value = SkillDocument("bad-link", "Bad link", "[click](javascript:alert(1))")
    folder = write_skill(tmp_path, value)
    with pytest.raises(SkillPathError, match="unsupported link scheme"):
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
