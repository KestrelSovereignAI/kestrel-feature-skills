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


def test_folder_name_must_match_frontmatter(tmp_path):
    folder = tmp_path / "other-name"
    folder.mkdir()
    (folder / "SKILL.md").write_text(
        serialize_skill_markdown(document()), encoding="utf-8"
    )
    with pytest.raises(SkillFormatError, match="must match"):
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


def test_local_resource_reference_must_exist_and_stay_inside(tmp_path):
    value = SkillDocument("linked", "Linked resource", "Read [notes](notes.md).")
    folder = write_skill(tmp_path, value)
    with pytest.raises(SkillPathError, match="does not exist"):
        validate_skill_folder(folder, source_root=tmp_path)
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
