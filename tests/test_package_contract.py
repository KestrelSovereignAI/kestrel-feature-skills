from __future__ import annotations

import tomllib
from importlib import metadata
from pathlib import Path

from kestrel_feature_skills import ProceduralSkillsFeature
from kestrel_feature_skills.format import parse_skill_markdown

ROOT = Path(__file__).resolve().parents[1]


def test_scaffold_shape_and_entry_point_contract():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    entry = pyproject["project"]["entry-points"]["kestrel_sovereign.features"]
    assert entry == {
        "ProceduralSkillsFeature": "kestrel_feature_skills:ProceduralSkillsFeature"
    }
    assert (ROOT / "src" / "kestrel_feature_skills" / "feature.py").is_file()
    assert (ROOT / "tests").is_dir()
    assert (ROOT / "SKILL.md").is_file()
    assert (ROOT / "README.md").is_file()


def test_repository_skill_manifest_is_a_valid_packaged_skill():
    document = parse_skill_markdown(
        (ROOT / "SKILL.md").read_bytes(),
        source="repository SKILL.md",
    )

    assert document.name == "kestrel-feature-skills"
    assert document.description


def test_installed_entry_point_when_distribution_metadata_is_available():
    entries = {
        entry.name: entry
        for entry in metadata.entry_points(group="kestrel_sovereign.features")
    }
    if "ProceduralSkillsFeature" not in entries:
        return
    assert entries["ProceduralSkillsFeature"].load() is ProceduralSkillsFeature


def test_static_assets_are_declared_as_wheel_artifacts():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    artifacts = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["artifacts"]
    assert "src/kestrel_feature_skills/static/**/*" in artifacts
    assert (ROOT / "src" / "kestrel_feature_skills" / "static" / "skills.js").is_file()
    assert (ROOT / "src" / "kestrel_feature_skills" / "static" / "skills.css").is_file()


def test_core_package_is_not_modified_or_vendored():
    assert not (ROOT / "kestrel_sovereign").exists()
    assert not (ROOT / "kestrel_sdk").exists()
