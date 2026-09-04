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


def test_runtime_dependencies_require_the_async_context_refresh_contract():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = set(pyproject["project"]["dependencies"])

    assert "kestrel-sovereign>=0.53.11,<0.54" in dependencies
    assert "kestrel-sovereign-sdk>=0.38.1,<0.39" in dependencies


def test_publish_workflow_gates_the_exact_tag_before_trusted_upload():
    publish = (ROOT / ".github" / "workflows" / "publish.yml").read_text(
        encoding="utf-8"
    )
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "uses: ./.github/workflows/ci.yml" in publish
    assert "ref: ${{ inputs.ref || github.ref }}" in publish
    assert 'tag_version="${tag_name#v}"' in publish
    assert "environment: pypi" in publish
    assert "id-token: write" in publish
    assert "workflow_call:" in ci
    assert ci.count("ref: ${{ inputs.ref || github.ref }}") == 2


def test_sdist_declares_the_runnable_browser_harness():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    included = set(pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]["include"])

    assert {
        "/package.json",
        "/package-lock.json",
        "/playwright.config.cjs",
    } <= included


def test_core_package_is_not_modified_or_vendored():
    assert not (ROOT / "kestrel_sovereign").exists()
    assert not (ROOT / "kestrel_sdk").exists()
