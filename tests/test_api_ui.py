from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from kestrel_sovereign.privacy import PrivacyConfig

from kestrel_feature_skills.errors import (
    GitSourceError,
    SkillConflictError,
    SkillPathError,
)
from kestrel_feature_skills.format import MAX_RESOURCE_PATH_BYTES


@pytest.fixture
def app(feature):
    value = FastAPI()
    value.state.demo_mode = False
    value.state.agent = feature.agent
    value.include_router(feature.get_router())
    return value


@pytest.fixture
async def client(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as value:
        yield value


@pytest.mark.asyncio
async def test_catalog_create_tree_open_edit_round_trip(client):
    response = await client.post(
        "/api/procedural-skills",
        json={
            "name": "console",
            "description": "Console round trip",
            "body": "# Procedure\n\nOriginal.",
        },
    )
    assert response.status_code == 200, response.text
    catalog = (await client.get("/api/procedural-skills")).json()
    assert catalog["skills"][0]["name"] == "console"
    assert catalog["skills"][0]["token_cost"] > 0
    tree = (await client.get("/api/procedural-skills/console/tree")).json()
    assert tree["entries"][0]["path"] == "SKILL.md"
    opened = await client.get(
        "/api/procedural-skills/console/file",
        params={"path": "SKILL.md"},
    )
    assert "Original." in opened.json()["content"]
    content = opened.json()["content"].replace("Original.", "Edited.")
    saved = await client.put(
        "/api/procedural-skills/console/file",
        json={"path": "SKILL.md", "content": content},
    )
    assert saved.status_code == 200, saved.text
    reopened = await client.get(
        "/api/procedural-skills/console/file",
        params={"path": "SKILL.md"},
    )
    assert "Edited." in reopened.json()["content"]


@pytest.mark.asyncio
async def test_invalid_frontmatter_rejected_at_save_with_visible_reason(client):
    await client.post(
        "/api/procedural-skills",
        json={"name": "invalid-save", "description": "Original", "body": "body"},
    )
    response = await client.put(
        "/api/procedural-skills/invalid-save/file",
        json={"path": "SKILL.md", "content": "---\nname: invalid-save\n---\nbody"},
    )
    assert response.status_code == 422
    assert "description" in response.json()["detail"]


@pytest.mark.asyncio
async def test_edit_api_maps_concurrent_writer_conflict_to_409(
    client, feature, monkeypatch
):
    async def conflict(**_kwargs):
        raise SkillConflictError("concurrent skill write is already in progress")

    monkeypatch.setattr(feature, "edit_skill", conflict)

    response = await client.put(
        "/api/procedural-skills/conflicted/file",
        json={"path": "SKILL.md", "content": "replacement"},
    )

    assert response.status_code == 409
    assert "concurrent skill write" in response.json()["detail"]


@pytest.mark.asyncio
@pytest.mark.parametrize("priority", (True, 1.5, "1"))
async def test_create_api_rejects_non_integer_priority_before_writing(
    priority, client, feature
):
    response = await client.post(
        "/api/procedural-skills",
        json={
            "name": "invalid-priority",
            "description": "Invalid priority",
            "body": "body",
            "priority": priority,
        },
    )

    assert response.status_code == 422
    assert not (feature.agent.procedural_skills_root / "invalid-priority").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source_url, ref",
    (
        ("http://example.com/skills.git", "HEAD"),
        ("https://example.com/skills.git", "bad..ref"),
        ("https://example.com/skills.git", "main/"),
    ),
)
async def test_install_api_maps_invalid_git_input_to_422(client, source_url, ref):
    response = await client.post(
        "/api/procedural-skills/install",
        json={"source_url": source_url, "skill_name": "example", "ref": ref},
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_install_api_keeps_upstream_git_failures_as_502(
    client, feature, monkeypatch
):
    async def fail_install(**_kwargs):
        raise GitSourceError("remote unavailable")

    monkeypatch.setattr(feature, "install_skill", fail_install)

    response = await client.post(
        "/api/procedural-skills/install",
        json={
            "source_url": "https://example.com/skills.git",
            "skill_name": "example",
            "ref": "HEAD",
        },
    )

    assert response.status_code == 502


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    ["../secret.md", "/etc/passwd", "scripts\\escape.py", "scripts/../../escape.py"],
)
async def test_api_refuses_path_escape(path, client):
    await client.post(
        "/api/procedural-skills",
        json={"name": "scoped", "description": "Scoped", "body": "body"},
    )
    response = await client.get(
        "/api/procedural-skills/scoped/file",
        params={"path": path},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_api_read_and_write_share_the_validated_resource_path_cap(client):
    await client.post(
        "/api/procedural-skills",
        json={"name": "bounded-path", "description": "Bounded", "body": "body"},
    )
    overlong = "a" * (MAX_RESOURCE_PATH_BYTES + 1)

    written = await client.put(
        "/api/procedural-skills/bounded-path/file",
        json={"path": overlong, "content": "notes"},
    )
    opened = await client.get(
        "/api/procedural-skills/bounded-path/file",
        params={"path": overlong},
    )

    assert written.status_code == 422
    assert f"exceeds {MAX_RESOURCE_PATH_BYTES}" in written.json()["detail"]
    assert opened.status_code == 422


@pytest.mark.asyncio
async def test_python_route_saves_text_and_reports_risk_without_execution(
    client, tmp_path
):
    await client.post(
        "/api/procedural-skills",
        json={"name": "python-risk", "description": "Risk", "body": "body"},
    )
    marker = tmp_path / "should-not-exist"
    source = f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n"
    saved = await client.put(
        "/api/procedural-skills/python-risk/file",
        json={"path": "scripts/danger.py", "content": source},
    )
    assert saved.status_code == 200, saved.text
    assert not marker.exists()
    opened = await client.get(
        "/api/procedural-skills/python-risk/file",
        params={"path": "scripts/danger.py"},
    )
    assert opened.json()["execution_risk"] is True
    assert opened.json()["content"] == source


@pytest.mark.asyncio
async def test_file_read_rejects_symlink_added_after_catalog_discovery(client, feature):
    await client.post(
        "/api/procedural-skills",
        json={"name": "late-link", "description": "Late link", "body": "body"},
    )
    saved = await client.put(
        "/api/procedural-skills/late-link/file",
        json={"path": "scripts/tool.py", "content": "SECRET = True\n"},
    )
    assert saved.status_code == 200, saved.text
    folder = feature.agent.procedural_skills_root / "late-link"
    (folder / "notes.md").symlink_to(folder / "scripts" / "tool.py")

    opened = await client.get(
        "/api/procedural-skills/late-link/file",
        params={"path": "notes.md"},
    )

    assert opened.status_code == 422
    assert "symlink" in opened.json()["detail"]


@pytest.mark.asyncio
async def test_reload_discovers_folder_without_restart(client, feature):
    folder = feature.agent.procedural_skills_root / "appeared"
    folder.mkdir()
    (folder / "SKILL.md").write_text(
        '---\nname: "appeared"\ndescription: "Appeared on disk"\n---\n\nbody\n',
        encoding="utf-8",
    )
    before = (await client.get("/api/procedural-skills")).json()
    assert not any(skill["name"] == "appeared" for skill in before["skills"])
    after = (await client.post("/api/procedural-skills/reload")).json()
    assert any(skill["name"] == "appeared" for skill in after["skills"])


@pytest.mark.asyncio
async def test_catalog_route_rehydrates_after_persistent_privacy_returns(
    client, feature
):
    await client.post(
        "/api/procedural-skills",
        json={"name": "privacy-ui", "description": "Privacy UI", "body": "body"},
    )
    feature.agent.privacy_config = PrivacyConfig(storage="none")
    await feature.refresh()
    assert (await client.get("/api/procedural-skills")).json()["count"] == 0

    feature.agent.privacy_config = PrivacyConfig(storage="full")

    catalog = await client.get("/api/procedural-skills")
    opened = await client.get(
        "/api/procedural-skills/privacy-ui/file",
        params={"path": "SKILL.md"},
    )
    assert catalog.status_code == 200
    assert catalog.json()["skills"][0]["name"] == "privacy-ui"
    assert opened.status_code == 200
    assert "Privacy UI" in opened.json()["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "path"),
    (("get", "/api/procedural-skills"), ("post", "/api/procedural-skills/reload")),
)
async def test_catalog_rehydration_maps_invalid_persistent_root_to_422(
    method, path, client, feature, monkeypatch
):
    feature.agent.privacy_config = PrivacyConfig(storage="none")
    await feature.refresh()
    feature.agent.privacy_config = PrivacyConfig(storage="full")

    def reject_invalid_root():
        raise SkillPathError("skills root became a symlink")

    monkeypatch.setattr(feature, "_ensure_persistent_services", reject_invalid_root)

    response = await getattr(client, method)(path)

    assert response.status_code == 422
    assert response.json()["detail"] == "skills root became a symlink"


@pytest.mark.asyncio
async def test_tree_and_file_routes_hold_privacy_lock_through_filesystem_reads(
    client, feature, monkeypatch
):
    await client.post(
        "/api/procedural-skills",
        json={"name": "privacy-http", "description": "Privacy HTTP", "body": "body"},
    )
    feature.agent._privacy_transition_lock = asyncio.Lock()
    feature.agent.privacy_config = PrivacyConfig(storage="full")
    observed = []
    original_tree = feature._store.tree
    original_read_file = feature._store.read_file

    def observed_tree(record):
        observed.append(("tree", feature.agent._privacy_transition_lock.locked()))
        return original_tree(record)

    def observed_read_file(record, relative_path):
        observed.append(("file", feature.agent._privacy_transition_lock.locked()))
        return original_read_file(record, relative_path)

    monkeypatch.setattr(feature._store, "tree", observed_tree)
    monkeypatch.setattr(feature._store, "read_file", observed_read_file)

    tree = await client.get("/api/procedural-skills/privacy-http/tree")
    opened = await client.get(
        "/api/procedural-skills/privacy-http/file", params={"path": "SKILL.md"}
    )

    assert tree.status_code == 200
    assert opened.status_code == 200
    assert observed == [("tree", True), ("file", True)]


@pytest.mark.asyncio
async def test_state_endpoint_changes_context_breakdown(client):
    await client.post(
        "/api/procedural-skills",
        json={
            "name": "context",
            "description": "Context sentinel",
            "body": "secret body",
        },
    )
    enabled = await client.patch(
        "/api/procedural-skills/context/state",
        json={"enabled": True, "priority": 3},
    )
    assert enabled.status_code == 200
    catalog = (await client.get("/api/procedural-skills")).json()
    assert catalog["context"]["included"] == ["context"]
    assert "Context sentinel" in catalog["context"]["text"]
    assert "secret body" not in catalog["context"]["text"]
    await client.patch(
        "/api/procedural-skills/context/state",
        json={"enabled": False},
    )
    assert (await client.get("/api/procedural-skills")).json()["context"]["text"] == ""


@pytest.mark.asyncio
async def test_state_endpoint_maps_invalid_skill_name_to_422(client):
    response = await client.patch(
        "/api/procedural-skills/INVALID!/state",
        json={"enabled": True},
    )

    assert response.status_code == 422
    assert "skill name" in response.json()["detail"]


@pytest.mark.asyncio
async def test_delete_api_removes_only_resolved_local_skill(client, feature):
    await client.post(
        "/api/procedural-skills",
        json={"name": "delete-me", "description": "Delete", "body": "body"},
    )
    refused = await client.delete("/api/procedural-skills/delete-me")
    assert refused.status_code == 403
    assert (feature.agent.procedural_skills_root / "delete-me").exists()
    response = await client.delete(
        "/api/procedural-skills/delete-me",
        headers={"X-Kestrel-Allow-Destructive": "operator-confirmed-ui"},
    )
    assert response.status_code == 200
    assert not (feature.agent.procedural_skills_root / "delete-me").exists()
    second = await client.delete(
        "/api/procedural-skills/delete-me",
        headers={"X-Kestrel-Allow-Destructive": "operator-confirmed-ui"},
    )
    assert second.status_code == 404


@pytest.mark.asyncio
async def test_delete_api_maps_invalid_skill_name_to_422(client):
    response = await client.delete(
        "/api/procedural-skills/INVALID!",
        headers={"X-Kestrel-Allow-Destructive": "operator-confirmed-ui"},
    )

    assert response.status_code == 422
    assert "skill name" in response.json()["detail"]


def test_ui_bundle_contains_required_rails_and_no_run_control(feature):
    ui = feature.get_ui_contributions()
    static = Path(ui.static_dir)
    source = (static / "skills.js").read_text(encoding="utf-8")
    assert ui.modules == ["skills.js"]
    assert ui.css == ["skills.css"]
    assert ui.capability is None
    assert "registerPanel" in source
    assert "name.pattern = '[a-z0-9](?:[a-z0-9_\\\\-]{0,62}[a-z0-9])?'" in source
    assert "skills-delete-approval" in source
    assert "X-Kestrel-Allow-Destructive" in source
    assert "python-execution-risk" in source
    assert "Discover / reload" in source
    assert "Save rejected:" in source
    assert "showModal" in source
    assert "agent:switch" in source
    assert "currentAgent() === payload.next" in source
    assert "capabilities:changed" in source
    assert "reconcileAvailability" in source
    assert "MutationObserver" in source
    assert "state.catalog = catalog" in source
    assert "editorOwner" in source
    assert "skill.deletable" in source
    assert "remaining_source_kind" in source
    assert "skill_run_script" not in source
    assert "/run" not in source
    assert ".execute(" not in source
    assert "run script" not in source.lower()
