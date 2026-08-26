from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI


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


def test_ui_bundle_contains_required_rails_and_no_run_control(feature):
    ui = feature.get_ui_contributions()
    static = Path(ui.static_dir)
    source = (static / "skills.js").read_text(encoding="utf-8")
    assert ui.modules == ["skills.js"]
    assert ui.css == ["skills.css"]
    assert ui.capability == "procedural-skills"
    assert "registerPanel" in source
    assert "skills-delete-approval" in source
    assert "X-Kestrel-Allow-Destructive" in source
    assert "python-execution-risk" in source
    assert "Discover / reload" in source
    assert "Save rejected:" in source
    assert "showModal" in source
    assert "agent:switch" in source
    assert "editorOwner" in source
    assert "skill_run_script" not in source
    assert "/run" not in source
    assert ".execute(" not in source
    assert "run script" not in source.lower()
