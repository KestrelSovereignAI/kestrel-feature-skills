# ProceduralSkillsFeature

> Folder-shaped procedural knowledge with progressive disclosure, provenance,
> per-agent enablement, and explicit code-risk rails.

## Tools

### skill_list

- **Description:** List names, descriptions, enablement, priority, provenance,
  and token estimates. Never returns procedure bodies.
- **Permission:** `ALLOW`

### skill_read

- **Description:** Explicitly disclose one skill's procedure body and resource
  inventory.
- **Permission:** `ALLOW`

### skill_search

- **Description:** Search names and descriptions. Never searches or returns
  procedure bodies.
- **Permission:** `ALLOW`

### skill_create

- **Description:** Create an agent-local skill. New skills are disabled unless
  the caller explicitly asks to enable them.
- **Permission:** `ASK`

### skill_edit

- **Description:** Edit `SKILL.md`, a Markdown resource, or `scripts/*.py`.
  Saving Python never executes it.
- **Permission:** `ASK`

### skill_enable / skill_disable

- **Description:** Change whether the skill's one-line description is eligible
  for the prompt catalog. Priority controls deterministic catalog order.
- **Permission:** `ASK`

### skill_delete

- **Description:** Permanently delete an agent-local skill folder and clean up
  its configuration and secondary graph index.
- **Permission:** `ALWAYS_ASK`

### skill_install

- **Description:** Install one validated skill from a credential-free HTTPS git
  source, record the exact commit, and leave it disabled.
- **Permission:** `ALWAYS_ASK`

## Configuration

| Environment variable | Meaning |
|---|---|
| `KESTREL_SHARED_SKILLS_DIR` | Optional host-shared, read-only skill root |
| `KESTREL_HOME` | Supplies the default host-shared `<home>/skills` root |

## Security invariants

- Absolute paths, `..`, backslashes, and symlink escapes are rejected.
- Local Markdown links must resolve inside their skill folder.
- Frontmatter accepts a strict scalar subset and exactly `name` + `description`.
- Remote installs require HTTPS with no embedded credentials.
- There is no script runner, subprocess route, or implicit execution path.
