# kestrel-feature-skills — Agent Instructions

This repository packages folder-shaped procedural skills for Kestrel Sovereign.

## Architectural rails

- A skill folder is authoritative. Graph nodes are a best-effort secondary index.
- `SKILL.md` contains only `name` and `description` frontmatter plus the procedure body.
- Only enabled skill names and descriptions may enter the system-prompt clause.
- Resources and scripts are opened explicitly; scripts are never executed by this package.
- Agent-local skills shadow host-shared skills, which shadow remote sources.
- Skill enablement and priority use namespaced rows in core's `bootstrap_config` table.
- Destructive or remote-code tools remain protected by declared SDK permission defaults.

## Running tests

```bash
uv run --extra test pytest -q
```

Run the package tests before integration or live-agent dogfooding. Mutation-check the
atomic-write and context byte-stability guards before claiming those rails verified.
