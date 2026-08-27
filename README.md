# kestrel-feature-skills

Folder-shaped, progressively disclosed procedural skills for Kestrel Sovereign.

A skill is an authored directory:

```text
skills/<name>/
  SKILL.md
  references.md       # optional
  scripts/example.py  # optional; never executed by this package
```

`SKILL.md` has exactly two frontmatter fields. Its folder name and frontmatter
name must match:

```markdown
---
name: "review-a-diff"
description: "Use when a complete branch diff needs an evidence-backed review."
---

# Procedure

1. Inspect the complete diff against its base.
2. Reproduce each finding before changing code.
```

Only the enabled skill's name and one-line description are eligible for the
system-prompt catalog. The procedure body appears only after `skill_read`.
Resources and scripts are opened explicitly, and there is no script execution
surface in this package. `skill_read <name> <relative-path>` opens a file from
the resource inventory as text; omitting the path returns the primary procedure
body and inventory.

## Sources and precedence

Resolution is deterministic:

1. agent-local — `<agent-data>/skills`
2. host-shared — `$KESTREL_SHARED_SKILLS_DIR`, or `$KESTREL_HOME/skills`
3. git-backed origin — installed explicitly into the agent-local root

An agent-local skill shadows a same-named host skill. Every resolved record
reports provenance, and a git install records the full source commit in
`.kestrel-provenance.json`. Remote installs refuse redirects and use a partial
sparse checkout of only the requested skill paths. They abort if checkout data
crosses 32 MiB; the validated skill folder itself remains capped at 2 MiB.

## Permission rails

The package declares permission defaults through the SDK contribution contract:

| Tools | Permission |
|---|---|
| `skill_list`, `skill_read`, `skill_search` | `ALLOW` |
| `skill_create`, `skill_edit`, `skill_enable`, `skill_disable` | `ASK` |
| `skill_delete`, `skill_install` | `ALWAYS_ASK` |

The feature default is `ASK`. The Console also requires a distinct destructive
confirmation before its authenticated operator delete route is called.

## Storage authority

The folder is authoritative. `procedural_skill` graph nodes are a recoverable,
best-effort index. Per-agent enablement and priority use `skill:<name>` rows in
the existing `bootstrap_config` table under an isolated
`procedural-skill-state:<agent DID>` logical-agent namespace. The isolation
prevents older core bootstrap loaders from treating a procedure as a full-text
bootstrap file; this package does not create a competing configuration table.

## Context-seam status

The package includes and tests the deterministic, descriptions-only context
renderer. Registering it with Sovereign is intentionally gated on epic #3018's
SDK/core contribution-seam tickets (#3021–#3026). This repository does not ship
a lookalike hook or per-turn fallback while that contract is unavailable.

## Installation

```bash
uv pip install -e .
```

The `ProceduralSkillsFeature` entry point is discoverable through
`kestrel_sovereign.features`. The distinct class name lets the package be
dogfooded alongside a core release that still contains the obsolete bundled
`SkillsFeature`; the allowlist should select only `ProceduralSkillsFeature`.

## Development

```bash
uv run --extra test pytest -q
uv build
```

Live-path verification follows Kestrel's Kite runbook and drives the feature
through `/api/agents/kite/api/agent/invoke`, in addition to its scoped HTTP and
Console surfaces.
