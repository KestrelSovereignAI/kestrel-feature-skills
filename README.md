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
body and inventory. Every inventoried file must be UTF-8 text and is capped at
256 KiB so every successfully discovered resource remains readable through the
advertised tool and Console surfaces.
Resource paths use portable POSIX spelling (no backslashes or drive prefixes)
and are capped at 1024 UTF-8 bytes so every discovered path fits the HTTP read
contract.

## Sources and precedence

Resolution is deterministic:

1. agent-local — `<agent-data>/skills`
2. host-shared — `$KESTREL_SHARED_SKILLS_DIR`, or `$KESTREL_HOME/skills`
3. git-backed origin — installed explicitly into the agent-local root

An agent-local skill shadows a same-named host skill. Every resolved record
reports provenance, and a git install records the full source commit in
`.kestrel-provenance.json`. Remote installs refuse redirects and use a partial
sparse checkout of only the requested skill paths. They abort if checkout data
crosses 32 MiB or 4,096 filesystem entries; the validated skill folder itself
remains capped at 2 MiB and 512 entries, in addition to the 256 KiB per-file
read limit. Each configured source root is also capped at 4,096 immediate
entries before sorting or per-skill validation, so reload work remains bounded
even when the directory contains non-skill or hidden files.

Mutation locks, durable fail-closed state, and crash-safe edit temporaries live
below the single hidden `<agent-data>/skills/.kestrel-internal/` directory.
They are never charged individually against source discovery. Each agent-local
folder also has an implementation-owned `.kestrel-generation` marker whose
random value prevents a stale approval from matching a deleted and recreated
folder even if its filesystem inode is reused. That marker is hidden from the
resource inventory and does not consume the documented user file or byte
budget; an atomic initializer may briefly use an equally hidden
`.kestrel-generation.tmp.*` hardlink source inside `.kestrel-internal`.
Marker writers and the startup reaper share a private lock, and startup retires
any crash-orphaned marker staging left by older or interrupted processes.
Deletion atomically retires the complete skill generation into this private
directory before returning; it never recursively unlinks a public or
recovery-visible name. It then best-effort purges the retired generation only
inside `.kestrel-internal`; a later store startup retries interrupted cleanup.
That directory is implementation-owned recovery state, not a supported
direct-edit surface; direct filesystem edits remain supported only in named
skill folders.

## Permission rails

The package declares permission defaults through the SDK contribution contract:

| Tools | Permission |
|---|---|
| `skill_list`, `skill_read`, `skill_search` | `ALLOW` |
| `skill_create`, `skill_edit`, `skill_enable`, `skill_disable` | `ASK` |
| `skill_delete`, `skill_install` | `ALWAYS_ASK` |

The feature default is `ASK`. The Console also requires a distinct destructive
confirmation before its authenticated operator delete route is called. Agent
tool deletion requires the target's `delete_revision` from `skill_list` or
`skill_search`, binding an `ALWAYS_ASK` approval to the exact observed skill
generation.

## Storage authority

The folder is authoritative. `procedural_skill` graph nodes are a recoverable,
best-effort index. Per-agent enablement and priority use `skill:<name>` rows in
the existing `bootstrap_config` table under an isolated
`procedural-skill-state:<agent DID>` logical-agent namespace. The isolation
prevents older core bootstrap loaders from treating a procedure as a full-text
bootstrap file; this package does not create a competing configuration table.

## Context contribution

The package registers its deterministic, descriptions-only renderer through
the SDK context-clause contract introduced by epic #3018. Sovereign resolves
those bytes only at lifecycle or configuration transitions, retains an
immutable cache between turns, and owns prompt budgeting and audit accounting.
There is no lookalike hook or per-turn feature callback.

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
npm ci
npx playwright install chromium
tests/e2e/run-live-suite.sh skills
tests/e2e/run-live-suite.sh core-only
```

The browser runner incepts a temporary test-only Kite home, starts Core on the
configured E2E port, executes Chromium against the live host, and removes the
temporary agent afterward. The browser contract does not invoke an LLM.

Live-path verification follows Kestrel's Kite runbook and drives the feature
through `/api/agents/kite/api/agent/invoke`, in addition to its scoped HTTP and
Console surfaces. The live test requires `KESTREL_KITE_HOSTED_PROVIDER` and
`KESTREL_KITE_HOSTED_MODEL`; it accepts only hosted Claude Haiku or GPT-5.6
Luna routes and asserts that the invoke response reports the exact pinned
provider and model. It never accepts a local-model fallback as release evidence.
The isolated Kite agent must enable both `BootstrapFeature` (so the test can
complete onboarding through `!skip-discovery`) and `ProceduralSkillsFeature`.

Version tags matching `vX.Y.Z` run the complete reusable CI workflow against
the exact tag SHA before building and uploading through PyPI trusted publishing.
The publish workflow rejects a tag whose version differs from `pyproject.toml`.
