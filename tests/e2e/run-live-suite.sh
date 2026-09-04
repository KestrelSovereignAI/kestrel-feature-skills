#!/usr/bin/env bash
set -euo pipefail

e2e_profile=${1:-}
case "$e2e_profile" in
  skills)
    e2e_features='["ProceduralSkillsFeature"]'
    e2e_expected=1
    e2e_spec=tests/e2e/skills.spec.cjs
    ;;
  core-only)
    e2e_features='[]'
    e2e_expected=0
    e2e_spec=tests/e2e/core_only.spec.cjs
    ;;
  *)
    echo "usage: $0 {skills|core-only}" >&2
    exit 2
    ;;
esac

e2e_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
e2e_python="$e2e_root/.venv/bin/python"
e2e_kestrel="$e2e_root/.venv/bin/kestrel"
e2e_port=${KESTREL_E2E_PORT:-8777}
e2e_url="http://127.0.0.1:$e2e_port"
e2e_key="skills-e2e-only-key"
e2e_tmp_root=${TMPDIR:-/tmp}
e2e_tmp_root=${e2e_tmp_root%/}
e2e_home=$(mktemp -d "$e2e_tmp_root/kestrel-skills-e2e.XXXXXX")
e2e_log_dir="$e2e_root/.e2e-artifacts"
e2e_log="$e2e_log_dir/$e2e_profile-server.log"
e2e_server_pid=

cleanup() {
  if [[ -n "$e2e_server_pid" ]] && kill -0 "$e2e_server_pid" 2>/dev/null; then
    kill "$e2e_server_pid"
    wait "$e2e_server_pid" 2>/dev/null || true
  fi
  case "$e2e_home" in
    "$e2e_tmp_root"/kestrel-skills-e2e.*) rm -rf -- "$e2e_home" ;;
  esac
}
trap cleanup EXIT INT TERM

if curl --silent --fail "$e2e_url/api/auth/key" >/dev/null 2>&1; then
  echo "refusing to reuse a server already listening at $e2e_url" >&2
  exit 1
fi

mkdir -p "$e2e_log_dir"
env \
  -u VIRTUAL_ENV \
  OPENAI_API_KEY= \
  ANTHROPIC_API_KEY= \
  GEMINI_API_KEY= \
  GOOGLE_API_KEY= \
  AWS_EC2_METADATA_DISABLED=true \
  KESTREL_HOME="$e2e_home" \
  KESTREL_API_KEY="$e2e_key" \
  KESTREL_DID_WEB_DOMAIN=localhost \
  "$e2e_kestrel" create kite --port 8801 --test

"$e2e_python" - "$e2e_home/multi_agent.toml" "$e2e_features" <<'PY'
from pathlib import Path
import sys

config_path = Path(sys.argv[1])
feature_list = sys.argv[2]
content = config_path.read_text(encoding="utf-8")
needle = "autostart = true\n"
if content.count(needle) != 1:
    raise SystemExit("generated multi_agent.toml has an unexpected agent shape")
config_path.write_text(
    content.replace(needle, f"{needle}features = {feature_list}\n", 1),
    encoding="utf-8",
)
PY

(
  cd "$e2e_home"
  exec env \
    -u VIRTUAL_ENV \
    OPENAI_API_KEY= \
    ANTHROPIC_API_KEY= \
    GEMINI_API_KEY= \
    GOOGLE_API_KEY= \
    AWS_EC2_METADATA_DISABLED=true \
    KESTREL_HOME="$e2e_home" \
    KESTREL_MULTI_AGENT=true \
    KESTREL_MULTI_AGENT_CONFIG="$e2e_home/multi_agent.toml" \
    KESTREL_API_KEY="$e2e_key" \
    "$e2e_python" -m uvicorn kestrel_sovereign.server:app \
      --host 127.0.0.1 --port "$e2e_port"
) >"$e2e_log" 2>&1 &
e2e_server_pid=$!

e2e_ready=0
for _e2e_attempt in $(seq 1 90); do
  if curl --silent --fail \
    -H "X-API-Key: $e2e_key" \
    "$e2e_url/api/agents" \
    | "$e2e_python" -c \
      'import json, sys; data = sys.stdin.read(); payload = json.loads(data) if data else {}; raise SystemExit(0 if any(a.get("name") == "kite" for a in payload.get("agents", ())) else 1)'
  then
    e2e_ready=1
    break
  fi
  if ! kill -0 "$e2e_server_pid" 2>/dev/null; then
    break
  fi
  sleep 1
done

if [[ "$e2e_ready" != 1 ]]; then
  echo "isolated $e2e_profile host did not become ready" >&2
  tail -200 "$e2e_log" >&2
  exit 1
fi

cd "$e2e_root"
env \
  KESTREL_URL="$e2e_url" \
  KESTREL_API_KEY="$e2e_key" \
  KESTREL_AGENT=kite \
  KESTREL_EXPECT_SKILLS="$e2e_expected" \
  KESTREL_KITE_SKILLS_ROOT="$e2e_home/agent_data/kite/skills" \
  npx playwright test "$e2e_spec"
