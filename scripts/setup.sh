#!/bin/zsh
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ ! -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  uv venv --python 3.13 "$PROJECT_ROOT/.venv"
fi
uv pip install --python "$PROJECT_ROOT/.venv/bin/python" -r "$PROJECT_ROOT/requirements-runner.txt"
uv pip install --python "$PROJECT_ROOT/.venv/bin/python" -r "$PROJECT_ROOT/requirements-test-tools.txt"
"$PROJECT_ROOT/scripts/build-app.sh"
