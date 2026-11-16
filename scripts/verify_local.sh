#!/bin/bash
# Full local source baseline. No hosted runner, status upload, signing or publication.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ "$(uname -s)" == Darwin ]] || { echo 'Requires the local macOS development host.' >&2; exit 2; }
[[ -x .venv/bin/python ]] || { echo 'Run scripts/setup.sh first.' >&2; exit 2; }
[[ "$(git rev-parse --is-shallow-repository)" == false ]] || { echo 'Full Git history is required.' >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo 'Commit or preserve edits first; this baseline requires a clean checkout.' >&2; exit 2; }
# The inherited lock spans the complete Swift/Python baseline. Keep the exact
# eligibility/clean-source checks below; this only serializes competing worktrees.
if [[ "${1:-}" != "--kwip-compute-locked" ]]; then
    exec .venv/bin/python scripts/verification_lock.py --purpose full-baseline -- bash scripts/verify_local.sh --kwip-compute-locked
fi
shift
verified_sha=$(git rev-parse HEAD)
receipt_dir=".local/verification/$(date -u +%Y%m%dT%H%M%SZ)-${verified_sha:0:12}"
mkdir -p "$receipt_dir"
exec > >(tee "$receipt_dir/output.log") 2>&1
trap 'result=$?; if [[ $result -ne 0 ]]; then echo "Verification failed (exit $result); no success receipt. Log: $receipt_dir/output.log"; fi' EXIT
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"
export KWIPML_TEST_DEVELOPMENT_ROOT="$PWD"
export KWIPML_SKLEARN_TEST_PYTHON="$PWD/.venv/bin/python"
printf 'Local verification: %s\nCommit: %s\n' "$receipt_dir" "$verified_sha"
sw_vers
xcodebuild -version
swift --version
.venv/bin/python --version
.venv/bin/python scripts/component_bundles.py --check
.venv/bin/python scripts/generate_data02_learning.py --check
.venv/bin/python scripts/documentation_index.py --check
.venv/bin/python scripts/coverage_report.py --output "$receipt_dir/coverage.json"
.venv/bin/python scripts/runtime_package.py --prepare-development-watchdog .local/native-watchdog
/usr/bin/time -p swift test
/usr/bin/time -p .venv/bin/python scripts/verify_python.py --output "$receipt_dir/python" --junitxml "$receipt_dir/python-tests.xml"
[[ "$(git rev-parse HEAD)" == "$verified_sha" && -z "$(git status --porcelain)" ]] || { echo 'Checkout changed during verification; refusing success receipt.' >&2; exit 1; }
.venv/bin/python - "$receipt_dir" "$verified_sha" <<'PY'
import datetime, json, pathlib, sys
folder = pathlib.Path(sys.argv[1])
(folder / 'PASSED.json').write_text(json.dumps({
    'commit': sys.argv[2], 'completedAt': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'scope': 'local Swift and pinned Python authoring/worker/runtime baseline; see output.log for skips',
    'hostedResult': False, 'nativeUAT': False, 'releaseAcceptance': False,
}, indent=2) + '\n')
PY
printf 'Local baseline passed. Receipt: %s/PASSED.json\n' "$receipt_dir"
