#!/bin/zsh
# Local development signing only. No App Store submission or Developer ID claim.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNTIME_SOURCE="${1:?Provide a prepared KwipMLRuntime payload directory}"
[[ -f "$RUNTIME_SOURCE/runtime-manifest.json" ]] || { print -u2 'Runtime manifest missing'; exit 1; }
# Verify against this checkout before building or touching a previous candidate.
# Use a build-host interpreter, never execute the supplied runtime to attest itself.
BUILD_PYTHON="${KWIPML_BUILD_PYTHON:-python3}"
"$BUILD_PYTHON" "$PROJECT_ROOT/scripts/runtime_package.py" --verify-current-worker "$RUNTIME_SOURCE"
"$PROJECT_ROOT/scripts/build-app.sh"
CANDIDATE="$PROJECT_ROOT/.local/KwipML Candidate.app"
STAGING="$PROJECT_ROOT/.local/packaged-stage-$(uuidgen)"
mkdir -p "$STAGING"
trap 'rm -rf "$STAGING"' EXIT
/usr/bin/ditto "$PROJECT_ROOT/.local/KwipML POC.app" "$STAGING/KwipML Candidate.app"
APP="$STAGING/KwipML Candidate.app"
rm -rf "$APP/Contents/Resources/KwipMLRuntime"
RUNTIME_TARGET="$APP/Contents/XPCServices/DatabaseSourceService.xpc/Contents/Resources/KwipMLRuntime"
mkdir -p "${RUNTIME_TARGET:h}"
/usr/bin/ditto "$RUNTIME_SOURCE" "$RUNTIME_TARGET"
# Attributed privacy resources are pinned to the copied runtime and staged before
# any native/app signing. Existing resources must verify exactly; never overwrite.
PRIVACY_DESTINATION="$APP/Contents/Resources/VendorPrivacy"
if [[ -e "$PRIVACY_DESTINATION" || -L "$PRIVACY_DESTINATION" ]]; then
  "$BUILD_PYTHON" "$PROJECT_ROOT/scripts/vendor_privacy.py" --runtime-root "$RUNTIME_TARGET" --destination "$PRIVACY_DESTINATION" --verify-existing
else
  "$BUILD_PYTHON" "$PROJECT_ROOT/scripts/vendor_privacy.py" --runtime-root "$RUNTIME_TARGET" --destination "$PRIVACY_DESTINATION"
fi
# Recheck the copied bytes before signing; integrity refresh must never turn a
# stale payload into a current one merely by regenerating hashes.
"$BUILD_PYTHON" "$PROJECT_ROOT/scripts/runtime_package.py" --verify-current-worker "$RUNTIME_TARGET"
"$BUILD_PYTHON" "$PROJECT_ROOT/scripts/app_store_api_audit.py" "$APP"
# Normalize dependency install identities on the staged copy, sign its native
# dependencies, and fail closed on external loads or invalid native signatures.
# The audit verifies sealed runner binaries without modifying their bytes.
"$BUILD_PYTHON" "$PROJECT_ROOT/scripts/runtime_audit.py" "$RUNTIME_TARGET" --sign-ad-hoc --require-signatures --report "$STAGING/runtime-native-audit.json"
"$BUILD_PYTHON" "$PROJECT_ROOT/scripts/runtime_package.py" --verify-current-worker "$RUNTIME_TARGET"
rm -f "$APP/Contents/Resources/ProjectRoot.txt"
/usr/bin/codesign --force --sign - --timestamp=none --entitlements "$PROJECT_ROOT/packaging/Worker.entitlements" "$RUNTIME_TARGET/python/bin/python3.13"
DATABASE_ENTITLEMENTS="$STAGING/DatabaseSourceService.entitlements"
"$BUILD_PYTHON" "$PROJECT_ROOT/scripts/database_broker_package.py" --app "$APP" --development-entitlements "$DATABASE_ENTITLEMENTS" --development-relocation "$CANDIDATE"
/usr/bin/codesign --force --sign - --timestamp=none --entitlements "$DATABASE_ENTITLEMENTS" "$APP/Contents/XPCServices/DatabaseSourceService.xpc"
/usr/bin/codesign --force --sign - --timestamp=none --entitlements "$PROJECT_ROOT/packaging/MCPService.entitlements" "$APP/Contents/XPCServices/MCPService.xpc"
/usr/bin/codesign --force --sign - --timestamp=none --entitlements "$PROJECT_ROOT/packaging/App.entitlements" "$APP"
/usr/bin/codesign --verify --deep --strict "$APP"
# Signing must not change any sealed worker bytes, including the native guard.
"$BUILD_PYTHON" "$PROJECT_ROOT/scripts/runtime_package.py" --verify-current-worker "$RUNTIME_TARGET"
# A previous local candidate is retained for rollback rather than silently removed.
if [[ -e "$CANDIDATE" ]]; then mv "$CANDIDATE" "$PROJECT_ROOT/.local/KwipML Candidate previous-$(uuidgen).app"; fi
mv "$APP" "$CANDIDATE"
print "$CANDIDATE"
