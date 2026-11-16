#!/bin/zsh
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"
BUILD_CONFIGURATION="${KWIPML_BUILD_CONFIGURATION:-debug}"
# Opt-in private development signing preserves the existing NetworkService container identity.
# Other helpers retain their established ad-hoc identities and container permissions.
# Default builds remain ad hoc; this does not assemble or authorize a release.
DEVELOPMENT_SIGNING_IDENTITY="${KWIPML_DEVELOPMENT_SIGNING_IDENTITY:--}"
case "$BUILD_CONFIGURATION" in
  debug|release) ;;
  *) print -u2 'KWIPML_BUILD_CONFIGURATION must be debug or release'; exit 1 ;;
esac
PRODUCT_DISPLAY_NAME="$("${KWIPML_BUILD_PYTHON:-python3}" "$PROJECT_ROOT/scripts/product_brand.py")"
SDK_VERSION="$(xcrun --sdk macosx --show-sdk-version)"
SDK_BUILD="$(xcrun --sdk macosx --show-sdk-build-version)"
[[ "$SDK_VERSION" =~ '^[0-9]+(\.[0-9]+){0,2}$' ]] || { print -u2 'Invalid macOS SDK version'; exit 1; }
[[ "$SDK_BUILD" =~ '^[A-Za-z0-9.]+$' && ${#SDK_BUILD} -le 64 ]] || { print -u2 'Invalid macOS SDK build'; exit 1; }
"${KWIPML_BUILD_PYTHON:-python3}" "$PROJECT_ROOT/scripts/runtime_package.py" --prepare-development-watchdog "$PROJECT_ROOT/.local/native-watchdog"
"${KWIPML_BUILD_PYTHON:-python3}" "$PROJECT_ROOT/scripts/component_bundles.py" --check
"${KWIPML_BUILD_PYTHON:-python3}" "$PROJECT_ROOT/scripts/generate_data02_learning.py" --check
swift build --package-path "$PROJECT_ROOT" --configuration "$BUILD_CONFIGURATION"
BUILD_DIR="$(swift build --package-path "$PROJECT_ROOT" --configuration "$BUILD_CONFIGURATION" --show-bin-path)"
APP_PATH="$PROJECT_ROOT/.local/KwipML POC.app"
mkdir -p "$APP_PATH/Contents/MacOS" "$APP_PATH/Contents/Resources"
cp "$BUILD_DIR/KwipML" "$APP_PATH/Contents/MacOS/KwipML"
cp -R "$BUILD_DIR/KwipML_KwipMLCore.bundle" "$APP_PATH/Contents/Resources/"
cp -R "$BUILD_DIR/KwipML_KwipMLApp.bundle" "$APP_PATH/Contents/Resources/"
"${KWIPML_BUILD_PYTHON:-python3}" "$PROJECT_ROOT/scripts/app_icon.py" --source "$PROJECT_ROOT/Sources/KwipMLApp/Resources/KwipMLWelcomeIcon.png" --output "$APP_PATH/Contents/Resources/KwipML.icns"
printf '%s\n' "$PROJECT_ROOT" > "$APP_PATH/Contents/Resources/ProjectRoot.txt"
cat > "$APP_PATH/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleIdentifier</key><string>info.kwip.KwipML.POC</string>
<key>CFBundleName</key><string>Kwipily Development</string>
<key>CFBundleDisplayName</key><string>Kwipily Development</string>
<key>CFBundleIconFile</key><string>KwipML.icns</string>
<key>CFBundleExecutable</key><string>KwipML</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleShortVersionString</key><string>1.3.4</string>
<key>CFBundleVersion</key><string>1</string>
<key>LSMinimumSystemVersion</key><string>14.0</string>
<key>NSHighResolutionCapable</key><true/>
<key>CFBundleSupportedPlatforms</key><array><string>MacOSX</string></array>
<key>LSApplicationCategoryType</key><string>public.app-category.developer-tools</string>
<key>DTPlatformName</key><string>macosx</string>
<key>NSHumanReadableCopyright</key><string>Copyright © 2026 KWIP LLC</string>
</dict></plist>
PLIST
/usr/libexec/PlistBuddy -c "Set :CFBundleName $PRODUCT_DISPLAY_NAME Development" "$APP_PATH/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleDisplayName $PRODUCT_DISPLAY_NAME Development" "$APP_PATH/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Add :DTSDKName string macosx$SDK_VERSION" "$APP_PATH/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Add :DTSDKBuild string $SDK_BUILD" "$APP_PATH/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Add :DTPlatformVersion string $SDK_VERSION" "$APP_PATH/Contents/Info.plist"
DATABASE_HELPER_ENTITLEMENTS="$PROJECT_ROOT/packaging/DatabaseSourceService.entitlements"
if [[ "$BUILD_CONFIGURATION" == debug ]]; then
  DATABASE_HELPER_ENTITLEMENTS="$PROJECT_ROOT/.local/DatabaseSourceService.debug.entitlements"
  "${KWIPML_BUILD_PYTHON:-python3}" "$PROJECT_ROOT/scripts/database_broker_package.py" --app "$APP_PATH" --products "$BUILD_DIR" --development-entitlements "$DATABASE_HELPER_ENTITLEMENTS"
else
  "${KWIPML_BUILD_PYTHON:-python3}" "$PROJECT_ROOT/scripts/database_broker_package.py" --app "$APP_PATH" --products "$BUILD_DIR"
fi
codesign --force --sign - --entitlements "$DATABASE_HELPER_ENTITLEMENTS" "$APP_PATH/Contents/XPCServices/DatabaseSourceService.xpc"
"${KWIPML_BUILD_PYTHON:-python3}" "$PROJECT_ROOT/scripts/mcp_broker_package.py" --app "$APP_PATH" --products "$BUILD_DIR" --development
codesign --force --sign - --entitlements "$PROJECT_ROOT/packaging/MCPService.entitlements" "$APP_PATH/Contents/XPCServices/MCPService.xpc"
MODEL_HELPER_ENTITLEMENTS="$PROJECT_ROOT/packaging/NetworkService.entitlements"
if [[ "$BUILD_CONFIGURATION" == debug ]]; then
  MODEL_HELPER_ENTITLEMENTS="$PROJECT_ROOT/.local/NetworkService.debug.entitlements"
  "${KWIPML_BUILD_PYTHON:-python3}" "$PROJECT_ROOT/scripts/model_hub_broker_package.py" --app "$APP_PATH" --products "$BUILD_DIR" --development-entitlements "$MODEL_HELPER_ENTITLEMENTS"
else
  "${KWIPML_BUILD_PYTHON:-python3}" "$PROJECT_ROOT/scripts/model_hub_broker_package.py" --app "$APP_PATH" --products "$BUILD_DIR"
fi
codesign --force --sign "$DEVELOPMENT_SIGNING_IDENTITY" --entitlements "$MODEL_HELPER_ENTITLEMENTS" "$APP_PATH/Contents/XPCServices/NetworkService.xpc"
codesign --force --sign "$DEVELOPMENT_SIGNING_IDENTITY" "$APP_PATH"
codesign --verify --deep --strict "$APP_PATH"
printf '%s\n' "$APP_PATH"
