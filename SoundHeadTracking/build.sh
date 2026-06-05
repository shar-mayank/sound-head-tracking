#!/bin/bash
set -euo pipefail

# -----------------------------------------------------------------------
# build.sh — Compile the Swift menu bar app and package as .app / .dmg
#
# Usage:
#     cd SoundHeadTracking && chmod +x build.sh && ./build.sh
# -----------------------------------------------------------------------

APP_NAME="Sound Head Tracking"
BUNDLE_NAME="SoundHeadTracking"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
DIST_DIR="${SCRIPT_DIR}/dist"
APP_BUNDLE="${DIST_DIR}/${APP_NAME}.app"

echo "=== Building ${APP_NAME} (native Swift) ==="

# Pre-flight check
if ! command -v swiftc &>/dev/null; then
    echo "ERROR: swiftc not found. Install Xcode or Xcode Command Line Tools."
    exit 1
fi

SDK_PATH="$(xcrun --show-sdk-path 2>/dev/null || true)"
if [ -z "$SDK_PATH" ]; then
    echo "ERROR: Could not locate macOS SDK. Run: xcode-select --install"
    exit 1
fi

# Clean
rm -rf "${BUILD_DIR}" "${DIST_DIR}"
mkdir -p "${BUILD_DIR}" "${DIST_DIR}"

# Detect architecture
ARCH="$(uname -m)"   # arm64 or x86_64

echo "  Architecture : ${ARCH}"
echo "  SDK          : ${SDK_PATH}"

# Compile
echo "  Compiling Swift sources..."
swiftc \
    -O \
    -whole-module-optimization \
    -target "${ARCH}-apple-macos13.0" \
    -sdk "${SDK_PATH}" \
    -framework AppKit \
    -framework AVFoundation \
    -framework Vision \
    -framework CoreAudio \
    -framework CoreMedia \
    -framework UserNotifications \
    -o "${BUILD_DIR}/${BUNDLE_NAME}" \
    "${SCRIPT_DIR}"/Sources/*.swift

echo "  Compilation successful."

# Create .app bundle
echo "  Creating app bundle..."
mkdir -p "${APP_BUNDLE}/Contents/MacOS"
mkdir -p "${APP_BUNDLE}/Contents/Resources"

cp "${BUILD_DIR}/${BUNDLE_NAME}" "${APP_BUNDLE}/Contents/MacOS/${APP_NAME}"
cp "${SCRIPT_DIR}/Info.plist"    "${APP_BUNDLE}/Contents/"

# Code-sign (ad-hoc)
echo "  Code signing..."
codesign --force --deep --sign - \
    --entitlements /dev/stdin "${APP_BUNDLE}" <<'ENTITLEMENTS'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.security.device.camera</key>
    <true/>
</dict>
</plist>
ENTITLEMENTS

echo "  App bundle ready: ${APP_BUNDLE}"

# Create DMG
echo "  Creating DMG..."
hdiutil create \
    -volname "${APP_NAME}" \
    -srcfolder "${APP_BUNDLE}" \
    -ov -format UDZO \
    "${DIST_DIR}/${BUNDLE_NAME}.dmg" \
    >/dev/null

echo ""
echo "=== Build complete ==="
echo "  App : ${APP_BUNDLE}"
echo "  DMG : ${DIST_DIR}/${BUNDLE_NAME}.dmg"
echo ""
echo "To install:"
echo "  open \"${DIST_DIR}/${BUNDLE_NAME}.dmg\""
echo "  # Drag to /Applications"
