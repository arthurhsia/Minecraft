#!/bin/bash
# Wrap dist/Minecraft Biogaming.app in a drag-to-Applications .dmg.
# Usage: bash packaging/make_dmg.sh [version]   (run from the repo root after pyinstaller)
set -euo pipefail
VERSION="${1:-${MCBIO_VERSION:-0.1.0}}"
APP="dist/Minecraft Biogaming.app"
case "$(uname -m)" in arm64) ARCH=apple-silicon ;; x86_64) ARCH=intel ;; *) ARCH="$(uname -m)" ;; esac
[ -d "$APP" ] || { echo "missing $APP - run pyinstaller first"; exit 1; }

# Ad-hoc sign the whole bundle consistently (no Developer ID: see HOW_TO_OPEN.txt).
codesign --force --deep --sign - "$APP"

STAGE="$(mktemp -d)"
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
cp packaging/HOW_TO_OPEN.txt "$STAGE/How to open - READ ME.txt"
OUT="dist/Minecraft-Biogaming-${VERSION}-mac-${ARCH}.dmg"
rm -f "$OUT"
hdiutil create -volname "Minecraft Biogaming" -srcfolder "$STAGE" -ov -format UDZO "$OUT"
rm -rf "$STAGE"
echo "built $OUT ($(du -h "$OUT" | cut -f1))"
