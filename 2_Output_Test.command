#!/bin/bash
# Output test - NO camera needed. A scripted "virtual player" drives the real
# keyboard/mouse output for 40 s so you can check that Minecraft responds.
# Start this, then click into your Minecraft world within 8 seconds.
source "$(dirname "$0")/_env.sh"
echo "Click into your Minecraft world now. Output starts in 8 seconds and runs ~40 s."
echo "Stop any time: press F8 (pause) or Ctrl-C in this window."
"$PY" -m mcbio --synthetic --realtime --backend pynput --no-hud --start-delay 8 --frames 1200
read -n 1 -s -r -p "Done. Press any key to close..."
