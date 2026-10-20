#!/bin/bash
# Play with your face and hands (profile: face + hand).
# Calibrates at every start (rest your hand BESIDE your face, where it sits while playing). Starts PAUSED: show both open palms for 1 s
# (or press F8 / Space in the HUD window) to go live. Esc in the HUD quits.
source "$(dirname "$0")/_env.sh"
"$PY" -m mcbio --profile profiles/face_hand.yaml --backend pynput --log-dir sessions --calibrate "$@"
read -n 1 -s -r -p "Press any key to close..."
