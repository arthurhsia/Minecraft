#!/bin/bash
# Step-by-step check of every Minecraft action (no camera needed).
source "$(dirname "$0")/_env.sh"
"$PY" scripts/diagnose_output.py
read -n 1 -s -r -p "Press any key to close..."
