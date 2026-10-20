#!/bin/bash
# Face-only control (no hands needed), e.g. for users without hand function.
source "$(dirname "$0")/_env.sh"
"$PY" -m mcbio --profile profiles/face_only.yaml --backend pynput --log-dir sessions --calibrate "$@"
read -n 1 -s -r -p "Press any key to close..."
