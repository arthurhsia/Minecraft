#!/bin/bash
# One-time setup: creates a private Python environment in this folder (.venv),
# installs the dependencies and downloads the two MediaPipe model files.
source "$(dirname "$0")/_env.sh"
pause_exit() { echo; read -n 1 -s -r -p "Press any key to close..."; exit "${1:-0}"; }
trap 'echo; echo "!! Setup stopped with an error (see the message above)."; pause_exit 1' ERR
set -eE
echo "== Minecraft Biogaming setup =="

CONDA=""
for c in "$(command -v conda 2>/dev/null)" "$HOME/opt/anaconda3/bin/conda" "$HOME/anaconda3/bin/conda" \
         "$HOME/miniconda3/bin/conda" "/opt/homebrew/anaconda3/bin/conda" "/opt/anaconda3/bin/conda"; do
  if [ -n "$c" ] && [ -x "$c" ]; then CONDA="$c"; break; fi
done

# Apple Silicon Macs need a NATIVE arm64 Python: MediaPipe stopped publishing
# Intel-Mac (x86_64) builds at 0.10.21. An Intel Anaconda can still create an
# arm64 environment when CONDA_SUBDIR=osx-arm64 is set.
ARM=0; [ "$(sysctl -n hw.optional.arm64 2>/dev/null)" = "1" ] && ARM=1
want_arch() { if [ $ARM = 1 ]; then echo arm64; else echo x86_64; fi; }
env_ok() {
  [ -x "$PY" ] && "$PY" -c "import sys,platform; assert sys.version_info >= (3,10); assert platform.machine() == '$(want_arch)'" 2>/dev/null
}

make_env() {
  # Try several Python versions: a corrupted cached package for one version
  # (seen with conda's python-3.11 cache) should not block setup.
  if [ -n "$CONDA" ]; then
    for v in 3.12 3.11 3.10; do
      rm -rf "$ENV_DIR"
      echo "Creating Python $v environment with conda ($CONDA) ..."
      if [ $ARM = 1 ]; then export CONDA_SUBDIR=osx-arm64; fi
      if "$CONDA" create -y -q -p "$ENV_DIR" "python=$v" pip 2>&1 | grep -v "anaconda-cloud-auth\|ChannelAuthBase\|newer version of conda\|conda update\|conda install conda\|current version\|latest version\|^$"; then :; fi
      if env_ok; then
        echo "OK: $("$PY" --version) ($(want_arch))"; return 0
      fi
      echo "  ...Python $v failed, trying the next version."
    done
  fi
  for p in python3.12 python3.11 python3.10 /opt/homebrew/bin/python3 python3; do
    if command -v "$p" >/dev/null 2>&1 && "$p" -c "import sys,platform; assert (3,10) <= sys.version_info < (3,13); assert platform.machine() == '$(want_arch)'" 2>/dev/null; then
      rm -rf "$ENV_DIR"; echo "Creating venv with $p ..."; "$p" -m venv "$ENV_DIR"; return 0
    fi
  done
  echo "Could not create a Python 3.10-3.12 environment."
  echo "Fix: install Python 3.12 from https://www.python.org/downloads/macos/ and run this again."
  return 1
}

if ! env_ok; then make_env; fi
echo "Using $("$PY" -c 'import sys,platform; print(sys.version.split()[0], platform.machine())')"
"$PY" -m pip install --upgrade pip -q
"$PY" -m pip install -r requirements.txt
"$PY" scripts/download_models.py
echo
"$PY" -c "import mediapipe, cv2, pynput; print('OK: mediapipe', mediapipe.__version__, '| opencv', cv2.__version__)"
echo
echo "Setup finished. Next: double-click 2_Output_Test.command"
pause_exit 0
