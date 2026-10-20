# Shared helper for the .command launchers: finds the project's Python environment.
cd "$(dirname "$0")"
ENV_DIR="$PWD/.venv"
PY="$ENV_DIR/bin/python"
