#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[say.sh] Virtual environment not found. Please run 'python3 -m venv .venv' and install requirements." >&2
  exit 1
fi
exec "$PYTHON_BIN" "$SCRIPT_DIR/say.py" "$@"
