#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")"

venv_python=".venv/bin/python"
requirements_marker=".venv/.requirements-installed"

if [[ ! -x "$venv_python" ]]; then
  if ! command -v python3 >/dev/null; then
    echo "Python 3 is required to build the site." >&2
    exit 1
  fi

  python3 -m venv .venv
fi

if [[ ! -f "$requirements_marker" || requirements.txt -nt "$requirements_marker" ]]; then
  "$venv_python" -m pip install --isolated --index-url https://pypi.org/simple -r requirements.txt
  touch "$requirements_marker"
fi

"$venv_python" -m mkdocs build --strict
