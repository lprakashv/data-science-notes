#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")"

venv_python=".venv/bin/python"

./build-site.sh
echo "Serving the latest site at http://127.0.0.1:8000/data-science-notes/"
exec "$venv_python" -m mkdocs serve --strict
