#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")"
export PIPENV_VENV_IN_PROJECT=1

while IFS= read -r -d '' notebook; do
  python3 -m pipenv run jupyter nbconvert "$notebook" --to markdown --output-dir ./markdown-book
done < <(find ./ipy-notebooks -name "*.ipynb" ! -path "*/.ipynb_checkpoints/*" -print0)
