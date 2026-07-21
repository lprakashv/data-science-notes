#!/usr/bin/env bash

set -euo pipefail

find ./ipy-notebooks -name "*.ipynb" ! -path "*/.ipynb_checkpoints/*" \
  -exec python3 -m pipenv run jupyter nbconvert {} --to markdown --output-dir ./markdown-book \;
