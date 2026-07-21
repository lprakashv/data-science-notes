#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")"

if ! command -v mdbook >/dev/null 2>&1; then
  echo "mdbook is required. Install it with: cargo install mdbook --version 0.5.4 --locked" >&2
  exit 1
fi

if ! python3 -m pipenv --version >/dev/null 2>&1; then
  echo "Pipenv is required. Install it with: python3 -m pip install pipenv" >&2
  exit 1
fi

./jnb_convert_script.sh
mdbook build
