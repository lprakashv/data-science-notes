#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")"

if ! command -v mdbook >/dev/null 2>&1; then
  echo "mdbook is required. Install it with: cargo install mdbook --version 0.5.4 --locked" >&2
  exit 1
fi

mdbook build
