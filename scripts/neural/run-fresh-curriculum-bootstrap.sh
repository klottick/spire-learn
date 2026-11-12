#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")/../.."
fresh_bootstrap_parent_start="$(python3.12 -B -c 'import time; print(time.monotonic())')"
export FRESH_BOOTSTRAP_PARENT_START="$fresh_bootstrap_parent_start"
exec /usr/bin/timeout --signal=TERM --kill-after=30s 90s env PYTHONPATH=src:. python3.12 -B scripts/neural/fresh_curriculum_bootstrap.py
