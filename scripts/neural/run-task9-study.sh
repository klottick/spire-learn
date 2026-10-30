#!/usr/bin/env bash
set -euo pipefail
if [[ ! -x /usr/bin/timeout ]]; then
    echo 'GNU timeout is required for the bounded Task 9 campaign watchdog' >&2
    exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SPIRE_LEARN_TASK9_WATCHDOG='4470+30'
exec /usr/bin/timeout --signal=TERM --kill-after=30s 4470s \
    python3.12 -B "$script_dir/task9_study_controls.py" run "$@"
