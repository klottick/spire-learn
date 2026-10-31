#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -x /usr/bin/timeout ]]; then
    echo 'GNU timeout is required for the sustained campaign watchdog' >&2
    exit 2
fi
# Derive the remaining allowance from the exact reviewed release/predecessor.
watchdog_seconds="$(python3.12 -B "$script_dir/sustained_controls.py" "$@" --watchdog-seconds)"
if [[ ! "$watchdog_seconds" =~ ^[0-9]+$ ]] || (( watchdog_seconds <= 0 )); then
    echo 'No remaining campaign watchdog allowance' >&2
    exit 2
fi
export SPIRE_LEARN_SUSTAINED_WATCHDOG='remaining+10'
exec /usr/bin/timeout --signal=TERM --kill-after=10s "${watchdog_seconds}s" \
    python3.12 -B "$script_dir/sustained_controls.py" "$@"
