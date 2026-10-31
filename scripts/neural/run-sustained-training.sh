#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -x /usr/bin/timeout ]]; then
    echo 'GNU timeout is required for the sustained campaign watchdog' >&2
    exit 2
fi
# Parent supplies the reviewed numeric ceiling. All Python validation and
# execution is inside this one guard; no derivation helper runs beforehand.
watchdog_seconds="${1:-}"
if [[ ! "$watchdog_seconds" =~ ^[1-9][0-9]{1,4}$ ]] || (( watchdog_seconds <= 10 || watchdog_seconds > 28800 )); then
    echo 'First argument must be the reviewed stage wall ceiling (11..28800 seconds)' >&2
    exit 2
fi
shift
export SPIRE_LEARN_SUSTAINED_WATCHDOG='parent-ceiling+10'
export SPIRE_LEARN_SUSTAINED_WATCHDOG_SECONDS="$watchdog_seconds"
exec /usr/bin/timeout --signal=TERM --kill-after=10s "$((watchdog_seconds - 10))s" \
    python3.12 -B "$script_dir/sustained_controls.py" "$@"
