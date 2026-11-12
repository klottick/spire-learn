#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")/../.."
# Reserve forty seconds inside the transport and bound host stalls as well.
# Qualified timeout TERM at 90s plus a 30s KILL grace bounds host stalls too.
exec timeout --signal=TERM --kill-after=30s 90s env PYTHONPATH=src python3.12 -B scripts/neural/qualify_gym_release.py
