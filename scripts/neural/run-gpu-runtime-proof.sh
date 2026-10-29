#!/usr/bin/env bash
set -euo pipefail

: "$SPIRE_LEARN_RUN_NEURAL_GPU_RUNTIME"
: "$SPIRE_LEARN_NEURAL_GPU_UUID"
: "$SPIRE_LEARN_GPU_PROOF_REPO_ROOT"
: "$SPIRE_LEARN_GPU_PROOF_SOURCE_SNAPSHOT"
: "$SPIRE_LEARN_GPU_PROOF_PARENT_RELEASE"
: "$SPIRE_LEARN_GPU_PROOF_EXCLUSIVE_CLAIM"
: "$SPIRE_LEARN_GPU_PROOF_RUN_ID"
: "$SPIRE_LEARN_GPU_PROOF_OUTPUT_CHILD"

if [[ "$SPIRE_LEARN_GPU_PROOF_REPO_ROOT" != /* ]]; then
    echo "GPU proof repository root must be an absolute path" >&2
    exit 2
fi
if [[ ! -x /usr/bin/timeout ]]; then
    echo "GNU timeout is required for the bounded proof watchdog" >&2
    exit 2
fi
exec /usr/bin/timeout --signal=TERM --kill-after=30s 270s python3.12 -B \
    "$SPIRE_LEARN_GPU_PROOF_REPO_ROOT/scripts/neural/gpu_runtime_proof_helpers.py" \
    run \
    --root "$SPIRE_LEARN_GPU_PROOF_REPO_ROOT" \
    --snapshot "$SPIRE_LEARN_GPU_PROOF_SOURCE_SNAPSHOT" \
    --release "$SPIRE_LEARN_GPU_PROOF_PARENT_RELEASE" \
    --claim "$SPIRE_LEARN_GPU_PROOF_EXCLUSIVE_CLAIM" \
    --run-id "$SPIRE_LEARN_GPU_PROOF_RUN_ID" \
    --output-child "$SPIRE_LEARN_GPU_PROOF_OUTPUT_CHILD"
