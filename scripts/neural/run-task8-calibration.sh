#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TASK_ROOT="$ROOT/artifacts/development/neural-combat/task-8"
CALIBRATION_ROOT="$TASK_ROOT/calibration"
HELPER="$ROOT/scripts/neural/task8_calibration_run_helpers.py"
SETUP_HELPER="$ROOT/scripts/neural/task8_gpu_setup_helpers.py"
SOURCE_SNAPSHOT="$CALIBRATION_ROOT/calibration-source-snapshot-v3.json"
RELEASE="$CALIBRATION_ROOT/phase-release-v3.json"
RELEASE_CLAIM="$CALIBRATION_ROOT/phase-release-continuation-1.claim.json"
VENV="$TASK_ROOT/runtime-output/cuda-setup-20261028-a/install-work/venv-cuda"
LOCK="$TASK_ROOT/runtime-output/cuda-setup-20261028-a/resolve-work/requirements-cuda.lock"
QUALIFICATION_REPORT="$TASK_ROOT/task-8-gpu-qualification-report.md"
QUALIFICATION_SUMMARY="$TASK_ROOT/task-8-gpu-qualification-summary.json"
IMAGE_ALIAS=spire-learn/task8-python:cuda-setup-20261028-a

if [[ "${SPIRE_LEARN_TASK8_CALIBRATION_BOUNDED:-}" != "1" ]]; then
    export SPIRE_LEARN_TASK8_CALIBRATION_BOUNDED=1
    export SPIRE_LEARN_TASK8_CALIBRATION_STARTED_EPOCH="$EPOCHREALTIME"
    exec timeout --signal=TERM --kill-after=90s 510s bash "$0"
fi
STARTED_EPOCH="${SPIRE_LEARN_TASK8_CALIBRATION_STARTED_EPOCH:-}"
if [[ ! "$STARTED_EPOCH" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    printf '%s\n' 'The bounded wrapper start timestamp is missing or malformed.' >&2
    exit 2
fi

if (( $# != 0 )); then
    printf '%s\n' 'usage: set SPIRE_LEARN_TASK8_CALIBRATION_RUN_ID and invoke this wrapper without arguments' >&2
    exit 2
fi
RUN_ID="${SPIRE_LEARN_TASK8_CALIBRATION_RUN_ID:-}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,39}$ ]]; then
    printf '%s\n' 'SPIRE_LEARN_TASK8_CALIBRATION_RUN_ID must be a fresh 1..40 character safe ID.' >&2
    exit 2
fi
if [[ "${SPIRE_LEARN_TASK8_CALIBRATION_RELEASED:-}" != "1" ]]; then
    printf '%s\n' 'Refusing Task 8 calibration: parent source and independent review release is unset.' >&2
    exit 2
fi
if [[ ! -f "$RELEASE" || ! -f "$SOURCE_SNAPSHOT" ]]; then
    printf '%s\n' 'Parent release or reviewed source snapshot is missing; no cohort may run.' >&2
    exit 2
fi
if [[ -e "$RELEASE_CLAIM" || -L "$RELEASE_CLAIM" ]]; then
    printf '%s\n' 'Parent release has already been claimed; it cannot authorize another cohort.' >&2
    exit 2
fi
if [[ ! -d "$VENV" || ! -r "$LOCK" || ! -r "$QUALIFICATION_REPORT" || ! -r "$QUALIFICATION_SUMMARY" ]]; then
    printf '%s\n' 'Retained CUDA runtime, lock, or qualification evidence is unavailable.' >&2
    exit 2
fi

RUNTIME_PARENT="$CALIBRATION_ROOT/runtime"
CONTROL_PARENT="$CALIBRATION_ROOT/runtime-control"
ADMIN_PARENT="$CALIBRATION_ROOT/runtime-admin"
RUN_ROOT="$RUNTIME_PARENT/$RUN_ID"
CONTROL_ROOT="$CONTROL_PARENT/$RUN_ID"
ADMIN_ROOT="$ADMIN_PARENT/$RUN_ID"
LOG_ROOT="$RUN_ROOT/logs"
OUTPUT_CHILD="$RUN_ROOT/output-child"
INVENTORY="$ADMIN_ROOT/evidence-inventory.json"
CONTAINER_NAME="spirelearn-task8-calibration-$RUN_ID"
CONTAINER_ID=
CAPTURED_RELEASE_SHA256=
CREATE_ATTEMPTED=0
CREATE_RETURNED=0
CONTAINER_OWNED=0
CONTAINER_STARTED=0
CONTAINER_REMOVED=0
RUN_ROOT_OWNED=0
CONTROL_ROOT_OWNED=0
ADMIN_ROOT_OWNED=0
SOURCE_PREFLIGHT_PASSED=0
CLEANUP_FAILED=0
FINALIZING=0
WRAPPER_STATUS=0

record() {
    local name="$1"
    shift
    local result=0
    printf '%q ' "$@" > "$LOG_ROOT/$name.command.log"
    printf '\n' >> "$LOG_ROOT/$name.command.log"
    if "$@" > "$LOG_ROOT/$name.stdout.log" 2> "$LOG_ROOT/$name.stderr.log"; then
        result=0
    else
        result=$?
    fi
    printf '%s\n' "$result" > "$LOG_ROOT/$name.exit.log"
    return "$result"
}

cleanup() {
    local saved_status=$?
    if (( FINALIZING )); then exit "$saved_status"; fi
    FINALIZING=1
    trap '' INT TERM
    trap - EXIT
    set +e

    if (( CONTAINER_OWNED )) && [[ -n "$CONTAINER_ID" ]]; then
        if record cleanup-inspect-before-stop docker inspect "$CONTAINER_ID"; then
            if record cleanup-owner-before-stop python3.12 -B "$HELPER" assert-owner \
                "$LOG_ROOT/cleanup-inspect-before-stop.stdout.log" "$CONTAINER_ID" "$RUN_ID"; then
                local running
                if record cleanup-running-state python3.12 -B "$HELPER" container-running \
                    "$LOG_ROOT/cleanup-inspect-before-stop.stdout.log" "$CONTAINER_ID" "$RUN_ID"; then
                    running="$(< "$LOG_ROOT/cleanup-running-state.stdout.log")"
                    if [[ "$running" == true ]]; then
                        if record container-stop docker stop --time 5 "$CONTAINER_ID"; then
                            :
                        else
                            CLEANUP_FAILED=1
                        fi
                    elif [[ "$running" != false ]]; then
                        CLEANUP_FAILED=1
                    fi
                else
                    CLEANUP_FAILED=1
                fi
            else
                CLEANUP_FAILED=1
            fi
        else
            CLEANUP_FAILED=1
        fi

        if record cleanup-inspect-before-remove docker inspect "$CONTAINER_ID"; then
            if record cleanup-owner-before-remove python3.12 -B "$HELPER" assert-owner \
                "$LOG_ROOT/cleanup-inspect-before-remove.stdout.log" "$CONTAINER_ID" "$RUN_ID"; then
                if record container-remove docker rm "$CONTAINER_ID"; then
                    CONTAINER_REMOVED=1
                else
                    CLEANUP_FAILED=1
                fi
            else
                CLEANUP_FAILED=1
            fi
        else
            CLEANUP_FAILED=1
        fi
    elif (( CREATE_ATTEMPTED )); then
        # A failed/uncertain create never grants ownership by container name.
        CLEANUP_FAILED=1
    fi

    if (( CREATE_ATTEMPTED )); then
        if record container-final-name-check docker ps -a -q --no-trunc \
            --filter "name=^/$CONTAINER_NAME$"; then
            if record container-final-name-assert python3.12 -B "$HELPER" assert-name-absent \
                "$LOG_ROOT/container-final-name-check.stdout.log"; then
                :
            else
                CLEANUP_FAILED=1
            fi
        else
            CLEANUP_FAILED=1
        fi
    fi

    if (( SOURCE_PREFLIGHT_PASSED )); then
        if record source-postcheck python3.12 -B "$HELPER" verify-source-snapshot \
            "$ROOT" "$SOURCE_SNAPSHOT"; then
            :
        else
            CLEANUP_FAILED=1
        fi
    fi

    if (( RUN_ROOT_OWNED )); then
        if record storage-postcheck python3.12 -B "$HELPER" verify-storage \
            "$ROOT" "$RUN_ROOT" "$CONTROL_ROOT" "$ADMIN_ROOT" "$OUTPUT_CHILD" \
            --reservation-bytes 16777216; then
            :
        else
            CLEANUP_FAILED=1
            if (( saved_status == 0 )); then saved_status=2; fi
        fi
    fi

    if (( RUN_ROOT_OWNED && CONTROL_ROOT_OWNED && ADMIN_ROOT_OWNED )); then
        if (( CLEANUP_FAILED )) && (( saved_status == 0 )); then saved_status=2; fi
        printf '{"container_create_attempted":%s,"create_returned_full_id":%s,"container_owned":%s,"container_started_once":%s,"container_removed":%s,"cleanup_failed":%s}\n' \
            "$CREATE_ATTEMPTED" "$CREATE_RETURNED" "$CONTAINER_OWNED" "$CONTAINER_STARTED" \
            "$CONTAINER_REMOVED" "$CLEANUP_FAILED" > "$CONTROL_ROOT/cleanup-status.json"
        printf '%s\n' "$saved_status" > "$CONTROL_ROOT/wrapper-preinventory-status.log"
        # All phase logs are closed before the immutable inventory begins.
        exec 1>&- 2>&-
        if timeout --signal=TERM --kill-after=2s 10s python3.12 -B "$HELPER" inventory-closed \
            "$INVENTORY" --logs-closed "$RUN_ROOT" "$CONTROL_ROOT" >/dev/null 2>&1; then
            printf '%s\n' 0 > "$ADMIN_ROOT/evidence-inventory.exit.log"
        else
            local inventory_status=$?
            printf '%s\n' "$inventory_status" > "$ADMIN_ROOT/evidence-inventory.exit.log"
            if (( saved_status == 0 )); then saved_status=2; fi
        fi
        if timeout --signal=TERM --kill-after=2s 10s python3.12 -B "$HELPER" write-host-lifecycle \
            "$ADMIN_ROOT/host-lifecycle.json" "$STARTED_EPOCH" "$saved_status" "$CLEANUP_FAILED" \
            --release-binding "$RUN_ROOT/release-binding.json" \
            >/dev/null 2>&1; then
            printf '%s\n' 0 > "$ADMIN_ROOT/host-lifecycle.exit.log"
        else
            local lifecycle_status=$?
            printf '%s\n' "$lifecycle_status" > "$ADMIN_ROOT/host-lifecycle.exit.log"
            if (( saved_status == 0 )); then saved_status=2; fi
        fi
    fi
    exit "$saved_status"
}

if [[ -e "$RUN_ROOT" || -L "$RUN_ROOT" || -e "$CONTROL_ROOT" || -L "$CONTROL_ROOT" \
      || -e "$ADMIN_ROOT" || -L "$ADMIN_ROOT" ]]; then
    printf '%s\n' 'Calibration run/control/admin path already exists; run IDs are never reused.' >&2
    exit 2
fi
mkdir -p "$RUNTIME_PARENT" "$CONTROL_PARENT" "$ADMIN_PARENT"
mkdir -m 700 "$RUN_ROOT"
RUN_ROOT_OWNED=1
mkdir -m 700 "$CONTROL_ROOT"
CONTROL_ROOT_OWNED=1
mkdir -m 700 "$ADMIN_ROOT"
ADMIN_ROOT_OWNED=1
mkdir -m 700 "$LOG_ROOT" "$OUTPUT_CHILD"
exec >> "$CONTROL_ROOT/wrapper.stdout.log" 2>> "$CONTROL_ROOT/wrapper.stderr.log"
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [[ "$(stat -c '%u:%g' "$OUTPUT_CHILD")" != "1000:1000" ]]; then
    printf '%s\n' 'Fresh writable output child must already be owned by UID:GID 1000:1000.' >&2
    exit 2
fi

record source-preflight python3.12 -B "$HELPER" verify-source-snapshot "$ROOT" "$SOURCE_SNAPSHOT"
SOURCE_PREFLIGHT_PASSED=1
record release-verification python3.12 -B "$HELPER" verify-release \
    "$ROOT" "$RELEASE" "$SOURCE_SNAPSHOT" --output "$RUN_ROOT/release-binding.json"
CAPTURED_RELEASE_SHA256="$(python3.12 -B -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["release_sha256"])' \
    "$RUN_ROOT/release-binding.json")"
if [[ ! "$CAPTURED_RELEASE_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    printf '%s\n' 'Captured parent release SHA256 is missing or malformed.' >&2
    exit 2
fi
record gpu-preflight python3.12 -B "$HELPER" verify-gpu
record storage-preflight python3.12 -B "$HELPER" verify-storage \
    "$ROOT" "$RUN_ROOT" "$CONTROL_ROOT" "$ADMIN_ROOT" "$OUTPUT_CHILD"
record image-inspect docker image inspect "$IMAGE_ALIAS"
record image-assertion python3.12 -B "$HELPER" verify-image-alias \
    "$LOG_ROOT/image-inspect.stdout.log" --output "$RUN_ROOT/image-identity.json"

IMAGE_ID="$(python3.12 -B -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["image_id"])' \
    "$RUN_ROOT/image-identity.json")"
python3.12 -B "$HELPER" write-container-argv "$ROOT" "$OUTPUT_CHILD" "$RUN_ID" \
    "$IMAGE_ID" "$RUN_ROOT/release-binding.json" "$CONTROL_ROOT/docker-create-argv.nul"
mapfile -d '' -t CREATE_ARGV < "$CONTROL_ROOT/docker-create-argv.nul"
if (( ${#CREATE_ARGV[@]} < 30 )); then
    printf '%s\n' 'Calibration Docker create argv did not pass its structural preflight.' >&2
    exit 2
fi

record container-name-preflight docker ps -a -q --no-trunc --filter "name=^/$CONTAINER_NAME$"
record container-name-assertion python3.12 -B "$HELPER" assert-name-absent \
    "$LOG_ROOT/container-name-preflight.stdout.log"
record release-single-use-claim python3.12 -B "$HELPER" claim-continuation \
    "$ROOT" "$RELEASE" "$SOURCE_SNAPSHOT" "$RELEASE_CLAIM" "$RUN_ID" \
    --expected-release-sha256 "$CAPTURED_RELEASE_SHA256"
record release-claim-binding-assert python3.12 -B "$HELPER" assert-claim-release-binding \
    "$LOG_ROOT/release-single-use-claim.stdout.log" "$CAPTURED_RELEASE_SHA256"

CREATE_ATTEMPTED=1
if record container-create "${CREATE_ARGV[@]}"; then
    CREATE_RETURNED=1
else
    WRAPPER_STATUS=$?
    exit "$WRAPPER_STATUS"
fi
record create-id-assertion python3.12 -B "$HELPER" parse-created-id \
    "$LOG_ROOT/container-create.stdout.log"
CONTAINER_ID="$(< "$LOG_ROOT/create-id-assertion.stdout.log")"
if [[ ! "$CONTAINER_ID" =~ ^[0-9a-f]{64}$ ]]; then
    printf '%s\n' 'The recorded create result is not one full Docker ID.' >&2
    exit 2
fi
printf '%s\n' "$CONTAINER_ID" > "$CONTROL_ROOT/owned-container-id.log"
record container-inspect-created docker inspect "$CONTAINER_ID"
record container-owner-assertion python3.12 -B "$HELPER" assert-owner \
    "$LOG_ROOT/container-inspect-created.stdout.log" "$CONTAINER_ID" "$RUN_ID"
CONTAINER_OWNED=1
record container-inspect-assertion python3.12 -B "$HELPER" assert-inspect \
    "$LOG_ROOT/container-inspect-created.stdout.log" "$CONTAINER_ID" "$RUN_ID" \
    "$IMAGE_ID" "$ROOT" "$OUTPUT_CHILD"

CONTAINER_STARTED=1
if record container-start docker start --attach "$CONTAINER_ID"; then
    WRAPPER_STATUS=0
else
    WRAPPER_STATUS=$?
fi
record container-inspect-finished docker inspect "$CONTAINER_ID"
record container-finished-assertion python3.12 -B "$HELPER" assert-finished \
    "$LOG_ROOT/container-inspect-finished.stdout.log" "$CONTAINER_ID" "$RUN_ID"
record unittest-transcript-assertion python3.12 -B "$HELPER" verify-transcript \
    "$LOG_ROOT/container-start.stdout.log"
if (( WRAPPER_STATUS != 0 )); then
    exit "$WRAPPER_STATUS"
fi
exit 0
