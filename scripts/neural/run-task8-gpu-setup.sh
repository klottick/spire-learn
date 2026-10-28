#!/usr/bin/env bash
set -Eeuo pipefail

INVOCATION_STARTED_EPOCH="${SPIRE_LEARN_TASK8_STARTED_EPOCH:-$EPOCHREALTIME}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TASK_ROOT="$ROOT/artifacts/development/neural-combat/task-8"
HELPER="$ROOT/scripts/neural/task8_gpu_setup_helpers.py"
CORRECTED_RED_SOURCE_SNAPSHOT="$TASK_ROOT/task-8-corrected-rng-reviewed-source-snapshot.json"
DEVICE_CHECK="$ROOT/scripts/neural/task8_device_check.py"
LOCATION="$ROOT/artifacts/development/neural-combat/cpu-runtime-location.json"
CPU_RUNTIME="$ROOT/artifacts/development/neural-combat/runtime/db72a7eb-c789-4cf4-bfa5-42c1ada5e3fc"
CPU_WHEEL_AUDIT="$CPU_RUNTIME/new-wheel-audit.json"
CPU_VENV_PATH="$CPU_RUNTIME/data/venv-cpu"
CPU_WHEEL_PATH="$CPU_RUNTIME/data/wheelhouse/spire_lab-0.1.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
TORCH_PREFLIGHT="$TASK_ROOT/parent-package-preflight/metadata-preflight.json"
TORCH_METADATA="$TASK_ROOT/parent-package-preflight/torch-wheel.metadata"
IMAGE_REFERENCE=docker.io/library/python@sha256:57d96f71de145a98e7ceca18ee4ade77bba4103a33a72dfaea026b23625c6bf0
GPU_UUID=GPU-c05e0ce3-c1c2-0c54-14e6-8a4a2eccaa1d
RNG_TEST_CLASS=tests.neural.test_gpu_qualification.GpuRngRegressionTests

if (( $# != 4 && $# != 6 )); then
    printf '%s\n' 'usage: run-task8-gpu-setup.sh {resolve|install|rng-red|rng-green|device-check} RUN_ID ABSOLUTE_PARENT_RELEASE.json ABSOLUTE_SOURCE_SNAPSHOT.json [--attempt-id corrected-1]' >&2
    exit 2
fi
PHASE="$1"
RUN_ID="$2"
RELEASE="$3"
SOURCE_SNAPSHOT="$4"
ATTEMPT_ID=
if (( $# == 6 )); then
    if [[ "$5" != --attempt-id || "$6" != corrected-1 ]]; then
        printf '%s\n' 'Only --attempt-id corrected-1 is accepted for the reviewed RNG correction.' >&2
        exit 2
    fi
    ATTEMPT_ID="$6"
fi
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,39}$ ]]; then
    printf '%s\n' 'RUN_ID must be a fresh 1..40 character safe identifier.' >&2
    exit 2
fi
case "$PHASE" in
    resolve|install|rng-red|rng-green|device-check) ;;
    *) printf '%s\n' 'Unsupported Task 8 phase.' >&2; exit 2 ;;
esac
if [[ -n "$ATTEMPT_ID" && ( "$PHASE" != rng-red || "$RUN_ID" != cuda-setup-20261028-a ) ]]; then
    printf '%s\n' 'The corrected attempt is limited to the retained cuda-setup-20261028-a RNG RED phase.' >&2
    exit 2
fi
if [[ ( "$PHASE" == rng-green || "$PHASE" == device-check ) && "$RUN_ID" != cuda-setup-20261028-a ]]; then
    printf '%s\n' 'Later workload phases must bind the retained corrected RNG run.' >&2
    exit 2
fi
if [[ "$RELEASE" != /* || "$SOURCE_SNAPSHOT" != /* ]]; then
    printf '%s\n' 'Parent release and source snapshot paths must be absolute.' >&2
    exit 2
fi
if [[ "$ATTEMPT_ID" == corrected-1 && "$SOURCE_SNAPSHOT" != "$CORRECTED_RED_SOURCE_SNAPSHOT" ]]; then
    printf '%s\n' 'The corrected RNG attempt requires the exact reviewed corrected RED source snapshot path.' >&2
    exit 2
fi
if [[ "${SPIRE_LEARN_TASK8_BOUNDED:-}" != 1 ]]; then
    python3.12 -B "$HELPER" verify-safe-inputs "$TASK_ROOT" "$RELEASE" "$SOURCE_SNAPSHOT" >/dev/null
    elapsed_rounded_up="$(python3.12 -B - "$INVOCATION_STARTED_EPOCH" <<'PY'
import math
import sys
import time
print(math.ceil(time.time() - float(sys.argv[1])))
PY
)"
    if [[ "$PHASE" == install ]]; then
        budget_path="$TASK_ROOT/runtime-control/$RUN_ID/resolve/setup-budget-state.json"
        budget_json="$(python3.12 -B "$HELPER" read-setup-budget-state "$budget_path")"
        hard_remaining="$(python3.12 -B -c 'import json,sys; print(json.load(sys.stdin)["remaining_hard_seconds"])' <<< "$budget_json")"
    elif [[ "$PHASE" == resolve ]]; then
        hard_remaining=1200
    else
        hard_remaining=120
    fi
    wrapper_seconds=$((hard_remaining - elapsed_rounded_up - 1))
    if (( wrapper_seconds <= 0 )); then
        printf '%s\n' 'Cumulative hard time allowance is exhausted before phase startup.' >&2
        exit 124
    fi
    export SPIRE_LEARN_TASK8_BOUNDED=1
    export SPIRE_LEARN_TASK8_STARTED_EPOCH="$INVOCATION_STARTED_EPOCH"
    if [[ -n "$ATTEMPT_ID" ]]; then
        exec timeout --signal=TERM --kill-after=1s "${wrapper_seconds}s" \
            bash "$0" "$PHASE" "$RUN_ID" "$RELEASE" "$SOURCE_SNAPSHOT" \
            --attempt-id "$ATTEMPT_ID"
    fi
    exec timeout --signal=TERM --kill-after=1s "${wrapper_seconds}s" \
        bash "$0" "$PHASE" "$RUN_ID" "$RELEASE" "$SOURCE_SNAPSHOT"
fi
if [[ "$RELEASE" != "$TASK_ROOT/"* || "$SOURCE_SNAPSHOT" != "$TASK_ROOT/"* ]]; then
    printf '%s\n' 'Parent release and source snapshot must remain inside the Task 8 evidence root.' >&2
    exit 2
fi
python3.12 -B "$HELPER" verify-safe-inputs "$TASK_ROOT" "$RELEASE" "$SOURCE_SNAPSHOT" >/dev/null
cd "$ROOT"

RUNTIME_ROOT="$TASK_ROOT/runtime/$RUN_ID"
OUTPUT_ROOT="$TASK_ROOT/runtime-output/$RUN_ID"
CONTROL_ROOT="$TASK_ROOT/runtime-control/$RUN_ID"
ADMIN_ROOT="$TASK_ROOT/runtime-admin/$RUN_ID"
DOCKER_CONFIG_ROOT="$TASK_ROOT/docker-config/$RUN_ID"
PHASE_KEY="$PHASE"
CONTAINER_NAME="spirelearn-task8-$PHASE-$RUN_ID"
CORRECTED_RNG_PROJECTION=
if [[ -n "$ATTEMPT_ID" || "$PHASE" == rng-green || "$PHASE" == device-check ]]; then
    CORRECTED_RNG_PROJECTION="$(python3.12 -B "$HELPER" project-corrected-rng-attempt \
        "$TASK_ROOT" "$RUN_ID" corrected-1)"
fi
CORRECTED_RED_TRANSCRIPT=
CORRECTED_RED_INVENTORY=
ORIGINAL_FAILED_INVENTORY=
if [[ -n "$CORRECTED_RNG_PROJECTION" ]]; then
    CORRECTED_RED_TRANSCRIPT="$(python3.12 -B -c 'import json,sys; print(json.load(sys.stdin)["corrected_transcript_path"])' <<< "$CORRECTED_RNG_PROJECTION")"
    CORRECTED_RED_INVENTORY="$(python3.12 -B -c 'import json,sys; print(json.load(sys.stdin)["corrected_inventory_path"])' <<< "$CORRECTED_RNG_PROJECTION")"
    ORIGINAL_FAILED_INVENTORY="$(python3.12 -B -c 'import json,sys; print(json.load(sys.stdin)["previous_failed_inventory_path"])' <<< "$CORRECTED_RNG_PROJECTION")"
fi
if [[ -n "$ATTEMPT_ID" ]]; then
    PHASE_KEY="$(python3.12 -B -c 'import json,sys; print(json.load(sys.stdin)["phase_key"])' <<< "$CORRECTED_RNG_PROJECTION")"
    CONTAINER_NAME="$(python3.12 -B -c 'import json,sys; print(json.load(sys.stdin)["container_name"])' <<< "$CORRECTED_RNG_PROJECTION")"
fi
PHASE_OUTPUT="$OUTPUT_ROOT/$PHASE_KEY-work"
PHASE_CONTROL="$CONTROL_ROOT/$PHASE_KEY"
LOG_ROOT="$ADMIN_ROOT/$PHASE_KEY/logs"
PHASE_ADMIN="$ADMIN_ROOT/$PHASE_KEY"
IMAGE_ALIAS="spire-learn/task8-python:$RUN_ID"
RESOLVE_OUTPUT="$OUTPUT_ROOT/resolve-work"
INSTALL_OUTPUT="$OUTPUT_ROOT/install-work"
LOCK_PATH="$RESOLVE_OUTPUT/requirements-cuda.lock"
WHEELHOUSE="$RESOLVE_OUTPUT/wheelhouse"
VENV_PATH="$INSTALL_OUTPUT/venv-cuda"
CONTAINER_ID=
CREATE_ATTEMPTED=0
CONTAINER_START_ATTEMPTED=0
CONTAINER_START_INVOCATION_RETURNED=0
CONTAINER_STARTED=0
ALIAS_CREATED=0
CLEANUP_FAILED=0
FINALIZING=0
ROOTS_OWNED=0
START_SECONDS=$SECONDS
START_EPOCH_US="${INVOCATION_STARTED_EPOCH/./}"
SETUP_ACTIVE_BEFORE=0
SETUP_SOFT_REMAINING=600
SETUP_HARD_REMAINING=1200
EXTERNAL_PARENT_PAUSE=0
SOURCE_HASH=
LOCK_SHA=
SOURCE_VERIFIED_BEFORE=0
PREVIOUS_FAILED_INVENTORY_SHA=
PRIOR_INVENTORIES=()
PRIOR_PHASES=()

case "$PHASE" in
    resolve) PRIOR_PHASES=() ;;
    install) PRIOR_PHASES=(resolve) ;;
    rng-red)
        if [[ -n "$ATTEMPT_ID" ]]; then PRIOR_PHASES=(resolve install rng-red); else PRIOR_PHASES=(resolve install); fi ;;
    rng-green) PRIOR_PHASES=(resolve install rng-red-corrected-1) ;;
    device-check) PRIOR_PHASES=(resolve install rng-red-corrected-1 rng-green) ;;
esac
for prior_phase in "${PRIOR_PHASES[@]}"; do
    PRIOR_INVENTORIES+=("$ADMIN_ROOT/$prior_phase/evidence-inventory.json")
done

if [[ -n "$ATTEMPT_ID" ]]; then
    if [[ ! -f "$LOCK_PATH" || -L "$LOCK_PATH" \
            || "$(realpath -e -- "$LOCK_PATH")" != "$LOCK_PATH" ]]; then
        printf '%s\n' 'The reviewed corrected RNG release requires the exact retained regular CUDA lock.' >&2
        exit 2
    fi
    SOURCE_HASH="$(sha256sum "$SOURCE_SNAPSHOT" | awk '{print $1}')"
    LOCK_SHA="$(sha256sum "$LOCK_PATH" | awk '{print $1}')"
    python3.12 -B "$HELPER" verify-source-snapshot "$SOURCE_SNAPSHOT" "$ROOT" >/dev/null
    python3.12 -B "$HELPER" verify-corrected-rng-red-release \
        "$RELEASE" "$TASK_ROOT" "$RUN_ID" "$SOURCE_HASH" "$LOCK_SHA" \
        --allow-venv "$CPU_VENV_PATH" --allow-venv "$VENV_PATH" >/dev/null
fi

elapsed_seconds_ceil() {
    local now_us="${EPOCHREALTIME/./}"
    echo "$(((now_us - START_EPOCH_US + 999999) / 1000000))"
}

if [[ "$PHASE" == install ]]; then
    BUDGET_STATE="$CONTROL_ROOT/resolve/setup-budget-state.json"
    BUDGET_JSON="$(python3.12 -B "$HELPER" read-setup-budget-state "$BUDGET_STATE")"
    SETUP_ACTIVE_BEFORE="$(python3.12 -B -c 'import json,sys; print(json.load(sys.stdin)["active_seconds_before"])' <<< "$BUDGET_JSON")"
    SETUP_SOFT_REMAINING="$(python3.12 -B -c 'import json,sys; print(json.load(sys.stdin)["remaining_soft_seconds"])' <<< "$BUDGET_JSON")"
    SETUP_HARD_REMAINING="$(python3.12 -B -c 'import json,sys; print(json.load(sys.stdin)["remaining_hard_seconds"])' <<< "$BUDGET_JSON")"
    EXTERNAL_PARENT_PAUSE="$(python3.12 -B - "$RELEASE" <<'PY'
import json
import sys
from pathlib import Path
record=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
value=record.get("external_parent_review_pause_seconds")
if type(value) is not int or value < 0:
    raise SystemExit("parent install release must carry external_parent_review_pause_seconds")
print(value)
PY
)"
fi

record() {
    local name="$1" status=0
    shift
    printf '%q ' "$@" > "$LOG_ROOT/$name.command.log"
    printf '\n' >> "$LOG_ROOT/$name.command.log"
    if "$@" > "$LOG_ROOT/$name.stdout.log" 2> "$LOG_ROOT/$name.stderr.log"; then status=0; else status=$?; fi
    printf '%s\n' "$status" > "$LOG_ROOT/$name.exit.log"
    return "$status"
}

record_started_container() {
    local status=0 timeout_seconds=60 kill_after=10 elapsed remaining
    if [[ "$PHASE" == resolve || "$PHASE" == install ]]; then
        timeout_seconds="$SETUP_SOFT_REMAINING"
    fi
    elapsed="$(elapsed_seconds_ceil)"
    remaining=$((timeout_seconds - elapsed - 15))
    if (( remaining <= 0 )); then
        printf '%s\n' 'Cumulative phase soft budget plus cleanup reserve is exhausted before container start.' >&2
        return 124
    fi
    timeout_seconds="$remaining"
    printf '%q ' timeout --signal=TERM --kill-after="${kill_after}s" "${timeout_seconds}s" docker start -a "$CONTAINER_ID" \
        > "$LOG_ROOT/container-start.command.log"
    printf '\n' >> "$LOG_ROOT/container-start.command.log"
    CONTAINER_START_ATTEMPTED=1
    if timeout --signal=TERM --kill-after="${kill_after}s" "${timeout_seconds}s" docker start -a "$CONTAINER_ID" \
        > "$LOG_ROOT/container-start.stdout.log" 2> "$LOG_ROOT/container-start.stderr.log"; then
        status=0
    else
        status=$?
    fi
    CONTAINER_START_INVOCATION_RETURNED=1
    printf '%s\n' "$status" > "$LOG_ROOT/container-start.exit.log"
    return "$status"
}

cleanup() {
    local status=$?
    if (( FINALIZING )); then exit "$status"; fi
    FINALIZING=1
    trap '' INT TERM
    trap - EXIT
    set +e
    if (( CREATE_ATTEMPTED )); then
        local cleanup_container_verified=0
        if [[ -n "$CONTAINER_ID" && -f "$PHASE_CONTROL/container-spec.json" ]]; then
            if record cleanup-container-inspect timeout --signal=TERM --kill-after=1s 5s docker inspect "$CONTAINER_ID" \
            && record cleanup-container-assert timeout --signal=TERM --kill-after=1s 5s \
                python3.12 -B "$HELPER" assert-container-inspect \
                "$LOG_ROOT/cleanup-container-inspect.stdout.log" "$PHASE_CONTROL/container-spec.json"; then
                cleanup_container_verified=1
            else
                CLEANUP_FAILED=1
            fi
        fi
        if (( cleanup_container_verified )); then
            if record cleanup-container-removal-authorization timeout --signal=TERM --kill-after=1s 5s \
                python3.12 -B "$HELPER" authorize-container-removal "$CONTAINER_ID" \
                "$LOG_ROOT/container-create.exit.log" "$LOG_ROOT/container-create.stdout.log" \
                "$LOG_ROOT/cleanup-container-assert.exit.log" \
                "$PHASE_CONTROL/container-removal-authorization.json"; then
                if ! record container-remove timeout --signal=TERM --kill-after=2s 8s docker rm -f "$CONTAINER_ID"; then
                    CLEANUP_FAILED=1
                fi
            else
                CLEANUP_FAILED=1
            fi
        fi
        if record container-final-list timeout --signal=TERM --kill-after=1s 4s \
            docker ps -a -q --no-trunc --filter "name=^/$CONTAINER_NAME$"; then
            if [[ -s "$LOG_ROOT/container-final-list.stdout.log" ]]; then CLEANUP_FAILED=1; fi
        else
            CLEANUP_FAILED=1
        fi
    fi
    if (( ALIAS_CREATED && status != 0 )); then
        if ! record image-alias-remove timeout --signal=TERM --kill-after=2s 8s docker image rm "$IMAGE_ALIAS"; then
            CLEANUP_FAILED=1
        fi
    fi
    if (( ROOTS_OWNED )); then
        local scope=setup
        if [[ "$PHASE" == rng-red || "$PHASE" == rng-green || "$PHASE" == device-check ]]; then scope=workload; fi
        if ! record storage-final timeout --signal=TERM --kill-after=2s 12s \
            python3.12 -B "$HELPER" verify-storage "$ROOT" "$ROOT/artifacts/development/neural-combat" \
            "$scope" "$TASK_ROOT/runtime" "$TASK_ROOT/runtime-output" "$TASK_ROOT/runtime-control" \
            "$TASK_ROOT/runtime-admin" "$TASK_ROOT/docker-config" \
            --reservation-bytes 1073741824 --protected-cpu-venv "$CPU_VENV_PATH" \
            --allow-venv "$CPU_VENV_PATH" --allow-venv "$VENV_PATH"; then
            CLEANUP_FAILED=1
        fi
        if [[ -n "$SOURCE_HASH" ]]; then
            if ! record source-snapshot-postcheck timeout --signal=TERM --kill-after=1s 8s \
                python3.12 -B "$HELPER" verify-source-snapshot "$SOURCE_SNAPSHOT" "$ROOT"; then
                CLEANUP_FAILED=1
            fi
            if ! record source-snapshot-file-postcheck timeout --signal=TERM --kill-after=1s 5s \
                python3.12 -B -c 'import hashlib,pathlib,sys; p=pathlib.Path(sys.argv[1]); print(hashlib.sha256(p.read_bytes()).hexdigest())' \
                "$SOURCE_SNAPSHOT"; then
                CLEANUP_FAILED=1
            elif [[ "$(< "$LOG_ROOT/source-snapshot-file-postcheck.stdout.log")" != "$SOURCE_HASH" ]]; then
                CLEANUP_FAILED=1
            fi
        fi
        if [[ -f "$PHASE_CONTROL/phase-input-identities.json" ]]; then
            if ! record prior-input-identities-postcheck timeout --signal=TERM --kill-after=1s 12s \
                python3.12 -B "$HELPER" verify-phase-input-identities \
                "$PHASE_CONTROL/phase-input-identities.json" \
                --allow-venv "$CPU_VENV_PATH" --allow-venv "$VENV_PATH"; then
                CLEANUP_FAILED=1
            fi
        fi
        if [[ -n "$(find "$DOCKER_CONFIG_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
            CLEANUP_FAILED=1
        fi
        if (( CLEANUP_FAILED == 0 )); then
            local budget_path="$PHASE_CONTROL/phase-budget-state.json"
            if [[ "$PHASE" == resolve || "$PHASE" == install ]]; then budget_path="$PHASE_CONTROL/setup-budget-state.json"; fi
            if ! record phase-budget-state timeout --signal=TERM --kill-after=1s 8s \
                python3.12 -B "$HELPER" write-phase-budget-state "$PHASE" "$budget_path" \
                "$INVOCATION_STARTED_EPOCH" --active-before "$SETUP_ACTIVE_BEFORE" \
                --external-parent-pause "$EXTERNAL_PARENT_PAUSE" --cleanup-reserve 15 --cleanup-completed; then
                CLEANUP_FAILED=1
            fi
        fi
        if (( CLEANUP_FAILED && status == 0 )); then status=2; fi
        printf '{"phase":"%s","run_id":"%s","attempt_id":"%s","container_create_attempted":%s,"container_start_attempted":%s,"container_start_invocation_returned":%s,"container_started":%s,"cleanup_failed":%s,"status":%s}\n' \
            "$PHASE" "$RUN_ID" "$ATTEMPT_ID" "$CREATE_ATTEMPTED" "$CONTAINER_START_ATTEMPTED" \
            "$CONTAINER_START_INVOCATION_RETURNED" "$CONTAINER_STARTED" "$CLEANUP_FAILED" "$status" \
            > "$PHASE_CONTROL/cleanup-status.json"
        printf '%s\n' "$status" > "$PHASE_CONTROL/wrapper-preinventory-status.log"
        exec 1>&- 2>&-
        printf '%q ' python3.12 -B "$HELPER" inventory "$PHASE_ADMIN/evidence-inventory.json" \
            "$PHASE_OUTPUT" "$PHASE_CONTROL" "$PHASE_ADMIN" \
            --allow-venv "$CPU_VENV_PATH" --allow-venv "$VENV_PATH" \
            > "$PHASE_ADMIN/evidence-inventory.command.log"
        printf '\n' >> "$PHASE_ADMIN/evidence-inventory.command.log"
        if timeout --signal=TERM --kill-after=2s 10s python3.12 -B "$HELPER" inventory \
            "$PHASE_ADMIN/evidence-inventory.json" "$PHASE_OUTPUT" "$PHASE_CONTROL" "$PHASE_ADMIN" \
            --allow-venv "$CPU_VENV_PATH" --allow-venv "$VENV_PATH" \
            > "$PHASE_ADMIN/evidence-inventory.stdout.log" 2> "$PHASE_ADMIN/evidence-inventory.stderr.log"; then
            printf '%s\n' 0 > "$PHASE_ADMIN/evidence-inventory.exit.log"
        else
            local inventory_status=$?
            printf '%s\n' "$inventory_status" > "$PHASE_ADMIN/evidence-inventory.exit.log"
            if (( status == 0 )); then status=2; fi
        fi
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p "$TASK_ROOT/runtime" "$TASK_ROOT/runtime-output" "$TASK_ROOT/runtime-control" \
    "$TASK_ROOT/runtime-admin" "$TASK_ROOT/docker-config"
if [[ "$PHASE" == resolve ]]; then
    for path in "$RUNTIME_ROOT" "$OUTPUT_ROOT" "$CONTROL_ROOT" "$ADMIN_ROOT" "$DOCKER_CONFIG_ROOT"; do
        if [[ -e "$path" || -L "$path" ]]; then
            printf '%s\n' 'Run ID has already been used; evidence paths are never reused.' >&2
            exit 2
        fi
        mkdir -m 700 "$path"
    done
    ROOTS_OWNED=1
else
    for path in "$RUNTIME_ROOT" "$OUTPUT_ROOT" "$CONTROL_ROOT" "$ADMIN_ROOT" "$DOCKER_CONFIG_ROOT"; do
        if [[ ! -d "$path" || -L "$path" ]]; then
            printf '%s\n' 'Earlier Task 8 run roots are missing or unsafe.' >&2
            exit 2
        fi
    done
fi
STARTED_MARKER="$CONTROL_ROOT/$PHASE_KEY.started"
if [[ -n "$ATTEMPT_ID" ]]; then
    ORIGINAL_STARTED_MARKER="$(python3.12 -B -c 'import json,sys; print(json.load(sys.stdin)["original_started_marker"])' <<< "$CORRECTED_RNG_PROJECTION")"
    if [[ ! -f "$ORIGINAL_STARTED_MARKER" || -L "$ORIGINAL_STARTED_MARKER" \
            || ! -f "$ORIGINAL_FAILED_INVENTORY" || -L "$ORIGINAL_FAILED_INVENTORY" ]]; then
        printf '%s\n' 'The immutable original failed RNG attempt and inventory are required.' >&2
        exit 2
    fi
else
    if [[ -e "$STARTED_MARKER" || -L "$STARTED_MARKER" ]]; then
        printf '%s\n' 'This phase was already attempted; automatic retries are disabled.' >&2
        exit 2
    fi
fi
if [[ -e "$STARTED_MARKER" || -L "$STARTED_MARKER" ]]; then
    printf '%s\n' 'This attempt was already attempted; automatic retries are disabled.' >&2
    exit 2
fi
mkdir -m 700 "$PHASE_OUTPUT" "$PHASE_CONTROL" "$PHASE_ADMIN" "$LOG_ROOT"
ROOTS_OWNED=1
if [[ "$(stat -c '%u:%g' "$PHASE_OUTPUT")" != "1000:1000" ]]; then
    printf '%s\n' 'Fresh /work output must be owned by UID:GID 1000:1000.' >&2
    exit 2
fi
if [[ -n "$ATTEMPT_ID" || "$PHASE" == rng-green || "$PHASE" == device-check ]]; then
    record previous-failed-attempt-assertion timeout --signal=TERM --kill-after=1s 8s \
        python3.12 -B "$HELPER" verify-previous-failed-rng-attempt "$TASK_ROOT" "$RUN_ID" \
        --allow-venv "$CPU_VENV_PATH" --allow-venv "$VENV_PATH"
    PREVIOUS_FAILED_INVENTORY_SHA="$(python3.12 -B -c \
        'import json,sys; print(json.load(sys.stdin)["inventory_sha256"])' \
        < "$LOG_ROOT/previous-failed-attempt-assertion.stdout.log")"
fi
: > "$STARTED_MARKER"
if [[ "$PHASE" == resolve ]]; then
    printf '{"run_id":"%s","image_alias":"%s","format":"spire-learn-task8-run-v1"}\n' \
        "$RUN_ID" "$IMAGE_ALIAS" > "$RUNTIME_ROOT/run-identity.json"
fi
if [[ -n "$(find "$DOCKER_CONFIG_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    printf '%s\n' 'Task 8 Docker config directory must remain empty.' >&2
    exit 2
fi
export DOCKER_CONFIG="$DOCKER_CONFIG_ROOT"
exec >> "$PHASE_CONTROL/wrapper.stdout.log" 2>> "$PHASE_CONTROL/wrapper.stderr.log"

SOURCE_HASH="$(sha256sum "$SOURCE_SNAPSHOT" | awk '{print $1}')"
record source-snapshot-precheck timeout --signal=TERM --kill-after=1s 8s \
    python3.12 -B "$HELPER" verify-source-snapshot "$SOURCE_SNAPSHOT" "$ROOT"
SOURCE_VERIFIED_BEFORE=1
record release-assertion timeout --signal=TERM --kill-after=1s 5s \
    python3.12 -B "$HELPER" verify-phase-release "$RELEASE" --phase "$PHASE" --source-hash "$SOURCE_HASH"
record free-space timeout --signal=TERM --kill-after=1s 4s df -B1 --output=avail "$ROOT"
record neural-storage timeout --signal=TERM --kill-after=1s 10s du -sb "$ROOT/artifacts/development/neural-combat"
STORAGE_SCOPE=setup
if [[ "$PHASE" == rng-red || "$PHASE" == rng-green || "$PHASE" == device-check ]]; then STORAGE_SCOPE=workload; fi
STORAGE_ARGS=("$ROOT" "$ROOT/artifacts/development/neural-combat" "$STORAGE_SCOPE"
    "$TASK_ROOT/runtime" "$TASK_ROOT/runtime-output" "$TASK_ROOT/runtime-control"
    "$TASK_ROOT/runtime-admin" "$TASK_ROOT/docker-config"
    --reservation-bytes 1073741824
    --protected-cpu-venv "$CPU_VENV_PATH"
    --allow-venv "$CPU_VENV_PATH" --allow-venv "$VENV_PATH")
if [[ "$PHASE" == resolve ]]; then
    STORAGE_ARGS+=(--reserve-remaining-setup)
elif [[ "$PHASE" == install ]]; then
    STORAGE_ARGS+=(--project-install-wheelhouse "$WHEELHOUSE" --local-wheel "$CPU_WHEEL_PATH")
fi
record storage-preflight timeout --signal=TERM --kill-after=1s 15s \
    python3.12 -B "$HELPER" verify-storage "${STORAGE_ARGS[@]}"
record gpu-wheel-metadata timeout --signal=TERM --kill-after=1s 6s \
    python3.12 -B "$HELPER" verify-candidate-metadata "$TORCH_PREFLIGHT" "$TORCH_METADATA"
record accepted-gym-wheel timeout --signal=TERM --kill-after=1s 6s \
    python3.12 -B "$HELPER" verify-cpu-wheel "$LOCATION" "$CPU_WHEEL_AUDIT"

if [[ "$PHASE" == install || "$PHASE" == rng-red || "$PHASE" == rng-green || "$PHASE" == device-check ]]; then
    if [[ ! -f "$LOCK_PATH" || -L "$LOCK_PATH" || ! -d "$WHEELHOUSE" || -L "$WHEELHOUSE" ]]; then
        printf '%s\n' 'Audited CUDA closure/lock is missing; offline install is blocked.' >&2
        exit 2
    fi
    LOCK_SHA="$(sha256sum "$LOCK_PATH" | awk '{print $1}')"
    record release-lock-assertion timeout --signal=TERM --kill-after=1s 5s \
        python3.12 -B "$HELPER" verify-phase-release "$RELEASE" --phase "$PHASE" \
        --source-hash "$SOURCE_HASH" --require "closure_lock_sha256=$LOCK_SHA"
    if [[ "$PHASE" == install ]]; then
        BUDGET_SHA="$(sha256sum "$BUDGET_STATE" | awk '{print $1}')"
        record release-setup-budget-assertion timeout --signal=TERM --kill-after=1s 5s \
            python3.12 -B "$HELPER" verify-phase-release "$RELEASE" --phase "$PHASE" \
            --source-hash "$SOURCE_HASH" --require "closure_lock_sha256=$LOCK_SHA" \
            --require "setup_budget_state_sha256=$BUDGET_SHA" \
            --require "setup_active_seconds_before=$SETUP_ACTIVE_BEFORE" \
            --require "setup_soft_remaining_seconds=$SETUP_SOFT_REMAINING" \
            --require "setup_hard_remaining_seconds=$SETUP_HARD_REMAINING" \
            --require "external_parent_review_pause_seconds=$EXTERNAL_PARENT_PAUSE"
    fi
    if [[ -n "$ATTEMPT_ID" ]]; then
        record corrected-rng-release-assertion timeout --signal=TERM --kill-after=1s 8s \
            python3.12 -B "$HELPER" verify-corrected-rng-red-release \
            "$RELEASE" "$TASK_ROOT" "$RUN_ID" "$SOURCE_HASH" "$LOCK_SHA" \
            --allow-venv "$CPU_VENV_PATH" --allow-venv "$VENV_PATH"
    fi
fi
if [[ "$PHASE" == rng-red || "$PHASE" == rng-green || "$PHASE" == device-check ]]; then
    if [[ ! -d "$VENV_PATH" || -L "$VENV_PATH" ]]; then
        printf '%s\n' 'Fresh offline CUDA venv is missing.' >&2
        exit 2
    fi
fi
if [[ "$PHASE" == rng-green || "$PHASE" == device-check ]]; then
    if [[ ! -r "$CORRECTED_RED_TRANSCRIPT" || -L "$CORRECTED_RED_TRANSCRIPT" \
            || ! -r "$CORRECTED_RED_INVENTORY" || -L "$CORRECTED_RED_INVENTORY" ]]; then
        printf '%s\n' 'Accepted corrected RED transcript/inventory is missing; later phases are blocked.' >&2
        exit 2
    fi
    RNG_RED_SHA="$(sha256sum "$CORRECTED_RED_TRANSCRIPT" | awk '{print $1}')"
    RNG_RED_INVENTORY_SHA="$(sha256sum "$CORRECTED_RED_INVENTORY" | awk '{print $1}')"
    record corrected-rng-red-evidence-assertion timeout --signal=TERM --kill-after=1s 10s \
        python3.12 -B "$HELPER" verify-corrected-rng-red-evidence \
        "$TASK_ROOT" "$RUN_ID" "$CORRECTED_RED_TRANSCRIPT" "$RNG_RED_SHA" \
        "$CORRECTED_RED_INVENTORY" "$RNG_RED_INVENTORY_SHA" \
        "$ORIGINAL_FAILED_INVENTORY" "$PREVIOUS_FAILED_INVENTORY_SHA" \
        --allow-venv "$CPU_VENV_PATH" --allow-venv "$VENV_PATH"
    if [[ "$PHASE" == device-check ]]; then
        RNG_GREEN_RESULT="$ADMIN_ROOT/rng-green/logs/rng-green-transcript.stdout.log"
        if [[ ! -r "$RNG_GREEN_RESULT" || -L "$RNG_GREEN_RESULT" ]]; then
            printf '%s\n' 'Green RNG evidence is missing; device qualification is blocked.' >&2
            exit 2
        fi
        RNG_GREEN_SHA="$(sha256sum "$RNG_GREEN_RESULT" | awk '{print $1}')"
        record release-red-assertion timeout --signal=TERM --kill-after=1s 5s \
            python3.12 -B "$HELPER" verify-corrected-rng-followup-release \
            "$RELEASE" "$TASK_ROOT" "$PHASE" "$SOURCE_HASH" "$LOCK_SHA" \
            "$LOG_ROOT/corrected-rng-red-evidence-assertion.stdout.log" \
            --rng-green-transcript-sha256 "$RNG_GREEN_SHA"
    else
        record release-red-assertion timeout --signal=TERM --kill-after=1s 5s \
            python3.12 -B "$HELPER" verify-corrected-rng-followup-release \
            "$RELEASE" "$TASK_ROOT" "$PHASE" "$SOURCE_HASH" "$LOCK_SHA" \
            "$LOG_ROOT/corrected-rng-red-evidence-assertion.stdout.log"
    fi
fi

PHASE_INPUT_ARGS=(record-phase-input-identities "$PHASE_CONTROL/phase-input-identities.json"
    --immutable "source_snapshot=$SOURCE_SNAPSHOT"
    --immutable "torch_preflight=$TORCH_PREFLIGHT"
    --immutable "torch_metadata=$TORCH_METADATA"
    --immutable "cpu_runtime_location=$LOCATION"
    --immutable "cpu_wheel_audit=$CPU_WHEEL_AUDIT"
    --immutable "accepted_cpu_wheel=$CPU_WHEEL_PATH"
    --allow-venv "$CPU_VENV_PATH" --allow-venv "$VENV_PATH")
for prior_phase in "${PRIOR_PHASES[@]}"; do
    PHASE_INPUT_ARGS+=(--prior "$prior_phase=$ADMIN_ROOT/$prior_phase/evidence-inventory.json")
done
if [[ -f "$LOCK_PATH" && ! -L "$LOCK_PATH" ]]; then
    PHASE_INPUT_ARGS+=(--immutable "closure_lock=$LOCK_PATH")
fi
if [[ "$PHASE" == rng-red || "$PHASE" == rng-green || "$PHASE" == device-check ]]; then
    PHASE_INPUT_ARGS+=(--venv "$VENV_PATH" --venv-inventory "$ADMIN_ROOT/install/evidence-inventory.json")
fi
record phase-input-identities-precheck timeout --signal=TERM --kill-after=1s 15s \
    python3.12 -B "$HELPER" "${PHASE_INPUT_ARGS[@]}"

if [[ "$PHASE" == resolve ]]; then
    record source-image-inspect timeout --signal=TERM --kill-after=1s 8s docker image inspect "$IMAGE_REFERENCE"
    record source-image-assertion timeout --signal=TERM --kill-after=1s 5s \
        python3.12 -B "$HELPER" verify-image-inspect "$LOG_ROOT/source-image-inspect.stdout.log"
    IMAGE_ID="$(python3.12 -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["image_id"])' \
        "$LOG_ROOT/source-image-assertion.stdout.log")"
    record image-alias-create timeout --signal=TERM --kill-after=2s 10s \
        docker image tag "$IMAGE_REFERENCE" "$IMAGE_ALIAS"
    ALIAS_CREATED=1
else
    if [[ ! -f "$RUNTIME_ROOT/image-identity.json" || -L "$RUNTIME_ROOT/image-identity.json" ]]; then
        printf '%s\n' 'Canonical image identity from the resolution phase is missing.' >&2
        exit 2
    fi
    IMAGE_ID="$(python3.12 -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["image_id"])' \
        "$RUNTIME_ROOT/image-identity.json")"
fi
record image-alias-inspect timeout --signal=TERM --kill-after=1s 5s docker image inspect "$IMAGE_ALIAS"
record image-alias-assertion timeout --signal=TERM --kill-after=1s 5s \
    python3.12 -B "$HELPER" verify-local-image-alias "$LOG_ROOT/image-alias-inspect.stdout.log" \
    --alias "$IMAGE_ALIAS" --expected-image-id "$IMAGE_ID"
if [[ "$PHASE" == resolve ]]; then
    cp --no-clobber "$LOG_ROOT/image-alias-assertion.stdout.log" "$RUNTIME_ROOT/image-identity.json"
fi

GPU_JSON=
if [[ "$PHASE" == rng-red || "$PHASE" == rng-green || "$PHASE" == device-check ]]; then
    record gpu-inventory timeout --signal=TERM --kill-after=1s 8s nvidia-smi \
        --query-gpu=uuid,name,driver_version,memory.total,memory.free --format=csv,noheader,nounits
    record gpu-inventory-assertion timeout --signal=TERM --kill-after=1s 5s \
        python3.12 -B "$HELPER" verify-gpu-inventory "$LOG_ROOT/gpu-inventory.stdout.log"
    GPU_JSON="$LOG_ROOT/gpu-inventory-assertion.stdout.log"
    record gpu-processes-before timeout --signal=TERM --kill-after=1s 8s nvidia-smi \
        --query-compute-apps=pid,used_memory,gpu_uuid --format=csv,noheader,nounits
fi

SOFT_REMAINING="$SETUP_SOFT_REMAINING"
if [[ "$PHASE" == rng-red || "$PHASE" == rng-green || "$PHASE" == device-check ]]; then
    SOFT_REMAINING=60
fi
if (( $(elapsed_seconds_ceil) + 15 >= SOFT_REMAINING )); then
    printf '%s\n' 'Phase soft budget plus finalization reserve is exhausted before container creation.' >&2
    exit 124
fi

python3.12 -B - "$PHASE_CONTROL/container-spec.json" "$IMAGE_ALIAS" "$IMAGE_ID" "$RUN_ID" "$PHASE" \
    "$PHASE_OUTPUT" "$HELPER" "$DEVICE_CHECK" "$LOCATION" "$CPU_WHEEL_AUDIT" "$RESOLVE_OUTPUT" \
    "$INSTALL_OUTPUT" "$ROOT/src" "$ROOT/tests" "$GPU_JSON" "$ATTEMPT_ID" <<'PY'
import json
import os
import sys
from pathlib import Path
from scripts.neural.task8_gpu_setup_helpers import build_phase_spec, verify_cpu_wheel_payload

(spec_path, image, image_id, run_id, phase, work, helper, device_script,
 location_path, wheel_audit_path, resolve_output, install_output, source_root,
 tests_root, gpu_inventory, attempt_id) = sys.argv[1:]
wheel = verify_cpu_wheel_payload(Path(location_path), Path(wheel_audit_path))["wheel_path"]
spec = build_phase_spec(
    phase=phase, run_id=run_id, image=image, expected_image_id=image_id,
    work=Path(work), helper=Path(helper), local_spire_wheel=Path(wheel),
    resolve_output=Path(resolve_output), install_output=Path(install_output),
    source_root=Path(source_root), tests_root=Path(tests_root),
    device_check_script=Path(device_script), gpu_inventory=Path(gpu_inventory) if gpu_inventory else Path("/missing"),
    attempt_id=attempt_id or None,
)
spec["expected_image_id"] = image_id
fd = os.open(spec_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
    json.dump(spec, stream, sort_keys=True, separators=(",", ":"))
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
PY

record docker-argv timeout --signal=TERM --kill-after=1s 8s \
    python3.12 -B "$HELPER" docker-create-argv "$PHASE_CONTROL/container-spec.json"
record docker-argv-nul timeout --signal=TERM --kill-after=1s 5s \
    python3.12 -B -c 'import json,sys; a=json.load(open(sys.argv[1]))["argv"]; sys.stdout.buffer.write(b"\0".join(x.encode() for x in a)+b"\0")' \
    "$LOG_ROOT/docker-argv.stdout.log"
mapfile -d '' -t CREATE_ARGS < "$LOG_ROOT/docker-argv-nul.stdout.log"
record container-name-preflight timeout --signal=TERM --kill-after=1s 4s \
    docker ps -a -q --no-trunc --filter "name=^/$CONTAINER_NAME$"
record container-name-assertion timeout --signal=TERM --kill-after=1s 5s \
    python3.12 -B "$HELPER" assert-container-name-absent \
    "$LOG_ROOT/container-name-preflight.exit.log" "$LOG_ROOT/container-name-preflight.stdout.log"
CREATE_ATTEMPTED=1
if record container-create timeout --signal=TERM --kill-after=2s 12s "${CREATE_ARGS[@]}"; then
    CREATE_STATUS=0
else
    CREATE_STATUS=$?
fi
if (( CREATE_STATUS != 0 )); then exit "$CREATE_STATUS"; fi
record container-create-id-assertion timeout --signal=TERM --kill-after=1s 5s \
    python3.12 -B "$HELPER" assert-container-create \
    "$LOG_ROOT/container-create.exit.log" "$LOG_ROOT/container-create.stdout.log"
CONTAINER_ID="$(< "$LOG_ROOT/container-create.stdout.log")"
record container-inspect timeout --signal=TERM --kill-after=1s 6s docker inspect "$CONTAINER_ID"
record container-assertion timeout --signal=TERM --kill-after=1s 6s \
    python3.12 -B "$HELPER" assert-container-inspect \
    "$LOG_ROOT/container-inspect.stdout.log" "$PHASE_CONTROL/container-spec.json"

if [[ "$PHASE" == rng-red ]]; then
    if record_started_container; then START_STATUS=0; else START_STATUS=$?; fi
    if (( START_STATUS != 1 )); then
        printf '%s\n' 'RNG RED invocation did not return unittest failure status 1.' >&2
        exit 2
    fi
    record rng-red-transcript timeout --signal=TERM --kill-after=1s 6s \
        python3.12 -B "$HELPER" verify-rng-transcript "$LOG_ROOT/container-start.stdout.log" \
        --expected-failure "test_cpu_and_initialized_cuda_rng_states_unchanged_by_build ($RNG_TEST_CLASS.test_cpu_and_initialized_cuda_rng_states_unchanged_by_build)" \
        --expected-failure "test_cpu_build_does_not_call_global_manual_seed_or_initialize_cuda ($RNG_TEST_CLASS.test_cpu_build_does_not_call_global_manual_seed_or_initialize_cuda)"
else
    record_started_container
    if [[ "$PHASE" == rng-green ]]; then
        record rng-green-transcript timeout --signal=TERM --kill-after=1s 6s \
            python3.12 -B "$HELPER" verify-rng-transcript "$LOG_ROOT/container-start.stdout.log"
    elif [[ "$PHASE" == device-check ]]; then
        record device-report-assertion timeout --signal=TERM --kill-after=1s 6s \
            python3.12 -B "$HELPER" verify-device-report "$PHASE_OUTPUT/device-check.json"
        record gpu-processes-after timeout --signal=TERM --kill-after=1s 8s nvidia-smi \
            --query-compute-apps=pid,used_memory,gpu_uuid --format=csv,noheader,nounits
    fi
fi
record container-exit-state timeout --signal=TERM --kill-after=1s 6s \
    docker inspect --format '{{.State.Status}}|{{.State.ExitCode}}' "$CONTAINER_ID"
EXPECTED_EXIT=0
if [[ "$PHASE" == rng-red ]]; then EXPECTED_EXIT=1; fi
if [[ "$(< "$LOG_ROOT/container-exit-state.stdout.log")" != "exited|$EXPECTED_EXIT" ]]; then
    printf '%s\n' 'Container final state or exit code differs from phase expectation.' >&2
    exit 2
fi
CONTAINER_STARTED=1
exit 0
