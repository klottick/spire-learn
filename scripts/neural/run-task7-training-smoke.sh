#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
TASK_ROOT="$ROOT/artifacts/development/neural-combat/task-7"
RUNTIME_ROOT="$ROOT/artifacts/development/neural-combat/runtime/db72a7eb-c789-4cf4-bfa5-42c1ada5e3fc"
LOCATION="$ROOT/artifacts/development/neural-combat/cpu-runtime-location.json"
VENV="$RUNTIME_ROOT/data/venv-cpu"
LOCK="$RUNTIME_ROOT/data/requirements-cpu.lock"
NATIVE="$VENV/lib/python3.12/site-packages/spire_lab/_native.cpython-312-x86_64-linux-gnu.so"
IMAGE=docker.io/library/python@sha256:57d96f71de145a98e7ceca18ee4ade77bba4103a33a72dfaea026b23625c6bf0
HELPER="$ROOT/scripts/neural/task7_training_run_helpers.py"
RELEASE="$TASK_ROOT/parent-smoke-release.json"

if [[ "${SPIRE_LEARN_TASK7_REVIEWED:-}" != "1" ]]; then
    printf '%s\n' 'Refusing Task 7 CPU smoke: parent source/control release gate is unset.' >&2
    exit 2
fi

RUN_ID="${SPIRE_LEARN_TASK7_RUN_ID:-}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,39}$ ]]; then
    printf '%s\n' 'SPIRE_LEARN_TASK7_RUN_ID must be a unique 1..40 character ID.' >&2
    exit 2
fi
if [[ ! -f "$RELEASE" ]]; then
    printf '%s\n' 'Parent release record is missing; do not run the held smoke.' >&2
    exit 2
fi

if [[ "${SPIRE_LEARN_TASK7_BOUNDED:-}" != "1" ]]; then
    export SPIRE_LEARN_TASK7_BOUNDED=1
    exec timeout --signal=TERM --kill-after=90s 210s bash "$0"
fi

RUN_ROOT="$TASK_ROOT/runtime/$RUN_ID"
LOG_ROOT="$RUN_ROOT/logs"
WRITABLE_ROOT="$RUN_ROOT/writable"
OUTPUT_CHILD="$WRITABLE_ROOT/artifacts"
CONTROL_ROOT="$TASK_ROOT/runtime-control/$RUN_ID"
ADMIN_ROOT="$TASK_ROOT/runtime-admin/$RUN_ID"
CONTAINER_NAME="spirelearn-task7-$RUN_ID"
CONTAINER_ID=
CREATE_ATTEMPTED=0
CONTAINER_VERIFIED=0
CONTAINER_STARTED=0
CONTAINER_REMOVED=0
SOURCE_SNAPSHOT_READY=0
CLEANUP_FAILED=0
FINALIZING=0
RUN_ROOT_OWNED=0
CONTROL_ROOT_OWNED=0
ADMIN_ROOT_OWNED=0

if [[ -e "$RUN_ROOT" || -e "$CONTROL_ROOT" || -e "$ADMIN_ROOT" ]]; then
    printf '%s\n' 'Task 7 run/control path already exists; run IDs are never reused.' >&2
    exit 2
fi
if [[ ! -d "$ROOT" || ! -d "$VENV" || ! -r "$LOCK" || ! -r "$NATIVE" || ! -r "$RELEASE" ]]; then
    printf '%s\n' 'Retained CPU runtime or parent release record is unavailable.' >&2
    exit 2
fi
record() {
    local name=$1
    shift
    local result
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

record_control() {
    local name=$1
    shift
    local result
    printf '%q ' "$@" > "$CONTROL_ROOT/$name.command.log"
    printf '\n' >> "$CONTROL_ROOT/$name.command.log"
    if "$@" > "$CONTROL_ROOT/$name.stdout.log" 2> "$CONTROL_ROOT/$name.stderr.log"; then
        result=0
    else
        result=$?
    fi
    printf '%s\n' "$result" > "$CONTROL_ROOT/$name.exit.log"
    return "$result"
}

cleanup() {
    local saved_status=$?
    if (( FINALIZING )); then exit "$saved_status"; fi
    FINALIZING=1
    trap '' INT TERM
    trap - EXIT
    set +e

    if (( CREATE_ATTEMPTED && ! CONTAINER_VERIFIED )); then
        if record container-create-recovery-lookup timeout --signal=TERM --kill-after=1s 4s \
            docker ps -a -q --no-trunc --filter "name=^/$CONTAINER_NAME$"; then
            if record container-create-recovery-resolve timeout --signal=TERM --kill-after=1s 3s \
                python3.12 -B "$HELPER" resolve-container-id "$LOG_ROOT/container-create-recovery-lookup.stdout.log"; then
                local recovered
                recovered="$(< "$LOG_ROOT/container-create-recovery-resolve.stdout.log")"
                if [[ -n "$recovered" ]]; then
                    if record container-create-recovery-inspect timeout --signal=TERM --kill-after=1s 4s \
                        docker inspect "$recovered" \
                    && record container-create-recovery-assertion timeout --signal=TERM --kill-after=1s 4s \
                        python3.12 -B "$HELPER" assert-inspect \
                        "$LOG_ROOT/container-create-recovery-inspect.stdout.log" \
                        "$LOG_ROOT/container-create-recovery-assertion.json" \
                        --run-id "$RUN_ID" --container-id "$recovered" --root "$ROOT" \
                        --image-identity "$RUN_ROOT/image-identity.json" \
                        --venv "$VENV" --lock "$LOCK" --output-child "$OUTPUT_CHILD"; then
                        CONTAINER_ID=$recovered
                        CONTAINER_VERIFIED=1
                    else
                        CLEANUP_FAILED=1
                    fi
                fi
            else
                CLEANUP_FAILED=1
            fi
        else
            CLEANUP_FAILED=1
        fi
    fi

    if (( CONTAINER_VERIFIED )); then
        if record container-remove timeout --signal=TERM --kill-after=1s 6s docker rm -f "$CONTAINER_ID"; then
            CONTAINER_REMOVED=1
        else
            CLEANUP_FAILED=1
        fi
    fi
    if (( CREATE_ATTEMPTED )); then
        if record container-final-list timeout --signal=TERM --kill-after=1s 4s \
            docker ps -a -q --no-trunc --filter "name=^/$CONTAINER_NAME$"; then
            if [[ -s "$LOG_ROOT/container-final-list.stdout.log" ]]; then CLEANUP_FAILED=1; fi
        else
            CLEANUP_FAILED=1
        fi
    fi

    if (( SOURCE_SNAPSHOT_READY )); then
        if record_control source-postcheck timeout --signal=TERM --kill-after=1s 8s \
            python3.12 -B "$HELPER" source-snapshot "$ROOT" "$CONTROL_ROOT/source-after.json" \
        && record_control source-diff-check timeout --signal=TERM --kill-after=1s 4s \
            python3.12 -B "$HELPER" compare-snapshots \
            "$RUN_ROOT/source-before.json" "$CONTROL_ROOT/source-after.json"; then
            :
        else
            CLEANUP_FAILED=1
        fi
    fi

    if (( RUN_ROOT_OWNED && CONTROL_ROOT_OWNED && ADMIN_ROOT_OWNED )); then
        printf '{"container_create_attempted":%s,"container_verified":%s,"container_started":%s,"container_removed":%s,"cleanup_failed":%s}\n' \
            "$CREATE_ATTEMPTED" "$CONTAINER_VERIFIED" "$CONTAINER_STARTED" \
            "$CONTAINER_REMOVED" "$CLEANUP_FAILED" > "$CONTROL_ROOT/cleanup-status.json"
        if (( CLEANUP_FAILED )) && (( saved_status == 0 )); then saved_status=2; fi
        printf 'preinventory_wrapper_status=%s\n' "$saved_status" \
            > "$CONTROL_ROOT/wrapper-preinventory-status.log"

        exec 1>&- 2>&-
        printf '%q ' python3.12 -B "$HELPER" inventory "$RUN_ROOT" "$CONTROL_ROOT" "$ADMIN_ROOT" \
            "$ADMIN_ROOT/evidence-inventory.json" > "$ADMIN_ROOT/evidence-inventory.command.log"
        printf '\n' >> "$ADMIN_ROOT/evidence-inventory.command.log"
        if timeout --signal=TERM --kill-after=2s 10s python3.12 -B "$HELPER" inventory \
            "$RUN_ROOT" "$CONTROL_ROOT" "$ADMIN_ROOT" "$ADMIN_ROOT/evidence-inventory.json" \
            > "$ADMIN_ROOT/evidence-inventory.stdout.log" \
            2> "$ADMIN_ROOT/evidence-inventory.stderr.log"; then
            printf '%s\n' 0 > "$ADMIN_ROOT/evidence-inventory.exit.log"
        else
            local inventory_status=$?
            printf '%s\n' "$inventory_status" > "$ADMIN_ROOT/evidence-inventory.exit.log"
            if (( saved_status == 0 )); then saved_status=2; fi
        fi
    fi
    exit "$saved_status"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
mkdir -p "$TASK_ROOT/runtime" "$TASK_ROOT/runtime-control" "$TASK_ROOT/runtime-admin"
mkdir -m 700 "$RUN_ROOT"
RUN_ROOT_OWNED=1
mkdir -m 700 "$CONTROL_ROOT"
CONTROL_ROOT_OWNED=1
mkdir -m 700 "$ADMIN_ROOT"
ADMIN_ROOT_OWNED=1
mkdir -m 700 "$LOG_ROOT" "$WRITABLE_ROOT" "$OUTPUT_CHILD"
record output-scaffold timeout --signal=TERM --kill-after=1s 4s \
    python3.12 -B "$HELPER" prepare-output-tree "$OUTPUT_CHILD"
if [[ "$(stat -c '%u:%g' "$OUTPUT_CHILD")" != "1000:1000" ]]; then
    printf '%s\n' 'Fresh output child must be owned by UID:GID 1000:1000.' >&2
    exit 2
fi
exec >> "$CONTROL_ROOT/wrapper.stdout.log" 2>> "$CONTROL_ROOT/wrapper.stderr.log"

record runtime-identity timeout --signal=TERM --kill-after=1s 6s \
    python3.12 -B "$HELPER" runtime-identity "$LOCATION" "$LOCK" "$NATIVE" \
    "$RUN_ROOT/runtime-identity.json"
record free-space timeout --signal=TERM --kill-after=1s 4s \
    python3.12 -B "$HELPER" verify-free-space "$ROOT" --minimum-gib 30
record source-preflight timeout --signal=TERM --kill-after=1s 8s \
    python3.12 -B "$HELPER" source-snapshot "$ROOT" "$RUN_ROOT/source-before.json"
SOURCE_SNAPSHOT_READY=1
record release-assertion timeout --signal=TERM --kill-after=1s 4s \
    python3.12 -B "$HELPER" verify-release "$RELEASE" "$RUN_ROOT/source-before.json"
record image-inspect timeout --signal=TERM --kill-after=1s 5s \
    docker image inspect --format '{{.Id}}|{{.Os}}|{{.Architecture}}|{{json .RepoDigests}}' "$IMAGE"
record image-assertion timeout --signal=TERM --kill-after=1s 4s \
    python3.12 -B "$HELPER" verify-image "$LOG_ROOT/image-inspect.stdout.log" \
    "$RUN_ROOT/image-identity.json"

mapfile -d '' -t CONTAINER_COMMAND_ARGS < <(python3.12 -B "$HELPER" container-args)
if (( ${#CONTAINER_COMMAND_ARGS[@]} != 4 )); then
    printf '%s\n' 'Task 7 container preflight command is incomplete.' >&2
    exit 2
fi

CREATE_ATTEMPTED=1
record container-create timeout --signal=TERM --kill-after=1s 10s \
    docker create --pull=never --name "$CONTAINER_NAME" \
    --label spire-learn.task=task7-training-smoke \
    --label "spire-learn.run-id=$RUN_ID" \
    --platform linux/amd64 --network none --read-only --user 1000:1000 --cpus 4 --memory 8g \
    --pids-limit 512 --cap-drop ALL --security-opt no-new-privileges \
    --tmpfs /tmp:rw,noexec,nosuid,size=64m --workdir /learner \
    --mount "type=bind,src=$ROOT/src,dst=/learner/src,readonly" \
    --mount "type=bind,src=$ROOT/tests,dst=/learner/tests,readonly" \
    --mount "type=bind,src=$ROOT/configs,dst=/learner/configs,readonly" \
    --mount "type=bind,src=$HELPER,dst=/learner/scripts/neural/task7_training_run_helpers.py,readonly" \
    --mount "type=bind,src=$OUTPUT_CHILD,dst=/learner/artifacts" \
    --mount "type=bind,src=$VENV,dst=/work/venv-cpu,readonly" \
    --mount "type=bind,src=$LOCK,dst=/work/requirements-cpu.lock,readonly" \
    --env SPIRE_LEARN_RUN_NEURAL_NATIVE=1 \
    --env SPIRE_LEARN_RUN_NEURAL_TENSOR=1 \
    --env PYTHONPATH=/learner/src \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --env PYTHONNOUSERSITE=1 \
    --env OMP_NUM_THREADS=4 \
    --env MKL_NUM_THREADS=4 \
    --env OPENBLAS_NUM_THREADS=4 \
    --env NUMEXPR_NUM_THREADS=4 \
    --env VECLIB_MAXIMUM_THREADS=4 \
    --env BLIS_NUM_THREADS=4 \
    --env HOME=/tmp \
    --entrypoint /bin/sh "$IMAGE" "${CONTAINER_COMMAND_ARGS[@]}"
CONTAINER_ID="$(< "$LOG_ROOT/container-create.stdout.log")"
if [[ ! "$CONTAINER_ID" =~ ^[0-9a-f]{64}$ ]]; then
    printf '%s\n' 'docker create did not return one full container ID.' >&2
    exit 2
fi
record container-inspect timeout --signal=TERM --kill-after=1s 5s docker inspect "$CONTAINER_ID"
record container-assertion timeout --signal=TERM --kill-after=1s 5s \
    python3.12 -B "$HELPER" assert-inspect "$LOG_ROOT/container-inspect.stdout.log" \
    "$RUN_ROOT/container-inspection.json" --run-id "$RUN_ID" --container-id "$CONTAINER_ID" \
    --image-identity "$RUN_ROOT/image-identity.json" \
    --root "$ROOT" --venv "$VENV" --lock "$LOCK" --output-child "$OUTPUT_CHILD"
CONTAINER_VERIFIED=1

CONTAINER_STARTED=1
record container-start timeout --signal=TERM --kill-after=2s 170s docker start -a "$CONTAINER_ID"
record container-final-inspect timeout --signal=TERM --kill-after=1s 5s \
    docker inspect --format '{{.State.Status}} {{.State.ExitCode}}' "$CONTAINER_ID"
record container-exit-assertion timeout --signal=TERM --kill-after=1s 4s \
    python3.12 -B "$HELPER" verify-container-status "$LOG_ROOT/container-final-inspect.stdout.log"
record transcript-assertion timeout --signal=TERM --kill-after=1s 4s \
    python3.12 -B "$HELPER" transcript "$LOG_ROOT/container-start.stdout.log" \
    "$LOG_ROOT/container-start.stderr.log"
record artifact-cap-assertion timeout --signal=TERM --kill-after=1s 6s \
    python3.12 -B "$HELPER" assert-smoke-artifacts "$OUTPUT_CHILD" \
    "$RUN_ROOT/smoke-artifact-summary.json"
