#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/klottick/repos/spire-learn
TASK_ROOT="$ROOT/artifacts/development/neural-combat/task-6"
LEDGER="$ROOT/artifacts/development/neural-combat/diagnostic-sampled-row-ledger.jsonl"
RUNTIME_ROOT="$ROOT/artifacts/development/neural-combat/runtime/db72a7eb-c789-4cf4-bfa5-42c1ada5e3fc"
VENV="$RUNTIME_ROOT/data/venv-cpu"
LOCK="$RUNTIME_ROOT/data/requirements-cpu.lock"
NATIVE="$VENV/lib/python3.12/site-packages/spire_lab/_native.cpython-312-x86_64-linux-gnu.so"
IMAGE=docker.io/library/python@sha256:57d96f71de145a98e7ceca18ee4ade77bba4103a33a72dfaea026b23625c6bf0

if [[ "${SPIRE_LEARN_TASK6_REVIEWED:-}" != "1" ]]; then
    printf '%s\n' 'Refusing protected Task 6 checkpoint tests: internal source/control review gate is unset.' >&2
    exit 2
fi

RUN_ID="${SPIRE_LEARN_TASK6_RUN_ID:-}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,57}$ ]]; then
    printf '%s\n' 'SPIRE_LEARN_TASK6_RUN_ID must be a unique 1..58 character ID.' >&2
    exit 2
fi
EXPECTED_OUTCOME="${SPIRE_LEARN_TASK6_EXPECT:-}"
if [[ "$EXPECTED_OUTCOME" != "red" && "$EXPECTED_OUTCOME" != "green" ]]; then
    printf '%s\n' 'SPIRE_LEARN_TASK6_EXPECT must be red or green.' >&2
    exit 2
fi

if [[ "${SPIRE_LEARN_TASK6_BOUNDED:-}" != "1" ]]; then
    export SPIRE_LEARN_TASK6_BOUNDED=1
    exec timeout --signal=TERM --kill-after=90s 210s bash "$0"
fi

RUN_ROOT="$TASK_ROOT/runtime/$RUN_ID"
LOG_ROOT="$RUN_ROOT/logs"
WRITABLE_ROOT="$RUN_ROOT/writable"
OUTER_LOG_ROOT="$TASK_ROOT/runtime-control/$RUN_ID"
ADMIN_ROOT="$TASK_ROOT/runtime-admin/$RUN_ID"
RESERVATION_ID="task6-$RUN_ID"
CONTAINER_NAME="spirelearn-task6-$RUN_ID"
CONTAINER_ID=
CONTAINER_CREATE_ATTEMPTED=0
CONTAINER_CREATED=0
CONTAINER_VERIFIED=0
CONTAINER_STARTED=0
CONTAINER_START_STATUS=143
RESERVATION_PENDING=0
SOURCE_PREFLIGHT_DONE=0
CLEANUP_FAILED=0
FINALIZING=0

if [[ ! -d "$ROOT" || ! -d "$VENV" || ! -r "$LOCK" || ! -r "$NATIVE" ]]; then
    printf '%s\n' 'Retained qualified CPU runtime is unavailable.' >&2
    exit 2
fi
if [[ ! -f "$LEDGER" ]]; then
    printf '%s\n' 'Shared diagnostic ledger is missing; refusing to initialize or reset it.' >&2
    exit 2
fi
if [[ -e "$RUN_ROOT" || -e "$OUTER_LOG_ROOT" || -e "$ADMIN_ROOT" ]]; then
    printf '%s\n' 'Task 6 run or control-log path already exists; IDs are never reused.' >&2
    exit 2
fi
mkdir -p "$TASK_ROOT/runtime" "$TASK_ROOT/runtime-control"
mkdir -p "$TASK_ROOT/runtime-admin"
mkdir -m 700 "$RUN_ROOT" "$OUTER_LOG_ROOT" "$ADMIN_ROOT"
mkdir -m 700 "$LOG_ROOT" "$WRITABLE_ROOT"
mkdir -m 700 "$WRITABLE_ROOT/home" "$WRITABLE_ROOT/tmp"

record() {
    local name=$1
    shift
    local command_status
    if ! printf '%q ' "$@" > "$LOG_ROOT/$name.command.log"; then return 125; fi
    if ! printf '\n' >> "$LOG_ROOT/$name.command.log"; then return 125; fi
    if "$@" > "$LOG_ROOT/$name.stdout.log" 2> "$LOG_ROOT/$name.stderr.log"; then
        command_status=0
    else
        command_status=$?
    fi
    if ! printf '%s\n' "$command_status" > "$LOG_ROOT/$name.exit.log"; then return 125; fi
    return "$command_status"
}

record_outer() {
    local name=$1
    shift
    local command_status
    if ! printf '%q ' "$@" > "$OUTER_LOG_ROOT/$name.command.log"; then return 125; fi
    if ! printf '\n' >> "$OUTER_LOG_ROOT/$name.command.log"; then return 125; fi
    if "$@" > "$OUTER_LOG_ROOT/$name.stdout.log" 2> "$OUTER_LOG_ROOT/$name.stderr.log"; then
        command_status=0
    else
        command_status=$?
    fi
    if ! printf '%s\n' "$command_status" > "$OUTER_LOG_ROOT/$name.exit.log"; then return 125; fi
    return "$command_status"
}

write_final_inventory() {
    local admin_root=$1
    local prior_status=$2
    shift 2
    if ! printf '%q ' "$@" > "$admin_root/evidence-inventory.command.log"; then
        FINAL_INVENTORY_STATUS=125
        FINAL_WRAPPER_STATUS=2
        return 2
    fi
    if ! printf '\n' >> "$admin_root/evidence-inventory.command.log"; then
        FINAL_INVENTORY_STATUS=125
        FINAL_WRAPPER_STATUS=2
        return 2
    fi
    if "$@" > "$admin_root/evidence-inventory.stdout.log" \
        2> "$admin_root/evidence-inventory.stderr.log"; then
        FINAL_INVENTORY_STATUS=0
        FINAL_WRAPPER_STATUS=$prior_status
    else
        FINAL_INVENTORY_STATUS=$?
        FINAL_WRAPPER_STATUS=2
    fi
    if ! printf 'inventory_command_exit=%s\nwrapper_exit=%s\n' \
        "$FINAL_INVENTORY_STATUS" "$FINAL_WRAPPER_STATUS" \
        > "$admin_root/evidence-inventory.exit.log"; then
        FINAL_WRAPPER_STATUS=2
    fi
    return "$FINAL_WRAPPER_STATUS"
}

finalize() {
    local saved_status=$?
    if (( FINALIZING )); then
        exit "$saved_status"
    fi
    FINALIZING=1
    trap '' INT TERM
    trap - EXIT

    if (( CONTAINER_CREATE_ATTEMPTED )); then
        if (( CONTAINER_VERIFIED )); then
            if ! record container-remove timeout --signal=TERM --kill-after=1s 5s \
                docker rm -f "$CONTAINER_ID"; then
                CLEANUP_FAILED=1
            fi
        else
            if ! record container-create-recovery-lookup timeout --signal=TERM --kill-after=1s 4s \
                docker ps -a -q --no-trunc --filter "name=^/$CONTAINER_NAME$"; then
                CLEANUP_FAILED=1
            elif ! record container-create-recovery-resolve timeout --signal=TERM --kill-after=1s 3s \
                python3.12 -B "$ROOT/scripts/neural/task6_checkpoint_run_helpers.py" \
                resolve-container-id "$LOG_ROOT/container-create-recovery-lookup.stdout.log"; then
                CLEANUP_FAILED=1
            else
                local recovered_id
                recovered_id="$(< "$LOG_ROOT/container-create-recovery-resolve.stdout.log")"
                if [[ -n "$recovered_id" ]]; then
                    if record container-create-recovery-inspect timeout --signal=TERM --kill-after=1s 4s \
                        docker inspect "$recovered_id" \
                    && record container-create-recovery-assertion timeout --signal=TERM --kill-after=1s 4s \
                        python3.12 -B "$ROOT/scripts/neural/task6_checkpoint_run_helpers.py" inspect \
                        "$LOG_ROOT/container-create-recovery-inspect.stdout.log" \
                        "$LOG_ROOT/container-create-recovery-assertion.json" \
                        --run-id "$RUN_ID" --source "$ROOT/src" --tests "$ROOT/tests" \
                        --venv "$VENV" --lock "$LOCK" --writable "$WRITABLE_ROOT" \
                        --command-file "$LOG_ROOT/container-command.txt" \
                        --container-id "$recovered_id"; then
                        CONTAINER_ID=$recovered_id
                        CONTAINER_CREATED=1
                        CONTAINER_VERIFIED=1
                        if ! record container-remove timeout --signal=TERM --kill-after=1s 5s \
                            docker rm -f "$CONTAINER_ID"; then
                            CLEANUP_FAILED=1
                        fi
                    else
                        # preserve any uncertain or unowned object for manual review.
                        CLEANUP_FAILED=1
                    fi
                else
                    printf '%s\n' 'no_container_with_exact_task6_name' \
                        > "$LOG_ROOT/container-create-recovery-result.log"
                fi
            fi
        fi
    fi
    if (( CONTAINER_CREATE_ATTEMPTED )); then
        if ! record container-final-inventory timeout --signal=TERM --kill-after=1s 4s \
            docker ps -a -q --no-trunc --filter "name=^/$CONTAINER_NAME$"; then
            CLEANUP_FAILED=1
        elif [[ -s "$LOG_ROOT/container-final-inventory.stdout.log" ]]; then
            CLEANUP_FAILED=1
        fi
    fi

    if (( SOURCE_PREFLIGHT_DONE )); then
        if ! record source-postcheck timeout --signal=TERM --kill-after=1s 6s \
            python3.12 -B "$ROOT/scripts/neural/task6_checkpoint_run_helpers.py" \
            source-after "$ROOT" "$RUN_ROOT"; then
            CLEANUP_FAILED=1
        fi
    fi

    if (( RESERVATION_PENDING )); then
        local sample_status=not_started
        if (( CONTAINER_STARTED )); then
            if (( CONTAINER_START_STATUS == 0 )); then sample_status=succeeded; else sample_status=failed; fi
        fi
        if ! record sample-row-finish timeout --signal=TERM --kill-after=1s 4s \
            python3.12 -B -m tests.neural._sample_guard --ledger "$LEDGER" finish \
            --reservation-id "$RESERVATION_ID" --status "$sample_status" \
            --evidence "$WRITABLE_ROOT/tensor-sampling.json"; then
            CLEANUP_FAILED=1
        fi
    fi

    local cleanup_args=()
    if (( CONTAINER_CREATE_ATTEMPTED )); then cleanup_args+=(--container-create-attempted); fi
    if (( CONTAINER_CREATED )); then cleanup_args+=(--container-created); fi
    if (( CONTAINER_STARTED )); then cleanup_args+=(--container-started); fi
    if (( CLEANUP_FAILED )); then cleanup_args+=(--cleanup-failed); fi
    if ! record_outer cleanup-status timeout --signal=TERM --kill-after=1s 3s \
        python3.12 -B "$ROOT/scripts/neural/task6_checkpoint_run_helpers.py" \
        cleanup-status "$RUN_ROOT" "${cleanup_args[@]}"; then
        CLEANUP_FAILED=1
    fi
    if (( CLEANUP_FAILED )) && (( saved_status == 0 )); then saved_status=2; fi
    local summary_args=(--expected-outcome "$EXPECTED_OUTCOME" --start-status "$CONTAINER_START_STATUS")
    if (( CONTAINER_CREATED )); then summary_args+=(--container-created); fi
    if (( CONTAINER_STARTED )); then summary_args+=(--container-started); fi
    if (( CLEANUP_FAILED )); then summary_args+=(--cleanup-failed); fi
    if ! record_outer run-summary timeout --signal=TERM --kill-after=1s 3s \
        python3.12 -B "$ROOT/scripts/neural/task6_checkpoint_run_helpers.py" \
        run-summary "$RUN_ROOT" "${summary_args[@]}"; then
        saved_status=2
    fi
    printf 'preinventory_wrapper_status=%s\n' "$saved_status" \
        > "$OUTER_LOG_ROOT/wrapper-preinventory-status.log"

    if write_final_inventory "$ADMIN_ROOT" "$saved_status" \
        timeout --signal=TERM --kill-after=2s 10s \
        python3.12 -B "$ROOT/scripts/neural/task6_checkpoint_run_helpers.py" \
        inventory "$RUN_ROOT" "$OUTER_LOG_ROOT" "$ADMIN_ROOT/evidence-inventory.json" \
        --admin-root "$ADMIN_ROOT"; then
        saved_status=$FINAL_WRAPPER_STATUS
    else
        saved_status=$FINAL_WRAPPER_STATUS
    fi
    exit "$saved_status"
}

trap finalize EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
exec >> "$OUTER_LOG_ROOT/wrapper.stdout.log" 2>> "$OUTER_LOG_ROOT/wrapper.stderr.log"

if [[ ! -L "$VENV/bin/python" || "$(readlink -- "$VENV/bin/python")" != "/usr/local/bin/python" ]]; then
    printf '%s\n' 'Qualified CPU venv Python link does not match the accepted runtime.' >&2
    exit 2
fi
if ! record source-preflight timeout --signal=TERM --kill-after=4s 8s \
    python3.12 -B "$ROOT/scripts/neural/task6_checkpoint_run_helpers.py" \
    source-before "$ROOT" "$RUN_ROOT"; then
    printf '%s\n' 'Task 6 source, control, free-space, CPU runtime, or Gym provenance preflight failed.' >&2
    exit 2
fi
SOURCE_PREFLIGHT_DONE=1

if ! record local-image-inspect timeout --signal=TERM --kill-after=4s 8s \
    docker image inspect "$IMAGE"; then
    printf '%s\n' 'The exact pinned Task 6 image is not available locally; no pull will be attempted.' >&2
    exit 2
fi
if ! record local-image-assertion timeout --signal=TERM --kill-after=2s 4s \
    python3.12 -B "$ROOT/scripts/neural/task6_checkpoint_run_helpers.py" assert-local-image \
    "$LOG_ROOT/local-image-inspect.stdout.log" "$LOG_ROOT/local-image-identity.json"; then
    printf '%s\n' 'The local image identity does not match the pinned linux/amd64 image.' >&2
    exit 2
fi

if ! record sample-row-reserve timeout --signal=TERM --kill-after=4s 8s \
    python3.12 -B -m tests.neural._sample_guard --ledger "$LEDGER" reserve \
    --task task6 --reservation-id "$RESERVATION_ID" --rows 64; then
    printf '%s\n' 'Shared Task 4/6 diagnostic-row reservation failed; no test container was created.' >&2
    exit 2
fi
RESERVATION_PENDING=1

if ! record container-name-precheck timeout --signal=TERM --kill-after=4s 8s \
    docker ps -a -q --no-trunc --filter "name=^/$CONTAINER_NAME$"; then
    printf '%s\n' 'Could not verify the unique Task 6 container name is unused.' >&2
    exit 2
fi
if [[ -s "$LOG_ROOT/container-name-precheck.stdout.log" ]]; then
    printf '%s\n' 'A container already uses the Task 6 run name; it will be left untouched.' >&2
    exit 2
fi

CONTAINER_COMMAND=$(cat <<'COMMAND'
set -eu
test "$(pwd)" = /learner
test -r /work/venv-cpu/bin/python
test -r /work/requirements-cpu.lock
test -d /learner/src
test -d /learner/tests
test ! -w /learner/src
test ! -w /learner/tests
test ! -w /work/venv-cpu
test ! -w /work/requirements-cpu.lock
mkdir -p /run/home /run/tmp
/work/venv-cpu/bin/python -B -m pip check
/work/venv-cpu/bin/python -B -c 'import hashlib,importlib.metadata as m,json,pathlib; p=pathlib.Path("/work/venv-cpu/lib/python3.12/site-packages/spire_lab/_native.cpython-312-x86_64-linux-gnu.so"); lock=pathlib.Path("/work/requirements-cpu.lock"); h=lambda x:hashlib.sha256(x.read_bytes()).hexdigest(); expected={"spire-lab":"0.1.0","gymnasium":"1.4.0","numpy":"2.5.4","torch":"2.14.1+cpu"}; actual={k:m.version(k) for k in expected}; identity={"packages":actual,"lock_sha256":h(lock),"native_sha256":h(p)}; print("TASK6_RUNTIME_IDENTITY="+json.dumps(identity,sort_keys=True)); assert actual==expected; assert h(lock)=="51714575fe45381716e06d482ad0f6b064c60ffd0e64ec5bf2753158cf31f2d0"; assert h(p)=="071069cea66fbf0bc7277002c6b8a21f138194cf692326595109a6741a5178e9"'
exec timeout --signal=TERM --kill-after=15s 120s /work/venv-cpu/bin/python -B -m unittest tests.neural.test_checkpoint -v
COMMAND
)
printf '%s' "$CONTAINER_COMMAND" > "$LOG_ROOT/container-command.txt"
printf '%q ' timeout --signal=TERM --kill-after=90s 210s bash "$0" > "$LOG_ROOT/wrapper-timeout.command.log"
printf '\n' >> "$LOG_ROOT/wrapper-timeout.command.log"

CREATE_ARGS=(
    --pull=never
    --name "$CONTAINER_NAME"
    --user 1000:1000
    --read-only
    --cap-drop ALL
    --security-opt no-new-privileges
    --cpus 4
    --memory 8g
    --pids-limit 512
    --network none
    --workdir /learner
    --tmpfs /tmp:rw,noexec,nosuid,size=64m
    --mount "type=bind,src=$ROOT/src,dst=/learner/src,readonly"
    --mount "type=bind,src=$ROOT/tests,dst=/learner/tests,readonly"
    --mount "type=bind,src=$VENV,dst=/work/venv-cpu,readonly"
    --mount "type=bind,src=$LOCK,dst=/work/requirements-cpu.lock,readonly"
    --mount "type=bind,src=$WRITABLE_ROOT,dst=/run"
    --env PYTHONPATH=/learner/src
    --env PYTHONDONTWRITEBYTECODE=1
    --env PIP_NO_INDEX=1
    --env PIP_NO_CACHE_DIR=1
    --env PIP_DISABLE_PIP_VERSION_CHECK=1
    --env HOME=/run/home
    --env TMPDIR=/run/tmp
    --env OMP_NUM_THREADS=4
    --env MKL_NUM_THREADS=4
    --env OPENBLAS_NUM_THREADS=1
    --env NUMEXPR_NUM_THREADS=1
    --env SPIRE_LEARN_RUN_NEURAL_TENSOR=1
    --env SPIRE_LEARN_TENSOR_TASK=task6
    --env "SPIRE_LEARN_TENSOR_RESERVATION_ID=$RESERVATION_ID"
    --env SPIRE_LEARN_TENSOR_MAX_SAMPLED_ROWS=64
    --env SPIRE_LEARN_TENSOR_EVIDENCE_PATH=/run/tensor-sampling.json
    --label spire-learn.phase=task6-checkpoint-cpu-tests
    --label "spire-learn.owner=$RUN_ID"
    --entrypoint /bin/sh
    "$IMAGE" -ec "$CONTAINER_COMMAND"
)

CONTAINER_CREATE_ATTEMPTED=1
if ! record container-create timeout --signal=TERM --kill-after=10s 30s docker create "${CREATE_ARGS[@]}"; then
    printf '%s\n' 'Docker create failed; no retry will be attempted.' >&2
    exit 2
fi
CONTAINER_ID="$(< "$LOG_ROOT/container-create.stdout.log")"
if [[ ! "$CONTAINER_ID" =~ ^[0-9a-f]{12,64}$ ]]; then
    printf '%s\n' 'Docker create returned an invalid container ID.' >&2
    exit 2
fi
CONTAINER_CREATED=1
if ! record container-inspect timeout --signal=TERM --kill-after=4s 10s docker inspect "$CONTAINER_ID"; then
    printf '%s\n' 'Could not inspect the created container before start.' >&2
    exit 2
fi
if ! record inspect-assertion timeout --signal=TERM --kill-after=4s 8s \
    python3.12 -B "$ROOT/scripts/neural/task6_checkpoint_run_helpers.py" inspect \
    "$LOG_ROOT/container-inspect.stdout.log" "$LOG_ROOT/container-inspect-assertion.json" \
    --run-id "$RUN_ID" --source "$ROOT/src" --tests "$ROOT/tests" --venv "$VENV" \
    --lock "$LOCK" --writable "$WRITABLE_ROOT" --command-file "$LOG_ROOT/container-command.txt" \
    --container-id "$CONTAINER_ID"; then
    printf '%s\n' 'Created container failed its exact mount/privilege/command inspection.' >&2
    exit 2
fi
CONTAINER_VERIFIED=1

CONTAINER_STARTED=1
if record container-start timeout --signal=TERM --kill-after=20s 150s docker start -a "$CONTAINER_ID"; then
    CONTAINER_START_STATUS=0
else
    CONTAINER_START_STATUS=$?
fi
if (( CONTAINER_START_STATUS != 0 )); then
    if [[ "$EXPECTED_OUTCOME" == "red" ]]; then
        exit 0
    fi
    printf '%s\n' 'Task 6 checkpoint test process failed or exceeded its bounded runtime.' >&2
    exit "$CONTAINER_START_STATUS"
fi
exit 0
