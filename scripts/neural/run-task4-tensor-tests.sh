#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/klottick/repos/spire-learn
TASK_ROOT="$ROOT/artifacts/development/neural-combat/task-4"
LEDGER="$ROOT/artifacts/development/neural-combat/diagnostic-sampled-row-ledger.jsonl"
RUNTIME_ROOT="$ROOT/artifacts/development/neural-combat/runtime/db72a7eb-c789-4cf4-bfa5-42c1ada5e3fc"
VENV="$RUNTIME_ROOT/data/venv-cpu"
LOCK="$RUNTIME_ROOT/data/requirements-cpu.lock"
NATIVE="$VENV/lib/python3.12/site-packages/spire_lab/_native.cpython-312-x86_64-linux-gnu.so"
RECONCILIATION="$RUNTIME_ROOT/cpu-source-reconciliation.json"
IMAGE=docker.io/library/python@sha256:57d96f71de145a98e7ceca18ee4ade77bba4103a33a72dfaea026b23625c6bf0
EXPECTED_VENV_PYTHON_TARGET=/usr/local/bin/python

if [[ "${SPIRE_LEARN_TASK4_REVIEWED:-}" != "1" ]]; then
    printf '%s\n' 'Refusing protected tensor tests: parent first-delivery review gate is unset.' >&2
    exit 2
fi

RUN_ID="${SPIRE_LEARN_TASK4_RUN_ID:-}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,57}$ ]]; then
    printf '%s\n' 'SPIRE_LEARN_TASK4_RUN_ID must be a unique 1..58 character ID.' >&2
    exit 2
fi
EXPECTED_OUTCOME="${SPIRE_LEARN_TASK4_EXPECT:-}"
if [[ "$EXPECTED_OUTCOME" != "red" && "$EXPECTED_OUTCOME" != "green" ]]; then
    printf '%s\n' 'SPIRE_LEARN_TASK4_EXPECT must be red or green.' >&2
    exit 2
fi

if [[ "${SPIRE_LEARN_TASK4_BOUNDED:-}" != "1" ]]; then
    export SPIRE_LEARN_TASK4_BOUNDED=1
    exec timeout --signal=TERM --kill-after=90s 210s bash "$0"
fi

RUN_ROOT="$TASK_ROOT/runtime/$RUN_ID"
LOG_ROOT="$RUN_ROOT/logs"
WRITABLE_ROOT="$RUN_ROOT/writable"
OUTER_LOG_ROOT="$TASK_ROOT/runtime-control/$RUN_ID"
RESERVATION_ID="task4-$RUN_ID"
CONTAINER_NAME="spirelearn-task4-$RUN_ID"
CONTAINER_ID=
CONTAINER_CREATED=0
CONTAINER_STARTED=0
CONTAINER_START_STATUS=143
DOCKER_CREATE_ATTEMPTED=0
RESERVATION_PENDING=0
SOURCE_PREFLIGHT_DONE=0
CLEANUP_FAILED=0
FINALIZING=0
FINAL_STATUS=0

if [[ ! -d "$ROOT" || ! -d "$VENV" || ! -r "$LOCK" || ! -r "$NATIVE" || ! -r "$RECONCILIATION" ]]; then
    printf '%s\n' 'Learner root or previously qualified CPU runtime is unavailable.' >&2
    exit 2
fi
if [[ ! -f "$LEDGER" ]]; then
    printf '%s\n' 'Shared diagnostic sampled-row ledger is missing; refusing to reset it.' >&2
    exit 2
fi
if [[ -e "$RUN_ROOT" || -e "$OUTER_LOG_ROOT" ]]; then
    printf '%s\n' 'Run directory or outer-control log already exists; refusing to reuse it.' >&2
    exit 2
fi

mkdir -p "$TASK_ROOT/runtime" "$TASK_ROOT/runtime-control"
if ! mkdir -m 700 "$RUN_ROOT"; then
    printf '%s\n' 'Could not create a new owned Task 4 runtime child.' >&2
    exit 2
fi
mkdir -m 700 "$LOG_ROOT" "$WRITABLE_ROOT"
mkdir -m 700 "$WRITABLE_ROOT/home" "$WRITABLE_ROOT/tmp"
if ! mkdir -m 700 "$OUTER_LOG_ROOT"; then
    printf '%s\n' 'Could not create a unique outer-control log directory.' >&2
    exit 2
fi

record() {
    local name=$1
    shift
    local status
    printf '%q ' "$@" > "$LOG_ROOT/$name.command.log"
    printf '\n' >> "$LOG_ROOT/$name.command.log"
    set +e
    "$@" > "$LOG_ROOT/$name.stdout.log" 2> "$LOG_ROOT/$name.stderr.log"
    status=$?
    set -e
    printf '%s\n' "$status" > "$LOG_ROOT/$name.exit.log"
    return "$status"
}

finalize() {
    local saved_status=$?
    if (( FINALIZING )); then
        exit "$saved_status"
    fi
    FINALIZING=1
    trap '' INT TERM
    trap - EXIT

    if (( DOCKER_CREATE_ATTEMPTED )); then
        if ! record container-remove timeout --signal=TERM --kill-after=4s 8s \
            docker rm -f "$CONTAINER_NAME"; then
            : # The final exact-name inventory determines whether a resource remains.
        fi
        if ! record container-final-inventory timeout --signal=TERM --kill-after=4s 8s \
            docker ps -a -q --no-trunc --filter "name=^/$CONTAINER_NAME$"; then
            CLEANUP_FAILED=1
        elif [[ -s "$LOG_ROOT/container-final-inventory.stdout.log" ]]; then
            CLEANUP_FAILED=1
        fi
    fi

    if (( SOURCE_PREFLIGHT_DONE )); then
        if ! record source-postcheck timeout --signal=TERM --kill-after=4s 8s \
            python3.12 -B "$ROOT/scripts/neural/task4_tensor_run_helpers.py" \
            source-after "$ROOT" "$RUN_ROOT" "$RUNTIME_ROOT"; then
            CLEANUP_FAILED=1
        fi
    fi

    if (( RESERVATION_PENDING )); then
        local sample_status=not_started
        if (( CONTAINER_STARTED )); then
            if (( CONTAINER_START_STATUS == 0 )); then sample_status=succeeded; else sample_status=failed; fi
        fi
        if ! record sample-row-finish timeout --signal=TERM --kill-after=4s 8s \
            python3.12 -B -m tests.neural._sample_guard \
            --ledger "$LEDGER" finish --reservation-id "$RESERVATION_ID" --status "$sample_status" \
            --evidence "$WRITABLE_ROOT/tensor-sampling.json"; then
            CLEANUP_FAILED=1
        fi
    fi

    local cleanup_args=()
    if (( DOCKER_CREATE_ATTEMPTED )); then cleanup_args+=(--create-attempted); fi
    if (( CONTAINER_CREATED )); then cleanup_args+=(--container-created); fi
    if (( CLEANUP_FAILED )); then cleanup_args+=(--cleanup-failed); fi
    if ! record cleanup-status timeout --signal=TERM --kill-after=2s 4s \
        python3.12 -B "$ROOT/scripts/neural/task4_tensor_run_helpers.py" \
        cleanup-status "$RUN_ROOT" "${cleanup_args[@]}"; then
        CLEANUP_FAILED=1
    fi

    if (( CLEANUP_FAILED )) && (( saved_status == 0 )); then
        saved_status=2
    fi
    local summary_args=(--pre-inventory-exit-status "$saved_status" --expected-outcome "$EXPECTED_OUTCOME")
    if (( CONTAINER_CREATED )); then summary_args+=(--container-created); fi
    if (( CONTAINER_STARTED )); then summary_args+=(--container-started); fi
    if ! record run-summary timeout --signal=TERM --kill-after=2s 4s \
        python3.12 -B "$ROOT/scripts/neural/task4_tensor_run_helpers.py" \
        run-summary "$RUN_ROOT" "${summary_args[@]}"; then
        saved_status=2
    fi

    printf '%q ' timeout --signal=TERM --kill-after=4s 14s \
        python3.12 -B "$ROOT/scripts/neural/task4_tensor_run_helpers.py" \
        inventory "$RUN_ROOT" > "$OUTER_LOG_ROOT/evidence-inventory.command.log"
    printf '\n' >> "$OUTER_LOG_ROOT/evidence-inventory.command.log"
    set +e
    timeout --signal=TERM --kill-after=4s 14s \
        python3.12 -B "$ROOT/scripts/neural/task4_tensor_run_helpers.py" \
        inventory "$RUN_ROOT" \
        > "$OUTER_LOG_ROOT/evidence-inventory.stdout.log" \
        2> "$OUTER_LOG_ROOT/evidence-inventory.stderr.log"
    local inventory_status=$?
    set -e
    printf '%s\n' "$inventory_status" > "$OUTER_LOG_ROOT/evidence-inventory.exit.log"
    if (( inventory_status != 0 )); then saved_status=2; fi
    printf '%s\n' "$saved_status" > "$OUTER_LOG_ROOT/wrapper-final-status.log"
    exit "$saved_status"
}

trap finalize EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

printf '%q ' timeout --signal=TERM --kill-after=90s 210s bash "$0" \
    > "$LOG_ROOT/wrapper-timeout.command.log"
printf '\n' >> "$LOG_ROOT/wrapper-timeout.command.log"

if [[ ! -L "$VENV/bin/python" ]]; then
    printf '%s\n' 'Qualified CPU venv Python link is missing.' >&2
    exit 2
fi
if ! record qualified-venv-python-link readlink -- "$VENV/bin/python"; then
    printf '%s\n' 'Could not inspect the qualified CPU venv Python link.' >&2
    exit 2
fi
if [[ "$(< "$LOG_ROOT/qualified-venv-python-link.stdout.log")" != "$EXPECTED_VENV_PYTHON_TARGET" ]]; then
    printf '%s\n' 'Qualified CPU venv Python link target changed.' >&2
    exit 2
fi

if ! record source-preflight timeout --signal=TERM --kill-after=4s 8s \
    python3.12 -B "$ROOT/scripts/neural/task4_tensor_run_helpers.py" \
    source-before "$ROOT" "$RUN_ROOT" "$RUNTIME_ROOT"; then
    printf '%s\n' 'Read-only source, free-space, control, CPU runtime, or Gym provenance preflight failed.' >&2
    exit 2
fi
SOURCE_PREFLIGHT_DONE=1

CONTAINER_COMMAND=$(cat <<'COMMAND'
set -eu
test "$(pwd)" = /learner
test -d /learner/src
test -d /learner/tests
test -d /learner/configs
test -r /work/venv-cpu/bin/python
test ! -w /learner/src
test ! -w /learner/tests
test ! -w /learner/configs
test ! -w /work/venv-cpu
mkdir -p /run/home /run/tmp
/work/venv-cpu/bin/python -B -m pip check
/work/venv-cpu/bin/python -B -c 'import importlib.metadata as m,json; expected={"spire-lab":"0.1.0","gymnasium":"1.4.0","numpy":"2.5.4","torch":"2.14.1+cpu"}; actual={k:m.version(k) for k in expected}; print(json.dumps(actual,sort_keys=True)); assert actual==expected'
exec timeout --signal=TERM --kill-after=15s 120s env SPIRE_LEARN_RUN_NEURAL_TENSOR=1 PYTHONPATH=/learner/src /work/venv-cpu/bin/python -B -m unittest discover -s tests/neural -v
COMMAND
)
printf '%s' "$CONTAINER_COMMAND" > "$LOG_ROOT/container-command.txt"

DOCKER_CREATE_FLAGS=(
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
    --mount "type=bind,src=$ROOT/configs,dst=/learner/configs,readonly"
    --mount "type=bind,src=$VENV,dst=/work/venv-cpu,readonly"
    --mount "type=bind,src=$WRITABLE_ROOT,dst=/run"
    --mount "type=bind,src=$WRITABLE_ROOT,dst=/learner/artifacts/development/neural-combat"
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
    --env SPIRE_LEARN_TENSOR_TASK=task4
    --env "SPIRE_LEARN_TENSOR_RESERVATION_ID=$RESERVATION_ID"
    --env SPIRE_LEARN_TENSOR_MAX_SAMPLED_ROWS=64
    --env SPIRE_LEARN_TENSOR_EVIDENCE_PATH=/run/tensor-sampling.json
    --env SPIRE_LEARN_TENSOR_METRICS_PATH=/run/model-metrics.json
    --label spire-learn.phase=task4-authored-frame-tensor-tests
    --label "spire-learn.owner=$RUN_ID"
)

if ! record docker-image-inspect timeout --signal=TERM --kill-after=3s 30s \
    docker image inspect --format '{{.Id}}' "$IMAGE"; then
    printf '%s\n' 'Pinned CPU image is unavailable locally; no image pull will be attempted.' >&2
    exit 2
fi
IMAGE_ID=$(< "$LOG_ROOT/docker-image-inspect.stdout.log")
if [[ ! "$IMAGE_ID" =~ ^sha256:[0-9a-f]{64}$ ]]; then
    printf '%s\n' 'Local image inspection did not return one immutable image ID.' >&2
    exit 2
fi

if ! record docker-existing-name timeout --signal=TERM --kill-after=3s 30s \
    docker ps -a -q --no-trunc --filter "name=^/$CONTAINER_NAME$"; then
    printf '%s\n' 'Docker container inventory preflight failed.' >&2
    exit 2
fi
if [[ -s "$LOG_ROOT/docker-existing-name.stdout.log" ]]; then
    printf '%s\n' 'Container name already exists; refusing to reuse it.' >&2
    exit 2
fi

# --help precedes IMAGE. This validates Docker's local argv grammar without a
# daemon request; later create uses exactly these flags and command arguments.
if ! record docker-create-cli-help timeout --signal=TERM --kill-after=3s 30s \
    docker create --pull=never "${DOCKER_CREATE_FLAGS[@]}" --entrypoint /bin/sh --help \
    "$IMAGE" -ec "$CONTAINER_COMMAND"; then
    printf '%s\n' 'Docker CLI rejected the reviewed create grammar; no container was created.' >&2
    exit 2
fi

if ! record sample-row-reserve python3.12 -B -m tests.neural._sample_guard \
    --ledger "$LEDGER" reserve --task task4 --reservation-id "$RESERVATION_ID" --rows 64; then
    printf '%s\n' 'Shared diagnostic sample reservation failed; no tensor test was started.' >&2
    exit 2
fi
RESERVATION_PENDING=1

DOCKER_CREATE_ATTEMPTED=1
if ! record container-create timeout --signal=TERM --kill-after=3s 30s \
    docker create --pull=never "${DOCKER_CREATE_FLAGS[@]}" --entrypoint /bin/sh \
    "$IMAGE" -ec "$CONTAINER_COMMAND"; then
    printf '%s\n' 'Docker create failed; tensor tests were not started.' >&2
    exit 2
fi
CONTAINER_CREATED=1
CONTAINER_ID=$(< "$LOG_ROOT/container-create.stdout.log")
if [[ ! "$CONTAINER_ID" =~ ^[0-9a-f]{64}$ ]]; then
    printf '%s\n' 'Docker create did not return one full immutable container ID.' >&2
    exit 2
fi

INSPECT_FORMAT='{"image_id":{{json .Image}},"user":{{json .Config.User}},"workdir":{{json .Config.WorkingDir}},"entrypoint":{{json .Config.Entrypoint}},"cmd":{{json .Config.Cmd}},"env":{{json .Config.Env}},"readonly_rootfs":{{json .HostConfig.ReadonlyRootfs}},"privileged":{{json .HostConfig.Privileged}},"cap_drop":{{json .HostConfig.CapDrop}},"security_opt":{{json .HostConfig.SecurityOpt}},"nano_cpus":{{json .HostConfig.NanoCpus}},"memory":{{json .HostConfig.Memory}},"pids_limit":{{json .HostConfig.PidsLimit}},"network":{{json .HostConfig.NetworkMode}},"mounts":{{json .Mounts}},"tmpfs":{{json .HostConfig.Tmpfs}}}'
if ! record container-inspect timeout --signal=TERM --kill-after=3s 30s \
    docker inspect --format "$INSPECT_FORMAT" "$CONTAINER_ID"; then
    printf '%s\n' 'Container control inspection failed; tests were not started.' >&2
    exit 2
fi
if ! record container-inspect-assertion python3.12 -B \
    "$ROOT/scripts/neural/task4_tensor_inspect.py" \
    "$LOG_ROOT/container-inspect.stdout.log" "$IMAGE_ID" "$ROOT" "$VENV" \
    "$WRITABLE_ROOT" "$RESERVATION_ID" "$LOG_ROOT/container-command.txt"; then
    printf '%s\n' 'Container control inspection differs from the reviewed CPU boundary.' >&2
    exit 2
fi

CONTAINER_STARTED=1
if record container-start timeout --signal=TERM --kill-after=5s 140s docker start -a "$CONTAINER_ID"; then
    CONTAINER_START_STATUS=0
else
    CONTAINER_START_STATUS=$?
fi

if record test-result timeout --signal=TERM --kill-after=3s 12s \
    python3.12 -B "$ROOT/scripts/neural/task4_tensor_run_helpers.py" \
    evaluate "$ROOT" "$RUN_ROOT" --expected-outcome "$EXPECTED_OUTCOME" \
    --reservation-id "$RESERVATION_ID" --max-rows 64; then
    FINAL_STATUS=0
else
    FINAL_STATUS=$?
fi
exit "$FINAL_STATUS"
