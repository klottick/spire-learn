#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  SPIRE_LEARN_TASK5_REVIEWED=1 SPIRE_LEARN_TASK5_RUN_ID=<fresh-id> \
    bash scripts/neural/run-task5-collection.sh --run

The run path is held until an internal parent and independent review release.
It uses the existing read-only qualified CPU runtime and runs the two gated
collection tests once in an offline non-root container. --help performs no
filesystem, Docker, runtime, Gym, or policy operation.
USAGE
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

if [[ "${1:-}" == "--run" ]]; then
    if [[ "${SPIRE_LEARN_TASK5_REVIEWED:-}" != "1" ]]; then
        printf '%s\n' 'Task 5 runtime is held for internal parent and independent review.' >&2
        exit 2
    fi
    if [[ -z "${SPIRE_LEARN_TASK5_RUN_ID:-}" ]]; then
        printf '%s\n' 'A fresh SPIRE_LEARN_TASK5_RUN_ID is required.' >&2
        exit 2
    fi
    exec timeout --signal=TERM --kill-after=90s 210s bash "$0" \
        --internal "$SPIRE_LEARN_TASK5_RUN_ID"
fi

if [[ "${1:-}" != "--internal" || $# -ne 2 ]]; then
    usage >&2
    exit 2
fi
if [[ "${SPIRE_LEARN_TASK5_REVIEWED:-}" != "1" ]]; then
    printf '%s\n' 'Task 5 runtime is held for internal parent and independent review.' >&2
    exit 2
fi

RUN_ID="$2"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{1,63}$ ]]; then
    printf '%s\n' 'Run ID must be a fresh 2..64 character alphanumeric slug.' >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
CONTROL_TOOL="$ROOT/scripts/neural/task5_collection_control.py"
RUN_ID_PREFIX="spire-learn-task5-"
CONTAINER_NAME="$RUN_ID_PREFIX$RUN_ID"
RUNTIME_ROOT="$ROOT/artifacts/development/neural-combat/runtime/db72a7eb-c789-4cf4-bfa5-42c1ada5e3fc"
VENV="$RUNTIME_ROOT/data/venv-cpu"
LOCK="$RUNTIME_ROOT/data/requirements-cpu.lock"
RUNTIME_AREA="$ROOT/artifacts/development/neural-combat/task-5/runtime"
CONTROL_AREA="$ROOT/artifacts/development/neural-combat/task-5/runtime-control"
RUN_ROOT="$RUNTIME_AREA/$RUN_ID"
CONTROL_ROOT="$CONTROL_AREA/$RUN_ID"
LOG_ROOT="$CONTROL_ROOT/logs"
IMAGE="docker.io/library/python@sha256:57d96f71de145a98e7ceca18ee4ade77bba4103a33a72dfaea026b23625c6bf0"
EXPECTED_IMAGE_ID="sha256:57d96f71de145a98e7ceca18ee4ade77bba4103a33a72dfaea026b23625c6bf0"
EXPECTED_PYTHON_LINK="/usr/local/bin/python"
EXPECTED_LOCK_SHA256="51714575fe45381716e06d482ad0f6b064c60ffd0e64ec5bf2753158cf31f2d0"
EXPECTED_NATIVE_SHA256="071069cea66fbf0bc7277002c6b8a21f138194cf692326595109a6741a5178e9"
EXPECTED_GYM_REVISION="533dad1c8aec86df624b3742b6b3ad1dbf1d0033"
CONTAINER_ID=""
DOCKER_CREATE_ATTEMPTED=0
CONTAINER_CREATED=0
CONTAINER_STARTED=0
CLEANUP_FAILED=0
TEST_EXIT=not_started
FINAL_STATUS=2

if [[ -e "$RUN_ROOT" || -L "$RUN_ROOT" || -e "$CONTROL_ROOT" || -L "$CONTROL_ROOT" ]]; then
    printf '%s\n' 'Run ID already exists; refusing to reuse prior outputs or container identity.' >&2
    exit 2
fi
mkdir -p "$LOG_ROOT"

record() {
    local name="$1"
    shift
    local status
    local restore_errexit=0
    [[ "$-" == *e* ]] && restore_errexit=1
    printf '%q ' "$@" > "$LOG_ROOT/$name.command.log"
    printf '\n' >> "$LOG_ROOT/$name.command.log"
    set +e
    "$@" > "$LOG_ROOT/$name.stdout.log" 2> "$LOG_ROOT/$name.stderr.log"
    status=$?
    if (( restore_errexit )); then
        set -e
    else
        set +e
    fi
    printf '%s\n' "$status" > "$LOG_ROOT/$name.exit.log"
    return "$status"
}

write_run_summary() {
    local status="$1"
    local cleanup_status="$2"
    local recovery_status="$3"
    local removal_status="$4"
    local inventory_status="$5"
    local source_status="$6"
    local final_inventory_status="$7"
    cat > "$CONTROL_ROOT/run-summary.json" <<SUMMARY
{
  "schema": "spire-learn-neural-collection-run-v1",
  "run_id": "$RUN_ID",
  "container_name": "$CONTAINER_NAME",
  "container_id": "$CONTAINER_ID",
  "container_create_attempted": $DOCKER_CREATE_ATTEMPTED,
  "container_created": $CONTAINER_CREATED,
  "container_started": $CONTAINER_STARTED,
  "unittest_exit": "$TEST_EXIT",
  "cleanup_failed": $cleanup_status,
  "exact_name_recovery_inventory_exit": "$recovery_status",
  "remove_exit": "$removal_status",
  "exact_name_inventory_exit": "$inventory_status",
  "source_postcheck_exit": "$source_status",
  "final_evidence_inventory_exit": "$final_inventory_status",
  "wrapper_exit": $status,
  "expected_gym_revision": "$EXPECTED_GYM_REVISION",
  "optimizer_updates": 0
}
SUMMARY
}

finish() {
    local prior_status=$?
    local removal_status=not_needed
    local recovery_status=not_run
    local inventory_query_status=not_run
    local exact_name_absent=false
    local source_status=not_run
    local final_inventory_status=not_run
    trap - EXIT
    trap '' INT TERM
    set +e

    if (( DOCKER_CREATE_ATTEMPTED && ! CONTAINER_CREATED )); then
        if record exact-name-recovery-inventory timeout --signal=TERM --kill-after=1s 7s \
            docker ps -a -q --no-trunc --filter "name=^/$CONTAINER_NAME$"; then
            recovery_status=0
            local recovery_path="$LOG_ROOT/exact-name-recovery-inventory.stdout.log"
            if [[ -s "$recovery_path" ]]; then
                local recovered_id
                recovered_id="$(< "$recovery_path")"
                if [[ "$recovered_id" =~ ^[0-9a-f]{64}$ ]]; then
                    CONTAINER_ID="$recovered_id"
                    CONTAINER_CREATED=1
                else
                    CLEANUP_FAILED=1
                fi
            fi
        else
            recovery_status=$?
            CLEANUP_FAILED=1
        fi
    fi

    if (( CONTAINER_CREATED )); then
        local removal_target="$CONTAINER_ID"
        if [[ ! "$removal_target" =~ ^[0-9a-f]{64}$ ]]; then
            removal_target="$CONTAINER_NAME"
        fi
        if record container-remove timeout --signal=TERM --kill-after=2s 8s \
            docker rm -f "$removal_target"; then
            removal_status=0
        else
            removal_status=$?
            CLEANUP_FAILED=1
        fi
    fi

    if (( DOCKER_CREATE_ATTEMPTED )); then
        if record final-name-inventory timeout --signal=TERM --kill-after=1s 6s \
            docker ps -a -q --no-trunc --filter "name=^/$CONTAINER_NAME$"; then
            inventory_query_status=0
            if [[ -s "$LOG_ROOT/final-name-inventory.stdout.log" ]]; then
                CLEANUP_FAILED=1
            else
                exact_name_absent=true
            fi
        else
            inventory_query_status=$?
            CLEANUP_FAILED=1
        fi
    fi

    if [[ -f "$RUN_ROOT/source-before.json" ]]; then
        if record source-after timeout --signal=TERM --kill-after=1s 6s \
            python3.12 -B "$CONTROL_TOOL" snapshot --root "$ROOT" \
            --output "$RUN_ROOT/source-after.json"; then
            if record source-compare timeout --signal=TERM --kill-after=1s 4s \
                python3.12 -B "$CONTROL_TOOL" compare \
                --before "$RUN_ROOT/source-before.json" --after "$RUN_ROOT/source-after.json"; then
                source_status=0
            else
                source_status=$?
                CLEANUP_FAILED=1
            fi
        else
            source_status=$?
            CLEANUP_FAILED=1
        fi
    fi

    if [[ -d "$RUN_ROOT" ]]; then
        if record final-evidence-inventory timeout --signal=TERM --kill-after=1s 6s \
            python3.12 -B "$CONTROL_TOOL" inventory --root "$RUN_ROOT" \
            --output "$CONTROL_ROOT/final-inventory.json"; then
            final_inventory_status=0
        else
            final_inventory_status=$?
            CLEANUP_FAILED=1
        fi
    fi

    local final_status="$prior_status"
    if (( CLEANUP_FAILED )) && (( final_status == 0 )); then
        final_status=2
    fi
    cat > "$CONTROL_ROOT/cleanup-status.json" <<CLEANUP
{
  "schema": "spire-learn-neural-collection-cleanup-v1",
  "container_create_attempted": $DOCKER_CREATE_ATTEMPTED,
  "container_created": $CONTAINER_CREATED,
  "container_started": $CONTAINER_STARTED,
  "recovery_inventory_exit": "$recovery_status",
  "remove_exit": "$removal_status",
  "exact_name_inventory_exit": "$inventory_query_status",
  "exact_name_absent": $exact_name_absent,
  "source_postcheck_exit": "$source_status",
  "final_evidence_inventory_exit": "$final_inventory_status",
  "cleanup_failed": $CLEANUP_FAILED,
  "wrapper_exit": $final_status
}
CLEANUP
    write_run_summary "$final_status" "$CLEANUP_FAILED" "$recovery_status" \
        "$removal_status" "$inventory_query_status" "$source_status" \
        "$final_inventory_status"
    printf '%s\n' "$final_status" > "$CONTROL_ROOT/wrapper-final-status.log"
    exit "$final_status"
}

trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ ! -L "$VENV/bin/python" ]]; then
    printf '%s\n' 'Qualified CPU venv Python link is missing; no target is followed on the host.' >&2
    exit 2
fi
if ! record qualified-python-link readlink -- "$VENV/bin/python"; then
    printf '%s\n' 'Could not read the qualified CPU venv Python link text.' >&2
    exit 2
fi
if [[ "$(< "$LOG_ROOT/qualified-python-link.stdout.log")" != "$EXPECTED_PYTHON_LINK" ]]; then
    printf '%s\n' 'Qualified CPU venv Python link target text changed.' >&2
    exit 2
fi
if [[ ! -f "$LOCK" || "$(sha256sum "$LOCK" | awk '{print $1}')" != "$EXPECTED_LOCK_SHA256" ]]; then
    printf '%s\n' 'Qualified CPU requirements lock is missing or changed.' >&2
    exit 2
fi
free_kb="$(df -Pk "$ROOT" | awk 'NR == 2 {print $4}')"
if [[ ! "$free_kb" =~ ^[0-9]+$ ]] || (( free_kb < 30 * 1024 * 1024 )); then
    printf '%s\n' 'At least 30 GiB must be free on the protected run filesystem.' >&2
    exit 2
fi
mkdir -p "$RUN_ROOT"

if ! record source-before timeout --signal=TERM --kill-after=2s 10s \
    python3.12 -B "$CONTROL_TOOL" snapshot --root "$ROOT" \
    --output "$RUN_ROOT/source-before.json"; then
    printf '%s\n' 'Source, dirty-state, or input hash preflight failed.' >&2
    exit 2
fi
if ! record pre-dispatch-manifest python3.12 -B "$CONTROL_TOOL" pre-dispatch \
    --source-before "$RUN_ROOT/source-before.json" \
    --output "$RUN_ROOT/pre-dispatch.json"; then
    printf '%s\n' 'Per-test slot limits and pre-dispatch provenance could not be recorded.' >&2
    exit 2
fi

if ! record docker-image-inspect timeout --signal=TERM --kill-after=2s 8s \
    docker image inspect --format '{{.Id}}' "$IMAGE"; then
    printf '%s\n' 'Pinned CPU image is unavailable locally; no pull is attempted.' >&2
    exit 2
fi
IMAGE_ID="$(< "$LOG_ROOT/docker-image-inspect.stdout.log")"
if [[ "$IMAGE_ID" != "$EXPECTED_IMAGE_ID" ]]; then
    printf '%s\n' 'Local CPU image identity differs from the reviewed digest.' >&2
    exit 2
fi
if ! record docker-existing-name timeout --signal=TERM --kill-after=2s 8s \
    docker ps -a -q --no-trunc --filter "name=^/$CONTAINER_NAME$"; then
    printf '%s\n' 'Docker exact-name inventory failed.' >&2
    exit 2
fi
if [[ -s "$LOG_ROOT/docker-existing-name.stdout.log" ]]; then
    printf '%s\n' 'The exact container name already exists; no container will be reused.' >&2
    exit 2
fi

CONTAINER_COMMAND=$(cat <<'COMMAND'
set -eu
test "$(pwd)" = /learner
test -r /work/venv-cpu/bin/python
test ! -w /learner/src
test ! -w /learner/tests
test ! -w /learner/configs
test ! -w /learner/scripts/neural/task5_collection_control.py
test ! -w /work/venv-cpu
test ! -w /work/requirements-cpu.lock
mkdir -p /run/home /run/tmp
/work/venv-cpu/bin/python -B -m pip check
/work/venv-cpu/bin/python -B -c 'import hashlib,importlib.metadata as m,importlib.util,json,pathlib; expected={"spire-lab":"0.1.0","gymnasium":"1.4.0","numpy":"2.5.4","torch":"2.14.1+cpu"}; actual={name:m.version(name) for name in expected}; assert actual==expected,actual; spec=importlib.util.find_spec("spire_lab"); assert spec is not None; roots=[pathlib.Path(item) for item in (spec.submodule_search_locations or [str(pathlib.Path(spec.origin).parent)])]; native=sorted({p for root in roots for p in root.rglob("*.so")}); hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in native}; expected_native="071069cea66fbf0bc7277002c6b8a21f138194cf692326595109a6741a5178e9"; assert expected_native in hashes.values(),hashes; lock=pathlib.Path("/work/requirements-cpu.lock"); lock_hash=hashlib.sha256(lock.read_bytes()).hexdigest(); assert lock_hash=="51714575fe45381716e06d482ad0f6b064c60ffd0e64ec5bf2753158cf31f2d0",lock_hash; report={"packages":actual,"native_sha256":expected_native,"lock_sha256":lock_hash,"cuda_available":False,"gym_revision":"533dad1c8aec86df624b3742b6b3ad1dbf1d0033"}; pathlib.Path("/run/runtime-identity.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n"); print(json.dumps(report,sort_keys=True))'
env SPIRE_LEARN_RUN_NEURAL_COLLECTION=1 SPIRE_LEARN_COLLECTION_EVIDENCE=/run/native-collection-evidence.json PYTHONPATH=/learner/src PYTHONDONTWRITEBYTECODE=1 /work/venv-cpu/bin/python -B -m unittest tests.integration.test_neural_collection -v
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
    --mount "type=bind,src=$CONTROL_TOOL,dst=/learner/scripts/neural/task5_collection_control.py,readonly"
    --mount "type=bind,src=$VENV,dst=/work/venv-cpu,readonly"
    --mount "type=bind,src=$LOCK,dst=/work/requirements-cpu.lock,readonly"
    --mount "type=bind,src=$RUN_ROOT,dst=/run"
    --mount "type=bind,src=$RUN_ROOT,dst=/learner/artifacts/development/neural-combat"
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
    --label spire-learn.phase=task5-both-mode-collection
    --label "spire-learn.owner=$RUN_ID"
)

if ! record docker-create-cli-help timeout --signal=TERM --kill-after=2s 10s \
    docker create --pull=never "${DOCKER_CREATE_FLAGS[@]}" \
    --entrypoint /bin/sh --help "$IMAGE" -ec "$CONTAINER_COMMAND"; then
    printf '%s\n' 'Docker rejected the reviewed local create grammar; no container was created.' >&2
    exit 2
fi

DOCKER_CREATE_ATTEMPTED=1
if ! record container-create timeout --signal=TERM --kill-after=3s 20s \
    docker create --pull=never "${DOCKER_CREATE_FLAGS[@]}" \
    --entrypoint /bin/sh "$IMAGE" -ec "$CONTAINER_COMMAND"; then
    printf '%s\n' 'Docker container creation failed.' >&2
    exit 2
fi
CONTAINER_CREATED=1
CONTAINER_ID="$(< "$LOG_ROOT/container-create.stdout.log")"
if [[ ! "$CONTAINER_ID" =~ ^[0-9a-f]{64}$ ]]; then
    printf '%s\n' 'Docker create did not return one full container ID.' >&2
    exit 2
fi

INSPECT_FORMAT='{"image_id":{{json .Image}},"user":{{json .Config.User}},"workdir":{{json .Config.WorkingDir}},"entrypoint":{{json .Config.Entrypoint}},"cmd":{{json .Config.Cmd}},"env":{{json .Config.Env}},"readonly_rootfs":{{json .HostConfig.ReadonlyRootfs}},"privileged":{{json .HostConfig.Privileged}},"cap_drop":{{json .HostConfig.CapDrop}},"security_opt":{{json .HostConfig.SecurityOpt}},"nano_cpus":{{json .HostConfig.NanoCpus}},"memory":{{json .HostConfig.Memory}},"pids_limit":{{json .HostConfig.PidsLimit}},"network":{{json .HostConfig.NetworkMode}},"mounts":{{json .Mounts}},"tmpfs":{{json .HostConfig.Tmpfs}}}'
if ! record container-inspect timeout --signal=TERM --kill-after=2s 10s \
    docker inspect --format "$INSPECT_FORMAT" "$CONTAINER_ID"; then
    printf '%s\n' 'Container control inspection failed; tests were not started.' >&2
    exit 2
fi
if ! record container-inspect-assertion python3.12 -B "$CONTROL_TOOL" \
    verify-inspect --input "$LOG_ROOT/container-inspect.stdout.log" \
    --root "$ROOT" --venv "$VENV" --run-root "$RUN_ROOT" \
    --expected-command-file "$LOG_ROOT/container-command.txt" \
    --image-id "$IMAGE_ID" --output "$RUN_ROOT/container-boundary.json"; then
    printf '%s\n' 'Container settings or mount boundary differs from the reviewed design.' >&2
    exit 2
fi

CONTAINER_STARTED=1
if record container-start timeout --signal=TERM --kill-after=15s 140s \
    docker start -a "$CONTAINER_ID"; then
    TEST_EXIT=0
    FINAL_STATUS=0
else
    TEST_EXIT=$?
    FINAL_STATUS=2
fi

if ! record result-classification python3.12 -B "$CONTROL_TOOL" classify \
    --stdout "$LOG_ROOT/container-start.stdout.log" \
    --stderr "$LOG_ROOT/container-start.stderr.log" \
    --evidence "$RUN_ROOT/native-collection-evidence.json" \
    --exit-code "$TEST_EXIT" --output "$RUN_ROOT/collection-result.json"; then
    FINAL_STATUS=2
fi
exit "$FINAL_STATUS"
