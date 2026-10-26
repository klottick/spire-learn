#!/usr/bin/env bash
set -Eeuo pipefail

# Bound the complete host wrapper. TERM starts cleanup at t=210; GNU timeout
# sends KILL at t=300 if the wrapper and bounded cleanup have not exited.
if [[ "${SPIRE_LEARN_TASK3_GUARDED:-}" != "1" ]]; then
    if ! command -v timeout >/dev/null 2>&1; then
        printf '%s\n' 'GNU timeout is required for the 210s plus 90s whole-wrapper bound.' >&2
        exit 2
    fi
    export SPIRE_LEARN_TASK3_GUARDED=1
    exec timeout --signal=TERM --kill-after=90s 210s "$0" "$@"
fi

ROOT=/home/klottick/repos/spire-learn
TASK_ROOT="$ROOT/artifacts/development/neural-combat/task-3"
RUNTIME_ROOT="$ROOT/artifacts/development/neural-combat/runtime/db72a7eb-c789-4cf4-bfa5-42c1ada5e3fc"
VENV="$RUNTIME_ROOT/data/venv-cpu"
LOCK="$RUNTIME_ROOT/data/requirements-cpu.lock"
NATIVE="$VENV/lib/python3.12/site-packages/spire_lab/_native.cpython-312-x86_64-linux-gnu.so"
RECONCILIATION="$RUNTIME_ROOT/cpu-source-reconciliation.json"
IMAGE=docker.io/library/python@sha256:57d96f71de145a98e7ceca18ee4ade77bba4103a33a72dfaea026b23625c6bf0
EXPECTED_LOCK=51714575fe45381716e06d482ad0f6b064c60ffd0e64ec5bf2753158cf31f2d0
EXPECTED_NATIVE=071069cea66fbf0bc7277002c6b8a21f138194cf692326595109a6741a5178e9
EXPECTED_SOURCE_MAP=dc16f78dba274c4e7bd560e424620b877c471cec0b736aa3afa2e2b48f300ea3
EXPECTED_GYM_REVISION=533dad1c8aec86df624b3742b6b3ad1dbf1d0033
EXPECTED_INPUT_TREE=5b1779d740ce3ebd6d419768bb10b72d10d0d5a640857e310780edbb8bd08f63
EXPECTED_VENV_PYTHON_TARGET=/usr/local/bin/python
MIN_FREE_BYTES=32212254720

if [[ "${SPIRE_LEARN_TASK3_REVIEWED:-}" != "1" ]]; then
    printf '%s\n' 'Refusing native invocation: parent code/runbook review gate is unset.' >&2
    exit 2
fi

check_host_venv_python_link() {
    local venv_path=$1
    local expected_target=$2
    local python_link="$venv_path/bin/python"
    local observed_target
    if [[ ! -L "$python_link" ]]; then
        printf 'Qualified CPU venv interpreter link is missing: %s\n' "$python_link" >&2
        return 1
    fi
    if ! observed_target=$(readlink -- "$python_link"); then
        printf 'Could not inspect qualified CPU venv interpreter link: %s\n' "$python_link" >&2
        return 1
    fi
    if [[ "$observed_target" != "$expected_target" ]]; then
        printf 'Qualified CPU venv interpreter link target changed: expected %s, got %s\n' \
            "$expected_target" "$observed_target" >&2
        return 1
    fi
    return 0
}

RUN_ID="${SPIRE_LEARN_TASK3_RUN_ID:-}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]]; then
    printf '%s\n' 'SPIRE_LEARN_TASK3_RUN_ID must be a unique 1..64 character ID.' >&2
    exit 2
fi

RUN_ROOT="$TASK_ROOT/runtime/$RUN_ID"
LOG_ROOT="$RUN_ROOT/logs"
WRITABLE_ROOT="$RUN_ROOT/writable"
CONTAINER_NAME="spirelearn-task3-$RUN_ID"
CONTAINER_ID=
CLEANUP_STARTED=0
CLEANUP_FAILED=0
SOURCE_GATE_PASSED=0
SOURCE_POSTCHECK_DONE=0
SOURCE_POSTCHECK_STATUS=0
NATIVE_INVOKE_STARTED=0
NATIVE_RESULT_DONE=0
NATIVE_STATUS=143
FINAL_STATUS=0
DOCKER_INVENTORY_EMPTY=not-applicable
CONTAINER_NAME_ABSENCE_VERIFIED=0

if [[ ! -d "$ROOT" || ! -d "$VENV" ]]; then
    printf '%s\n' 'Learner root or previously qualified CPU venv is unavailable.' >&2
    exit 2
fi
if ! check_host_venv_python_link "$VENV" "$EXPECTED_VENV_PYTHON_TARGET"; then
    printf '%s\n' 'Qualified CPU venv interpreter link metadata differs from the reviewed runtime.' >&2
    exit 2
fi
if [[ ! -r "$LOCK" || ! -r "$NATIVE" || ! -r "$RECONCILIATION" ]]; then
    printf '%s\n' 'Previously qualified CPU lock, native extension or reconciliation is unavailable.' >&2
    exit 2
fi
FREE_BYTES=$(python3.12 -B -c 'import shutil,sys; print(shutil.disk_usage(sys.argv[1]).free)' "$TASK_ROOT")
if [[ ! "$FREE_BYTES" =~ ^[0-9]+$ ]] || (( FREE_BYTES < MIN_FREE_BYTES )); then
    printf '%s\n' 'At least 30 GiB free is required before creating a Task 3 run directory.' >&2
    exit 2
fi

mkdir -p "$TASK_ROOT/runtime"
if ! mkdir "$RUN_ROOT"; then
    printf '%s\n' 'Run directory already exists; refusing to reuse outputs or logs.' >&2
    exit 2
fi
mkdir "$LOG_ROOT" "$WRITABLE_ROOT"
mkdir "$WRITABLE_ROOT/home" "$WRITABLE_ROOT/tmp"

write_source_check_helper() {
    cat > "$LOG_ROOT/source-check.py" <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

mode, root_arg, run_arg, runtime_arg, native_arg, expected_tree, free_bytes = sys.argv[1:]
root = Path(root_arg)
run_root = Path(run_arg)
runtime_root = Path(runtime_arg)
native_path = Path(native_arg)

def source_files() -> dict[str, str]:
    result: dict[str, str] = {}
    for top in ("src", "tests", "configs"):
        for path in sorted((root / top).rglob("*")):
            if path.is_symlink():
                raise SystemExit(f"symlink is not permitted in reviewed mounted inputs: {path.relative_to(root)}")
            if path.is_file():
                relative = path.relative_to(root).as_posix()
                result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result

files = source_files()
tree_hash = hashlib.sha256(
    json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
if tree_hash != expected_tree:
    raise SystemExit("mounted learner src/tests/configs tree differs from the reviewed digest")
lock_path = runtime_root / "data/requirements-cpu.lock"
reconciliation_path = runtime_root / "cpu-source-reconciliation.json"
reconciliation = json.loads(reconciliation_path.read_text(encoding="utf-8"))
lock_hash = hashlib.sha256(lock_path.read_bytes()).hexdigest()
native_hash = hashlib.sha256(native_path.read_bytes()).hexdigest()
if lock_hash != "51714575fe45381716e06d482ad0f6b064c60ffd0e64ec5bf2753158cf31f2d0":
    raise SystemExit("qualified CPU lock hash differs from the reviewed value")
if native_hash != "071069cea66fbf0bc7277002c6b8a21f138194cf692326595109a6741a5178e9":
    raise SystemExit("qualified CPU extension hash differs from the reviewed value")
if reconciliation.get("source_map_sha256") != "dc16f78dba274c4e7bd560e424620b877c471cec0b736aa3afa2e2b48f300ea3":
    raise SystemExit("durable Gym source-map hash differs from the reviewed reconciliation")
if reconciliation.get("original_source_posthash_available") is not False:
    raise SystemExit("Gym source post-hash qualification changed")
if reconciliation.get("verified_files") != 636 or reconciliation.get("normalized_git_modes_match") is not True:
    raise SystemExit("durable Gym source reconciliation differs from its reviewed evidence")

if mode == "before":
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
        check=True, capture_output=True, text=True,
    ).stdout
    record = {
        "git_head": revision,
        "git_status_porcelain_v1": status,
        "reviewed_input_tree_sha256": expected_tree,
        "current_input_tree_sha256": tree_hash,
        "learner_reviewed_file_sha256": files,
        "free_bytes_at_preflight": int(free_bytes),
        "cpu_lock_sha256": lock_hash,
        "native_sha256": native_hash,
        "gym_revision": "533dad1c8aec86df624b3742b6b3ad1dbf1d0033",
        "gym_source_map_sha256": reconciliation["source_map_sha256"],
        "gym_source_postcheck_kind": reconciliation["source_postcheck_kind"],
        "gym_original_source_posthash_available": reconciliation["original_source_posthash_available"],
        "gym_durable_verified_files": reconciliation["verified_files"],
        "gym_durable_modes_match": reconciliation["normalized_git_modes_match"],
        "container_output_aliases": [
            {"host": str(run_root / "writable"), "container": "/run", "read_write": True},
            {"host": str(run_root / "writable"), "container": "/learner/artifacts/development/neural-combat", "read_write": True},
        ],
        "native_invocation": "one gated three-test conformance command; no retry",
    }
    (run_root / "host-preflight.json").write_text(
        json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"git_head": revision, "tree_sha256": tree_hash, "file_count": len(files)}))
elif mode == "after":
    before = json.loads((run_root / "host-preflight.json").read_text(encoding="utf-8"))
    if files != before["learner_reviewed_file_sha256"]:
        raise SystemExit("a mounted learner src/test/config file changed during conformance")
    if tree_hash != before["current_input_tree_sha256"]:
        raise SystemExit("learner input tree digest changed during conformance")
    if lock_hash != before["cpu_lock_sha256"] or native_hash != before["native_sha256"]:
        raise SystemExit("read-only CPU runtime identity changed during conformance")
    record = {
        "input_tree_sha256_before": before["current_input_tree_sha256"],
        "input_tree_sha256_after": tree_hash,
        "learner_reviewed_file_sha256_after": files,
        "cpu_lock_sha256_after": lock_hash,
        "native_sha256_after": native_hash,
        "unchanged": True,
    }
    (run_root / "source-postcheck.json").write_text(
        json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"tree_sha256": tree_hash, "file_count": len(files), "unchanged": True}))
else:
    raise SystemExit("source-check mode must be before or after")
PY
}

write_inspection_assertion_helper() {
    cat > "$LOG_ROOT/assert-container.py" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

inspect_path, image_id_path, root_arg, venv_arg, writable_arg, expected_command = sys.argv[1:]
inspect = json.loads(Path(inspect_path).read_text(encoding="utf-8"))
image_id = Path(image_id_path).read_text(encoding="utf-8").strip()
root = str(Path(root_arg))
venv = str(Path(venv_arg))
writable = str(Path(writable_arg))
assert inspect["image_id"] == image_id, "container image ID differs from inspected pinned image"
assert inspect["user"] == "1000:1000"
assert inspect["workdir"] == "/learner"
assert inspect["readonly_rootfs"] is True
assert inspect["privileged"] is False
assert "ALL" in inspect["cap_drop"]
assert any(item in {"no-new-privileges", "no-new-privileges:true"} for item in inspect["security_opt"])
assert inspect["nano_cpus"] == 4_000_000_000
assert inspect["memory"] == 8 * 1024**3
assert inspect["pids_limit"] == 512
assert inspect["network"] == "none"
assert set(inspect["tmpfs"]) == {"/tmp"}
tmpfs_options = set(inspect["tmpfs"]["/tmp"].split(","))
assert {"rw", "noexec", "nosuid"} <= tmpfs_options
tmpfs_size = next((item.split("=", 1)[1] for item in tmpfs_options if item.startswith("size=")), None)
assert tmpfs_size in {"64m", str(64 * 1024 * 1024)}
mounts = {item["Destination"]: item for item in inspect["mounts"]}
expected = {
    "/learner/src": (str(Path(root) / "src"), False),
    "/learner/tests": (str(Path(root) / "tests"), False),
    "/learner/configs": (str(Path(root) / "configs"), False),
    "/work/venv-cpu": (venv, False),
    "/run": (writable, True),
    "/learner/artifacts/development/neural-combat": (writable, True),
}
assert set(mounts) == set(expected), f"unexpected mount destinations: {sorted(mounts)}"
for destination, (source, read_write) in expected.items():
    item = mounts[destination]
    assert item["Type"] == "bind", f"{destination} is not a bind mount"
    assert item["Source"] == source, f"{destination} source changed"
    assert item["RW"] is read_write, f"{destination} access mode changed"
env = set(inspect["env"])
required_env = {
    "PYTHONPATH=/learner/src",
    "SPIRE_LEARN_RUN_NEURAL_NATIVE=1",
    "SPIRE_LEARN_TASK3_EVIDENCE_PATH=/run/task3-counters.json",
    "PYTHONDONTWRITEBYTECODE=1",
    "PIP_NO_INDEX=1",
    "PIP_NO_CACHE_DIR=1",
    "HOME=/run/home",
    "TMPDIR=/run/tmp",
}
assert required_env <= env
assert inspect["entrypoint"] == ["/bin/sh"]
assert inspect["cmd"] == ["-ec", expected_command], "container command differs from the reviewed create argv"
assert expected_command.splitlines()[-1] == (
    "exec /work/venv-cpu/bin/python -B -m unittest tests.integration.test_neural_envs -v"
), "reviewed container command does not end with the expected native unittest command"
print(json.dumps({"controls_asserted": True, "mount_destinations": sorted(mounts)}))
PY
}

write_native_result_helper() {
    cat > "$LOG_ROOT/native-result.py" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

exit_status = int(sys.argv[1])
counter_path = Path(sys.argv[2])
stdout_path = Path(sys.argv[3])
stderr_path = Path(sys.argv[4])
result_path = Path(sys.argv[5])
output = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
output += "\n" + (stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else "")
match = re.search(r"Ran\s+(\d+)\s+tests?\s+in\s+[0-9.]+s", output)
run_count = int(match.group(1)) if match else None
skipped_match = re.search(r"\(skipped=(\d+)\)", output)
skipped_count = int(skipped_match.group(1)) if skipped_match else 0
counters = None
if counter_path.is_file():
    counters = json.loads(counter_path.read_text(encoding="utf-8"))
expected_tests = {
    "test_standalone_public_profile",
    "test_manual_vector_mixed_reset",
    "test_normal_run_common_view",
}
actions = counters.get("action_budget", {}) if counters is not None else {}
resets = counters.get("reset_counters", {}) if counters is not None else {}
closes = counters.get("close_counters", {}) if counters is not None else {}
lifecycle = counters.get("lifecycle_counters", {}) if counters is not None else {}
run_outcomes = lifecycle.get("exit_outcomes_by_mode", {}).get("run", {})
suite_ok = (
    exit_status == 0
    and run_count == 3
    and skipped_count == 0
    and counters is not None
    and counters.get("tests_run") == 3
    and counters.get("tests_skipped") == 0
    and counters.get("native_enabled") is True
    and set(counters.get("test_names", [])) == expected_tests
    and counters.get("cohort_fault_latched") is False
    and counters.get("public_catalog_splits_validated") == ["train", "selection", "final"]
    and actions.get("slot_actions_upper_bound", 4097) <= 4096
    and actions.get("max_fight_slot_actions", 1025) <= 1024
    and closes.get("driver_close_api_failures", 1) == 0
    and closes.get("driver_close_api_calls") == counters.get("driver_constructions")
    and closes.get("automatic_fault_close_attempts", 1) == 0
    and resets.get("delivered_calls") == resets.get("calls")
    and run_outcomes.get("victory", 0) >= 1
)
record = {
    "native_invocation_exit": exit_status,
    "unittest_tests_run": run_count,
    "unittest_tests_skipped": skipped_count,
    "suite_success": suite_ok,
    "counters_available": counters is not None,
    "counters": counters,
}
result_path.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"tests_run": run_count, "tests_skipped": skipped_count, "suite_success": suite_ok}))
if not suite_ok:
    raise SystemExit(1)
PY
}

record_command() {
    local name="$1"
    shift
    local status=0
    printf '%q ' "$@" > "$LOG_ROOT/$name.command.log"
    printf '\n' >> "$LOG_ROOT/$name.command.log"
    if "$@" > "$LOG_ROOT/$name.stdout.log" 2> "$LOG_ROOT/$name.stderr.log"; then
        status=0
    else
        status=$?
    fi
    printf '%s\n' "$status" > "$LOG_ROOT/$name.exit.log"
    return "$status"
}

cleanup() {
    local status=0
    CLEANUP_STARTED=1
    trap '' TERM INT
    set +e
    if [[ -d "$LOG_ROOT" ]]; then
        if [[ "$CONTAINER_NAME_ABSENCE_VERIFIED" == "1" ]] && \
            record_command cleanup-before-remove timeout --signal=TERM --kill-after=2s 8s \
                docker ps -a -q --no-trunc --filter "name=^/$CONTAINER_NAME$"; then
            local listed_id
            listed_id=$(< "$LOG_ROOT/cleanup-before-remove.stdout.log")
            if [[ -n "$listed_id" ]]; then
                if [[ -n "$CONTAINER_ID" && "$listed_id" == "$CONTAINER_ID" ]]; then
                    if ! record_command cleanup-remove timeout --signal=TERM --kill-after=2s 8s \
                        docker rm -f "$CONTAINER_ID"; then
                        CLEANUP_FAILED=1
                    fi
                else
                    printf '%s\n' 'Refused removal: exact container name resolved to an ID this run did not create.' \
                        > "$LOG_ROOT/cleanup-remove-refused.log"
                    CLEANUP_FAILED=1
                fi
            fi
        elif [[ "$CONTAINER_NAME_ABSENCE_VERIFIED" == "1" ]]; then
            CLEANUP_FAILED=1
        fi
        if [[ "$CONTAINER_NAME_ABSENCE_VERIFIED" == "1" ]]; then
            if record_command cleanup-final-inventory timeout --signal=TERM --kill-after=2s 8s \
                docker ps -a -q --no-trunc --filter "name=^/$CONTAINER_NAME$"; then
                if [[ -s "$LOG_ROOT/cleanup-final-inventory.stdout.log" ]]; then
                    CLEANUP_FAILED=1
                    DOCKER_INVENTORY_EMPTY=false
                else
                    DOCKER_INVENTORY_EMPTY=true
                fi
            else
                CLEANUP_FAILED=1
                DOCKER_INVENTORY_EMPTY=false
            fi
        else
            DOCKER_INVENTORY_EMPTY=not-applicable
        fi
        if [[ "$NATIVE_INVOKE_STARTED" == "1" && "$NATIVE_RESULT_DONE" != "1" ]]; then
            write_native_result_helper
            if record_command native-result timeout --signal=TERM --kill-after=2s 15s \
                python3.12 -B "$LOG_ROOT/native-result.py" "$NATIVE_STATUS" \
                "$WRITABLE_ROOT/task3-counters.json" "$LOG_ROOT/native-invocation.stdout.log" \
                "$LOG_ROOT/native-invocation.stderr.log" "$RUN_ROOT/native-result.json"; then
                NATIVE_RESULT_DONE=1
            else
                if [[ -s "$RUN_ROOT/native-result.json" ]]; then
                    NATIVE_RESULT_DONE=1
                else
                    CLEANUP_FAILED=1
                fi
            fi
        fi
        if [[ "$SOURCE_GATE_PASSED" == "1" && "$SOURCE_POSTCHECK_DONE" != "1" ]]; then
            if record_command source-postcheck timeout --signal=TERM --kill-after=2s 15s \
                python3.12 -B "$LOG_ROOT/source-check.py" after "$ROOT" "$RUN_ROOT" \
                "$RUNTIME_ROOT" "$NATIVE" "$EXPECTED_INPUT_TREE" "$FREE_BYTES"; then
                SOURCE_POSTCHECK_STATUS=0
            else
                SOURCE_POSTCHECK_STATUS=$?
                CLEANUP_FAILED=1
            fi
            SOURCE_POSTCHECK_DONE=1
        fi
    fi
    if [[ "$CLEANUP_FAILED" == "1" ]]; then
        FINAL_STATUS=2
    fi
    if [[ "$SOURCE_POSTCHECK_STATUS" != "0" ]]; then
        FINAL_STATUS=2
    fi
    write_final_inventory
    return 0
}

write_final_inventory() {
    local status=$FINAL_STATUS
    python3.12 -B - "$RUN_ROOT" "$status" "$DOCKER_INVENTORY_EMPTY" \
        "$CLEANUP_FAILED" "$SOURCE_POSTCHECK_STATUS" <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import os
import sys

run_root = Path(sys.argv[1])
final_status = int(sys.argv[2])
inventory_empty = sys.argv[3]
cleanup_failed = sys.argv[4] == "1"
source_postcheck_status = int(sys.argv[5])
items = {}
for path in sorted(run_root.rglob("*")):
    if path.is_file() and path.name != "evidence-inventory.json":
        relative = path.relative_to(run_root).as_posix()
        items[relative] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
record = {
    "run_id": run_root.name,
    "final_exit_status": final_status,
    "docker_inventory_empty_after_cleanup": inventory_empty,
    "cleanup_failed": cleanup_failed,
    "source_postcheck_exit": source_postcheck_status,
    "closed_file_inventory": items,
    "inventory_file_excluded_from_its_own_hash_list": "evidence-inventory.json",
    "no_more_run_artifact_writers_after_inventory": True,
}
temporary = run_root / ".evidence-inventory.tmp"
temporary.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, run_root / "evidence-inventory.json")
PY
    local write_status=$?
    if [[ "$write_status" -ne 0 ]]; then
        FINAL_STATUS=2
    fi
}

on_exit() {
    local original=$?
    trap - EXIT
    FINAL_STATUS=$original
    cleanup
    exit "$FINAL_STATUS"
}

trap 'exit 130' INT
trap 'exit 143' TERM
trap on_exit EXIT

write_source_check_helper
write_inspection_assertion_helper
write_native_result_helper
printf '%q ' timeout --signal=TERM --kill-after=90s 210s "$0" > "$LOG_ROOT/wrapper-timeout.command.log"
printf '\n' >> "$LOG_ROOT/wrapper-timeout.command.log"

if ! record_command runtime-lock-native-hashes sha256sum "$LOCK" "$NATIVE"; then
    printf '%s\n' 'Could not inspect the qualified CPU lock/native hashes.' >&2
    exit 2
fi
if [[ "$(awk 'NR == 1 { print $1 }' "$LOG_ROOT/runtime-lock-native-hashes.stdout.log")" != "$EXPECTED_LOCK" || \
      "$(awk 'NR == 2 { print $1 }' "$LOG_ROOT/runtime-lock-native-hashes.stdout.log")" != "$EXPECTED_NATIVE" ]]; then
    printf '%s\n' 'Qualified CPU lock/native hash changed; stopping before container creation.' >&2
    exit 2
fi

if ! record_command host-preflight python3.12 -B "$LOG_ROOT/source-check.py" before \
    "$ROOT" "$RUN_ROOT" "$RUNTIME_ROOT" "$NATIVE" "$EXPECTED_INPUT_TREE" "$FREE_BYTES"; then
    printf '%s\n' 'Reviewed source or host provenance preflight failed.' >&2
    exit 2
fi
SOURCE_GATE_PASSED=1

if ! record_command image-inspect docker image inspect --format '{{.Id}}' "$IMAGE"; then
    printf '%s\n' 'Pinned CPU base image is not present locally; no pull will be attempted.' >&2
    exit 2
fi
if ! record_command existing-container-check docker ps -a -q --no-trunc --filter "name=^/$CONTAINER_NAME$"; then
    printf '%s\n' 'Docker inventory preflight failed.' >&2
    exit 2
fi
if [[ -s "$LOG_ROOT/existing-container-check.stdout.log" ]]; then
    printf '%s\n' 'Container name already exists; refusing to reuse it.' >&2
    exit 2
fi
CONTAINER_NAME_ABSENCE_VERIFIED=1

CONTAINER_COMMAND='set -eu
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
export SPIRE_LEARN_TASK3_STARTED_MONOTONIC=$(/work/venv-cpu/bin/python -B -c "import time; print(time.monotonic())")
/work/venv-cpu/bin/python -B -m pip check
/work/venv-cpu/bin/python -B -c "import importlib.metadata as m,json; expected={\"spire-lab\":\"0.1.0\",\"gymnasium\":\"1.4.0\",\"numpy\":\"2.5.4\",\"torch\":\"2.14.1+cpu\"}; actual={k:m.version(k) for k in expected}; print(json.dumps(actual,sort_keys=True)); assert actual==expected"
exec /work/venv-cpu/bin/python -B -m unittest tests.integration.test_neural_envs -v'

if ! record_command container-create docker create --pull=never --name "$CONTAINER_NAME" --rm \
    --user 1000:1000 --read-only --cap-drop ALL --security-opt no-new-privileges \
    --cpus 4 --memory 8g --pids-limit 512 --network none \
    --workdir /learner \
    --tmpfs /tmp:rw,noexec,nosuid,size=64m \
    --mount "type=bind,src=$ROOT/src,dst=/learner/src,readonly" \
    --mount "type=bind,src=$ROOT/tests,dst=/learner/tests,readonly" \
    --mount "type=bind,src=$ROOT/configs,dst=/learner/configs,readonly" \
    --mount "type=bind,src=$VENV,dst=/work/venv-cpu,readonly" \
    --mount "type=bind,src=$WRITABLE_ROOT,dst=/run" \
    --mount "type=bind,src=$WRITABLE_ROOT,dst=/learner/artifacts/development/neural-combat" \
    --env PYTHONPATH=/learner/src \
    --env SPIRE_LEARN_RUN_NEURAL_NATIVE=1 \
    --env SPIRE_LEARN_TASK3_EVIDENCE_PATH=/run/task3-counters.json \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --env PIP_NO_INDEX=1 \
    --env PIP_NO_CACHE_DIR=1 \
    --env PIP_DISABLE_PIP_VERSION_CHECK=1 \
    --env HOME=/run/home \
    --env TMPDIR=/run/tmp \
    --label spire-learn.phase=task3-native-conformance \
    --label "spire-learn.owner=$RUN_ID" \
    --entrypoint /bin/sh "$IMAGE" -ec "$CONTAINER_COMMAND"; then
    printf '%s\n' 'Container create failed; native test command was not started.' >&2
    exit 2
fi
CONTAINER_ID=$(< "$LOG_ROOT/container-create.stdout.log")
if [[ ! "$CONTAINER_ID" =~ ^[0-9a-f]{64}$ ]]; then
    printf '%s\n' 'Container create did not return one full immutable container ID.' >&2
    exit 2
fi

if ! record_command container-inspect docker inspect --format \
    '{"image_id":{{json .Image}},"user":{{json .Config.User}},"workdir":{{json .Config.WorkingDir}},"entrypoint":{{json .Config.Entrypoint}},"cmd":{{json .Config.Cmd}},"env":{{json .Config.Env}},"readonly_rootfs":{{json .HostConfig.ReadonlyRootfs}},"privileged":{{json .HostConfig.Privileged}},"cap_drop":{{json .HostConfig.CapDrop}},"security_opt":{{json .HostConfig.SecurityOpt}},"nano_cpus":{{json .HostConfig.NanoCpus}},"memory":{{json .HostConfig.Memory}},"pids_limit":{{json .HostConfig.PidsLimit}},"network":{{json .HostConfig.NetworkMode}},"mounts":{{json .Mounts}},"tmpfs":{{json .HostConfig.Tmpfs}}}' \
    "$CONTAINER_ID"; then
    printf '%s\n' 'Container control inspection failed; native test command was not started.' >&2
    exit 2
fi
if ! record_command container-inspect-assertion python3.12 -B "$LOG_ROOT/assert-container.py" \
    "$LOG_ROOT/container-inspect.stdout.log" "$LOG_ROOT/image-inspect.stdout.log" \
    "$ROOT" "$VENV" "$WRITABLE_ROOT" "$CONTAINER_COMMAND"; then
    printf '%s\n' 'Container controls differ from the reviewed CPU boundary; native test command was not started.' >&2
    exit 2
fi

NATIVE_INVOKE_STARTED=1
if record_command native-invocation docker start -a "$CONTAINER_ID"; then
    NATIVE_STATUS=0
else
    NATIVE_STATUS=$?
fi
if record_command native-result python3.12 -B "$LOG_ROOT/native-result.py" "$NATIVE_STATUS" \
    "$WRITABLE_ROOT/task3-counters.json" "$LOG_ROOT/native-invocation.stdout.log" \
    "$LOG_ROOT/native-invocation.stderr.log" "$RUN_ROOT/native-result.json"; then
    NATIVE_RESULT_DONE=1
else
    NATIVE_RESULT_DONE=1
    if [[ "$NATIVE_STATUS" == "0" ]]; then
        NATIVE_STATUS=2
    fi
fi

if record_command source-postcheck timeout --signal=TERM --kill-after=2s 15s \
    python3.12 -B "$LOG_ROOT/source-check.py" after "$ROOT" "$RUN_ROOT" \
    "$RUNTIME_ROOT" "$NATIVE" "$EXPECTED_INPUT_TREE" "$FREE_BYTES"; then
    SOURCE_POSTCHECK_STATUS=0
else
    SOURCE_POSTCHECK_STATUS=$?
    if [[ "$NATIVE_STATUS" == "0" ]]; then
        NATIVE_STATUS=2
    fi
fi
SOURCE_POSTCHECK_DONE=1

if [[ "$NATIVE_STATUS" -ne 0 ]]; then
    printf '%s\n' "Native conformance exited $NATIVE_STATUS. Preserve this run; no retry is permitted." >&2
    exit "$NATIVE_STATUS"
fi

printf '%s\n' 'Native conformance passed; EXIT cleanup verifies the named container inventory and closes the artifact inventory.'
