"""Zero-effect fixtures for the Task 4 offline container control assertion."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

from scripts.neural.task4_tensor_inspect import assert_container_inspect, expected_environment


def _fixture():
    root = Path("/learner")
    venv = Path("/qualified/venv-cpu")
    writable = Path("/run-owned/task4-a")
    command = "set -eu\nexec /work/venv-cpu/bin/python -B -m unittest discover -s tests/neural -v"
    reservation_id = "task4-review-a"
    return {
        "image_id": "sha256:local-pinned-image",
        "user": "1000:1000",
        "workdir": "/learner",
        "entrypoint": ["/bin/sh"],
        "cmd": ["-ec", command],
        "readonly_rootfs": True,
        "privileged": False,
        "cap_drop": ["ALL"],
        "security_opt": ["no-new-privileges"],
        "nano_cpus": 4_000_000_000,
        "memory": 8 * 1024**3,
        "pids_limit": 512,
        "network": "none",
        "tmpfs": {"/tmp": "rw,noexec,nosuid,size=64m"},
        "mounts": [
            {"Destination": "/learner/src", "Source": "/learner/src", "Type": "bind", "RW": False},
            {"Destination": "/learner/tests", "Source": "/learner/tests", "Type": "bind", "RW": False},
            {"Destination": "/learner/configs", "Source": "/learner/configs", "Type": "bind", "RW": False},
            {"Destination": "/work/venv-cpu", "Source": "/qualified/venv-cpu", "Type": "bind", "RW": False},
            {"Destination": "/run", "Source": "/run-owned/task4-a", "Type": "bind", "RW": True},
            {"Destination": "/learner/artifacts/development/neural-combat", "Source": "/run-owned/task4-a", "Type": "bind", "RW": True},
        ],
        "env": sorted(expected_environment(reservation_id)),
    }, root, venv, writable, command, reservation_id


class NeuralTensorInspectTests(unittest.TestCase):
    def test_reviewed_offline_container_fixture_is_accepted(self):
        inspect, root, venv, writable, command, reservation_id = _fixture()
        assert_container_inspect(
            inspect,
            image_id="sha256:local-pinned-image",
            root=root,
            venv=venv,
            writable=writable,
            command=command,
            reservation_id=reservation_id,
        )

    def test_changed_access_network_or_command_fails_closed(self):
        inspect, root, venv, writable, command, reservation_id = _fixture()
        cases = []
        changed = copy.deepcopy(inspect)
        changed["mounts"][0]["RW"] = True
        cases.append(changed)
        changed = copy.deepcopy(inspect)
        changed["network"] = "bridge"
        cases.append(changed)
        changed = copy.deepcopy(inspect)
        changed["cmd"] = ["-ec", command + " --unexpected"]
        cases.append(changed)
        changed = copy.deepcopy(inspect)
        changed["entrypoint"] = ["/bin/sh", "-ec"]
        cases.append(changed)
        changed = copy.deepcopy(inspect)
        changed["readonly_rootfs"] = False
        cases.append(changed)
        changed = copy.deepcopy(inspect)
        changed["mounts"].pop()
        cases.append(changed)
        for index, candidate in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(AssertionError):
                assert_container_inspect(
                    candidate,
                    image_id="sha256:local-pinned-image",
                    root=root,
                    venv=venv,
                    writable=writable,
                    command=command,
                    reservation_id=reservation_id,
                )


if __name__ == "__main__":
    unittest.main()
