"""Strict dependency doubles for the real Task 9 _run_mode orchestration."""

from __future__ import annotations

from contextlib import ExitStack
from collections.abc import Mapping
from pathlib import Path
import json
from types import SimpleNamespace
from unittest.mock import patch

from spire_learn.neural.collector import CollectorRecovery
from spire_learn.neural.contracts import (
    CollectionDiagnostics,
    CumulativeCounters,
    NeuralIdentity,
    PolicyChoice,
    Rollout,
)
from spire_learn.neural.device import RuntimeDevicePlan


class FakeClock:
    def __init__(self, value: float = 100.0, *, advances_enabled=True):
        self.now = value
        self.advances_enabled = advances_enabled

    def __call__(self):
        return self.now

    def advance(self, seconds: float):
        if self.advances_enabled:
            self.now += seconds


class FakeTensor:
    def __init__(self, value, device="cuda", *, dtype="float32", shape=(1,)):
        self.value = value
        self.device = SimpleNamespace(type=device)
        self.dtype = dtype
        self.shape = shape

    def detach(self):
        return self

    def cpu(self):
        return self.to("cpu")

    def to(self, device):
        return FakeTensor(self.value, device, dtype=self.dtype, shape=self.shape)

    def clone(self):
        return self.to(self.device.type)

    def contiguous(self):
        return self

    def numpy(self):
        return SimpleNamespace(tobytes=lambda: float(self.value).hex().encode("ascii"))


class _Finite:
    def all(self):
        return self

    def item(self):
        return True


class FakeGenerator:
    def __init__(self, device="cpu"):
        self.device = device
        self.state = 0

    def manual_seed(self, seed):
        self.state = int(seed)

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = int(state)


class FakeTorch:
    Generator = FakeGenerator

    @staticmethod
    def is_tensor(value):
        return isinstance(value, FakeTensor)

    @staticmethod
    def equal(left, right):
        if isinstance(left, FakeTensor) and isinstance(right, FakeTensor):
            return left.value == right.value
        return left == right

    @staticmethod
    def isfinite(value):
        del value
        return _Finite()


class FakeAdam:
    def __init__(self, *, device="cuda", state=None):
        self.device = device
        self.state = dict(state or {})
        self.param_groups = [{"lr": 0.001}]
        self.steps = len(self.state)

    def step(self):
        self.steps += 1
        self.state["parameter"] = {
            "step": FakeTensor(self.steps, "cpu", dtype="int64"),
            "exp_avg": FakeTensor(0.25, self.device),
            "exp_avg_sq": FakeTensor(0.5, self.device),
        }

    def state_dict(self):
        return {"state": self.state, "param_groups": self.param_groups}


class FakePolicy:
    def __init__(self, *, device, version=0, state=None, action_offset=0):
        self.device = device
        self._behavior_policy_version = version
        self.action_offset = action_offset
        self._state = state or {
            "body.weight": FakeTensor(0.25, device),
            "candidate_head.weight": FakeTensor(0.5, device),
            "value_head.weight": FakeTensor(0.75, device),
        }

    def state_dict(self):
        return self._state

    def load_state_dict(self, state, strict=True):
        del strict
        self._state = {name: tensor.to(self.device) for name, tensor in state.items()}

    def act(self, frames, *, generator, deterministic=False):
        base = 0 if deterministic else generator.state
        if not deterministic:
            generator.state += len(frames)
        return tuple(
            PolicyChoice(
                base + index + self.action_offset,
                log_probability=-0.25,
                value=0.5,
                behavior_policy_version=self._behavior_policy_version,
            )
            for index, _ in enumerate(frames)
        )

    def change_weights(self):
        for name, replacement in zip(self._state, (0.375, 0.625, 0.875), strict=True):
            self._state[name] = FakeTensor(replacement, self.device)


class FakeDriver:
    def __init__(self, mode, clock):
        self.mode = mode
        self.schema = object()
        self.clock = clock
        self.reset_calls = 0
        self.step_calls = 0
        self.close_calls = 0
        self.close_error = None

    def reset(self, *, seeds, entry_ids, reset_mask):
        del seeds, entry_ids
        self.reset_calls += sum(bool(value) for value in reset_mask)

    def step(self, actions):
        self.step_calls += len(actions)

    def close(self):
        self.close_calls += 1
        self.clock.advance(0.3)
        if self.close_error is not None:
            raise self.close_error


class FakeCollector:
    def __init__(self, driver, policy, config, *, identity, clock, generator, recovery=None):
        self.driver = driver
        self.policy = policy
        self.config = config
        self.identity = identity
        self.clock = clock
        self.policy_generator = generator
        self.budget = None
        self._closed = False
        self._partial_rollout = None
        self._recovery = recovery

    @property
    def partial_rollout(self):
        return self._partial_rollout

    @property
    def recovery_state(self):
        if self._closed:
            raise RuntimeError("closed collector recovery state is unavailable")
        if self._recovery is not None:
            return self._recovery
        return self._current_recovery()

    def _current_recovery(self):
        rows = self.config.rollout_rows
        resets = 36 if self.config.mode == "standalone" else 33
        quotient, remainder = divmod(resets, rows)
        episodes = tuple(quotient + (index < remainder) for index in range(rows))
        steps = 32
        return CollectorRecovery(resets, steps // rows, episodes)

    def collect(self, **kwargs):
        del kwargs
        reset_slots = 36 if self.config.mode == "standalone" else 33
        self.driver.reset(
            seeds=(1,) * reset_slots,
            entry_ids=("entry",) * reset_slots,
            reset_mask=(True,) * reset_slots,
        )
        frames = tuple(SimpleNamespace(public=True) for _ in range(32))
        choices = self.policy.act(
            frames, generator=self.policy_generator,
            deterministic=False,
        )
        self.driver.step(tuple(choice.original_action_index for choice in choices))
        diagnostics = CollectionDiagnostics(
            attempted_environment_slots=32,
            forwarded_environment_slots=32,
            known_accepted_environment_slots=32,
            delivered_environment_slots=32,
            attempted_reset_slots=reset_slots,
            forwarded_reset_slots=reset_slots,
            known_accepted_reset_slots=reset_slots,
            delivered_reset_slots=reset_slots,
            learned_decisions=32,
            next_seed_ordinal=reset_slots,
            stop_reason="fixed_case_budget",
        )
        self._recovery = self._current_recovery()
        self._partial_rollout = Rollout(
            self.config.mode,
            "train",
            self.identity,
            (),
            CumulativeCounters(0, 0, 0, 0, 0, 0),
            diagnostics,
        )
        records = tuple(
            SimpleNamespace(source=frame, actor_training_mask=True)
            for frame in frames
        )
        return SimpleNamespace(records=records, collection=diagnostics)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.driver.close()


class ProjectedHarness:
    """Runs the integration module's real _run_mode with strict fakes below it."""

    def __init__(self, root: Path, integration, *, mode_clock=True):
        from spire_learn.neural import checkpoint, envs, runner

        self.root = Path(root)
        self.integration = integration
        self.checkpoint_module = checkpoint
        self.envs = envs
        self.runner = runner
        self.clock = FakeClock(advances_enabled=mode_clock)
        self.mode_clock = mode_clock
        self.torch = FakeTorch()
        self.checkpoints = {}
        self.loaded_checkpoints = []
        self.runtimes = []
        self.drivers = []
        self.create_calls = []
        self.writer_calls = []
        self.corrupt_prefix = False
        self.corrupt_recovery = False
        self.corrupt_restored_recovery = False
        self.mismatch_choice = False
        self.mismatch_rng = False
        self.mismatch_device = False
        self.mismatch_version = False
        self.close_error = None
        self.save_error = None
        self.train_calls = 0

    def config(self, mode, *, device="cuda:0", operation="train", split="train"):
        from types import SimpleNamespace

        rows, steps = (4, 8) if mode == "standalone" else (1, 32)
        identity = NeuralIdentity(
            model_id="fixture-model",
            features_id="fixture-features",
            action_id="fixture-actions",
            objective_id="fixture-objective",
            accepted_raw_mode_mappings={mode: "fixture-mode"},
            config_hashes={"model": "a" * 64, "ppo": "b" * 64},
        )
        return SimpleNamespace(
            mode=mode,
            operation=operation,
            split=split,
            partition=split,
            rollout_rows=rows if operation == "train" else 1,
            rollout_steps=steps if operation == "train" else 1,
            budgets=SimpleNamespace(
                max_training_decisions=32,
                max_environment_actions=32,
                max_optimizer_steps=4,
                max_wall_seconds=120.0,
            ),
            device=device,
            runtime_gpu_uuid=(self.integration._HELPER.GPU_UUID if device == "cuda:0" else None),
            seed_streams=SimpleNamespace(policy=13, optimizer=17, curriculum=19, noncombat=23),
            config_sha256="c" * 64,
            manifest_sha256="d" * 64,
            environment_revision="fixture-gym-revision",
            accepted_identities=(identity,),
            output_root=self.root / mode,
            cases=object(),
            identity=identity,
        )

    def setup_case(self, mode):
        test = self.integration.NeuralGpuRuntimeProofTests("test_01_standalone_b4_h8")
        test.ledger = self.integration._HELPER.EffectLedger(
            deadline=1000.0, clock=self.clock,
        )
        test.configs = {
            current: self.config(current)
            for current in ("standalone", "run")
        }
        test.torch = self.torch
        test.RuntimeOperationLimits = self.runner.RuntimeOperationLimits
        test.device_identity = {
            "device_name": "strict fake accelerator",
            "device_total_memory_bytes": 1,
            "allocator_fraction": 1.0,
            "cpu_threads": 1,
            "interop_threads": 1,
        }
        test.shared_deadline = 1000.0
        test.started = self.clock.now
        test.create_runtime = self.create_runtime
        test.train = self.train
        test.load_neural_config = self.load_cpu_config
        test._first_mode_failure = None
        return test

    def activate(self):
        stack = ExitStack()
        stack.enter_context(patch.object(self.integration.time, "monotonic", self.clock))
        stack.enter_context(patch.object(self.envs, "make_driver", side_effect=self.make_driver))
        stack.enter_context(patch.object(self.runner, "save_checkpoint", side_effect=self.save_checkpoint))
        stack.enter_context(patch.object(self.checkpoint_module, "load_checkpoint", side_effect=self.load_checkpoint))
        return stack

    def make_driver(self, config, cases):
        del cases
        driver = FakeDriver(config.mode, self.clock)
        driver.close_error = self.close_error
        self.drivers.append(driver)
        return driver

    def load_cpu_config(self, path):
        name = Path(path).name
        mode = "run" if name.startswith("run") else "standalone"
        operation = "evaluate" if "selection" in name else "train"
        split = "selection" if operation == "evaluate" else "train"
        return self.config(mode, device="cpu", operation=operation, split=split)

    def create_runtime(self, config, *, purpose, checkpoint=None, **kwargs):
        self.create_calls.append((config.mode, purpose))
        driver = self.envs.make_driver(config, config.cases)
        saved = self.checkpoints.get(str(Path(checkpoint))) if checkpoint is not None else None
        device = "cuda" if config.device == "cuda:0" else "cpu"
        if purpose == "train":
            policy = FakePolicy(device=device)
            adam = FakeAdam(device=device)
            optimizer = self.runner._CountingOptimizer(adam)
            policy_generator = FakeGenerator("cuda:0")
            shuffle_generator = FakeGenerator("cpu")
            policy_generator.manual_seed(config.seed_streams.policy)
            shuffle_generator.manual_seed(config.seed_streams.optimizer)
            recovery = None
            progress = {}
            version = 0
        else:
            if saved is None:
                raise AssertionError("inert runtime requested without a saved checkpoint")
            policy_state = saved["model_state"]
            if purpose == "resume" and self.mismatch_device:
                policy_state = {
                    name: tensor.to("cpu") for name, tensor in policy_state.items()
                }
            state_device = "cpu" if purpose == "resume" and self.mismatch_device else device
            policy = FakePolicy(
                device=device,
                version=(
                    0 if purpose == "warm_start"
                    else saved["metadata"]["training_state"]["behavior_policy_version"]
                    + int(purpose == "resume" and self.mismatch_version)
                ),
                state={name: tensor.to(state_device) for name, tensor in policy_state.items()},
                action_offset=1 if purpose == "resume" and self.mismatch_choice else 0,
            )
            if purpose in {"resume", "warm_start"}:
                state = saved["optimizer_state"]["state"] if purpose == "resume" else None
                adam = FakeAdam(device=device, state=state)
                optimizer = self.runner._CountingOptimizer(adam)
            else:
                optimizer = None
            policy_generator = FakeGenerator("cuda:0" if device == "cuda" else "cpu")
            shuffle_generator = FakeGenerator("cpu")
            policy_generator.manual_seed(config.seed_streams.policy)
            shuffle_generator.manual_seed(config.seed_streams.optimizer)
            if purpose == "resume":
                policy_generator.set_state(saved["rng_state"]["policy"] + int(self.mismatch_rng))
                shuffle_generator.set_state(saved["rng_state"]["minibatch"])
                recovery_payload = saved["metadata"]["run_metadata"]["collector_recovery"]
                recovery = CollectorRecovery(
                    recovery_payload["next_seed_ordinal"] + int(self.corrupt_restored_recovery),
                    recovery_payload["next_decision_sequence"],
                    tuple(recovery_payload["next_episode_ordinals"]),
                )
                progress = dict(saved["metadata"]["training_state"]["counters"])
            else:
                recovery = None
                progress = {}
        collector = FakeCollector(
            driver, policy, config,
            identity=config.identity,
            clock=self.clock,
            generator=policy_generator,
            recovery=recovery,
        )
        plan = RuntimeDevicePlan(
            "cuda:0", "cuda:0", "cpu", "cuda", self.integration._HELPER.GPU_UUID,
        ) if device == "cuda" else RuntimeDevicePlan("cpu", "cpu", "cpu", "cpu", None)
        runtime = self.runner._NeuralTrainingRuntime(
            config=config,
            collector=collector,
            policy=policy,
            optimizer=optimizer,
            torch=self.torch,
            policy_generator=policy_generator,
            shuffle_generator=shuffle_generator,
            curriculum_rng=__import__("random").Random(config.seed_streams.curriculum),
            controller_rng=__import__("random").Random(config.seed_streams.noncombat),
            device_plan=plan,
            device_report={},
            budget=self.runner._RunnerBudget(120.0, self.clock, None),
            recovery=recovery,
            recovered_progress=progress,
        )

        def fake_update(rollout):
            del rollout
            for _ in range(4):
                runtime.optimizer.step()
            runtime.policy.change_weights()
            runtime.policy._behavior_policy_version += 1
            metrics = SimpleNamespace(
                attempted_steps=4, accepted_steps=4,
                policy_loss=0.0, value_loss=0.0, entropy=0.0,
            )
            runtime.record_update(metrics)
            return metrics

        runtime.update = fake_update
        runtime._fake_saved = saved
        self.runtimes.append(runtime)
        return runtime

    def train(self, config, *, runtime, clock):
        self.train_calls += 1
        runtime.set_budget(self.runner._RunnerBudget(
            config.budgets.max_wall_seconds,
            clock,
            runtime.operation_limits.monotonic_deadline,
        ))
        rollout = runtime.collector.collect()
        runtime.update(rollout)
        run_dir = config.output_root / "fake-run"
        run_dir.mkdir(parents=True, exist_ok=True)
        runtime.save(run_dir / "checkpoint-final", "final")
        diagnostics = rollout.collection
        self.clock.advance(0.1)
        runtime.close()
        elapsed = runtime.budget.elapsed()
        summary = {
            "status": "complete",
            "elapsed_seconds": elapsed,
            "wall_overshoot_seconds": 0.0,
            "lifecycle_receipt_required": True,
            "lifecycle": {
                "status": "complete",
                "exit_code": 0,
                "elapsed_seconds": elapsed,
                "wall_overshoot_seconds": 0.0,
                "close_errors": [],
                "timing_boundary": (
                    "after runtime and artifact writer close; before lifecycle receipt serialization and fsync"
                ),
            },
            "optimizer": {
                "updates": runtime.update_count,
                "accepted_steps": runtime.optimizer_accepted_steps,
                "attempted_steps": runtime.optimizer_attempted_steps,
                "uncertain_steps": runtime.optimizer_uncertain_steps,
            },
            "collection": self.runner._diagnostic_projection(diagnostics),
        }
        return self.runner.NeuralReport(summary, 0, run_dir)

    def save_checkpoint(self, path, *, policy, optimizer, rng_state, metadata):
        self.writer_calls.append(path)
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(metadata, sort_keys=True, allow_nan=False)
        (path / "metadata.json").write_text(encoded + "\n", encoding="utf-8")
        self.checkpoints[str(path)] = {
            "metadata": metadata,
            "model_state": dict(policy.state_dict()),
            "optimizer_state": optimizer.state_dict(),
            "rng_state": dict(rng_state),
        }
        self.clock.advance(0.25)
        if self.save_error is not None:
            raise self.save_error

    def load_checkpoint(self, path, identity, *, purpose, device):
        del identity, device
        saved = self.checkpoints[str(Path(path))]
        metadata = json.loads(json.dumps(saved["metadata"]))
        if self.corrupt_prefix:
            metadata["training_state"]["counters"]["learned_decisions"] += 1
        if self.corrupt_recovery:
            metadata["run_metadata"]["collector_recovery"]["next_seed_ordinal"] += 1
        model_tensors = {
            name: tensor.to("cpu") for name, tensor in saved["model_state"].items()
        }
        training_state = metadata["training_state"]
        saved_inflight = training_state["saved_inflight_record_count"]
        metadata["recovery"] = {
            "purpose": purpose,
            "parent_checkpoint": str(Path(path)),
            "dropped_inflight_record_count": saved_inflight if purpose == "resume" else 0,
        }
        frozen_metadata = self.checkpoint_module._freeze_json(metadata)
        if not isinstance(frozen_metadata, Mapping):
            raise AssertionError("production checkpoint freezer returned non-mapping metadata")
        resumable = purpose == "resume"
        loaded = self.checkpoint_module.LoadedCheckpoint(
            metadata=frozen_metadata,
            model_tensors=model_tensors,
            optimizer_state=saved["optimizer_state"] if resumable else None,
            rng_state=saved["rng_state"] if resumable else None,
            optimizer_step=training_state["optimizer_step"] if resumable else 0,
            inflight_record_count=0,
        )
        self.loaded_checkpoints.append((Path(path), loaded))
        return loaded
