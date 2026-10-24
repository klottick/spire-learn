# Headless combat environments

The current version is **source-grounded, not original-runtime validated**.
It supports supplied first-ready Ironclad A0 combat states, base/upgraded
Strike/Defend/Bash, Burning Blood and one Cultist or ordinary Jaw Worm. No initial
shuffle, draw, enemy-HP sampling or entry-hook execution occurs. Unsupported
content/fields and capacity overflow reject, rather than silently becoming inert.
The required future product remains full runs, keys, Act 4 and the Heart for
**Ironclad, Silent, Defect and Watcher at Ascension 0–20**. No run engine,
learner, learned-strength, original-runtime fidelity or throughput result is claimed.

## Installation and example

Gymnasium is optional; ordinary native APIs do not import it. For a fresh setup
that needs it, use `uv sync --locked --dev --extra gymnasium`, then rebuild with
`uv run --no-sync maturin develop --release --locked --offline`. Sync removes the
editable package; rebuild after sync. Routine edits/tests need **no sync**. See
[development](development.md) for the base toolchain.

Run `.venv/bin/python -B examples/combat_gymnasium.py`. This is a finite masked
caller rollout, not a learner, benchmark, environment cap or full-game example.
Stopping collection does not truncate, reset or score an unfinished episode.

```python
from spire_lab.gymnasium import CombatEnv, CombatVectorEnv

single = CombatEnv()  # manual reset after natural termination
vector = CombatVectorEnv(3, workers=2)  # one native batch, SAME_STEP autoreset
```

Both constructors accept keyword `catalog=None`, `partition="train"`,
`render_mode=None`, `record_replay=False`; vector additionally requires positive
`num_envs` and accepts positive `workers=1`. Partitions are `train`, `selection`,
`final`, `test`.
Effective workers are `min(workers, num_envs)`; one is direct serial iteration,
otherwise one private persistent native Rayon pool. Rendering is absent; only None
is accepted, `render()` returns None, and `close()` is idempotent. Always close
in `finally`. There is no collection of Python Env instances behind the vector.

`catalog=None` selects `starter-cultist-a0-v1` / `starter-jaw-worm-a0-v1`.
A custom catalog is a nonempty list of strict template dictionaries with `id`,
`encounter`, `player_hp`, `player_max_hp`, `energy_per_turn`, `draw_per_turn`,
`enemy_hp`, `hand`, `draw`, and optional `context`. Cards are `{kind, upgraded}`;
context accepts `act` and `next_encounter`, omitted/None for unknowns. See the
[native fixtures](../crates/spire-env/src/tests/fixtures/scenario_catalog_v1.json)
and [binding test support](../tests/python/env/support.py) for exact examples.
Only supported content and validated numeric bounds are accepted.

## Reset, seeds and legal actions

Single: `reset(seed=None, options=None)` returns `(obs, info)`. Seed must be an
exact nonnegative built-in Python integer or None, including integers above 256
bits. Options accepts only `scenario_id` (known string). Vector accepts None,
an exact nonnegative scalar seed, or a length-B list of integer/None entries.
Scalar expansion is native `seed + actual_slot_index`; None continues that slot's
sequence. Explicit reseeding restarts episode ordinals; unseeded reset/autoreset
allocates the next episode. Scenario scheduling/seed derivation/game RNG are Rust
owned, not Gymnasium's compatibility `np_random` or action-space sampling RNG.

Vector options accepts only `scenario_ids` (length-B list of known string/None)
and `reset_mask` (NumPy bool `[B]`, at least one true). Omission selects all rows.
All entries, including unselected ones, validate before mutation; partial reset
requires unselected slots initialized. The base facades never mutate seeds,
options or masks. Unselected rows remain unchanged, with false info presence masks.

Single `step(action)` accepts Python/NumPy integer scalars, not bool, arrays,
timedelta or float. Vector `step(actions)` requires an actual native-endian NumPy
**int64 `[B]`** array, with no coercion from lists/other dtypes. Strided/read-only
inputs are accepted. Candidate indices belong to the current decision, not stable
card identities. Select only from `obs["action_mask"]`; e.g. single:

```python
obs, info = single.reset(seed=7)
action = single.action_space.sample(mask=obs["action_mask"])
obs, reward, terminated, truncated, info = single.step(action)
single.close()
vector.close()
```

Spaces are standard Dict with Box/MultiBinary arrays and Discrete(32), or the
standard batched counterparts. Padding is not legal. Whole-batch invalid input
rejects atomically without an action, reward, RNG advancement or automatic retry.
Native explicit decision tokens reject stale/wrong-pool actions. The facade holds
its latest token privately; a bare integer cannot prove it came from the latest
observation if that index remains legal. Tokens/seeds are not learned features.

## Results, final resources and ownership

Schema `spire-lab-combat-observation-v1` has 26 fixed numeric array fields. Bounds
and vocabularies come from native `observation_spec()` (fresh descriptor and hex
fingerprint), not a second Python feature table. Versioned capacities: hand 10,
enemy 1, previous observed intents 2, candidates 32. Quantities/categories are
int32, observation masks/flags int8. Knowledge includes public composition, not
unseen draw order, RNG keys/cursors or private AI history.

Reward `spire-lab-combat-winloss-v1` is +1 victory, −1 death, 0 otherwise.
There is **no episode cap**; successful built-in transitions always have
`truncated=False`. Single returns float/bool scalars. It preserves terminal obs
and requires reset before stepping again. Vector returns float32 `[B]` rewards,
bool `[B]` endings, and dictionary-of-arrays info with `_key` bool presence masks.
Fields: `episode_id`, `scenario_id`, `episode_steps`, `outcome` (0 ongoing, 1 win,
2 death), and optional `hp_before_healing`, `hp_after_cleanup`.

The vector declares `AutoresetMode.SAME_STEP`. At endings, main obs/info describe
the replacement episode at step zero; reward/ending flags belong to the old one:

```python
# After vector.step(actions), for each terminated slot i:
final_obs = info["final_obs"][i]       # dict of single-row numeric arrays
assert info["_final_obs"][i] and info["_final_info"][i]
old_episode = info["final_info"]["episode_id"][i]
hp_before = info["final_info"]["hp_before_healing"][i]
hp_after = info["final_info"]["hp_after_cleanup"][i]
```

`final_obs` is object `[B]`, None for nonended rows. `final_info` is a nested
dictionary of full-width arrays and per-field masks; outer `_final_info` marks
ended rows. All four final keys are omitted if none end. Preserve final public
resources/context, including terminal killing-card in-play accounting and both
HP checkpoints, rather than using reset resources. This is independent scenario
reset, not run progression or a resource-aware exit-value reward.

All exported arrays are independently owned, C-contiguous and native-endian.
Retained current/final obs/info remain unchanged across steps, explicit resets
and close. Caller mutations cannot change native state or later outputs. Final
rows do not alias replacement storage. Collecting indefinitely still consumes
caller memory; ownership is not a bounded rollout buffer.

## Errors and privileged evidence

`spire_lab._native` defines `CombatInputError(ValueError)` (slot/reason),
`CombatStateError(RuntimeError)` (uninitialized/terminal/busy/closed/faulted),
`CombatSetupError(RuntimeError)`, and `CombatFaultError(RuntimeError)` with ordered
fault summaries. Concurrent/reentrant mutable calls fail Busy. Pre-dispatch input
and setup errors are recoverable. Runtime faults can leave partially advanced
slots: all workers quiesce, the pool faults, ordinary operations reject. There is
no rollback, reset-after-fault, retry or fabricated loss. Post-commit Python
packaging failures also fault before rethrowing the original exception. Construct
a fresh pool only after investigating; do not resubmit an uncertain action.

Facade delivery retains the original native atomic operation lease through all
Python packaging, including final rows and reset compatibility bookkeeping. A
delivery failure marks that call's newly retired records and installed episodes
faulted, never older successfully delivered undrained records. The internal
attached packaging callback runs only after native workers quiesce; it cannot
reenter mutable pool operations. No reusable delivery receipt escapes. Ordinary
raw `NativeCombatPool.reset`/`step_batch` calls still finish delivery at numeric
return; a later `mark_export_fault` call cannot claim their retired records.

`diagnostics()` explicitly exports privileged fault/status/live-token metadata,
not Python core snapshots. With `record_replay=True`, `reproduction()` (single)
or `reproduction(slot)` (vector) exports privileged from-start job/action evidence;
`take_completed_records()` drains owned retired records once. Records contain
seeds/roots, full catalog/schema fingerprints and attempted-fault metadata: never
feed them to a policy. Rust `spire_env::replay` re-executes from the initial job,
not snapshot import. Arbitrary external/injected faults are not magically
reproducible; incomplete/faulted prefixes are not successful completed episodes.
Recording is off by default; opting in grows action traces/retired queues until
drained and leaves memory retention the caller's responsibility. Ordinary stepping
does not snapshot, serialize JSON, perform trace I/O or construct worker pools.

## Verified compatibility subset and limits

Pinned Gymnasium **1.4.0** tests execute upstream `check_reset_seed_determinism`,
`check_reset_options`, `check_reset_return_info_deprecation`,
`check_seed_deprecation`, `check_reset_return_type`, `check_action_space`,
`check_observation_space`, `check_space_limit` for both spaces, and passive reset/
step checks on **valid masked actions**. The fixed Ascension=0 space warning is
expected/asserted. Signature checks and mask-aware deterministic trajectories,
single/serial/worker comparisons, invalid-batch atomicity, ownership and final
resources complement these official checks.

Tested wrappers: single identity `TransformObservation` and single statistics;
vector `RecordEpisodeStatistics` supports SAME_STEP and verifies episode return/
length plus partial-reset isolation. Its wall-clock `episode.t` is excluded from
deterministic comparisons. Upstream vector statistics **pops `reset_mask` from
options**: pass a caller-created copy when retaining options. This upstream mutation
does not weaken the base facade promise. `DictInfoToList` is also tested.

Generic v1.4.0 `VectorObservationWrapper` rejects SAME_STEP with ValueError;
that rejection is tested. No universal wrapper, reflection-helper, SB3 VecEnv or
learner compatibility is promised. Unmodified full `check_env`,
`check_step_determinism` and `check_returned_data_not_reused` sample without masks
and are **not claimed**. No checker/sampling monkeypatch, deterministic metadata
disablement or padded-action acceptance is used. Framework adapters and performance/
learning milestones require separate scope and verification.
