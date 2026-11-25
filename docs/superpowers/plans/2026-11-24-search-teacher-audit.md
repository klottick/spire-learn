# Search Teacher and Audit (Stage 1 of Search Distillation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a public-information tree-search teacher on Gym v1.2.0 with a faster training path and a boss panel. Re-pin to Gym v1.2.3 after Task 8, audit the teacher and run gate G1 on v1.2.3, then stop before labelling.

**Architecture:**
- The teacher is a Learn-side Monte Carlo tree search over the Gym's `PublicSearch` sampled worlds. Its nodes are keyed by public view, and it records a per-action value and visit count at the root.
- It runs from a container entry over the existing Standard panels and a new boss panel.
- An audit script measures self-consistency, strength against budget, root refusals and label coverage. It then applies gate G1 and the teacher-selection rule.
- Two exact training-path speed-ups (trimming padding, sampled validation) land first, because every later stage trains on that path.

**Tech Stack:** Python 3.12, PyTorch (already pinned in the runtime profile), the Spire Gym v1.2.0 Python API (`StandardCombatEnv`, `StandardCombatVectorEnv`, `PublicSearch`), `unittest`, and the offline runtime container (`scripts/neural/runtime_container.py`).

**Spec:** `docs/superpowers/specs/2026-11-21-search-distillation-design.md`. This plan covers the spec's Throughput section, Gym feature audit, Part 1 (the teacher) and Part 2 (audit and panels), plus gate G1 and the teacher-selection rule. Parts 3–5 (dataset, students, PPO arm) and gates G2–G4 get their own plans after the Stage 1 audit.

## Global Constraints

- Tasks 1–8 use Gym v1.2.0: revision `4d198064b3b568b6f0d08d6e3937d236fbc48e9b`, runtime profile `configs/neural/gym-1.2.0/runtime.json`, rules epoch `gym-1.2.0`, Standard rules, Ironclad, ascension 20. The training layout is the compact view, preset `standard-combat-compact-v1`.
- The search uses only the Gym's `PublicSearch` roots and branches (`env.public_search(...)`, `search.branch(k)`, `branch.step`, `branch.fork`, `search.step_many`). It never uses exact checkpoints, forks of the live session, seeds, records or any other privileged surface. Research seeds and search statistics are labels and provenance, never network inputs.
- Networks receive only public observations and legal-action masks.
- Gym faults (`RunFaultError`, `BatchFaultError`, and typed capacity faults) are errors. Tasks 1–8 use the v1.2.0 fault contract. The separate re-pin plan establishes the v1.2.3 fault contract for Tasks 9–10. Faults are never scored as losses and never silently retried. A branch refusal inside a simulation counts against that simulation and is reported.
- Fights cap at 1,024 decisions with no per-case wall-clock cap; a capped fight is censored. The per-turn action budget is `max_turn_actions = 500`: censored in evaluation, truncated in training.
- Selection-split and final-split states are never labelled for training. The final panel stays unscored.
- Search constants (the upper-confidence constant, depth limit, progressive widening and world count) are tuned on training-split fights only.
- The heuristic playout policy is unchanged. Leaf value is the terminal reward (+1 win, −1 death, 0 escape), otherwise half the HP balance, as in `scripts/neural/boss_winnability.py`.
- CPU: at most 24 of 32 cores, at low priority. GPU work uses the `gpu_wait` gate; Learn is second to Hermes. Nothing is committed while a container run is in progress.
- Do not modify any Gym checkout. Read v1.2.0 docs from `artifacts/development/neural-combat/gym-1.2.0/source-ref/`.
- Baalorlord-derived data, run outputs and datasets stay under `artifacts/` and are never committed. Committed docs carry aggregates only, with no run or case IDs.
- Tests are deterministic and CPU-only, on synthetic fixtures. Gym-dependent tests are gated by `SPIRE_LEARN_RUN_NATIVE_INTEGRATION=1` and run in the container by the controller.
- Experiments record:
  - Gym revision and dirty state;
  - recipe, table and panel hashes;
  - seeds and budgets;
  - CPU and GPU time and GPU waits.
- M10 and existing Standard behaviour stay byte-identical where untouched. v1.1.1 configs are unchanged.
- Delivery: one PR per task, reviewed by Claude Sonnet (medium effort) and Gemini 4 Neon Low, merged when clean. Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **A Gym fault inside one simulation of a tree search.** It must abort that decision's search as an error, with every branch and the search handle closed, and never become a −1 value. Tests in Task 6.
2. **A `RunBudgetError` from a `PublicSearch` limit mid-search.** The teacher sizes its limits from its own budget, so hitting one is a Learn defect. It must raise, not return a partial tree. Tests in Task 6.
3. **A fight that begins terminal at reset, or a decision with one legal action.** The first produces zero decisions; the second produces no search and no label, recorded as `single_legal`. Neither crashes. Tests in Task 7.
4. **Two sampled worlds that reach the same public view after different hidden draws.** They must share one tree node. Two different public views must never share one, even if the worlds are identical. Tests in Task 6.
5. **The boss panel's identity.** It must not change the existing selection panel (`eb4a50d8…`, identity `2247fe2b…`), and training-only recipe fields must not change it. Tests in Task 8.

---

### Task 1: Gym v1.2.0 feature audit note

**Files:**
- Create: `docs/neural-gym-1.2.0-feature-audit.md`

**Interfaces:**
- Produces: a list that Tasks 5–7 consult. For each Gym feature relevant to search, labelling, batching or observation size, it names the doc section and says whether Learn uses it. Each entry is marked "used", "used from Task N" or "not used, because …".

- [ ] **Step 1: Read the v1.2.0 docs.** Read from `artifacts/development/neural-combat/gym-1.2.0/source-ref/docs/`, covering at least `standard-run-api.md`, `batching.md`, `combat-environments.md`, `combat-training.md`, `card-metadata.md`, `combat-entry-resources.md`, `performance.md` and `releases/v1.2.0.md`. Also read the v1.2.2 release notes from the Gym PM message in the ledger, for `bound` faults only. Do not adopt anything from v1.2.2.
- [ ] **Step 2: Write the note.** It must cover at least:
  - `step_many` and `workers`;
  - `view="mask"`. The heuristic playout needs action rows, so mask is usable only for playouts that need only the mask;
  - `fork`;
  - compact branch layout via `public_search(..., observation="compact", capacity_preset=...)`;
  - the four `PublicSearch` limits and their hard caps (4,096 live branches, 1,048,576 branches, 100,000 steps per branch, and the total-step cap as `standard-run-api.md` states it);
  - `counters()` and `spec()["root_digest"]`;
  - the root refusal classes (L1–L6);
  - common random numbers across equal `(research_seed, k)`;
  - `card_metadata()`;
  - the combat-entry resource descriptor;
  - vector `reset_mask`.

  End with "Adjustments to the plan". This is empty if the audit finds nothing new; otherwise the controller amends this plan before Task 5.
- [ ] **Step 3: Commit.** `git add docs/neural-gym-1.2.0-feature-audit.md && git commit -m "docs: Gym v1.2.0 feature audit for the search teacher"`

### Task 2: Trim padding in model batches

**Files:**
- Modify: `src/spire_learn/neural/features.py` (`encode_frames` and `EncodedBatch`), `src/spire_learn/neural/model.py` where it consumes row tensors.
- Test: `tests/unit/test_neural_padding_trim.py`, plus a gated test `tests/integration/test_padding_trim_native.py`.

**Interfaces:**
- Produces: `encode_frames(frames, schema, *, device, trim_padding: bool = False) -> EncodedBatch`. With `trim_padding=True`, every row-group tensor is cut to the largest occupied row count in the batch, with a minimum of 1. Coverage is still computed over the view's dense row count, as now.
- Produces: config field `[model] trim_padding` (bool, default false). When absent, config and stream hashes are byte-identical. It is not part of the model, features or objective identity, so checkpoints load either way.

- [ ] **Step 1: Write the failing tests** in `tests/unit/test_neural_padding_trim.py`:
  - `test_trimmed_batch_cuts_rows_to_batch_max_occupied`. Given three synthetic frames whose `combat_cards` occupy 3, 7 and 0 rows, assert `values["combat_cards"].shape[1] == 7`. Assert coverage is equal to the untrimmed coverage.
  - `test_trimmed_and_untrimmed_give_identical_policy_outputs`. For a fresh-weight `spire-learn-combat-ff-standard-v1` model on CPU over 64 synthetic frames, assert that logits, values and sampled and greedy actions are byte-identical (`torch.equal`) with `trim_padding` true and false.
  - `test_trimmed_training_segment_is_byte_identical`. Run one CPU training segment of 2 updates on a scripted driver with both settings. Assert equal parameter tensors after each update, and equal loss values.
  - `test_trim_padding_absent_keeps_config_hash`. Assert that the canonical config hash of an existing Standard template is unchanged when the key is absent.
- [ ] **Step 2: Run them to verify they fail.** Run `PYTHONPATH=src python3.12 -B -m unittest tests.unit.test_neural_padding_trim -v`. Expected: FAIL, because `trim_padding` is an unknown argument.
- [ ] **Step 3: Implement the trim.**
  - Pick the batch maximum from the occupied tensors before transfer.
  - The model must mask or pool only over occupied rows, so outputs are exactly equal. If any model operation depends on the padded row count (for example a mean over rows), change it to the dense-row-count form, as `encode_frames` already does for coverage. Pin that operation with the identical-outputs test.
- [ ] **Step 4: Run the tests to verify they pass.** Same command. Expected: PASS.
- [ ] **Step 5: Add the gated equivalence test.** `tests/integration/test_padding_trim_native.py::test_trimmed_outputs_identical_on_real_v1_2_0_frames`: 2,400 decisions from real v1.2.0 entries of all four characters, in the compact layout, with byte-identical outputs. Run the full suite. The known failure is `test_complete_controller_projects_real_gate_baseline_training_and_time_cutoff`.
- [ ] **Step 6: Commit.** `feat: trim padding in model batches, byte-identical outputs`. The controller runs the gated test in the v1.2.0 container before review.

### Task 3: Sampled validation during training

**Files:**
- Modify: `src/spire_learn/neural/envs.py` (`_row_from_raw` validation), `src/spire_learn/neural/features.py` (`encode_frames` re-validation), `src/spire_learn/neural/runner.py` (`_frame_hash` on records), `src/spire_learn/neural/config.py`.
- Test: `tests/unit/test_neural_sampled_validation.py`.

**Interfaces:**
- Produces: config field `[budgets] validation_stride` (int ≥ 1; absent means 1, which is full validation). It applies only when `operation == "train"`. Evaluation and every test keep full validation.
  - The first observation of every rollout is always validated and hashed. After that, every `validation_stride`-th observation in each slot is.
  - A failed sampled check raises, as now.
  - When absent, config and stream hashes are byte-identical.
- Produces: in training records, `source_public_sha256` and `successor_public_sha256` are `null` for observations that were not sampled. The run-record schema documents the field as nullable in training only. Every reader of those fields (resume, calibration, `standard_results.py`) accepts `null` only in training records. Each reader gets a test.

- [ ] **Step 1: Write the failing tests:**
  - `test_stride_16_validates_first_of_rollout_and_every_16th`. A scripted driver counts `validate_frame` calls. Assert the validated indices per slot are `{0, 16, 32, …}` within each rollout.
  - `test_sampled_validation_raises_on_corrupted_sampled_observation`. Corrupt observation 16. Assert `ContractError`.
  - `test_unsampled_corruption_is_not_checked_but_eval_still_validates_all`. For training with stride 16, corrupting observation 5 is not detected. For `operation="evaluate"` with stride 16 in the config, it raises.
  - `test_validation_stride_absent_keeps_hashes`.
  - `test_resume_accepts_null_hashes_only_in_training_records`.
- [ ] **Step 2: Run them to verify they fail.** Run `PYTHONPATH=src python3.12 -B -m unittest tests.unit.test_neural_sampled_validation -v`. Expected: FAIL.
- [ ] **Step 3: Implement the sampling.** Also remove the duplicate validation: `encode_frames` re-validates frames that `_row_from_raw` already validated. Pass a validated-frame marker so each frame is validated at most once on every path. This is exact on its own, and tests must show evaluation still validates every frame once.
- [ ] **Step 4: Run the tests to verify they pass.** Then run the full suite.
- [ ] **Step 5: Commit.** `feat: sampled validation during training (stride 16), full validation elsewhere`

### Task 4: Throughput profile before and after (controller runs)

**Files:**
- Create: `configs/neural/profile-standard-1.2.0/plan-{16,64}-compact-fast.json` and `template-{16,64}-compact-fast.toml`. These copy the `-compact` pair plus `trim_padding = true` and `validation_stride = 16`, and also exist as `-w8` variants with Gym batch `workers = 8`.
- Modify: `docs/neural-throughput-gym-1.2.0.md`, appending a section.
- Test: extend `tests/unit/test_neural_standard_baseline_configs.py`, or the profile config test if one exists, so the new plans differ from `-compact` only in those keys.

**Interfaces:**
- Consumes: `trim_padding` (Task 2) and `validation_stride` (Task 3).
- Produces: measured decisions per second and the phase split at 16 and 64 rows for: compact, compact-fast, compact-w8 and compact-fast-w8. Later plans choose their training settings from this.

- [ ] **Step 1: Write the configs and the config test.** Run it, and confirm it passes.
- [ ] **Step 2: Profile (controller).** Use `scripts/neural/profile_entry.py` in the container with `gpu_wait`, exactly as `docs/neural-throughput-gym-1.2.0.md` "Method" describes. Run all eight plans back to back. Each run must exit 0 with `learner_changed` false.
- [ ] **Step 3: Append the results.** Add a "Padding trim, sampled validation and workers" section: the same tables as the existing doc, with the speed-up against compact.
- [ ] **Step 4: Commit.** `docs: v1.2.0 throughput with padding trim, sampled validation and workers`

### Task 5: Public search view, batched one-move search and its cost

**Files:**
- Create: `src/spire_learn/neural/search_view.py`. Move `View` (descriptor columns, `legal`, `heuristic_choice`, `balance`) and `refusal_class` out of `scripts/neural/boss_winnability.py` unchanged. `boss_winnability.py` then imports them.
- Create: `src/spire_learn/neural/search_one_move.py`.
- Test: `tests/unit/test_search_one_move.py`, plus a gated test `tests/integration/test_search_one_move_native.py`.

**Interfaces:**
- Produces: `View(descriptor: Mapping)` with the same methods as today, and `refusal_class(error: BaseException) -> str`.
- Produces: `one_move_values(search, legal: Sequence[int], view: View, *, samples: int, playout_steps: int, research_seed: int, branch_view: str = "full") -> dict[int, ActionStat]`.
  - `ActionStat` is a frozen dataclass with `mean: float`, `visits: int` and `refused: int`.
  - It uses common random numbers: sample `k` is `search.branch(k)`, and its playout RNG is `random.Random(research_seed * 1_000_003 + k)`, exactly as today.
  - It steps all `samples × len(legal)` playouts together through `search.step_many`, one lock-step heuristic move per call.
- Produces: `boss_winnability.search_choice` now calls `one_move_values`. Its choice and statistics are unchanged.
- Produces: `step_branches(search, branches, actions, *, view: str) -> list[StepOutcome]`, the shared wrapper over `search.step_many` that Task 6 also uses (adjustment 1 from the Task 1 audit note).
  - Gym sets the worker count when the caller opens the arena.
  - If the whole call is refused for an unsupported boundary, the Gym names no entry. The wrapper then re-issues each entry with `branch.step` to find which entries are refused. Each refused entry is a per-branch refusal, scored −1 for its simulation and reported.
  - Any other whole-call preflight error (`RunInputError`, `RunStateError` or `RunBudgetError`) is a Learn defect and raises.
  - A per-entry fault in the results raises after every handle is closed. Its zero-filled row is never read as an observation.

- [ ] **Step 1: Write the failing tests:**
  - `test_batched_values_equal_sequential_values`. With a fake `PublicSearch` with deterministic scripted worlds, assert `one_move_values` equals the sequential per-action means exactly. The native parity test covers both worker counts.
  - `test_branch_refusal_counts_minus_one_and_is_reported`.
  - `test_fault_in_one_branch_raises_and_closes_all_handles`. The fake records `close()` calls; assert that every branch and the search are closed.
  - `test_whole_call_unsupported_refusal_is_attributed_per_entry`. The fake `step_many` refuses the whole call; assert the wrapper re-steps entries one by one, scores only the refusing entry −1, and reports one refusal.
  - `test_whole_call_input_state_or_budget_error_raises`.
  - `test_search_choice_unchanged_after_refactor`. On recorded fake decisions, assert `boss_winnability.search_choice` returns the same actions and the same `stats` as before the move.
- [ ] **Step 2: Run them to verify they fail,** then implement, then run them to verify they pass.
- [ ] **Step 3: Add the gated parity test.** `test_one_move_batched_matches_sequential_on_real_roots`, on 200 real v1.2.0 training-split roots. It checks dense branch observations, and compact `standard-combat-compact-v1` branch observations against dense, giving identical values.
- [ ] **Step 4: Measure (controller).** Re-run the selection-panel search reference with the batched search, in dense and compact branch layouts and with `workers` 1 and 4, under the Task 4 container settings. Record:
  - wins;
  - CPU seconds and wall seconds per searched decision;
  - the share of time in Python round trips, from profiler spans around `step_many` against the rest.

  Wins must equal the Task 4 reference of 221 wherever values are identical. Append the results to `docs/neural-standard-baseline-gym-1.2.0.md` under "Search reference cost".
- [ ] **Step 5: Post the measurements to the Gym bead `spire-n6b` (controller).** Use `bd -C /home/klottick/repos/spire-lab --sandbox --actor claude-learn-capability-requests-20261121 note spire-n6b --file <note>`, then read it back.
- [ ] **Step 6: Commit.** `feat: batched one-move public search with measured cost`

### Task 6: Tree teacher

**Files:**
- Create: `src/spire_learn/neural/search_teacher.py`.
- Test: `tests/unit/test_search_teacher.py`.

**Interfaces:**
- Consumes: `View` and `refusal_class` (Task 5), and a `PublicSearch`-like object.
- Produces: `TeacherConfig`, a frozen dataclass with:
  - `simulations: int`;
  - `min_root_visits: int = 8`;
  - `worlds: int`;
  - `ucb_c: float`;
  - `max_depth: int`;
  - `widening: tuple[float, float] | None`;
  - `playout_steps: int = 80`;
  - `branch_view: str = "compact"`, selecting the arena layout with preset `standard-combat-compact-v1`. Branch steps request `view="full"` to supply the heuristic with action rows;
  - `workers: int = 1`.

  `config_sha256(TeacherConfig) -> str` is its provenance hash.
- Produces: `teach(search, legal: Sequence[int], view: View, config: TeacherConfig, *, research_seed: int) -> TeacherResult`. `TeacherResult` is a frozen dataclass with:
  - `choice: int`;
  - `stats: dict[int, ActionStat]`, the root mean and visits for every legal action;
  - `simulations: int`;
  - `refused: int`;
  - `cpu_seconds: float`.
- Algorithm (the spec fixes these choices; the rest are the builder's):
  - Each simulation `i` runs in world `k = i mod worlds`, drawn as `search.branch(k)`.
  - Tree nodes are keyed by `sha256` of the public observation bytes plus the legal mask at that node. Children are keyed by `(action, successor public key)`, so chance outcomes that look different publicly lead to different children.
  - Selection uses UCB1 with constant `ucb_c`. Expansion is one node per simulation, up to `max_depth`. The leaf value is the heuristic playout value (terminal reward, else `0.5 * view.balance`) with `playout_steps`.
  - Every legal root action gets at least `min_root_visits` simulations before UCB selection at the root. Successive halving over the remaining root budget is allowed.
  - The root choice is the highest mean, with ties broken by the lowest action index, so the choice is deterministic.
  - Re-descending a node reached in a branch uses `branch.fork()` from the deepest stored fork point, or a fresh `branch(k)` replay. Use the cheaper method from the pre-implementation measurement. Both methods use public calls only.
  - `PublicSearch` limits are sized from `config` with a 10% margin. A `RunBudgetError` raises.

- [ ] **Before tree descent implementation, measure the best case first (user rule, 2026-11-25).** On 50 real training-split roots using compact branches and one worker, measure CPU cost per simulation for `fork()` from a stored node and for creating `branch(k)` and re-stepping the same public action prefix. Compare equivalent node positions and record the costs and the cheaper method in the task report. Implement tree descent with that method. Run labelling fights in parallel processes, with one worker per search by default.
- [ ] **Step 1: Write the failing tests** with a fake search whose worlds are scripted:
  - `test_same_public_view_from_two_worlds_shares_a_node`. Two worlds differ in a hidden draw but produce identical observations after action 0. Assert one child node with visits from both worlds.
  - `test_different_public_views_never_share_a_node`.
  - `test_every_root_action_gets_min_visits`. With `simulations=40`, three legal actions and `min_root_visits=8`, assert every root action has at least 8 visits and the visits sum to 40.
  - `test_choice_is_highest_mean_and_deterministic`. Same seed gives the same result. A different `research_seed` may change `stats`, but not the API shape.
  - `test_branch_refusal_counts_against_that_simulation`. The refused simulation contributes −1 to its path, and `refused` increments.
  - `test_fault_aborts_search_and_closes_handles`. A fake `RunFaultError` in simulation 5 propagates, and every branch and fork is closed.
  - `test_budget_error_raises`. A fake `RunBudgetError` from `branch.step` propagates.
  - `test_no_privileged_surface_is_called`. The fake env's `checkpoint`, `restore`, `fork_session`, `seed`, `record` and `snapshot` attributes raise if touched, and `teach` never touches them. Also assert that `search_teacher.py` source contains none of those attribute names.
  - `test_teacher_is_one_move_search_when_depth_1_and_uniform`. With `max_depth=1`, `worlds=samples` and `simulations = samples × len(legal)`, the root means equal `one_move_values`.
- [ ] **Step 2: Run them to verify they fail,** then implement `teach`, then run them to verify they pass.
- [ ] **Step 3: Run the full suite,** then commit. `feat: public-view tree search teacher`

### Task 7: Teacher fight entry and budget calibration

**Files:**
- Create: `scripts/neural/search_teacher_entry.py`, the container entry.
- Create: `configs/neural/search-teacher-gym-1.2.0/teachers.json`. It holds named teacher configs: `current`, which is one-move with 32 samples and 80 playout steps, and `tree-1x`, `tree-3x` and `tree-10x`. Their simulation counts are filled in Step 4.
- Test: `tests/unit/test_search_teacher_entry.py`.

**Interfaces:**
- Consumes: `teach` and `TeacherConfig` (Task 6), `one_move_values` (Task 5), and `build_panel` / `build_boss_panel` (Task 8 for the boss panel; Step 2 uses the selection panel only).
- Produces the CLI:

  ```
  search_teacher_entry.py --recipe RECIPE --panel {selection,boss} | --train-sample N
      --teacher NAME --teachers FILE --seed S --output DIR
      [--max-steps 1024] [--max-turn-actions 500] [--workers W] [--limit K]
  ```

  It writes:
  - `fights.jsonl`, one row per case: outcome in `victory|death|escaped|censored`, censor reason, steps, `hp_after_cleanup`, CPU seconds and wall seconds;
  - `decisions.jsonl`, one row per decision, with:
    - `case_id`, `step` and the action prefix length;
    - `root` in `searched|single_legal|<refusal class>`;
    - per-action `{action: [mean, visits]}`, `choice`, `cpu_seconds` and `research_seed`;
  - `summary.json`, with wins, the root table and provenance (Gym revision, recipe and table SHA, panel identity, teacher `config_sha256` and seeds).

  `decisions.jsonl` stays under `artifacts/` only.
- Fight rules:
  - Fights reset with Learn's driver construction: `StandardCombatVectorEnv(catalog, 1, partition=<split>, workers=1, autoreset_mode="disabled", fault_policy="freeze")` with the compact preset, and `reset(seed=[case seed], options={"reset_mask": [True], "entry_ids": [id]})`. These are the same seeds the PPO panels use.
  - Censoring at 1,024 steps and at 500 actions in one public turn.
  - Faults are errors.
  - `--train-sample N` draws N train-split level-0 states deterministically from `seed`, for calibration and tuning only.

- [ ] **Step 1: Write the failing tests** with fake envs:
  - `test_terminal_at_reset_fight_has_zero_decisions`.
  - `test_single_legal_decision_has_no_search_and_no_label`.
  - `test_refused_root_records_class_plays_heuristic_and_has_no_label`.
  - `test_turn_cap_censors_at_500_and_step_cap_at_1024`.
  - `test_fault_is_an_error_not_a_loss`.
  - `test_train_sample_uses_train_split_only`. Every sampled state's run is a train run, per `boss_winnability._is_train`.
  - `test_outputs_have_no_hidden_state`. The JSON keys come from an allow-list.
- [ ] **Step 2: Run them to verify they fail,** then implement, then run them to verify they pass. Commit `feat: search teacher fight entry`.
- [ ] **Step 3: Measure the cost unit (controller).** Run `current` on `--train-sample 200` using compact branches (`standard-combat-compact-v1`) and one worker. Record the mean CPU seconds per searched decision. That is the 1× budget.
- [ ] **Step 4: Calibrate and tune (controller, user ruling 2026-11-25).**
  - On `--train-sample 100`, with a different sample seed from Step 3, score the four combinations of `ucb_c` in {1.0, 2.0} and `max_depth` in {4, 8} by train-sample wins for the 3× tree. Hold widening and world count fixed across the grid.
  - Calibrate each tree config on 100 training-split cases too. Set `simulations` so its mean CPU per searched decision is within ±15% of 1×, 3× and 10× the cost unit. The smaller grid and sample reach G1 sooner.
  - Record the grid and the chosen values in `teachers.json` with a `"tuned_on": "train-sample-100 seed …"` note.
  - Commit `configs: calibrated search teacher budgets`, after the runs end.

### Task 8: Boss panel

**Files:**
- Modify: `src/spire_learn/neural/standard_stream.py` (generalize `build_standard_panel`), `src/spire_learn/neural/recipes.py`.
- Modify: `scripts/neural/baseline_entry.py`. Add `--reevaluate` with `--panel boss` to score existing checkpoints on the boss panel, by evaluation only.
- Test: `tests/unit/test_standard_boss_panel.py`.

**Interfaces:**
- Produces: `build_boss_panel(recipe: StandardRecipe, split: str = "selection", *, validate) -> Mapping[str, object]`, exposed through `recipes.build_panel(recipe, split, kind="boss")`.
  - 256 cases: 64 at each of fuzz levels 0, 1, 2 and 4.
  - Per level, 40 boss cases and 24 elite cases. Within each kind, cases are stratified by act in proportion to the available base states, using the existing `allocate_quotas`.
  - Base states are distinct within a level.
  - It has its own `panel_identity(recipe, kind="boss")`, seeded only from that identity.
  - Shortfall rules are as in `build_standard_panel`: a stratum that runs out gives its shortfall to the same kind's other acts.
- Produces: `baseline_entry.py --reevaluate PRIOR --panel boss` writes a summary whose evaluations are scored on the boss panel. Plan-compatibility rules are unchanged.

- [ ] **Step 1: Write the failing tests:**
  - `test_boss_panel_quotas`. Per level: 64 cases, exactly 40 boss and 24 elite, acts proportional within ±1 of the `allocate_quotas` result, and base states distinct within the level.
  - `test_boss_panel_is_deterministic`.
  - `test_selection_panel_identity_unchanged`. On the pinned v1.2.0 recipe fixture, `build_panel(recipe, "selection")` still has identity `2247fe2b…` and its manifest is unchanged.
  - `test_boss_panel_independent_of_training_only_fields`. Changing the recipe's training schedule or fuzz mix for training leaves the boss panel identity and manifest unchanged.
  - `test_reevaluate_boss_panel_scores_on_boss_cases`.
- [ ] **Step 2: Run them to verify they fail,** then implement, then run them to verify they pass. Then run the full suite.
- [ ] **Step 3: Build the panel (controller).** Build it in the container against the live Gym validator, and record its identity and manifest hash in `docs/neural-standard-baseline-gym-1.2.0.md` under "Boss panel". Say whether its cases share base states with the selection panel, and how many.
- [ ] **Step 4: Commit.** `feat: Standard boss panel and boss-panel re-evaluation`

### Gym v1.2.3 re-pin between Tasks 8 and 9

The user ruled on 2026-11-25 to re-pin to Gym v1.2.3 after Task 8. The re-pin
gets its own plan, including how to qualify the runtime and establish the
compatible PPO checkpoints, search reference, recipes and panels. Run Tasks
9–10 on the v1.2.3 pins established by that plan.

### Task 9: Teacher and baseline runs on both panels (controller)

**Files:**
- No code. Run outputs go under `artifacts/development/neural-combat/runs/gym-1.2.3/teacher/`.

**Interfaces:**
- Consumes: Tasks 7 and 8.
- Produces: `fights.jsonl`, `decisions.jsonl` and `summary.json` for:
  - teachers `current`, `tree-1x`, `tree-3x` and `tree-10x` on the selection panel and the boss panel;
  - the PPO baselines r1 and r2, re-evaluated on the boss panel using the v1.2.3 plans and compatible checkpoints established by the re-pin. Keep the turn cap.

- [ ] **Step 1: Launch the runs.** Run all ten as one detached chain: `setsid nohup`, at most 24 cores in total, `nice`, and `gpu_wait` for the PPO evaluations. Each run starts only if the previous one exits 0 with `learner_changed` false. The repo is untouched until the chain ends.
- [ ] **Step 2: Check each output.**
  - Every summary has the expected case count.
  - There are zero faults.
  - Censors are reported by reason.
  - Provenance matches the pins.

### Task 10: Teacher audit, gate G1 and the teacher choice

**Files:**
- Create: `scripts/neural/teacher_audit.py`.
- Create: `docs/neural-search-teacher-gym-1.2.3.md`.
- Test: `tests/unit/test_teacher_audit.py`.

**Interfaces:**
- Consumes: the Task 9 outputs, `standard_results.mcnemar_exact(only_a, only_b) -> float`, and `search_teacher_entry.py` for the replays.
- Produces: `teacher_audit.py consistency --decisions D --sample 2000 --seed S --teacher NAME --output O`.
  - It picks 2,000 searched decisions from the `tree-3x` selection-panel run, deterministically from `S`. The audit set is selection-split and is never training data.
  - It reproduces each decision's state by resetting with the case seed and replaying the recorded action prefix.
  - It re-runs the same teacher with `research_seed + 1`.
  - It reports:
    - top-choice agreement;
    - mean value lost: run A's value of its own choice minus run A's value of run B's choice, with the symmetric figure as well.
- Produces: `teacher_audit.py report --runs DIR --output O`, which reports:
  - strength against budget: wins, mean HP kept and CPU seconds per decision, for each teacher on each panel;
  - root refusals by reason, act and room kind, on both panels;
  - label coverage: the share of decisions with a root, by act and room kind;
  - the stop rule: `boss_root_share < 0.5` in the chosen teacher's own fights gives `STOP`;
  - G1: `tree-1x` against `current` and `tree-3x` against `current`, each an exact McNemar test with censored and escaped cases dropped from the pair. A teacher "beats" `current` if it has more wins and p < 0.025;
  - the teacher choice: the most selection-panel wins among `current`, `tree-1x` and `tree-3x`, with ties going to the cheaper teacher. `tree-10x` is shown on the strength curve only.

- [ ] **Step 1: Write the failing tests** on synthetic run outputs:
  - `test_g1_uses_mcnemar_with_censored_and_escaped_dropped`.
  - `test_beats_requires_more_wins_and_p_below_0_025`.
  - `test_teacher_choice_ties_go_to_cheaper`.
  - `test_stop_rule_triggers_below_half_boss_roots`.
  - `test_consistency_value_lost_is_zero_when_choices_agree`.
  - `test_audit_sample_is_selection_split_and_deterministic`.
- [ ] **Step 2: Run them to verify they fail,** then implement, then run them to verify they pass. Commit `feat: teacher audit and gate G1`.
- [ ] **Step 3: Run the consistency audit (controller),** in the container on the `tree-3x` selection decisions.
- [ ] **Step 4: Write `docs/neural-search-teacher-gym-1.2.3.md`.** It contains:
  - each teacher's config and calibration;
  - the strength-against-budget table for both panels, with PPO r1 and r2 and Baalorlord's level-0 outcomes on the boss panel;
  - the self-consistency ceiling;
  - refusals and coverage;
  - the stop-rule result;
  - G1 with p-values and effect sizes;
  - the chosen teacher;
  - the note that differences below about 4–5 points read as no difference.

  Aggregates only. Commit `docs: search teacher audit and gate G1 on Gym v1.2.3`.
- [ ] **Step 5: Report the results and stop (controller).** Report the stop-rule result, G1 and the chosen teacher to the user. Then stop for the Parts 3–5 plan (dataset, students, PPO arm, gates G2–G4).
