# Sustained Ironclad training campaign

Status: proposed design; no new workload authorized or started.

## Intent and decisions

Move beyond the short plumbing study toward sustained Ironclad combat learning
in both standalone fights and normal Runs. The user selected an **eight-hour
overnight budget** and **warm-starting the best checkpoint in each mode**.
The allocations and implementation details below are the author's proposals
for approving that work together.

Use Standalone B for standalone training and Run B for normal-Run training.
These are development-panel choices, not independently established best policies.
The accepted study used only 43,163 learned decisions overall; all normal-Run
selection trials ended in death. The next campaign should measure improvement,
failure patterns and throughput, without promising full-Run wins.

Approaches considered:

1. Fresh models: useful for a later controlled comparison, but discards the
   progress the user chose to retain.
2. One long process: retains all transition frames and eventually exceeds
   consumed-ID, event-log or summary limits. It also lacks intermediate saves.
3. **Warm-start new experiments, then resume bounded segments:** chosen direction.
   Preserves current guards and saves progress frequently, at the cost of fresh
   environment episodes and startup/checkpoint overhead between segments.

Do not extend the old checkpoints' configurations in place. Exact resume checks
the entire config and manifest hashes. This campaign creates new fixed configs,
loads parent weights with fresh Adam and RNG state, then exactly resumes only
its own checkpoints. Parent artifacts and the sealed final split stay unchanged.

## Scope and unchanged contracts

Use the existing public-view model (16-wide embeddings, 64-wide hidden layers,
265,122 parameters), PPO settings, combat win/loss objective and seeded fixed
noncombat controller. Standalone remains native B16 with 32-step rollouts; Run
remains B1 with 128-step rollouts. No Run-vector facade is invented. No reward,
architecture, simulator, acquisition rule or private actor input changes.

Pin Gym `533dad1c8aec86df624b3742b6b3ad1dbf1d0033`, the accepted common-view
fingerprint `02009686282c592f971eece3d5853eb54f2a482dde9dae1d0cf380ecdaf8fd78`,
and explicit `m10-ironclad-policy` Run v3. Reuse the qualified CUDA environment,
image, native extension and exact NVIDIA CDI profile from completed study B.
The lock SHA is `92022c4b8f68dd7c5bad091b558744ffb30c627292135b7ab18a179f2fd65b45`;
the native binary SHA is
`071069cea66fbf0bc7277002c6b8a21f138194cf692326595109a6741a5178e9`.
All other exact runtime identities must be copied from verified retained
records into the new reviewed manifest, never inferred from current Gym main.

No installs, Gym edits, host Python/driver changes, service stops, pushes or
publication. Work stays in this repository. Preserve the two original user
files and all prior evidence. Source work may use the standing local-commit
authority; experiment outputs remain ignored.

## Time, effects and storage

Treat eight hours as **28,800 seconds of cumulative owned runtime-stage wall
time**, including CPU evaluation and setup/cleanup, rather than active CUDA
kernel time. Source editing and review with no workload running are excluded.
Only one runtime job runs at a time.

| Allocation | Ceiling |
| --- | ---: |
| Profiling, catalog validation, diagnostic traces and runtime verification | 15 minutes |
| Standalone replica 1 | 100 minutes |
| Standalone replica 2 | 100 minutes |
| Run replica 1 | 100 minutes |
| Run replica 2 | 100 minutes |
| Frozen CPU evaluation, cumulatively across batches | 45 minutes |
| Shared staging, controller work and final cleanup | 20 minutes |
| Total | 480 minutes |

Charge each interval once, using outer monotonic receipts. Per-job setup,
checkpointing and cleanup count against its allocation; shared work has its own
allocation. Reserve worst-case cleanup before launching a job. A controller
watchdog bounds each active invocation and the remaining campaign allowance.
Do not count an idle gap between evaluation batches as evaluation time or reuse
an expired absolute deadline. Unused sub-budgets do not enlarge other ceilings.

Each training replica has fixed cumulative caps of 500,000 learned decisions,
1,000,000 environment-slot attempts, 32,768 optimizer attempts, 6,000 seconds,
and at most 256 segments. A segment permits at most 4,096 environment-slot
attempts, eight standalone or 32 Run updates, and 600 seconds including cleanup,
always narrowed by remaining limits. Existing operational update caps are
cumulative, so each segment binds the restored update prefix plus that delta. Reset slots are separately counted and
bounded by attempted actions plus 16 initial slots per admitted segment.
An exhausted allowance stops before forwarding another effect.

The profiling allocation permits at most 8,192 environment-slot attempts,
16,384 actor/diagnostic sample rows and 256 disposable optimizer attempts.
Evaluation permits 704 case executions, each with one initial reset and at most
256 actions: 180,224 action reservations total, zero optimizer steps. Action,
reset, sample and optimizer receipts distinguish attempted, delivered and
uncertain effects. Evaluation's shorter trace-only cases also obey their own
lower caps. No outcome-dependent extension or silent replacement is allowed.

Retain current non-root, network-disabled, read-only-root containment: four
CPUs, four intra-op/one inter-op Torch threads, 8 GiB host memory with the same
combined memory/swap ceiling, 512 PIDs and 8 GiB Torch allocator limit. The latter
is not a hard total-VRAM limit. Require 12 GiB free VRAM on the pinned RTX 5090
before every GPU admission; retain unrelated processes. If unavailable, stop
before effects and report it. Do not automatically stop Ollama.

Allow at most 12 GiB new retained storage, 40 GiB cumulative experiment storage,
and require 30 GiB filesystem headroom. Admit conservatively from measured
growth and stop before exceeding a cap. Retain every published checkpoint and
log; do not prune historical data to make admission pass.

## Measure first

Prior production rates were approximately 19.65 learned decisions/s standalone
and 16.36/s Run. At those rates 400 training minutes would yield roughly 430,000
learned decisions before changed cases and segment overhead; this is a scale
illustration, not a forecast. The prior B16 calibration was a different workload.

Profile real collection and PPO boundaries: native/reset dispatch, public export
and snapshot validation, feature preparation/transfers, policy and bootstrap
evaluation, PPO forward/backward/Adam, hashing/event writes, checkpoint and
finalization. Report inclusive versus exclusive timings and CUDA synchronization
method so asynchronous work is not double-counted. Capture process/cgroup RSS,
retained records/frame bytes, event growth and Torch allocated/reserved peaks.
Current inference is already batched; event writes flush, but do not fsync every
event. Neither is an established bottleneck.

Permit at most one measured, behavior-preserving optimization within batch-local
feature preparation or repeated encoding. It must retain feature meaning,
owned storage, gradients, masks, exact same-backend choices and checkpoint
identity, pass focused correctness checks and a bounded paired timing check.
No long-lived unbounded frame cache. If no clear gain is established, use the
instrumented reference. Disposable profiling updates never seed campaign models.
All optional-runtime verification and paired measurements debit the profiling
budget. Freeze and review the selected source before the first sustained segment;
do not change it between segments. If that budget cannot qualify the path, report incomplete preparation
instead of starting sustained training.

Also record eight diagnostic episodes: old selection cases 00, 04, 01 and 03 on
frozen Standalone A and B, capped at 32 actions each/256 total and 16 MiB total.
Keep only descriptor-defined public player/enemy features, legal candidate
semantics and probabilities, selected log-probability/value, reward and ending
flags, with at most 64 KiB per decision. Hidden RNG, unseen draw order and
private diagnostics remain excluded. This explains examples; it is not a new
held-out evaluation or proof of why Standalone A failed.

## Segment continuity

Add one narrow optional operational environment-action ceiling to
`RuntimeOperationLimits`, defaulting to no additional restriction. It is an
absolute cumulative high-water mark, checked before collection and passed as an
effective collector-local ceiling. Update caps alone cannot bound Run loops
containing only fixed-controller actions. Test this path explicitly.

Each replica's raw config and manifest stay fixed from first warm start onward.
For each segment, mount a fresh host output directory at the **same logical
guest output path**, and mount only its parent checkpoint read-only elsewhere.
Record the host/guest mapping and actual run UUID. This preserves honest config
identity without granting write access to older outputs or replacing dataclass
fields while retaining a stale hash. Staged input trees are read-only and
contain only required public source/config/helper files, never credentials,
`.git`, trackers or historical scratch trees.

After healthy finalization, verify and bind the new checkpoint before another
segment. Resume restores model/Adam, behavior version, generators, cumulative
progress and seed/decision/episode high-water marks. Native environments start
fresh; it does not restore in-flight combats. Report unfinished tasks as
incomplete, not deaths. Restore the saved elapsed prefix exactly once; the
campaign's outer accounting additionally charges serialization, close and
cleanup beyond that save boundary. Never freeze production clocks to force
checkpoint and later lifecycle durations to match.

Round-robin the four replicas at segment boundaries. A failed/ambiguous native,
optimizer, artifact or close effect halts the campaign and retains its prefix.
No retry, fallback checkpoint or next-mode continuation after such failure.
Genuine death is a normal training outcome. Cleanup removes only successfully
created, re-inspected owned container IDs; uncertain creation is reported and
never authorizes removal by name.

## Broader training and frozen evaluation

Create a new standalone catalog: all 13 supported Act 1 setups, the three
existing deck templates, and 64/80 or 80/80 HP: **78 entries**. The setups are
`cultist`, `jaw_worm`, `two_louse`, `small_slimes`, `large_slime`,
`lots_of_slimes`, `three_louse`, `gremlin_nob`, `lagavulin`, `three_sentries`,
`guardian`, `hexaghost`, and `slime_boss`. Retain Burning Blood and empty potion
slots. Use the existing uniform curriculum; log difficult encounter outcomes
separately. Validate all 78 entries against the pinned Gym constructor/reset
and one legal action before training. Do not search for successful seeds.
Run training uses the existing normal-Run controller and public combat routing.

Create two fresh stochastic replicas per mode with distinct policy, optimizer,
curriculum and controller seed quartets. Environment training keeps the accepted
replica-0 ordinal seed domain: disclose matched streams and parental exposure,
not four wholly independent environment populations. Parent checkpoint paths
and exact file inventories are frozen in
`artifacts/development/neural-combat/sustained-v1/preparation/parent-checkpoints.json`.

Before any workload, freeze a new **64-case selection panel**: 32 standalone and
32 normal Runs, using distinct new selection-domain seeds and lineages. The
standalone panel has 16 familiar setup/new-seed cases, eight new deck-composition
cases (starter plus two unupgraded Shrug It Off), and eight new HP-condition
cases (56/80 or 72/80 on familiar decks). Balance these across supported setups
in the authored manifest. These are catalog-relative strata, not claims that
Run policies or parent checkpoints never saw those resources or enemies.

Freeze a 16-case sentry subset (eight standalone: 4/2/2 across those strata;
eight Runs). Evaluate each new replica's first healthy checkpoint after 25%,
50% and 75% of its charged training allowance on that subset, then its final
healthy checkpoint on all 64 cases. If a threshold/final checkpoint is
unavailable, record `not_started`/unavailable rather than inventing a result.
Bind actual checkpoint hashes after publication; do not invent future hashes.

Evaluate the two frozen parent anchors and two references once on the same full
panel. That is 448 new-model plus 256 anchor/reference operations, **704 total**.
Interleave batches at segment boundaries under the cumulative evaluation budget.
No training/hyperparameter/time allocation changes based on interim scores.
References keep their accepted attribution and optional-import isolation.

Report paired standalone victory/HP/actions by stratum, and Run termination,
combat victories per started Run, pre-healing HP and settled-fight counts.
Do not compare policy-dependent pooled fight denominators as independent trials.
Report budget-cutoff cases separately from death. Keep
`configs/cases/neural/final.json` untouched and unexecuted; new panel results
are model-selection evidence. Preserve RNG-001/TIME-001, finite-content and
simulator-versus-original-game qualifications.

## Build, review and authority

Following design approval, write the implementation plan. Use one senior
6.1 Sol/high source executor with Luna/MAX assistants for separate bounded
analysis/review; no competing writers or nested reviewer dispatch by the writer.
Reuse the existing trainer, collector, checkpoint and ownership controls.
Builder choices include exact module boundaries, seed values, balanced case
assignment and timing mechanism; record them before execution.

Required checks include real loaded-checkpoint metadata shapes, projected
container imports and CLI paths, fixed-only Run action ceilings, preserved
checkpoint-boundary timing, stable config hashes across physical output mounts,
exact resume/identity rejection, budget exhaustion before effects, once-only
owned cleanup, and actual producer-to-ledger-to-validator integration. Hand-made
passing result dictionaries are insufficient. Run focused tests, optional-import
isolation and the final default suite; independently review source and controls
before profiling, then review retained qualification before sustained release.

Approval of this design is requested to cover **implementation, review and the
bounded profiling/training/evaluation stages above**. The parent may capture
fresh source hashes and internally release reviewed stages within these limits
without another routine approval question. Broader effects, installs, changed
pins, host-service interruption, retry after accepted-effect failure, increased
budgets or changing the learning problem require a new user decision. Prior
claims stay consumed; new one-use stage records never overwrite them.

The final deliverable is immutable checkpoints, learning curves and paired
evaluation results, measured throughput/memory, complete budget/effect and
cleanup accounting, and an honest recommendation for the next training scale.

## Preparation evidence

- `artifacts/development/neural-combat/sustained-v1/preparation/throughput-audit.md`
- `artifacts/development/neural-combat/sustained-v1/preparation/generalization-audit.md`
- Accepted study: `docs/neural-study.md` and its linked runtime review.
- Design writing and source audits have run no optional runtime or new workload.
