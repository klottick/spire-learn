# Milestone 1: evaluation and artifact foundation design

**Status:** User-approved scoped M1 design. This specification implements the
foundation portion of the approved [master architecture](2026-10-24-spire-learn-architecture-design.md).
It is not authorization to implement, install dependencies, or run integrations.
A detailed implementation plan and review must precede any execution choice.
The approved master is unchanged.

## 1. Purpose and proposed scope

Milestone 1 (M1) establishes reproducible, inspectable evaluation of fixed
combat cases through the public Spire Gym interface. It supplies legal-random
and deliberately simple scripted baselines, learner-owned decision/transition/
case-result contracts, a real but isolated native-vector conformance path, and
an early TorchRL compatibility probe. The milestone is infrastructure and
evidence, not a playing-strength result.

M1 does **not** implement a learned agent, PPO optimizer, GRU training, LLM
server, cloud orchestrator, or VOD pipeline. It does not copy or approximate
the game engine. Gym remains authoritative for observations, actions, rewards,
termination and reset behavior.

### Agreed scope

Fixed-case evaluation scheduling is **single-environment only** in M1. Native
vector integration is nevertheless real: it is used for conformance and
diagnostic transition collection and for the adapter probe. It is not used to
schedule the fixed evaluation manifest. The public `CombatVectorEnv` requires
an action for every row, uses `SAME_STEP` autoreset, and exposes no public
`step_mask`. A batched fixed-case scheduler would therefore need explicit
refill, drain, and unused-episode accounting. M1 will not disguise diagnostic
replacement episodes as scheduled evaluations, nor implement a filler/drain
scheduler. Defer that scheduler explicitly. This is part of the user-approved
M1 scope, not a rejection of native-vector support.

## 2. Context and evidence boundaries

The read-only source inspection recorded Spire Gym at clean `main`, commit
`4874fb398128c142bd56f10a12e6fca939a88f93`, on 2026-10-24. This is source
checkout evidence, not proof of the installed native extension. Python 3.12.15
is the current interpreter. Metadata queries in that interpreter found
`numpy`, `gymnasium`, `torch`, `torchrl`, `pytest`, and `spire-lab` absent; no
borrowed Gym virtual environment is in use. Thus runtime integration gates are
blocked pending separately authorized isolated setup. Absence is not evidence
of incompatibility and does not authorize installation.

The documented public facade is `spire_lab.gymnasium.CombatEnv` and
`CombatVectorEnv`, with public `observation_spec` descriptor. Its documented
starter Ironclad A0 scenarios are `starter-cultist-a0-v1` and
`starter-jaw-worm-a0-v1`. M1 uses `catalog=None`; it neither copies gym fixtures
nor invents a catalog enumeration API. An authored manifest names supported
scenario IDs explicitly. Unknown IDs fail setup rather than being treated as
supported. Documented reset overrides still require an integration check at
the selected installed facade.

The public observation descriptor declares schema identifier
`spire-lab-combat-observation-v1`, a SHA fingerprint, tensor descriptors, named fields,
and vocabularies. Validate its schema identifier and public descriptor at
construction, and always record the actual descriptor fingerprint. If a case
or config declares an expected schema/fingerprint, mismatch is a setup failure
before actions; an initial probe may discover the public fingerprint without
pretending an expected value is already known. Do not recompute Gym's
fingerprint. Learner-owned record/encoder/config fingerprints use canonical
JSON and SHA-256. Consumers read the descriptor rather than hardcoding tensor
counts, shapes, or numeric enum IDs. Action vocabulary includes padding,
`end_turn`, and `play_card`; masks are `int8`; native vector actions are
native-endian `int64[B]`. Public player/candidate printed stats are not
predicted damage or outcomes. There is no general public action-effect text
API, so M1 does not add LLM rendering or hidden feature inputs.

## 3. Component boundaries

The project remains one Python 3.12 package, `spire_learn`, with modules grouped
by responsibility. Exact filenames and signatures belong to the later
implementation plan, not this design.

* Core contracts, configuration, artifacts, and evaluation aggregation use the
  standard library. Gymnasium, NumPy, and native imports are lazy and isolated
  in environment/collection adapters. Ordinary package/core imports do not
  import optional Torch or TorchRL. Gymnasium remains the environment API;
  learner records are not a second generic `Env` abstraction.
* The environment factory constructs only the existing public facade and
  closes it in `finally`. It records source revision/dirty state and available
  public schema. Where public build metadata exists, installed native build
  identity is recorded separately from checkout identity; unavailable identity
  is explicitly unavailable, not presented as proven exact revision. Never
  enter or copy a Gym virtual environment or its artifacts.
* A schema reader resolves public descriptor field names, vocabularies, and
  candidate masks. It rejects unknown/incompatible required schema and
  candidate contracts. Optional future fields do not silently become learned
  features. Learner-owned feature selections and representations are versioned;
  they do not redefine environment semantics.
* Baseline agents implement a decision protocol and own opaque per-stream
  state. Collection routes lifecycle/correlation metadata, while actors receive
  public observations only. The random baseline uses an independent learner
  Python `Random` stream, seeded separately from the environment, and samples
  uniformly from the current legal set. The `play-first` baseline chooses the
  lowest current legal index whose descriptor-resolved kind is `play_card`, or
  the legal `end_turn` action (lowest index on ties). It calculates no damage
  and implements no game rules; label it a crude reference, not a strong
  heuristic. No choice or invalid choice is an explicit failure, never a random
  fallback.
* A terminal observation may have an empty mask and must not be sent for an
  action. A nonterminal observation with no legal candidates is a contract
  error, not a fabricated loss.
* `collection.single` performs one initial reset for the declared case
  (environment seed, scenario, and partition), then processes until natural
  end, configured cap, or error. It does not autoreset into an extra case. It
  validates action/request correlation and the current mask before stepping,
  records the exact delivered transition, and keeps termination and
  truncation distinct. Agent memory advances once per decision; no response
  retry implicitly advances it again. Policy RNG stream initialization/reset
  rules are explicit per execution seed and outside observations. Initialize
  policy RNG once per execution from its manifest policy seed; do not carry its
  RNG state between cases. Seed numeric equality is not itself leakage: require
  independent streams and never derive actor RNG from environment seed.
  Repeatability is expected only within the same recorded revisions/build,
  not across devices.
* `collection.vector` is a native conformance path, not the fixed-case
  evaluator. Each step submits full-batch native-endian `int64` actions for
  every initialized row, including automatic replacements, chosen from each
  row's current public mask; no step mask is implied. It uses a small
  diagnostic cohort. It captures mixed ended and ongoing rows, `final_obs`/
  `final_info` and presence masks; final keys may be absent when no row ends.
  Main reset observations remain distinct from old final observations. Reset
  the relevant slot's memory before any replacement-episode decision. With no
  public park operation, replacement decisions and resulting work are labeled
  diagnostic overhead, never scheduled evaluation cases or training data.
  Stop at a diagnostic budget. Do not expose a fixed-case vector evaluator or
  implement refill/drain scheduling in M1.
* Evaluation owns single-environment fixed-case scheduling, results, and status
  aggregation. Experiment configuration composes factories; artifact
  persistence does not couple to a trainer. Public decisions/traces are kept
  separate from orchestration seeds, case IDs, routing, and privileged
  reproduction data. Replay recording is off by default; no privileged trace
  becomes a feature input.

## 4. Versioned semantic records

The following are semantic version-one contracts, not Python signatures or an
implementation prescription. Stable record schema identifiers are
`spire-learn-decision-v1`, `spire-learn-transition-v1`,
`spire-learn-cases-v1`, and `spire-learn-evaluation-v1`.

### Decision request and result

A decision request contains the current public observation, current legal mask
and candidates, observation-schema fingerprint, and learner action-mapping
identity. Opaque correlation, slot, and episode-start routing metadata is
separate; it is not numeric or text model input. State is opaque per policy;
the runner stores, passes, and resets it. M1 baselines may use `None` state.
CPU recurrence-sentinel fixtures test lifecycle routing without a trained RNN.

A result contains one valid current integer choice, returned opaque state, and
optional diagnostics. It is associated with the exact observation/request so
an asynchronous future cannot apply stale indices. Native tokens are not
introduced as features. Failures are typed and explicit. There is no M1
pre-submit repair/retry. If later introduced, correction must not advance
history twice.

### Transition and boundary semantics

A transition retains the old source public observation and mask, submitted
action, reward, separate `terminated`/`truncated` flags, and old-episode
successor/final observation. For same-step vector autoreset, the replacement
reset observation is a distinct field. Per-slot old recurrent context is
retained for a bootstrappable final observation; replacement memory resets
before replacement processing. True termination suppresses bootstrap under
the selected target contract. A valid truncation contract and old-versus-reset
context arithmetic are tested with fixtures, not a new simulation. A
collection cut preserves memory if execution continues. Fresh checkpoint
recovery is a separately recorded later event.

At each record boundary, raw arrays are independently represented by an owned
snapshot, or an owned original is retained with an actor read-only view, so
policy mutation cannot change a recorded input. M1 may copy rows for
correctness; it makes no zero-copy claim. Versioned disk normalization retains
tensor field/dtype/shape information through schema plus numeric list values.
No private fields, object serialization/pickle, or silent NaN/Infinity
conversion are allowed. Internal numeric arrays do not force core imports of
NumPy. Persisted debug traces are optional; required in-memory transitions are
not optional.

### Case manifest and split identity

Evaluation manifests are JSON with explicit manifest schema/version, split
(`train`, `selection`, or `final`), `case_id`, `scenario_id`, environment
partition, environment seed, repetition/execution identity, and independently
specified policy seed for each scheduled execution. The manifest's partition
is authoritative case identity; config may assert an expected partition but
cannot override it. A mismatch is setup failure before actions. Conflicting
expected schema/fingerprint declarations likewise fail setup; no competing
values have silent precedence. Construction uses the declared case partition.
Harness fixtures are
separate test records, not a fourth experiment split or a mapping to a native
environment partition; native partition behavior is tested only by protocol/
integration tests. M1 diagnostic examples use `selection` and never auto-use
`final`. Validate seed integer semantics as nonnegative where required;
booleans do not count as integers. Preserve the exact accepted Python
environment-seed value: do not silently truncate or coerce. No arbitrary huge
seed rollout is required now. IDs and seeds remain orchestration data, not
actor inputs. Validate actual schema/descriptor at construction; an expected
schema/fingerprint declared by case or config mismatch fails setup before
action, while discovery probes need not pretend an expected fingerprint is
known.

Split validation detects duplicated declared case IDs and overlapping
environment case identities as appropriate; merely sharing a scenario does
not establish leakage. Match cases by scenario, environment seed, partition,
and canonical environment content where supplied. A partition namespace must
not disguise the same underlying case. Record disjoint environment
seed/scenario identities in manifests, but do not claim this proves
distribution independence. Final split use is explicit opt-in. Ordinary smoke
integration does not consume final cases.

### Case result and outcome

One result represents one uniquely scheduled execution. A successful initial
reset marks it started. It records a natural outcome only from a successfully
delivered public terminal transition/info under a verified reward/outcome
contract. Do not infer outcomes from private diagnostics, partial faulted
advancement, or incomplete data. Execution status and game outcome are
separate. Started execution statuses are disjoint: `terminal`, `env_truncated`,
`budget_stopped`, `agent_error`, `simulator_fault`, `contract_error`, and
other explicit `exec_error`. Setup failures/not-started are accounted for
outside started statuses. Record stop reason, steps, reward sum as a
diagnostic, and final public resource fields when present; unavailable values
are null, not zero. A known terminal outcome survives a later close/artifact
failure, which is reported as a secondary failure rather than erasing the
outcome. Primary exceptions are not hidden by close/logging failures.

## 5. Configuration, CLI, and artifacts

Authored configuration is TOML parsed by `tomllib`; CLI parsing uses
`argparse`; manifests and artifacts use JSON/JSONL. No YAML or hosted tracker
dependency is needed. This choice does not prohibit future adapters.

Named smoke, development, and overnight profiles refer to explicit configs.
There is no implicit overnight action or hardware-hardcoded duration. Config
names mode/agent, case manifest, environment constructor kwargs (single M1
evaluation uses `catalog=None` and a partition), explicit positive decision
and time caps, trace option, and output root. The manifest is authoritative for
each execution's environment and policy seeds; config selects the agent,
manifest, partition, and budgets and never silently overrides those seeds.
Reject unknown fields and unsafe coercions. Decision cap applies per
scheduled case; a diagnostic cap applies per diagnostic stream/cohort as
separately labeled. Diagnostic vector cap counts full-batch steps, while
metrics separately count per-slot decisions, avoiding an ambiguous factor of
batch size. The total soft wall-time cap applies to the whole command/run.
Check before scheduling the next reset/step. If total budget expires, record
future planned cases as `not_started` with budget reason. Finish an in-flight
native operation safely and record overshoot. A successfully delivered
terminal or valid environment-truncation result from that operation remains
the case's natural execution outcome even if the step overshoots the deadline.
Mark the current case `budget_stopped` only if it is still ongoing when no
next action is scheduled. Do not interrupt or retry an uncertain effect. This
design chooses no numeric caps; user-approved config supplies values at run
time.

Proposed command surface:

```text
PYTHONPATH=src python3.12 -B -m spire_learn evaluate --config <path>
PYTHONPATH=src python3.12 -B -m spire_learn probe-vector --config <path>
PYTHONPATH=src python3.12 -B -m spire_learn probe-torchrl --config <path>
```

These are proposed M1 contracts, not code or present commands. `--help` and
core configuration validation must work without Gym, Torch, or TorchRL.

Each generated run lives under ignored `runs/<unique-id>/` and includes
`config.json` (resolved TOML), `manifest.json`, `episodes.jsonl`,
`metrics.jsonl`, `summary.json`, and optional `traces/`. M1 creates no
checkpoint infrastructure; it records only compatible shape/record identity
for future use. Preserve existing directories; never overwrite arbitrarily.
Writer failure is a fatal diagnostic, not a reason to retry an environment
action. Flush case results to retain accepted results. Write final summary
atomically within the newly created run directory where feasible. Best-effort
diagnostics must not fabricate success or hide the primary exception. Artifact
schemas contain no secrets.

Record learner revision/unborn-repository state and known dirty state, Gym
source revision/dirty state, separately available installed build metadata,
Python/platform, actually loaded package versions in the isolated setup,
descriptor fingerprint, reward/profile, agent/encoder/config/case identity,
budget, RNG-001/TIME-001 qualifications, and vector diagnostic overhead.
Missing identity or measurements is explicitly unavailable. GPU measurements
are not required for this CPU foundation. Do not copy Gym dependency lists,
lockfiles, virtual environments, or native artifacts. Exact dependency setup
and pins require separately authorized compatibility/setup work. NumPy and
Gym facade runtime are needed for actual native integration. `pytest` is not
selected; standard-library `unittest` suffices for core tests.

## 6. Evaluation accounting and split limits

Report scheduled, started, not-started, and setup-failure totals. Among
started executions, report disjoint execution statuses. Count known game
outcomes without turning faults into losses. Completed-game win rate is
`wins / (wins + deaths)`, null when its denominator is zero. Terminal
completion coverage is terminal started executions / started executions,
also null at zero denominator. Always show raw numbers and incomplete, fault,
and setup counts. Future outcome types need explicit mapping; do not assume
all terminal states are wins or deaths. Distinguish wall-time from step caps.
Single fixed-case evaluation must not count incidental replacement episodes.
Preserve failed attempts; do not auto-retry.

M1 manifests are diagnostic selections and harness fixtures, not a statistical
final benchmark. Support clear split tags and overlap checks; reserve final for
later deliberate, user-approved use. Demonstration contracts in the master
remain future reference only: M1 has no dataset importer or video pipeline.
Split-lineage fixture tests may belong to a later importer milestone; do not
claim an unimplemented importer. The separate one-VOD task is not run here.

## 7. Early framework and native integration probe

TorchRL's `GymWrapper` supports existing Gymnasium `VectorEnv`/`Dict`, but its
`SAME_STEP` info mapping is unverified. An `ActionMask` transform alone
does not establish policy-side training masking. Probe the observed original,
final, and reset observations; mixed ended/ongoing rows; and old/reset
per-slot-sentinel state. Check the public native vector path and, optionally,
TorchRL `MaskedCategorical` sampling/log-probability consistency, without an
optimizer. Compare actual delivered data. Never resubmit an accepted action
to the same environment for comparison; use fresh matched cases if separate
runs are required. Deterministic protocol fixtures are stubs, not game
simulators.

Evidence records the candidate version matrix for Torch, TensorDict, TorchRL,
Gymnasium, NumPy, and Gym build. Docs or version numbers alone do not establish
adoption. Probe verdicts are:

* **compatible** — tested mapping is faithful;
* **adaptable** — a bounded adaptation was actually demonstrated and tested;
* **incompatible** — an observed contract violation cannot safely be
  accommodated; or
* **blocked** — dependencies, authorization, or actual engine are missing.

Blocked is not a successful test. The approved master architecture can remain
in force while M1 integration signoff remains incomplete; after plan approval,
stdlib components may be implemented separately if authorized. The early
probe precedes freezing collection-record contracts. Full M4 PPO/RNN trainer
adoption remains a later gate. Gymnasium source pin 1.4.0 constrains the
initial tested matrix; PyTorch/TorchRL are optional probe dependencies chosen
later from recorded compatibility evidence. Do not install now.

## 8. Verification and acceptance

Core CPU `unittest` coverage includes malformed records/config, bool-as-seed
and bool-as-action, illegal/padded actions, nonterminal all-zero masks versus
terminal empty masks, stale decisions, mutation isolation, status counts and
zero denominators, secondary failures, budget cuts, deterministic policy RNG
independent of environment seeds, per-execution policy RNG reset, selection/
train/final split distinction and overlap, total-budget expiry leaving future
cases `not_started`, terminal/environment-truncated completion at deadline,
per-case and diagnostic budget scope, vector full-batch step versus
slot-decision accounting, manifest/config partition and expected-schema
mismatch failures, artifact overwrite and writer errors,
plus no optional imports for package import/help. Protocol
fixtures cover terminal, truncation, ongoing, and cuts; same-step final data
and missing-mask mismatch; mixed rows; and different old/reset hidden-state
sentinels in bootstrap arithmetic. No game rules are simulated and no Gym
fixtures are copied.

Opt-in actual Gym integration tests both documented scenarios singly, native
vector boundary/action dtype/mask-presence semantics, full-batch step versus
slot-decision accounting, and single/vector equivalence only for matched
inputs and observed supported contracts—not
universal hidden-state exact equivalence. Fault behavior uses unit stubs and
safe public invalid-input recovery; do not manufacture simulator faults via
private hooks. TorchRL evidence is a separate optional CPU integration check
after authority. General tests require no native package or ML framework.

M1 cannot be called complete when integration is skipped or blocked. Acceptance
requires a verified standard-library suite; actual single and vector integration
evidence; reconciled baseline fixed-case artifacts/status counts; and an early
framework verdict of compatible, demonstrated adaptable, or a deliberate,
tested PyTorch fallback. Freeze the collector contract only after evidence.
Reproducibility is scoped to the same recorded revision/build/config; there is
no cross-device bitwise or playing-strength claim. No VOD, LLM weights,
full-run support, GPU training, or distribution services are acceptance items.

## 9. Risks, rationale, and references

* Gym descriptor documentation differs in field counts: consume the live
  descriptor, never a copied constant. A public facade without catalog
  enumeration means authored scenario IDs must be validated at setup.
* Public printed stats do not explain action effects. This is a deliberate
  capability boundary, not a reason to reconstruct rules; readable LLM
  semantics are a later M2 concern.
* Missing dependencies yield a **blocked** runtime probe verdict, not evidence
  of compatibility or incompatibility. They do not prevent offline design and
  unit work after subsequent authority.
* Native vector autoreset prevents honest fixed-case batching without explicit
  refill/drain and incidental-episode accounting. M1’s single scheduler plus
  separately labeled vector diagnostic preserves both useful evaluation and
  real vector conformance without conflating their samples.
* Simulator findings retain RNG-001/TIME-001 qualifications; simulator
  performance is not original-game transfer.

Evidence and context: the [approved master design](2026-10-24-spire-learn-architecture-design.md),
[prior-art research report](../../research/2026-10-24-prior-art.md), and
[copied reference index](../../reference/README.md), especially its
[combat-environment guide](../../reference/spire-gym/docs/combat-environments.md).
Official [TorchRL `GymWrapper` documentation](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.envs.GymWrapper.html)
and [TorchRL `MaskedCategorical` documentation](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.modules.MaskedCategorical.html)
inform the probe targets, not proof that this Gym mapping is tested or
compatible. No missing local artifact is presumed to exist and no external documentation
claim substitutes for actual integration evidence.
