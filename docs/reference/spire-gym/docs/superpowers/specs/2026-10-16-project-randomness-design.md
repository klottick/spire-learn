# Project randomness under RNG-001

Date: 2026-10-16

**Status: standalone Python randomness foundation implemented and verified on
2026-10-16.** Its subsequent narrow Python combat integration is also implemented
and reviewed; original-runtime validation and Rust parity remain incomplete. See
[repository scope](../../scope.md) for the optimized Rust/learning goals this
foundation supports. The user approved this design and its implementation plan
separately. It
replaces the unapproved exact-original-RNG draft, not the accepted evidence history.
No staging, commits, dependencies, original-game access or simulator edits are
authorized by writing this document.

## 1. Intent and boundaries

Train policies to play the supported game rather than exploit incidental original
seed/stream coupling. Preserve intended probabilities and intentional dependencies,
including draws without replacement and enemy move restrictions, while making
unrelated randomness sources reproducible and independently derived.

This is the intentional deviation
[RNG-001](../../intentional-deviations.md#rng-001--decorrelated-reproducible-project-randomness),
not a claim of stochastic equivalence to the base game. Exact original words,
counter behavior, floating RNG overloads and seed-to-outcome mapping are not goals.
The old original-vector/NaN gates do not apply to this design.

The next implementation slice is the standalone Python randomness foundation:
stream derivation, word generation, unbiased integer/probability/weighted sampling,
shuffle and replay tests. Do not yet change `model.RngState`, `Scenario`,
`CombatState`, content IDs or the existing snapshot schema. Later combat integration
must explicitly assign streams and update the provisional state representation.

No Rust implementation, training loop, scenario sampler, capture adapter or combat
mechanic is included. A generic shuffle primitive does not establish when the game
shuffles, which pile is used or its surrounding card/effect order. Those remain
separate mechanics/evidence requirements.

## 2. Alternatives and recommendation

| Approach | Benefit | Cost / limitation |
|---|---|---|
| **Keyed counter streams using HMAC-SHA256 — recommended** | Standard primitives, small readable reference, explicit derivation, no linear-state RNG imitation, straightforward cross-language bit contract | Hash cost may exceed a dedicated generator; must measure later |
| Dedicated counter generator such as Philox | Fast and naturally parallel; plausible future native throughput choice | More algorithm code/reference work and statistical assumptions; not the simplest first readable slice |
| Per-domain instances of a conventional stateful generator | Familiar sampling APIs | Large/opaque or library-specific state, harder cross-language stability; tempting reliance on defaults |

Do not add multiple backends now. Use the first approach as a clear reference
contract, not as an asserted wall-clock optimum. Later measurements can justify a
new, explicitly versioned contract; silently replacing the algorithm is forbidden.
Python uses ordinary standard-library hashing/HMAC primitives, not custom crypto,
NumPy, tensors, the game JAR or the project's native extension. A future native
implementation may optimize evaluation while preserving this approved contract.

## 3. Version, identity and stream derivation

Export `RNG_CONTRACT_ID = "spire-lab-rng-hmac-sha256-v1"`. This identifies the whole
derivation/word/sampling contract, not just the primitive. Changes to output,
consumption, encoding or existing stream identity require a new contract identity.
Later integrated snapshots/experiments must record it alongside rules/content
identity; this document does not retrofit it into current snapshots.

All data types below belong to `python/spire_lab/combat/rng.py`:

- Frozen `StreamId(domain: str, scope: tuple[str, ...])`.
- Frozen `RandomStreamState(stream_id: StreamId, key: bytes, word_index: int)`.
- `RngExhaustedError`, carrying the latest `RandomStreamState` for diagnostics.

Initial domain registry, with exact scope arity:

| Domain | Scope | Intended future owner |
|---|---|---|
| `combat.shuffle` | `(combat_id,)` | Deck shuffles within one combat |
| `combat.enemy.hp` | `(combat_id, enemy_id)` | Encounter/scenario HP sampling |
| `combat.enemy.ai` | `(combat_id, enemy_id)` | One enemy's stochastic decisions |

These identify project streams, not original stream names or evidence that these
mechanics are integrated. Future random mechanics need an explicit domain/scope
assignment; they must not borrow whichever existing stream is convenient.

Scenario selection, policy/action sampling and optimizer randomness remain outside
combat randomness, with separate randomness namespaces. No learner/scenario API is
introduced in this slice.

`make_stream(root_seed: bytes, partition: str, episode_id: str,
stream_id: StreamId) -> RandomStreamState` is deterministic and consumes no parent
stream. Requirements:

- `root_seed` is exactly 32 bytes; no implicit integer seed coercion.
- `partition` is one of `train`, `selection`, `final`, `test`.
- Episode/combat/enemy IDs are stable semantic IDs, not worker numbers, memory
  addresses, execution order or Python's randomized `hash()` output.
- Every string field is 1–255 printable non-space ASCII bytes (`0x21`–`0x7e`).
  Domains and scope arities must match the registry. Validate before derivation.
- A new sampled episode needs a new identity. Reusing identical derivation inputs
  deliberately reproduces the same stream; the factory does not maintain a global
  uniqueness registry. Episode allocation belongs to later integration.

Canonical encoding is project-owned and fixed. For fields
`(partition, episode_id, domain, *scope)`, encode the field count as unsigned
big-endian32, then each field's byte length as unsigned big-endian32 followed by
its ASCII bytes. This prevents ambiguous concatenations and needs no JSON.

Let `frame` be that encoding. Derive the 32-byte stream key as:

`HMAC-SHA256(key=root_seed, message=b"spire-lab/rng/v1/stream\0" || frame)`.

New state has `word_index=0`. Never initialize all domain streams with the same
unmodified seed/state, derive one stream by advancing another or share their cursor.
Changing HP sampling must not advance AI or shuffle; one enemy's draws must not
advance another's. Dependencies imposed by game rules remain in mechanics state.

## 4. Word sequence and replay state

For a state at word index `p`, let `block = p // 4` and `lane = p % 4`. The block
digest is:

`HMAC-SHA256(key=state.key, message=b"spire-lab/rng/v1/block\0" || BE64(block))`.

Split its 32 bytes into four consecutive eight-byte words, interpreted as unsigned
big-endian64, in lane order 0–3. `draw_u64(state)` returns the selected word and a
new state at `p+1`; the input state is unchanged. There is no native-endian cast,
floating conversion, hidden global cache or mutable generator.

Valid keys are exactly 32 bytes; valid `word_index` values are `0 <= p <= 2**64`.
The upper endpoint is an exhausted sentinel, not a drawable word. A draw there
raises `RngExhaustedError` instead of wrapping or reusing output. The block counter
for every drawable index fits BE64. A later Rust representation may use a wider
checked cursor or an explicit exhaustion flag; its semantic values must agree.

The complete replay state is stream identity, key and next-word index. All fields
are immutable ordinary values. Retaining/deep-copying a value and replaying the same
calls suffices; there is no reseed-by-counter loop or original counter-restoration
API. Any derived block cache added later must be non-semantic and reconstructible.

The reference may recompute a block on each word request. It must not introduce a
global memoization cache to save work. Native block buffering is a later optimization,
not a claim of measured performance or an extra field needed for replay correctness.

## 5. Small sampling API

All operation signatures take state first and return `(value, next_state)`.
Sampling is pure and modifies neither input state nor caller-owned collections.

| Function | Contract |
|---|---|
| `draw_u64(state)` | One unsigned64 word; advances by one |
| `uniform_below(state, bound)` | Uniform integer in `[0, bound)` for `1 <= bound <= 2**64` |
| `bernoulli(state, numerator, denominator)` | Boolean with exact rational probability `numerator/denominator`; `0 <= numerator <= denominator <= 2**64`, denominator positive |
| `weighted_index(state, weights)` | Index chosen by nonnegative integer weights; exact nonempty tuple of ints, positive sum at most `2**64` |
| `shuffle(state, items)` | New list containing the input elements, uniformly permuted; input is a list or tuple and remains unchanged |

Every integer must be an exact Python int, not bool. Invalid types raise authored
`TypeError`; invalid values/unknown domains raise authored `ValueError` before
sampling. These are project input errors, not reproduced Java exceptions. State
fields and stream IDs are validated at construction. No APIs for arbitrary Java
floating bounds, NaN arithmetic or original failure/counter quirks are needed.
Require exact builtin bytes/str/tuple types for owned key/identity/scope fields,
and exact list/tuple inputs for shuffle; do not accept foreign mutable wrappers or
an iterator whose evaluation may hide unrelated state changes. `shuffle` returns
a detached list but preserves the input elements themselves.

**Bounded sampling:** for bound `b > 1`, set
`L = 2**64 - (2**64 mod b)`. Consume words until `x < L`, then return `x mod b`.
Every rejection advances only this stream. This removes modulo bias under uniform
word sampling. For `b=1`, return zero without consuming a word.

**Bernoulli:** endpoints zero and one consume no words. Otherwise compare
`uniform_below(denominator)` with the numerator. Callers must supply the evidenced
probability law; do not guess an enemy probability or silently turn a conditional
move rule into IID sampling. There is no implicit conversion from float thresholds.

**Weighted choice:** validate all weights before any draw, sample below their sum,
and select the first cumulative sum strictly greater than the sampled value. Zero
weights are never selected; a sole unit of weight consumes no word through the
bound-one rule. Ordering is caller-supplied semantic order, not dictionary order.

**Shuffle:** copy the input to a new list, then use descending Fisher–Yates: for
indices `i=n-1` down to `1`, choose `j` below `i+1` and swap positions `i,j`.
Lengths zero/one consume no words. Preserve element identity/multiplicity; do not
deep-copy card instances or draw with replacement. This specifies a project
primitive, not the game's pile orientation or shuffle-call timing.

Exhaustion is an administrative error, never a loss, resample or wrapped stream.
If a multi-draw operation exhausts its stream, raise with its latest exhausted
cursor, without returning a fabricated result or modifying caller inputs. Later
combat integration must retain this diagnostic RNG state and apply the existing
combat fault policy; no normal recovery from exhaustion is introduced here.

## 6. Reproducibility and the information boundary

Stream derivation depends on declared IDs and seed, never scheduling. Assign
episode IDs before parallel dispatch. Worker count, batching, inspection and
tracing must not change a given episode's game draws. Changing an unrelated
stream's usage does not alter another's sequence. Within a stream, changing call
order/consumption intentionally changes its future outputs.

Keep roots, keys, cursors, derivation IDs and debug metadata out of policy inputs,
candidate ordering and other observable side channels. Training/selection/final
partitions have distinct derivation inputs; do not reset every episode to one fixed
stream or let a reused environment slot define an episode seed. Scenario sampling
and learner randomness must not consume these combat streams.

The purpose is robust pseudorandomness and isolation, not a proof of mathematical
independence, cryptographic security of the simulator or impossibility of policy
memorization. Use varied episodes and held-out evaluation as already required.
RNG-001 remains disclosed in training/evaluation and original-game transfer reports.

## 7. Validation without original-RNG parity

All new tests/fixtures are ordinary project-authored or public-standard material;
no private game evidence loader or new original execution is needed. Proposed files:

- `python/spire_lab/combat/rng.py` — one small module for this slice.
- `tests/python/combat/test_rng.py` — derivation, words, domains, input/state/replay.
- `tests/python/combat/test_rng_sampling.py` — bounded/probability/weighted/shuffle.
- `tests/python/fixtures/combat/project_rng_v1.json` — public/project known answers.

Do not add a generic generator backend framework, evidence SDK or optional original
RNG mode. Keep `model.py`, `snapshot.py` and all existing structural tests intact.

Required checks for the concrete implementation plan:

1. Public HMAC-SHA256 known answers from [RFC4231](https://www.rfc-editor.org/rfc/rfc4231),
   plus independently checked project framing/stream/block known answers. Expected
   values must be fixed before product implementation, not generated by the code
   under test. These establish this contract, not original-game correctness.
2. Exact framing bytes, byte order, all four lanes, a block boundary, replay from
   each lane and the final drawable cursor/exhausted sentinel.
3. Unambiguous identity encodings, partition/domain/enemy isolation and arbitrary
   interleavings that leave another stream's outputs/cursor unchanged. Different
   chosen test outputs alone are not a statistical independence proof.
4. Forced rejection and exact consumption; bound-one/endpoints; malformed inputs
   rejected without sampling. Use self-authored test seams, not a runtime plugin API.
5. Small exhaustive combinatorial checks for weighted choice and all shuffle paths
   on a three-element input, plus preservation of duplicate elements/input ownership.
6. Fixed project-fixture comparisons, repeat/restore determinism, structural-suite
   regressions and independent code review. No skipped private-oracle gate exists.

Statistical smoke tests can supplement these checks, but cannot prove independence
or replace the rejection/Fisher–Yates reasoning. Test results are not throughput
measurements. Profile later in the actual native training workload before choosing
a more complex primitive solely for speed.

Mechanics validation remains separately grounded in original evidence. Where
feasible, compare mechanics under matched supplied random outcomes, not matching
base-game seeds. A future adapter for this needs its own explicit plan; do not
silently alter frozen trace/import contracts or claim complete original validation.

## 8. Handoff

User review of this written design precedes the concrete TDD implementation plan.
That plan must pin project/public known answers, exact test names/assertions and
red/green commands, then obtain the applicable execution approval. No additional
original-RNG semantic review or raw-vector transport is a prerequisite for this
project-owned contract. All accepted old evidence and failed attempts stay intact.

Implementation plan: [`2026-10-16-project-randomness.md`](../plans/2026-10-16-project-randomness.md).
The user approved execution on 2026-10-16. All three standalone tasks and their
fresh reviews are complete; final controller verification passed 395 tests.
`model.RngState` and existing combat snapshots remain unchanged. These checks do
not establish original-game mechanics correctness, Rust parity or performance.
