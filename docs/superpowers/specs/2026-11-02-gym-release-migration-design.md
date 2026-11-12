# Learn migration to the November 2 Gym release

Status: **implemented and qualified on 2026-11-02 within the zero-gameplay scope**.
The user has lifted the old observation-shape freeze. This proposal covers
compatibility work and a bounded check with no gameplay, not another campaign.

## Purpose

Use the accepted Gym build `afb51f5f222b4c4331ed34067c5eebeda606db91`
for future standalone Ironclad combat and combat within normal Runs. Preserve
the failed campaigns and their checkpoints as historical evidence. The new
engine-registry vocabularies change the meaning of embedding indices, so an old
checkpoint is not a valid warm start merely because some tensor sizes match.
Fresh weights are the proposed starting point; conversion or distillation from
an old policy would be a separate, measured experiment.

The user selected "Proceed with fresh-weight migration", including the bounded
check with no gameplay. The Luna builder check asked for exact mapping IDs;
they follow the existing deterministic naming convention, with these values:

- standalone: `spire-learn-standalone-common-v1@4c73f7de04d7b2875f473f8034c1cbacf984ea2d52d0a9428289c30cf1dff463`
- run: `spire-learn-run-policy-v3-common-v1@28d92d89c437eea7fdc260240585bca466dee1fd74b9b3eb38513d2b34b0be64:4c73f7de04d7b2875f473f8034c1cbacf984ea2d52d0a9428289c30cf1dff463`

## Verified release

| Artifact or interface | Identity |
| --- | --- |
| Accepted source | `afb51f5f222b4c4331ed34067c5eebeda606db91` |
| CPython 3.12 wheel SHA256 | `65b5367183d1ab530dbce123e01812749a6def0e7e9f74e8f9afe1f804af75af` |
| Native member SHA256 | `1656e912d88b91043f3a1bc7632989638b63bd2b5554b53703d9484658990eac` |
| Common view v1 | `4c73f7de04d7b2875f473f8034c1cbacf984ea2d52d0a9428289c30cf1dff463` |
| Standalone view v2, deferred | `5ba729fd1b10a067fb8fc5d777d78b47fc9526ec0af0030d6868a32b52a7824b` |
| Policy Run observation v3 | `28d92d89c437eea7fdc260240585bca466dee1fd74b9b3eb38513d2b34b0be64` |
| Public card metadata v1 | `b60b66c69bce54a6652442e54bc42a2a446d05ec7efa39bb1b11cf4b8c6ff5fb` |

The durable packet is
`~/.local/share/spire-gym/evidence/builds/afb51f5f/packet/`.
Learn independently matched all **17 packet entries**, **735 tracked extension
inputs**, and **23 packaged Python/stub members** against the clean source at
that revision. Wheel and native-member hashes match the delivery. This read-only
check imported no Gym, NumPy or Torch and executed no native operations.
[Machine-readable assessment](../../../artifacts/development/neural-combat/gym-migration-20261102/assessment.json).

Upstream reports 281 passing Python facade tests and retained replays crossing
the three earlier fault points. Those are upstream results, not Learn gameplay
qualification or proof of long-run reliability. The packet's comparisons are
against build `8137c664`; Learn is upgrading from the older `a86df922`, so its
vocabulary changes must also be included.

## Compatibility changes

The common view keeps its schema name but has different bytes and semantics:
registry-derived categorical IDs, combat-card and known-draw capacities of
1,024 rows, and public phase/resume-phase lifecycle fields. In the inspected
Ironclad profile these phases remain metadata; numeric phase columns belong to
Watcher profiles. The lifecycle can report
selection owners `relic` and `power`, which Learn currently rejects. New
fingerprints must be authenticated from the complete descriptor, never accepted
by checking the schema name alone.

The existing feature planner already derives fields and categorical embeddings
from an authenticated descriptor. Keep that design, including known versus
occupied masks and original dispatch indices. Recheck fixed assumptions in
frame validation, references, candidate decoding, PPO admission and size limits.
Lifecycle/provenance metadata remains separate from actor and critic inputs;
only descriptor-declared public numeric fields enter features.

## Approaches

1. **Explicit release contracts — recommended builder choice.** Add a small
   dependency-free registry binding exact runtime revisions to exact common and
   Run fingerprints and raw-mode mapping IDs. Retain the existing legacy
   contract for historical config validation; add the new contract for new
   configs. Descriptor parsing and driver/PPO admission use the selected
   contract. This preserves reproducibility without freezing the current shape.
2. **Replace every old constant globally.** Simpler initially, but makes old
   configs unreadable and encourages accidental rewrites of historical support
   inventories and experiment templates. Rejected for provenance and maintenance
   reasons, not because the user requires old shapes to remain current.

Neither approach permits arbitrary revisions or fingerprints. Do not create an
"accept any compatible shape" fallback.

## Implementation scope

- Introduce the release-contract lookup and use it in `config.py`, `schema.py`,
  `contracts.py`, `envs.py` and `ppo.py`. Reject mixed old/new revision,
  fingerprint and raw-mode mapping combinations before optional imports.
- Retain the original schema/support provenance. Add a separately identified
  support snapshot if current scenario generation needs it; do not relabel the
  old snapshot as newly validated. Revalidate authored catalog entries against
  the new native constructor before any future training admission.
- Extend validated lifecycle metadata for the public selection owners and
  phase labels. Test that these fields do not leak provenance or hidden state
  into features.
- Keep standalone `ironclad-combat-v1` and normal
  `RunEnv("ironclad", 0, profile="m10-ironclad-policy")` initially, so both
  modes use the same common-view-v1 contract. Wider standalone v2 encounters
  require a separate decision about cross-mode representation and qualification.
- Add new templates and a release profile under a new dated directory. Keep
  old templates, overlays, release claims and campaign artifacts unchanged.
- Use fresh model, optimizer, RNG and progress state. New raw-mode mapping
  identities make existing checkpoint preflight reject old checkpoints before
  Torch loading; add explicit regressions for evaluation, warm start and resume.
- Bind the delivered static card-metadata identity in the new release profile.
  Metadata describes printed cost, target, keywords and values; it does not
  provide complete card-effect text. A new semantic encoder or teacher loss is
  outside this migration.

## Verification and runtime boundary

First run focused pure regressions for release pairing, descriptor body hashes,
capacity/reference bounds, lifecycle values, both factory paths and checkpoint
rejection. Check fresh import isolation and default test discovery with runtime
gates disabled. Old deterministic pure tests must remain useful under their
explicit legacy contract.

For native qualification, stage a new read-only wheel overlay using the
existing locked dependencies, without an installer or dependency resolver.
Use one reviewed, non-root, offline CPU container with a 120-second whole-run
limit, four CPU threads, 8 GiB memory, no GPU, and owned-ID cleanup. The builder
uses the existing transport's exact 8 GiB ceiling rather than modifying shared
historical controls for the initially proposed 4 GiB ceiling. It may
import and verify the package, obtain complete descriptors and card metadata,
and construct/close both Learn adapters. It forwards **zero resets, steps,
samples or optimizer updates**. Qualify the actual parser and factory, not a
stand-in that merely prints expected fingerprints. Retain exact command,
source/package hashes, exit status, cleanup and protected-file postchecks.
Review the concrete wrapper before execution; on failure retain evidence and
stop rather than silently retrying.

This gate establishes release identity and constructor/schema compatibility.
It does not establish gameplay, GPU performance or learning strength. Gameplay
qualification and any future campaign need their own concrete bounds. The
previous campaign remains failed, and consumed claims are never reused.

## Acceptance

Both current-mode configs validate; both real adapter constructors accept the
new exact release in the zero-gameplay gate; old checkpoints are rejected by
the new mapping; focused/default pure checks and import isolation pass; source,
package and preserved-file postchecks reconcile. Documentation must distinguish
this result from the upstream test lane and from later gameplay evidence.
No dependency installation, Gym edits, training, commit or push is included.
RNG-001, TIME-001, finite-content and simulator-only limits remain explicit.
