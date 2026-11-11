# Gym migration on November 2

Learn now supports Gym revision
`afb51f5f222b4c4331ed34067c5eebeda606db91` in both standalone Ironclad
combat and combat inside normal Runs. The approved migration passed a single
released, CPU-only check of the actual Learn adapters. **No gameplay or training
was run.** Historical campaigns and checkpoints retain their original identities.

## What changed

Gym moved categorical vocabularies to its engine-owned registry. Integer IDs
can therefore mean different cards or powers even where array sizes match.
Combat-card and known-draw capacity increased to **1,024 rows**. The common
view keeps its name, but its complete descriptor fingerprint changed.

Learn selects an exact release contract linking the native revision, common
and Run descriptors, and checkpoint raw-mode mappings. Mixed old/new pairs are
rejected before runtime imports. The descriptor-driven feature planner remains
in use; dispatch indices, known/occupied masks and typed row references retain
their meaning. Public `relic`/`power` selection owners and phase/resume-phase
lifecycle metadata are now accepted and validated. In this Ironclad profile,
phases are metadata, not added numeric actor columns.

**Future runs on this release start from fresh weights.** Existing checkpoint
evaluation, warm start and resume are rejected under the new mappings before
Torch loading. Changing the labels on old embedding weights would not convert
them. Old contracts remain available to validate historical configs; that does
not keep the old observation shape frozen for current work.

## Pinned runtime and configs

| Input | Pin |
| --- | --- |
| Wheel SHA256 | `65b5367183d1ab530dbce123e01812749a6def0e7e9f74e8f9afe1f804af75af` |
| Native SHA256 | `1656e912d88b91043f3a1bc7632989638b63bd2b5554b53703d9484658990eac` |
| Common view v1 | `4c73f7de04d7b2875f473f8034c1cbacf984ea2d52d0a9428289c30cf1dff463` |
| Policy Run v3 | `28d92d89c437eea7fdc260240585bca466dee1fd74b9b3eb38513d2b34b0be64` |
| Card metadata v1 | `b60b66c69bce54a6652442e54bc42a2a446d05ec7efa39bb1b11cf4b8c6ff5fb` |

[runtime.json](../configs/neural/gym-20261102/runtime.json) records the new
read-only wheel overlay and existing locked dependency environment. No installer
or dependency resolver ran. New templates are
[standalone.toml](../configs/neural/gym-20261102/standalone.toml) and
[run.toml](../configs/neural/gym-20261102/run.toml). These checks are safe without
the optional runtime and create no experiment output:

```sh
PYTHONPATH=src python3.12 -B -m spire_learn.neural train --config configs/neural/gym-20261102/standalone.toml --validate-only
PYTHONPATH=src python3.12 -B -m spire_learn.neural train --config configs/neural/gym-20261102/run.toml --validate-only
```

The standalone template still selects `ironclad-combat-v1`; normal Run selects
`m10-ironclad-policy`. Both share the new common-view-v1 representation. The
delivered wider standalone v2 encounter profile is not yet admitted by Learn.
Source-only broader Run milestones do not imply a new Python profile.

## Qualification and limits

Final verification passed: **102 focused tests**, default discovery with
**1,046 tests (1,004 passed, 42 gated skips)**, and fresh import isolation across
**56 product modules** with no optional imports. An independent source/control
review and retained-runtime review passed. The protected user files and final
case manifest remain unchanged. Changes are uncommitted on
`feat/gym-contract-20261102`.

Learn matched **17 packet entries, 735 extension source inputs and 23 packaged
Python/stub files** to the accepted clean Gym source. The released container
then checked the imported binary, five descriptor identities, complete common
schema/feature plan, and both actual adapter constructors. It authenticated
**369 public card metadata rows** by the native descriptor's complete hash.
Seven constructed environments each closed once. The owned container was
removed and its final lookup was empty; source, old-package/protected-file and
overlay postchecks all passed. Observed process exit was **0**, with
**1.569 seconds** reported inside the 120-second outer bound.

An earlier unreleased launcher attempt failed at a host import path before
creating any release seal, attempt marker, container or native effect. Its
records remain separate. The corrected entrypoint is regression-tested; the
released native check ran once. The pending runtime-profile copy and reviewed
source snapshot remain retained unchanged. Later documentation, evidence-pointer
updates and the public descriptor test fixture are explicitly subsequent edits.

The [migration report](../artifacts/development/neural-combat/gym-migration-20261102/report.md)
links retained source review, runtime evidence and verification. This is
constructor/schema compatibility evidence. It does not prove gameplay
reliability, GPU performance or learning strength. Gameplay and model
forward/backward checks against this schema belong to the next qualification
stage. Any campaign needs fresh bounds and authority; consumed campaign controls
cannot be reused. RNG-001, TIME-001, finite-content and simulator-only limits
remain in force.

Gym's new [public card metadata](gym-learning-capabilities.md#public-card-metadata-is-now-delivered)
can support future semantic encoders or teachers, but does not supply complete
card-effect text or automatically enable a new learner. Public-information
search branching and validated mid-combat tactical starts remain separate gaps.
