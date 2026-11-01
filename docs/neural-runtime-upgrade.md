# Gym native-fault release: 2026-11-01

Learn has staged and checked the Gym delivery from
[`a86df922`](https://github.com/klottick/spire-gym/pull/67).
The delivered wheel SHA256 is `06a008cced5e7dcd61c9c36085190f9160a3a01368c35e46f72b6b0de26ce2be`;
the imported native binary is `96d54621aa533f27ca720621dfa17664609e51e5b9311fed1636cffa8cd9d787`.

Gym's diagnosis distinguishes two defects:

- `spire-tsw`: Infernal Blade could generate Clash without a defined starting
  cost, faulting normal Run play. Its cost definition was fixed and pinned by
  a regression.
- `spire-bhl`: Thorns/Bronze Scales could kill or half-down a multi-hit attacker
  while its remaining hits stayed queued. The engine now drops those queued
  hits for dead or half-dead attackers.

Gym's retained imported-extension replays crossed both former fault points
without a fault and reported no prefix divergences. Its facade lane recorded
279 passing tests. These are upstream results, not replays performed by Learn.
The delivery and replay evidence is retained independently of managed worktrees
under `~/.local/share/spire-gym/evidence/native-faults-delivery-20261101/packet-v1/`.

Learn independently verified all 16 delivery-manifest entries and 706 extension
source inputs against the accepted export. A protected non-root CPU container
then imported the new wheel contents, checked all five descriptor fingerprints
and constructed/closed both actual Learn adapters. It forwarded zero resets,
steps, samples or optimizer updates. No GPU was requested. The container was
removed by its owned full ID and its subsequent lookup was empty. This is an
adoption/identity check, not a learning or long-run reliability claim.

The wheel is deployed as a separate read-only package overlay. The existing
qualified dependency venv and old Gym package are unchanged; no installer or
dependency resolver was run. [runtime.json](../configs/neural/native-fixes-20261101/runtime.json)
records the exact artifacts and dependency lock. New templates are
[standalone.toml](../configs/neural/native-fixes-20261101/standalone.toml) and
[run.toml](../configs/neural/native-fixes-20261101/run.toml). The overlay must
precede the dependency venv on the guest Python path. Match the selected config's
exact revision to `build_info()`; a schema match alone never authorizes an
arbitrary runtime build.

Historical configs, schema/support provenance and consumed campaign releases
retain their original identities. The loader now accepts these two exact
runtime revisions; the original schema provenance constant is unchanged.
Historical one-use host controllers still bind their original binary and cannot
be reused for this release. Any new campaign needs fresh source/runtime pins,
output identities and execution authority. This update did not restart training.

Old weights may be considered through the explicit warm-start path with fresh
optimizer/RNG/progress and recorded lineage. Exact resume still rejects a change
of config/environment revision. Neither old checkpoint metadata nor the failed
campaign's uncertainty is rewritten to make a resume pass. The separate Learn
fault-evidence capture issue `spire_learn-pa7` remains open.

See the [Learn qualification and refresh report](../artifacts/development/neural-combat/gym-refresh-20261101/report.md)
for retained commands, tests, source changes, preservation and limits. RNG-001,
TIME-001, finite content and simulator-only qualifications remain in force.

## Subsequent campaign result

The separately approved campaign `curriculum-20261101-c` used this exact overlay
and passed its native/CUDA qualification. It then stopped on another standalone
native fault after six healthy training segments, including two randomized
checkpoints. Gym issue `spire-14g` tracks this new incident; its internal cause
and relationship to earlier bugs remain unknown. The [campaign results](neural-curriculum.md#fixed-gym-campaign-c)
retain the saved work, incomplete evaluation and reviewed failure evidence.
Training is stopped with no retry. The adoption check and earlier upstream
replays remain historical evidence; they do not establish long-run reliability.
