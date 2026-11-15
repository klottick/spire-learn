# Spire Gym 1.0.0 and Standard adoption

Updated **2026-11-15** (current build `8726d467`; see the re-pin section). Originally written 2026-11-12 for build `1929e18b`. Spire Learn now selects Gym by the configured revision
and supports the Gym 1.0.0 common, Standard combat and Standard Run contracts.
The native extension revision is
`1929e18b9f44372e020877bd62f2a5c9dee3e64e` (tag `v1.0.0` is `fc3c0118`,
documentation/data only). The release build identity is `release`, clean, with
feature `spire-python/extension`, source digest
`d4b3121226cfbb030b89731ed7ea3773f5c2b44988c9ec01c2355d53f2dfa9b2` and file
digest `dce682976f221f5686d16beda7ced03f0221624da0a9b83e8b5969f7300f60ce`.

## Re-pin to build 8726d467 (2026-11-15)

Learn's current templates are `configs/neural/gym-8726d467/`. They pin Gym
revision `8726d467ad8579730ddf4fa236b24aeb2206b71a`, still version 1.0.0, from
Gym PR 112. The `1929e18b` contract and its `configs/neural/gym-v1.0.0/`
templates stay valid for that revision.

| Item | Value |
| --- | --- |
| Wheel SHA-256 | `96321606aa43ef73d50af4e6ed8d401aee26b88bc1044b16132935fd9c379fb7` |
| `_native` SHA-256 | `7019351538ae567abc2ffd7f40cc98d5c83419065685fcae12e13dd83867c86d` |
| `source_digest` / `file_digest` | `73f554e2a9a2cb20c23efff0daba83e4ff6a6bcd90e0fbd33b48f8534bce1d7a` / `98db0a9ac92e05c45424fb5658485ed16ee92a0cfcd49063728c2d9a905a6ccf` |
| Rules IDs | Standard `spire-gym-native-standard-rules-v5`; M10/Gym66 `spire-gym-native-m10-rules-v2` (`1929e18b`: `v3` / `v1`) |
| Fingerprints | Unchanged: every descriptor is byte-identical to the `1929e18b` fixtures |
| `rules_epoch` | `gym-1.0.0+8726d467` |

**Learn changes.** Each `GymContract` now carries its own `m10_rules_id` and
`standard_rules_id`. The drivers and run metadata read these from the
configured revision instead of hard-coded literals. A config therefore always
expects its own release's rules IDs, and a mismatched build fails closed.
`scripts/neural/qualify_gym_v1.py` and `soak_gym_v1.py` take `--profile-dir`,
defaulting to the current templates.

**Gym changes in this build** (see the packet's `DELIVERY.md`):
- **Speed.** Batch steps are faster, with byte-identical observations.
- **Compact observations.** `observation="compact"` is new and opt-in. Learn
  does not use it yet: its parser assumes the dense int64 layout.
- **Former fault paths fixed** (spire-3xq, Standard rules v4):
  - Berserk grants stacks above four;
  - queued Fire Breathing hits keep their damage;
  - post-victory cleanup opens no player choice;
  - a choice between a finished card and its queued repeat resolves before the
    repeat;
  - the 999 generated-card bound reports a typed `Unsupported`;
  - inconsistent offered choices keep an `Invariant` fault.
- **Shops** (spire-0kr, Standard rules v5). The Courier restocks each bought
  row in place, and Circlet costs 400 in a shop.

Rules changed while fingerprints did not. The new `rules_epoch` therefore
rejects `1929e18b` checkpoints and mappings, and vice versa. Results from the
two builds are not interchangeable.

**Verification** (2026-11-15). All runs used the offline runtime container
described below.
- **Packet.** All 84 `MANIFEST.sha256` entries verify. `build_info()` matches
  the table above, with `dirty=false`.
- **Descriptors.** Captured descriptors and the Standard catalog are
  byte-identical to the `1929e18b` fixtures. The Standard environment specs
  differ only in `rules_id`.
- **Qualification.** `qualify_gym_v1.py` reports `qualified_one_reset_per_mode`:
  four adapters, four resets, zero steps and one close each.
- **Soak.** `soak_gym_v1.py --steps 3000 --characters` passed with no contract
  fault: 3,000 steps in each of the four modes, plus 3,000-step Standard Runs
  for every character at A0 and A20.
  - The counts equal the `1929e18b` soak's under the same seeds, so these
    random rollouts did not reach the changed gameplay paths.
  - The soak therefore establishes adapter acceptance of the new build, not
    coverage of those fixes.

Outputs are under `artifacts/development/neural-combat/gym-8726d467/`, which
Git ignores.

## Contract identities

The wheel and native extension hashes below identify the local release packet;
the view fingerprints identify the public schemas Learn validates.

| Item | Gym 1.0.0 identity |
| --- | --- |
| Native build revision | `1929e18b9f44372e020877bd62f2a5c9dee3e64e` |
| Tag | `v1.0.0` = `fc3c0118` (docs/data only) |
| Build state / profile / features | clean / `release` / `spire-python/extension` |
| Source digest | `d4b3121226cfbb030b89731ed7ea3773f5c2b44988c9ec01c2355d53f2dfa9b2` |
| File digest | `dce682976f221f5686d16beda7ced03f0221624da0a9b83e8b5969f7300f60ce` |
| Wheel SHA-256 | `d157065d9389c92094bef34fc3a58024dd7c2632486dd60a7b4a9f6ace4774ad` |
| `_native` extension SHA-256 | `5398d2fd591e025f84435efb4d80909995a764303ce0bc6d87d74c68d4be88cf` |
| Common combat view v1 | `4c73f7de04d7b2875f473f8034c1cbacf984ea2d52d0a9428289c30cf1dff463` |
| Policy Run v3 | `28d92d89c437eea7fdc260240585bca466dee1fd74b9b3eb38513d2b34b0be64` |
| Card metadata v1 | `effbb810d345e906f6e20b47bd972e35d72ef48141378d54bd28b54afac5f406` |
| Standard combat view v1 | `5d8147e13ded030c7faa501a8202eb2588915c2e1330b30cac5b06afc9f9916e` |
| Standard Run observation v1 | `bf8e1d111bbf2171594231b99d8d6b701244249abfbb90a11e86c4b4a405f7ae` |

These values are the verified release identity recorded in the
[Gym 1.0.0 design](superpowers/specs/2026-11-11-gym-v1-standard-design.md).
Learn's `GYM_V1_CONTRACT` requires the native build identity, clean state and
the selected revision. Shared fingerprints do not select a contract: the
configured revision is authoritative.

## Learn adapters and decisions

The authored templates are
`configs/neural/gym-v1.0.0/{standalone,run,standard-standalone,standard-run}.toml`.
The two Standard modes use Standard combat features and public legal actions.
Standard standalone uses `StandardCombatVectorEnv`; Standard Run uses scalar
`RunEnv(profile="standard")` with one row and one worker. Its Run-mode driver is
batch-1. At noncombat Run boundaries Learn continues to use its existing random
controller; a learned Run policy is not part of this migration.

The contract is chosen by Gym revision, because common and Run fingerprints
are shared by earlier releases. `rules_epoch="gym-1.0.0"` is included in the
new raw-mode mappings. Gym changed rules without changing all schema
fingerprints, so this migration requires fresh weights. Earlier checkpoints
are rejected under the new epoch even when their tensor shapes match; their
identities and historical records remain tied to their original contracts.

`COMMON_V1` routings include `combat`, `combat_selection` and `noncombat`.
Standard combat also supports those routings and the `escaped` outcome. An
escape settles naturally with reward 0: it is neither a win nor a death, fault
or retry. Run-mode escapes continue the Run.

Both Standard modes run through the existing collector and runner by driver
family: `standard-standalone` behaves as standalone combat and `standard-run`
as combat inside a Run. Rollouts, identities and run metadata keep the exact
mode, and run metadata records the Standard profile, content, rules and reward
identities. Curriculum case sampling, scenario streams and calibration remain
legacy Ironclad standalone features.

### Conformance with the real 1.0.0 build

Native qualification and a driver soak found three Standard-only differences
from the common view. Learn now accepts each one explicitly, per view:

- Hidden draw-pile cards are `combat_cards` rows in zone `draw` with
  `zone_position` −1 and known 0 (descriptor `semantics.draw_rows`). This is the
  only unknown non-reference value that is not padding. The model already
  zeroes unknown numeric inputs.
- Potion capacity (16) exceeds every character's slots. Real empty slots are
  known zeroes, and unused capacity rows are unknown zeroes. Empty orb slots are
  also published as known zeroes. The common view keeps its rule that every
  empty potion slot is a known zero.
- A `standard-combat-v1` episode ends at the combat's own terminal, before Run
  settlement. Gym keeps the lifecycle active with phase `terminal` and reports
  the result in `terminal_reason` and the HP fields. Learn's adapter settles
  that visit as standalone combat does: inactive, with `last_exit` built from
  `terminal_reason`, the HP fields and the case's own catalog encounter. A
  terminal with no recognised reason is a contract fault.

## Rules changes affecting comparability

Gym 1.0.0 changes the rules represented by the combat score, independently of
the schema identity. Old panel results such as **41/64** and **35/64** therefore
cannot be compared as if they came from the same game. A fresh-weight
experiment on this release needs its own development baseline and selection
process.

- **Turn order** changed, affecting when player and enemy effects resolve.
- **Louse openings** changed, changing initial enemy behavior and damage.
- **Lagavulin** woken by Poison no longer keeps its Block.
- **Burning elite buffs** now scale with act; the power identifier was renamed
  from `regeneration` to `regenerate`.
- **Autoplayed cards** (for example Havoc, then Exhume) no longer keep their
  free-play waiver.
- **Stasis** changed its roll mapping, altering the selected card outcomes.
- **Power row order** changed in public observations, affecting the mapping of
  categorical rows even when the information remains public.

These rule and ordering changes alter encounters, action consequences or the
meaning of categorical positions. Matching dimensions or relabeling a
checkpoint fingerprint cannot establish score comparability.

## Privileged surfaces

Only public observations, public lifecycle fields and legal action masks reach
the learner. The following Gym surfaces must never be copied to policy or critic
features, lifecycle data or ordinary logs:

- `diagnostic()` and `effective_seed`;
- `exact_checkpoint`, `exact_restore`, `exact_fork` and `from_record`;
- `record` and `reproduction`;
- `take_completed_records()` and `take_fault_records()`;
- `np_random`.

Batch `diagnostics()` is reserved for fault evidence only. It is not an actor or
critic input. Public project seeds are not a substitute for original-game RNG
parity; RNG-001 and TIME-001 qualifications remain in force.

## Known limits

- Gym 1.0.0 delivers PublicSearch branching and tactical starts on Standard
  roots. Learn has not integrated PublicSearch or tactical-start curricula;
  both are follow-up work.
- Run noncombat decisions still use the random controller.
- No training run was part of this migration, and no learned policy was
  evaluated on 1.0.0.
- The soak below uses uniformly random legal actions. It establishes that
  Learn's adapters accept real 1.0.0 frames and boundaries; it does not
  establish playing strength or full rules coverage.
- Standard standalone was soaked with the single authored Ironclad case;
  other characters were soaked through Standard Runs.
- Simulator results do not establish transfer to the original game.

## Qualification status

**Qualified on 2026-11-12** against the verified release build. Every run
used the offline runtime container `spire-learn/task8-python:cuda-setup-20261028-a`
(`--network none`, read-only root, repository mounted read-only), the existing
locked dependency environment and the read-only 1.0.0 overlay. Nothing was
installed.

- **Build identity.** The wheel (`d157065d…`) and `_native` (`5398d2fd…`)
  hashes matched the release. `build_info()` reported revision `1929e18b…`,
  `dirty=false`, profile `release` and the published source and file digests.
- **Descriptors.** Fixtures captured with `scripts/neural/capture_gym_descriptors.py`
  matched all three published fingerprints. Common view v1 is byte-identical to
  the `afb51f5` fixture, and the Standard Run combat view equals the standalone
  Standard combat view.
- **Qualification.** `scripts/neural/qualify_gym_v1.py` exited 0 with status
  `qualified_one_reset_per_mode`: four adapters, four resets, zero steps, each
  driver closed once, and no training imports. The first two attempts failed
  on the draw-row and empty-slot differences above. Their outputs are kept
  beside the passing one.
- **Driver soak.** `scripts/neural/soak_gym_v1.py --steps N --output DIR`
  drives Learn's four real drivers with uniformly random
  legal actions; no learning was involved.
  - 3,000 steps each in `standalone`, `run`, `standard-standalone` and
    `standard-run`: 755, 41, 635 and 43 terminals respectively.
  - 2,000 Standard Run steps for each character at A0 and A20, ending 315 Runs
    in total.
  - It exposed the unsettled Standard terminal, and passed with no contract
    fault after the fix.

The generated outputs are under `artifacts/development/neural-combat/gym-v1.0.0/`
(`qualification/`, `qualification-failed-1/`, `qualification-failed-2/` and
`debug/`), which Git ignores.
