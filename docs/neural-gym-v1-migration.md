# Spire Gym 1.0.0 and Standard adoption

Updated **2026-11-12**. Spire Learn now selects Gym by the configured revision
and supports the Gym 1.0.0 common, Standard combat and Standard Run contracts.
The native extension revision is
`1929e18b9f44372e020877bd62f2a5c9dee3e64e` (tag `v1.0.0` is `fc3c0118`,
documentation/data only). The release build identity is `release`, clean, with
feature `spire-python/extension`, source digest
`d4b3121226cfbb030b89731ed7ea3773f5c2b44988c9ec01c2355d53f2dfa9b2` and file
digest `dce682976f221f5686d16beda7ced03f0221624da0a9b83e8b5969f7300f60ce`.

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

## Rules changes affecting comparability

Gym 1.0.0 changes the rules represented by the combat score, independently of
the schema identity. Old panel results such as **41/64** and **35/64** therefore
cannot be compared as if they came from the same game. A fresh-weight
experiment on this release needs its own development baseline and selection
process.

- **Turn order** changed, affecting when player and enemy effects resolve.
- **Louse openings** changed, changing initial enemy behavior and damage.
- **Lagavulin Poison wake block** changed the block applied when Poison wakes
  Lagavulin.
- **Burning elite buffs** now scale with act; the power identifier was renamed
  from `regeneration` to `regenerate`.
- **Havoc and Exhume** now waive free-play restrictions for their forced card
  plays.
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
- No training run was part of this migration.
- Standard combat qualification covers one reset boundary per mode and no
  gameplay step. It does not establish gameplay conformance or playing strength.
- Simulator results do not establish transfer to the original game.

## Qualification status

**pending controller run**

The authored `scripts/neural/qualify_gym_v1.py --output DIR` probe constructs
each of the four adapters, performs one reset per mode, validates the parsed
public rows, checks the complete native build identity, and closes every
driver. It has a meta-path guard against importing Torch, TorchRL or
TensorDict, and it never calls `step`. Its unit tests use fakes and do not
import `spire_lab`. No native qualification result is recorded here; the
controller fills in the container run result and evidence.
