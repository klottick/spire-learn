# Gym 1.0.0 and Standard profile adoption: design

Status: proposed, 2026-11-11. No training, gameplay or native execution is
authorized by this document.

## Why

Spire Gym released **1.0.0**, a frozen contract. Learn still pins Gym
`afb51f5f` from 2 November. Since then Gym has added the **Standard** profile:
all four characters, A0–A20, Acts 1–4 and the Heart. Gym also delivered the two
capabilities Learn requested:

- public belief branching (`PublicSearch`, request `spire-2o7`);
- tactical combat starts (request `spire-831`).

Both run only on Standard roots. Adopting Standard is therefore the
prerequisite for the search-teacher direction (option C in
[the 4 November investigation](../../learning-investigation-20261104.md)) and
for training beyond Ironclad A0.

## Verified release identity

All values come from the release asset `DELIVERY.md`. The wheel and `.so`
hashes were re-checked locally on 2026-11-11.

| Item | Value |
| --- | --- |
| Native build revision (`build_info()["revision"]`) | `1929e18b9f44372e020877bd62f2a5c9dee3e64e` (the tag `v1.0.0` is `fc3c0118`; docs/data only) |
| `dirty` / `profile` / `features` | `false` / `release` / `["spire-python/extension"]` |
| `source_digest` | `d4b3121226cfbb030b89731ed7ea3773f5c2b44988c9ec01c2355d53f2dfa9b2` |
| `file_digest` | `dce682976f221f5686d16beda7ced03f0221624da0a9b83e8b5969f7300f60ce` |
| Wheel SHA-256 | `d157065d9389c92094bef34fc3a58024dd7c2632486dd60a7b4a9f6ace4774ad` |
| `_native` `.so` SHA-256 | `5398d2fd591e025f84435efb4d80909995a764303ce0bc6d87d74c68d4be88cf` |
| Common view v1 (unchanged) | `4c73f7de04d7b2875f473f8034c1cbacf984ea2d52d0a9428289c30cf1dff463` |
| Policy Run v3 (unchanged) | `28d92d89c437eea7fdc260240585bca466dee1fd74b9b3eb38513d2b34b0be64` |
| Card metadata v1 | `effbb810d345e906f6e20b47bd972e35d72ef48141378d54bd28b54afac5f406` (was `b60b66…`) |
| Standard combat view v1 | `5d8147e13ded030c7faa501a8202eb2588915c2e1330b30cac5b06afc9f9916e` |
| Standard Run observation v1 | `bf8e1d111bbf2171594231b99d8d6b701244249abfbb90a11e86c4b4a405f7ae` |

The local inputs are in `artifacts/development/neural-combat/gym-v1.0.0/`,
which Git ignores:

- `inputs/` holds the wheel and `DELIVERY.md`;
- `site/` holds the extracted overlay.

## Decisions

1. **One new contract, `GYM_V1_CONTRACT`, selected only by revision.** The
   common and Run fingerprints did not change between `afb51f5` and 1.0.0, so
   looking a contract up by fingerprint is now ambiguous.
   - Every schema parse and checkpoint check receives the configured revision
     explicitly.
   - `contract_for_fingerprint` becomes `contracts_for_fingerprint`, which
     returns a tuple; it is used only for diagnostics.
2. **Fresh weights under 1.0.0.** Gym changed Ironclad combat rules globally
   without bumping rules IDs. The changes include turn order, Louse openings,
   Lagavulin, burning elites (the power ID was renamed), Havoc/Exhume and
   Stasis.
   - `GymContract` gains a `rules_epoch`. When it is set, the raw-mode mapping
     strings end in `+{rules_epoch}`.
   - Earlier checkpoints are therefore rejected under 1.0.0 even where tensor
     shapes match.
   - The two earlier contracts keep their mapping strings byte-for-byte, so
     their historical configs still validate.
3. **Full build identity is verified.** `_verify_native_build` checks
   `extension`, `revision`, `dirty is False`, `profile`, `features`,
   `source_digest` and `file_digest` whenever the contract supplies them.
4. **A view registry replaces the hard-coded common view.** Each view's
   specification records:
   - `view_id`, `profile_id` and fingerprint domain;
   - the allowed tensor groups;
   - the lifecycle `combat_view_id`;
   - allowed routings and outcomes.

   There are two entries:

   | View | `view_id` | `profile_id` | Domain | Groups | Routings | Outcomes |
   | --- | --- | --- | --- | --- | --- | --- |
   | Common v1 | `spire-gym-combat-policy-view-v1` | `IroncladCombatPolicy` | `spire-gym/combat-policy-view/v1\0` | the existing 12 | `combat`, `combat_selection` | `victory`, `death` |
   | Standard combat v1 | `spire-gym-standard-combat-view-v1` | `StandardCombat` | `spire-gym/combat-view/standard/v1\0` | the 16 below | `combat`, `combat_selection`, `noncombat` | `victory`, `death`, `escaped` |

   The 16 Standard combat groups are `combat_context`, `player`,
   `permanent_cards`, `combat_cards`, `enemies`, `powers`, `relics`,
   `potions`, `orbs`, `selection`, `selection_counts`, `actions`,
   `combat_state`, `known_draw`, `stance_choices` and `combat_card_offers`.

   The feature planner and model are already descriptor-driven and need no
   group-specific code. The Standard view is a separate schema with its own
   vocabularies, not a superset of common v1.
5. **Two new environment modes.**
   - **`standard-standalone`** uses `StandardCombatVectorEnv` with:
     - `autoreset_mode="disabled"`, `fault_policy="freeze"`;
     - `workers` from config (1–256);
     - a `spire-gym-standard-combat-catalog-v1` catalog.

     This mode takes only `fresh_start` entries.
   - **`standard-run`** uses `RunVectorEnv(profile="standard")` with:
     - `autoreset_mode="disabled"`, `fault_policy="freeze"`;
     - `combat_views()` for combat;
     - Run `action_mask` legal indices for non-combat decisions, which are
       chosen by the existing random controller (`collector.py:605`).
   - Cases allow `character` in `ironclad|silent|defect|watcher` and
     `ascension` in 0–20 only in these modes. Existing modes stay Ironclad A0.
6. **Escape is a natural outcome.** `escaped` settles a Standard combat with
   reward 0. It is neither a win nor a fault, and it must never be retried.
   Run-mode escapes continue the Run.
7. **Privacy is enforced by copy-in allow-lists, and tests prove it.** Only
   descriptor tensors and the documented lifecycle fields are copied. The
   following never reach the actor or critic:
   - `diagnostic()`, `effective_seed`;
   - `exact_checkpoint`, `exact_restore`, `exact_fork`, `from_record`;
   - `record`, `reproduction`;
   - batch `diagnostics()`, `take_completed_records()`, `take_fault_records()`;
   - `env.np_random`.

   Learn calls none of these, except `diagnostics()` on a fault, and only into
   fault evidence.

## Out of scope (separate later plans)

- **`PublicSearch` teacher and expert iteration.** This depends on this plan's
  Standard combat schema and providers.
- **Tactical-start curricula** (`synthetic_position`, `reachable_recipe`).
- **A learned non-combat Run policy** (path, rewards, shops, events).
- **Re-evaluating or warm-starting older checkpoints.**
- **Any training run or campaign profile.**

## Acceptance

- `--validate-only` passes for the four new 1.0.0 templates:
  - `standalone`
  - `run`
  - `standard-standalone`
  - `standard-run`
- Unit tests cover the new contract, build identity, the view registry,
  Standard parsing against real captured descriptors, both providers through
  fakes, escape settlement and the privacy allow-lists.
- A released, CPU-only qualification probe runs each actual adapter
  constructor and parses real 1.0.0 descriptors. It runs no gameplay and no
  training. It requires explicit user approval to run the native binary.
