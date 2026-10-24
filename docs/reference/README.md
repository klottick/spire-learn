# References copied from Spire Gym

Copied on **2026-10-23** from the local `spire-lab` checkout of
[klottick/spire-gym](https://github.com/klottick/spire-gym), without editing or
removing the originals. The repository's historical local/package name differs
from its GitHub name. See [the source manifest](spire-gym/SOURCE.json) for the
exact source checkout/revision, paths and SHA-256 hashes of the copied bytes.
These snapshots do not automatically track subsequent gym changes.

## Reading order

| Copy | Why it is relevant | Interpretation |
| --- | --- | --- |
| [Combat environment guide](spire-gym/docs/combat-environments.md) | Public observations, masks, seeds, native batching, Gymnasium reset/final-observation and fault semantics | The documented supplied-entry Ironclad A0 interface, not full-run Python support. Installation commands apply to a gym checkout, not this scaffold. |
| [Repository scope](spire-gym/docs/scope.md) | Simulator/learner boundary, player-visible information and historical combat/run-policy direction | Gym context only. Its task status and approvals do not establish learner implementation or an installed extension's capabilities. |
| [Intentional deviations](spire-gym/docs/intentional-deviations.md) | RNG-001 and TIME-001 limitations to disclose in experiment/transfer reports | Project seeds are not original-game seeds; simulator performance does not prove original-game transfer. |
| [Original ML-first design](spire-gym/docs/superpowers/specs/2026-10-11-spire-lab-ml-first-design.md) | Historical model/training ideas (§8), split/evaluation/reporting guidance (§9), and two-policy amendment | Historical only. Its combined-repo layout, proposed tensor shapes, PPO defaults, budgets, capture gates and roadmap are not current learner requirements or verified APIs. |
| [Project randomness design](spire-gym/docs/superpowers/specs/2026-10-16-project-randomness-design.md) | Distinguishes environment randomness from scenario/policy/optimizer randomness | Simulator-owned contract, not code to reimplement in the learner. Dated status statements are not current acceptance evidence. |

The copies preserve their original text and relative directory structure.
References to files not included in this small selection still belong to the gym;
follow the corresponding path in its checkout/repository rather than assuming
every relative link resolves here. No simulator source, datasets, task database,
private evidence, environment or native binary was copied.

## What carries forward—and what does not

- The learner consumes one authoritative Rust simulator, with public information
  only. Legal-action indices are decision-local; padding is not a legal action.
- Respect actual terminal versus reset observations, resource outcomes and faults.
  Do not silently score faults as game losses or bootstrap from reset states.
- Keep environment, policy and optimizer randomness separate; freeze evaluation
  cases independently of training/model selection and report resource budgets.
- Separate combat/run policies are a previously discussed direction, not a mandate
  to implement two models now. PPO and architecture constants need fresh selection
  against the interface/profile actually used.
- The historical combat +1/−1 reward is not automatically a full-run reward
  contract. Read the actual selected environment's schema/reward metadata.
- Full-game/all-character/A0–20 coverage is a target, not an availability claim.
  New native milestones do not imply a rebuilt full-run Python/Gymnasium API.
- Source-grounded/reference-informed simulation is not original-runtime validation.
  Recorded upstream tests are historical evidence, not tests run for this bootstrap.
- Historical Daily-mode goals and one-repo requirements do not expand this new
  repo's scope or authorize training, installations or publication.

## Attribution

The copied documents are authored Spire Gym project material. The upstream
[MIT license](spire-gym/LICENSE) is retained alongside them. This does not select
a license for new Spire Learn code, relicense third-party datasets or grant rights
to Slay the Spire game code/assets (owned by Mega Crit).
