# Fresh Gym curriculum launch plan

> Use the existing reviewed ten-hour curriculum controller, focused TDD and fresh independent launch review. User instruction: "now launch soime training like before". This authorizes the new launch and necessary setup; do not ask routine reapproval. No commits, pushes, installs or Gym changes.

**Goal:** Launch the prior ten-hour fixed/randomized/mixed Ironclad experiment on the migrated exact Gym release, with one shared newly initialized policy and fresh state in every arm.

**Architecture:** Reuse the existing finite scenarios, 64-case development panel, controller, leases, effect journal, checkpoint continuation and independent watchdog. Add an explicit current release profile and exact current raw-mode mappings. A separate bounded CPU bootstrap constructs an untrained checkpoint under the new schema; it performs no gameplay or optimizer steps. The existing first campaign job qualifies actual GPU training/resume before sustained work. Luna owns retained-only monitoring.

**Runtime:** Gym `afb51f5f222b4c4331ed34067c5eebeda606db91`, native `1656e912d88b91043f3a1bc7632989638b63bd2b5554b53703d9484658990eac`, current read-only overlay from `configs/neural/gym-20261102/runtime.json`, existing qualified dependencies and RTX5090. No new package resolution or installs.

## Constraints and scope

- Campaign `curriculum-20261102-a`; profile `fresh-gym-afb51f5f`; owned domain `artifacts/development/neural-combat/curriculum-campaign-v4/`.
- Reuse prior 36,000-second whole-run cap; stop training at second33,600. Qualification first; scheduled CPU development comparisons; no final-split execution.
- Reuse unchanged global ceilings:1,200,000 environment attempts;1,100,000 sampled rows;70,000 Adam attempts;1,100,000 reset slots. Preserve per-segment4,096 actions/samples,256 Adam,4,112 resets,8 updates,600s hard/480s soft; qualification512actions/512samples/8Adam/256resets,600s hard/120s soft.
- Fixed and randomized each four matched segments, then up to256 mixed segments under time/cumulative ceilings. Mixed remains70% realistic,20% lessons,10% unusual. No score-directed schedule changes.
- Shared initial checkpoint has current-schema weights, zero trained decisions/updates, no old-policy conversion or old checkpoint consumption. Warm-starting that untrained checkpoint is only a mechanism to make arm initialization identical; every arm gets fresh Adam, RNG and progress.
- Source support snapshot and scenario DATA retain original provenance. Current native constructors validate selected catalogs before stepping. Do not relabel historical support as newly measured content coverage.
- Protect the existing Task6 report, root review diff, final cases, old packages, claims and campaign evidence. Preserve unrelated GPU process3772; current observed free GPU memory31,604MiB exceeds12GiB floor, so no service stop is needed.
- Native fault or uncertain effect stops the campaign without automatic retry. Keep failed/unstarted/censored evaluation rows distinct. RNG-001/TIME-001, simulator-only, finite Ironclad A0 and one paired replicate remain explicit.

## Task 1 — Current deployment and immutable inputs

- [ ] Add the exact current profile and per-profile pins/qualification-source paths in `curriculum_profiles.py`.
- [ ] Adapt only current-profile package projection to replace environment fingerprints and raw-mode mappings as well as revision; leave old profiles unchanged.
- [ ] Author current campaign/template/overlay/qualification data under `configs/neural/experiments/curriculum-v4/`; prepare current cases by reusing verified scenario bytes.
- [ ] Prove current real loaders and immutable scenario reuse using pure tests; retain RED/GREEN and source scope.

## Task 2 — Fresh parent and actual qualification

- [ ] Add a bounded CPU bootstrap with current configs, exact imported native identity and zero reset/step/sample/optimizer effects; save an untrained owned checkpoint, close once, and retain hashes/source/cleanup.
- [ ] Independent source/control review before bootstrap; no retry of a started native operation after failure.
- [ ] Bind the actual fresh-parent inventory into campaign/qualification inputs and current config hashes. Add exact source-bound authored qualification selection to the controller.
- [ ] Required focused/full default checks, import isolation and independent final launch review. Capture dirty state/source file hashes and modes; never require an unauthorized commit to launch.

## Task 3 — Launch once and delegate monitoring

- [ ] Prepare new user-authorization record, reviewed snapshot, package and fresh one-use release. Validate all actual release inputs and headroom before launch.
- [ ] Launch the reviewed existing watchdog/controller once, preserve the actual owned host exec session, process identities, monotonic start/deadline and result-observation requirements.
- [ ] Confirm actual GPU qualification and initial training effects before claiming training started.
- [ ] Create Luna/MAX retained-only monitor with its own bounded heartbeat. No monitor restarts, source edits, runtime imports, new GPU/Docker work or service control. Terminal callback requests primary actual OS-exit observation and audit.
- [ ] Report the live training scope, deadline and monitor; leave terminal audit to the callback without primary routine polling.

## Preflight continuation ruling

The first controller attempt failed before environment construction or Torch work: its final guest pin guard still expected the historical native hash. Preserve `curriculum-20261102-a` as failed, with actual launcher OS1 and its full failed lease. One separately reviewed continuation `curriculum-20261102-b` uses a new v5 domain, correct selected-overlay pin admission, and the same already retained fresh parent. Its remaining effect ceilings subtract the prior entire qualification lease; its outer limit is35,400 seconds, so prior600-second reserved lease plus continuation cannot exceed the original ten-hour cap. No accepted gameplay effect is retried; no old claim/release is changed. Require source-bound predecessor evidence and pure actual-pin admission tests, then independent source review/new one-use release. Stop on another started failure. Routine monitoring still delegates to Luna after actual training begins.
