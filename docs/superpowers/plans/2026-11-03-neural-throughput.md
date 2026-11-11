# Neural throughput implementation plan

> Use superpowers:subagent-driven-development. User approved the preceding optimization proposal; execute without new routine approval gates. No commits.

**Goal:** Remove repeated frame validation, tiny tensor transfers and PPO re-encoding without changing learning semantics.
**Architecture:** Weak successful-validation memo; owned CPU batch packing; raw encoded rollout row selection.
**Tech stack:** Python3.12, stdlib pure validation, existing qualified PyTorch2.14.1+cu130 in protected containers.
**Spec:** docs/superpowers/specs/2026-11-03-neural-throughput-design.md

## Global constraints

Only this worktree may change. Main /home/klottick/repos/spire-learn frozen source/claims/releases and user files remain unchanged while active. No installs, commits, Gym edits, gameplay, actor sampling or concurrent GPU workload. Same masks/schema/checkpoints/algorithm/RNG/error prefixes. Synthetic optimizer diagnostics cap64 steps. Persist evidence outside managed workspaces; do not archive worktree.

## Review focus

- Invalid/mutated/replaced frames and distinct schemas cannot inherit validation authority.
- Weak memo must bound memory and handle identity reuse/GC/thread interleavings.
- Packing and row selection must preserve independent writable ownership after GC.
- PPO must cache raw constants only and preserve gradients, shuffle state, admission and partial-failure counts.
- Benchmark must compare actual baseline/candidate at identical shapes/seeds with synchronization, not call-count claims alone.

## Task1 — Successful validation reuse

Files schema.py plus a new pure unit test. TDD serial hits, first-time invalid frames, schema identity, replacement, GC, bounded cache and subclass behavior. Implement private bounded weak identity memo, preserve public validate_frame API and deep validation. Run affected schema/snapshot/contracts tests and fresh optional-import isolation. Independent scoped review before task2.

## Task2 — Batch encoding and PPO reuse

Files features.py, ppo.py and focused pure/gated tests. TDD batch packing/ownership/parity, malformed row indices and minibatch data independence. Add EncodedBatch.select_rows;34tensor transfers per encode. Reuse source_batch in PPO after behavior checks; preserve all optimizer/control/budget semantics. Run pure import-safe tests, author no-sampling tensor parity suite and bounded baseline/candidate CPU benchmark. Independent review of actual tensor evidence.

## Task3 — Measurement and durable delivery

Reuse qualified CPU container with isolated baseline/candidate source binds and no Gym import. Run both variants on current authored frames B1/B16/B64 and paired synthetic PPO, no actor draws, <=64Adam. Compare exact output/state/RNG/ownership and measured phase times. GPU benchmark only after campaign terminal and cleanup. Run appropriate default/focused checks once final source is stable; broad review. Export only owned patch/source/evidence, preserve current root104/protected hashes, and arrange safe root integration after active campaign postchecks. No new training launch.
