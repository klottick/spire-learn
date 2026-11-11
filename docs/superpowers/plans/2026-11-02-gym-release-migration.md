# Gym release migration implementation plan

> **For agentic workers:** Use superpowers:executing-plans with focused TDD and one fresh independent source/control review. Preserve the existing user files; no commits or pushes are authorized.

**Goal:** Admit the exact November 2 Gym contract in both Learn combat modes, reject old checkpoints under it, and qualify the actual adapters without gameplay.

**Architecture:** An immutable stdlib release-contract registry binds revisions, fingerprints and raw-mode mappings. Existing historical contracts remain explicit. Descriptor-driven encoders consume the new schema without duplicating game rules.

**Tech stack:** Python 3.12, unittest, existing qualified dependencies and owned-ID CPU container transport; no installers.

**Spec:** `docs/superpowers/specs/2026-11-02-gym-release-migration-design.md`.

## Global constraints

- Current revision `afb51f5f222b4c4331ed34067c5eebeda606db91`; exact pins from the spec.
- Zero gameplay, samples, optimizer steps, GPU requests or dependency installs.
- Existing legacy configs, runtime overlays, campaigns, claims and checkpoints retain their identities.
- Protected foundation Task 6 report, root review diff and final case manifest retain assessment hashes.
- Source work stays in this repository on an isolated feature branch; no commit or push.

## Review focus

- Mixed revision/schema and raw-mode mapping pairs fail before optional imports.
- Old checkpoint metadata cannot enter evaluation, warm start or resume with current identities.
- Real descriptors authenticate by body hash, with 1,024 rows and reference bounds preserved.
- New lifecycle owners/phases are validated; hidden/provenance fields remain outside actor features.
- The zero-gameplay wrapper owns only its created container and proves cleanup even on failure.

## Task 1: Exact contracts and pure admission

Files: new `src/spire_learn/neural/gym_contracts.py`, new `tests/unit/test_neural_gym_migration.py`; modify `config.py`, `contracts.py`, `schema.py`, `envs.py`, `ppo.py`.

Interface: frozen `GymContract`, `contract_for_revision(revision)`, `contract_for_fingerprint(fingerprint)`, and immutable exact contract tables. Mapping IDs are derived from fingerprints using the existing convention.

- [x] Add failing pure tests for new revision/config admission, crossed pairs, lifecycle owners/phases and old checkpoint incompatibility.
- [x] Run the focused tests and retain RED output.
- [x] Implement exact contract lookup and use it consistently across config/schema/frame/factory/PPO admission. Keep parser authority checks and legacy synthetic fixtures intact.
- [x] Run focused config/schema/frame/driver/checkpoint/PPO admission tests; retain GREEN.

## Task 2: New templates and zero-gameplay qualification

Files: new `configs/neural/gym-20261102/{standalone.toml,run.toml,runtime.json}`; owned scripts and evidence under `artifacts/development/neural-combat/gym-migration-20261102/`.

Consumes Task 1 contract lookups and existing `make_driver(config, cases)` / owned-ID container transport.

- [x] Create new pinned config templates without modifying legacy templates or final cases.
- [x] Stage wheel contents in a new overlay with bounded ZIP validation, exact member hashes and read-only package files; preserve the existing dependency environment.
- [x] Prepare a CPU-only metadata/constructor guest and one-use host wrapper under the spec's caps; test its pure boundaries.
- [x] Fresh independent source/control review; fix material findings before release.
- [x] Invoke once, retaining actual exit, complete descriptors, metadata hash, both actual Learn factory results, denied gameplay counts and cleanup receipts. A failed attempt stops; no silent retry.

## Task 3: Verification and maintained documentation

Files: maintained capability/upgrade docs, README links, migration report and this plan's ledger.

- [x] Run default discovery with all runtime gates unset and fresh optional-import isolation.
- [x] Verify source/overlay/protected hashes, exact release identity and owned cleanup; preserve original assessment.
- [x] Update documentation with actual compatibility status, fresh-weight requirement, card metadata limits and test/runtime evidence.
- [x] Retain uncommitted changes for user review; report unresolved limitations without starting training.
