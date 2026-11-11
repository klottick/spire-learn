# Neural data-path throughput optimization

The user approved the proposed validation, batched-transfer and encoded-rollout reuse improvements. The current campaign stays on its frozen source. Implement in this isolated worktree, preserving the dirty public baseline copied from the live checkout; no commits, installations, Gym edits, training restarts or new gameplay.

## Selected approach

Use a bounded weak identity memo for successful validation of immutable public CombatFrame/schema pairs; batch-pack each descriptor tensor on CPU before one owned device transfer; reuse the already encoded raw rollout through row selection in PPO. Preserve all schema/feature/model/checkpoint IDs, field meanings, padding, known/occupied/legal masks, floating arithmetic, shuffle RNG, optimizer counts and error/control accounting. Actor/critic inputs remain only the existing34 public tensors. Never cache trainable embeddings, model outputs or autograd graphs across optimizer steps.

Alternatives considered: retain repeated conversion (known CPU bottleneck); add parallel collectors or change the Gym export API (larger algorithm/runtime surface before fixing measured Learn overhead). The selected approach implements the user's preceding optimization request; environment batch size stays16 until evidence warrants a separate change.

## Validation and ownership

Memoize only completed successful deep validations, keyed by exact frame and authenticated schema identity, with weak references and a4096-entry bound. Invalid frames never receive a certificate; changed/replaced frames or schemas must revalidate. Do not trust a caller-provided validated flag or fingerprint alone. Preserve first-time checks, integer domains, binary masks, references, empty-potion semantics and ownership. Support normal immutable exact classes; any subclass with weaker invariants must receive full checks. Cache lookup must never retain frame buffers, and weakref callbacks cannot delete newer entries after identity reuse. Existing Python frozen-record contracts remain unchanged.

For each tensor key, concatenate frame bytes in batch order into fresh writable owned storage, tensorize/reshape to[B,*shape], then transfer once. Preserve CPU/GPU lifetime and contiguity without readonly-buffer warnings; no alias to immutable source bytes or another encode call. Synchronous copies are sufficient initially: do not add unmeasured pinned-memory/async lifetime complexity. EncodedBatch.select_rows accepts a nonempty1-D int64 tensor of valid indices, moves its index vector at most once to the batch device, and returns independent writable selected tensors plus unchanged fingerprint/metadata. Validate malformed or out-of-range indices before GPU indexing can poison a device.

PPO reuses only the existing source_batch built for behavior-data admission, after that validation succeeds. Select mini-batches from its constant tensors; recompute policy outputs and gradients for every optimizer step. Preserve budget-check placement, rollout consumption, version increments, UpdateFault prefixes and control-exception precedence.

## Verification and resources

Use deterministic pure tests for memo misses/hits/GC/bounds/schema separation/invalid inputs, then no-gameplay tensor parity and whole-update parity against baseline. Require exact raw tensors/masks/coverage, identical CPU outputs/gradients/Adam/RNG and documented CUDA tolerance if kernels differ. Meaningful optimized-path call counts:34 transfers per encode independent of batch size; one full encode per PPO update, not per minibatch/epoch. Include both current4c73...descriptor and supported legacy fixtures.

Use existing qualified locked dependencies; no host imports of ML runtimes. A bounded CPU container may run synthetic tests while the campaign is active (host has32cores; cap CPU2/4 and memory8GiB). No concurrent CUDA benchmark, gameplay reset/step or actor sampling; protect process3772. GPU benchmark waits until campaign terminal/cleanup and actual OS observation; no training changes. Synthetic PPO diagnostic steps are explicitly bounded64 total, actor draws0. Public authored frames only; no simulator approximations.

## Delivery

Keep original root/source104 snapshot unchanged while campaign active. Retain baseline/candidate source and benchmark evidence outside Paseo-managed workspace lifetime. Produce a patch containing only optimization-owned files, independent review and a measured speed report. Stage root integration only after campaign postchecks; if still active, prepare exact safe apply/handoff for its terminal callback. Never archive this worktree before durable delivery and explicit retention coordination. Report actual measured CPU and CUDA results separately; do not promise an inferred end-to-end speedup.
