# Neural data-path performance

The data-path optimizations were implemented and reviewed in an isolated worktree. At the user's request, they were then activated through a checkpoint cutover during the November 2–3 campaign. The original source records remain unchanged; the continuation records its new source identity and inherited state. The completed campaign's [learning and throughput results](neural-curriculum.md#fresh-gym-campaign-results-2026-11-03) distinguish both phases.

The retained campaign measurements showed about 14 learned decisions per second and very low GPU utilization. An earlier detailed profile put most measured time in Learn's feature conversion and repeated observation validation. That profile used an earlier Gym schema, so it identifies likely bottlenecks rather than proving the current exact time split. Low end-to-end throughput alone does not show that Gym's native engine is slow.

The implementation preserves the schema, model, PPO objective, checkpoint identities, legal-action masks and environment batch width. It changes three data-handling operations:

- Successful validation is reused only for the same immutable frame and authenticated schema objects. Bounded weak references avoid retaining observations. New, invalid or mutable inputs still receive full checks.
- Each public tensor is packed into an owned CPU batch before device transfer, replacing transfers of individual frames. This reduces the raw-tensor transfer calls from 34 times the batch width to 34 per encoding.
- PPO selects minibatches from the raw tensors already encoded for behavior-data admission. Policy outputs, losses and gradients are recomputed after every optimizer step; learned activations and autograd graphs are never reused across steps.

The benchmark uses the current public descriptor and 64 distinct synthetic observations with varying values, known coverage and legal actions. Separate baseline and candidate interpreters compare exact CPU tensors, outputs, PPO metrics, gradients, weights, Adam state and shuffle RNG. Actor sampling and Gym imports are blocked. These are synthetic diagnostics, not gameplay or learning-efficacy results.

The bounded CPU comparison used PyTorch 2.14.1+cu130 with two intra-op threads and one inter-op thread. Encoding times are medians of three measured calls after two warmups on 64 distinct, already-validated frames. First-time deep validation remains necessary: validating all 64 frames took 0.313 seconds before and 0.319 seconds after.

| Diagnostic | Baseline | Optimized | Ratio |
|---|---:|---:|---:|
| Hot encoding, batch 1 | 5.153 ms | 0.194 ms | 26.6× |
| Hot encoding, batch 16 | 81.761 ms | 0.451 ms | 181.4× |
| Hot encoding, batch 64 | 323.935 ms | 1.719 ms | 188.5× |
| Synthetic PPO, 16 rows / 4 Adam steps | 0.747 s | 0.420 s | 1.78× |

The PPO comparison encoded the full rollout five times before and once after. All 24 independent paired checks passed, including ordered inputs, masks, outputs, final gradients, parameters, Adam state, metrics, consumption and shuffle RNG. Six gated tensor tests also passed, covering exact per-step gradients/Adam states, ownership, row ordering and failure/control prefixes; 180 focused pure tests passed. A fresh process imported all product modules without optional-runtime imports.

This is one paired synthetic CPU measurement, with no gameplay, sampling or CUDA context. Shared host load can affect timing, and hot encoding excludes the initial validation cost. The evidence is retained under `artifacts/development/neural-combat/throughput-20261103/`, including the expected pre-change RED and final GREEN. After the cutover, actual CUDA/native qualification and six nonempty resumed training segments completed successfully. Original mixed training averaged 13.60 decisions/s; the optimized source averaged 61.92/s, about 4.55× faster including job setup and cleanup. This is a campaign comparison across different shards, not a controlled CUDA microbenchmark or a measurement of GPU utilization. The much larger hot-encoding ratio must not be presented as the end-to-end gain. Environment width stayed at 16.
