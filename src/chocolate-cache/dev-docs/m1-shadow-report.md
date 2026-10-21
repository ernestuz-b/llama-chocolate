# M1R: reliable shadow observation and metadata management

Validated on 2026-10-21 in the private Chocolate fork. Starting SHA: `ed8f18609a6bbbb9ee86c23301786ce76478ad2d`. The completion message supplies the final local SHA. No push or merge is performed. This report replaces the M1 report; the original evidence and per-layer statistical convention remain available in the starting commit.

## Changed files

- `src/chocolate-cache/chocolate-shadow.{h,cpp}`: compute samples, failure accounting, mirror validity, explicit LRU membership, lifetime counters and elastic window buckets.
- `src/chocolate-cache/chocolate-observation.{h,cpp}`: bounded capture ring and one metadata worker per native cache/context.
- `src/chocolate-cache/llama-moe-chocolate-cache.cpp`: capture adapters, independent backend-completion notifications and shutdown reporting.
- `src/llama-context.cpp`: initialization, explicit graph-compute sample markers and notifications after existing synchronization calls.
- `src/llama-moe-cache.h`: guard all Chocolate declarations.
- Root and `src` CMake files: thread dependency and Chocolate source selection.
- `tests/CMakeLists.txt`, `tests/test-chocolate-shadow.cpp`: retain replay/fuzz checks; add failure, sample, LRU, resize, threaded handoff and performance coverage.
- `tests/test-llama-archs.cpp`, `tests/test_chocolate_integration.py`: committed synthetic inference fixture and ON/OFF comparison driver.
- This report.

The user's pre-existing change to `chocolate-cache-invariants.md` is preserved and excluded from this commit.

## Observation-to-management handoff

The serialized context thread is the sole producer. It copies router IDs from existing host readbacks into a fixed SPSC ring, and records sample markers, native fill facts and existing synchronization notifications. The ring contains values and logical identities; it retains no graph, tensor, backend or router pointers. Producer methods do not allocate, resize, rank, scan the expert population, update statistics or wait for queue space. Existing native readback-vector allocations are unchanged.

One worker owns the original `shadow_cache`, tracking LRU, statistical buckets, scores, actual mirror and independent simulator. It also owns reusable router scratch storage. There is no new task pool or inference-thread mutex. Notifications wake a condition variable; a worker-side 1 ms timed recheck prevents lost notifications from stalling management. Only synchronized context destruction joins the worker. No GPU call is made by the worker.

Capacity is derived from eligible layers, maximum routed width and `cparams.n_batch`. It budgets every token as a separate microbatch, including router, fill and completion records, and reserves two batch budgets for independent producer/consumer progress. This conservative bound also covers small microbatches inside a larger decode batch. It is fixed at initialization and has no production tuning parameter. An unbounded stream without sufficient existing completion points, or a persistently slower worker, can still exhaust it.

On exhaustion, the producer immediately counts lost events and disables further capture. Management invalidates history and all residency mirrors; it does not turn missing selections into zero use. Management allocation/processing exceptions follow the same invalidation path and set a separate failure flag. Context recreation is the recovery boundary. No speculative reconstruction or additional GPU wait is used.

## Compute sample and completion boundaries

A sample is exactly one invocation of `llama_context::graph_compute()` from `process_ubatch()`. Begin/end markers surround the existing `ggml_backend_sched_graph_compute_async()` call. Thus one decoder microbatch, including its observed layers, contributes at most one ordinal and one domain scan. Encode also uses this graph-compute boundary. A whole multi-microbatch `llama_decode()` call is not one sample.

Scheduler return means submission status. It is not GPU completion. The worker retains an end marker until an existing full scheduler synchronization acknowledges its serial. Notifications come from `llama_context::synchronize()` and the existing pipeline graph-reuse synchronization. A synchronization acknowledges earlier explicitly delimited samples; it never creates or merges them. Metadata can lag inference until such an acknowledgement. Shutdown without an end marker or completion acknowledgement invalidates the sample.

- Successful submission plus complete router coverage over the eligible topology, followed by completion acknowledgement, commits a complete sample.
- Failed submission with a fully captured eligible-layer prefix, followed by acknowledgement, commits an aborted sample using Fakecache's convention. An empty known aborted prefix also advances its ordinal.
- Rejected routers, duplicate layer views, missing eligible layers, gaps in an aborted prefix or explicitly unknown coverage prevent a statistical commit. Ordinal, buckets and decay do not advance. History stays invalid afterward; later observations cannot repair an unknown interval silently.
- Factual router selections remain separate from successful execution. Known selections and lifetime counters can be captured even for an aborted/incomplete sample.
- A user eval callback can truncate execution while the scheduler returns success. Such samples are explicitly marked as having unknown coverage. The callback and scheduler are not modified.

Output pruning may shorten later token rows. No transition alignment is inferred. The clock is per actual graph compute, so changing microbatch grouping can change the number of samples, as in Fakecache's multi-token compute convention. It no longer changes decay frequency merely by changing the number of observed layers.

## Residency consistency and failures

Backend completion is captured immediately after the native synchronization, before router-shape validation, in both existing readback paths. It confirms only groups owned by that backend. Full scheduler completion confirms all groups. A rejected observation therefore cannot strand an otherwise completed native fill.

Fill facts are captured after native copies are queued. Management stages them independently of usage acceptance. Slot replacement demotes the previous logical occupant; pending entries become actual only at a recorded completion. Completion checks pending ownership before publication. One group completing cannot publish another group's pending data.

Every staging rejection is counted and invalidates the affected group's mirror. An unknown logical identity invalidates all groups conservatively. Invalid groups clear their claimed placements and count later residency selections as unknown. Their simulated occupants remain independent. A router rejection invalidates usage history at sample closure without invalidating an otherwise consistent residency mirror. Overflow or management failure invalidates both because transfer notifications may also be missing.

Shutdown logs aggregate router/staging failures, incomplete samples, missing routers, lost events, desynchronized groups, history validity and management failure. `LLAMA_CHOCOLATE_DUMP=/path` adds group validity, pending counts and full records. There is no per-token failure logging. Aggregates preceding invalidation describe a captured prefix; validity flags must accompany their interpretation. There is no in-place resynchronization after a mirror failure in this milestone.

The native SLRU, admission, victims, slots, uploads and execution remain authoritative. `src/llama-moe-cache.cpp`, `src/llama-graph.cpp`, `ggml/src/ggml-backend.cpp` and `include/llama.h` are unchanged. The SLRU definition in the Chocolate implementation is byte-identical to the starting version. Neither `build_moe_ffn()` nor `build_lora_mm_id()` changes. No extra readback or GPU synchronization is added.

## Statistics, LRU and allocation

Fakecache reuse is source-grounded in `src/fakecache/statistics.{h,cpp}`, `expert-state.h` and `resident-policy.cpp` from the sibling Fakecache checkout. The adaptation retains dense numeric identity, one ordinal per complete/known-aborted compute sample, once-per-sample selected membership, exact rolling expiration, full-domain bounded eager decay/rise, saturating factual counters and lexicographic `(score,last_use)` ranking. Empty-slot, strictly stronger admission and weakest-slot tie rules are unchanged.

Scores now decay over the complete simulation domain, including experts not yet selected. First tracking admission inherits that expert's prior bounded score. Tracking eviction deliberately forgets tracked score, recency and window history; readmission creates a new generation. Generation-tagged buckets cannot expire a later generation's membership. Per-expert factual lifetime selection counters survive eviction and readmission. Pending transfer evidence survives tracking eviction and retains its committed router ordinal at completion.

LRU membership is an explicit `linked` flag. Unlink ignores nonmembers and cannot decrement an empty population. Touch, removal, admission and eviction remain O(1), with numeric head/tail and neighbor links. Cached records remain exempt from nonresident tracking eviction. Simulation score/recency/occupants are separate and are not chosen using native placement or tracking protection.

Window vectors start empty. They grow for actual distinct sample membership and contract when a reused bucket needs much less capacity. Allocation is prepared before changing statistical history. Reusable scratch can also be contracted on the management path; resize/regrowth tests retain numeric identities. Stable dense expert records remain allocated for the fixed eligible topology. Live topology changes and elastic GPU storage are not implemented.

## Validation

Final logs, binaries, complete inference traces and a hash manifest are in `/tmp/chocolate-m1r-evidence/`.

| Check | Result |
| --- | --- |
| Strict standalone GCC, C++17, `-Wall -Wextra -Wpedantic -Werror` | 362,868 checks, zero failures |
| CTest `^test-chocolate-shadow` | 2/2 passed, including unchanged golden file replay |
| ASan/UBSan | 362,868 checks, zero failures; `ASAN_OPTIONS=detect_leaks=0`; LeakSanitizer unvalidated |
| Direct compilation against Fakecache statistics | 24,000 multi-layer complete/aborted sample comparisons passed |
| Committed scalar Fakecache oracle, replay, randomized LRU and threaded queue tests | Passed |
| Allocation fault, overflow, missing-end, unconfirmed-end, router rejection and staging failure injection | Passed |
| Existing DeepSeek2 architecture/cache regression on CUDA0, ON and OFF | 3/3 in each; host experts, native cache, round-trip and mixed-batch comparisons retained |
| Committed ON/OFF inference driver, CPU and CUDA0 | 47,616 logits per backend bitwise identical; nonzero LoRA effects verified for both microbatch settings; native counters and graph traces identical |

The inference fixture has two Qwen3MoE layers, eight experts/layer, K=2, generated F32 weights (seed 1234), and nonzero rank-four expert-up LoRA (seed 99). Batches are 1, 2, 16, 64, 1, 1 and 8; microbatch limits are 2 and 64. `GGML_OP_OFFLOAD_MIN_BATCH=1` is set before initialization. CUDA0 is the RTX 4060. Each GPU context observes all 372 selections: 96 router views/48 samples or 14 views/7 samples, with no rejected, missing or lost observations and valid mirrors. CPU-only contexts have no native cache observer.

The final CPU logit SHA256 is `0c49a02c81878bc8fda9e39fd15054edf1b6de724c362a3aa42c0a811178309d`; CUDA0 is `51733c9b14bcb12ff91efb82a296fe0f507c7d85b1785db68a1cbe0097bada19`. Equal graph trace counts are 5,822 and 6,140 respectively.

CUDA1 (RTX 5060 Ti) and two-GPU execution are not validated for M1R. CUDA1 initialization failed with out-of-memory; a read-only device check showed only 227 MiB free. The general architecture run also encountered that occupied device. The retained `both` fixture verifies one layer per GPU when resources permit. These failures do not replace current dual-GPU evidence with M1's historical results.

ThreadSanitizer compiled but failed before the tests with `FATAL: ThreadSanitizer: unexpected memory mapping`. Concurrency is covered by the real worker regression and ASan/UBSan, without a ThreadSanitizer claim. Python type-check and editorconfig executables are not installed; syntax, changed-line whitespace and ASCII checks are used locally.

## Capture, processing and memory measurements

The committed `--benchmark` measures 100 samples, 48 layers x 128 experts, K=8 and 256 shared slots. A one-token sample also captures 384 deterministic native fill notifications and matching completions; the 256-token large-batch case captures no cache fills. Processing occurs separately on the management/test owner. The benchmark capture capacity equals the tested token count; production instead uses `cparams.n_batch`, independently of the current sample size. No timing threshold is asserted.

| Tokens/sample | Window | Capture ns/layer, including notifications | Total management us/sample | Statistics us/sample | Simulation/ranking us/sample | Metadata bytes | Bytes including transport |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 91.5 | 105.8 | 10.9 | 70.4 | 561,616 | 611,304 |
| 256 | 1 | 580.1 | 3,553.5 | 53.8 | 2,765.4 | 653,776 | 13,294,344 |
| 1 | 256 | 75.9 | 92.9 | 12.0 | 60.7 | 1,175,992 | 1,225,680 |
| 256 | 256 | 549.4 | 3,373.8 | 82.3 | 2,605.3 | 10,391,992 | 23,032,560 |

All four capture measurements recorded zero producer allocations and exactly 614,400 scan visits: one 6,144-expert scan per sample. Total management includes event consumption, factual counters, LRU/placement maintenance and scratch copies; statistics includes bucket allocation/expiration and the domain update; simulation includes sorting, ranking and trimming. These are host point measurements, not end-to-end inference slowdown, worker scheduling latency, throughput gains or CPU/GPU overlap evidence. M1's 10.3/79.4 us reference measured synchronous per-layer processing, so it is not directly comparable to per-sample management here.

The record sizes remain 64 bytes actual metadata plus 16 bytes independent simulation, with additional dense indices, flags and reusable scratch. At 6,144 experts, an empty window of 1/256/4,096 consumes 555,472/561,592/653,752 metadata bytes. Window cost starts with O(window) vector headers, not reserved O(window x experts) entries. Membership allocation reflects actual distinct selections in retained samples. The window-256 benchmark has populated 100 buckets, not the full 256-bucket steady state; dense continued use will allocate further required memberships.

With a capture budget for 256 tokens and K=8:

| Experts | Empty window | Metadata bytes | One context, transport included | Two independent contexts |
| --- | --- | --- | --- | --- |
| 1,024 | 1 | 99,080 | 2,245,696 | 4,491,392 |
| 1,024 | 4,096 | 197,360 | 2,343,976 | 4,687,952 |
| 6,144 | 1 | 555,472 | 13,187,848 | 26,375,696 |
| 6,144 | 4,096 | 653,752 | 13,286,128 | 26,572,256 |

The conservative capture ring can dominate memory; its bound grows with batch capacity, eligible layers and routed width, independently of statistics window. The 6,144-expert, 256-token ring alone is 12,632,112 bytes. This is a declared CPU-memory tradeoff for bounded capture across the smallest possible microbatches. Per-context costs add; contexts do not share mutable metadata. Figures include persistent vector capacities and object storage, and exclude allocator bookkeeping, transient resize replacements, thread runtime/stack and the enclosing owner's pointer. Each context also has one implementation-dependent worker stack. These figures do not describe GPU cache allocation.

## Reproduction and remaining limits

From the repository root, create separate static ON/OFF builds using the same compiler/backend options:

```sh
cmake -S . -B /tmp/chocolate-on -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_CHOCOLATE_CACHE=ON -DGGML_CUDA=ON
cmake -S . -B /tmp/chocolate-off -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_CHOCOLATE_CACHE=OFF -DGGML_CUDA=ON
cmake --build /tmp/chocolate-on --target test-chocolate-shadow test-llama-archs -j8
cmake --build /tmp/chocolate-off --target test-llama-archs -j8
ctest --test-dir /tmp/chocolate-on -R '^test-chocolate-shadow' --output-on-failure
python3 tests/test_chocolate_integration.py --on /tmp/chocolate-on/bin/test-llama-archs --off /tmp/chocolate-off/bin/test-llama-archs --backends CPU CUDA0 --output-dir /tmp/chocolate-integration
/tmp/chocolate-on/bin/test-chocolate-shadow --benchmark
```

Use `-DGGML_CUDA=OFF` and `--backends CPU` for CPU-only reproduction. Add `CUDA1 both` when both devices have resources. The actual validation reused `/tmp/chocolate-m1-on` between static ON/OFF configurations and preserved each executable in the evidence directory. The committed driver does not depend on temporary M1 probe sources.

Coverage still requires an eligible native cache layer and one of the two existing host-router readback paths. CPU-only/ineligible layers and generic splits without the qualifying first `MUL_MAT_ID` remain outside coverage. Missing eligible views invalidate statistics. Actual cross-group/device identity is fixed; no expert migrates between groups. Tests cover cross-layer replacement within a shared group and independent completion/replacement across groups.

Defaults remain `window=1, decay=1, rise=2`; production score/window selection, tracking retention policy and hypothetical admission policy refinements remain decisions for later milestones. Synthetic F32 inference is validated on CPU/CUDA0; pretrained, quantized-cache, current two-GPU, real inference abort and malformed-router integration fault injection remain unvalidated. Rejection/staging/abort mechanics are covered in committed manager/transport tests. Dense expert topology remains fixed; metadata history/scratch elasticity does not implement GPU residency elasticity.
