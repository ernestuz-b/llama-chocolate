# M1: shadow metadata and cache simulation

Validated on 2026-10-21 in the private `ernestuz-b/llama-chocolate` fork. Starting commit: `660bd6aa6e46ca9f26b1b896f7582a43f52de242`, after the user updated the checkout. The final local commit SHA is supplied in the completion message; this report is included in that commit. No push or merge is authorized or performed.

## Changes and ownership

- `src/chocolate-cache/chocolate-shadow.{h,cpp}`: metadata, tracking LRU, statistics, independent simulation and explicit dump.
- `src/chocolate-cache/llama-moe-chocolate-cache.cpp`: observer ownership and adapters around existing readbacks, fill notifications and shutdown telemetry.
- `src/llama-context.cpp`: feed the generic copy callback's existing host IDs to Chocolate; confirm residency after an existing scheduler synchronization. Both hooks compile only with Chocolate enabled.
- `src/llama-moe-cache.h`: two internal observer method declarations required by those hooks. This is the shared internal header, not the installed public API.
- `src/CMakeLists.txt`: select the shadow source and private observation macro only for Chocolate.
- `tests/test-chocolate-shadow.cpp`, `tests/chocolate-shadow-replay.txt`, `tests/CMakeLists.txt`: standalone tests, deterministic file replay, a golden replay result and a processing benchmark.
- This report.

The Chocolate cache implementation owns one `chocolate::shadow_cache`. It owns a fixed dense expert table, immutable layer offsets and group identities, numeric tracking links, rolling buckets, reusable selection scratch and separate actual/pending/simulated slot tables. Logical identity is `layer.begin + expert`, independent of a physical slot. Lookup, LRU touch, removal and admission are O(1).

The metadata record is 64 bytes on this host. Its history contains `last_use`, `score`, `window_uses` and a saturating selection count. The remaining fields hold numeric links, tracking generation and observed actual/pending placement. Only confirmed router use admits a tracking record. Promotion removes it from the nonresident LRU; demotion returns it at MRU without clearing history. Tracking eviction clears history. Generation-tagged buckets cannot expire history belonging to a later readmission.

All vectors and bucket capacities are allocated during initialization. Growing and shrinking the active tracked population does not move logical identities or resize backing storage. There are no retained router pointers. Physical release/compaction of metadata backing storage is not implemented. Production retains every used expert in the eligible topology; only tests/replay can impose a smaller nonresident tracking capacity. No production tracking threshold was invented.

Simulation has its own 16-byte-per-expert score/recency/slot record. This separate history is necessary: actual residency can protect an observed tracking record from eviction, and must not thereby determine simulated ranking. Tests compare simulation with different actual placements and tracking capacities.

## Fakecache reuse

Inspected Fakecache checkout `/mnt/AI/AiProgs/experiments/llama-fakecache`, HEAD `bfae85005f36c529e66c58c8663411349335f781`: `expert-state.h`, `statistics.{h,cpp}`, `cache-controller.{h,cpp}`, `iresident-policy.h`, `resident-policy.{h,cpp}`, `resident-cache.h`, and resident policy/storage/statistics tests.

Adapted the dense owner/table contract, minimum history fields, bounded eager score decay/rise, once-per-request rolling-window membership, saturating factual counters and reusable markers/buckets. The resident ranking implementation preserves Fakecache's lexicographic `(score, last_use)` ordering, strictly stronger admission, equality without churn and lowest-index weakest tie. Boundary tests include the unique weakest final slot. A separate probe compiled against the existing Fakecache statistics source passed 32,000 direct history comparisons.

The controller, resident storage and lifecycle tests informed ownership, numeric identities and publication at completion boundaries. Their device storage machinery was not ported. No rotating-buffer architecture or prediction/publication/execution subsystem was introduced.

## Observation and residency

1. The Chocolate slot-map `copy()` callback consumes IDs after its existing readback and synchronization, before the unchanged SLRU plan. `build_moe_cache_slots()` flattens IDs; the adapter obtains the original token/lane shape from the reshape source without another readback or graph scan.
2. The generic `llama_context::sched_copy_experts()` path consumes the existing host readback when its ID cache is refreshed. The immutable host weight binding identifies the logical layer. Observation remains valid if this operation executes on a backend different from that layer's cache backend. The real cache's backend check remains unchanged.
3. Fill notifications update only shadow actual/pending tables after vanilla queues its original copies. Pending fills become confirmed residency at an existing matching-backend synchronization, the context's existing scheduler synchronization, or synchronized context destruction. No new backend wait, event or readback is introduced. Before a known completion boundary, pending metadata is conservative.

Router observations count confirmed selections even if a later compute fails; they do not claim successful execution. Actual-residency selection counts describe availability in the native cache, not measured execution hits. Native cache counters remain separately reported.

Coverage is bounded to existing cache bindings and readbacks. CPU-only contexts without a native cache, ineligible layers, a generic copy whose split does not start with the qualifying `MUL_MAT_ID`, and execution paths without either host readback are not observed. No eval callback, graph/scheduler change or extra synchronization was added to close those gaps. All selections were observed in the tested eligible single- and two-GPU workloads, including the large-batch path.

## Simulation contract and remaining policy decisions

Each simulated group uses exactly its vanilla group's usable `n_slots` and sum of expert tensor `nb[2]` values. The native extra guard slot and alignment padding are not usable capacity. Groups retain vanilla's existing layer/device partition. Actual admissions, victims and slot assignments are never simulator inputs.

Each request counts hits/misses against pre-request simulated residency. A miss is hypothetically CPU-served for that request. After updating history, distinct selected candidates are considered in logical identity order using Fakecache ranking. A promotion becomes available for the next request and estimates one full expert upload; eviction and population counters follow these hypothetical changes. There is no timing, execution graph, bandwidth or throughput simulation.

The inherited Fakecache defaults are `window=1`, `decay=1`, `rise=2`; constructor tests exercise other values. Score bounds are `[-decay*window, (rise-decay)*window]`. Here an ordinal/window bucket is one observed layer router request, not a whole compute. Duplicate selections count separately in factual frequency/hits/misses but affect score/window membership once per request. Replay accepts duplicate lanes as well as duplicates across token rows.

Production tracking capacity, the production score/window settings, a compute/token epoch convention and any additional candidate eligibility rules remain unresolved. The simulator currently considers every selected expert in its eligible group. Its ranking history is not limited by the observed metadata's test-only tracking cap. Results depend on batching/observation grouping and must include that configuration. Simulated hits and CPU service are not measured performance improvements.

## Validation

All artifacts and complete logs are in `/tmp/chocolate-m1-evidence/`. The temporary inference probe reuses the existing architecture-test fixture helpers; it is a validation artifact, not a new execution implementation.

| Check | Result |
| --- | --- |
| Standalone strict build: `g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror -Isrc tests/test-chocolate-shadow.cpp src/chocolate-cache/chocolate-shadow.cpp -o /tmp/test-chocolate-shadow` | Passed |
| `/tmp/test-chocolate-shadow` | 58,973 checks, 0 failures |
| CTest `^test-chocolate-shadow` | 2/2 passed, including golden file replay |
| ASan/UBSan standalone build/run | 58,973 checks, 0 failures; `ASAN_OPTIONS=detect_leaks=0`, LeakSanitizer unvalidated |
| Existing Fakecache statistics parity probe | 32,000 comparisons passed |
| CUDA-enabled `test-llama-archs` build, Chocolate OFF and ON | Both passed |
| Existing `test-llama-archs -a deepseek2 -s 1234`, OFF and ON | 6/6 passed in each build, including native cache, host experts, round-trip and mixed-batch checks |
| Seeded Qwen3 MoE inference probe, OFF versus ON | Bitwise equality, identical graph assignment traces and identical native cache counter lines on CPU, CUDA0, CUDA1 and an explicit two-GPU layer split |
| Nonzero MoE LoRA on the up-expert matrix | Changed outputs; retained bitwise OFF/ON equality in every backend case |
| `git diff --check` | Passed |

The Qwen3 probe uses two layers, eight experts per layer, top-K two, seed 1234 and generated F32 weights. It decodes batches of 1, 2, 16, 64, 1, 1 and 8 tokens, with `n_ubatch=2` and `64`, with and without the LoRA. `GGML_OP_OFFLOAD_MIN_BATCH=1` is set before initialization. Each backend case compares 47,616 logits. CUDA0 is the RTX 4060; CUDA1 is the RTX 5060 Ti. The two-GPU case explicitly verifies one MoE layer on each device, with group capacities 2 and 8; individual GPU cases use a shared five-slot group. Both microbatch configurations observe all 372 selections per workload, with zero rejected observations. The larger microbatch configuration observes all 14 layer requests; the smaller one observes 96.

| Backend | Equal ON/OFF logit SHA256 | Equal graph trace lines |
| --- | --- | --- |
| CUDA0 | `51733c9b14bcb12ff91efb82a296fe0f507c7d85b1785db68a1cbe0097bada19` | 6,140 |
| CUDA1 | `f1e47992c92b6473d0f7c6ea1c2fd7cbbd4e3cc20dd3caecd069838f5fd10e71` | 6,140 |
| Both | `f1967f8f8957ac717cdb72d81df7caf04fe2042231d60b30a15ead29493e2ba8` | 6,202 |
| CPU | `0c49a02c81878bc8fda9e39fd15054edf1b6de724c362a3aa42c0a811178309d` | 5,822 |

Full pretrained-model and quantized-cache numerical comparisons were not run. The tests above exercise real inference on synthetic fixtures, not just replay. They do not establish speed gains or CPU/GPU overlap.

## Actual versus simulated example

CUDA0, no LoRA, five usable slots, 372 selections. Values are from `on-CUDA0.log` and `validated-on-CUDA0.dump`; OFF has identical native counter lines.

| Statistic | `n_ubatch=2` | `n_ubatch=64` |
| --- | --- | --- |
| Native measured unique cache hits / misses | 62 / 247 | 12 / 6 |
| Native measured cache-fill bytes | 291,373,056 | 7,077,888 |
| Native large-batch cache copies, per tensor expert slice | 0 | 45, totaling 17,694,720 bytes |
| Observed resident / nonresident selections | 73 / 299 | 101 / 271 |
| Simulated GPU hits / CPU-served misses | 100 / 272 | 25 / 347 |
| Simulated promotions / evictions | 230 / 225 | 35 / 30 |
| Simulated uploaded expert bytes | 271,319,040 | 41,287,680 |
| Final observed / simulated residency | 5 / 5 | 5 / 5 |
| Selection placement disagreements | 81 | 92 |

Native fill counters count distinct experts per native plan, while selection counters include duplicates. Native cache-fill bytes omit generic host uploads; cache-copy counters describe device copies. These are not total measured transfer traffic and must not be compared as if the units and coverage were identical.

## Memory and processing overhead

`test-chocolate-shadow --benchmark` uses 48 layers, 128 experts/layer, top-K eight, 256 shared slots, inherited statistics defaults and 4,800 layer requests per case. The one-token case also stages/confirms actual mirror replacements; both cases include completion checks. On this host, the recorded sample was:

| Tokens per layer request | Time/request | Time/selection | Capacity-based host storage | C++ allocations during processing |
| --- | --- | --- | --- | --- |
| 1 | 10.475 us | 1,309.3 ns | 653,600 bytes, about 638.3 KiB | 0 |
| 256 | 79.399 us | 38.8 ns | 653,600 bytes | 0 |

Storage includes the manager object, topology, vector capacities, observed/simulated records, markers, ranks and preallocated generation-tagged window buckets. It excludes allocator bookkeeping and the enclosing cache owner's pointer. Storage grows with topology, window length and usable slots; active tracking contraction does not release the reserved storage. Processing includes an eager scan of the expert table and resident-rank scans; these timings are a sample of observer processing, not end-to-end inference overhead or an improvement claim. Existing inference allocations/readbacks were not removed; no new ones were added by metadata processing.

## Reproduction and telemetry

```sh
cmake -S . -B /tmp/chocolate-m1-on -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES='89;120' -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CHOCOLATE_CACHE=ON
cmake --build /tmp/chocolate-m1-on --target test-chocolate-shadow test-llama-archs -j 12
ctest --test-dir /tmp/chocolate-m1-on -R '^test-chocolate-shadow' --output-on-failure -V
/tmp/chocolate-m1-on/bin/test-chocolate-shadow --replay tests/chocolate-shadow-replay.txt
/tmp/chocolate-m1-on/bin/test-chocolate-shadow --benchmark
```

The inference comparison reconfigures that same build directory with OFF, builds, and runs `python3 /tmp/chocolate-m1-evidence/compile-probe.py off`, then repeats with ON and `compile-probe.py on`. CUDA backend objects are reused; each static probe embeds its selected cache implementation. Run each `inference-{on,off} OUTPUT BACKEND` for `CPU`, `CUDA0`, `CUDA1`, `both`, then `python3 /tmp/chocolate-m1-evidence/compare.py`. GPU execution required access outside the filesystem sandbox. The build directory finishes configured ON.

Temporary probe source SHA256: `3ed5f9ddd0ff569dbec79478871ec0e229b74c77f650e5a7ea69f0efc29459e5`. The helper source, compiler/link helper, comparison script, binary logits, logs and detailed dumps remain in the evidence directory. `comparison.log`, `unit.log`, `ctest.log`, `sanitizer.log`, `fakecache-parity.log`, `benchmark.log` and `archs-{on,off}.log` record the results.

Replay format starts with layer/group counts, then one `n_expert group` pair per layer and one `capacity expert_bytes` pair per group. `R layer lanes tokens ids...` supplies a router request, `S layer expert slot` records a queued vanilla fill, `C group` records completion, and `T capacity` sets the test-only tracking cap. Blank lines and full-line `#` comments are accepted. The fixture exercises cross-layer replacement, duplicates, tracking eviction and actual/simulated independence. Replay prints aggregates and records deterministically.

Normal inference adds only aggregated shutdown summaries after the native measured statistics. Set `LLAMA_CHOCOLATE_DUMP=/path/to/file` to append detailed records, topology, history, actual/pending/simulated placement and LRU links at cache destruction. This dump setting does not change policy. File I/O occurs only at destruction, after synchronization.

## Execution boundary and review

`src/llama-moe-cache.cpp`, `src/llama-graph.cpp` and `ggml/src/ggml-backend.cpp` are byte-identical to the starting commit. The complete SLRU source is unchanged, SHA256 `c94fd6e0fc7a93debb1909cef0db6539262d125086277ebd85525b3ac30f732c`. Original graph construction, LoRA execution, native cache allocation, slot selection, transfer ordering, copy/execution dispatch and scheduling remain authoritative. New simulation state is never read to decide real work.

Self-review applied the repository skill's scope, security, approach/design, tests and general checklists. IDs, shapes, strides, score arithmetic and capacities are checked; invalid observations leave history/placement unchanged except for their rejection counter. Metadata setup failure disables observation with a warning and preserves native inference. There is no public API change, production tracking tuning parameter, new dependency, actual promotion policy, elasticity, speculative scheduler or mixed-execution implementation.
