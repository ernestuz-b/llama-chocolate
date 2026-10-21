# Chocolate Cache — consolidated source and experimental audit

**Audit date:** 20 October 2026  
**Purpose:** factual technical reference for later design discussions; **not an implementation proposal or migration plan**.  
**Revision:** 20 October 2026, revised after independent line-by-line review and clarification of the Fakecache 10 execution-initiation example. The source baseline remains pinned to the commit below; this revision does not claim a new checkout or benchmark.  
**Current source baseline:** [`ernestuz-b/llama-chocolate@39da130fcd1b6ca47d3c0522a40eccff41ea3dfb`](https://github.com/ernestuz-b/llama-chocolate/commit/39da130fcd1b6ca47d3c0522a40eccff41ea3dfb) (`master` at inspection).  
**Uploaded independent audit:** `moe-cache-lru-surface-audit.md`, audited against [`098ce526b546b5a190327c1da82929cee051b9ac`](https://github.com/ernestuz-b/llama-chocolate/commit/098ce526b546b5a190327c1da82929cee051b9ac).  
**Earlier ChatGPT integration audit (18 October):** Fakecache P8 [`895e306b96a9e1c8d9ae3c8bddf8e8287f786c69`](https://github.com/ernestuz-b/Fakecache/commit/895e306b96a9e1c8d9ae3c8bddf8e8287f786c69) versus then-current llama.cpp `5aeac37a35b44fde4bdffaec9159f2fcb253b622`. The historical audit's conclusions have been recovered from the prior discussion and relevant source inspected again; the entire original audit text was **not** independently supplied as a document here.  
**Method:** live GitHub source reads, commit-to-commit comparison, earlier audit and historical experiment context. An independent review supplied on 20 October was checked against selected current source ranges. No build, GPU inference, profiler run, concurrency test or numerical comparison was performed for this audit. The separately uploaded LRU audit remains available as `moe-cache-lru-surface-audit.md`; this consolidated audit does not overwrite it.

### Interpretation of the requirements labels

The independent `098ce526b` audit evaluated issues against three **project intentions**, not claims about existing source: **R1:** elastic, chunk-granular cache sizing; **R2:** do not block inference waiting for expert-weight transfers (CPU execution is an ordinary path); **R3:** evidence-driven, gradual resident promotion rather than admission driven only by immediate use. A source observation may be *material to R1/R2/R3* without being a present-day bug or dictating a particular architecture. Numerical priorities and proposed implementation schemes in the independent review are not adopted here.

## A. Executive findings

1. **The code is no longer a plain LRU.** Commit [`33d6135c`](https://github.com/ernestuz-b/llama-chocolate/commit/33d6135cbe20e90e67d9db1c874fbb4a9dc64b19) (19 October 2026) replaced it with a **50/50 segmented LRU (SLRU)**. A probationary queue gets new admissions, hits promote entries to a protected queue, excess protected entries are demoted, and victim selection scans probation before protected while excluding experts selected by **this layer in this particular `plan()` request**. It can still evict any other layer's slot, including one needed later in the same forward pass. This changes both semantics and worst-case policy cost. The uploaded audit's LRU-specific claims are now historically correct, **not current**.
2. **Most external contracts are unchanged** by the SLRU commit. Host slot maps, internal raw pointers, sharing by same-layout layers, host-WEIGHTS scheduler interception, router-ID readback/synchronization, synchronous-on-the-critical-path reactive admission, and fixed group/device bank capacity all remain.
3. **Native expert-ID remapping and execution are real and reusable facts.** `selected_experts` remains a logical ID tensor; `slots = ggml_get_rows(slot_map, ids)` supplies physical cache indices to the existing `ggml_mul_mat_id` projections. Bias, scale and LoRA paths still use *logical* IDs in several places.
4. **The native cache is not a mixed CPU/GPU expert executor.** In the cache-enabled path, selected resident or freshly uploaded experts are executed via cached GPU banks. In the non-cache/large-batch GPU path, the copy callback can stage host experts and reuse cached copies, but this is still a way to feed a GPU matmul, not a CPU computation of cache misses with a merged GPU result.
5. **The significant serialization remains in source.** Each per-layer cache copy reads device router IDs to host and immediately synchronizes its backend; cache uploads and slot-map publication occur within the same scheduler copy interception. The policy currently runs reactively, not off the compute-critical path.
6. **Fakecache P8 demonstrated partitioned CPU/GPU expert programs, not effective concurrent overlap.** Its execution loop runs CPU and rotating/GPU programs in order, immediately synchronizes each backend, reads output back and merges on host. This distinction matches the prior performance investigation.
7. **No new source evidence establishes throughput, correctness or concurrency guarantees.** Earlier measured slowdowns matter as motivating observations but are not general proof of a particular cause; no new benchmark is claimed here.
8. **Execution *initiation order* is a performance-critical dependency, distinct from computation correctness.** Fakecache's P8 CPU-first loop submitted work and synchronized it before initiating GPU work. In the subsequent Fakecache 10 discussion, the explicit GPU-first code example illustrated how a potentially blocking CPU backend call can nevertheless overlap with previously submitted GPU work. A later review reported **4.45 ms overlap vs zero** in an experiment, but the exact producing commit and raw trace have **not been verified in this audit**. The source-verifiable P8 behavior and later historical report must not be conflated.
9. **Indirect native cache numerical tests do exist.** `tests/test-llama-archs.cpp` defines GPU-dependent MoE-cache cases with small capacities, including host-resident DeepSeek2 expert models and logits comparison/round-trip checks. This is not a direct SLRU policy-decision test.

## B. Precisely what changed since the uploaded audit

A GitHub comparison of `098ce526b` to `39da130fc` reports **35 commits ahead, zero behind**. `src/llama-moe-cache.cpp` changed **+59/−25** lines (from 639 to 673 lines); `src/llama-moe-cache.h` changed only a descriptive comment (**+1/−1**). The cache changes come from the SLRU commit above. `src/llama-graph.cpp`, `src/llama-context.cpp` and the relevant cache interception logic were **not listed as changed** in this comparison. `ggml/src/ggml-backend.cpp` added `GGML_OP_ARGMAX` to `ggml_op_alloc_size_may_expand` (`:2221–2234`); its host-WEIGHTS copy-interception contract is unchanged. This change belongs to general backend allocation/sizing behavior, **not** to SLRU or the cache callback. It is potentially relevant when interpreting overall VRAM accounting, but the diff alone does **not** establish a measurable cache-capacity or free-VRAM effect. A number of other backend files changed (notably Metal, OpenCL, SYCL and CUDA), which could affect platform operation/performance, but the commit comparison **does not demonstrate** a change to the cache management or execution boundary. Source diff: [old](https://github.com/ernestuz-b/llama-chocolate/blob/098ce526b546b5a190327c1da82929cee051b9ac/src/llama-moe-cache.cpp) / [new](https://github.com/ernestuz-b/llama-chocolate/blob/39da130fcd1b6ca47d3c0522a40eccff41ea3dfb/src/llama-moe-cache.cpp).

### SLRU behavioral delta, verified

| Aspect | Previous `moe_cache_lru` | Current `moe_cache_slru` |
|---|---|---|
| Type | One doubly linked LRU list | Probationary (`0`) and protected (`1`) lists |
| New miss | Insert/move to overall MRU | Insert/move to probationary MRU |
| Hit | Move to overall MRU | Move to protected MRU; demote oldest protected when protected count exceeds `n_slots/2` |
| Victim | `head`, O(1) selection | Scan probation then protected; skip a slot **only when its owner is this `il` and its expert belongs to the current request** |
| Worst-case victim lookup | O(1) | O(number of slots) *per miss*; repeated scans can amplify work |
| Current-request selection preservation | Achieved principally through hit promotion and capacity guard | Explicit exclusion in `victim(il)` using `seen`/`seen_gen`, including prior fills; **not** protection for other layers or later calls |
| Cross-layer churn | One group-wide queue, no inter-layer protection | Probation/protected ordering **may** preserve repeatedly used entries better, but foreign-layer victims remain eligible; no reservation spans the remaining layers in one compute |
| Long-term frequency/statistics | None | Still none: segment membership is a small implicit repeated-hit signal, **not** a statistical predictor |
| External cache API, slot maps, callback | Existing | Unchanged |

Source: [`src/llama-moe-cache.cpp:15–154`](https://github.com/ernestuz-b/llama-chocolate/blob/39da130fcd1b6ca47d3c0522a40eccff41ea3dfb/src/llama-moe-cache.cpp#L15-L154). SLRU keeps the generation-stamped deduplication, capacity guard, sorted unique IDs, `fill{expert,slot}` output and foreign slot-map invalidation. The SLRU commit did not add a dedicated policy test. **Indirect cache execution tests already exist:** `tests/test-llama-archs.cpp:1029–1039` adds GPU/iGPU-dependent `MoE cache` and multilayer split configurations with a `1536*1024`-byte budget, documented as a few slots to exercise eviction and re-upload; `:157–165` currently enables host-expert coverage for DeepSeek2, `:1125–1133` compares logits to CPU using an NMSE threshold, and `:1191–1200` checks the save/load round-trip when that path is enabled. Whether these tests ran and passed on any specific GPU was **not** established in this audit. They do not inspect SLRU victim order, segment membership, or statistics.

### Source-visible SLRU caveats

- Policy frequency is **per planning/cache-admission request**, not per physical compute: `touch` is called by `plan` for cache hits, not by the `copy_experts` large-batch read path. The earlier audit's “large-batch reads do not update recency” still applies, but “no evidence of repeat use” is no longer literally true—protected status encodes a limited repeat-hit signal.
- A victim search traverses both segment lists and is therefore no longer constant-time. No measured runtime impact or pathological case is established by this audit.
- Cross-layer entries continue competing for one group's slots. Precisely, `victim(il)` skips a candidate only when `key_of[s] / n_expert == il` **and** `seen[key_of[s] % n_expert] == seen_gen`; a foreign-layer slot is eligible even if its expert number matches a current selected ID. The condition preserves current-request entries, **not** a reservation for later layers in the same graph/ubatch. Probationary MRU admission and protected hits may mitigate cross-layer churn, but no quantitative improvement is demonstrated.
- `init` sets the two list heads/tails and protection flags but does not reset `n_protected` (`:29`, `:35–50`). **Repeated initialization of an existing policy instance with protected occupants would violate its segment-count invariant**: the lists and flags would start in probation while `n_protected` retained a stale count. This is a concrete conditional invariant defect, not a presently exercised failure: current construction calls `init` only once per initialized group. There is no current SLRU resize/re-init or policy-state persistence path.
- For a hypothetically initialized zero-slot SLRU, `head[0]=0` while there are no nodes. Production **skips `init` for zero-slot groups**, so the uploaded audit's zero-slot point describes a representation edge case, not a demonstrated live failure.
- `victim()` ends with `GGML_ABORT("no SLRU victim")` (`:98`). With valid, internally consistent maps and `uniq.size() <= n_slots` (`:124–126`), its exhaustive-scan failure is unreachable during a miss: if every slot were excluded, every slot would already contain a distinct current-request expert, leaving no requested miss that could reach `victim`. This is an invariant guard, not observed crash evidence.
- `is_protected` is `std::vector<bool>` (`:28`), meaning a bit-packed proxy representation rather than a byte array. This is not an observed defect; it matters only to potential direct serialization, instrumentation or atomicity assumptions.

## C. Actual native execution and ownership, at current source

### C1. Eligibility, placement and storage

- `llama_moe_cache` is enabled by a nonzero `cparams.moe_cache_size` and built by `llama_context` at startup. Cache use disqualifies pipeline-parallel scheduling via `cparams.moe_cache_size == 0` in the pipeline condition. `src/llama-context.cpp:515–555`.
- Constructor admits GPU/iGPU devices, rejects tensor split mode, requires an MoE model; a layer is eligible only when **all** its routed expert tensors are host-resident WEIGHTS and the graph's layer-assigned device is among the cache devices. `src/llama-moe-cache.cpp:255–300`.
- Eligible layers are grouped by **device and compatible expert tensor layout** (count, type, shape and expert stride). One SLRU and one set of physical banks exist **per group**, not per layer; a group may contain multiple layers. `:156–177`, `:201–216`, `:275–298`.
- Cache allocation is **fixed during construction**. A device weight `d.split` is derived from tensor-split values, or from `ggml_backend_dev_memory` **once** during construction when tensor split is unspecified; group `g` then receives a nominal budget `(size * d.split / split_sum) * (g.host_bytes / d.host_bytes)` (`:314–350`). The constructor increments `g.n_slots` while the padded size of `n_slots+1`, including the MMQ guard element for each bank, fits that fixed budget (`:304–311`, `:351–353`), capped at `g.layers.size()*n_expert`. No subsequent dynamic free-VRAM feedback/resizing is visible. A group below global `n_expert_used_max()` capacity is disabled; when all groups fall below this threshold, construction fails. `:255–256`, `:304–365`.
- Each expert-tensor bank has an **extra slot** (`n_slots+1`), explicitly documented as a CUDA MMQ guard against reading beyond the final expert. This is an allocation/layout invariant independent of policy. `:304–311`, `:391–413`.
- Each cached layer has its own host `GGML_TYPE_I32` slot-map tensor mapping logical expert IDs to physical slot IDs (or `−1`). Maps are allocated in a CPU buffer, cleared to `0xff`, and tagged as WEIGHTS so the scheduler intercepts transfer. The SLRU holds **borrowed pointers** into those host tensor buffers. `:388–458`.
- `no_alloc` has a size-accounting construction path; cache memory is reported in `llama_context::memory_breakdown()`. These are live integration contracts. `:421–453`, `:662–673` and `src/llama-context.cpp:3700–3714`.

### C2. Cache-enabled graph path (eligible layer/ubatch)

1. Graph construction asks `get_slot_map(il, n_tokens, n_expert_used)`. It returns null if the layer is uncached, the ubatch is empty or exceeds **32 tokens**, or the upper bound `min(n_tokens*n_expert_used, layer_expert_count)` exceeds the group's slots. The bound may reject a ubatch that happens to select fewer distinct experts; it is conservative. `src/llama-moe-cache.cpp:496–507`.
2. Graph builder retains logical `selected_experts`, makes a contiguous, flattened selection as needed, creates `ggml_get_rows(slot_map, ids)`, verifies that the chosen backend supports the lookup, assigns its backend, and expands the lookup into the graph. `src/llama-graph.cpp:2532–2571`.
3. The host map becomes a graph/scheduler input. The scheduler copies **non-host-WEIGHTS inputs first**, then host-WEIGHTS inputs, invoking a copy callback for suitable inputs. Its source comment states: *"the weights in host memory are copied last, so that the copy callback can read the other inputs of the split"* (`:1887`). `ggml/src/ggml-backend.cpp:1817–1821`, `:1823–1849`, `:1886–1897`.
4. `llama_context::sched_copy_experts` forwards the callback to the cache. Cache `copy()` resolves the layer by slot-map **tensor identity**, searches the split graph for the consuming `GGML_OP_GET_ROWS`, and expects router IDs to have been computed in an earlier split. It aborts if selection appears in this same split. `src/llama-context.cpp:2762–2768`; `src/llama-moe-cache.cpp:549–580`.
5. It copies selected router IDs device→host asynchronously **then immediately calls `ggml_backend_synchronize`**. It invokes `slru.plan`, uploads missed experts to GPU banks with opportunistic contiguous-run merges, and uploads the map last. `src/llama-moe-cache.cpp:582–616`.
6. `ggml_mul_mat_id` uses the cached bank view with physical slot IDs. The normal `selected_experts` logical IDs are preserved for bias, scale, LoRA, and related operations. `src/llama-graph.cpp:1669–1710`, `:2333–2380`, `:2468–2480`.

This mechanism is **router-reactive**: a requested miss is uploaded before the GPU cached path can use it. No source-visible CPU execution of that cache miss is substituted at this boundary.

### C3. Non-cache / large-batch GPU execution is different from CPU fallback

When `get_slot_map` returns null, graph-building uses the ordinary `ggml_mul_mat_id(w, cur, selected_experts)`. The generic `sched_copy_experts` path may then inspect selected logical IDs and stage just the routed host experts into a GPU destination. It optionally copies the already-resident run from the device cache, then uploads nonresident runs from host; it preserves CUDA MMQ padding. This optimization **uses the cache as a source for filling a temporary GPU weight copy**, without creating a mixed CPU/GPU expert computation. `src/llama-graph.cpp:1676–1679`; `src/llama-context.cpp:2770–2843`; `src/llama-moe-cache.cpp:514–547`. Exactly where an *uncached* graph executes depends on the scheduler/backend assignment; this audit does not claim all non-cache graphs run on GPU.

### C4. Publication/synchronization/lifetime facts

- The host slot map is mutable across calls, and its returned `ggml_tensor *` is **not an immutable policy snapshot**. `get_slot_map` is `const` but shares writable tensor storage. `src/llama-moe-cache.cpp:496–507`.
- A normal reactive `copy` orders expert bank uploads before slot-map upload; it documents reliance on the **next** callback's backend synchronization before rewriting the host map. Queued DMA source-lifetime management is therefore part of the behavior. `:582–616`.
- The scheduler has copy-buffer reuse events (`events[backend][cur_copy]`) protecting its own split-copy buffer reuse, but those events do **not by themselves** protect the cache's persistent **host source** when a prior H2D read is outstanding. Removing the present synchronization without providing equivalent source-lifetime/order guarantees would introduce a mapping data race. Generation-separated host-map storage and event-ordered publication are one possible mechanism discussed in the independent review, **not an established uniquely required implementation**. `ggml/src/ggml-backend.cpp:1823–1849`. This is directly material to R2.
- The class is internal to the cache `.cpp`. `impl::group` holds the policy directly; construction writes its borrowed map pointers, while the large-batch callback reads them directly. `src/llama-moe-cache.cpp:13–154`, `:201–249`, `:454–458`, `:525–531`.
- The scheduler condition for special interception (`host buffer` **and** `WEIGHTS usage`) is duplicated in a similar predicate in the cache. Maintaining the meaning of that host-WEIGHTS contract is essential to understanding this behavior. `src/llama-moe-cache.cpp:179–183`; `ggml/src/ggml-backend.cpp:1817–1821`.

### C5. Data and telemetry visible today

- Internal cache accounting contains `hits`, `misses`, and `bytes`, two classification buckets labelled `ubatch <= 8` and `ubatch > 8`, plus cached-to-GPU copies during large batches. These are cumulative aggregate counts logged at destruction. `src/llama-moe-cache.cpp:224–236`, `:608–633`.
- The small/large stats bucket uses `ids.size() <= 8*n_expert_used`, where `n_expert_used` is the model-wide **maximum**. On heterogeneous top-k architectures the log labels need not correspond exactly to eight tokens for the current layer. This does **not** mean all runtime capacity gating is model-wide: the graph passes the current selected tensor dimensions into `get_slot_map`.
- SLRU has only recency/segment state and temporary `seen`/`uniq` arrays, **not** persistent per-expert use frequency, transition distributions, deadline evidence or historical prediction scores. Protected/probationary membership plus order encodes approximately one repeat-hit distinction, not a counter of how often a given logical expert was selected.

## D. Earlier Fakecache integration audit — verified compatibility and differences

| Subject | Already present in native Chocolate baseline | What Fakecache P8 added or showed | Important boundary of the evidence |
|---|---|---|---|
| Logical→physical expert IDs | `get_rows(slot_map, selected_experts)` plus existing GPU `mul_mat_id` | General placement representations and logical/physical-ID separation | Same conceptual remap, but logical ID must remain authoritative outside the weight matmul |
| Router/post-selection boundary | Scheduler splits at host-WEIGHTS slot-map lookup; callback gets split graph | Explicit router-frontier discovery, graph scopes and scheduler/split interception | Native boundary is conditional on host-WEIGHTS eligibility and split assumptions; not a generic expert executor callback |
| GPU banks/slot storage | Grouped same-layout banks and cache mapping | Resident and transient/rotating banks with separate lifetime states | Native owns only fixed group cache; no independently executable rotating epoch semantics |
| Weight transfer | Reactive H2D missing-expert uploads, merged consecutive runs, generic selective copies for uncached graph | Ahead-of-use epoch transfers and READY/FREE lifecycle tests | Native transfer callback still synchronizes on router readback; it does not prove a nonblocking predictive runway |
| CPU execution for routed misses | **Not present as a native per-expert choice** inside cached `mul_mat_id` | CPU/GPU partition and output merge demonstrably implemented | Fakecache P8's inner program loop synchronizes between CPU and GPU, so overlap is **not proven** |
| Statistics and policy | LRU historically; now SLRU; hits/misses/bytes | Counter/statistics provider, ranking/rotating policy and published presence plans | Native SLRU protected segment is not equivalent to Fakecache statistical history |
| Graph integration | Native graph builder, normal scheduler/bank views, copy callback | Externalized graph replacement, graph-scope and custom scheduler seams | Native indirection reduces some duplication but does not implement partitioned CPU/GPU graph execution |
| Multi-token handling | Graph builder makes selected IDs contiguous/flat; per-layer admission can be gated off | Fakecache had multi-token frontier/contiguity limitations in earlier milestones | These are **distinct** code paths; the earlier Fakecache guard is not automatically a present native bug |
| Lifetime and memory accounting | Constructor allocation, extra MMQ slot, `memory_breakdown`, scheduler events/realloc callback | Producer/consumer READY/FREE, resident/rotating state, admission, fixed/elastic concepts | Native reallocation callback is mainly an *observer* of graph reallocation; it does not dynamically resize banks |

Fakecache P8 source check: [`src/fakecache/expert-execution.cpp:824–919`](https://github.com/ernestuz-b/Fakecache/blob/895e306b96a9e1c8d9ae3c8bddf8e8287f786c69/src/fakecache/expert-execution.cpp#L824-L919) partitions routes using grouping, runs `c=0` CPU and `c=1` rotating programs in series, performs `ggml_backend_graph_compute_async` followed by `ggml_backend_synchronize`, copies each result back and adds it to the shared publication buffer. This validates the earlier audit's **lack of CPU/GPU overlap in that implementation**; it does not establish impossibility of overlap in some other implementation.

Fakecache's `cache_controller` documented that plans were published **between computes**, not mid-graph. That is established Fakecache policy behavior, **not something the current native cache provides**: [`src/fakecache/cache-controller.h:28–43`](https://github.com/ernestuz-b/Fakecache/blob/895e306b96a9e1c8d9ae3c8bddf8e8287f786c69/src/fakecache/cache-controller.h#L28-L43). Historical Fakecache experiments also demonstrated explicit slot mapping, cross-backend event ordering and failure recovery; those experiments do not by themselves establish performance after integration with this newer upstream cache.

### Historical measurements from the 18 October discussion (not repeated here)

| Configuration discussed | Reported throughput / counts | Evidential interpretation |
|---|---|---|
| Native upstream cache, 4 GiB, generation | **8.92 tok/s** | One measured environment, not a universal cache result |
| Same discussion, cache disabled | **16.9–17.0 tok/s** | Demonstrates material observed regression in that comparison; does not by itself isolate the cause |
| Native cache, small-ubatch routing | **1,024,912 hits**, **1,183,139 misses**, **46.42% hit rate** | Counts were recorded in prior audit; hit rate alone omits synchronization, transfers, splitting, backend work |
| Native cache, larger ubatches | **5.74% hit rate** | Consistent with small residency opportunity and conservative cache gate; no causal proof without workload controls |
| Native cache, uploads | **~3.38 TiB** | Transfer volume reported previously; its latency and overlap were not separated by this audit |
| Native cache, prefill | **163.86 tok/s** cached vs **190.25 tok/s** disabled | The cache's costs were visible in that setup; not reproduced here |
| Graph split comparison | **51** cache-on vs **99** cache-off | Fewer splits do not imply faster execution; other work and waits changed |
| Fakecache P8 comparison in earlier discussion | **11.50 tok/s** disabled vs **6.16 tok/s** ordinary fallback after OOM vs **6.21 tok/s** streaming | **Different experiment** from the native comparison. Failed fallback could retain Fakecache scheduler split overhead; cause was *suspected*, not proven. The 6.16 run must not be labelled an ordinary cache-off baseline. |

These numbers are **historical statements from the earlier conversation**, not fresh artifact measurements; complete configuration logs, model identity, workload and run-to-run distributions were not reproduced in this audit. Comparison across the two experiments is **not** apples-to-apples. The source does prove that cache-on and cache-off can alter graph construction, scheduler boundaries, copy paths and pipeline-parallel eligibility; therefore a raw tok/s difference cannot be assigned to replacement policy alone.

## E. Reconciliation with the uploaded audit's leak register

The ID numbers below match the uploaded independent audit. **Status** means validity against *current* code, not a priority or task order. Line numbers have been translated from its `098ce526b` baseline to current `39da130fc` where practical.

| ID | Current status | Verified fact and adjustment |
|---|---|---|
| **LR-01** foreign mutation on eviction | **Confirmed, unchanged** | `slru.plan` still writes `slot_map[old_layer][old_expert] = -1` through borrowed pointers (`:143–149`). Eviction has no separately returned victim/reason/cost channel. Claim that a given future policy is “impossible” is a design judgment, not a source-level proof. |
| **LR-02** direct field access | **Confirmed, updated name** | Constructor writes `g.slru.slot_map[il]` (`:454–458`); large-batch copy reads it (`:525–531`). Large-batch reads do not update protection/recency. |
| **LR-03** const getter, mutable map | **Confirmed** | `impl::get_slot_map(...) const` returns an aliased mutable `ggml_tensor *` (`:496–507`); graph reads that tensor later (`llama-graph.cpp:2561`). There is no snapshot/version API in native cache. A specific `shared_ptr` snapshot scheme in the uploaded audit is a *proposal*, not an observed necessity. |
| **LR-04** reuse protected by synchronization | **Confirmed** | Router D2H readback synchronizes (`:583–584`); next callback sync protects prior map source before it is mutated (`:613–614`). The exact required replacement mechanism was not established by this audit. |
| **LR-05** capacity failure abort | **Confirmed with reachability qualification** | `plan` returns false when actual distinct IDs exceed slots (`:124–126`), `copy` aborts (`:587–589`). The earlier conservative `get_slot_map` gate (`:503`) makes this unlikely/unreachable for valid selections entering this path; it is **not evidence of frequent runtime aborts**. |
| **LR-06** topology encoded in SLRU | **Confirmed, scope qualified** | `key_of = il*n_expert+e` with global layer-sized pointer table (`:20–21`, `:35–38`, `:144–149`, `:360`). This is a coupling and extra metadata cost, not a proven correctness fault. |
| **LR-07** internal type in impl | **Confirmed, name changed** | `moe_cache_slru` is anonymous-namespace and its `fill` appears in `impl::fills` (`:16`, `:245`). Direct independent linkage/tests are awkward, but “cannot test” is too absolute (tests can access through higher-level behavior or special harnesses). |
| **LR-08** callback contract outside policy | **Confirmed** | `layer_of` tensor identity (`:415`), host WEIGHTS usage (`:465–470`), scheduler predicate (`ggml-backend.cpp:1817–1821`) and late-copy order (`:1886–1897`) collectively make interception work. Native caller contracts have not changed. |
| **LR-09** multiple contiguous-run encodings | **Confirmed; more accurately layout coupling** | Sorting (`:138`) shapes placement to make contiguous expert/slot upload runs likely; actual run-merge logic is present in cache `copy` (`:597–605`), cache `copy_experts` (`:530–541`) and context `sched_copy_experts` (`:2806–2841`). It is three different range-handling paths plus a policy-level layout choice, not four identical algorithms. |
| **LR-10** batch thresholds and max-top-k | **Confirmed, narrowed** | `max_batch=32` (`:230`), bucket classified using model-wide max (`:232`, `:608`), and minimum group slots also use model-wide max (`:355`). However `get_slot_map` gets current layer's top-k from the selected tensor (`llama-graph.cpp:2543`). The claim “wrong capacity arithmetic everywhere” was overstated. |
| **LR-11** zero-slot validity | **Representation caveat, not observed production failure** | `slru.init(…,0)` would set `head[0]=0` (`:46`), but construction skips `init` for groups with `g.n_slots==0` (`:355–360`), and they never expose a map. |
| **LR-12** misleading pure LRU semantics | **Superseded / partly survives** | The object is now called `slru` and *does* distinguish an expert hit from a new admission using one protection bit/list membership and list position. It observes `plan` hits, not every physical execution; `seen` is request-scoped, and no per-expert count, decay, confidence or transition evidence is retained. |
| **LR-13** member destruction ordering | **Latent lifetime constraint; no active UAF shown** | `groups` precede `ctx_host` / `buf_host` in member declaration (`:238–249`), so host storage dies before the vector of holders with borrowed pointers. Current policy destructors don't dereference them. The uploaded audit's projected UAF requires hypothetical teardown logic, not observed source behavior. |
| **LR-14** per-layer host tensor interception | **Confirmed, unchanged; see F-13** | One slot map per cached layer (`:413–416`), one graph lookup (`llama-graph.cpp:2561`) and per-layer callback scan/sync (`:560–584`). Frequency/CPU cost was **not measured** here; any alternative map granularity remains hypothetical. |

Additional precision: the independent audit's proposed immutable snapshots, `score_table`, `publisher`, `transfer_engine`, `capacity_controller`, `plan_result`, and its migration checklist are **design content**. They have intentionally **not** been adopted as audit conclusions. Likewise, its labels “blocking for R2/R3” refer to that document's desired future requirements, not a present runtime severity classification.

## F. Additional cross-audit findings not adequately captured by either audit alone

**F-01 — Two meanings of “fallback.”** A cache miss during the *cached* path causes a GPU upload, whereas a cache *bypass* switches to an ordinary graph path. The ordinary path can still invoke selected-host-expert GPU staging. Neither is the mixed CPU/GPU reactive executor demonstrated experimentally by Fakecache. Source: cache `:496–507`, `:549–616`; context `:2762–2843`; graph `:1669–1680`.

**F-02 — Cache admission remains in the critical path.** The SLRU is invoked *after* device router-ID readback and synchronization, inside the scheduler copy callback that prepares the consuming graph split. The policy's operations may be relatively small, but their timing and transfer demands are tied to the current layer. Source: cache `:549–616`; scheduler `:1823–1849`, `:1886–1897`.

**F-03 — The conservative upper-bound gate limits opportunities before policy is consulted.** For eligible ubatches under 32 tokens, `min(n_tokens*top_k, n_expert) > n_slots` bypasses cache even if many routed IDs repeat and the observed unique set would fit. This matters when interpreting low hit rates; it does **not** prove a single causal explanation for them. Source: cache `:496–507`.

**F-04 — Layer- and backend-eligibility constraints are strong.** All experts must be host WEIGHTS and the layer must map to a cache GPU; tensor parallelism is explicitly rejected; the `GET_ROWS` backend capability is separately checked. “Native cache supports MoE” does not imply support for every MoE architecture, backend and placement. Source: cache `:255–300`; graph `:2543–2565`.

**F-05 — CPU RAM is the canonical expert source in eligible cache groups.** Cache banks are derived copies of host-resident expert tensors. Native code has no independent ownership of an authoritative GPU-only expert. This is an important agreement with Fakecache's canonical-host-weight experimental assumption. Source: cache `:275–298`, `:403–416`, `:591–604`.

**F-06 — SLRU performance assertions are presently source-only.** The 50/50 policy protects hits and scans for victims; no comparative SLRU hit-rate, overhead or token/s data were available in the verified commit/diff. A policy change by itself does not establish improvement or regression.

**F-07 — Source code shows serialized Fakecache P8 CPU/GPU groups, not merely a suspicion.** `expert_transaction::execute` invokes CPU then rotating/GPU programs inside its group loop; each `program.run` reaches `ggml_backend_graph_compute_async`, immediately calls `ggml_backend_synchronize`, and retrieves the result before the next path. The inline producer-start mechanism concerns streaming **bank fills**, not proof of overlapping CPU/GPU expert compute. This is stronger evidence for the absence of the intended compute overlap in that version than the historical throughput numbers. Source: Fakecache `expert-execution.cpp:878–919` (P8); the later reported overlap experiment is distinct and not attested by this commit.

**F-08 — The mapping's correctness depends on mixed index domains.** Weights use physical slots only when cache-backed. Biases, per-expert scales, LoRA adapter matmuls and selected logical identity remain tied to expert IDs. Remapping every operation blindly would change semantics. Source: graph `:1669–1710`, `:2333–2380`, `:2468–2480`.

**F-09 — The cache's aggregation and cost observability are incomplete.** It reports hits, misses and byte totals, not per-layer/group residency score, individual eviction identities, callback wall time, sync duration, H2D concurrency or CPU/GPU overlap. The source therefore cannot by itself apportion throughput regression among admission, stalls, copied bytes, graph placement and kernel dispatch.

**F-10 — “No allocations in steady-state planning” is narrower than “zero runtime overhead.”** The policy reuses its arrays and a generation stamp, but SLRU now scans slot lists, host router IDs are resized/read back, backend sync happens, and DMA/callback work follows. Initialization and eventual `uniq`/`fills` vector capacity growth may allocate. Source: cache `:31–50`, `:108–153`, `:582–616`.

**F-11 — There is no production resize/re-initialization or persistence protocol for policy state.** SLRU is initialized during cache construction (`:360`); cache bank and map storage are also constructed there. No current source path re-initializes SLRU or restores its protected/probationary ordering from persisted state. The stale-`n_protected` condition above becomes a concrete state-lifecycle constraint if reinitialization is introduced; it does **not** imply an active cache correctness bug.

**F-12 — No runtime VRAM-slack feedback loop is present.** Free memory is read, where applicable, for **initial** proportional budget distribution (`:314–353`), not continuously. `memory_breakdown()` reports allocated cache buffers (`:662–673`) into context memory reporting (`llama-context.cpp:3710–3714`). These facts are relevant to R1 and accounting under changing capacity; they do not prescribe a growth/shrink mechanism.

**F-13 — Slot-map interception is currently per layer.** Each eligible layer gets its own host map (`:413–416`) and `GET_ROWS` lookup (`llama-graph.cpp:2561`), triggering a callback with a graph scan (`:560–568`) and synchronous ID readback (`:582–584`). Other map granularities may change interception count and lifetime scope, but no alternative has been implemented or measured here.

**F-14 — Transfer layout is coupled to policy ordering.** Sorting unique IDs (`:138`) creates opportunities to upload adjacent selected experts into contiguous target slots; cache `copy` coalesces such runs (`:597–605`), while large-batch cached copies and the generic selected-host-expert path have their own run handling. The present policy therefore affects transfer shapes as well as who survives eviction. Under a hypothetical chunked allocator, this becomes an occupancy/contiguity question; no chunk layout exists in the current code.

## G. Execution initiation and ordering — Fakecache 10 performance lesson

This section records the **specific execution-order observation** from the Fakecache 10 discussion, not an implementation plan. It concerns when work is **submitted**, not merely which backend computes which expert. Submission and completion are different events, and a function with an `*_async` name does not guarantee nonblocking behavior on every backend.

### G1. The illustrative C++ ordering (from the earlier discussion)

The earlier example was conceptually:

```cpp
// The router has finished; both programs' required inputs are ready.
// On an asynchronous GPU backend, enqueue GPU work first.
ggml_backend_graph_compute_async(gpu, gpu_graph);

// This CPU-backend call may execute synchronously on the calling thread.
// If so, GPU execution can progress concurrently with CPU work.
ggml_backend_graph_compute_async(cpu, cpu_graph);

// Only at the result-dependency boundary, ensure outstanding work is done.
ggml_backend_synchronize(gpu);
// A truly asynchronous CPU backend would also require CPU completion here.
// ggml_backend_synchronize(cpu);
merge_expert_outputs();
```

**Why order matters:** if the CPU call blocks, submitting **CPU first** delays the GPU submission until CPU work is over. Submitting **GPU first** permits overlap while the CPU call runs. This is *not* guaranteed by either call's name: GPU work must actually be enqueued without a hidden blocking wait; CPU behavior varies by backend; input copies must have completed or be correctly ordered; and any asynchronous CPU work must be complete before merge. The example is deliberately schematic—it is **not** production-ready error/lifetime handling or a claim that synchronization can simply be removed.

```text
Accidentally serialized (CPU call blocks):
  CPU submit + execute + return ────────> GPU submit + execute ─────> merge

Potentially overlapping (GPU submit does not block):
  GPU submit ── GPU executing ─────────────────────────────────────> merge
                 CPU submit + execute + return ──────┘
```

The more general invariant, independent of a particular cache policy, is: **initiate independent, dependency-ready work before entering a potentially blocking operation, and wait at the first genuine consumer dependency.** A premature weight-READY wait, synchronous D2H routing readback, host tensor-copy API, backend/scheduler split transition, event wait, or buffer-reuse barrier can defeat overlap *even if graph compute is nominally async*. A GPU-first call is only safe when its own prerequisites are satisfied; it is not a license to submit against unready data.

### G2. Evidence strength and relation to the cache

- **P8 source demonstrated serialization:** `expert-execution.cpp:878–919` submitted CPU and GPU program work in the same loop and synchronized each before proceeding. The earlier producer-thread/bank-fill overlap did not establish compute overlap.
- **Historical Fakecache 10 follow-up:** the conversation reports an Agent 3 review measuring **4.45 ms CPU/GPU overlap**, compared with zero in the older arrangement, with a CPU interval around **143 ms**. The full trace, benchmark configuration and exact producing commit have not been independently located in this audit. Treat the numbers as historical reported evidence, **not reproduced source-verified measurements**. The report did not demonstrate full-model throughput recovery.
- **Chocolate source today:** native `copy()` still performs backend synchronization after router-ID readback (`llama-moe-cache.cpp:582–584`) before choosing/uploads misses, and scheduler copy input processing can synchronize or enqueue event waits (`ggml-backend.cpp:1823–1862`). The cache provides no mixed CPU/GPU complement launch as of the audited commit. Consequently, native async GPU operations alone do not establish useful CPU/GPU overlap for uncached routed experts.

**Audit implication:** correctness-only assertions about CPU/GPU partitioning would miss a failure mode previously observed in Fakecache: valid outputs with work effectively serialized by initiation order or hidden waits. Actual overlap requires timing evidence from both execution intervals and their submission/dependency boundaries; speed comparisons alone cannot identify the stall.

---

## H. Source map and evidence limits

Main current source permalinks:

- [Current SLRU and its `plan`](https://github.com/ernestuz-b/llama-chocolate/blob/39da130fcd1b6ca47d3c0522a40eccff41ea3dfb/src/llama-moe-cache.cpp#L15-L154)
- [Grouping, eligibility, fixed budget and buffer allocation](https://github.com/ernestuz-b/llama-chocolate/blob/39da130fcd1b6ca47d3c0522a40eccff41ea3dfb/src/llama-moe-cache.cpp#L187-L470)
- [Cache graph gate, callback, upload, stats](https://github.com/ernestuz-b/llama-chocolate/blob/39da130fcd1b6ca47d3c0522a40eccff41ea3dfb/src/llama-moe-cache.cpp#L496-L633)
- [Graph cache slot lookup](https://github.com/ernestuz-b/llama-chocolate/blob/39da130fcd1b6ca47d3c0522a40eccff41ea3dfb/src/llama-graph.cpp#L2532-L2572)
- [Logical-vs-physical matmul/bias/scale/LoRA handling](https://github.com/ernestuz-b/llama-chocolate/blob/39da130fcd1b6ca47d3c0522a40eccff41ea3dfb/src/llama-graph.cpp#L1669-L1710)
- [Scheduler host-WEIGHTS interception and ordering](https://github.com/ernestuz-b/llama-chocolate/blob/39da130fcd1b6ca47d3c0522a40eccff41ea3dfb/ggml/src/ggml-backend.cpp#L1817-L1897)
- [Context callback, selected-expert GPU copies](https://github.com/ernestuz-b/llama-chocolate/blob/39da130fcd1b6ca47d3c0522a40eccff41ea3dfb/src/llama-context.cpp#L2762-L2844)
- [P8 Fakecache CPU/GPU execution source](https://github.com/ernestuz-b/Fakecache/blob/895e306b96a9e1c8d9ae3c8bddf8e8287f786c69/src/fakecache/expert-execution.cpp#L824-L919)
- [P8 Fakecache controller publication contract](https://github.com/ernestuz-b/Fakecache/blob/895e306b96a9e1c8d9ae3c8bddf8e8287f786c69/src/fakecache/cache-controller.h#L28-L43)
- [Conditional architecture tests of MoE cache](https://github.com/ernestuz-b/llama-chocolate/blob/39da130fcd1b6ca47d3c0522a40eccff41ea3dfb/tests/test-llama-archs.cpp#L1023-L1039)
- [Architecture logits comparison and save/load round-trip](https://github.com/ernestuz-b/llama-chocolate/blob/39da130fcd1b6ca47d3c0522a40eccff41ea3dfb/tests/test-llama-archs.cpp#L1119-L1200)
- [Backend allocation-size expansion classification (`GGML_OP_ARGMAX`)](https://github.com/ernestuz-b/llama-chocolate/blob/39da130fcd1b6ca47d3c0522a40eccff41ea3dfb/ggml/src/ggml-backend.cpp#L2221-L2234)

**Confidence:** high for current source-level facts and the SLRU commit delta, including the verified indirect architectural test definitions; high for the explicitly re-read P8 sequential-execution code; moderate for prior-discussion performance/overlap reports because this revision did not recover raw traces and configurations. No new inference-speed, overlap, elastic-capacity, no-stall publication, backend-portability or future-implementation numerical equivalence claim is made.

**Scope boundary:** this is an audit of *what exists and what was already demonstrated*, not a claim that any specific replacement implementation, class hierarchy, snapshot design, schedule, buffer geometry, or migration sequence is required. The C++ example records the performance lesson from the previous discussion, not a normative API contract.

**Independent review reconciliation:** adopted the corrected SLRU victim scope, cross-layer observations, `n_protected` re-init invariant, architecture-test coverage, current allocation formula, scheduler comment, host-map lifetime risk and run/layout coupling. Retained the `GGML_OP_ARGMAX` observation only as a bounded allocator-context note; did **not** infer that the change affects Chocolate cache VRAM measurably. Did **not** elevate an unexercised re-init defect into a present bug, assume an immutable snapshot implementation is mandatory, or treat historical overlap measurements as newly reproducible results.
