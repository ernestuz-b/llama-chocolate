# Chocolate Cache — Invariants
The finished project MUST comply with the invariants listed here.

## Invariants
### Experts MUST be able to execute in parallel across available devices when beneficial.
Chocolate must support distributing experts across devices (for instance CUDA0, CUDA1 and CPU) so independent expert work can execute concurrently. Distribution is not mandatory when it would add more overhead than benefit; no device should be forced to execute experts solely to satisfy a placement rule.

### CPU RAM is authoritative. 
GPU residency is optional. An uncached expert executes using its canonical CPU weights.

### Cache management stays off the critical path.
Promotion, demotion and transfers must not unnecessarily delay inference.

### No mandatory transfer on a cache miss.
CPU execution is a normal path, not an emergency fallback requiring synchronous GPU admission.

### Safe residency publication.
An expert becomes GPU-resident only after its transfer completes. Storage cannot be reused while an outstanding GPU operation still needs it.

### No regression in the zero-hit case.
If no experts are available in VRAM, execution should perform approximately as well as ordinary CPU execution.

### Correct results regardless of placement.
CPU-only, mixed CPU/GPU and all-GPU execution must preserve the model's numerical behavior. Predictions affect performance, never correctness.

### Correctness alone isn't sufficient.
A Chocolate implementation that produces perfect numerical results but serializes CPU and GPU execution violates our performance requirements.

### Execution initiation MUST preserve concurrency.
After routing and when inputs are ready, initiate independent GPU work on all participating GPUs before entering potentially blocking CPU computation. A CPU backend's `compute_async` call may block its caller even though GPU submission is asynchronous; launching CPU work first can therefore serialize execution. Avoid hidden waits in copy callbacks, transfers and scheduler transitions, and synchronize only where results are genuinely required, after independent work has had an opportunity to start. Verify actual overlap by measurement.

### Chocolate cache memory usage is elastic.
The cache grows and shrinks, depending on the available VRAM across all devices. It will leave a slack, a portion of free memory, in each device, and it will try to keep it at the correct size plus a hysteresis.

### Elastic eviction releases whole chunks.
Storage detects low free VRAM and requests that the controller release capacity (possibly through a queue). At a safe point, the controller uses previously prepared chunk values to select the chunk(s), marks every occupant's residency as CPU RAM, and makes the chunks unavailable. Storage then frees each whole chunk once outstanding users have finished. Experts spill naturally to CPU and may be opportunistically promoted elsewhere later; no forced relocation, compaction or synchronous repopulation is needed. Statistical history survives eviction, and maintenance must not block ordinary inference. See [Elastic chunk lifecycle](chocolate-cache-elasticity.md).

### The 'statician' and other metadata operators should be kept out the critical path.
Moving the statistician off the inference path must not introduce a blocking handoff in its place. Most metadata operations should not compete with inference for time, the only allowed ones will be siple ones only affecting a small subset, like updating a counter on each routed expert.

### LoRA execution should be possible and correct.
A distributed Chocolate executor cannot simply bypass build_lora_mm_id() and expect LoRA compatibility to survive. Preserving the complete LoRA-aware computation is an explicit requirement of the execution integration.

