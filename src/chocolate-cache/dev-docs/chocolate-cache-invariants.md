# Chocolate Cache — Invariants
The finished project MUST comply with the invariants listed here.

## Invariants
### Experts MUST be spread over all the devices so they can be executed in parallel. 
For instance, if CUDA0 and CUDA1 devices are available, the cache should spread the experts across both devices so execution of them can be made in parallel.

### CPU RAM is authoritative. 
GPU residency is optional. An uncached expert executes using its canonical CPU weights.

### Cache management stays off the critical path.
Promotion, demotion and transfers must not unnecessarily delay inference.

### No mandatory transfer on a cache miss.
CPU execution is a normal path, not an emergency fallback requiring synchronous GPU admission.

### Safe residency publication.
An expert becomes GPU-resident only after its transfer completes. Storage cannot be reused while an outstanding GPU operation still needs it.

## No regression in the zero-hit case.
If no experts are available in VRAM, execution should perform approximately as well as ordinary CPU execution.

## Correct results regardless of placement.
CPU-only, mixed CPU/GPU and all-GPU execution must preserve the model's numerical behavior. Predictions affect performance, never correctness.

## correctness alone isn't sufficient.
A Chocolate implementation that produces perfect numerical results but serializes CPU and GPU execution violates our performance requirements.


