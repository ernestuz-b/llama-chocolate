# Chocolate Cache — Elastic chunk lifecycle

**Status:** agreed design requirements; not an implementation claim about llama.cpp's current fixed-size MoE cache.
See [Chocolate Cache invariants](chocolate-cache-invariants.md) and the [native-cache surface audit](moe-cache-lru-surface-audit.md).

## Storage model

- GPU cache capacity consists of independently allocated chunks containing whole expert slots. The storage owner allocates/releases chunks; the controller owns expert residency metadata and decides which chunks to retain.
- Every expert has canonical weights in CPU RAM. The controller's `location[layer][expert]` is **0** when there is no *usable* GPU residency, and otherwise encodes the group and slot. Zero does not imply that no physical GPU copy is still draining.
- Resident experts are not relocated to different slots for resizing. Growing adds chunks; shrinking removes entire chunks. Unaffected chunks retain their slot identities.
- Grow/shrink decisions account for per-device free-VRAM slack and hysteresis. The controller can value candidate chunks using the statistics of their occupants; the exact valuation formula is not specified here.

## Shrinking: evict the entire chunk

**Selecting a chunk for release means evicting every expert resident in that chunk.** This is not a partial eviction followed by compacting or copying survivors into other chunks.

1. **Select and retire.** The controller chooses a chunk to relinquish, and it becomes unavailable for new admissions or GPU work. No new execution may begin using the retiring chunk after the residency change is published.
2. **Demote all occupants.** At a coherent publication boundary, invalidate the GPU location of *every* expert in the chunk (`location[layer][expert] = 0`), and update the reverse residency metadata consistently. Later routing uses canonical CPU weights for those experts. Their scores, `last_use`, counts and other statistical history remain intact.
3. **Drain existing users.** Already-submitted GPU work and transfers that reference the old chunk must finish before its memory can be reused or freed. Retire/drain bookkeeping remains valid even after the logical expert locations have been cleared. This work must not impose a blocking wait on ordinary inference.
4. **Release.** Once the chunk has no outstanding users, storage releases the **whole allocation** and the controller removes its capacity/slot bookkeeping. No surviving chunk is moved.

An expert demoted because its chunk is released can later be promoted again to *any* available slot, using the normal promotion process. Such a promotion is a new transfer and becomes visible only when ready.

## Growth and promotion

When capacity permits, storage adds a chunk without reallocating or relocating other chunks. The controller selects an expert and slot; until the weight copy completes, that expert continues to have location 0 and can execute from CPU RAM. Only a completed, usable GPU copy is published as resident.

## Ownership and correctness boundaries

- **Controller:** select/retire chunks, choose all affected residents, maintain both directions of residency metadata, publish nonresident/resident states.
- **Storage:** allocate and release physical chunks, resolve encoded group/slot locations, protect allocations until their GPU users have completed.
- **Statistician:** supply expert usage history used to value residents/chunks; eviction does not erase history.
- **llama.cpp integration:** ensure new execution uses the published locations and earlier graph/backend uses are respected.

A chunk's logical eviction and physical release are different events. The ordering must prevent both new readers of a retired chunk and use-after-free by earlier readers; the precise synchronization mechanism is an implementation decision. Cache maintenance should stay off the inference critical path.
