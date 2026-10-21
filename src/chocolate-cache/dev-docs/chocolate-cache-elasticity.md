# Chocolate Cache — Elastic chunk lifecycle

**Status:** agreed design requirements; not an implementation claim about llama.cpp's current fixed-size MoE cache.
See [Chocolate Cache invariants](chocolate-cache-invariants.md) and the [native-cache surface audit](moe-cache-lru-surface-audit.md).

## Storage model

- GPU cache capacity consists of independently allocated chunks containing whole expert slots. Storage allocates and frees physical chunks; the controller owns expert residency metadata and decides which chunks to evict.
- Every expert has canonical weights in CPU RAM. The controller's `location[layer][expert]` is **0** when there is no usable GPU residency, and otherwise encodes the group and slot.
- Resident experts are never physically relocated merely to resize the cache. Growth adds chunks; shrinkage discards whole chunks. The slot identities of surviving chunks remain unchanged.
- Storage observes device memory pressure and the required free-VRAM slack/hysteresis. The controller maintains chunk valuations based on its expert statistics **ahead of time, off the inference critical path**. The exact valuation formula remains to be designed.

## Shrinking: storage requests, controller decides

**Releasing a chunk means evicting every resident expert in that chunk.** Storage detects the need; it does not choose victims or manipulate expert metadata.

1. **Pressure notification.** Storage detects low available memory (relative to the desired slack/hysteresis) and asks the controller to free capacity. The request may be queued so that storage never forces a synchronous cache-management operation onto inference.
2. **Safe-point decision.** At a safe point, the controller consumes pending requests and consults its *already prepared chunk values* to choose the lowest-value chunk(s) needed. It should not run expensive statistics or re-rank the entire cache in this critical section.
3. **Publish eviction.** At that safe point, the controller sets `location[layer][expert] = 0` for **every** occupant of each chosen chunk, updates its reverse maps coherently, and marks the entire chunk unavailable for further cache use. Its experts now naturally follow the CPU RAM path. Scores, `last_use`, counts and other accounting remain intact.
4. **Release.** The controller instructs storage to free the retired chunk(s). Storage releases the whole allocation as soon as it is safe to do so; no compaction, survivor copying or immediate replacement transfers are required.

**Safe-point requirement:** invalidating locations prevents *new* GPU readers, but is not by itself proof that *earlier* GPU readers or transfers have completed. Either the chosen safe point already guarantees that no operation still uses the retired chunk, or storage defers the physical free until those uses finish. Normal inference must not wait for this maintenance.

## Natural spill and repopulation

No explicit migration pass follows chunk release:

- Evicted experts execute using their authoritative CPU RAM weights.
- Their metadata remains available and continues to be observed; future controller decisions can promote valuable experts into slots in other surviving chunks, if capacity and policy permit.
- Promotion is opportunistic, never a mandatory transfer on the first cache miss. A new GPU location is published only after the new copy is usable.
- Other chunks and their resident experts are not moved simply because one chunk disappeared.

This is the intended *natural* reorganization: as residency metadata is consulted and policy decisions continue, the cache adapts without a synchronous reshuffle.

## Growth and promotion

When capacity permits, storage adds a chunk without relocating other chunks. The controller selects an expert and a free slot; until its weight copy completes, that expert continues to have location 0 and can execute from CPU RAM. Only a completed, usable GPU copy is published as resident.

## Ownership boundaries

- **Storage:** observe free VRAM, request shrinkage (possibly by a queue), allocate/free chunks, ensure no outstanding GPU use survives a physical free.
- **Controller:** maintain prepared chunk values, consume capacity requests at safe points, select victim chunks, atomically make their expert locations CPU-resident and their slots unavailable, then request release.
- **Statistician:** maintain expert usage knowledge used in valuations; evictions do not erase history.
- **llama.cpp integration:** provide the correct safe publication point and ensure execution respects the resulting locations.

The queue and safe-point mechanism are not specified as additional classes or threads here. Their precise integration with llama.cpp is to be settled when implementing the existing class interfaces.
