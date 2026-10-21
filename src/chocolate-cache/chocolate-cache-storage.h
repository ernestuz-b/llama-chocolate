#pragma once

#include <cstdint>
#include <vector>

// DESIGN DRAFT: physical GPU cache storage, separate from residency policy.
// Storage owns the device/group buffers, tensor views, independent allocations
// and the authoritative list of cache chunks. It never owns the expert
// statistics, decides eviction victims, or changes expert residency metadata.
//
// A group identifies a device + compatible expert layout. A chunk contains
// a contiguous range of COMPLETE expert slots in that group (including every
// expert weight tensor needed by the backend, fused or separate).
class chocolate_cache_storage {
public:
    using chunk_id = uint32_t; // Stable across growth and shrinkage.

    struct chunk {
        chunk_id id;
        uint32_t group;
        uint32_t first_slot;
        uint32_t n_slots;
        // Physical ggml allocation(s) and tensor views are owned by storage;
        // their exact representation is deliberately not fixed yet.
    };

    // Chunk lifecycle / memory-pressure contract:
    //
    // - Storage monitors free VRAM and its desired slack/hysteresis.
    // - If VRAM is low, storage asks chocolate_cache_control to free capacity.
    //   The request can be queued; it must not synchronously stall inference.
    // - The controller has independently prepared values for candidate
    //   chunks. At a safe point it selects a chunk, invalidates ALL resident
    //   experts in it (packed location = 0, meaning CPU RAM), and declares
    //   the entire chunk unavailable for future cache use.
    // - Only then does the controller request its physical release. Storage
    //   frees it when no previously submitted GPU compute or transfer still
    //   references its memory. A logically retired chunk may therefore
    //   remain physically allocated until those uses have drained.
    // - There is no relocation/compaction. Other chunks and their slot IDs
    //   remain unchanged. Experts naturally spill to CPU RAM; subsequent
    //   policy decisions may promote some into surviving or newly added
    //   chunks, without forcing an immediate migration.
    // - Growth simply allocates another chunk; a newly copied expert does
    //   not become GPU-resident until its upload is complete.
    //
    // Physical allocation, pressure reporting and release API signatures
    // are to be agreed here before implementation.

private:
    // Indexed chunk registry; do not erase/reorder entries in a way that
    // renumbers live chunks. Retired entries can remain as tombstones.
    // No duplicate controller-owned physical chunk list is needed.
    std::vector<chunk> chunks_;
};
