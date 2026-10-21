#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

// DESIGN DRAFT: metadata and policy owner for the whole Chocolate cache.
// This class owns one coherent expert history and residency mapping across
// layers, devices and storage groups. Storage owns GPU memory; the statistician
// updates history but does not own another copy of it. No transfers, ggml
// tensors or physical allocations belong here.
//
// We keep dense, stable expert identities even for experts never routed.
// Metadata persists when an expert is evicted. All experts are tracked; there
// is no "untracked" state, bounded tracking LRU or per-expert admission state.
// AoS is for fields touched together per expert, SoA for independently swept
// arrays. This class is responsible for their consistency.
class chocolate_cache_control {
public:
    using expert_id = uint32_t; // Dense across layers; stable for cache lifetime.
    using group_id  = uint32_t; // Storage group: device + expert tensor layout.
    using slot_id   = uint32_t; // Stable logical slot within a group.
    using location_id = uint32_t;

    struct cache_slot {
        group_id group;
        slot_id  slot;
    };

    struct expert_view {
        int32_t     score;
        uint64_t    last_use;
        uint32_t    window_uses;
        location_id location; // 0 = no usable GPU copy; otherwise packed group/slot.
    };

    // Stable layer/expert -> dense index. Initially every expert has empty
    // history and no GPU residency; CPU RAM remains canonical at all times.
    explicit chocolate_cache_control(std::vector<uint32_t> experts_per_layer);
    chocolate_cache_control(const chocolate_cache_control &) = delete;
    chocolate_cache_control & operator=(const chocolate_cache_control &) = delete;

    expert_id index(uint32_t layer, uint32_t expert) const;
    size_t expert_count() const noexcept;
    expert_view inspect(expert_id expert) const;

    // Logical 2D residency table: location[layer][expert], stored flat for
    // compactness. Zero means "use CPU RAM"; it does not imply an old GPU
    // allocation is already safe to free.
    // Nonzero packs group + physical slot into one uint32_t; no second
    // expert->slot-handle lookup is required. Exact bit widths are undecided.
    static location_id encode_location(cache_slot slot) noexcept;
    static cache_slot decode_location(location_id location); // requires != 0
    location_id location(uint32_t layer, uint32_t expert) const;

    // last_use is a monotonic accepted-compute ordinal, not a timestamp or
    // promotion time. 0 = never observed. Each accepted compute advances once;
    // only router-selected experts receive that ordinal, resident or not.
    // Aborted/incomplete computes must not silently advance the ordinal.
    // The exact transactional interface with the statistician is still open.
    void begin_compute() noexcept;
    void observe(expert_id expert) noexcept;
    uint64_t compute_ordinal() const noexcept;

    // Residency is independent of statistics. The controller owns both the
    // packed forward table and slot->expert reverse mapping; updates must be
    // coherent. Storage never chooses victims or writes expert locations.
    // New residency becomes visible only after transfer completion.
    void publish_cached(expert_id expert, cache_slot slot);
    void uncache(expert_id expert); // location -> 0; preserve all history.
    std::optional<expert_id> occupant(cache_slot slot) const;

    // Cache decisions are made here, not in storage or llama.cpp integration.
    // Off-path work prepares per-chunk values using expert statistics.
    // Storage detects low VRAM/slack and requests release (possibly queued).
    // At a safe point the controller consumes the request, selects chunks
    // using precomputed values, clears ALL occupants' locations to zero,
    // marks those chunks unavailable, then asks storage to release them.
    // Existing GPU users must have completed, or storage defers the free.
    // No relocation, compaction or synchronous cache repopulation: experts
    // naturally use CPU RAM and may later promote into surviving chunks.
    // Chunk-registration, valuation and request APIs remain to be designed
    // with the storage header rather than guessed here.

    // The statistician updates controller-owned accounting. Evictions do not
    // reset scores, recency, window counts or lifetime counters. The complete
    // statistics interface will be agreed when its header is drafted.
    void update_statistics(expert_id expert, int32_t score, uint32_t window_uses);

private:
    // AoS: fields normally touched together for a routed expert.
    struct expert_record {
        uint64_t last_use = 0;
    };
    std::vector<uint32_t> layer_offsets_; // n_layers + 1
    std::vector<expert_record> experts_;

    // SoA: independently read/swept by policy or statistician.
    std::vector<int32_t>  scores_;
    std::vector<uint32_t> window_uses_;

    // Exactly one canonical location per expert; logically [layer][expert].
    // Packed group/slot values, with 0 reserved for CPU RAM.
    std::vector<location_id> locations_;
    // Reverse mapping. Empty slots use an out-of-range expert_id sentinel.
    std::vector<std::vector<expert_id>> slot_owners_;

    // Precomputed chunk valuations belong to the controller; the physical
    // chunk registry, allocation/retirement state and VRAM pressure detection
    // belong to chocolate_cache_storage. Their concrete layout is still open.
    uint64_t compute_ordinal_ = 0;
};
