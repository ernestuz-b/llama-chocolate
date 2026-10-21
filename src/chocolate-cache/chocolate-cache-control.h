#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

// Owns all Chocolate cache metadata for one cache instance, across all groups.
// No ggml tensors, GPU buffers, transfers, or statistical algorithms live here.
// Metadata updates must keep the record (AoS) and array (SoA) fields coherent.
class chocolate_cache_control {
public:
    using expert_id = uint32_t; // Dense index across all layers.
    using group_id  = uint32_t; // Storage group (device + compatible tensor layout).
    using slot_id   = uint32_t; // Slot index within a group.

    struct cache_slot {
        group_id group;
        slot_id  slot;
    };

    struct expert_view {
        int32_t  score;
        uint64_t last_use;
        uint32_t window_uses;
        std::optional<cache_slot> location; // Present only when cached.
    };

    // Every expert has a stable index and persistent accounting from initialization.
    // No cache storage is allocated here.
    explicit chocolate_cache_control(std::vector<uint32_t> experts_per_layer);

    chocolate_cache_control(const chocolate_cache_control &) = delete;
    chocolate_cache_control & operator=(const chocolate_cache_control &) = delete;

    // Convert model-local expert identity into one stable dense index.
    expert_id index(uint32_t layer, uint32_t expert) const;
    size_t expert_count() const noexcept;
    expert_view inspect(expert_id expert) const;

    // Advance once per accepted compute; 0 means no compute observed yet.
    // Only router-selected experts receive the new ordinal as last_use.
    // Observation is independent of GPU residency.
    void begin_compute() noexcept;
    void observe(expert_id expert) noexcept;
    uint64_t compute_ordinal() const noexcept;

    // Physical slot capacities, provided by storage. Resizing is an
    // off-critical-path operation; shrinking requires affected slots to be
    // vacated first.
    void resize_group(group_id group, size_t n_slots);
    std::optional<expert_id> occupant(cache_slot slot) const;

    // Residency changes are committed only when the copy is usable or the
    // old slot is no longer in use. No in-flight state is represented here.
    void publish_cached(expert_id expert, cache_slot slot);
    void uncache(expert_id expert);

    // The statistician computes these values; this class owns their storage.
    // Further statistical history structures will be designed with its header.
    void update_statistics(expert_id expert, int32_t score, uint32_t window_uses);

private:
    // AoS: residency metadata accessed together when changing placement.
    struct expert_record {
        std::optional<cache_slot> location; // Only valid after a completed transfer.
    };

    std::vector<uint32_t> layer_offsets_; // n_layers + 1.
    std::vector<expert_record> experts_;

    // SoA: independently scanned and updated statistical fields.
    // These are the only copies of these values.
    std::vector<int32_t>  scores_;
    std::vector<uint64_t> last_use_;
    std::vector<uint32_t> window_uses_;

    // Reverse residency map; both directions change in one operation.
    std::vector<std::vector<expert_id>> slot_owners_;

    // One monotonic ordinal for the cache instance, not wall-clock time.
    uint64_t compute_ordinal_ = 0;
};
