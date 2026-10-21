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

    enum class expert_state : uint8_t {
        untracked,
        tracked, // Non-resident, with retained usage history.
        cached,  // Usable GPU copy, not merely a pending transfer.
    };

    struct cache_slot {
        group_id group;
        slot_id  slot;
    };

    struct expert_view {
        expert_state state;
        int32_t  score;
        uint64_t last_use;
        uint32_t window_uses;
        std::optional<cache_slot> location; // Present only when cached.
    };

    // A tracking limit bounds non-resident tracked experts; cached experts do
    // not occupy the tracking LRU. No cache storage is allocated here.
    chocolate_cache_control(std::vector<uint32_t> experts_per_layer, size_t tracking_limit);

    chocolate_cache_control(const chocolate_cache_control &) = delete;
    chocolate_cache_control & operator=(const chocolate_cache_control &) = delete;

    // Convert model-local expert identity into one stable dense index.
    expert_id index(uint32_t layer, uint32_t expert) const;
    size_t expert_count() const noexcept;
    expert_view inspect(expert_id expert) const;

    // Router-confirmed use: enter/touch the tracking LRU if non-resident.
    // When full, discard the oldest tracked (never a cached) expert's history.
    // Must be bounded, with no steady-state allocation.
    void observe(expert_id expert);

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
    void update_statistics(expert_id expert, int32_t score,
                           uint64_t last_use, uint32_t window_uses);

private:
    static constexpr expert_id no_expert = static_cast<expert_id>(-1);

    // AoS: small fields normally touched together on use/state transitions.
    struct expert_record {
        expert_state state = expert_state::untracked;
        expert_id prev = no_expert;
        expert_id next = no_expert;
        cache_slot location{}; // Meaningful only when state == cached.
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

    // Tracking LRU contains exactly the non-resident tracked experts.
    size_t tracking_limit_ = 0;
    size_t tracked_count_ = 0;
    expert_id tracked_head_ = no_expert;
    expert_id tracked_tail_ = no_expert;
};
