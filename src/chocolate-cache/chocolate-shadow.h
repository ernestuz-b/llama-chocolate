#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <optional>
#include <vector>

namespace chocolate {

constexpr uint32_t no_expert = UINT32_MAX;

// Adapted from Fakecache expert_state and statistics_params; no execution placement is owned here.
struct statistics_params {
    uint32_t window = 1;
    uint32_t decay = 1;
    uint32_t rise = 2;
};

struct expert_state {
    uint64_t last_use = 0;
    int32_t score = 0;
    uint32_t window_uses = 0;
    uint64_t selections = 0;
    uint64_t generation = 0;
    uint32_t prev = no_expert;
    uint32_t next = no_expert;
    int32_t actual_slot = -1;
    int32_t pending_slot = -1;
    uint64_t pending_use = 0;
    bool tracked = false;
    bool linked = false;
};

struct simulated_expert {
    uint64_t last_use = 0;
    int32_t score = 0;
    int32_t slot = -1;
};

struct layer_config {
    uint32_t n_expert = 0;
    int32_t group = -1;
};

struct group_config {
    uint32_t capacity = 0;
    uint64_t expert_bytes = 0;
};

struct router_view {
    int32_t layer = -1;
    const int32_t * ids = nullptr;
    size_t size = 0;
    size_t n_lanes = 0;
    size_t n_tokens = 0;
    size_t lane_stride = 1;
    size_t token_stride = 0;
};

struct resident_rank {
    int32_t score;
    uint64_t last_use;
};

// Fakecache resident_policy_score: strictly stronger rank, lowest slot on weakest ties.
std::optional<size_t> select_slot(const resident_rank & candidate, const resident_rank * slots, size_t n_slots) noexcept;

struct shadow_totals {
    uint64_t requests = 0;
    uint64_t token_rows = 0;
    uint64_t selections = 0;
    uint64_t distinct_selections = 0;
    uint64_t admissions = 0;
    uint64_t tracking_evictions = 0;
    uint64_t actual_resident_selections = 0;
    uint64_t actual_nonresident_selections = 0;
    uint64_t simulated_hits = 0;
    uint64_t simulated_misses = 0;
    uint64_t simulated_cpu_selections = 0;
    uint64_t simulated_promotions = 0;
    uint64_t simulated_evictions = 0;
    uint64_t simulated_upload_bytes = 0;
    uint64_t placement_disagreements = 0;
    uint64_t rejected_requests = 0;
    uint64_t rejected_staging = 0;
    uint64_t mirror_desynchronizations = 0;
    uint64_t unknown_residency_selections = 0;
    uint64_t samples = 0;
    uint64_t complete_samples = 0;
    uint64_t aborted_samples = 0;
    uint64_t incomplete_samples = 0;
    uint64_t missing_routers = 0;
    uint64_t scan_visits = 0;
    uint64_t statistics_ns = 0;
    uint64_t simulation_ns = 0;
};

// One management owner. Numeric identities survive vector relocation; no router pointers are retained.
class shadow_cache {
public:
    shadow_cache(const std::vector<layer_config> & layers, const std::vector<group_config> & groups,
                 statistics_params params = {}, size_t tracking_capacity = std::numeric_limits<size_t>::max());

    shadow_cache(const shadow_cache &) = delete;
    shadow_cache & operator=(const shadow_cache &) = delete;

    void begin_sample(bool expect_all_layers = false);
    bool finish_sample(bool compute_complete = true, bool coverage_complete = true);
    // Without an explicit begin, a view is a standalone replay sample.
    bool observe(const router_view & router);
    // Stage after vanilla queues a fill; confirm only at an existing backend completion boundary.
    bool stage_actual(int32_t layer, uint32_t expert, uint32_t slot) noexcept;
    void complete_actual(size_t group) noexcept;
    void complete_actual() noexcept;
    void set_tracking_capacity(size_t capacity) noexcept;
    void invalidate_actual(size_t group) noexcept;
    void invalidate_observation() noexcept;
    bool actual_valid(size_t group) const noexcept;
    bool statistics_valid() const noexcept { return history_valid; }
    size_t expert_count() const noexcept { return experts.size(); }
    void release_unused_storage();

    const expert_state * find(int32_t layer, uint32_t expert) const noexcept;
    const shadow_totals & totals() const noexcept { return counts; }
    size_t tracked_count() const noexcept { return n_tracked; }
    size_t nonresident_count() const noexcept { return n_nonresident; }
    size_t actual_count() const noexcept { return n_actual; }
    size_t simulated_count() const noexcept { return n_simulated; }
    uint32_t lru() const noexcept { return head; }
    uint32_t mru() const noexcept { return tail; }
    uint32_t identity(int32_t layer, uint32_t expert) const noexcept;
    const expert_state & at(uint32_t id) const { return experts.at(id); }
    const simulated_expert & simulated_at(uint32_t id) const { return simulation.at(id); }
    size_t memory_bytes() const noexcept;
    void dump(FILE * out, bool records = false) const;

private:
    struct layer_state {
        uint32_t begin;
        layer_config config;
    };
    struct group_state {
        group_config config;
        std::vector<uint32_t> actual;
        std::vector<uint32_t> pending;
        std::vector<uint32_t> simulated;
        std::vector<resident_rank> ranks;
        size_t n_pending = 0;
        bool valid = true;
    };
    struct window_entry {
        uint32_t id;
        uint64_t generation;
    };

    void unlink(uint32_t id) noexcept;
    void touch(uint32_t id) noexcept;
    void admit(uint32_t id) noexcept;
    void trim() noexcept;
    void evict_simulated(uint32_t id) noexcept;
    bool observe_router(const router_view & router);
    bool prepare_bucket(size_t size, std::vector<window_entry> & replacement);

    std::vector<layer_state> layers;
    std::vector<group_state> groups;
    std::vector<expert_state> experts;
    std::vector<simulated_expert> simulation;
    std::vector<uint32_t> group_of;
    std::vector<uint8_t> selected;
    std::vector<uint32_t> used;
    std::vector<uint8_t> observed_layers;
    std::vector<std::vector<window_entry>> buckets;
    statistics_params params;
    int32_t score_min;
    int32_t score_max;
    size_t next_bucket = 0;
    size_t tracking_capacity;
    size_t n_tracked = 0;
    size_t n_nonresident = 0;
    size_t n_actual = 0;
    size_t n_simulated = 0;
    uint32_t head = no_expert;
    uint32_t tail = no_expert;
    shadow_totals counts;
    bool sample_active = false;
    bool sample_valid = false;
    bool expect_all = false;
    bool history_valid = true;
};

} // namespace chocolate
