#include "chocolate-shadow.h"

#include <algorithm>
#include <cinttypes>
#include <stdexcept>

namespace chocolate {
namespace {

bool weaker(const resident_rank & a, const resident_rank & b) noexcept {
    return a.score < b.score || (a.score == b.score && a.last_use < b.last_use);
}

void add(uint64_t & value, uint64_t delta = 1) noexcept {
    value = delta > UINT64_MAX - value ? UINT64_MAX : value + delta;
}

} // namespace

// Adapted from Fakecache src/fakecache/resident-policy.cpp, with the same rank and tie rules.
std::optional<size_t> select_slot(const resident_rank & candidate, const resident_rank * slots, size_t n_slots) noexcept {
    if (slots == nullptr || n_slots == 0) {
        return std::nullopt;
    }
    size_t weakest = 0;
    for (size_t i = 1; i < n_slots; ++i) {
        if (weaker(slots[i], slots[weakest])) {
            weakest = i;
        }
    }
    if (weaker(slots[weakest], candidate)) {
        return weakest;
    }
    return std::nullopt;
}

shadow_cache::shadow_cache(const std::vector<layer_config> & layer_configs, const std::vector<group_config> & group_configs,
                           statistics_params config, size_t capacity) : params(config), tracking_capacity(capacity) {
    if (!params.window || params.window > 65536 || params.rise < params.decay) {
        throw std::invalid_argument("invalid shadow statistics parameters");
    }
    const uint64_t minimum = uint64_t(params.decay) * params.window;
    const uint64_t maximum = uint64_t(params.rise - params.decay) * params.window;
    if (minimum > uint64_t(INT32_MAX) + 1 || maximum > INT32_MAX || layer_configs.size() > INT32_MAX || group_configs.size() > INT32_MAX) {
        throw std::overflow_error("shadow domain or score bounds overflow");
    }
    score_min = int32_t(-int64_t(minimum));
    score_max = int32_t(maximum);
    uint64_t n_experts = 0;
    std::vector<uint64_t> group_sizes(group_configs.size(), 0);
    for (const auto & layer : layer_configs) {
        if ((layer.n_expert && (layer.group < 0 || size_t(layer.group) >= group_configs.size())) ||
            layer.n_expert > uint64_t(no_expert) - n_experts) {
            throw std::invalid_argument("invalid shadow layer domain");
        }
        layers.push_back({uint32_t(n_experts), layer});
        n_experts += layer.n_expert;
        if (layer.n_expert) {
            group_sizes[layer.group] += layer.n_expert;
        }
    }
    if (n_experts > SIZE_MAX / sizeof(window_entry) / params.window) {
        throw std::overflow_error("shadow window storage overflow");
    }
    for (size_t i = 0; i < group_configs.size(); ++i) {
        const auto & group = group_configs[i];
        if (group.capacity > INT32_MAX || group.capacity > group_sizes[i] ||
            (group.capacity && group.expert_bytes > UINT64_MAX / group.capacity)) {
            throw std::invalid_argument("invalid shadow group capacity");
        }
        groups.push_back({group, std::vector<uint32_t>(group.capacity, no_expert),
                         std::vector<uint32_t>(group.capacity, no_expert), std::vector<uint32_t>(group.capacity, no_expert),
                         std::vector<resident_rank>(group.capacity)});
    }
    experts.resize(size_t(n_experts));
    simulation.resize(size_t(n_experts));
    group_of.resize(size_t(n_experts));
    selected.resize(size_t(n_experts));
    used.reserve(size_t(n_experts));
    for (const auto & layer : layers) {
        for (uint32_t e = 0; e < layer.config.n_expert; ++e) {
            group_of[layer.begin + e] = uint32_t(layer.config.group);
        }
    }
    buckets.resize(params.window);
    for (auto & bucket : buckets) {
        bucket.reserve(size_t(n_experts));
    }
}

uint32_t shadow_cache::identity(int32_t layer, uint32_t expert) const noexcept {
    if (layer < 0 || size_t(layer) >= layers.size() || expert >= layers[layer].config.n_expert) {
        return no_expert;
    }
    return layers[layer].begin + expert;
}

const expert_state * shadow_cache::find(int32_t layer, uint32_t expert) const noexcept {
    const uint32_t id = identity(layer, expert);
    return id == no_expert || !experts[id].tracked ? nullptr : &experts[id];
}

void shadow_cache::unlink(uint32_t id) noexcept {
    auto & e = experts[id];
    if (e.prev != no_expert) { experts[e.prev].next = e.next; } else { head = e.next; }
    if (e.next != no_expert) { experts[e.next].prev = e.prev; } else { tail = e.prev; }
    e.prev = e.next = no_expert;
    --n_nonresident;
}

void shadow_cache::touch(uint32_t id) noexcept {
    auto & e = experts[id];
    if (e.actual_slot >= 0) { return; }
    if (head == id || e.prev != no_expert || e.next != no_expert) { unlink(id); }
    e.prev = tail;
    if (tail != no_expert) { experts[tail].next = id; } else { head = id; }
    tail = id;
    ++n_nonresident;
}

void shadow_cache::admit(uint32_t id) noexcept {
    auto & e = experts[id];
    if (!e.tracked) {
        const uint64_t generation = e.generation + 1;
        const int32_t pending_slot = e.pending_slot;
        const uint64_t pending_use = e.pending_use;
        e = {};
        e.generation = generation;
        e.pending_slot = pending_slot;
        e.pending_use = pending_use;
        e.tracked = true;
        ++n_tracked;
        add(counts.admissions);
    }
    touch(id);
}

void shadow_cache::evict_simulated(uint32_t id) noexcept {
    auto & e = simulation[id];
    if (e.slot >= 0) {
        groups[group_of[id]].simulated[e.slot] = no_expert;
        e.slot = -1;
        --n_simulated;
        add(counts.simulated_evictions);
    }
}

void shadow_cache::trim() noexcept {
    while (n_nonresident > tracking_capacity) {
        const uint32_t id = head;
        unlink(id);
        auto & e = experts[id];
        const uint64_t generation = e.generation;
        const int32_t pending_slot = e.pending_slot;
        const uint64_t pending_use = e.pending_use;
        e = {};
        e.generation = generation;
        e.pending_slot = pending_slot;
        e.pending_use = pending_use;
        --n_tracked;
        add(counts.tracking_evictions);
    }
}

void shadow_cache::set_tracking_capacity(size_t capacity) noexcept {
    tracking_capacity = capacity;
    trim();
}

bool shadow_cache::observe(const router_view & router) noexcept {
    auto reject = [&]() { add(counts.rejected_requests); return false; };
    if (router.layer < 0 || size_t(router.layer) >= layers.size() || !router.ids || !router.n_lanes || !router.n_tokens ||
        !router.lane_stride || !router.token_stride || counts.requests == UINT64_MAX) {
        return reject();
    }
    const auto & l = layers[router.layer];
    if (!l.config.n_expert || router.n_lanes > l.config.n_expert || router.size > SIZE_MAX / sizeof(int32_t) ||
        router.n_lanes - 1 > (SIZE_MAX - 1) / router.lane_stride) {
        return reject();
    }
    const size_t row_size = (router.n_lanes - 1) * router.lane_stride + 1;
    if (router.token_stride < row_size || router.n_tokens - 1 > (SIZE_MAX - row_size) / router.token_stride ||
        (router.n_tokens - 1) * router.token_stride + row_size > router.size) {
        return reject();
    }
    for (size_t t = 0; t < router.n_tokens; ++t) {
        for (size_t lane = 0; lane < router.n_lanes; ++lane) {
            const int32_t e = router.ids[t * router.token_stride + lane * router.lane_stride];
            if (e < 0 || uint32_t(e) >= l.config.n_expert || experts[l.begin + e].generation == UINT64_MAX) {
                return reject();
            }
        }
    }

    ++counts.requests;
    add(counts.token_rows, router.n_tokens);
    for (uint32_t id : used) { selected[id] = 0; }
    auto & bucket = buckets[next_bucket];
    for (const auto & entry : bucket) {
        auto & e = experts[entry.id];
        if (e.tracked && e.generation == entry.generation) { --e.window_uses; }
    }
    bucket.clear();
    used.clear();
    for (size_t t = 0; t < router.n_tokens; ++t) {
        for (size_t lane = 0; lane < router.n_lanes; ++lane) {
            const uint32_t id = l.begin + router.ids[t * router.token_stride + lane * router.lane_stride];
            auto & e = experts[id];
            admit(id);
            add(e.selections);
            add(counts.selections);
            const bool actual = e.actual_slot >= 0;
            const bool simulated = simulation[id].slot >= 0;
            add(actual ? counts.actual_resident_selections : counts.actual_nonresident_selections);
            add(simulated ? counts.simulated_hits : counts.simulated_misses);
            if (!simulated) { add(counts.simulated_cpu_selections); }
            if (actual != simulated) { add(counts.placement_disagreements); }
            if (!selected[id]) {
                selected[id] = 1;
                used.push_back(id);
                ++e.window_uses;
                e.last_use = counts.requests;
                simulation[id].last_use = counts.requests;
                bucket.push_back({id, e.generation});
                add(counts.distinct_selections);
            }
        }
    }
    // Fakecache's bounded eager decay/rise, once per distinct expert in an accepted request.
    for (size_t id = 0; id < experts.size(); ++id) {
        auto & e = experts[id];
        if (e.tracked) {
            const int64_t score = int64_t(e.score) - params.decay + (selected[id] ? int64_t(params.rise) : 0);
            e.score = int32_t(std::clamp(score, int64_t(score_min), int64_t(score_max)));
        }
        auto & s = simulation[id];
        if (s.last_use) {
            const int64_t score = int64_t(s.score) - params.decay + (selected[id] ? int64_t(params.rise) : 0);
            s.score = int32_t(std::clamp(score, int64_t(score_min), int64_t(score_max)));
        }
    }
    next_bucket = (next_bucket + 1) % buckets.size();

    auto & group = groups[l.config.group];
    for (size_t slot = 0; slot < group.simulated.size(); ++slot) {
        const uint32_t id = group.simulated[slot];
        group.ranks[slot] = id == no_expert ? resident_rank{score_min, 0} : resident_rank{simulation[id].score, simulation[id].last_use};
    }
    // Candidate order is logical identity, independent of vanilla's slot and victim choices.
    std::sort(used.begin(), used.end());
    for (uint32_t id : used) {
        auto & e = simulation[id];
        if (e.slot < 0) {
            const resident_rank rank{e.score, e.last_use};
            const auto slot = select_slot(rank, group.ranks.data(), group.ranks.size());
            if (slot) {
                const uint32_t old = group.simulated[*slot];
                if (old != no_expert) { evict_simulated(old); }
                group.simulated[*slot] = id;
                group.ranks[*slot] = rank;
                e.slot = int32_t(*slot);
                ++n_simulated;
                add(counts.simulated_promotions);
                add(counts.simulated_upload_bytes, group.config.expert_bytes);
            }
        }
    }
    trim();
    return true;
}

bool shadow_cache::stage_actual(int32_t layer, uint32_t expert, uint32_t slot) noexcept {
    const uint32_t id = identity(layer, expert);
    if (id == no_expert) { return false; }
    auto & group = groups[group_of[id]];
    if (slot >= group.actual.size() || experts[id].actual_slot >= 0 || experts[id].pending_slot >= 0 || group.pending[slot] != no_expert) { return false; }
    if (!experts[id].tracked) {
        if (!selected[id] || experts[id].generation == UINT64_MAX) { return false; }
        admit(id);
        experts[id].last_use = counts.requests;
    }
    const uint32_t old = group.actual[slot];
    if (old != no_expert) {
        experts[old].actual_slot = -1;
        group.actual[slot] = no_expert;
        --n_actual;
        touch(old);
    }
    group.pending[slot] = id;
    ++group.n_pending;
    experts[id].pending_slot = int32_t(slot);
    experts[id].pending_use = experts[id].last_use;
    return true;
}

void shadow_cache::complete_actual(size_t index) noexcept {
    if (index >= groups.size()) { return; }
    auto & group = groups[index];
    if (group.n_pending == 0) { return; }
    for (size_t slot = 0; slot < group.pending.size(); ++slot) {
        const uint32_t id = group.pending[slot];
        if (id == no_expert) { continue; }
        auto & e = experts[id];
        if (!e.tracked && e.generation < UINT64_MAX) {
            admit(id);
            e.last_use = e.pending_use;
        }
        if (e.tracked && e.actual_slot < 0) {
            unlink(id);
            e.actual_slot = int32_t(slot);
            group.actual[slot] = id;
            ++n_actual;
        }
        group.pending[slot] = no_expert;
        e.pending_slot = -1;
        e.pending_use = 0;
    }
    group.n_pending = 0;
    trim();
}

void shadow_cache::complete_actual() noexcept {
    for (size_t group = 0; group < groups.size(); ++group) { complete_actual(group); }
}

size_t shadow_cache::memory_bytes() const noexcept {
    size_t bytes = sizeof(*this) + layers.capacity() * sizeof(layer_state) + groups.capacity() * sizeof(group_state) +
        experts.capacity() * sizeof(expert_state) + simulation.capacity() * sizeof(simulated_expert) + group_of.capacity() * sizeof(uint32_t) + selected.capacity() +
        used.capacity() * sizeof(uint32_t) + buckets.capacity() * sizeof(std::vector<window_entry>);
    for (const auto & group : groups) {
        bytes += (group.actual.capacity() + group.pending.capacity() + group.simulated.capacity()) * sizeof(uint32_t) +
            group.ranks.capacity() * sizeof(resident_rank);
    }
    for (const auto & bucket : buckets) { bytes += bucket.capacity() * sizeof(window_entry); }
    return bytes;
}

void shadow_cache::dump(FILE * out, bool records) const {
    std::fprintf(out, "chocolate shadow: requests=%" PRIu64 " selections=%" PRIu64 " distinct=%" PRIu64 " rejected=%" PRIu64 " window=%u decay=%u rise=%u\n",
                 counts.requests, counts.selections, counts.distinct_selections, counts.rejected_requests, params.window, params.decay, params.rise);
    std::fprintf(out, "observed metadata: tracked=%zu nonresident=%zu actual_resident=%zu admissions=%" PRIu64 " tracking_evictions=%" PRIu64 " memory_bytes=%zu\n",
                 n_tracked, n_nonresident, n_actual, counts.admissions, counts.tracking_evictions, memory_bytes());
    std::fprintf(out, "observed residency selections: resident=%" PRIu64 " nonresident=%" PRIu64 "\n", counts.actual_resident_selections, counts.actual_nonresident_selections);
    std::fprintf(out, "simulated: hits=%" PRIu64 " misses=%" PRIu64 " cpu=%" PRIu64 " promotions=%" PRIu64 " evictions=%" PRIu64 " upload_bytes=%" PRIu64 " resident=%zu disagreements=%" PRIu64 "\n",
                 counts.simulated_hits, counts.simulated_misses, counts.simulated_cpu_selections, counts.simulated_promotions, counts.simulated_evictions,
                 counts.simulated_upload_bytes, n_simulated, counts.placement_disagreements);
    for (size_t i = 0; i < groups.size(); ++i) {
        std::fprintf(out, "group=%zu capacity=%u expert_bytes=%" PRIu64 "\n", i, groups[i].config.capacity, groups[i].config.expert_bytes);
    }
    if (records) {
        for (size_t il = 0; il < layers.size(); ++il) {
            const auto & layer = layers[il];
            std::fprintf(out, "layer=%zu experts=%u group=%d\n", il, layer.config.n_expert, layer.config.group);
            for (uint32_t expert = 0; expert < layer.config.n_expert; ++expert) {
                const uint32_t id = layer.begin + expert;
                const auto & e = experts[id];
                const auto & s = simulation[id];
                if (!e.tracked && !s.last_use) { continue; }
                std::fprintf(out, "id=%u layer=%zu expert=%u tracked=%d last_use=%" PRIu64 " score=%d window_uses=%u selections=%" PRIu64 " actual=%d pending=%d simulated=%d prev=%u next=%u\n",
                    id, il, expert, int(e.tracked), e.last_use, e.score, e.window_uses, e.selections, e.actual_slot, e.pending_slot, s.slot, e.prev, e.next);
                std::fprintf(out, "simulated_history id=%u last_use=%" PRIu64 " score=%d\n", id, s.last_use, s.score);
            }
        }
    }
}

} // namespace chocolate
