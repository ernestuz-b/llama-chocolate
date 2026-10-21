#include "chocolate-cache/chocolate-shadow.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>

void operator delete(void * ptr, size_t size) noexcept;
void operator delete[](void * ptr, size_t size) noexcept;

static size_t allocation_calls = 0;

void * operator new(size_t size) {
    ++allocation_calls;
    void * ptr = std::malloc(size ? size : 1);
    if (!ptr) { throw std::bad_alloc(); }
    return ptr;
}

void * operator new[](size_t size) { return ::operator new(size); }

// Keep GCC from reporting malloc-backed replacement new/delete as mismatched.
#if defined(__GNUC__)
__attribute__((noinline))
#endif
static void free_allocation(void * ptr) noexcept { std::free(ptr); }

void operator delete(void * ptr) noexcept { free_allocation(ptr); }
void operator delete[](void * ptr) noexcept { free_allocation(ptr); }
void operator delete(void * ptr, size_t) noexcept { free_allocation(ptr); }
void operator delete[](void * ptr, size_t) noexcept { free_allocation(ptr); }

using namespace chocolate;

static int checks = 0;
static int failures = 0;

static void check(bool ok, const char * label) {
    ++checks;
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", label);
    }
}

static void observe(shadow_cache & cache, int32_t layer, std::initializer_list<int32_t> ids) {
    check(cache.observe({layer, ids.begin(), ids.size(), ids.size(), 1, 1, ids.size()}), "accepted router request");
}

static void lru_order(const shadow_cache & cache, std::initializer_list<uint32_t> expected) {
    uint32_t id = cache.lru();
    uint32_t prev = no_expert;
    for (uint32_t wanted : expected) {
        check(id == wanted, "LRU identity order");
        if (id == no_expert) { return; }
        check(cache.at(id).prev == prev && cache.at(id).tracked && cache.at(id).actual_slot < 0, "LRU backward link and class");
        prev = id;
        id = cache.at(id).next;
    }
    check(id == no_expert && prev == cache.mru(), "LRU tail and length");
    check(cache.nonresident_count() == expected.size(), "nonresident population");
}

static void test_tracking() {
    shadow_cache cache({{5, 0}}, {{0, 1024}}, {4, 1, 3}, 2);
    check(!cache.find(0, 0) && cache.tracked_count() == 0, "untracked initially");
    observe(cache, 0, {0});
    const auto first = *cache.find(0, 0);
    check(first.last_use == 1 && first.score == 2 && first.window_uses == 1 && first.selections == 1, "first use admits fresh history");
    lru_order(cache, {0});
    observe(cache, 0, {1});
    observe(cache, 0, {0, 0});
    lru_order(cache, {1, 0});
    check(cache.find(0, 0)->last_use == 3 && cache.find(0, 0)->window_uses == 2 && cache.find(0, 0)->selections == 3, "duplicates count selections but one window use");
    observe(cache, 0, {2});
    check(!cache.find(0, 1) && cache.totals().tracking_evictions == 1, "tracking LRU eviction");
    lru_order(cache, {0, 2});
    cache.set_tracking_capacity(5);
    observe(cache, 0, {3});
    observe(cache, 0, {1});
    check(cache.identity(0, 0) == 0 && cache.identity(0, 3) == 3, "growth preserves logical identity");
    check(cache.find(0, 1)->selections == 1 && cache.find(0, 1)->generation == 2 && cache.find(0, 1)->window_uses == 1, "readmission resets evicted history");
    lru_order(cache, {0, 2, 3, 1});
    cache.set_tracking_capacity(1);
    lru_order(cache, {1});
    check(cache.identity(0, 3) == 3 && !cache.find(0, 3), "contraction preserves logical identity");
    observe(cache, 0, {1});
    observe(cache, 0, {1});
    check(cache.find(0, 1)->window_uses == 3, "expired old generation does not decrement readmitted history");
    cache.set_tracking_capacity(0);
    lru_order(cache, {});
    check(cache.tracked_count() == 0, "zero tracking capacity");

    shadow_cache generations({{3, 0}}, {{0, 1}}, {4, 1, 3}, 1);
    observe(generations, 0, {0});
    observe(generations, 0, {1});
    observe(generations, 0, {0});
    observe(generations, 0, {0});
    observe(generations, 0, {0});
    check(generations.find(0, 0)->generation == 2 && generations.find(0, 0)->window_uses == 3, "old bucket expires after readmission without corrupting the new generation");
}

static void test_actual_lifecycle() {
    shadow_cache cache({{4, 0}, {3, 0}}, {{2, 100}}, {4, 1, 3});
    observe(cache, 0, {0, 1});
    check(cache.stage_actual(0, 0, 0), "stage first real fill");
    check(cache.find(0, 0)->actual_slot == -1 && cache.actual_count() == 0, "pending upload is not cached");
    check(!cache.stage_actual(0, 0, 1) && !cache.stage_actual(0, 1, 0), "duplicate pending expert or destination rejected");
    cache.complete_actual(0);
    check(cache.find(0, 0)->actual_slot == 0 && cache.actual_count() == 1, "completed upload is cached");
    lru_order(cache, {1});
    observe(cache, 0, {0});
    const auto history = *cache.find(0, 0);
    observe(cache, 1, {2});
    check(cache.stage_actual(1, 2, 0), "cross-layer actual replacement");
    check(cache.find(0, 0)->actual_slot == -1 && cache.actual_count() == 0, "old resident demoted before overwrite completes");
    const auto & demoted = *cache.find(0, 0);
    check(demoted.last_use == history.last_use && demoted.window_uses == history.window_uses && demoted.selections == history.selections, "demotion retains history");
    lru_order(cache, {1, 6, 0});
    cache.complete_actual();
    lru_order(cache, {1, 0});
    check(cache.find(1, 2)->actual_slot == 0 && cache.actual_count() == 1, "new actual slot identity");
    cache.set_tracking_capacity(0);
    check(cache.find(1, 2) && cache.tracked_count() == 1 && cache.nonresident_count() == 0, "tracking eviction excludes actual residents");
    check(!cache.stage_actual(1, 2, 2) && !cache.stage_actual(9, 0, 0) && !cache.stage_actual(0, 0, 0), "invalid actual fill fails closed");

    shadow_cache capped({{3, 0}}, {{1, 100}}, {}, 0);
    observe(capped, 0, {0});
    check(!capped.find(0, 0) && capped.simulated_count() == 1, "tracking eviction does not evict simulated residency");
    check(capped.stage_actual(0, 0, 0), "same-request router evidence permits a real fill after tracking eviction");
    capped.set_tracking_capacity(0);
    capped.complete_actual();
    check(capped.find(0, 0) && capped.find(0, 0)->last_use == 1 && capped.find(0, 0)->actual_slot == 0, "pending router evidence survives tracking eviction");
}

static void decision(resident_rank candidate, std::initializer_list<resident_rank> ranks, std::optional<size_t> expected) {
    const size_t allocations = allocation_calls;
    check(select_slot(candidate, ranks.begin(), ranks.size()) == expected, "Fakecache ranking decision");
    check(allocations == allocation_calls, "rank decision allocates nothing");
}

static void test_ranking() {
    decision({2, 1}, {{1, 100}}, 0);
    decision({1, 100}, {{2, 1}}, std::nullopt);
    decision({2, 11}, {{2, 10}}, 0);
    decision({2, 10}, {{2, 10}}, std::nullopt);
    decision({2, 9}, {{2, 10}}, std::nullopt);
    decision({0, 6}, {{4, 1}, {-1, 9}, {-2, 20}, {0, 2}}, 2);
    decision({0, 6}, {{0, 4}, {0, 2}, {0, 3}}, 1);
    decision({0, 6}, {{4, 1}, {0, 2}, {0, 2}, {4, 1}}, 1);
    decision({0, 6}, {{0, 2}, {0, 2}, {0, 2}}, 0);
    decision({INT32_MAX, UINT64_MAX}, {}, std::nullopt);
    decision({INT32_MAX, 0}, {{INT32_MIN, UINT64_MAX}}, 0);
    decision({INT32_MIN, UINT64_MAX}, {{INT32_MAX, 0}}, std::nullopt);
    decision({-16, UINT64_MAX}, {{-16, 0}}, 0);
    decision({2, 8}, {{5, 1}, {2, 20}, {1, 30}, {4, 2}, {2, 6}, {3, 1}, {1, 25}, {4, 1}, {1, 2}}, 8);
    check(!select_slot({1, 1}, nullptr, 9), "null ranks fail closed");
}

static void test_statistics() {
    shadow_cache cache({{8, 0}}, {{0, 10}}, {4, 1, 3});
    std::array<int32_t, 8> scores{};
    std::array<uint64_t, 8> last{};
    std::array<uint64_t, 8> selections{};
    std::array<std::array<bool, 8>, 4> window{};
    uint32_t random = 7;
    for (uint64_t request = 1; request <= 100; ++request) {
        std::array<int32_t, 4> ids{};
        std::array<bool, 8> selected{};
        for (auto & id : ids) {
            random = random * 1664525u + 1013904223u;
            id = int32_t((random >> 16) % 8);
            selected[id] = true;
            ++selections[id];
            last[id] = request;
        }
        window[(request - 1) % 4] = selected;
        check(cache.observe({0, ids.data(), ids.size(), 2, 2, 1, 2}), "two-token statistics observation");
        for (size_t id = 0; id < 8; ++id) {
            if (!last[id]) { check(!cache.find(0, uint32_t(id)), "never-used experts stay untracked"); continue; }
            scores[id] = std::clamp(scores[id] - 1 + (selected[id] ? 3 : 0), -4, 8);
            uint32_t uses = 0;
            for (const auto & bucket : window) { uses += bucket[id]; }
            const auto & actual = *cache.find(0, uint32_t(id));
            check(actual.score == scores[id] && actual.last_use == last[id] && actual.window_uses == uses && actual.selections == selections[id], "bounded score, ordinal, rolling window and selection oracle");
            check(cache.simulated_at(uint32_t(id)).score == scores[id] && cache.simulated_at(uint32_t(id)).last_use == last[id], "independent simulated ranking history matches the score oracle");
        }
    }
    check(cache.totals().selections == 400 && cache.totals().token_rows == 200 && cache.totals().requests == 100, "statistics totals");
    check(cache.totals().simulated_cpu_selections == 400 && cache.simulated_count() == 0 && cache.totals().simulated_promotions == 0, "zero GPU capacity is CPU-only simulation");
}

static bool same_history(const expert_state & a, const expert_state & b) {
    return a.last_use == b.last_use && a.score == b.score && a.window_uses == b.window_uses && a.selections == b.selections &&
        a.generation == b.generation && a.prev == b.prev && a.next == b.next && a.actual_slot == b.actual_slot &&
        a.pending_slot == b.pending_slot && a.pending_use == b.pending_use && a.tracked == b.tracked;
}

static void test_replay_and_allocations() {
    const std::vector<layer_config> layers{{11, 0}, {5, 0}, {7, 1}, {3, 1}};
    shadow_cache a(layers, {{4, 100}, {2, 200}}, {4, 1, 3});
    shadow_cache b(layers, {{4, 100}, {2, 200}}, {4, 1, 3});
    const size_t allocations = allocation_calls;
    uint32_t random = 42;
    for (uint32_t request = 0; request < 1000; ++request) {
        const size_t layer = request % layers.size();
        std::array<int32_t, 12> ids{};
        for (auto & id : ids) {
            random = random * 1664525u + 1013904223u;
            id = int32_t((random >> 16) % layers[layer].n_expert);
        }
        const router_view router{int32_t(layer), ids.data(), ids.size(), 3, 4, 1, 3};
        check(a.observe(router) && b.observe(router), "deterministic synthetic replay");
        size_t tracked = 0, actual = 0, simulated = 0;
        std::array<size_t, 2> populations{};
        for (size_t il = 0; il < layers.size(); ++il) {
            for (uint32_t id = 0; id < layers[il].n_expert; ++id) {
                const auto & e = a.at(a.identity(int32_t(il), id));
                const auto & s = a.simulated_at(a.identity(int32_t(il), id));
                const auto & other = b.simulated_at(b.identity(int32_t(il), id));
                check(same_history(e, b.at(b.identity(int32_t(il), id))), "replay histories and placement match");
                check(s.last_use == other.last_use && s.score == other.score && s.slot == other.slot, "replay simulated histories match");
                tracked += e.tracked;
                actual += e.actual_slot >= 0;
                simulated += s.slot >= 0;
                populations[layers[il].group] += s.slot >= 0;
            }
        }
        check(tracked == a.tracked_count() && actual == a.actual_count() && simulated == a.simulated_count(), "population consistency");
        check(populations[0] <= 4 && populations[1] <= 2, "each group respects vanilla capacity");
        check(a.totals().simulated_hits + a.totals().simulated_misses == a.totals().selections, "simulation hit/miss accounting");
        check(a.totals().actual_resident_selections + a.totals().actual_nonresident_selections == a.totals().selections, "actual residency accounting");
    }
    check(allocation_calls == allocations, "first-use, rolling history and simulation allocate nothing during replay");
    check(a.memory_bytes() == b.memory_bytes(), "replay memory footprint is deterministic");

    shadow_cache actual({{4, 0}}, {{2, 100}}, {4, 1, 3});
    shadow_cache independent({{4, 0}}, {{2, 100}}, {4, 1, 3});
    independent.set_tracking_capacity(0);
    const size_t calls_before = allocation_calls;
    for (uint32_t i = 0; i < 20; ++i) {
        observe(actual, 0, {int32_t(i % 4)});
        observe(independent, 0, {int32_t(i % 4)});
        check(actual.stage_actual(0, i % 4, 0), "actual fill stages independently");
        actual.complete_actual();
        for (uint32_t id = 0; id < 4; ++id) {
            check(actual.simulated_at(id).slot == independent.simulated_at(id).slot, "vanilla placements cannot determine simulated placements");
        }
    }
    check(allocation_calls == calls_before, "actual fill and demotion maintenance allocate nothing");
    check(actual.totals().simulated_hits == independent.totals().simulated_hits &&
          actual.totals().simulated_upload_bytes == independent.totals().simulated_upload_bytes, "actual residency does not change simulated counters");
}

static void test_rejections_and_strides() {
    shadow_cache cache({{4, 0}}, {{2, 100}});
    const std::array<int32_t, 8> data{{0, -1, 1, -1, 2, -1, 3, -1}};
    check(cache.observe({0, data.data(), data.size(), 2, 2, 2, 4}), "strided router view");
    check(cache.tracked_count() == 4 && cache.totals().selections == 4, "only logical strided entries observed");
    std::array<expert_state, 4> before{};
    for (size_t i = 0; i < 4; ++i) { before[i] = cache.at(uint32_t(i)); }
    const size_t calls = allocation_calls;
    check(!cache.observe({0, data.data(), data.size(), 2, 2, 1, 2}), "negative expert rejected");
    check(!cache.observe({0, data.data(), 3, 2, 2, 2, 4}), "short view rejected");
    check(!cache.observe({1, data.data(), data.size(), 1, 1, 1, 1}), "invalid layer rejected");
    check(!cache.observe({0, nullptr, 0, 1, 1, 1, 1}), "null router rejected");
    check(!cache.observe({0, data.data(), data.size(), 2, SIZE_MAX, 2, 4}), "row arithmetic overflow rejected");
    check(!cache.observe({0, data.data(), data.size(), 3, 1, SIZE_MAX, 1}), "lane arithmetic overflow rejected");
    check(!cache.observe({0, data.data(), data.size(), 1, 1, 1, 0}), "zero token stride rejected");
    for (size_t i = 0; i < 4; ++i) { check(same_history(before[i], cache.at(uint32_t(i))), "invalid observation leaves history and placement unchanged"); }
    check(cache.totals().requests == 1 && cache.totals().selections == 4 && cache.totals().rejected_requests == 7, "rejection accounting");
    check(calls == allocation_calls, "rejection allocates nothing");
    for (const statistics_params params : {statistics_params{0, 1, 2}, {1, 3, 2}, {65537, 1, 2}, {2, UINT32_MAX, UINT32_MAX}}) {
        bool caught = false;
        try { shadow_cache invalid({{1, 0}}, {{0, 1}}, params); } catch (const std::exception &) { caught = true; }
        check(caught, "invalid statistical parameters rejected before use");
    }
    bool caught = false;
    try { shadow_cache invalid({{1, 0}}, {{2, 1}}); } catch (const std::exception &) { caught = true; }
    check(caught, "capacity cannot exceed eligible logical domain");
}

static int replay(const char * path) {
    std::ifstream in(path);
    size_t n_layers, n_groups;
    if (!(in >> n_layers >> n_groups) || n_layers > 1000000 || n_groups > n_layers) { throw std::runtime_error("invalid replay header"); }
    std::vector<layer_config> layers(n_layers);
    std::vector<group_config> groups(n_groups);
    for (auto & l : layers) { if (!(in >> l.n_expert >> l.group)) { throw std::runtime_error("invalid replay layer"); } }
    for (auto & g : groups) { if (!(in >> g.capacity >> g.expert_bytes)) { throw std::runtime_error("invalid replay group"); } }
    shadow_cache cache(layers, groups);
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
        std::istringstream row(line);
        char op;
        if (!(row >> op) || op == '#') { continue; }
        if (op == 'R') {
            int32_t layer;
            size_t lanes, tokens;
            if (!(row >> layer >> lanes >> tokens) || !lanes || !tokens || lanes > SIZE_MAX / tokens || lanes * tokens > 10000000) {
                throw std::runtime_error("invalid replay request shape");
            }
            std::vector<int32_t> ids(lanes * tokens);
            for (auto & id : ids) { if (!(row >> id)) { throw std::runtime_error("missing replay expert"); } }
            if (!cache.observe({layer, ids.data(), ids.size(), lanes, tokens, 1, lanes})) { throw std::runtime_error("rejected replay request"); }
        } else if (op == 'S') {
            int32_t layer;
            uint32_t expert, slot;
            if (!(row >> layer >> expert >> slot) || !cache.stage_actual(layer, expert, slot)) { throw std::runtime_error("invalid replay fill"); }
        } else if (op == 'C') {
            size_t group;
            if (!(row >> group) || group >= groups.size()) { throw std::runtime_error("invalid replay completion"); }
            cache.complete_actual(group);
        } else if (op == 'T') {
            size_t capacity;
            if (!(row >> capacity)) { throw std::runtime_error("invalid replay tracking capacity"); }
            cache.set_tracking_capacity(capacity);
        } else {
            throw std::runtime_error("unknown replay operation");
        }
        std::string extra;
        if (row >> extra) { throw std::runtime_error("trailing replay input"); }
    }
    cache.dump(stdout, true);
    return 0;
}

static int benchmark() {
    std::vector<layer_config> layers(48, {128, 0});
    for (size_t tokens : {size_t(1), size_t(256)}) {
        shadow_cache cache(layers, {{256, 1024 * 1024}});
        std::vector<int32_t> ids(8 * tokens);
        for (size_t i = 0; i < ids.size(); ++i) { ids[i] = int32_t((i * 7) % 128); }
        const size_t calls = allocation_calls;
        const auto start = std::chrono::steady_clock::now();
        for (size_t iteration = 0; iteration < 100; ++iteration) {
            for (int32_t layer = 0; layer < 48; ++layer) {
                cache.complete_actual(0);
                if (!cache.observe({layer, ids.data(), ids.size(), 8, tokens, 1, 8})) { return 1; }
                if (tokens == 1) {
                    for (size_t lane = 0; lane < 8; ++lane) {
                        const uint32_t expert = uint32_t(ids[lane]);
                        if (cache.find(layer, expert)->actual_slot < 0 && !cache.stage_actual(layer, expert, uint32_t((size_t(layer) * 8 + lane) % 256))) { return 1; }
                    }
                }
            }
        }
        const double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count();
        std::printf("tokens=%zu requests=4800 ns_per_request=%.1f ns_per_selection=%.1f memory_bytes=%zu allocations=%zu record_bytes=%zu\n",
                    tokens, ns / 4800, ns / (4800 * ids.size()), cache.memory_bytes(), allocation_calls - calls, sizeof(expert_state));
        cache.dump(stdout);
        if (allocation_calls != calls) { return 1; }
    }
    return 0;
}

int main(int argc, char ** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--replay") { return replay(argv[2]); }
        if (argc == 2 && std::string(argv[1]) == "--benchmark") { return benchmark(); }
        if (argc != 1) { std::fprintf(stderr, "usage: %s [--replay FILE | --benchmark]\n", argv[0]); return 1; }
        test_tracking();
        test_actual_lifecycle();
        test_ranking();
        test_statistics();
        test_replay_and_allocations();
        test_rejections_and_strides();
        std::printf("%d checks, %d failures\n", checks, failures);
        return failures ? 1 : 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
