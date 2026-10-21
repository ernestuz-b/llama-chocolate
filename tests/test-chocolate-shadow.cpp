#include "chocolate-cache/chocolate-observation.h"

#include <atomic>

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

static std::atomic<size_t> allocation_calls{0};
static thread_local bool fail_allocations = false;

void * operator new(size_t size) {
    ++allocation_calls;
    if (fail_allocations) { throw std::bad_alloc(); }
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
    check(cache.find(0, 1)->selections == 2 && cache.find(0, 1)->generation == 2 && cache.find(0, 1)->window_uses == 1, "readmission resets history and retains factual lifetime selections");
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
    shadow_cache invalid({{3, 0}}, {{1, 1}});
    check(invalid.stage_actual(0, 0, 0), "residency facts are independent of accepted routers");
    check(!invalid.stage_actual(0, 1, 0), "duplicate pending destination rejected");
    check(invalid.totals().rejected_staging == 1 && !invalid.actual_valid(0), "staging rejection invalidates affected mirror");
    invalid.complete_actual();
    check(invalid.actual_count() == 0, "invalid mirror cannot publish residency");
    check(!invalid.stage_actual(1, 2, 2) && !invalid.stage_actual(9, 0, 0), "invalid domain fills fail closed with accounting");
    check(invalid.totals().rejected_staging == 3, "all staging failures counted");

    shadow_cache capped({{3, 0}}, {{1, 100}}, {}, 0);
    observe(capped, 0, {0});
    check(!capped.find(0, 0) && capped.simulated_count() == 1, "tracking eviction does not evict simulated residency");
    check(capped.stage_actual(0, 0, 0), "same-request router evidence permits a real fill after tracking eviction");
    capped.set_tracking_capacity(0);
    capped.complete_actual();
    check(capped.find(0, 0) && capped.find(0, 0)->last_use == 1 && capped.find(0, 0)->actual_slot == 0, "pending router evidence survives tracking eviction");

    shadow_cache pending({{3, 0}}, {{1, 100}}, {}, 0);
    const int32_t id = 0;
    pending.begin_sample(true);
    check(pending.observe({0, &id, 1, 1, 1, 1, 1}) && pending.stage_actual(0, 0, 0), "fill stages before off-path sample commit");
    check(pending.finish_sample() && !pending.find(0, 0), "tracking can evict a pending record after commit");
    pending.complete_actual();
    check(pending.find(0, 0)->last_use == 1 && pending.find(0, 0)->selections == 1 && pending.find(0, 0)->actual_slot == 0, "pending completion retains committed router evidence after tracking eviction");
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
            scores[id] = std::clamp(scores[id] - 1 + (selected[id] ? 3 : 0), -4, 8);
            if (!last[id]) { check(!cache.find(0, uint32_t(id)), "never-used experts stay untracked"); continue; }
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
        a.pending_slot == b.pending_slot && a.pending_use == b.pending_use && a.tracked == b.tracked && a.linked == b.linked;
}

static void test_replay_and_independence() {
    const std::vector<layer_config> layers{{11, 0}, {5, 0}, {7, 1}, {3, 1}};
    shadow_cache a(layers, {{4, 100}, {2, 200}}, {4, 1, 3});
    shadow_cache b(layers, {{4, 100}, {2, 200}}, {4, 1, 3});
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
    check(a.statistics_valid() && b.statistics_valid(), "management replay retains valid history with elastic storage");
    check(a.memory_bytes() == b.memory_bytes(), "replay memory footprint is deterministic");

    shadow_cache actual({{4, 0}}, {{2, 100}}, {4, 1, 3});
    shadow_cache independent({{4, 0}}, {{2, 100}}, {4, 1, 3});
    independent.set_tracking_capacity(0);
    for (uint32_t i = 0; i < 20; ++i) {
        observe(actual, 0, {int32_t(i % 4)});
        observe(independent, 0, {int32_t(i % 4)});
        check(actual.stage_actual(0, i % 4, 0), "actual fill stages independently");
        actual.complete_actual();
        for (uint32_t id = 0; id < 4; ++id) {
            check(actual.simulated_at(id).slot == independent.simulated_at(id).slot, "vanilla placements cannot determine simulated placements");
        }
    }
    check(actual.actual_valid(0), "actual fill and demotion remain consistent");
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

static void test_sample_clock() {
    shadow_cache cache({{4, 0}, {4, 1}}, {{0, 1}, {0, 1}}, {3, 1, 3});
    const int32_t first[] = {0, 1};
    const int32_t second[] = {2, 3};
    cache.begin_sample(true);
    check(cache.observe({0, first, 2, 2, 1, 1, 2}), "first layer captured in compute sample");
    check(cache.totals().samples == 0 && cache.at(0).score == 0, "router capture does not advance statistical clock");
    check(cache.observe({1, second, 2, 2, 1, 1, 2}), "second layer captured in same compute sample");
    check(cache.finish_sample(), "complete multi-layer sample accepted");
    check(cache.totals().samples == 1 && cache.at(0).score == 2 && cache.at(6).score == 2, "one decay/rise for all layers");
    check(cache.totals().scan_visits == 8, "one full-domain scan per sample");
    cache.begin_sample(true);
    check(cache.observe({0, first, 2, 2, 1, 1, 2}), "known prefix of aborted sample captured");
    check(cache.finish_sample(false), "aborted sample with complete prefix coverage accepted like Fakecache");
    check(cache.totals().aborted_samples == 1 && cache.at(6).score == 1, "aborted compute has its own ordinal");
    const auto history = cache.at(6);
    cache.begin_sample(true);
    check(cache.observe({0, first, 2, 2, 1, 1, 2}), "partial router coverage captured");
    check(!cache.finish_sample(), "missing layer prevents sample commit");
    check(cache.totals().samples == 2 && cache.totals().missing_routers == 1 && cache.totals().incomplete_samples == 1, "unknown absence has explicit coverage accounting");
    check(!cache.statistics_valid() && cache.at(6).score == history.score && cache.at(6).window_uses == history.window_uses, "incomplete sample freezes history rather than decaying omitted experts");

    shadow_cache rejected({{2, 0}}, {{1, 1}}, {2, 1, 2});
    observe(rejected, 0, {0});
    observe(rejected, 0, {1});
    check(!rejected.observe({}), "rejection at window boundary");
    check(rejected.totals().samples == 2 && rejected.find(0, 0)->window_uses == 1, "rejection does not expire an old bucket");
    rejected.begin_sample(true);
    check(rejected.observe({0, first, 2, 2, 1, 1, 2}), "factual observations remain usable after history invalidation");
    check(!rejected.finish_sample() && rejected.totals().samples == 2, "history is not silently resumed across coverage loss");
}

static void test_handoff() {
    const int32_t ids[] = {0, 1};
    observation_queue queue({{3, 0}, {3, 1}}, {{1, 1}, {1, 1}}, 8, 2, false);
    const size_t before = allocation_calls;
    queue.begin();
    queue.complete_group(0);
    queue.router({0, ids, 2, 2, 1, 1, 2});
    queue.stage(0, 0, 0);
    queue.router({1, ids, 2, 2, 1, 1, 2});
    queue.stage(1, 1, 0);
    queue.end(true);
    check(allocation_calls == before, "inference observation and staging capture allocate nothing");
    check(queue.metadata().totals().requests == 0, "inference producer never maintains metadata");
    queue.process_available();
    check(queue.metadata().totals().requests == 2 && queue.metadata().totals().samples == 0, "submission alone cannot confirm complete compute");
    queue.complete_all();
    queue.process_available();
    check(queue.metadata().totals().samples == 1 && queue.metadata().actual_count() == 2, "explicit sample closes after existing full synchronization");

    queue.begin();
    queue.stage(0, 1, 0);
    queue.stage(1, 2, 0);
    queue.complete_group(0);
    queue.router({});
    queue.process_available();
    check(queue.metadata().find(0, 1)->actual_slot == 0, "completion survives rejected router observation");
    check(queue.metadata().find(1, 2)->pending_slot == 0 && queue.metadata().find(1, 2)->actual_slot == -1, "one group completion cannot publish another group's transfer");
    queue.stage(0, 2, 0);
    queue.complete_group(0);
    queue.end(true);
    queue.complete_all();
    queue.process_available();
    check(queue.metadata().totals().rejected_staging == 0 && queue.metadata().find(0, 2)->actual_slot == 0, "rejected observations leave no stale pending slot on subsequent replacement");
    check(queue.metadata().actual_valid(0) && !queue.metadata().statistics_valid(), "router failure invalidates usage history independently of residency");
    queue.begin();
    queue.stage(1, 0, 0);
    queue.stage(1, 1, 0);
    queue.end(false);
    queue.complete_all();
    queue.process_available();
    check(!queue.metadata().actual_valid(1) && queue.metadata().actual_valid(0), "staging failure invalidates only affected group");
    check(queue.metadata().totals().rejected_staging == 1 && queue.metadata().totals().mirror_desynchronizations == 1, "staging failure diagnostics are aggregated");

    observation_queue unconfirmed({{2, 0}}, {{1, 1}}, 1, 1, false);
    unconfirmed.begin();
    unconfirmed.router({0, ids, 1, 1, 1, 1, 1});
    unconfirmed.end(true);
    unconfirmed.stop();
    unconfirmed.process_available();
    check(unconfirmed.metadata().totals().samples == 0 && !unconfirmed.metadata().statistics_valid(), "shutdown without synchronization cannot invent a completed sample");

    observation_queue interrupted({{2, 0}}, {{1, 1}}, 1, 1, false);
    interrupted.begin();
    interrupted.router({0, ids, 1, 1, 1, 1, 1});
    interrupted.complete_all();
    interrupted.stop();
    interrupted.process_available();
    check(interrupted.metadata().totals().incomplete_samples == 1 && !interrupted.metadata().statistics_valid(), "missing end marker at shutdown cannot claim complete sample coverage");

    observation_queue overflow({{2, 0}}, {{1, 1}}, 1, 1, false);
    const size_t capture_calls = allocation_calls;
    for (size_t i = 0; i < 100; ++i) { overflow.begin(); }
    check(allocation_calls == capture_calls && overflow.failed() && overflow.lost_events() > 0, "bounded producer reports overflow without allocating or waiting");
    overflow.process_available();
    check(!overflow.metadata().actual_valid(0) && !overflow.metadata().statistics_valid(), "lost notifications invalidate mirror and history");

    observation_queue allocation({{2, 0}}, {{1, 1}}, 1, 1, false);
    allocation.begin();
    allocation.router({0, ids, 1, 1, 1, 1, 1});
    allocation.end(true);
    allocation.complete_all();
    fail_allocations = true;
    allocation.process_available();
    fail_allocations = false;
    check(allocation.failed() && allocation.management_failed(), "management allocation failure disables observer");
    check(!allocation.metadata().statistics_valid() && !allocation.metadata().actual_valid(0), "allocation failure cannot expose valid history or residency");

    observation_queue callbacks({{2, 0}}, {{1, 1}}, 1, 1, false);
    callbacks.begin();
    callbacks.router({0, ids, 1, 1, 1, 1, 1});
    callbacks.end(true, false);
    callbacks.complete_all();
    callbacks.process_available();
    check(!callbacks.metadata().statistics_valid() && callbacks.metadata().totals().incomplete_samples == 1, "eval callback coverage is not claimed complete from scheduler success");
}

static void test_memory_scaling() {
    const std::vector<layer_config> layers(48, {128, 0});
    shadow_cache short_window(layers, {{0, 1}}, {1, 1, 2});
    shadow_cache long_window(layers, {{0, 1}}, {4096, 1, 2});
    check(long_window.memory_bytes() - short_window.memory_bytes() < 4096 * 32, "empty window scales with bucket headers rather than domain times window");
    const size_t empty = long_window.memory_bytes();
    observe(long_window, 0, {0});
    check(long_window.memory_bytes() - empty < 1024, "sparse sample allocates only observed window membership");
    shadow_cache elastic({{128, 0}}, {{0, 1}}, {1, 1, 2});
    std::array<int32_t, 128> dense{};
    for (size_t i = 0; i < dense.size(); ++i) { dense[i] = int32_t(i); }
    check(elastic.observe({0, dense.data(), dense.size(), dense.size(), 1, 1, dense.size()}), "dense management bucket grows on demand");
    const size_t dense_bytes = elastic.memory_bytes();
    observe(elastic, 0, {0});
    check(elastic.memory_bytes() < dense_bytes && elastic.find(0, 127)->window_uses == 0, "expired dense bucket contracts without stale membership");
    elastic.release_unused_storage();
    check(elastic.observe({0, dense.data(), dense.size(), dense.size(), 1, 1, dense.size()}), "scratch and bucket regrow after contraction");
    check(elastic.identity(0, 127) == 127 && elastic.find(0, 127)->selections == 2, "resizing retains numeric identities and lifetime counters");
    long_window.release_unused_storage();
    check(long_window.identity(47, 127) == 6143 && long_window.find(0, 0)->window_uses == 1, "storage contraction preserves numeric identity and history");
}

static void test_lru_randomized() {
    shadow_cache cache({{16, 0}, {16, 0}}, {{4, 1}}, {7, 1, 3}, 4);
    std::array<uint64_t, 32> lifetime{};
    uint32_t random = 9182;
    for (size_t step = 0; step < 4000; ++step) {
        random = random * 1664525u + 1013904223u;
        const uint32_t id = (random >> 16) % 32;
        const int32_t value = int32_t(id % 16);
        observe(cache, int32_t(id / 16), {value});
        ++lifetime[id];
        if (step % 3 == 0 && cache.at(id).actual_slot < 0) {
            check(cache.stage_actual(int32_t(id / 16), id % 16, uint32_t(step % 4)), "randomized replacement stages");
            cache.complete_actual();
        }
        cache.set_tracking_capacity((random >> 8) % 12);
        uint32_t current = cache.lru(), previous = no_expert;
        size_t linked = 0;
        while (current != no_expert && linked <= 32) {
            const auto & e = cache.at(current);
            check(e.linked && e.tracked && e.actual_slot < 0 && e.prev == previous, "randomized LRU membership and backward links");
            previous = current;
            current = e.next;
            ++linked;
        }
        check(current == no_expert && previous == cache.mru() && linked == cache.nonresident_count(), "randomized LRU has no cycle, underflow or invalid tail");
        size_t tracked = 0, actual = 0;
        for (uint32_t e = 0; e < 32; ++e) {
            const auto & record = cache.at(e);
            tracked += record.tracked;
            actual += record.actual_slot >= 0;
            check(record.selections == lifetime[e], "factual lifetime selections survive eviction and readmission");
            check(!record.linked || (record.tracked && record.actual_slot < 0), "only tracked nonresidents can be linked");
        }
        check(tracked == cache.tracked_count() && actual == cache.actual_count(), "randomized actual and tracked populations");
    }
}

static void test_multilayer_reference() {
    // Scalar oracle for Fakecache statistics::consume: one ordinal, one bucket and one eager domain update.
    shadow_cache cache({{8, 0}, {8, 0}, {8, 0}}, {{3, 1}}, {7, 1, 3});
    std::array<int32_t, 24> scores{};
    std::array<uint64_t, 24> last{}, selections{};
    std::array<std::array<bool, 24>, 7> window{};
    uint32_t random = 31337;
    for (uint64_t ordinal = 1; ordinal <= 500; ++ordinal) {
        std::array<bool, 24> selected{};
        const bool complete = ordinal % 7 != 0;
        const int32_t n_layers = complete ? 3 : int32_t(ordinal % 3 + 1);
        cache.begin_sample(true);
        for (int32_t layer = 0; layer < n_layers; ++layer) {
            std::array<int32_t, 4> ids{};
            for (size_t row = 0; row < 2; ++row) {
                random = random * 1664525u + 1013904223u;
                ids[2 * row] = int32_t((random >> 16) % 8);
                ids[2 * row + 1] = (ids[2 * row] + 1) % 8;
            }
            check(cache.observe({layer, ids.data(), ids.size(), 2, 2, 1, 2}), "multi-layer reference router accepted");
            for (int32_t id : ids) { selected[size_t(layer) * 8 + id] = true; ++selections[size_t(layer) * 8 + id]; }
        }
        check(cache.finish_sample(complete), "multi-layer reference sample accepted");
        window[(ordinal - 1) % 7] = selected;
        for (uint32_t id = 0; id < 24; ++id) {
            scores[id] = std::clamp(scores[id] - 1 + (selected[id] ? 3 : 0), -7, 14);
            if (selected[id]) { last[id] = ordinal; }
            uint32_t uses = 0;
            for (const auto & bucket : window) { uses += bucket[id]; }
            const auto & sim = cache.simulated_at(id);
            check(sim.score == scores[id] && sim.last_use == last[id], "simulation matches complete-domain Fakecache score and recency");
            const auto & e = cache.at(id);
            if (last[id]) { check(e.score == scores[id] && e.last_use == last[id] && e.window_uses == uses && e.selections == selections[id], "multi-layer score, recency, exact window and factual selections match Fakecache oracle"); }
        }
    }
    check(cache.totals().scan_visits == 24 * 500 && cache.totals().samples == 500, "multi-layer scan frequency independent of router count");
}

static void test_threaded_handoff() {
    observation_queue queue({{4, 0}, {4, 0}}, {{2, 1}}, 256, 2);
    const int32_t ids[] = {0, 1};
    fail_allocations = true;
    for (size_t i = 0; i < 200; ++i) {
        queue.begin();
        queue.router({0, ids, 2, 2, 1, 1, 2});
        queue.router({1, ids, 2, 2, 1, 1, 2});
        queue.end(true);
        queue.complete_all();
    }
    fail_allocations = false;
    queue.stop();
    check(!queue.failed() && queue.lost_events() == 0, "threaded handoff retains all events without producer allocation");
    check(queue.metadata().totals().samples == 200 && queue.metadata().totals().selections == 800, "worker drains confirmed samples before destruction");
    check(queue.metadata().statistics_valid() && queue.metadata().actual_valid(0), "threaded metadata remains valid");
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
    for (const uint32_t window : {1u, 256u}) {
        for (size_t tokens : {size_t(1), size_t(256)}) {
            observation_queue queue(layers, {{256, 1024 * 1024}}, tokens, 8, false, {window, 1, 2});
            std::vector<int32_t> ids(8 * tokens);
            for (size_t i = 0; i < ids.size(); ++i) { ids[i] = int32_t((i * 7) % 128); }
            double capture_ns = 0, management_ns = 0;
            size_t capture_allocations = 0;
            for (size_t iteration = 0; iteration < 100; ++iteration) {
                const size_t calls = allocation_calls;
                const auto start = std::chrono::steady_clock::now();
                queue.begin();
                for (int32_t layer = 0; layer < 48; ++layer) {
                    queue.complete_group(0);
                    queue.router({layer, ids.data(), ids.size(), 8, tokens, 1, 8});
                    if (tokens == 1) {
                        for (uint32_t lane = 0; lane < 8; ++lane) {
                            queue.stage(layer, uint32_t(ids[lane]), (uint32_t(layer) * 8 + lane + uint32_t(iteration) * 128) % 256);
                        }
                    }
                }
                queue.end(true);
                queue.complete_all();
                const auto captured = std::chrono::steady_clock::now();
                capture_ns += std::chrono::duration<double, std::nano>(captured - start).count();
                capture_allocations += allocation_calls.load() - calls;
                queue.process_available();
                management_ns += std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - captured).count();
            }
            const auto & cache = queue.metadata();
            const auto & totals = cache.totals();
            std::printf("tokens=%zu window=%u samples=%llu capture_ns_per_layer=%.1f capture_ns_per_selection=%.1f capture_allocations=%zu management_ns_per_sample=%.1f statistics_ns_per_sample=%.1f simulation_ns_per_sample=%.1f metadata_bytes=%zu transport_inclusive_bytes=%zu scan_visits=%llu\n",
                        tokens, window, (unsigned long long) totals.samples, capture_ns / 4800, capture_ns / (4800 * ids.size()), capture_allocations,
                        management_ns / 100, totals.statistics_ns / 100.0, totals.simulation_ns / 100.0, cache.memory_bytes(), queue.memory_bytes(), (unsigned long long) totals.scan_visits);
            if (capture_allocations || queue.failed() || totals.samples != 100 || totals.scan_visits != 614400 ||
                totals.rejected_staging || (tokens == 1 && cache.actual_count() != 256)) { return 1; }
        }
    }
    for (size_t n_layers : {size_t(8), size_t(48)}) {
        const std::vector<layer_config> domain(n_layers, {128, 0});
        for (uint32_t window : {1u, 256u, 4096u}) {
            observation_queue first(domain, {{256, 1}}, 256, 8, false, {window, 1, 2});
            observation_queue second(domain, {{256, 1}}, 256, 8, false, {window, 1, 2});
            std::printf("empty_experts=%zu window=%u metadata_bytes=%zu one_context_bytes=%zu two_context_bytes=%zu\n",
                        128 * n_layers, window, first.metadata().memory_bytes(), first.memory_bytes(), first.memory_bytes() + second.memory_bytes());
        }
    }
    return 0;
}

int main(int argc, char ** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--replay") { return replay(argv[2]); }
        if (argc == 2 && std::string(argv[1]) == "--benchmark") { return benchmark(); }
        if (argc != 1) { std::fprintf(stderr, "usage: %s [--replay FILE | --benchmark]\n", argv[0]); return 1; }
        test_multilayer_reference();
        test_threaded_handoff();
        test_sample_clock();
        test_handoff();
        test_memory_scaling();
        test_lru_randomized();
        test_tracking();
        test_actual_lifecycle();
        test_ranking();
        test_statistics();
        test_replay_and_independence();
        test_rejections_and_strides();
        std::printf("%d checks, %d failures\n", checks, failures);
        return failures ? 1 : 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
