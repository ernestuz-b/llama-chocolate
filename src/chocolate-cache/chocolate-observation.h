#pragma once

#include "chocolate-shadow.h"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace chocolate {

// One context thread produces; one worker owns all metadata. The ring never waits for space.
class observation_queue {
public:
    observation_queue(const std::vector<layer_config> & layers, const std::vector<group_config> & groups,
                      size_t max_tokens, size_t max_lanes, bool threaded = true, statistics_params params = {});
    ~observation_queue();

    void begin() noexcept;
    void end(bool submitted, bool coverage_complete = true) noexcept;
    void router(const router_view & view) noexcept;
    void stage(int32_t layer, uint32_t expert, uint32_t slot) noexcept;
    void complete_group(size_t group) noexcept;
    // Called only after the context's existing full scheduler synchronization.
    void complete_all() noexcept;
    void stop();

    // Management/test owner only; do not call while the worker runs.
    bool process_available() noexcept;
    const shadow_cache & metadata() const { return cache; }
    uint64_t lost_events() const { return lost.load(std::memory_order_relaxed); }
    bool failed() const { return disabled.load(std::memory_order_acquire); }
    bool management_failed() const { return processing_failed.load(std::memory_order_acquire); }
    size_t memory_bytes() const noexcept;
    void dump(FILE * out, bool records = false) const;

private:
    enum class kind : uint64_t { BEGIN, END, ROUTER, STAGE, COMPLETE_GROUP, COMPLETE_ALL };
    struct packet {
        kind type;
        uint64_t serial;
        uint64_t layer;
        uint64_t a;
        uint64_t b;
        uint64_t size;
    };
    static constexpr size_t header_words = sizeof(packet) / sizeof(int32_t);

    bool reserve(size_t payload) noexcept;
    void publish(const packet & header) noexcept;
    void send(kind type, uint64_t layer = 0, uint64_t a = 0, uint64_t b = 0) noexcept;
    void write_words(size_t offset, const void * data, size_t words) noexcept;
    void read_words(size_t offset, void * data, size_t words) const noexcept;
    void run() noexcept;
    bool process_events();

    shadow_cache cache;
    std::vector<int32_t> ring;
    std::vector<int32_t> scratch;
    size_t producer = 0;
    size_t consumer = 0;
    uint64_t serial = 0;
    bool active = false;
    std::atomic<size_t> written{0};
    std::atomic<size_t> consumed{0};
    std::atomic<uint64_t> confirmed{0};
    std::atomic<uint64_t> lost{0};
    std::atomic<bool> disabled{false};
    std::atomic<bool> processing_failed{false};
    std::atomic<bool> stopping{false};
    std::mutex mutex;
    std::condition_variable wake;
    std::thread worker;
};

} // namespace chocolate
