#include "chocolate-observation.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstring>
#include <stdexcept>

namespace chocolate {

observation_queue::observation_queue(const std::vector<layer_config> & layers, const std::vector<group_config> & groups,
                                     size_t max_tokens, size_t max_lanes, bool threaded, statistics_params params) :
    cache(layers, groups, params) {
    static_assert(sizeof(packet) == 6 * sizeof(uint64_t), "packet must have no padding");
    if (!max_lanes || !max_tokens || max_lanes > SIZE_MAX / max_tokens) { throw std::overflow_error("observation shape overflow"); }
    size_t words = 2 * header_words;
    for (const auto & layer : layers) {
        if (!layer.n_expert) { continue; }
        // Budget every token as a separate microbatch, including its native fill and completion notifications.
        const size_t fills = std::min(max_lanes, size_t(layer.n_expert));
        if (fills > SIZE_MAX / header_words - 2 || max_lanes > SIZE_MAX - (fills + 2) * header_words ||
            words > SIZE_MAX - max_lanes - (fills + 2) * header_words) {
            throw std::overflow_error("observation storage overflow");
        }
        words += max_lanes + (fills + 2) * header_words;
    }
    if (words > (SIZE_MAX / 2 / sizeof(int32_t) - header_words) / max_tokens) { throw std::overflow_error("observation storage overflow"); }
    // Two batches allow the producer and management owner to make progress independently.
    ring.resize(2 * (words * max_tokens + header_words));
    if (threaded) { worker = std::thread(&observation_queue::run, this); }
}

observation_queue::~observation_queue() { stop(); }

bool observation_queue::reserve(size_t payload) noexcept {
    if (disabled.load(std::memory_order_acquire)) { lost.fetch_add(1, std::memory_order_relaxed); return false; }
    if (payload > ring.size() - header_words || producer > SIZE_MAX - header_words - payload ||
        header_words + payload > ring.size() - (producer - consumed.load(std::memory_order_acquire))) {
        lost.fetch_add(1, std::memory_order_relaxed);
        disabled.store(true, std::memory_order_release);
        return false;
    }
    return true;
}

void observation_queue::write_words(size_t offset, const void * data, size_t words) noexcept {
    if (!words) { return; }
    const size_t start = offset % ring.size();
    const size_t first = std::min(words, ring.size() - start);
    std::memcpy(ring.data() + start, data, first * sizeof(int32_t));
    if (first < words) { std::memcpy(ring.data(), static_cast<const char *>(data) + first * sizeof(int32_t), (words - first) * sizeof(int32_t)); }
}

void observation_queue::read_words(size_t offset, void * data, size_t words) const noexcept {
    if (!words) { return; }
    const size_t start = offset % ring.size();
    const size_t first = std::min(words, ring.size() - start);
    std::memcpy(data, ring.data() + start, first * sizeof(int32_t));
    if (first < words) { std::memcpy(static_cast<char *>(data) + first * sizeof(int32_t), ring.data(), (words - first) * sizeof(int32_t)); }
}

void observation_queue::publish(const packet & header) noexcept {
    write_words(producer, &header, header_words);
    producer += header_words + size_t(header.size);
    written.store(producer, std::memory_order_release);
}

void observation_queue::send(kind type, uint64_t layer, uint64_t a, uint64_t b) noexcept {
    if (reserve(0)) { publish({type, serial, layer, a, b, 0}); }
}

void observation_queue::begin() noexcept {
    if (serial == UINT64_MAX) { disabled.store(true, std::memory_order_release); }
    else { ++serial; }
    send(kind::BEGIN);
}

void observation_queue::end(bool submitted, bool coverage_complete) noexcept {
    send(kind::END, 0, submitted, coverage_complete);
    wake.notify_one();
}

void observation_queue::router(const router_view & view) noexcept {
    const bool shape = view.ids && view.n_lanes && view.n_tokens && view.lane_stride && view.token_stride &&
        view.n_lanes - 1 <= (SIZE_MAX - 1) / view.lane_stride;
    const size_t row = shape ? (view.n_lanes - 1) * view.lane_stride + 1 : 0;
    if (!shape || view.token_stride < row || view.n_tokens - 1 > (SIZE_MAX - row) / view.token_stride ||
        (view.n_tokens - 1) * view.token_stride + row > view.size || view.n_lanes > SIZE_MAX / view.n_tokens) {
        send(kind::ROUTER, uint64_t(view.layer));
        return;
    }
    const size_t count = view.n_lanes * view.n_tokens;
    if (!reserve(count)) { return; }
    size_t offset = producer + header_words;
    if (view.lane_stride == 1 && view.token_stride == view.n_lanes) {
        write_words(offset, view.ids, count);
    } else {
        for (size_t token = 0; token < view.n_tokens; ++token) {
            for (size_t lane = 0; lane < view.n_lanes; ++lane) {
                ring[offset++ % ring.size()] = view.ids[token * view.token_stride + lane * view.lane_stride];
            }
        }
    }
    publish({kind::ROUTER, serial, uint64_t(view.layer), view.n_lanes, view.n_tokens, count});
}

void observation_queue::stage(int32_t layer, uint32_t expert, uint32_t slot) noexcept { send(kind::STAGE, uint64_t(layer), expert, slot); }
void observation_queue::complete_group(size_t group) noexcept { send(kind::COMPLETE_GROUP, group); }

void observation_queue::complete_all() noexcept {
    send(kind::COMPLETE_ALL);
    confirmed.store(serial, std::memory_order_release);
    wake.notify_one();
}

bool observation_queue::process_events() {
    if (disabled.load(std::memory_order_acquire)) {
        cache.invalidate_observation();
        consumer = written.load(std::memory_order_acquire);
        consumed.store(consumer, std::memory_order_release);
        return false;
    }
    bool progress = false;
    while (consumer < written.load(std::memory_order_acquire)) {
        packet header;
        read_words(consumer, &header, header_words);
        if (header.type == kind::END && header.serial > confirmed.load(std::memory_order_acquire)) {
            if (!stopping.load(std::memory_order_acquire)) { break; }
            cache.invalidate_observation();
        } else if (header.type == kind::BEGIN) {
            cache.begin_sample(true);
            active = true;
        } else if (header.type == kind::END) {
            if (!active) { cache.invalidate_observation(); }
            else { cache.finish_sample(header.a != 0, header.b != 0); }
            active = false;
        } else if (header.type == kind::ROUTER) {
            scratch.resize(size_t(header.size));
            read_words(consumer + header_words, scratch.data(), scratch.size());
            if (!active) { cache.invalidate_observation(); }
            else { cache.observe({int32_t(header.layer), scratch.data(), scratch.size(), size_t(header.a), size_t(header.b), 1, size_t(header.a)}); }
        } else if (header.type == kind::STAGE) {
            cache.stage_actual(int32_t(header.layer), uint32_t(header.a), uint32_t(header.b));
        } else if (header.type == kind::COMPLETE_GROUP) {
            cache.complete_actual(size_t(header.layer));
        } else if (header.type == kind::COMPLETE_ALL) {
            cache.complete_actual();
        }
        consumer += header_words + size_t(header.size);
        consumed.store(consumer, std::memory_order_release);
        progress = true;
    }
    return progress;
}

bool observation_queue::process_available() noexcept {
    try {
        const bool progress = process_events();
        if (stopping.load(std::memory_order_acquire) && consumer == written.load(std::memory_order_acquire) && active) {
            cache.invalidate_observation();
            active = false;
        }
        return progress;
    } catch (const std::exception &) {
        processing_failed.store(true, std::memory_order_release);
        disabled.store(true, std::memory_order_release);
        cache.invalidate_observation();
        consumer = written.load(std::memory_order_acquire);
        consumed.store(consumer, std::memory_order_release);
        return false;
    }
}

void observation_queue::run() noexcept {
    try {
        for (;;) {
            process_available();
            if (stopping.load(std::memory_order_acquire) && consumer == written.load(std::memory_order_acquire)) { break; }
            std::unique_lock<std::mutex> lock(mutex);
            // Timed recheck avoids a lost notification without taking a producer-side mutex.
            wake.wait_for(lock, std::chrono::milliseconds(1));
        }
    } catch (const std::exception &) {
        processing_failed.store(true, std::memory_order_release);
        disabled.store(true, std::memory_order_release);
        cache.invalidate_observation();
    }
}

void observation_queue::stop() {
    stopping.store(true, std::memory_order_release);
    wake.notify_one();
    if (worker.joinable()) { worker.join(); }
}

size_t observation_queue::memory_bytes() const noexcept {
    return sizeof(*this) - sizeof(cache) + cache.memory_bytes() + (ring.capacity() + scratch.capacity()) * sizeof(int32_t);
}

void observation_queue::dump(FILE * out, bool records) const {
    cache.dump(out, records);
    std::fprintf(out, "capture: lost_events=%" PRIu64 " disabled=%d management_failed=%d buffer_bytes=%zu total_host_bytes=%zu\n",
                 lost_events(), int(failed()), int(management_failed()), ring.capacity() * sizeof(int32_t), memory_bytes());
}

} // namespace chocolate
