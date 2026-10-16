#include "ditserve/benchmark/metrics.hpp"
#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

namespace ditserve {
Json summarize(const Json& records, double start, double end, double busy, int num_gpus) {
    number(start, "start_ms"); number(end, "end_ms"); number(busy, "gpu_busy_ms");
    if (!records.is_array() || end < start || num_gpus < 1) throw std::invalid_argument("invalid metrics records or observation window");
    const double duration = end - start, capacity = duration * num_gpus;
    if (busy > capacity + std::max(1e-9, capacity * 1e-9)) throw std::invalid_argument("GPU busy time exceeds capacity");
    int completed = 0, rejected = 0, timed_out = 0, on_time = 0, passed = 0, failed = 0, unknown = 0, slo = 0;
    double gpu_ms = 0;
    std::vector<double> latencies;
    std::set<std::string> seen;
    for (const auto& record : records) {
        const auto id = string_field(record, "request_id", "");
        if (!seen.insert(id).second) throw std::invalid_argument("duplicate metric request ID");
        const auto status = string_field(record, "status", "");
        if (status == "completed") ++completed;
        else if (status == "rejected") ++rejected;
        else if (status == "timed_out") ++timed_out;
        else throw std::invalid_argument("unknown terminal status: " + status);
        const double arrival = number(record.at("arrival_ms"), "arrival_ms");
        const double deadline = number(record.at("deadline_ms"), "deadline_ms");
        if (deadline < arrival || arrival < start || arrival > end) throw std::invalid_argument("metric arrival/deadline outside observation window");
        gpu_ms += number_field(record, "gpu_ms", 0);
        if (status != "completed") continue;
        const double completion = number(record.at("completion_ms"), "completion_ms");
        if (completion < arrival || completion > end) throw std::invalid_argument("metric completion outside observation window");
        const double latency = completion - arrival;
        if (record.contains("latency_ms") && !record.at("latency_ms").is_null()) {
            const double provided = number(record.at("latency_ms"), "latency_ms");
            if (std::abs(provided - latency) > std::max(1e-6, 1e-8 * std::max(std::abs(provided), std::abs(latency))))
                throw std::invalid_argument("inconsistent latency");
        }
        latencies.push_back(latency);
        const bool timely = completion <= deadline;
        on_time += timely;
        const auto quality = record.value("quality_ok", Json(nullptr));
        if (quality.is_boolean() && quality.get<bool>()) { ++passed; slo += timely; }
        else if (quality.is_boolean()) ++failed;
        else ++unknown;
    }
    std::sort(latencies.begin(), latencies.end());
    const auto percentile = [&](double fraction) -> Json {
        return latencies.empty() ? Json(nullptr) : Json(latencies.at(static_cast<std::size_t>(std::ceil(fraction * latencies.size())) - 1));
    };
    const auto total = records.size();
    return {{"provenance", "synthetic"}, {"quality_semantics", "reference_parity_only_not_perceptual_quality"},
            {"total_arrivals", total}, {"completed", completed}, {"rejected", rejected}, {"timed_out", timed_out},
            {"on_time_completed", on_time}, {"deadline_misses", total - on_time}, {"completed_deadline_misses", completed - on_time},
            {"quality_passed_completed", passed}, {"quality_failed_completed", failed}, {"quality_unknown_completed", unknown},
            {"slo_met", slo}, {"slo_denominator", total}, {"slo_attainment", total ? static_cast<double>(slo) / total : 0},
            {"goodput_per_sec", duration ? slo * 1000.0 / duration : 0}, {"p95_latency_ms", percentile(0.95)}, {"p99_latency_ms", percentile(0.99)},
            {"latency_sample_count", latencies.size()}, {"latency_population", "completed_requests_only"}, {"percentile_method", "nearest_rank"},
            {"gpu_seconds", gpu_ms / 1000.0}, {"gpu_seconds_per_completed_request", completed ? Json(gpu_ms / (1000.0 * completed)) : Json(nullptr)},
            {"gpu_busy_ms", busy}, {"num_gpus", num_gpus}, {"resource_utilization", capacity ? busy / capacity : 0},
            {"start_ms", start}, {"end_ms", end}, {"duration_ms", duration}};
}
}
