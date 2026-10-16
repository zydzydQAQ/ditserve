#include "ditserve/benchmark/workload.hpp"
#include "ditserve/benchmark/cost.hpp"
#include <set>
#include "ditserve/runtime/random.hpp"
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace ditserve {
void RequestSpec::validate() const {
    if (request_id.empty() || model.empty()) throw std::invalid_argument("request_id and model must be nonempty");
    number(arrival_ms, "arrival_ms"); number(deadline_ms, "deadline_ms"); number(cache_threshold, "cache_threshold");
    if (deadline_ms < arrival_ms) throw std::invalid_argument("deadline precedes arrival");
    if (height < 1 || width < 1 || frames < 1 || steps < 1) throw std::invalid_argument("dimensions and steps must be positive");
}

std::tuple<std::string, int, int, int> RequestSpec::shape_key() const { return {model, height, width, frames}; }

Json RequestSpec::to_json() const {
    return {{"request_id", request_id}, {"arrival_ms", arrival_ms}, {"deadline_ms", deadline_ms}, {"model", model},
            {"height", height}, {"width", width}, {"frames", frames}, {"steps", steps}, {"seed", seed},
            {"cache_threshold", cache_threshold}, {"cache_enabled", cache_enabled}};
}

RequestSpec RequestSpec::from_json(const Json& value) {
    check_keys(value, {"request_id", "arrival_ms", "deadline_ms", "model", "height", "width", "frames", "steps", "seed", "cache_threshold", "cache_enabled"});
    RequestSpec spec;
    spec.request_id = string_field(value, "request_id", "");
    spec.arrival_ms = number(value.at("arrival_ms"), "arrival_ms");
    spec.deadline_ms = number(value.at("deadline_ms"), "deadline_ms");
    spec.model = string_field(value, "model", spec.model);
    spec.height = int_field(value, "height", spec.height); spec.width = int_field(value, "width", spec.width);
    spec.frames = int_field(value, "frames", spec.frames); spec.steps = int_field(value, "steps", spec.steps);
    spec.seed = integer(value.value("seed", Json(0)), "seed");
    spec.cache_threshold = number_field(value, "cache_threshold", spec.cache_threshold);
    spec.cache_enabled = bool_field(value, "cache_enabled", spec.cache_enabled);
    spec.validate();
    return spec;
}

void validate_workload(const std::vector<RequestSpec>& specs) {
    std::set<std::string> ids;
    for (const auto& spec : specs) {
        spec.validate();
        if (!ids.insert(spec.request_id).second) throw std::invalid_argument("duplicate request_id: " + spec.request_id);
    }
}

std::vector<RequestSpec> read_workload(const Json& value) {
    if (!value.is_array()) throw std::invalid_argument("workload must be an array");
    std::vector<RequestSpec> specs;
    for (const auto& item : value) specs.push_back(RequestSpec::from_json(item));
    validate_workload(specs);
    return specs;
}

std::vector<RequestSpec> generate_workload(const Json& generator, const CostModel& cost) {
    check_keys(generator, {"count", "seed", "arrival_rate", "pattern", "slo_scale", "shapes", "step_choices", "model", "start_ms", "cache_threshold", "cache_enabled", "base_service_ms", "burst_period_ms", "burst_active_fraction"});
    const int count = int_field(generator, "count", 32, 0);
    const auto seed = integer(generator.value("seed", Json(0)), "seed");
    const double rate = number_field(generator, "arrival_rate", 2, true);
    const double scale = number_field(generator, "slo_scale", 2, true);
    const double start = number_field(generator, "start_ms", 0);
    const double threshold = number_field(generator, "cache_threshold", 0.15);
    const double period = number_field(generator, "burst_period_ms", 1000, true);
    const double fraction = number_field(generator, "burst_active_fraction", 0.2, true);
    const std::string pattern = string_field(generator, "pattern", "steady");
    const std::string model = string_field(generator, "model", "reference");
    const bool enabled = bool_field(generator, "cache_enabled", true);
    if (fraction > 1 || (pattern != "steady" && pattern != "bursty")) throw std::invalid_argument("invalid arrival pattern or active fraction");
    const auto shapes = generator.value("shapes", Json::array({Json::array({512, 512, 1})}));
    const auto steps = generator.value("step_choices", Json::array({20}));
    if (!shapes.is_array() || shapes.empty() || !steps.is_array() || steps.empty()) throw std::invalid_argument("shapes and step_choices require nonempty arrays");
    for (const auto& shape : shapes) {
        if (!shape.is_array() || shape.size() != 3) throw std::invalid_argument("shape requires height,width,frames");
        for (const auto& dimension : shape) int_field(Json{{"dimension", dimension}}, "dimension", 1);
    }
    for (const auto& step : steps) int_field(Json{{"steps", step}}, "steps", 1);
    const double base = number_field(generator, "base_service_ms", 100, true);
    Random arrivals(static_cast<std::uint64_t>(seed)), requests(static_cast<std::uint64_t>(seed) ^ 0x5DEECE66DULL);
    double exposure = 0;
    std::vector<RequestSpec> result;
    for (int index = 0; index < count; ++index) {
        exposure += arrivals.exponential(rate / (1000.0 * (pattern == "steady" ? 1 : fraction)));
        const double active = period * fraction;
        const double arrival = start + (pattern == "steady" ? exposure : std::floor(exposure / active) * period + std::fmod(exposure, active));
        const auto& shape = shapes.at(requests.bounded(shapes.size()));
        RequestSpec spec;
        std::ostringstream id; id << "req-" << std::setw(6) << std::setfill('0') << index;
        spec.request_id = id.str(); spec.arrival_ms = arrival; spec.deadline_ms = arrival; spec.model = model;
        spec.height = shape[0].get<int>(); spec.width = shape[1].get<int>(); spec.frames = shape[2].get<int>();
        spec.steps = steps.at(requests.bounded(steps.size())).get<int>();
        spec.seed = static_cast<std::int64_t>(requests.bounded(1ULL << 31));
        spec.cache_threshold = threshold; spec.cache_enabled = enabled;
        const double estimate = generator.contains("base_service_ms") ? base * cost.scale(spec) * (spec.steps / 20.0) : cost.full_service_ms(spec);
        number(estimate, "estimated_service_ms", true);
        spec.deadline_ms = arrival + scale * estimate;
        spec.validate(); result.push_back(spec);
    }
    return result;
}
}
