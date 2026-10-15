#include "ditserve/types.hpp"

#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>

namespace ditserve {
double number(const Json& value, const std::string& name, bool positive) {
    if (!value.is_number()) throw std::invalid_argument(name + " must be numeric");
    const double result = value.get<double>();
    if (!std::isfinite(result) || (positive ? result <= 0 : result < 0))
        throw std::invalid_argument(name + " must be finite and " + (positive ? "positive" : "nonnegative"));
    return result;
}

std::int64_t integer(const Json& value, const std::string& name) {
    if (!value.is_number_integer() || (value.is_number_unsigned() &&
        value.get<std::uint64_t>() > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())))
        throw std::invalid_argument(name + " must be a signed 64-bit integer");
    return value.get<std::int64_t>();
}

double number_field(const Json& object, const char* key, double fallback, bool positive) {
    return number(object.contains(key) ? object.at(key) : Json(fallback), key, positive);
}

int int_field(const Json& object, const char* key, int fallback, int minimum) {
    const auto result = integer(object.contains(key) ? object.at(key) : Json(fallback), key);
    if (result < minimum || result > std::numeric_limits<int>::max())
        throw std::invalid_argument(std::string(key) + " is outside its supported integer range");
    return static_cast<int>(result);
}

bool bool_field(const Json& object, const char* key, bool fallback) {
    if (!object.contains(key)) return fallback;
    if (!object.at(key).is_boolean()) throw std::invalid_argument(std::string(key) + " must be boolean");
    return object.at(key).get<bool>();
}

std::string string_field(const Json& object, const char* key, const std::string& fallback) {
    const auto result = object.contains(key) ? object.at(key).get<std::string>() : fallback;
    if (result.empty()) throw std::invalid_argument(std::string(key) + " must be nonempty");
    return result;
}

void check_keys(const Json& object, std::initializer_list<const char*> allowed) {
    if (!object.is_object()) throw std::invalid_argument("expected a JSON object");
    const std::set<std::string> names(allowed.begin(), allowed.end());
    for (auto it = object.begin(); it != object.end(); ++it)
        if (!names.count(it.key())) throw std::invalid_argument("unknown field: " + it.key());
}

Json core_version() {
    return {{"version", "ditserve-core.v2"}, {"rng_version", "mt19937_64-box-muller.v1"},
            {"fingerprint_version", "sha256-nlohmann-json.v2"}, {"checkpoint_version", 2}};
}

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
}
