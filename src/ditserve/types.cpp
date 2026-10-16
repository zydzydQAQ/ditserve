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

}
