#pragma once

#include <cstdint>
#include <initializer_list>
#include <string>
#include <tuple>
#include <vector>
#include <nlohmann/json.hpp>

namespace ditserve {
using Json = nlohmann::json;

double number(const Json& value, const std::string& name, bool positive = false);
std::int64_t integer(const Json& value, const std::string& name);
double number_field(const Json& object, const char* key, double fallback, bool positive = false);
int int_field(const Json& object, const char* key, int fallback, int minimum = 1);
bool bool_field(const Json& object, const char* key, bool fallback);
std::string string_field(const Json& object, const char* key, const std::string& fallback);
void check_keys(const Json& object, std::initializer_list<const char*> allowed);
Json core_version();

struct RequestSpec {
    std::string request_id;
    double arrival_ms = 0, deadline_ms = 0;
    std::string model = "reference";
    int height = 512, width = 512, frames = 1, steps = 20;
    std::int64_t seed = 0;
    double cache_threshold = 0.15;
    bool cache_enabled = true;
    void validate() const;
    std::tuple<std::string, int, int, int> shape_key() const;
    Json to_json() const;
    static RequestSpec from_json(const Json& value);
};

std::vector<RequestSpec> read_workload(const Json& value);
void validate_workload(const std::vector<RequestSpec>& specs);
}
