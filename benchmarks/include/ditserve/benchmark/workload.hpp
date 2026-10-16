#pragma once

#include "ditserve/types.hpp"
#include <tuple>

namespace ditserve {
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

struct CostModel;
std::vector<RequestSpec> generate_workload(const Json& generator, const CostModel& cost);
}
