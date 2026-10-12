#pragma once
#include "ditserve/cost.hpp"

namespace ditserve {
std::vector<RequestSpec> generate_workload(const Json& generator, const CostModel& cost);
}
