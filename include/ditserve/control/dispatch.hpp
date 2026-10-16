#pragma once

#include "ditserve/types.hpp"

namespace ditserve {
// These functions consume observations only. They never load models or inspect
// future/unready requests when making a dispatch decision.
Json select_action(const Json& rows, double now_ms, const std::string& policy,
                   const Json& costs = Json::object(), const std::string& active = "",
                   const Json& quantum = 1);
Json study_workload(const Json& phase);
// Request IDs are unique within observation_id (absent means one trace).
// Exposure sums each observation's maximum observation_ms or terminal finish_ms.
Json request_summary(const Json& rows);
Json evaluate_gates(const Json& evidence, const Json& rules);
Json expand_matrix(const Json& phase);
}
