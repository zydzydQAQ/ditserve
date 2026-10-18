#pragma once

#include "ditserve/types.hpp"

namespace ditserve {
// Workload generation, request metrics, gates and matrix expansion shared by
// the serving executor, the simulator and the rpc interface.
Json study_workload(const Json& phase);
// Request IDs are unique within observation_id (absent means one trace).
// Exposure sums each observation's maximum observation_ms or terminal finish_ms.
Json request_summary(const Json& rows);
// Deadline attainment and latency quantiles for one request kind; failed and
// unfinished requests stay in the denominator.
Json kind_summary(const Json& rows, const std::string& kind);
Json evaluate_gates(const Json& evidence, const Json& rules);
Json expand_matrix(const Json& phase);
}
