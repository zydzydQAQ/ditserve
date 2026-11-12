#pragma once

#include "ditserve/types.hpp"

namespace ditserve {
// Discrete-event replay of a frozen workload on N GPUs. It calls the same
// cluster_plan as the real executor, with durations from a ClusterCost
// profile; results describe the cost model, never measured GPU behavior.
//
// config: {workload:[rows from study_workload], profile, policy, gpus,
//          options (cluster_plan), encode_slots (default gpus), dispatch_ms}
Json simulate_cluster(const Json& config);
}
