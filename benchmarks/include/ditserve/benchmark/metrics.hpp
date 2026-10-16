#pragma once
#include "ditserve/types.hpp"

namespace ditserve {
Json summarize(const Json& records, double start_ms, double end_ms, double gpu_busy_ms, int num_gpus);
}
