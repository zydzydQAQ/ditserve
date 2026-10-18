#pragma once
#include "ditserve/types.hpp"
namespace ditserve {
Json run_study(const Json& request);
Json simulate_study(const Json& request);
// The checks run_study applies before any worker starts; throws on the first problem.
void validate_study(const Json& config);
// Illustrative cluster cost profile used when a simulation names none.
Json synthetic_profile();
}
