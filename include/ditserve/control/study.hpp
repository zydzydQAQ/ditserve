#pragma once
#include "ditserve/types.hpp"
namespace ditserve {
Json run_study(const Json& request);
Json simulate_study(const Json& request);
// Illustrative cluster cost profile used when a simulation names none.
Json synthetic_profile();
}
