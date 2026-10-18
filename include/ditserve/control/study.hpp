#pragma once
#include "ditserve/types.hpp"
namespace ditserve {
Json run_study(const Json& request);
Json simulate_study(const Json& request);
// Heartbeat verdict for one GPU occupancy observation during a phase: true keeps the
// run. A foreign process or any other non-exclusive state ends it at once; a probe
// that could not answer ("unknown") is tolerated until `tolerance` happen in a row.
bool isolation_holds(const Json& observation, int& unknown_streak, int tolerance = 2);
// Illustrative cluster cost profile used when a simulation names none.
Json synthetic_profile();
}
