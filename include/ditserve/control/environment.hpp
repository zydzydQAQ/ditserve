#pragma once

#include "ditserve/types.hpp"

namespace ditserve {
// One machine's binding for environment-neutral studies (configs/env/*.json,
// schema ditserve.env.v1): which GPUs, where the models live, worker
// environment and backend defaults, and the measured cost profile. The
// caller loads referenced files first; these functions do no I/O.

// GPUs the profile selects from a node inventory [{index, uuid, name,
// memory_mib}], in the profile's order. Pinned UUIDs must all be present;
// otherwise the lowest-index GPUs that match name_contains and
// min_memory_mib and are not in forbid_uuids are taken.
Json select_gpus(const Json& env, const Json& inventory);

// Returns a concrete study for this machine:
//   * workers with per_gpu become name.0 .. name.{k-1}, one per selected GPU;
//     gpu: N binds the Nth selected GPU; model: <logical name> becomes
//     config.model_path (relative paths resolve against models_root);
//   * env worker_environment and backends.<backend> are defaults the study
//     overrides;
//   * a cluster phase's gpus {video, image} expands to one slot per GPU (or
//     gpu_count); warmup and operations workers expand to their instances;
//   * phases with min_gpus above the selection, and phases that require
//     them, are skipped and recorded;
//   * any string "@env.KEY" anywhere is replaced by the profile's KEY;
//   * the experiment is prefixed with the environment name, and
//     "environment" records the inventory and every binding.
// With a null inventory (simulations) only the substitution, prefix and
// record apply, and GPU-bound workers are rejected.
Json resolve_study(const Json& study, const Json& env, const Json& inventory);
}
