#pragma once

#include "ditserve/control/cluster.hpp"
#include "ditserve/types.hpp"
#include <string>

namespace ditserve {
// Cluster policy registry. Every behaviour a policy name implies lives in one
// PolicyTraits row, so the planner, the executor's capability checks and the
// simulator read the same facts instead of comparing names. Study options may
// still override the documented defaults (see each field).
//
// Families:
//   ours       DitServe and its approximate variant.
//   baseline   re-implementations of published or preprint serving methods.
//              Preprint-only methods are named by method, not by system.
//   heuristic  simple rules the system must beat.
//   ablation   DitServe with one mechanism removed, plus the earlier designs.
//   control    fixed-form arms for calibration and paired digest checks.
//   diagnostic research paths outside the paper tables.
struct PolicyTraits {
    std::string id;           // canonical name written to new records
    std::string display;      // name used in tables
    std::string family;       // ours | baseline | heuristic | ablation | control | diagnostic
    std::string source;       // where the method comes from and what is not reproduced
    Form fixed_form;          // static forms: the one form every video runs
    int fixed_degree = 0;     // 0: the policy chooses among options.forms
    bool elastic = false;     // regroup at step boundaries
    bool blocks = false;      // block quanta and sub-step (doorbell) preemption
    bool rounds = false;      // regroup only every options.round_steps steps
    bool knapsack = false;    // DitServe v1 hand-weighted knapsack utility
    bool deadline_dp = false; // 0/1 group knapsack: maximise requests on track, minimal feasible degree
    bool best_fit = false;    // EDF best fit plus surplus redistribution
    bool approx = false;      // whole-step feature-cache research path
    bool v3 = false;          // INT8 Ring + Taylor research path
    bool v5 = false;          // late-step CFG folding (approximate)
    bool fission = false;     // release the unconditional CFG subgroup; main Approx does so on demand
    // Defaults that options may override.
    double widen = 0.0;                    // options.widen_efficiency
    std::string image_placement = "first"; // options.image_placement
    bool hold_diffuse = false;             // options.hold_diffuse: a started video keeps its GPUs between steps
    bool demote_late = true;               // options.demote_late
    std::string order = "edf";             // options.order
    bool prune_dominated = false;          // options.prune_dominated
    int min_video_degree = 1;              // options.min_video_degree
    int partition_divisor = 0;             // >0: reserve gpus / divisor GPUs for images (options.partition_image_gpus)
    int round_steps = 5;                   // options.round_steps
    bool tiered_memory = false;            // options.tiered_memory
    bool image_batching = false;           // batch waiting images when the profile prices batches
    bool pin_started = false;              // options.pin_started
    bool allow_cfg_forms = false;          // options.allow_cfg_forms: allow CFG=2 forms for baseline planners
    bool feasible_img_priority = true;     // options.feasible_img_priority: boost feasible images in DitServe EDF queue
};

// Throws std::invalid_argument for unknown or retired names. Accepts the
// legacy names of earlier records (see legacy_policy_alias).
const PolicyTraits& policy_traits(const std::string& name);
// Canonical ID for a name, resolving legacy aliases.
std::string canonical_policy(const std::string& name);
// Non-empty when `name` is a legacy alias: the canonical ID it maps to.
std::string legacy_policy_alias(const std::string& name);
// Every registered policy (parameterised static-/block-/xdit- forms listed by pattern).
Json policy_catalog();
}
