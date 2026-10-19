#pragma once

#include "ditserve/types.hpp"
#include <string>
#include <vector>

namespace ditserve {
// Multi-GPU dispatch shared by the real cluster executor and the simulator.
// GPUs are numbered in topology order, so aligned power-of-two groups map to
// PCIe switch pairs, socket quads and the whole node. Requests advance in
// quanta; between quanta a request owns no GPU and the plan is recomputed.

// Aligned groups of each allowed degree that fit in `gpus`.
std::vector<std::vector<int>> cluster_groups(int gpus, const std::vector<int>& degrees);

// How a video's group of cfg * sp GPUs runs one solver step: the two CFG
// branches on separate halves (cfg = 2, one prediction exchange per step) and
// Ulysses sequence parallelism of degree sp inside each branch. Names:
// sp1, sp2, sp4, sp8, cfg2, cfg2sp2, cfg2sp4.
struct Form {
    int cfg = 1;
    int sp = 1;
    int degree() const { return cfg * sp; }
    bool operator==(const Form& other) const { return cfg == other.cfg && sp == other.sp; }
    bool operator!=(const Form& other) const { return !(*this == other); }
};
Form parse_form(const std::string& name);
std::string form_name(const Form& form);

// Cost profile per request class and parallel form. Units are milliseconds.
//   step_ms(form, n) = step_compute_ms / (cfg * sp * sp_efficiency)
//                    + comm_mb_per_step / cfg * (sp-1)/sp^2 / gbps(sp, n)   (Ulysses all-to-all per rank)
//                    + collectives_per_step / cfg * links[sp].latency_ms
//                    + [cfg = 2] cfg_exchange_mb / links[2].gbps + links[2].latency_ms
// gbps(sp, n) is links[sp].gbps unless n >= 2 SP groups advance at once and the
// link names concurrent_gbps {"k": rate}: then the rate of the largest k <= n
// (groups contend for the host bridge). With cfg = 1 the worker may interleave
// the two branches so one branch's all-to-all overlaps the other's compute:
// sp_overlap in [0, 1] is the fraction of the shorter of the compute and
// communication terms hidden that way (1: max of the two). quantum_overhead_ms
// is the fixed end-to-end cost of dispatching one quantum.
class ClusterCost {
public:
    explicit ClusterCost(const Json& profile);
    const Json& profile() const { return profile_; }
    const Json& klass(const std::string& name) const;
    // sp_groups: SP groups (sp >= 2) advancing concurrently, this one included.
    double step_ms(const std::string& klass, const Form& form, int sp_groups = 1) const;
    double block_ms(const std::string& klass, const Form& form, int sp_groups = 1) const;
    // Remaining GPU time on `form` from the request's current progress.
    double remaining_ms(const Json& request, const Form& form, int sp_groups = 1) const;
    // Plain sequence parallelism of `degree` GPUs.
    double step_ms(const std::string& klass, int degree, int sp_groups = 1) const { return step_ms(klass, Form{1, degree}, sp_groups); }
    double block_ms(const std::string& klass, int degree, int sp_groups = 1) const { return block_ms(klass, Form{1, degree}, sp_groups); }
    double remaining_ms(const Json& request, int degree, int sp_groups = 1) const { return remaining_ms(request, Form{1, degree}, sp_groups); }
    int blocks_per_step(const std::string& klass) const;
    // VAE decode splits into decode_chunks latent-frame chunks (1 = atomic).
    int decode_chunks(const std::string& klass) const;
    double chunk_ms(const std::string& klass) const;
    double resize_ms() const { return resize_ms_; }
    double quantum_overhead_ms() const { return overhead_ms_; }
private:
    Json profile_;
    double efficiency_ = 1, resize_ms_ = 0, overhead_ms_ = 0, overlap_ = 0;
};

// state: {now_ms, gpus, busy:[bool], requests:[...], sp_groups_running (SP
// advances still running, default 0)}; request fields:
//   request_id, kind (video|image), class, arrival_ms, deadline_ms,
//   status waiting|suspended, ready (encoded), prepared, steps_done,
//   blocks_done (inside the current step), group ([] when unplaced), form (the
//   group's parallel form, default sp<group size>), optional steps_total
//   (overrides the class), affinity (GPU that must host prepare) and
//   decode_done (latent-frame chunks already decoded).
// Policies: static_sp1|static_sp2|static_sp4|static_sp8, static_cfg2,
// static_cfg2sp2, static_cfg2sp4, step_elastic, block_elastic, round_elastic
// (regroups only every options.round_steps steps). Elastic policies choose
// among options.forms (default: sp forms of options.degrees, default
// {1, 2, 4, 8}); for each group size the form with the smallest predicted
// step time wins. options.order (edf|srtf|fcfs) ranks feasible requests.
// Returns {assignments:[{request_id, group, degree, form, quantum, resize, reason}]}
// where quantum is "atomic" (image), "finish" (with decode_frames when the
// decode is chunked), "step" or a block count.
Json cluster_plan(const Json& state, const std::string& policy, const ClusterCost& cost, const Json& options = Json::object());
bool cluster_policy(const std::string& policy);
}
