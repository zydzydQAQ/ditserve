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

// Cost profile per request class and SP degree. Units are milliseconds.
//   step_ms(d, n) = step_compute_ms / (d * sp_efficiency)
//                 + comm_mb_per_step * (d-1)/d^2 / gbps(d, n)   (Ulysses all-to-all per rank)
//                 + collectives_per_step * links[d].latency_ms
// gbps(d, n) is links[d].gbps unless n >= 2 SP groups advance at once and the
// link names concurrent_gbps {"k": rate}: then the rate of the largest k <= n
// (groups contend for the host bridge). quantum_overhead_ms is the fixed
// end-to-end cost of dispatching one quantum.
class ClusterCost {
public:
    explicit ClusterCost(const Json& profile);
    const Json& profile() const { return profile_; }
    const Json& klass(const std::string& name) const;
    // sp_groups: SP groups (degree >= 2) advancing concurrently, this one included.
    double step_ms(const std::string& klass, int degree, int sp_groups = 1) const;
    double block_ms(const std::string& klass, int degree, int sp_groups = 1) const;
    int blocks_per_step(const std::string& klass) const;
    // VAE decode splits into decode_chunks latent-frame chunks (1 = atomic).
    int decode_chunks(const std::string& klass) const;
    double chunk_ms(const std::string& klass) const;
    // Remaining GPU time on `degree` GPUs from the request's current progress.
    double remaining_ms(const Json& request, int degree, int sp_groups = 1) const;
    double resize_ms() const { return resize_ms_; }
    double quantum_overhead_ms() const { return overhead_ms_; }
private:
    Json profile_;
    double efficiency_ = 1, resize_ms_ = 0, overhead_ms_ = 0;
};

// state: {now_ms, gpus, busy:[bool], requests:[...], sp_groups_running (SP
// advances still running, default 0)}; request fields:
//   request_id, kind (video|image), class, arrival_ms, deadline_ms,
//   status waiting|suspended, ready (encoded), prepared, steps_done,
//   blocks_done (inside the current step), group ([] when unplaced), optional
//   steps_total (overrides the class), affinity (GPU that must host prepare) and
//   decode_done (latent-frame chunks already decoded).
// Policies: static_sp1|static_sp2|static_sp4|static_sp8, step_elastic, block_elastic.
// Returns {assignments:[{request_id, group, degree, quantum, resize, reason}]}
// where quantum is "atomic" (image), "finish" (with decode_frames when the
// decode is chunked), "step" or a block count.
Json cluster_plan(const Json& state, const std::string& policy, const ClusterCost& cost, const Json& options = Json::object());
bool cluster_policy(const std::string& policy);
}
