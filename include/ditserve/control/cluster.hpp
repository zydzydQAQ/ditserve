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
//   step_ms(d) = step_compute_ms / (d * sp_efficiency)
//              + comm_mb_per_step * (d-1)/d^2 / (links[d].gbps)   (Ulysses all-to-all per rank)
//              + collectives_per_step * links[d].latency_ms
class ClusterCost {
public:
    explicit ClusterCost(const Json& profile);
    const Json& profile() const { return profile_; }
    const Json& klass(const std::string& name) const;
    double step_ms(const std::string& klass, int degree) const;
    double block_ms(const std::string& klass, int degree) const;
    int blocks_per_step(const std::string& klass) const;
    // VAE decode splits into decode_chunks latent-frame chunks (1 = atomic).
    int decode_chunks(const std::string& klass) const;
    double chunk_ms(const std::string& klass) const;
    // Remaining GPU time on `degree` GPUs from the request's current progress.
    double remaining_ms(const Json& request, int degree) const;
    double resize_ms() const { return resize_ms_; }
private:
    Json profile_;
    double efficiency_ = 1, resize_ms_ = 0;
};

// A workload generator may state `load` (offered GPU utilization) instead of
// rate_rps and `count_per_gpu` instead of count, so one config scales across
// clusters. Missing per-class service_ms (the SLO base) comes from the
// profile: encode + prepare + steps * step_ms(1) + finish for videos. The
// result carries `derived` {mean_gpu_service_ms, capacity_rps, load}.
Json resolve_generator(const Json& generator, const ClusterCost& cost, int gpus);

// state: {now_ms, gpus, busy:[bool], requests:[...]}; request fields:
//   request_id, kind (video|image), class, arrival_ms, deadline_ms,
//   status waiting|suspended, ready (encoded), prepared, steps_done,
//   blocks_done (inside the current step), group ([] when unplaced), optional
//   steps_total (overrides the class), affinity (GPU that must host prepare) and
//   decode_done (latent-frame chunks already decoded). State may also carry
//   last_image_arrival_ms, updated only on image admission (never a future time).
// Policies: static_sp1|static_sp2|static_sp4|static_sp8, step_elastic, block_elastic.
// Options: quantum_ms (required by block_elastic), degrees, pin_started,
// reserve_image_gpus, demote_late, late_short_ms, chunked_decode,
// reserve_transition_union (reserve the source and target during state handoff).
// adaptive_quantum (block_elastic only, default false): use
// adaptive_hot_quantum_ms (default 100, positive and < quantum_ms <= 10000)
// while an arrived image waits or its last arrival is within
// adaptive_hot_window_ms (default 1000, in (0,60000]); otherwise quantum_ms.
// decode_frames (optional integer [1,64]) fixes latent frames independently of
// DiT quantum; requires chunked_decode. Existing options retain their behavior.
// Adaptive or explicit decode_frames assignments include quantum_feedback;
// numeric DiT assignments also carry chosen_quantum_ms and block_count.
// Returns {assignments:[{request_id, group, degree, quantum, resize, reason}]}
// with reserved_group when union reservation is requested.
// where quantum is "atomic" (image), "finish" (with decode_frames when the
// decode is chunked), "step" or a block count.
Json cluster_plan(const Json& state, const std::string& policy, const ClusterCost& cost, const Json& options = Json::object());
bool cluster_policy(const std::string& policy);
}
