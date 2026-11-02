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

// How a video's group of cfg * sp * ring GPUs runs one solver step: the two CFG
// branches on separate halves (cfg = 2, one prediction exchange per step),
// Ulysses sequence parallelism of degree sp inside each branch/socket, and
// optional Ring Attention P2P degree ring across sockets (2D Hybrid USP).
// Names: sp1, sp2, sp4, sp8, cfg2, cfg2sp2, cfg2sp4, u2r2, u4r2, cfg2u2r2.
struct Form {
    int cfg = 1;
    int sp = 1;
    int ring = 1;
    int degree() const { return cfg * sp * ring; }
    int sp_total() const { return sp * ring; }
    bool operator==(const Form& other) const { return cfg == other.cfg && sp == other.sp && ring == other.ring; }
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
// (groups contend for the host bridge). Rates belong to a placement:
// links.groups may name aligned groups by their first and last GPU ("0-1",
// "2-3", "0-3", ...) with their own gbps, latency_ms and sp_overlap; a form is
// then priced on the GPUs it runs on by the slowest of its SP subgroups,
// contention scaling that rate as it scales the degree's. With cfg = 1 the
// worker may interleave the two branches so one branch's all-to-all overlaps
// the other's compute: sp_overlap in [0, 1] is the fraction of the shorter of
// the compute and communication terms hidden that way (1: max of the two).
// quantum_overhead_ms is the fixed end-to-end cost of dispatching one quantum;
// block_overhead_ms is the executor's extra time per executed block outside
// measured model time, so a step costs blocks_per_step / cfg of it per rank.
// classes.<name>.forms.<form>.placements maps "first-last" rank ranges to
// {step_ms}; alternatively {step_ms, group:[ranks]} binds one calibration.
// Unmatched placements and concurrent SP execution use the analytical model.
// Legacy {step_ms} entries apply across homogeneous, unscoped placements only.
class ClusterCost {
public:
    explicit ClusterCost(const Json& profile);
    const Json& profile() const { return profile_; }
    const Json& klass(const std::string& name) const;
    // Whether global dominance can be inferred from one uncontended placement.
    bool placement_invariant(const std::string& name) const;
    // sp_groups: SP groups (sp >= 2) advancing concurrently, this one included.
    // group: the GPUs the form runs on, for per-group link rates (optional).
    // cache_threshold: dynamic residual feature cache threshold in [0, 1); >0 skips redundant DiT blocks.
    double step_ms(const std::string& klass, const Form& form, int sp_groups = 1, const std::vector<int>* group = nullptr,
                   double cache_threshold = 0.0) const;
    double block_ms(const std::string& klass, const Form& form, int sp_groups = 1, const std::vector<int>* group = nullptr,
                    double cache_threshold = 0.0) const;
    // Remaining GPU time on `form` from the request's current progress.
    double remaining_ms(const Json& request, const Form& form, int sp_groups = 1, const std::vector<int>* group = nullptr,
                        double cache_threshold = 0.0) const;
    // Fraction of DiT blocks skipped (compute + communication) at `cache_threshold`.
    double cache_skip_ratio(const std::string& klass, double cache_threshold) const;
    // One step of two requests advancing in lockstep on `form` (request-level interleaving):
    // the sum of both steps less pair_overlap of the shorter of their compute and
    // communication sums; neither request interleaves its own branches.
    double pair_ms(const std::string& first, const std::string& second, const Form& form, int sp_groups = 1,
                   const std::vector<int>* group = nullptr) const;
    double pair_overlap() const { return pair_overlap_; }
    // Service time of `size` images of an image class generated as one batch: the class's
    // batch_ms table ({"1": ms, "2": ms, ...}, interpolated between listed sizes), else
    // size times service_ms (no gain).
    double batch_ms(const std::string& klass, int size) const;
    // One denoise round for up to 8 resident/new images in continuous mode.
    // batch_step_ms is a separate measured table, never the atomic pipeline batch_ms.
    double continuous_step_ms(const std::string& klass, int size) const;
    // Plain sequence parallelism of `degree` GPUs.
    double step_ms(const std::string& klass, int degree, int sp_groups = 1) const { return step_ms(klass, Form{1, degree}, sp_groups); }
    double block_ms(const std::string& klass, int degree, int sp_groups = 1) const { return block_ms(klass, Form{1, degree}, sp_groups); }
    double remaining_ms(const Json& request, int degree, int sp_groups = 1) const { return remaining_ms(request, Form{1, degree}, sp_groups); }
    int blocks_per_step(const std::string& klass) const;
    // VAE decode splits into decode_chunks latent-frame chunks (1 = atomic).
    int decode_chunks(const std::string& klass) const;
    double chunk_ms(const std::string& klass) const;
    // Extra time of the first quantum a video runs after another request used its GPU (cold caches,
    // allocator growth); 0 unless the profile class carries resume_ms (measured by the executor's
    // block_ms_resumed - block_ms_steady per request).
    double resume_ms(const std::string& klass) const;
    double resize_ms() const { return resize_ms_; }
    double quantum_overhead_ms() const { return overhead_ms_; }
    double block_overhead_ms() const { return block_overhead_ms_; }
private:
    struct Terms { double compute = 0, comm = 0, latency = 0, exchange = 0, overlap = 0, blocks = 0; };
    Terms terms(const std::string& klass, const Form& form, int sp_groups, const std::vector<int>* group,
                double cache_threshold = 0.0) const;
    Json profile_;
    double efficiency_ = 1, resize_ms_ = 0, overhead_ms_ = 0, block_overhead_ms_ = 0, overlap_ = 0, pair_overlap_ = 0;
};

// state: {now_ms, gpus, busy:[bool], requests:[...], sp_groups_running (SP
// advances still running, default 0)}; request fields:
//   request_id, kind (video|image), class, arrival_ms, deadline_ms,
//   status waiting|suspended, ready (encoded), prepared, steps_done,
//   blocks_done (inside the current step), group ([] when unplaced), form (the
//   group's parallel form, default sp<group size>), optional steps_total
//   (overrides the class), affinity (GPU that must host prepare) and
//   decode_done (latent-frame chunks already decoded), pair (the request this
//   one is inside a step with, in lockstep on the same group).
// options.image_batch (default 1) lets a plan send up to that many waiting
// images of one class to a GPU as one batch when every member still meets its
// deadline at the batch's service time; the assignment lists them in `batch`.
// options.continuous_batch (default 0) uses explicit-step image states instead:
// 1 is the unbatched baseline, 2-8 admits resident and newly arrived states on
// one worker, with batch_step_ms pricing and per-member deadline checks.
// options.pairing lets an elastic policy advance two videos in lockstep on one
// group when the profile's pair_overlap makes that cheaper than their steps
// in sequence; the assignment then names the partner in `pair`.
// Policies: static_sp1|static_sp2|static_sp4|static_sp8, static_cfg2,
// static_cfg2sp2, static_cfg2sp4, step_elastic, block_elastic, round_elastic
// (regroups only every options.round_steps steps). Elastic policies choose
// among options.forms (default: sp forms of options.degrees, default
// {1, 2, 4, 8}); for each group size the form with the smallest predicted
// step time wins. options.order (edf|srtf|fcfs) ranks feasible requests.
// Returns {assignments:[{request_id, group, degree, form, quantum, resize, reason, pair?, pair_resize?}]}
// where quantum is "atomic" (image), "continuous_batch" (explicit-step image),
// "finish" (with decode_frames when the decode is chunked), "step" or a block count.
Json cluster_plan(const Json& state, const std::string& policy, const ClusterCost& cost, const Json& options = Json::object());
bool cluster_policy(const std::string& policy);

struct PreemptionTiming {
    double stop_ms = 0, remaining_ms = 0;
};
// Rank-local executed blocks, matching the worker's progress counter. A Wan
// multi-GPU poll consumes the preceding checkpoint's flag; interleaved CFG
// executes two blocks per iteration. These are ranking estimates, not bounds
// on collective, synchronization or worker queueing latency.
PreemptionTiming estimate_preemption_timing(double block_ms, int blocks_total, int blocks_done,
                                            double elapsed_ms, int check_blocks,
                                            int blocks_per_iteration, bool delayed_group_check);
// True only when stopping before the step boundary changes this image from late to on time.
bool preemption_converts_deadline(double now_ms, double deadline_ms, double service_ms,
                                  double stop_ms, double boundary_ms);

// Disjoint running groups that can stop at a block boundary. Already-rung groups
// reserve their future capacity; do not interrupt another video for the same demand.
struct PreemptionGroup {
    std::string request_id;
    std::vector<int> group;
    double stop_ms = 0, remaining_ms = 0, resume_ms = 0;
    bool rung = false;
};
// One entry per waiting image: -1 for any GPU, otherwise its resident GPU.
// Returns only newly selected request IDs. Cost is displaced GPU time at the
// safe point plus measured resume overhead, with fewer interrupted GPUs on ties.
std::vector<std::string> select_preemption_groups(const std::vector<int>& image_affinities,
                                                 const std::vector<PreemptionGroup>& running);
struct PreemptionDemand {
    int affinity = -1;
    double deadline_ms = 0, service_ms = 0;
};
struct DeadlinePreemptionSelection {
    std::vector<std::string> selected;
    std::size_t deadline_skips = 0;
};
// EDF demand matching: every newly selected group must save the image assigned
// to its reserved capacity. Already-rung capacity is consumed at most once.
DeadlinePreemptionSelection select_deadline_preemption_groups(double now_ms,
    const std::vector<PreemptionDemand>& waiting, const std::vector<PreemptionGroup>& running);
}
