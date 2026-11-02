#include "ditserve/benchmark/cluster_sim.hpp"
#include "ditserve/control/policy.hpp"
#include "ditserve/control/cluster.hpp"
#include "ditserve/control/dispatch.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace ditserve {
namespace {
struct Running {
    std::size_t row;
    Json assignment;
    double end;
    int blocks;
    std::size_t partner = SIZE_MAX;   // the second request of a lockstep pair
    std::vector<std::size_t> batch;   // the other images of a batch
};
}

Json simulate_cluster(const Json& config) {
    // `profile` gives the durations; the planner may believe another one.
    const ClusterCost cost(config.at("profile"));
    const ClusterCost planner(config.contains("planner_profile") ? config.at("planner_profile") : config.at("profile"));
    const auto policy = config.at("policy").get<std::string>();
    const int gpus = int_field(config, "gpus", 1);
    const Json options = config.value("options", Json::object());
    const int encode_slots = int_field(config, "encode_slots", gpus);
    const double dispatch = number_field(config, "dispatch_ms");
    Json rows = config.at("workload");
    if (!rows.is_array() || rows.empty()) throw std::invalid_argument("simulation workload must be a nonempty array");
    std::map<std::string, std::size_t> index;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        auto& row = rows[i];
        const auto klass = row.at("class").get<std::string>();
        if (cost.klass(klass).at("kind") != row.at("kind")) throw std::invalid_argument("request kind differs from its cost class");
        // Keep only scheduling fields: every plan copies the queue.
        Json light;
        for (const auto* key : {"request_id", "kind", "class", "arrival_ms", "deadline_ms", "slo_ms", "service_ms", "steps_total", "decode_done", "batch_spec"})
            if (row.contains(key)) light[key] = row.at(key);
        row = light;
        row["status"] = "pending"; row["steps_done"] = 0; row["blocks_done"] = 0; row["prepared"] = false;
        if (row.at("kind") == "image" && int_field(options, "continuous_batch", 0, 0) > 0) row["continuous_batch"] = true;
        row["group"] = Json::array(); row["ready"] = false; row["degrees"] = Json::array(); row["forms"] = Json::array();
        index[row.at("request_id").get<std::string>()] = i;
    }
    std::vector<double> encoder_free(encode_slots, 0);
    std::vector<bool> busy(gpus, false);
    std::vector<Running> running;
    // The request that last ran on each GPU: a video resuming behind another request pays resume_ms.
    std::vector<std::size_t> last_on_gpu(gpus, SIZE_MAX);
    // Per-request action time by quantum kind, like the executor's action_ms.
    std::vector<Json> action_ms(rows.size(), Json::object());
    std::vector<int> quanta(rows.size(), 0);
    auto form_of = [](const Json& assignment) {
        return parse_form(assignment.value("form", "sp" + std::to_string(assignment.at("degree").get<int>())));
    };
    // Only sequence-parallel advances share the host bridge; a CFG pair exchanges one prediction per step.
    auto sp_advance = [&](const Json& assignment) {
        return form_of(assignment).sp_total() >= 2 && assignment.at("quantum") != "atomic" && assignment.at("quantum") != "finish";
    };
    double now = 0, busy_ms = 0, planner_cpu_ms_total = 0, planner_cpu_ms_max = 0;
    std::size_t completed = 0, preemptions = 0, resizes = 0, plans = 0, pairs = 0, batches = 0, batched = 0;
    std::size_t tiered_offloads = 0, tiered_prefetches = 0, approx_quanta = 0;
    std::vector<bool> prefetched(rows.size(), false);
    std::vector<std::size_t> arrival_order(rows.size());
    for (std::size_t i = 0; i < rows.size(); ++i) arrival_order[i] = i;
    std::stable_sort(arrival_order.begin(), arrival_order.end(), [&](std::size_t a, std::size_t b) {
        return number_field(rows[a], "arrival_ms") < number_field(rows[b], "arrival_ms");
    });

    for (;;) {
        // Arrivals in trace order; videos queue for a CPU prompt encoder.
        for (std::size_t i : arrival_order) {
            auto& row = rows[i];
            if (row.at("status") != "pending" || number_field(row, "arrival_ms") > now) continue;
            const auto& klass = cost.klass(row.at("class").get<std::string>());
            if (row.at("kind") == "video") {
                auto slot = std::min_element(encoder_free.begin(), encoder_free.end());
                const double start = std::max(number_field(row, "arrival_ms"), *slot);
                *slot = start + number_field(klass, "encode_ms");
                row["ready_ms"] = *slot; row["status"] = "encoding";
            } else { row["status"] = "waiting"; row["ready"] = true; }
        }
        for (auto& row : rows)
            if (row.at("status") == "encoding" && number_field(row, "ready_ms") <= now) { row["status"] = "waiting"; row["ready"] = true; }

        Json busy_flags = Json::array();
        for (bool flag : busy) busy_flags.push_back(flag);
        Json requests = Json::array();
        for (const auto& row : rows) {
            if (row.at("status") != "waiting" && row.at("status") != "suspended") continue;
            Json view = row;
            if (options.contains("cfg_fold_eligible_from_step") && row.at("kind") == "video") {
                const auto klass = row.at("class").get<std::string>();
                const int steps = int_field(row, "steps_total", int_field(cost.klass(klass), "steps", 0), 1);
                const int start = int_field(options, "cfg_fold_eligible_from_step", steps, 0);
                view["cfg_fold_eligible"] = int_field(row, "steps_done", 0, 0) >= start
                                            && int_field(row, "steps_done", 0, 0) < steps - 1;
            }
            requests.push_back(view);
        }
        if (!requests.empty() && std::find(busy.begin(), busy.end(), false) != busy.end()) {
            ++plans;
            int sp_running = 0;
            for (const auto& item : running) sp_running += sp_advance(item.assignment);
            const auto plan_t0 = std::chrono::steady_clock::now();
            const auto plan = cluster_plan({{"now_ms", now}, {"gpus", gpus}, {"busy", busy_flags}, {"requests", requests}, {"sp_groups_running", sp_running}},
                                           policy, planner, options);
            const double plan_dt_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - plan_t0).count();
            planner_cpu_ms_total += plan_dt_ms;
            planner_cpu_ms_max = std::max(planner_cpu_ms_max, plan_dt_ms);
            // SP quanta of this plan run alongside the running ones. Each is priced at
            // dispatch; a later change in concurrency does not re-time it.
            int sp_groups = sp_running;
            for (const auto& assignment : plan.at("assignments")) sp_groups += sp_advance(assignment);
            for (const auto& assignment : plan.at("assignments")) {
                const std::size_t at = index.at(assignment.at("request_id").get<std::string>());
                auto& row = rows[at];
                const auto klass = row.at("class").get<std::string>();
                const int degree = assignment.at("degree").get<int>();
                const Form form = form_of(assignment);
                if (form.degree() != degree || degree != static_cast<int>(assignment.at("group").size())) throw std::logic_error("assignment form does not match its group");
                const auto quantum = assignment.at("quantum");
                double duration = dispatch + cost.quantum_overhead_ms();
                const double resume_scale = number_field(options, "resume_scale", 1.0);
                const double cross_modal_swap_ms = number_field(options, "cross_modal_swap_ms", 0.0);
                if (cross_modal_swap_ms > 0.0) {
                    for (const auto& gpu : assignment.at("group")) {
                        const int g = gpu.get<int>();
                        if (last_on_gpu[g] != SIZE_MAX && rows[last_on_gpu[g]].at("kind") != row.at("kind")) {
                            duration += cross_modal_swap_ms;
                            break;
                        }
                    }
                }
                int blocks = 0;
                if (quantum == "atomic") duration += assignment.contains("batch") ? cost.batch_ms(klass, static_cast<int>(assignment.at("batch").size()))
                                                                               : number_field(cost.klass(klass), "service_ms");
                else if (quantum == "continuous_batch") {
                    const bool bubble_free = bool_field(options, "bubble_free", false);
                    const auto& batch_arr = assignment.at("batch");
                    duration += cost.continuous_step_ms(klass, static_cast<int>(batch_arr.size()));
                    if (!bubble_free) {
                        for (const auto& member : batch_arr) {
                            const auto& mrow = rows[index.at(member.get<std::string>())];
                            if (!mrow.value("prepared", false)) duration += number_field(cost.klass(klass), "prepare_ms");
                            const int steps = int_field(mrow, "steps_total", int_field(cost.klass(klass), "steps", 2), 1);
                            if (int_field(mrow, "steps_done", 0, 0) + 1 >= steps) duration += number_field(cost.klass(klass), "finish_ms");
                        }
                    }
                }
                else if (quantum == "finish") {
                    blocks = assignment.value("decode_frames", cost.decode_chunks(klass) - row.value("decode_done", 0));
                    const bool async_decode = bool_field(options, "async_rank0_decode", policy_traits(policy).v5);
                    double async_scale = 1.0;
                    if (async_decode) {
                        if (!cost.klass(klass).contains("async_decode_scale") && !options.contains("async_decode_scale"))
                            throw std::invalid_argument("async decode simulation requires an explicit measured scale");
                        async_scale = number_field(cost.klass(klass), "async_decode_scale",
                                                   number_field(options, "async_decode_scale", 1.0));
                        if (async_scale <= 0.0 || async_scale > 1.0)
                            throw std::invalid_argument("async decode scale must be in (0, 1]");
                    }
                    duration += blocks * cost.chunk_ms(klass) * async_scale;
                }
                else {
                    const int remaining = cost.blocks_per_step(klass) - row.at("blocks_done").get<int>();
                    blocks = quantum == "step" ? remaining : std::min(quantum.get<int>(), remaining);
                    if (!row.at("prepared").get<bool>()) duration += number_field(cost.klass(klass), "prepare_ms");
                    if (assignment.at("resize").get<bool>()) { duration += cost.resize_ms(); ++resizes; }
                    std::vector<int> group;
                    for (const auto& gpu : assignment.at("group")) group.push_back(gpu.get<int>());
                    const double cache_threshold = number_field(assignment, "cache_threshold", 0.0);
                    const bool cfg_folded = assignment.value("cfg_folded", false);
                    if (cache_threshold > 0.0 || cfg_folded) ++approx_quanta;
                    if (row.at("prepared").get<bool>()) {
                        for (int gpu : group) if (last_on_gpu[gpu] != at) {
                            duration += cost.resume_ms(klass) * (prefetched[at] ? 0.25 : resume_scale);
                            prefetched[at] = false;
                            break;
                        }
                    }
                    if (assignment.contains("pair")) {
                        // Two requests in lockstep: the pair's block time, plus the partner's prepare/resize.
                        auto& other = rows[index.at(assignment.at("pair").get<std::string>())];
                        const auto oklass = other.at("class").get<std::string>();
                        if (!other.at("prepared").get<bool>()) duration += number_field(cost.klass(oklass), "prepare_ms");
                        if (assignment.value("pair_resize", false)) { duration += cost.resize_ms(); ++resizes; }
                        duration += blocks * cost.pair_ms(klass, oklass, form, form.sp_total() >= 2 ? sp_groups : 1, &group) / cost.blocks_per_step(klass);
                    } else {
                        double fold_scale = 1.0;
                        if (cfg_folded && form.cfg == 1) {
                            if (!cost.klass(klass).contains("cfg_fold_step_scale") && !options.contains("cfg_fold_step_scale"))
                                throw std::invalid_argument("CFG-fold simulation requires an explicit measured scale");
                            fold_scale = number_field(cost.klass(klass), "cfg_fold_step_scale",
                                                      number_field(options, "cfg_fold_step_scale", 1.0));
                            if (fold_scale <= 0.0 || fold_scale > 1.0)
                                throw std::invalid_argument("CFG-fold scale must be in (0, 1]");
                        }
                        duration += blocks * cost.block_ms(klass, form, form.sp_total() >= 2 ? sp_groups : 1, &group, cache_threshold) * fold_scale;
                    }
                }
                const auto kind = quantum.is_string() ? quantum.get<std::string>() : std::string("blocks");
                std::vector<std::size_t> members = {at};
                if (assignment.contains("pair")) members.push_back(index.at(assignment.at("pair").get<std::string>()));
                if (assignment.contains("batch")) for (const auto& member : assignment.at("batch")) {
                    const std::size_t m = index.at(member.get<std::string>());
                    if (m != at) members.push_back(m);
                }
                for (std::size_t member : members) {
                    auto& item = rows[member];
                    action_ms[member][kind] = action_ms[member].value(kind, 0.) + duration;
                    ++quanta[member];
                    if (!item.contains("start_ms")) item["start_ms"] = now;
                    if (std::find(item["degrees"].begin(), item["degrees"].end(), degree) == item["degrees"].end()) item["degrees"].push_back(degree);
                    if (kind == "step" || kind == "blocks") {
                        const auto name = form_name(form);
                        if (std::find(item["forms"].begin(), item["forms"].end(), name) == item["forms"].end()) item["forms"].push_back(name);
                    }
                    item["status"] = "running";
                }
                preemptions += assignment.at("reason") == "block_preempt";
                pairs += assignment.contains("pair");
                batches += assignment.contains("batch");
                if (assignment.contains("batch")) batched += assignment.at("batch").size();
                for (const auto& gpu : assignment.at("group")) { busy[gpu.get<int>()] = true; last_on_gpu[gpu.get<int>()] = at; }
                busy_ms += duration * degree;
                Running item{at, assignment, now + duration, blocks, assignment.contains("pair") ? members[1] : SIZE_MAX, {}};
                if (assignment.contains("batch")) item.batch.assign(members.begin() + 1, members.end());
                running.push_back(item);
            }
            if (plan.contains("tiered_hints")) {
                const auto& hints = plan.at("tiered_hints");
                if (hints.contains("offload")) tiered_offloads += hints.at("offload").size();
                if (hints.contains("prefetch")) {
                    for (const auto& pid : hints.at("prefetch")) {
                        const auto s = pid.get<std::string>();
                        if (index.count(s)) { prefetched[index.at(s)] = true; ++tiered_prefetches; }
                    }
                }
            }
        }

        double next = std::numeric_limits<double>::infinity();
        for (const auto& item : running) next = std::min(next, item.end);
        for (const auto& row : rows) {
            if (row.at("status") == "pending") next = std::min(next, number_field(row, "arrival_ms"));
            if (row.at("status") == "encoding") next = std::min(next, number_field(row, "ready_ms"));
        }
        if (!std::isfinite(next)) {
            if (completed != rows.size()) throw std::runtime_error("simulation stalled: policy cannot place the remaining requests");
            break;
        }
        now = std::max(now, next);
        for (auto it = running.begin(); it != running.end();) {
            if (it->end > now) { ++it; continue; }
            auto& row = rows[it->row];
            const auto quantum = it->assignment.at("quantum");
            for (const auto& gpu : it->assignment.at("group")) busy[gpu.get<int>()] = false;
            const auto klass = row.at("class").get<std::string>();
            if (quantum == "finish" && row.value("decode_done", 0) + it->blocks < cost.decode_chunks(klass)) {
                row["decode_done"] = row.value("decode_done", 0) + it->blocks; row["status"] = "suspended";
                row["group"] = it->assignment.at("group"); row["form"] = "sp1";
            } else if (quantum == "atomic" || quantum == "finish") {
                row["status"] = "completed"; row["finish_ms"] = it->end; row["group"] = Json::array(); row.erase("form"); ++completed;
                for (std::size_t m : it->batch) {
                    auto& other = rows[m];
                    other["status"] = "completed"; other["finish_ms"] = it->end; other["group"] = Json::array(); ++completed;
                }
            } else if (quantum == "continuous_batch") {
                const bool bubble_free = bool_field(options, "bubble_free", false);
                std::vector<std::size_t> members = {it->row};
                members.insert(members.end(), it->batch.begin(), it->batch.end());
                for (std::size_t m : members) {
                    auto& mrow = rows[m];
                    const auto mklass = mrow.at("class").get<std::string>();
                    const int steps = int_field(mrow, "steps_total", int_field(cost.klass(mklass), "steps", 2), 1);
                    const int next_step = int_field(mrow, "steps_done", 0, 0) + 1;
                    mrow["steps_done"] = next_step;
                    mrow["prepared"] = true;
                    if (next_step >= steps) {
                        mrow["status"] = "completed";
                        mrow["finish_ms"] = it->end + (bubble_free ? number_field(cost.klass(mklass), "finish_ms") : 0.0);
                        mrow["group"] = Json::array();
                        mrow.erase("form");
                        ++completed;
                    } else {
                        mrow["status"] = "suspended";
                        mrow["group"] = it->assignment.at("group");
                        mrow["form"] = "sp1";
                    }
                }
            } else {
                std::vector<std::size_t> members = {it->row};
                if (it->partner != SIZE_MAX) members.push_back(it->partner);
                for (std::size_t member : members) {
                    auto& item = rows[member];
                    const auto iklass = item.at("class").get<std::string>();
                    int blocks = item.at("blocks_done").get<int>() + it->blocks;
                    if (blocks >= cost.blocks_per_step(iklass)) { blocks = 0; item["steps_done"] = item.at("steps_done").get<int>() + 1; }
                    item["blocks_done"] = blocks; item["prepared"] = true;
                    item["cfg_folded"] = it->assignment.value("cfg_folded", false);
                    item["group"] = it->assignment.at("group"); item["form"] = form_name(form_of(it->assignment)); item["status"] = "suspended";
                    // A pair stopped inside a step resumes together; at a step boundary it dissolves.
                    if (members.size() > 1 && blocks > 0) item["pair"] = rows[members[member == members[0] ? 1 : 0]].at("request_id");
                    else item.erase("pair");
                }
            }
            it = running.erase(it);
        }
    }
    const double makespan = now;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        auto& row = rows[i];
        row["observation_ms"] = makespan; row.erase("spec"); row.erase("ready");
        row["action_ms"] = action_ms[i]; row["quanta"] = quanta[i];
    }
    return {{"policy", policy}, {"gpus", gpus}, {"metrics", request_summary(rows)},
            {"by_kind", {{"video", kind_summary(rows, "video")}, {"image", kind_summary(rows, "image")}}},
            {"makespan_ms", makespan}, {"gpu_utilization", makespan > 0 ? busy_ms / (gpus * makespan) : 0.},
            {"block_preemptions", preemptions}, {"resizes", resizes}, {"plans", plans}, {"pairs", pairs},
            {"batches", batches}, {"batched_images", batched},
            {"tiered_offloads", tiered_offloads}, {"tiered_prefetches", tiered_prefetches}, {"approx_quanta", approx_quanta},
            {"provenance", cost.profile().at("provenance")}, {"requests", rows},
            {"planner_cpu_ms_total", bool_field(options, "measure_planner_cpu", false) ? planner_cpu_ms_total : 0.},
            {"planner_cpu_ms_mean", (bool_field(options, "measure_planner_cpu", false) && plans > 0) ? planner_cpu_ms_total / static_cast<double>(plans) : 0.},
            {"planner_cpu_ms_max", bool_field(options, "measure_planner_cpu", false) ? planner_cpu_ms_max : 0.},
            {"notice", "Cost-model simulation; durations come from the profile, not from executing models."}};
}
}
