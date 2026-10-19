#include "ditserve/benchmark/cluster_sim.hpp"
#include "ditserve/control/cluster.hpp"
#include "ditserve/control/dispatch.hpp"
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ditserve;
namespace {
int failures = 0, checks = 0;
void require(bool passed, const std::string& message) { if (!passed) throw std::runtime_error(message); }
void near(double actual, double expected, double tolerance = 1e-9) {
    require(std::abs(actual - expected) <= tolerance, "expected " + std::to_string(expected) + ", got " + std::to_string(actual));
}
void rejects(const std::function<void()>& action) {
    try { action(); } catch (const std::exception&) { return; }
    throw std::runtime_error("invalid input unexpectedly accepted");
}
void check(const std::string& name, const std::function<void()>& action) {
    ++checks;
    try { action(); }
    catch (const std::exception& error) { ++failures; std::cerr << name << ": " << error.what() << '\n'; }
}

Json profile() {
    return {{"provenance", "synthetic"},
            {"classes", {{"video", {{"kind", "video"}, {"steps", 4}, {"blocks_per_step", 10}, {"prepare_ms", 0}, {"encode_ms", 0},
                                    {"step_compute_ms", 1000}, {"comm_mb_per_step", 800}, {"collectives_per_step", 10}, {"finish_ms", 100}}},
                         {"image", {{"kind", "image"}, {"service_ms", 50}}}}},
            {"links", {{"2", {{"gbps", 20}, {"latency_ms", 0.1}}}, {"4", {{"gbps", 10}, {"latency_ms", 0.1}}},
                       {"8", {{"gbps", 5}, {"latency_ms", 0.1}}}}},
            {"sp_efficiency", 1}, {"resize_ms", 10}};
}
Json video(const std::string& id, double deadline, Json extra = Json::object()) {
    Json row = {{"request_id", id}, {"kind", "video"}, {"class", "video"}, {"arrival_ms", 0}, {"deadline_ms", deadline},
                {"status", "waiting"}, {"ready", true}};
    row.merge_patch(extra);
    return row;
}
Json image(const std::string& id, double deadline, double arrival = 0) {
    return {{"request_id", id}, {"kind", "image"}, {"class", "image"}, {"arrival_ms", arrival}, {"deadline_ms", deadline},
            {"status", "waiting"}, {"ready", true}};
}
Json plan(const Json& requests, const std::string& policy, int gpus, Json busy = Json::array(), Json options = Json::object()) {
    return cluster_plan({{"now_ms", 0}, {"gpus", gpus}, {"busy", busy}, {"requests", requests}}, policy, ClusterCost(profile()), options).at("assignments");
}
Json simulate(const Json& workload, const std::string& policy, int gpus, Json options = Json::object()) {
    return simulate_cluster({{"workload", workload}, {"profile", profile()}, {"policy", policy}, {"gpus", gpus}, {"options", options}});
}
Json trace_row(const std::string& id, const std::string& kind, double arrival, double deadline) {
    return {{"request_id", id}, {"kind", kind}, {"class", kind}, {"arrival_ms", arrival}, {"deadline_ms", deadline}};
}
}

int main() {
    check("aligned groups", [] {
        const auto groups = cluster_groups(8, {1, 2, 4, 8});
        require(groups.size() == 15, "8+4+2+1 groups");
        for (const auto& group : groups) require(group.front() % static_cast<int>(group.size()) == 0, "group is aligned");
        require(cluster_groups(6, {4}).size() == 1, "partial node keeps only whole groups");
        rejects([] { cluster_groups(8, {3}); });
        rejects([] { cluster_groups(0, {1}); });
    });
    check("cost model", [] {
        const ClusterCost cost(profile());
        near(cost.step_ms("video", 1), 1000);
        // compute/2 + 800 MB * (1/4) / 20 GB/s + 10 collectives * 0.1 ms
        near(cost.step_ms("video", 2), 500 + 10 + 1);
        near(cost.step_ms("video", 4), 250 + 800. * 3 / 16 / 10 + 1);
        near(cost.block_ms("video", 1), 100);
        near(cost.remaining_ms(video("v", 1e9, {{"steps_done", 1}, {"blocks_done", 5}, {"prepared", true}}), 1), 3 * 1000 - 500 + 100);
        Json bad = profile(); bad["provenance"] = "guess";
        rejects([&] { ClusterCost{bad}; });
        Json missing = profile(); missing["links"].erase("8");
        rejects([&] { ClusterCost(missing).step_ms("video", 8); });
        rejects([] { ClusterCost(profile()).step_ms("image", 1); });
    });
    check("host-bridge contention and per-quantum overhead", [] {
        Json shared = profile(); shared["links"]["2"]["concurrent_gbps"] = {{"2", 10}};
        const ClusterCost cost(shared);
        near(cost.step_ms("video", 2), 511);
        near(cost.step_ms("video", 2, 2), 500 + 800. / 4 / 10 + 1);
        near(cost.step_ms("video", 2, 3), 521);                       // largest listed count <= 3
        near(ClusterCost(profile()).step_ms("video", 2, 2), 511);      // no table: the isolated rate
        near(cost.step_ms("video", 4, 2), 250 + 800. * 3 / 16 / 10 + 1);
        for (const Json& bad : {Json{{"1", 10}}, Json{{"x", 10}}, Json{{"2", 0}}, Json::array({10})}) {
            Json broken = profile(); broken["links"]["2"]["concurrent_gbps"] = bad;
            rejects([&] { ClusterCost{broken}; });
        }
        Json negative = profile(); negative["quantum_overhead_ms"] = -1;
        rejects([&] { ClusterCost{negative}; });

        // Two static SP2 videos share the bridge for every step; a lone one does not.
        const Json pair = Json::array({trace_row("a", "video", 0, 1e9), trace_row("b", "video", 0, 1e9)});
        auto sim = [&](const Json& trace, const std::string& policy, int gpus, const Json& prof, Json options = Json::object()) {
            return simulate_cluster({{"workload", trace}, {"profile", prof}, {"policy", policy}, {"gpus", gpus}, {"options", options}});
        };
        near(sim(pair, "static_sp2", 4, shared).at("makespan_ms").get<double>(), 4 * 521. + 100);
        near(sim(Json::array({trace_row("a", "video", 0, 1e9)}), "static_sp2", 4, shared).at("makespan_ms").get<double>(), 4 * 511. + 100);
        // 4 steps of 10 blocks in 4-block quanta (4+4+2) plus one decode: 13 quanta.
        Json overhead = profile(); overhead["quantum_overhead_ms"] = 7;
        const auto one = sim(Json::array({trace_row("v", "video", 0, 1e9)}), "block_elastic", 1, overhead);
        near(one.at("makespan_ms").get<double>(), 4100 + 13 * 7.);
        require(one.at("requests")[0].at("quanta") == 13, "quanta counted");
        near(one.at("requests")[0].at("action_ms").at("blocks").get<double>(), 4000 + 12 * 7.);
        // The planner believes `planner_profile`; durations follow `profile`.
        const auto believed = simulate_cluster({{"workload", pair}, {"profile", profile()}, {"planner_profile", shared},
                                                {"policy", "static_sp2"}, {"gpus", 4}});
        near(believed.at("makespan_ms").get<double>(), 4 * 511. + 100);

        // Planner: "a" meets 2.16 s on an isolated pair (2.144 s) but not while another
        // SP group runs (2.184 s), so it takes the quad; a fixed overhead per quantum
        // (4 steps + 1 decode) has the same effect.
        const Json rows = {video("a", 2160), image("i", 1e9)};
        auto degree = [&](const Json& prof, int running) {
            return cluster_plan({{"now_ms", 0}, {"gpus", 4}, {"busy", Json::array()}, {"requests", rows}, {"sp_groups_running", running}},
                                "step_elastic", ClusterCost(prof)).at("assignments")[0].at("degree").get<int>();
        };
        require(degree(shared, 0) == 2 && degree(shared, 1) == 4, "contention widens the group");
        require(degree(profile(), 1) == 2, "no table: running groups change nothing");
        Json slow = profile(); slow["quantum_overhead_ms"] = 10;
        require(degree(slow, 0) == 4, "overhead counts every remaining quantum");
    });
    check("parallel forms", [] {
        require(form_name(parse_form("cfg2sp2")) == "cfg2sp2" && parse_form("cfg2").degree() == 2 && parse_form("sp8").sp == 8, "form names round-trip");
        require(form_name(Form{}) == "sp1" && form_name(Form{2, 1}) == "cfg2", "default form is sp1");
        for (const auto* bad : {"", "sp3", "cfg3", "cfg2sp3", "cfg", "sp", "sp16"}) rejects([bad] { parse_form(bad); });
        Json prof = profile(); prof["classes"]["video"]["cfg_exchange_mb"] = 20;
        const ClusterCost cost(prof);
        // Each half runs one branch: half the compute, one 20 MB exchange on the pair link plus its latency.
        near(cost.step_ms("video", parse_form("cfg2")), 500 + 1 + 0.1);
        // CFG2 x SP2: a quarter of the compute, half the all-to-all volume and collectives, plus the exchange.
        near(cost.step_ms("video", parse_form("cfg2sp2")), 250 + 400. / 4 / 20 + 5 * 0.1 + 1.1);
        near(cost.step_ms("video", parse_form("sp2")), cost.step_ms("video", 2));
        near(cost.remaining_ms(video("v", 1e9), parse_form("cfg2")), 4 * 501.1 + 100);
        rejects([&] { cost.step_ms("video", Form{3, 1}); });
        // Interleaved branches hide part of the shorter term; a CFG half has no partner branch.
        Json overlapped = prof; overlapped["sp_overlap"] = 1;
        near(ClusterCost(overlapped).step_ms("video", 2), 500 + 1);
        near(ClusterCost(overlapped).step_ms("video", 4), 250 + 1);
        near(ClusterCost(overlapped).step_ms("video", parse_form("cfg2sp2")), 250 + 5 + 0.5 + 1.1);
        overlapped["sp_overlap"] = 0.5;
        near(ClusterCost(overlapped).step_ms("video", 2), 500 + 5 + 1);
        overlapped["links"]["4"]["sp_overlap"] = 0.2;   // per-link value wins
        near(ClusterCost(overlapped).step_ms("video", 4), 250 + 15 - 3 + 1);
        near(ClusterCost(overlapped).step_ms("video", 2), 500 + 5 + 1);
        overlapped["sp_overlap"] = 2;
        rejects([&] { ClusterCost{overlapped}; });
        overlapped["sp_overlap"] = 0; overlapped["links"]["4"]["sp_overlap"] = 1.5;
        rejects([&] { ClusterCost(overlapped).step_ms("video", 4); });

        auto run = [&](const Json& requests, const std::string& policy, int gpus, const Json& options, const Json& costs = Json()) {
            return cluster_plan({{"now_ms", 0}, {"gpus", gpus}, {"busy", Json::array()}, {"requests", requests}}, policy,
                                ClusterCost(costs.is_null() ? profile() : costs), options).at("assignments");
        };
        // The planner takes the cheapest listed form of a group size: cfg2 (501.1 ms) over sp2 (511 ms).
        const Json lone = Json::array({video("v", 1e9)});
        const Json forms = {{"forms", {"sp1", "sp2", "cfg2"}}};
        auto chosen = run(lone, "step_elastic", 2, forms, prof);
        require(chosen.size() == 1 && chosen[0].at("form") == "cfg2" && chosen[0].at("degree") == 2, "cheapest form of the pair");
        Json dear = prof; dear["classes"]["video"]["cfg_exchange_mb"] = 400;   // 520.1 ms: SP2 wins
        require(run(lone, "step_elastic", 2, forms, dear)[0].at("form") == "sp2", "the exchange cost counts");
        require(run(lone, "step_elastic", 2, Json::object())[0].at("form") == "sp2", "default forms are plain SP");
        require(run(lone, "static_cfg2", 4, Json::object(), prof)[0].at("form") == "cfg2", "static CFG2 takes a pair");
        require(run(lone, "static_cfg2sp2", 4, Json::object(), prof)[0].at("degree") == 4, "static CFG2 x SP2 takes a quad");
        rejects([&] { run(lone, "static_cfg2", 1, Json::object(), prof); });
        rejects([&] { run(lone, "static_cfg3", 4, Json::object(), prof); });
        rejects([&] { run(lone, "step_elastic", 4, {{"forms", {"sp1"}}, {"degrees", {1}}}); });
        rejects([&] { run(lone, "step_elastic", 4, {{"forms", {"sp1", "cfgx"}}}); });
        // A request inside a step keeps its form; the state names it and it must fit the group.
        const auto mid = video("m", 1e9, {{"status", "suspended"}, {"prepared", true}, {"group", {0, 1}}, {"form", "cfg2"}, {"blocks_done", 4}});
        auto resumed = run(Json::array({mid}), "block_elastic", 2, forms, prof);
        require(resumed.size() == 1 && resumed[0].at("form") == "cfg2" && resumed[0].at("reason") == "resume", "form kept inside a step");
        rejects([&] { run(Json::array({video("x", 1e9, {{"status", "suspended"}, {"prepared", true}, {"group", {0}}, {"form", "cfg2"}, {"blocks_done", 4}})}),
                           "block_elastic", 2, Json::object(), prof); });
        // CFG pairs exchange one prediction per step and do not share the host bridge like SP groups.
        Json shared = prof; shared["links"]["2"]["concurrent_gbps"] = {{"2", 10}};
        const Json pair = Json::array({trace_row("a", "video", 0, 1e9), trace_row("b", "video", 0, 1e9)});
        const auto twin = simulate_cluster({{"workload", pair}, {"profile", shared}, {"policy", "static_cfg2"}, {"gpus", 4}});
        near(twin.at("makespan_ms").get<double>(), 4 * 501.1 + 100, 1e-6);
        require(twin.at("requests")[0].at("forms") == Json::array({"cfg2"}) && twin.at("requests")[0].at("degrees") == Json::array({2, 1}), "forms recorded");
        near(simulate_cluster({{"workload", pair}, {"profile", shared}, {"policy", "static_sp2"}, {"gpus", 4}}).at("makespan_ms").get<double>(), 4 * 521. + 100);
        // Elastic with forms: the lone video on 4 GPUs takes the cheapest quad form, CFG2 x SP2 (256.6 ms) over SP4 (266 ms).
        const auto quad = simulate_cluster({{"workload", Json::array({trace_row("v", "video", 0, 1e9)})}, {"profile", prof}, {"policy", "step_elastic"}, {"gpus", 4},
                                            {"options", {{"forms", {"sp1", "cfg2", "sp2", "cfg2sp2", "sp4"}}}}});
        require(quad.at("requests")[0].at("forms") == Json::array({"cfg2sp2"}), "elastic picks CFG2 x SP2");
        near(quad.at("makespan_ms").get<double>(), 4 * 256.6 + 100, 1e-6);
    });
    check("edf images first and videos share remaining GPUs", [] {
        const auto result = plan({video("v", 1e9), image("i", 100)}, "static_sp1", 2);
        require(result.size() == 2 && result[0].at("request_id") == "i" && result[0].at("quantum") == "atomic", "image first");
        require(result[1].at("quantum") == "step" && result[1].at("group").size() == 1, "video on one GPU");
        rejects([] { plan(Json::array({video("v", 1)}), "static_sp3", 4); });
    });
    check("elastic degree: work conserving alone, minimal feasible when contended", [] {
        auto alone = plan(Json::array({video("v", 1e9)}), "step_elastic", 8);
        require(alone.size() == 1 && alone[0].at("degree") == 8, "lone video takes the node");
        // 4 steps at SP1 take 4.1 s; deadline 3 s requires SP2 (2.14 s).
        auto shared = plan({video("a", 3000), video("b", 1e9)}, "step_elastic", 8);
        require(shared.size() == 2 && shared[0].at("degree") == 2, "minimal feasible degree for the urgent video");
        require(shared[1].at("degree") == 4, "the last request takes the largest remaining aligned group");
        auto busy = plan(Json::array({video("v", 1e9)}), "step_elastic", 4, Json::array({true, false, false, false}));
        require(busy.size() == 1 && busy[0].at("group") == Json::array({2, 3}), "aligned free pair when GPU 0 is busy");
    });
    check("a video at a step boundary does not return to a GPU another video holds mid-step", [] {
        // Both are late, so EDF puts "a" first; its old GPU 0 is free only between "b"'s quanta.
        const Json rows = {video("a", 100, {{"status", "suspended"}, {"prepared", true}, {"group", {0}}, {"steps_done", 1}}),
                           video("b", 200, {{"status", "suspended"}, {"prepared", true}, {"group", {0}}, {"blocks_done", 4}})};
        const auto result = plan(rows, "block_elastic", 1);
        require(result.size() == 1 && result[0].at("request_id") == "b" && result[0].at("reason") == "resume",
                "the mid-step owner resumes; the other video waits");
        auto suspended = [](const std::string& id, double deadline, Json group, Json extra) {
            Json row = video(id, deadline, {{"status", "suspended"}, {"prepared", true}, {"group", group}});
            row.merge_patch(extra);
            return row;
        };
        // "b" is inside a step on the pair; "a" last ran on GPU 1 alone.
        const auto pair = plan({suspended("a", 100, {1}, {{"steps_done", 1}}), suspended("b", 200, {0, 1}, {{"blocks_done", 4}})}, "block_elastic", 2);
        require(pair.size() == 1 && pair[0].at("request_id") == "b", "a GPU of a mid-step pair is not free for another video");
        // "a" last ran on the pair; "c" is inside a step on GPU 1, so "a" moves to GPU 0.
        const auto moved = plan({suspended("a", 100, {0, 1}, {{"steps_done", 1}}), suspended("c", 200, {1}, {{"blocks_done", 4}})}, "block_elastic", 2);
        require(moved.size() == 2 && moved[0].at("group") == Json::array({0}) && moved[0].at("resize") == true, "regroup around the mid-step GPU");
        // A chunked decode holds its GPU too, as a scheduling choice.
        Json chunked = profile(); chunked["classes"]["video"]["decode_chunks"] = 10;
        // "a" is planned first (earlier deadline, neither late).
        const Json decoding = {suspended("a", 1e5, {0}, {{"steps_done", 1}}), suspended("d", 1e9, {0}, {{"steps_done", 4}, {"decode_done", 3}})};
        const auto decode = cluster_plan({{"now_ms", 0}, {"gpus", 1}, {"busy", Json::array()}, {"requests", decoding}}, "block_elastic",
                                         ClusterCost(chunked), Json::object()).at("assignments");
        require(decode.size() == 1 && decode[0].at("quantum") == "finish", "no advance onto a decoding GPU");
    });
    check("the plan never advances onto another request's solver step", [] {
        // An inconsistent queue: two videos claim to be inside a step on GPU 0.
        const Json rows = {video("a", 100, {{"status", "suspended"}, {"prepared", true}, {"group", {0}}, {"blocks_done", 2}}),
                           video("b", 200, {{"status", "suspended"}, {"prepared", true}, {"group", {0}}, {"blocks_done", 4}})};
        bool contract = false;
        try { plan(rows, "block_elastic", 1); } catch (const std::logic_error&) { contract = true; }
        require(contract, "the post-condition names the conflict");
        require(plan(rows, "block_elastic", 1, Json::array(), {{"pin_started", true}}).size() == 1, "pinned workers may interleave");
        // K11 shape: 4 GPUs, SP1/2/4, ~300 ms quanta, overloaded mixed traffic with
        // tight video deadlines. Any interleaving the SP workers reject throws from the
        // post-condition; without the step-boundary resume check 5 of the 6 "free" runs do.
        for (const auto* feasibility : {"widest", "free"})
            for (int load = 2; load <= 3; ++load)
                for (int seed = 1; seed <= 3; ++seed) {
                    Json generator = {{"count", 120}, {"seed", seed}, {"pattern", "poisson"}, {"rate_rps", 3 * load},
                        {"classes", {{{"class", "video"}, {"kind", "video"}, {"weight", 1}, {"service_ms", 4000}, {"slo_scale", 1.25}},
                                     {{"class", "image"}, {"kind", "image"}, {"weight", 3}, {"service_ms", 50}, {"slo_ms", 400}}}}};
                    const auto result = simulate_cluster({{"workload", study_workload({{"generator", generator}})}, {"profile", profile()},
                        {"policy", "block_elastic"}, {"gpus", 4},
                        {"options", {{"degrees", {1, 2, 4}}, {"quantum_ms", 250}, {"feasibility_degree", feasibility}}}});
                    require(result.at("metrics").at("completed") == 120, "every request completes");
                }
    });
    check("lateness test degree", [] {
        // One free GPU. "a" needs SP4 (1.16 s) to meet 3 s, SP1 takes 4.1 s; "b" meets 5 s on SP1.
        const Json rows = {video("a", 3000), video("b", 5000)};
        const Json busy = {true, true, true, false};
        require(plan(rows, "step_elastic", 4, busy, {{"feasibility_degree", "widest"}})[0].at("request_id") == "a", "widest: SP4 feasibility keeps EDF order");
        for (const auto* policy : {"step_elastic", "block_elastic"})
            require(plan(rows, policy, 4, busy)[0].at("request_id") == "b", "elastic default is free: the request late on the free GPU yields");
        require(plan(rows, "step_elastic", 4, Json::array())[0].at("request_id") == "a", "free: an idle node still saves it");
        // "a" now meets 2 s on SP4 (1.16 s) but not on SP2 (2.14 s). A reserved image GPU or a
        // video inside a step on GPU 3 leaves only pairs free.
        const Json tight = {video("a", 2000), video("b", 5000)};
        require(plan(tight, "step_elastic", 4)[0].at("request_id") == "a", "an idle node saves the tight video");
        require(plan(tight, "step_elastic", 4, Json::array(), {{"reserve_image_gpus", 1}})[0].at("request_id") == "b", "reserved GPUs are not free");
        Json held = tight;
        held.push_back(video("m", 1e9, {{"status", "suspended"}, {"prepared", true}, {"group", {3}}, {"blocks_done", 2}}));
        require(plan(held, "block_elastic", 4)[0].at("request_id") == "b", "GPUs held inside a step are not free");
        // Static policies test lateness at their fixed degree whatever the option says.
        const Json one_busy = {true, false, false, false};
        const auto fixed = plan(rows, "static_sp2", 4, one_busy, {{"feasibility_degree", "free"}});
        require(fixed.size() == 1 && fixed == plan(rows, "static_sp2", 4, one_busy, {{"feasibility_degree", "widest"}}),
                "static ignores feasibility_degree");
        rejects([] { plan(Json::array({video("v", 1)}), "step_elastic", 4, Json::array(), {{"feasibility_degree", "any"}}); });
    });
    check("lazy block preemption", [] {
        // Mid-step video on GPU 0: 6 blocks (600 ms) to the step boundary.
        const auto mid = video("v", 1e9, {{"status", "suspended"}, {"prepared", true}, {"group", {0}}, {"blocks_done", 4}, {"steps_done", 1}});
        auto patient = plan({image("i", 2000), mid}, "block_elastic", 1);
        require(patient.size() == 1 && patient[0].at("request_id") == "v" && patient[0].at("quantum") == 4, "image waits for the boundary");
        auto urgent = plan({image("i", 300), mid}, "block_elastic", 1);
        require(urgent.size() == 1 && urgent[0].at("reason") == "block_preempt", "urgent image preempts inside the step");
        auto eager = plan({image("i", 2000), mid}, "block_elastic", 1, Json::array(), {{"lazy_preemption", false}});
        require(eager.size() == 1 && eager[0].at("request_id") == "i", "eager preemption takes the GPU");
        // 250 ms target at 100 ms per SP1 block: two blocks per quantum.
        auto timed = plan(Json::array({mid}), "block_elastic", 1, Json::array(), {{"quantum_ms", 250}});
        require(timed.size() == 1 && timed[0].at("quantum") == 2, "time-targeted quantum");
    });
    check("resident groups, finish and pinning", [] {
        const auto suspended = video("v", 1e9, {{"status", "suspended"}, {"prepared", true}, {"group", {2, 3}}, {"steps_done", 1}});
        require(plan(Json::array({suspended}), "static_sp2", 4, Json::array({false, false, false, true})).empty(), "static waits for its own group");
        auto moved = plan(Json::array({suspended}), "step_elastic", 4, Json::array({false, false, false, true}));
        require(moved.size() == 1 && moved[0].at("resize") == true, "elastic regroups at a step boundary");
        require(plan(Json::array({suspended}), "step_elastic", 4, Json::array({false, false, false, true}), {{"pin_started", true}}).empty(), "pinned");
        const auto done = video("v", 1e9, {{"status", "suspended"}, {"prepared", true}, {"group", {2, 3}}, {"steps_done", 4}});
        auto finish = plan(Json::array({done}), "block_elastic", 4);
        require(finish.size() == 1 && finish[0].at("quantum") == "finish" && finish[0].at("group") == Json::array({2}), "decode on the first rank");
    });
    check("minimum SP degree for classes that do not fit one GPU", [] {
        Json big = profile(); big["classes"]["video"]["min_degree"] = 2;
        const ClusterCost cost(big);
        auto run = [&](const std::string& policy, int gpus) {
            return cluster_plan({{"now_ms", 0}, {"gpus", gpus}, {"busy", Json::array()}, {"requests", Json::array({video("v", 1e9)})}},
                                policy, cost).at("assignments");
        };
        require(run("static_sp1", 4)[0].at("degree") == 2, "static SP1 raised to the class minimum");
        require(run("static_sp4", 4)[0].at("degree") == 4, "larger fixed degree kept");
        rejects([&] { run("step_elastic", 1); });
        Json odd = profile(); odd["classes"]["video"]["min_degree"] = 3;
        rejects([&] { ClusterCost{odd}; });
    });
    check("GPUs reserved for images", [] {
        auto result = plan(Json::array({video("v", 1e9), image("i", 100)}), "step_elastic", 4, Json::array(), {{"reserve_image_gpus", 2}});
        require(result.size() == 2 && result[0].at("group") == Json::array({2}), "image on a reserved GPU");
        require(result[1].at("group") == Json::array({0, 1}), "video limited to unreserved GPUs");
        rejects([] { plan(Json::array({video("v", 1e9)}), "step_elastic", 2, Json::array(), {{"reserve_image_gpus", 2}}); });
    });
    check("chunked decode is preemptible between latent frames", [] {
        Json chunked = profile(); chunked["classes"]["video"]["decode_chunks"] = 10;
        const ClusterCost cost(chunked);
        near(cost.chunk_ms("video"), 10);
        const auto decoding = video("v", 1e9, {{"status", "suspended"}, {"prepared", true}, {"group", {0}}, {"steps_done", 4}, {"decode_done", 3}});
        near(cost.remaining_ms(decoding, 1), 70);
        auto run = [&](const Json& rows, const Json& options) {
            return cluster_plan({{"now_ms", 0}, {"gpus", 1}, {"busy", Json::array()}, {"requests", rows}}, "block_elastic", cost, options).at("assignments");
        };
        auto next = run(Json::array({decoding}), {{"quantum_ms", 25}});
        require(next.size() == 1 && next[0].at("quantum") == "finish" && next[0].at("decode_frames") == 2, "two 10 ms chunks per 25 ms quantum");
        // 70 ms of decode remain: a patient image waits, an urgent one preempts.
        require(run(Json::array({image("i", 200), decoding}), {{"quantum_ms", 25}})[0].at("request_id") == "v", "patient image waits");
        require(run(Json::array({image("i", 90), decoding}), {{"quantum_ms", 25}})[0].at("reason") == "block_preempt", "urgent image preempts decode");
        // Steps end at 4000 ms; the image arrives 20 ms into the 100 ms decode.
        const Json trace = Json::array({trace_row("v", "video", 0, 1e9), trace_row("i", "image", 4020, 4100)});
        const auto atomic = simulate_cluster({{"workload", trace}, {"profile", chunked}, {"policy", "block_elastic"}, {"gpus", 1}, {"options", {{"quantum_ms", 25}, {"chunked_decode", false}}}});
        const auto split = simulate_cluster({{"workload", trace}, {"profile", chunked}, {"policy", "block_elastic"}, {"gpus", 1}, {"options", {{"quantum_ms", 25}}}});
        require(atomic.at("by_kind").at("image").at("ontime") == 0 && split.at("by_kind").at("image").at("ontime") == 1, "decode no longer blocks the image");
        near(split.at("makespan_ms").get<double>(), 4000 + 100 + 50, 1e-6);
    });
    check("late short requests keep their EDF position", [] {
        // An image that cannot meet its deadline ahead of a feasible video.
        const Json rows = Json::array({image("i", 10), video("v", 1e9)});  // 50 ms image, 10 ms deadline: late
        require(plan(rows, "static_sp1", 1)[0].at("request_id") == "i", "short late image is not demoted");
        require(plan(rows, "static_sp1", 1, Json::array(), {{"late_short_ms", 0}})[0].at("request_id") == "v", "strict demotion");
        require(plan(rows, "static_sp1", 1, Json::array(), {{"demote_late", false}})[0].at("request_id") == "i", "plain EDF");
    });
    check("simulation accounting", [] {
        Json images = Json::array();
        for (int i = 0; i < 4; ++i) images.push_back(trace_row("i" + std::to_string(i), "image", 0, 1000));
        const auto result = simulate(images, "static_sp1", 1);
        near(result.at("makespan_ms").get<double>(), 200);
        require(result.at("metrics").at("completed") == 4 && result.at("by_kind").at("image").at("ontime") == 4, "all complete");
        near(result.at("gpu_utilization").get<double>(), 1);
        require(simulate(images, "static_sp1", 1) == result, "deterministic");
    });
    check("block preemption shortens image latency on one GPU", [] {
        Json mixed = Json::array({trace_row("v", "video", 0, 1e9)});
        for (int i = 0; i < 6; ++i) mixed.push_back(trace_row("i" + std::to_string(i), "image", 250. + 700 * i, 250. + 700 * i + 200));
        const auto step = simulate(mixed, "static_sp1", 1), block = simulate(mixed, "block_elastic", 1, {{"lazy_preemption", false}});
        require(block.at("by_kind").at("image").at("ontime") > step.at("by_kind").at("image").at("ontime"), "more on-time images");
        require(block.at("block_preemptions").get<int>() > 0 && step.at("block_preemptions") == 0, "preemptions counted");
        require(block.at("metrics").at("completed") == 7 && step.at("metrics").at("completed") == 7, "all complete");
    });
    check("elastic SP finishes a lone video sooner", [] {
        const Json trace = Json::array({trace_row("v", "video", 0, 1e9)});
        const auto one = simulate(trace, "static_sp1", 4), elastic = simulate(trace, "step_elastic", 4);
        require(elastic.at("makespan_ms").get<double>() < one.at("makespan_ms").get<double>() / 2, "SP4 speedup");
        require(elastic.at("requests")[0].at("degrees") == Json::array({4, 1}), "SP4 steps, single-GPU decode");
    });
    check("generator classes", [] {
        Json generator = {{"count", 200}, {"seed", 3}, {"pattern", "poisson"}, {"rate_rps", 5},
            {"classes", {{{"class", "video"}, {"kind", "video"}, {"weight", 1}, {"service_ms", 4000}, {"slo_scale", 3}},
                         {{"class", "image"}, {"kind", "image"}, {"weight", 3}, {"service_ms", 50}, {"slo_ms", 400}}}}};
        const auto rows = study_workload({{"generator", generator}});
        int videos = 0;
        for (const auto& row : rows) {
            require(row.at("class") == row.at("kind"), "class recorded");
            if (row.at("kind") == "video") { ++videos; near(row.at("slo_ms").get<double>(), 12000); }
            else near(row.at("slo_ms").get<double>(), 400);
        }
        require(videos > 30 && videos < 70, "weights respected");
        require(study_workload({{"generator", generator}}) == rows, "deterministic");
        simulate_cluster({{"workload", rows}, {"profile", profile()}, {"policy", "block_elastic"}, {"gpus", 8}});
    });
    std::cout << checks - failures << "/" << checks << " cluster checks passed\n";
    return failures ? 1 : 0;
}
