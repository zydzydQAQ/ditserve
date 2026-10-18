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
