#include "ditserve/benchmark/cluster_sim.hpp"
#include "ditserve/control/cluster.hpp"
#include "ditserve/control/policy.hpp"
#include "ditserve/control/dispatch.hpp"
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <set>
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
    check("20261103 fixed-form resume waits for an interrupted solver-step owner", [] {
        // Reduced from the failed 12b0918 P0 contention cell: v0012 completed
        // one step, then feasible-first EDF admitted v0013 on the same CFG pair.
        // Its ninth quantum stopped early for an image; afterward both videos
        // were late and the earlier deadline selected v0012 again. The raw run
        // records no final block counter, so cover every possible stopped block.
        Json prof = profile();
        auto& klass = prof["classes"]["video"];
        klass["steps"] = 20; klass["blocks_per_step"] = 60;
        klass["step_compute_ms"] = 1158.2064293324947;
        klass["finish_ms"] = 1867.474514991045;
        klass["forms"]["cfg2"]["placements"]["0-1"]["step_ms"] = 582.8467737883329;
        prof["quantum_overhead_ms"] = 1.2567808018311073;
        const ClusterCost cost(prof);
        Json earlier = video("w0-v0012", 533208.9355873662,
            {{"arrival_ms", 473466.1357828203}, {"status", "suspended"}, {"prepared", true},
             {"steps_done", 1}, {"blocks_done", 0}, {"group", {0, 1}}, {"form", "cfg2"}});
        Json owner = video("w0-v0013", 534618.0826134813, {{"arrival_ms", 474875.2828089354}});
        const Json options = {{"forms", {"cfg2"}}, {"quantum_blocks", 4}, {"doorbell", true}, {"lazy_preemption", false}};
        auto dispatch = [&](double now) {
            return cluster_plan({{"now_ms", now}, {"gpus", 2}, {"busy", {false, false}},
                                 {"requests", {earlier, owner}}}, "block-cfg2", cost, options).at("assignments");
        };
        const auto admitted = dispatch(520274.0100566298);
        require(admitted.size() == 1 && admitted[0].at("request_id") == "w0-v0013",
                "the measured lateness crossover admits the later feasible video");
        owner.merge_patch({{"status", "suspended"}, {"prepared", true}, {"steps_done", 8},
                           {"group", {0, 1}}, {"form", "cfg2"}});
        for (int block = 1; block < 60; ++block) {
            owner["blocks_done"] = block;
            const auto resumed = dispatch(527733.4191929549);
            require(resumed.size() == 1 && resumed[0].at("request_id") == "w0-v0013"
                    && resumed[0].at("group") == Json::array({0, 1}) && !resumed[0].at("resize").get<bool>(),
                    "a fixed placement is not permission to enter another video's unfinished step");
        }
        owner["blocks_done"] = 0; owner["steps_done"] = 9;
        const auto boundary = dispatch(527733.4191929549);
        require(boundary.size() == 1 && boundary[0].at("request_id") == "w0-v0012",
                "ordinary EDF resumes once the worker is back at a solver-step boundary");
    });
    check("20261103 fixed and round-held boundary resumes honor solver ownership", [] {
        const Json rows = {
            video("boundary", 100, {{"status", "suspended"}, {"prepared", true}, {"group", {0, 1}},
                                    {"form", "cfg2"}, {"steps_done", 1}}),
            video("owner", 200, {{"status", "suspended"}, {"prepared", true}, {"group", {0, 1}},
                                 {"form", "cfg2"}, {"blocks_done", 4}})};
        for (const auto* policy : {"block-cfg2", "edf-cfg2", "tetriserve"}) {
            const auto result = plan(rows, policy, 2, Json::array(), {{"forms", {"cfg2"}}, {"round_steps", 5}});
            require(result.size() == 1 && result[0].at("request_id") == "owner",
                    std::string(policy) + " must defer a fixed boundary resume to the unfinished step");
        }
        Json partial = rows; partial[0]["group"] = {1}; partial[0]["form"] = "sp1";
        const auto overlap = plan(partial, "tetriserve", 2, Json::array(), {{"forms", {"sp1", "cfg2"}}, {"round_steps", 5}});
        require(overlap.size() == 1 && overlap[0].at("request_id") == "owner",
                "a round-held single GPU must not split a suspended CFG pair");
        Json demand = rows; demand.push_back(image("urgent-image", 80));
        const auto preempt = plan(demand, "block-cfg2", 2, Json::array(), {{"lazy_preemption", false}});
        require(preempt.size() == 1 && preempt[0].at("request_id") == "urgent-image"
                && preempt[0].at("reason") == "block_preempt",
                "the solver hold still permits an image on its separate worker after the video stops");
    });
    check("20261103 deadline doorbells select only a group that saves its reserved image", [] {
        const std::vector<PreemptionGroup> heterogeneous = {
            {"fast-expensive", {0}, 10, 100, 1000, false},
            {"slow-cheap", {1}, 30, 200, 0, false}};
        auto selected = select_deadline_preemption_groups(0, {{-1, 60, 40}}, heterogeneous);
        require(selected.selected == std::vector<std::string>{"fast-expensive"},
                "a cheap group cannot borrow another group's faster stop estimate");
        require(select_deadline_preemption_groups(0, {{1, 60, 40}}, heterogeneous).selected.empty(),
                "an image pinned to the slow group cannot use the fast group's timing");
        const auto patient = select_deadline_preemption_groups(0, {{-1, 60, 40}}, {
            {"natural-boundary", {0}, 5, 5, 0, false}, {"interrupt", {1}, 10, 100, 0, false}});
        require(patient.selected.empty(), "an available natural boundary avoids an unnecessary interruption");
        const std::vector<PreemptionGroup> pending = {
            {"already-rung", {0, 1}, 5, 100, 0, true}, {"extra", {2}, 5, 100, 0, false}};
        require(select_deadline_preemption_groups(0, {{-1, 50, 40}, {-1, 50, 40}}, pending).selected.empty(),
                "already-rung capacity covers two waiting images without another interruption");
        require(select_deadline_preemption_groups(0, {{-1, 50, 40}, {-1, 50, 40}, {-1, 50, 40}}, pending).selected
                    == std::vector<std::string>{"extra"}, "already-rung capacity is not reused for a third image");
        const auto deadlines = select_deadline_preemption_groups(0, {{-1, 70, 40}, {-1, 50, 40}}, {
            {"slow", {0}, 20, 100, 0, false}, {"fast", {1}, 5, 100, 100, false}});
        require(deadlines.selected == std::vector<std::string>{"fast", "slow"},
                "each differently constrained image receives its own feasible stop capacity");
        require(select_deadline_preemption_groups(0, {{-1, 30, 40}}, heterogeneous).deadline_skips == 1,
                "unsalvageable demand remains a recorded skip");
    });
    check("20261103 remaining-work widening cannot use a single-step pair exemption", [] {
        Json prof = profile(); prof["resize_ms"] = 10000; prof["pair_overlap"] = 1;
        const Json finishing = video("a", 1100, {{"status", "suspended"}, {"prepared", true},
            {"steps_done", 3}, {"group", {0}}, {"form", "sp1"}});
        const Json state = {{"now_ms", 0}, {"gpus", 2}, {"busy", Json::array()},
                            {"requests", {finishing, video("b", 1e9)}}};
        Json options = {{"forms", {"sp1", "sp2"}}, {"pairing", true}, {"widen_efficiency", 0.99}};
        const auto safe = cluster_plan(state, "step_elastic", ClusterCost(prof), options).at("assignments");
        require(safe[0].at("form") == "sp1" && !safe[0].at("resize").get<bool>(),
                "pair speculation cannot make a feasible last step late through regrouping");
        options["widen_remaining_work"] = false;
        const auto legacy = cluster_plan(state, "step_elastic", ClusterCost(prof), options).at("assignments");
        require(legacy[0].at("form") == "sp2" && legacy[0].at("resize").get<bool>(),
                "the explicitly requested single-step pair ablation remains available");
    });
    check("20261103 fission respects the model memory minimum while permitting CFG2 to SP1", [] {
        Json request = video("fold", 1e9, {{"status", "suspended"}, {"prepared", true},
            {"steps_done", 2}, {"group", {0, 1}}, {"form", "cfg2"}, {"cfg_fold_eligible", true}});
        const Json state = {{"now_ms", 0}, {"gpus", 2}, {"busy", Json::array()},
                            {"requests", {request, image("image", 2e9)}}};
        Json prof = profile(); prof["classes"]["video"]["min_degree"] = 2;
        auto held = cluster_plan(state, "ditserve-approx", ClusterCost(prof), {{"forms", {"cfg2"}}}).at("assignments");
        require(held[0].at("degree") == 2 && !held[0].value("fission", true),
                "folding does not establish that a memory-constrained model fits half as many GPUs");
        prof["classes"]["video"]["min_degree"] = 1;
        auto released = cluster_plan(state, "ditserve-approx", ClusterCost(prof), {{"forms", {"cfg2"}}}).at("assignments");
        require(released[0].at("form") == "sp1" && released[0].value("fission", false) && released.size() == 2,
                "a verified single-GPU class may release its CFG companion to the waiting image");
    });
    check("20261102 mixed resolutions retain independent communication fits", [] {
        Json measured = profile();
        measured["classes"]["large"] = measured["classes"]["video"];
        measured["classes"]["large"]["links"] = {{"2", {{"gbps", 10}, {"latency_ms", 0.1}}}};
        const ClusterCost cost(measured);
        near(cost.step_ms("video",parse_form("sp2")),511);
        near(cost.step_ms("large",parse_form("sp2")),521);
        rejects([&]{ cost.step_ms("large",parse_form("sp4")); });
        measured["classes"]["large"]["links"]["groups"]["0-1"]={{"gbps",5}};
        const std::vector<int> pair{0,1};
        near(ClusterCost(measured).step_ms("large",parse_form("sp2"),1,&pair),541);
        require(!ClusterCost(measured).placement_invariant("large"),"class topology must disable global pruning");
        require(ClusterCost(measured).placement_invariant("video"),"unrelated class keeps its own topology");
        measured["classes"]["large"]["links"]["2"]["gbps"]=-1;
        rejects([&]{ClusterCost invalid(measured);});
        measured["classes"]["large"]["links"]=Json::array();
        rejects([&]{ClusterCost invalid(measured);});
    });
    check("measured forms bind to placements and keep homogeneous compatibility", [] {
        Json measured = profile();
        measured["classes"]["video"]["forms"]["sp2"] = {{"step_ms", 123}};
        const std::vector<int> first{0,1}, second{2,3};
        near(ClusterCost(measured).step_ms("video",parse_form("sp2"),1,&second),123);
        measured["classes"]["video"]["forms"]["sp2"]["group"] = first;
        const ClusterCost bound(measured);
        near(bound.step_ms("video",parse_form("sp2"),1,&first),123);
        near(bound.step_ms("video",parse_form("sp2"),1,&second),511);
        near(bound.step_ms("video",parse_form("sp2")),511);
        require(!bound.placement_invariant("video"),"bound measurement cannot justify global pruning");
        measured["classes"]["video"]["forms"]["sp2"]["placements"] = {{"2-3",{{"step_ms",234}}}};
        near(ClusterCost(measured).step_ms("video",parse_form("sp2"),1,&second),234);
        near(ClusterCost(measured).step_ms("video",parse_form("sp2"),2,&second),511);
        auto placements_only=measured;
        placements_only["classes"]["video"]["forms"]["sp2"].erase("group");
        placements_only["classes"]["video"]["forms"]["sp2"].erase("step_ms");
        near(ClusterCost(placements_only).step_ms("video",parse_form("sp2"),1,&second),234);
        near(ClusterCost(placements_only).step_ms("video",parse_form("sp2"),1,&first),511);
        measured["classes"]["video"]["forms"]["sp2"].erase("group");
        measured["classes"]["video"]["forms"]["sp2"].erase("placements");
        measured["links"]["groups"]["2-3"] = {{"gbps",10}};
        near(ClusterCost(measured).step_ms("video",parse_form("sp2"),1,&second),521);
        measured["classes"]["video"]["forms"]["sp2"]["placements"] = {{"0-2",{{"step_ms",100}}}};
        rejects([&]{ClusterCost invalid(measured);});
        measured["classes"]["video"]["forms"]["sp2"]["placements"] = {{"0-1",{{"step_ms",0}}}};
        rejects([&]{ClusterCost invalid(measured);});
    });
    check("20261102 doorbell timing follows rank-local worker checkpoints", [] {
        // A flag raised immediately after a poll is seen at the next local
        // checkpoint, and one checkpoint later by Wan's pipelined group poll.
        const auto local=estimate_preemption_timing(10,60,0,0,1,1,false);
        const auto grouped=estimate_preemption_timing(10,30,0,0,1,1,true);
        near(local.stop_ms,10);near(grouped.stop_ms,20);
        const std::vector<PreemptionGroup> quantum_one={
            {"cfg2",{0,1},grouped.stop_ms,grouped.remaining_ms,0,false},
            {"sp1",{2},30,600,0,false}};
        require(select_preemption_groups({-1},quantum_one)==std::vector<std::string>{"sp1"},
                "quantum=1 must price the second group checkpoint before choosing a victim");
        const auto sequential=estimate_preemption_timing(10,60,0,0,2,1,true);
        const auto interleaved=estimate_preemption_timing(8,60,0,0,2,2,true);
        near(sequential.stop_ms,40);near(interleaved.stop_ms,64);
        const std::vector<PreemptionGroup> branches={
            {"interleaved",{0,1},interleaved.stop_ms,interleaved.remaining_ms,0,false},
            {"sequential",{2,3},sequential.stop_ms,sequential.remaining_ms,0,false}};
        require(select_preemption_groups({-1},branches)==std::vector<std::string>{"sequential"},
                "two interleaved branch blocks per iteration must not look cheaper than one");
        // A folded or CFG-parallel rank has 30 units; 28 units are already
        // complete. Keep the same per-block price when shortening the step.
        const auto finishing=estimate_preemption_timing(10,30,28,5,2,1,true);
        near(finishing.remaining_ms,15);
        near(estimate_preemption_timing(10,60,28,5,2,1,true).remaining_ms,315);
        near(estimate_preemption_timing(10,30,28,100,2,1,true).remaining_ms,10);
        rejects([]{estimate_preemption_timing(10,30,0,0,0,1,true);});
        require(preemption_converts_deadline(0,90,40,20,60),"urgent image becomes on time");
        require(!preemption_converts_deadline(0,100,40,20,30),"patient image waits for the step boundary");
        require(!preemption_converts_deadline(0,100,100,20,60),"unsalvageable image does not interrupt video");
        require(!preemption_converts_deadline(0,100,40,60,60),"stopping no sooner than the boundary has no benefit");
        rejects([]{preemption_converts_deadline(0,100,-1,20,60);});
    });
    check("selective doorbells cover demand without repeated global interruption", [] {
        const std::vector<PreemptionGroup> pairs = {
            {"slow",{0,1},20,100,1,false}, {"fast",{2,3},2,100,1,false},
            {"finishing",{4,5},20,1,0,false}, {"last",{6,7},10,100,1,false}};
        require(select_preemption_groups({},pairs).empty(),"idle queue rang videos");
        require(select_preemption_groups({-1},pairs)==std::vector<std::string>{"finishing"},"one image must stop only the cheapest group");
        require(select_preemption_groups({-1,-1,-1},pairs)==std::vector<std::string>({"fast","finishing"}),"three images need two pairs");
        auto rung=pairs;rung[0].rung=true;
        require(select_preemption_groups({-1,-1},rung).empty(),"already-rung capacity must be reserved");
        require(select_preemption_groups({-1,-1,-1},rung)==std::vector<std::string>{"finishing"},"new demand needs only incremental capacity");
        require(select_preemption_groups({0},pairs)==std::vector<std::string>{"slow"},"resident image must reclaim its own GPU");
        require(select_preemption_groups({0,0},pairs)==std::vector<std::string>{"slow"},"same-GPU images must not interrupt unrelated groups");
        require(select_preemption_groups({99},pairs).empty(),"unavailable resident worker cannot use other video GPUs");
        require(select_preemption_groups(std::vector<int>(10,-1),pairs).size()==4,"saturated queue must release all useful capacity");
        auto overlap=pairs;overlap[1].group={1,2};
        rejects([&]{select_preemption_groups({-1},overlap);});
    });
    check("placement-dependent forms are not globally pruned", [] {
        Json measured=profile();
        measured["classes"]["video"]["forms"] = {
            {"cfg2",{{"step_ms",500}}}, {"sp2",{{"step_ms",700},{"placements",{{"2-3",{{"step_ms",100}}}}}}}};
        const auto result=cluster_plan({{"now_ms",0},{"gpus",4},{"busy",{true,true,false,false}},
            {"requests",{video("v",1e9)}}},"ditserve",ClusterCost(measured),{{"forms",{"cfg2","sp2"}}});
        require(result.at("assignments")[0].at("form")=="sp2","globally slower form must remain usable on its fast placement");
        measured["classes"]["video"]["forms"]["sp2"].erase("placements");
        require(ClusterCost(measured).placement_invariant("video"),"homogeneous legacy profile remains prunable");
        measured["links"]["2"]["concurrent_gbps"]={{"2",1}};
        require(!ClusterCost(measured).placement_invariant("video"),"isolated dominance does not prove contended dominance");
    });
    check("aligned groups", [] {
        const auto groups = cluster_groups(8, {1, 2, 3, 4, 6, 8});
        require(groups.size() == 18, "8+4+2+2+1+1 groups");
        for (const auto& group : groups) require(group.front() % static_cast<int>(group.size()) == 0, "group is aligned");
        require(cluster_groups(6, {4}).size() == 1, "partial node keeps only whole groups");
        require(cluster_groups(8, {6})[0] == std::vector<int>({0, 1, 2, 3, 4, 5}), "six-GPU video group");
        rejects([] { cluster_groups(8, {5}); });
        rejects([] { cluster_groups(0, {1}); });
    });
    check("image placement keeps aligned pairs whole", [] {
        // GPU 3 busy, 0-2 free: "first" breaks the free pair (0,1); "pair_fit" takes 2, whose partner is busy.
        const Json busy = {false, false, false, true};
        const auto first = plan(Json::array({image("i", 1e9)}), "block_elastic", 4, busy);
        require(first.size() == 1 && first[0].at("group") == Json::array({0}), "first takes the lowest free GPU");
        const auto fitted = plan(Json::array({image("i", 1e9)}), "block_elastic", 4, busy, {{"image_placement", "pair_fit"}});
        require(fitted.size() == 1 && fitted[0].at("group") == Json::array({2}), "pair_fit keeps the free pair (0,1)");
        // Nothing busy: every pair is free, the tie stays with the lowest index.
        const auto idle = plan(Json::array({image("i", 1e9)}), "block_elastic", 4, Json::array(), {{"image_placement", "pair_fit"}});
        require(idle[0].at("group") == Json::array({0}), "ties keep the lowest index");
        rejects([] { plan(Json::array({image("i", 1e9)}), "block_elastic", 4, Json::array(), {{"image_placement", "best"}}); });
    });
    check("per-block overhead", [] {
        Json slow = profile(); slow["block_overhead_ms"] = 2;
        const ClusterCost base(profile()), extra(slow);
        near(base.block_overhead_ms(), 0);
        near(extra.step_ms("video", 1) - base.step_ms("video", 1), 20);                    // 10 blocks x 2 ms
        near(extra.step_ms("video", 4) - base.step_ms("video", 4), 20);                    // every SP rank runs all blocks
        near(extra.step_ms("video", Form{2, 1}) - base.step_ms("video", Form{2, 1}), 10);  // a CFG rank runs half
        near(extra.block_ms("video", 1) - base.block_ms("video", 1), 2);
        near(extra.pair_ms("video", "video", Form{1, 2}) - base.pair_ms("video", "video", Form{1, 2}), 40);
        Json negative = profile(); negative["block_overhead_ms"] = -1;
        rejects([&] { ClusterCost{negative}; });
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
        require(parse_form("sp3").degree() == 3 && parse_form("cfg2sp3").degree() == 6, "three-head-group forms");
        require(form_name(parse_form("cfg2u3r1")) == "cfg2sp3", "Ulysses alias canonicalizes without Ring");
        for (const auto* bad : {"", "sp5", "cfg3", "cfg2sp5", "cfg", "sp", "sp16"}) rejects([bad] { parse_form(bad); });
        Json prof = profile(); prof["classes"]["video"]["cfg_exchange_mb"] = 20;
        const ClusterCost cost(prof);
        // Each half runs one branch: half the compute, one 20 MB exchange on the pair link plus its latency.
        near(cost.step_ms("video", parse_form("cfg2")), 500 + 1 + 0.1);
        // CFG2 x SP2: a quarter of the compute, half the all-to-all volume and collectives, plus the exchange.
        near(cost.step_ms("video", parse_form("cfg2sp2")), 250 + 400. / 4 / 20 + 5 * 0.1 + 1.1);
        near(cost.step_ms("video", parse_form("sp2")), cost.step_ms("video", 2));
        near(cost.remaining_ms(video("v", 1e9), parse_form("cfg2")), 4 * 501.1 + 100);
        rejects([&] { cost.step_ms("video", Form{3, 1}); });
        Json measured = prof;
        measured["classes"]["video"]["forms"]["cfg2sp3"]["step_ms"] = 123;
        measured["links"]["3"] = {{"gbps", 15}, {"latency_ms", 0.1}};
        const ClusterCost measured_cost(measured);
        near(measured_cost.step_ms("video", parse_form("cfg2sp3")), 123);
        near(measured_cost.block_ms("video", parse_form("cfg2sp3")), 12.3);
        require(measured_cost.step_ms("video", parse_form("cfg2sp3"), 2) != 123, "concurrent groups use analytical fallback");
        require(measured_cost.step_ms("video", parse_form("cfg2sp3"), 1, nullptr, 0.16) != 123, "cache uses analytical fallback");
        Json bad_measured = prof; bad_measured["classes"]["video"]["forms"]["sp1"]["step_ms"] = -1;
        rejects([&] { ClusterCost{bad_measured}; });
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
        // Work-neutral widening: two waiting videos on 4 GPUs run SP1 by default (minimal feasible);
        // with widen_efficiency 0.97 each takes a CFG pair (efficiency 0.998; SP2 0.978 would pass too,
        // SP4 0.94 would not), with 0.999 nothing qualifies.
        const Json two = Json::array({video("a", 1e9), video("b", 1e9)});
        require(run(two, "step_elastic", 4, forms, prof)[0].at("degree") == 1, "minimal feasible by default");
        auto widened = run(two, "step_elastic", 4, {{"forms", {"sp1", "sp2", "cfg2", "sp4"}}, {"widen_efficiency", 0.97}}, prof);
        require(widened.size() == 2 && widened[0].at("form") == "cfg2" && widened[1].at("form") == "cfg2", "widened to CFG pairs");
        require(run(two, "step_elastic", 4, {{"forms", {"sp1", "sp2", "cfg2", "sp4"}}, {"widen_efficiency", 0.999}}, prof)[0].at("degree") == 1, "threshold respected");
        Json costly_resize = prof; costly_resize["resize_ms"] = 10000;
        const Json finishing = video("a", 1e9, {{"status", "suspended"}, {"prepared", true}, {"steps_done", 3},
                                               {"group", {0}}, {"form", "sp1"}});
        const auto held = run(Json::array({finishing, video("b", 1e9)}), "step_elastic", 2,
                              {{"forms", {"sp1", "cfg2"}}, {"widen_efficiency", 0.97}}, costly_resize);
        require(held[0].at("form") == "sp1" && !held[0].at("resize").get<bool>(),
                "remaining-work pricing refuses an unamortized final-step regroup");
        const auto legacy = run(Json::array({finishing, video("b", 1e9)}), "step_elastic", 2,
                                {{"forms", {"sp1", "cfg2"}}, {"widen_efficiency", 0.97}, {"widen_remaining_work", false}}, costly_resize);
        require(legacy[0].at("form") == "cfg2" && legacy[0].at("resize").get<bool>(),
                "single-step ablation retains the negative regroup for comparison");
        rejects([&] { run(two, "step_elastic", 4, {{"widen_efficiency", 1.5}}); });
        // prefer_exact: the CFG pair (501.1 ms, output-preserving) wins over SP2 at 511 ms either way;
        // with a dear exchange (520.1 ms) SP2 is 1.7% faster, which a 5% preference forgoes and a 1% one takes.
        require(run(lone, "step_elastic", 2, {{"forms", {"sp1", "sp2", "cfg2"}}, {"prefer_exact", 0.05}}, dear)[0].at("form") == "cfg2", "exact form within the margin");
        require(run(lone, "step_elastic", 2, {{"forms", {"sp1", "sp2", "cfg2"}}, {"prefer_exact", 0.01}}, dear)[0].at("form") == "sp2", "margin exceeded");
        // Elastic with forms: the lone video on 4 GPUs takes the cheapest quad form, CFG2 x SP2 (256.6 ms) over SP4 (266 ms).
        const auto quad = simulate_cluster({{"workload", Json::array({trace_row("v", "video", 0, 1e9)})}, {"profile", prof}, {"policy", "step_elastic"}, {"gpus", 4},
                                            {"options", {{"forms", {"sp1", "cfg2", "sp2", "cfg2sp2", "sp4"}}}}});
        require(quad.at("requests")[0].at("forms") == Json::array({"cfg2sp2"}), "elastic picks CFG2 x SP2");
        near(quad.at("makespan_ms").get<double>(), 4 * 256.6 + 100, 1e-6);
    });
    check("per-group link rates", [] {
        Json prof = profile();
        prof["links"]["groups"] = {{"0-1", {{"gbps", 5}}}, {"2-3", {{"gbps", 20}}}, {"0-3", {{"gbps", 10}, {"sp_overlap", 0.5}}}};
        const ClusterCost cost(prof);
        const std::vector<int> slow = {0, 1}, fast = {2, 3}, quad = {0, 1, 2, 3};
        near(cost.step_ms("video", 2), 511);                                              // no group: the degree's rate
        near(cost.step_ms("video", parse_form("sp2"), 1, &fast), 511);
        near(cost.step_ms("video", parse_form("sp2"), 1, &slow), 500 + 800. / 4 / 5 + 1);   // 541
        // CFG2 x SP2 on the quad runs at its slower half's rate.
        near(cost.step_ms("video", parse_form("cfg2sp2"), 1, &quad), 250 + 400. / 4 / 5 + 0.5 + 0.1);
        // The quad entry carries its own overlap; contention scales the per-group rate as the degree's.
        near(cost.step_ms("video", parse_form("sp4"), 1, &quad), 250 + 15 - 0.5 * 15 + 1);
        Json shared = prof; shared["links"]["2"]["concurrent_gbps"] = {{"2", 10}};
        near(ClusterCost(shared).step_ms("video", parse_form("sp2"), 2, &slow), 500 + 800. / 4 / 2.5 + 1);
        rejects([&] { cost.step_ms("video", parse_form("sp2"), 1, &quad); });
        for (const auto* bad : {"1-2", "0-2", "x", "3-0"}) {
            Json broken = profile(); broken["links"]["groups"] = {{bad, {{"gbps", 5}}}};
            rejects([&] { ClusterCost{broken}; });
        }
        // Placement takes the fastest free group of a size: a lone video on the pairs takes {2,3}.
        auto run = [&](const Json& requests, const Json& options) {
            return cluster_plan({{"now_ms", 0}, {"gpus", 4}, {"busy", Json::array()}, {"requests", requests}}, "step_elastic", cost, options).at("assignments");
        };
        const auto lone = run(Json::array({video("v", 1e9)}), {{"degrees", {1, 2}}});
        require(lone.size() == 1 && lone[0].at("group") == Json::array({2, 3}), "fastest pair first");
        // Two videos widened to pairs: the first takes the fast pair, the second the slow one.
        const auto two = run(Json::array({video("a", 1e9), video("b", 1e9)}), {{"degrees", {1, 2}}, {"widen_efficiency", 0.9}});
        require(two.size() == 2 && two[0].at("group") == Json::array({2, 3}) && two[1].at("group") == Json::array({0, 1}), "fast pair, then slow pair");
        // The simulator prices each quantum on the group it ran on.
        const auto sim = simulate_cluster({{"workload", Json::array({trace_row("v", "video", 0, 1e9)})}, {"profile", prof}, {"policy", "static_sp2"}, {"gpus", 4}});
        near(sim.at("makespan_ms").get<double>(), 4 * 511. + 100);
        require(sim.at("requests")[0].at("degrees") == Json::array({2, 1}), "pair then single-GPU decode");
    });
    check("lockstep pairs", [] {
        // Pair step: both computes and both all-to-alls, pair_overlap of the shorter sum hidden, latencies added.
        Json prof = profile(); prof["pair_overlap"] = 1;
        const ClusterCost cost(prof);
        near(cost.pair_ms("video", "video", parse_form("sp2")), 1000 + 20 - 20 + 2);
        near(cost.pair_ms("video", "video", parse_form("sp4")), 500 + 30 - 30 + 2);
        near(ClusterCost(profile()).pair_ms("video", "video", parse_form("sp2")), 2 * 511);
        Json half = profile(); half["pair_overlap"] = 0.5;
        near(ClusterCost(half).pair_ms("video", "video", parse_form("sp2")), 1022 - 10);
        half["pair_overlap"] = 2;
        rejects([&] { ClusterCost{half}; });
        // Two waiting videos on one pair of GPUs: with pairing on they advance in lockstep (1002 ms a
        // step for both, against 511 each in sequence); off, the second waits.
        auto plan2 = [&](const Json& options, const Json& costs = Json()) {
            return cluster_plan({{"now_ms", 0}, {"gpus", 2}, {"busy", Json::array()}, {"requests", {video("a", 1e9), video("b", 1e9)}}},
                                "block_elastic", ClusterCost(costs.is_null() ? prof : costs), options).at("assignments");
        };
        const Json options = {{"degrees", {2}}, {"quantum_ms", 300}, {"pairing", true}};
        auto paired = plan2(options);
        require(paired.size() == 1 && paired[0].at("pair") == "b" && paired[0].at("group") == Json::array({0, 1}), "pair formed");
        require(paired[0].at("quantum") == 2 && paired[0].at("pair_resize") == false, "quantum from the pair's block time (100.2 ms)");
        require(plan2({{"degrees", {2}}, {"quantum_ms", 300}}).size() == 1 && !plan2({{"degrees", {2}}, {"quantum_ms", 300}})[0].contains("pair"), "no pairing by default");
        require(!plan2(options, profile())[0].contains("pair"), "no overlap, no pair");
        // Inside a step the pair resumes together, and a diverged state is rejected.
        const Json mid = {video("a", 1e9, {{"status", "suspended"}, {"prepared", true}, {"group", {0, 1}}, {"form", "sp2"}, {"blocks_done", 4}, {"pair", "b"}}),
                          video("b", 1e9, {{"status", "suspended"}, {"prepared", true}, {"group", {0, 1}}, {"form", "sp2"}, {"blocks_done", 4}, {"pair", "a"}})};
        auto resumed = cluster_plan({{"now_ms", 0}, {"gpus", 2}, {"busy", Json::array()}, {"requests", mid}}, "block_elastic", cost, options).at("assignments");
        require(resumed.size() == 1 && resumed[0].at("reason") == "resume" && resumed[0].at("pair") == "b", "mid-step pair resumes together");
        Json diverged = mid; diverged[1]["blocks_done"] = 3;
        rejects([&] { cluster_plan({{"now_ms", 0}, {"gpus", 2}, {"busy", Json::array()}, {"requests", diverged}}, "block_elastic", cost, options); });
        // Simulated: two videos on two GPUs finish sooner paired than in sequence, and the record counts the pairs.
        const Json trace = Json::array({trace_row("a", "video", 0, 1e9), trace_row("b", "video", 0, 1e9)});
        const auto together = simulate_cluster({{"workload", trace}, {"profile", prof}, {"policy", "block_elastic"}, {"gpus", 2}, {"options", options}});
        const auto apart = simulate_cluster({{"workload", trace}, {"profile", prof}, {"policy", "block_elastic"}, {"gpus", 2}, {"options", {{"degrees", {2}}, {"quantum_ms", 300}}}});
        require(together.at("pairs").get<int>() > 0 && apart.at("pairs") == 0, "pairs counted");
        require(together.at("makespan_ms").get<double>() < apart.at("makespan_ms").get<double>(), "pairing shortens the makespan");
        require(together.at("metrics").at("completed") == 2, "both complete");
        // Images still preempt a paired step at a block boundary.
        const Json mixed = Json::array({trace_row("a", "video", 0, 1e9), trace_row("b", "video", 0, 1e9), trace_row("i", "image", 500, 700)});
        const auto preempted = simulate_cluster({{"workload", mixed}, {"profile", prof}, {"policy", "block_elastic"}, {"gpus", 2}, {"options", options}});
        require(preempted.at("by_kind").at("image").at("ontime") == 1 && preempted.at("metrics").at("completed") == 3, "image on time beside the pair");
    });
    check("image batches", [] {
        Json prof = profile(); prof["classes"]["image"]["batch_ms"] = {{"2", 70}, {"4", 110}};
        const ClusterCost cost(prof);
        near(cost.batch_ms("image", 1), 50);
        near(cost.batch_ms("image", 2), 70);
        near(cost.batch_ms("image", 3), 90);                                  // interpolated
        near(cost.batch_ms("image", 8), 220);                                 // beyond the table: the last per-image rate
        near(ClusterCost(profile()).batch_ms("image", 4), 200);               // no table: no gain
        rejects([] { ClusterCost(profile()).batch_ms("video", 2); });
        Json broken = profile(); broken["classes"]["image"]["batch_ms"] = {{"0", 10}};
        rejects([&] { ClusterCost{broken}; });
        // Four waiting images, one GPU: image_batch 4 sends them as one batch when their deadlines allow it.
        const Json rows = {image("a", 1000), image("b", 1000), image("c", 1000), image("d", 1000)};
        auto run = [&](const Json& options) {
            return cluster_plan({{"now_ms", 0}, {"gpus", 1}, {"busy", Json::array()}, {"requests", rows}}, "static_sp1", cost, options).at("assignments");
        };
        const auto single = run(Json::object());
        require(single.size() == 1 && !single[0].contains("batch"), "one image at a time by default");
        const auto batched = run({{"image_batch", 4}});
        require(batched.size() == 1 && batched[0].at("batch") == Json::array({"a", "b", "c", "d"}), "all four in one batch");
        // A tight deadline keeps the batch small: with "c" due at 80 ms a batch of three would miss.
        const Json tight = {image("a", 1000), image("b", 1000), image("c", 80), image("d", 1000)};
        const auto limited = cluster_plan({{"now_ms", 0}, {"gpus", 1}, {"busy", Json::array()}, {"requests", tight}}, "static_sp1", cost, {{"image_batch", 4}}).at("assignments");
        require(limited.size() == 1 && limited[0].at("request_id") == "c" && limited[0].at("batch") == Json::array({"c", "a"}), "EDF first, then only members that fit");
        // Simulated: four images on one GPU finish in 110 ms batched against 200 ms in sequence.
        Json trace = Json::array();
        for (int i = 0; i < 4; ++i) trace.push_back(trace_row("i" + std::to_string(i), "image", 0, 1000));
        const auto together = simulate_cluster({{"workload", trace}, {"profile", prof}, {"policy", "static_sp1"}, {"gpus", 1}, {"options", {{"image_batch", 4}}}});
        near(together.at("makespan_ms").get<double>(), 110);
        require(together.at("batches") == 1 && together.at("batched_images") == 4 && together.at("metrics").at("completed") == 4, "batch counted");
        near(simulate_cluster({{"workload", trace}, {"profile", prof}, {"policy", "static_sp1"}, {"gpus", 1}}).at("makespan_ms").get<double>(), 200);
    });
    check("continuous image batches", [] {
        Json prof = profile();
        prof["classes"]["image"].merge_patch({{"steps", 2}, {"prepare_ms", 5}, {"finish_ms", 10},
            {"batch_step_ms", {{"1", 30}, {"2", 40}, {"4", 60}, {"8", 100}}}});
        const ClusterCost cost(prof);
        near(cost.continuous_step_ms("image", 1), 30);
        near(cost.continuous_step_ms("image", 3), 50);
        near(cost.continuous_step_ms("image", 8), 100);
        Json active = image("a", 1000);
        active.merge_patch({{"status", "suspended"}, {"prepared", true}, {"steps_done", 1}, {"steps_total", 2},
                            {"group", {0}}, {"continuous_batch", true}});
        Json fresh = image("b", 1000); fresh["continuous_batch"] = true; fresh["steps_total"] = 2;
        Json other = image("c", 1000);
        other.merge_patch({{"status", "suspended"}, {"prepared", true}, {"steps_done", 1}, {"steps_total", 2},
                           {"group", {1}}, {"continuous_batch", true}});
        near(cost.remaining_ms(active, 1), 40);
        near(cost.remaining_ms(fresh, 1), 75);
        near(cost.remaining_ms(image("plain", 1000), 1), 50);
        const Json requests = {active, fresh, other};
        const auto assignments = cluster_plan({{"now_ms", 0}, {"gpus", 2}, {"busy", Json::array()}, {"requests", requests}},
                                               "static_sp1", cost, {{"continuous_batch", 4}}).at("assignments");
        require(assignments.size() == 2, "each resident GPU gets its own continuous round");
        require(assignments[0].at("group") == Json::array({0}) && assignments[0].at("batch") == Json::array({"a", "b"}),
                "active A joins new B on GPU0");
        require(assignments[0].at("quantum") == "continuous_batch", "explicit-step quantum");
        require(assignments[1].at("group") == Json::array({1}) && assignments[1].at("batch") == Json::array({"c"}),
                "active state on another GPU is not concatenated");
        Json tight = active; tight["deadline_ms"] = 80;
        Json late = fresh; late["deadline_ms"] = 90;
        const auto limited = cluster_plan({{"now_ms", 0}, {"gpus", 1}, {"busy", Json::array()}, {"requests", {tight, late}}},
                                           "static_sp1", cost, {{"continuous_batch", 4}}).at("assignments");
        require(!limited.empty() && limited[0].at("batch") == Json::array({"a"}), "new member is rejected before it misses A's deadline");
        rejects([&] { cluster_plan({{"now_ms", 0}, {"gpus", 1}, {"requests", {fresh}}}, "static_sp1", cost,
                                    {{"continuous_batch", 4}, {"image_batch", 2}}); });
        rejects([&] { cluster_plan({{"now_ms", 0}, {"gpus", 1}, {"requests", {fresh}}}, "static_sp1", cost,
                                    {{"continuous_batch", 9}}); });
        rejects([] { ClusterCost(profile()).continuous_step_ms("image", 1); });
        Json broken = profile(); broken["classes"]["image"]["steps"] = 0;
        rejects([&] { ClusterCost{broken}; });
    });
    check("queue order and round-based regrouping", [] {
        // "b" is due later but arrived earlier and has less work left (3 steps of 4): edf takes "a",
        // srtf and fcfs take "b".
        const Json rows = {video("a", 5000, {{"arrival_ms", 20}}),
                           video("b", 9000, {{"status", "suspended"}, {"prepared", true}, {"group", {0}}, {"steps_done", 1}, {"arrival_ms", 10}})};
        auto first = [&](const Json& options) {
            return cluster_plan({{"now_ms", 100}, {"gpus", 1}, {"busy", Json::array()}, {"requests", rows}}, "static_sp1", ClusterCost(profile()), options)
                .at("assignments")[0].at("request_id");
        };
        require(first(Json::object()) == "a", "edf by deadline");
        require(first({{"order", "srtf"}}) == "b", "srtf by remaining work");
        require(first({{"order", "fcfs"}}) == "b", "fcfs by arrival");
        rejects([&] { first({{"order", "lifo"}}); });
        // A video after 2 of 4 steps on GPU 3 keeps its group under rounds of 2 steps... only when
        // the count is not a multiple: after 1 step it stays, after 2 it may regroup to the free pair.
        auto placed = [](int steps) {
            return Json::array({video("v", 1e9, {{"status", "suspended"}, {"prepared", true}, {"group", {3}}, {"steps_done", steps}, {"steps_total", 10}})});
        };
        const Json busy = Json::array({false, false, true, false});
        auto stays = plan(placed(1), "round_elastic", 4, busy, {{"round_steps", 2}});
        require(stays.size() == 1 && stays[0].at("group") == Json::array({3}) && stays[0].at("reason") == "resume", "inside a round the group holds");
        auto moves = plan(placed(2), "round_elastic", 4, busy, {{"round_steps", 2}});
        require(moves.size() == 1 && moves[0].at("group") == Json::array({0, 1}) && moves[0].at("resize") == true, "at a round boundary it regroups");
        require(plan(placed(1), "step_elastic", 4, busy)[0].at("group") == Json::array({0, 1}), "step_elastic regroups every step");
        require(plan(placed(1), "round_elastic", 4, busy)[0].at("group") == Json::array({3}), "default round of 5 steps");
        require(plan(placed(5), "round_elastic", 4, busy)[0].at("group") == Json::array({0, 1}), "fifth step ends the round");
        const auto sim = simulate(Json::array({trace_row("v", "video", 0, 1e9)}), "round_elastic", 4);
        require(sim.at("metrics").at("completed") == 1 && sim.at("requests")[0].at("degrees") == Json::array({4, 1}), "lone video still takes the node");
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
        // 4 GPUs, SP1/2/4, ~300 ms quanta, overloaded mixed traffic with
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
        near(cost.resume_ms("video"), 0);
        Json resumed = profile(); resumed["classes"]["video"]["resume_ms"] = 250;
        near(ClusterCost(resumed).resume_ms("video"), 250);
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
    check("2D hybrid USP forms and laxity-driven feature caching", [] {
        const Form u4r2 = parse_form("u4r2"), cfg2u2r2 = parse_form("cfg2u2r2");
        require(u4r2.cfg == 1 && u4r2.sp == 4 && u4r2.ring == 2 && u4r2.degree() == 8 && form_name(u4r2) == "u4r2", "u4r2 parsed");
        require(cfg2u2r2.cfg == 2 && cfg2u2r2.sp == 2 && cfg2u2r2.ring == 2 && cfg2u2r2.degree() == 8 && form_name(cfg2u2r2) == "cfg2u2r2", "cfg2u2r2 parsed");
        const ClusterCost cost(profile());
        // On 8-GPU dual-socket PCIe (link 4 = 10 GB/s, link 8 = 5 GB/s), u4r2 confines All-to-All to quad links
        // and overlaps cross-socket Ring P2P, beating flat sp8 All-to-All.
        require(cost.step_ms("video", u4r2) < cost.step_ms("video", parse_form("sp8")), "u4r2 beats flat sp8 across sockets");
        require(cost.step_ms("video", u4r2, 1, nullptr, 0.16) < cost.step_ms("video", u4r2, 1, nullptr, 0.0), "feature cache skips compute and comm");
    });
    check("knapsack_dp, heterodit_exact and heterodit_approx policies", [] {
        require(cluster_policy("knapsack_dp") && cluster_policy("heterodit_exact") && cluster_policy("heterodit_approx"), "policies registered");
        Json prof = profile();
        prof["classes"]["video"]["cfg_exchange_mb"] = 20;
        prof["classes"]["video"]["resume_ms"] = 80;
        prof["classes"]["image"]["batch_ms"] = {{"2", 70}, {"4", 110}};
        const ClusterCost cost(prof);
        // Urgent video that cannot meet 1500 ms deadline strictly on 2 GPUs (needs ~2004 ms exact),
        // but meets it under heterodit_approx via dynamic feature caching.
        const Json tight = Json::array({video("v1", 1500), video("v2", 2500)});
        const auto exact_plan = cluster_plan({{"now_ms", 0}, {"gpus", 4}, {"busy", Json::array()}, {"requests", tight}}, "heterodit_exact", cost);
        const auto approx_plan = cluster_plan({{"now_ms", 0}, {"gpus", 4}, {"busy", Json::array()}, {"requests", tight}}, "heterodit_approx", cost);
        require(exact_plan.contains("tiered_hints") && approx_plan.contains("tiered_hints"), "tiered state hints emitted");
        require(approx_plan.at("assignments")[0].value("cache_threshold", 0.0) > 0.0, "approx selects positive cache_threshold for tight video");

        prof["links"]["3"] = {{"gbps", 15}, {"latency_ms", 0.1}};
        const ClusterCost morph_cost(prof);
        Json morph = video("m", 1e9, {{"status", "suspended"}, {"prepared", true}, {"steps_done", 12}, {"steps_total", 20},
            {"group", {0, 1, 2, 3, 4, 5}}, {"form", "cfg2sp3"}, {"cfg_fold_eligible", true}});
        const auto morph_plan = cluster_plan({{"now_ms", 0}, {"gpus", 8}, {"busy", Json::array()},
            {"requests", {morph, image("i", 1e9)}}}, "heterodit_v5", morph_cost, {{"forms", {"cfg2sp3"}}});
        Json folded;
        for (const auto& assignment : morph_plan.at("assignments")) if (assignment.at("request_id") == "m") folded = assignment;
        require(!folded.is_null() && folded.at("group") == Json::array({0, 1, 2}) && folded.at("form") == "sp3"
                && folded.value("cfg_folded", false), "worker-gated 6+2 to 3+5 morphing");
        morph["blocks_done"] = 4; morph["group"] = {0, 1, 2}; morph["form"] = "sp3"; morph["cfg_folded"] = true;
        const auto resumed = cluster_plan({{"now_ms", 0}, {"gpus", 8}, {"busy", Json::array()}, {"requests", {morph}}},
            "heterodit_v5", morph_cost, {{"forms", {"cfg2sp3"}}}).at("assignments")[0];
        require(resumed.value("cfg_folded", false), "mid-step resume preserves CFG-fold mode");

        Json generator = {{"count", 80}, {"seed", 7}, {"pattern", "poisson"}, {"rate_rps", 4},
            {"classes", {{{"class", "video"}, {"kind", "video"}, {"weight", 1}, {"service_ms", 4000}, {"slo_scale", 1.5}},
                         {{"class", "image"}, {"kind", "image"}, {"weight", 3}, {"service_ms", 50}, {"slo_ms", 300}}}}};
        const auto trace = study_workload({{"generator", generator}});
        const auto sim_exact = simulate_cluster({{"workload", trace}, {"profile", prof}, {"policy", "heterodit_exact"}, {"gpus", 8}});
        const auto sim_approx = simulate_cluster({{"workload", trace}, {"profile", prof}, {"policy", "heterodit_approx"}, {"gpus", 8}});
        const Json v5_options = {{"cfg_fold_eligible_from_step", 1}, {"cfg_fold_step_scale", 0.52}, {"async_decode_scale", 0.15}};
        const auto sim_v5 = simulate_cluster({{"workload", trace}, {"profile", prof}, {"policy", "heterodit_v5"}, {"gpus", 8},
                                             {"options", v5_options}});
        require(sim_exact.at("metrics").at("completed") == 80 && sim_approx.at("metrics").at("completed") == 80
                && sim_v5.at("metrics").at("completed") == 80, "all requests complete");
        require(sim_approx.at("approx_quanta").get<int>() > 0 && sim_v5.at("approx_quanta").get<int>() > 0, "approx quanta used under load");
        require(sim_v5.at("makespan_ms").get<double>() <= sim_exact.at("makespan_ms").get<double>(), "heterodit_v5 reduces or matches makespan");
        rejects([&] { simulate_cluster({{"workload", trace}, {"profile", prof}, {"policy", "heterodit_v5"}, {"gpus", 8},
                                       {"options", {{"cfg_fold_eligible_from_step", 1}}}}); });
    });
    check("async decode holds video placement but not independent images", [] {
        const Json requests = {video("a", 1e9), video("b", 1e9)};
        const auto videos = cluster_plan({{"now_ms", 0}, {"gpus", 2}, {"busy", Json::array()},
            {"async_decoding", {true, false}}, {"requests", requests}}, "block_elastic", ClusterCost(profile()),
            {{"degrees", {1}}}).at("assignments");
        require(videos.size() == 1 && videos[0].at("group") == Json::array({1}), "decode GPU cannot start another video");
        const auto images = cluster_plan({{"now_ms", 0}, {"gpus", 2}, {"busy", Json::array()},
            {"async_decoding", {true, false}}, {"requests", {image("i", 100)}}}, "static_sp1", ClusterCost(profile())).at("assignments");
        require(images.size() == 1 && images[0].at("group") == Json::array({0}), "independent image can overlap decode");
    });
    check("FlashPS-inspired bubble-free continuous batching and heterogeneous sequence packing", [] {
        Json prof = profile();
        prof["classes"]["image"] = {{"kind", "image"}, {"service_ms", 100}, {"steps", 2},
                                    {"prepare_ms", 20}, {"finish_ms", 40},
                                    {"batch_ms", {{"2", 80}, {"4", 120}}},
                                    {"batch_step_ms", {{"1", 50}, {"2", 50}, {"4", 50}}}};
        const ClusterCost cost(prof);
        auto img_with = [](const std::string& id, double deadline, const Json& extra) {
            Json row = image(id, deadline);
            row.merge_patch(extra);
            return row;
        };
        // Request 'a' is at step 1/2 (exiting after this step); request 'b' arrives at step 0/2 with deadline 160 ms.
        // Without bubble_free: round_cost({a, b}) = 50 + 20 (b prepare) + 40 (a finish) = 110 ms,
        // so b's total completion time is 110 + 50 + 40 = 200 ms > 160 ms (cannot join).
        // With bubble_free: round_cost({a, b}) = 50 ms (b's prepare overlapped, a's 2D-VAE finish decoupled on side stream),
        // so b completes at 50 + 50 + 40 = 140 ms <= 160 ms (joins 'a' in the same continuous batch!).
        const Json cb_rows = Json::array({
            img_with("a", 120, {{"status", "suspended"}, {"prepared", true}, {"group", {0}}, {"steps_done", 1}, {"steps_total", 2}, {"continuous_batch", true}}),
            img_with("b", 160, {{"status", "waiting"}, {"prepared", false}, {"steps_done", 0}, {"steps_total", 2}, {"continuous_batch", true}})
        });
        const auto serial_cb = cluster_plan({{"now_ms", 0}, {"gpus", 1}, {"busy", Json::array()}, {"requests", cb_rows}},
                                            "static_sp1", cost, {{"continuous_batch", 4}, {"bubble_free", false}}).at("assignments");
        const auto bubble_free_cb = cluster_plan({{"now_ms", 0}, {"gpus", 1}, {"busy", Json::array()}, {"requests", cb_rows}},
                                                 "static_sp1", cost, {{"continuous_batch", 4}, {"bubble_free", true}}).at("assignments");
        require(serial_cb.size() == 1 && serial_cb[0].at("batch").size() == 1, "serial prepare+finish prevents late join under tight deadline");
        require(bubble_free_cb.size() == 1 && bubble_free_cb[0].at("batch") == Json::array({"a", "b"})
                && bubble_free_cb[0].value("bubble_free", false), "bubble_free decouples encode/VAE so late join fits");

        // Heterogeneous resolution packing: 512x512 and 768x768 requests have different batch_spec.height/width.
        const Json res_rows = Json::array({
            img_with("r512", 500, {{"batch_spec", {{"steps", 12}, {"height", 512}, {"width", 512}}}}),
            img_with("r768", 500, {{"batch_spec", {{"steps", 12}, {"height", 768}, {"width", 768}}}})
        });
        const auto unpacked = cluster_plan({{"now_ms", 0}, {"gpus", 1}, {"busy", Json::array()}, {"requests", res_rows}},
                                           "static_sp1", cost, {{"image_batch", 4}, {"sequence_packing", false}}).at("assignments");
        const auto packed = cluster_plan({{"now_ms", 0}, {"gpus", 1}, {"busy", Json::array()}, {"requests", res_rows}},
                                         "static_sp1", cost, {{"image_batch", 4}, {"sequence_packing", true}}).at("assignments");
        require(unpacked.size() == 1 && !unpacked[0].contains("batch"), "without sequence_packing heterogeneous resolutions cannot batch");
        require(packed.size() == 1 && packed[0].at("batch") == Json::array({"r512", "r768"})
                && packed[0].value("sequence_packing", false), "sequence_packing batches heterogeneous resolutions together");

        // Simulate continuous_batch with and without bubble_free
        const Json sim_trace = Json::array({
            trace_row("i0", "image", 0, 500),
            trace_row("i1", "image", 50, 500)
        });
        const auto sim_serial = simulate_cluster({{"workload", sim_trace}, {"profile", prof}, {"policy", "static_sp1"},
                                                  {"gpus", 1}, {"options", {{"continuous_batch", 4}, {"bubble_free", false}}}});
        const auto sim_bf = simulate_cluster({{"workload", sim_trace}, {"profile", prof}, {"policy", "static_sp1"},
                                              {"gpus", 1}, {"options", {{"continuous_batch", 4}, {"bubble_free", true}}}});
        require(sim_serial.at("metrics").at("completed") == 2 && sim_bf.at("metrics").at("completed") == 2, "both simulations complete");
        require(sim_bf.at("makespan_ms").get<double>() < sim_serial.at("makespan_ms").get<double>(), "bubble_free reduces continuous batch makespan");
    });

    // ---- policy registry and the baseline / heuristic / ablation policies ----
    // Measured Wan2.1 480x832x33f step times on 8x A100-PCIe (same-allocation calibration, 2026-11-01).
    auto pcie_profile = [] {
        Json p = profile();
        Json forms = {{"sp1", {{"step_ms", 1147.0}}}, {"cfg2", {{"step_ms", 583.4}}}, {"sp2", {{"step_ms", 1196.9}}},
                      {"cfg2sp2", {{"step_ms", 790.6}}}, {"sp3", {{"step_ms", 1101.1}}}, {"cfg2sp3", {{"step_ms", 622.0}}},
                      {"sp4", {{"step_ms", 1091.6}}}, {"cfg2sp4", {{"step_ms", 567.1}}}, {"sp6", {{"step_ms", 827.9}}}};
        p["classes"]["video"] = {{"kind", "video"}, {"steps", 20}, {"blocks_per_step", 60}, {"prepare_ms", 0}, {"encode_ms", 0},
                                 {"step_compute_ms", 1147.0}, {"comm_mb_per_step", 10351}, {"collectives_per_step", 240},
                                 {"finish_ms", 1840}, {"forms", forms}};
        p["classes"]["image"] = {{"kind", "image"}, {"service_ms", 6450}};
        for (const char* d : {"2", "3", "4", "6", "8"}) p["links"][d] = {{"gbps", 3.75}, {"latency_ms", 0.5}};
        return p;
    };
    const Json pcie_forms = {{"forms", {"sp1", "cfg2", "sp2", "cfg2sp2", "sp3", "cfg2sp3", "sp4", "cfg2sp4", "sp6"}}};
    auto pcie_plan = [&](const Json& requests, const std::string& policy, Json busy = Json::array(), Json extra = Json::object()) {
        Json options = pcie_forms;
        options.merge_patch(extra);
        return cluster_plan({{"now_ms", 0}, {"gpus", 8}, {"busy", busy}, {"requests", requests}}, policy,
                            ClusterCost(pcie_profile()), options);
    };
    check("policy registry", [] {
        for (const char* id : {"xdit", "tetriserve", "sbp-esp", "edf-bestfit", "edf-cfg2", "cfg2-first", "partition", "ditserve",
                               "ditserve-approx", "abl-no-block", "abl-no-prune", "abl-no-mindeg", "abl-first-fit", "ditserve-v1",
                               "block-cfg2", "static-cfg2sp3", "xdit-sp4"})
            require(cluster_policy(id), std::string("registered policy: ") + id);
        require(canonical_policy("heterodit_exact") == "ditserve-v1" && canonical_policy("step_elastic") == "abl-step-elastic"
                && canonical_policy("static_cfg2") == "static-cfg2" && canonical_policy("block_cfg2") == "block-cfg2"
                && canonical_policy("heterodit_v5") == "ditserve-v1-approx" && canonical_policy("xdit") == "xdit-cfg2",
                "legacy names resolve to canonical IDs");
        for (const char* retired : {"vanilla_xdit", "tridentserve_stage", "vllm_omni", "gf_dit", "genserve_40gb"})
            require(!cluster_policy(retired), std::string("retired name rejected: ") + retired);
        rejects([] { policy_traits("no_such_policy"); });
        rejects([] { policy_traits("static-cfg3"); });
        const auto& d = policy_traits("ditserve");
        require(d.family == "ours" && d.blocks && d.elastic && d.prune_dominated && d.min_video_degree == 2 && !d.knapsack,
                "DitServe: block preemption, pruning, CFG-pair admission, no hand-weighted knapsack");
        require(policy_traits("abl-no-block").family == "ablation" && !policy_traits("abl-no-block").blocks, "ablation removes one mechanism");
        require(policy_traits("sbp-esp").display == "SBP-ESP" && policy_traits("edf-bestfit").display == "EDF-BF",
                "preprint methods are named by method");
        require(policy_traits("xdit").hold_diffuse && policy_traits("xdit").order == "fcfs" && policy_traits("xdit").partition_divisor == 4,
                "xDiT: disjoint model GPUs, FCFS, no preemption");
        const auto catalog = policy_catalog();
        require(catalog.is_array() && catalog.size() >= 20, "catalog lists every policy");
    });
    check("legacy alias plans name the canonical policy", [] {
        const auto result = cluster_plan({{"now_ms", 0}, {"gpus", 2}, {"busy", Json::array()}, {"requests", Json::array({video("v", 1e9)})}},
                                         "step_elastic", ClusterCost(profile()));
        require(result.at("policy") == "abl-step-elastic" && result.at("policy_alias_of") == "step_elastic", "alias recorded");
    });
    check("dominated forms pruned on PCIe", [&] {
        // A lone video with nobody waiting: DitServe keeps one CFG pair (cfg2sp4 saves 3% for 4x the GPUs).
        const auto lone = pcie_plan(Json::array({video("v", 1e9)}), "ditserve").at("assignments");
        require(lone.size() == 1 && lone[0].at("form") == "cfg2", "lone video stays on a CFG pair: " + lone.dump());
        // Without pruning the work-conserving rule hands it the widest group.
        const auto wide = pcie_plan(Json::array({video("v", 1e9)}), "abl-no-prune").at("assignments");
        require(wide.size() == 1 && wide[0].at("degree") == 8, "unpruned lone video takes the node: " + wide.dump());
        // cfg2sp2 (4 GPUs, slower than cfg2) is never chosen with pruning even when a deadline is tight.
        const auto tight = pcie_plan(Json::array({video("v", 20 * 700.0 + 1840), video("w", 1e9)}), "ditserve").at("assignments");
        for (const auto& a : tight) require(a.at("form") != "cfg2sp2" && a.at("form") != "sp2" && a.at("form") != "sp4", "no dominated form");
    });
    check("CFG-pair admission waits instead of starting narrow", [&] {
        // Only GPU 7 free: abl-no-mindeg starts the video on SP1, DitServe and CFG2-First wait.
        const Json busy = {true, true, true, true, true, true, true, false};
        for (const char* policy : {"ditserve", "cfg2-first"})
            require(pcie_plan(Json::array({video("v", 1e9), video("w", 1e9)}), policy, busy).at("assignments").empty(),
                    std::string(policy) + " waits for a pair");
        const auto narrow = pcie_plan(Json::array({video("v", 1e9), video("w", 1e9)}), "abl-no-mindeg", busy).at("assignments");
        require(narrow.size() == 1 && narrow[0].at("degree") == 1, "without the minimum the video starts on SP1");
        // Images still take the lone GPU.
        const auto img = pcie_plan(Json::array({video("v", 1e9), image("i", 7000)}), "ditserve", busy).at("assignments");
        require(img.size() == 1 && img[0].at("request_id") == "i", "image placed while the video waits");
    });
    check("deadline knapsack keeps the most requests on track", [&] {
        // Four GPUs free, three videos: two can make it on cfg2, the third cannot on any form.
        const Json busy = {true, true, true, true, false, false, false, false};
        const double ok = 20 * 583.4 + 1840 + 500, hopeless = 5000;
        const auto result = pcie_plan(Json::array({video("late", hopeless), video("a", ok), video("b", ok + 1)}), "tetriserve", busy,
                                      {{"demote_late", false}}).at("assignments");
        std::set<std::string> placed;
        for (const auto& a : result) { placed.insert(a.at("request_id").get<std::string>()); require(a.at("degree") == 2, "minimal feasible pair"); }
        require(placed == std::set<std::string>{"a", "b"}, "the knapsack drops the hopeless request: " + result.dump());
        // sbp-esp re-plans every step with the same objective.
        const auto genserve = pcie_plan(Json::array({video("late", hopeless), video("a", ok), video("b", ok + 1)}), "sbp-esp", busy,
                                        {{"demote_late", false}}).at("assignments");
        require(genserve.size() == 2, "SBP-ESP keeps the two feasible videos");
    });
    check("EDF best fit with surplus redistribution", [&] {
        // One feasible video, all 8 GPUs free, nothing else waiting: best fit takes the smallest feasible
        // size, then surplus promotion widens it step by step while GPUs remain.
        const auto result = pcie_plan(Json::array({video("a", 1e9)}), "edf-bestfit").at("assignments");
        require(result.size() == 1 && result[0].at("degree") == 8, "surplus promotes to the widest size: " + result.dump());
        // Two videos: both placed, the earlier deadline first.
        const auto two = pcie_plan(Json::array({video("a", 1e9), video("b", 2e9)}), "edf-bestfit").at("assignments");
        require(two.size() == 2, "both placed");
    });
    check("xDiT and Static-Partition reserve image GPUs", [&] {
        const auto x = pcie_plan(Json::array({video("v1", 1e9), video("v2", 1e9), video("v3", 1e9), video("v4", 1e9)}), "xdit").at("assignments");
        require(x.size() == 3, "two of eight GPUs serve images only, so three CFG pairs: " + x.dump());
        for (const auto& a : x) for (const auto& g : a.at("group")) require(g.get<int>() < 6, "videos stay off image GPUs");
        const auto sized = pcie_plan(Json::array({video("v1", 1e9), video("v2", 1e9), video("v3", 1e9)}), "partition", Json::array(),
                                     {{"partition_image_gpus", 4}}).at("assignments");
        require(sized.size() == 2, "partition_image_gpus resizes the image share");
        // reserve_image_gpus is unchanged for other policies.
        require(pcie_plan(Json::array({video("v1", 1e9), video("v2", 1e9), video("v3", 1e9), video("v4", 1e9)}), "edf-cfg2").at("assignments").size() == 4,
                "EDF-CFG2 uses every GPU");
    });
    check("fold-only and fission resource arms remain distinct", [] {
        Json request = video("fold", 1e9, {{"status", "suspended"}, {"prepared", true}, {"steps_done", 12}, {"steps_total", 20},
            {"group", {0, 1, 2, 3, 4, 5}}, {"form", "cfg2sp3"}, {"cfg_fold_eligible", true}});
        const Json state = {{"now_ms", 0}, {"gpus", 8}, {"busy", Json::array()}, {"requests", {request}}};
        const Json options = {{"forms", {"cfg2sp3"}}};
        Json prof = profile(); prof["links"]["3"] = {{"gbps", 15}, {"latency_ms", 0.1}};
        const ClusterCost cost(prof);
        const auto held = cluster_plan(state, "fold-cfg2sp3", cost, options).at("assignments")[0];
        require(held.at("form") == "cfg2sp3" && held.at("group").size() == 6 && held.value("cfg_folded", false),
                "fold-only keeps the complete CFG2 group reserved");
        Json continuation = request;
        continuation["cfg_folded"] = true; continuation["cfg_fold_eligible"] = false; continuation["blocks_done"] = 4;
        const auto resumed = cluster_plan({{"now_ms", 0}, {"gpus", 8}, {"busy", Json::array()}, {"requests", {continuation}}},
            "fold-cfg2sp3", cost, options).at("assignments")[0];
        require(resumed.value("cfg_folded", false), "fold-only preserves the mode inside an interrupted solver step");
        continuation["blocks_done"] = 0; continuation["steps_done"] = 19;
        const auto anchor = cluster_plan({{"now_ms", 0}, {"gpus", 8}, {"busy", Json::array()}, {"requests", {continuation}}},
            "fold-cfg2sp3", cost, options).at("assignments")[0];
        require(!anchor.value("cfg_folded", false) && anchor.at("form") == "cfg2sp3",
                "fold-only re-evaluates eligibility and executes the final exact anchor");
        const auto released = cluster_plan(state, "approx-cfg2sp3", cost, options).at("assignments")[0];
        require(released.at("form") == "sp3" && released.at("group").size() == 3 && released.value("cfg_folded", false)
                && released.value("fission", false), "fission contracts the conditional subgroup");
        const auto idle = cluster_plan(state, "ditserve-approx", cost, options).at("assignments")[0];
        require(idle.at("form") == "cfg2sp3" && idle.at("group").size() == 6 && idle.value("cfg_folded", false)
                && !idle.value("fission", true) && !idle.at("resize").get<bool>(),
                "without demand, folding avoids a pointless regroup");
        request["group"] = {0, 1}; request["form"] = "cfg2";
        const auto pair_plan = cluster_plan({{"now_ms", 0}, {"gpus", 2}, {"busy", Json::array()}, {"requests", {request, image("i", 2e9)}}},
            "ditserve-approx", ClusterCost(profile()), {{"forms", {"cfg2"}}}).at("assignments");
        require(pair_plan.size() == 2, "fission releases capacity for a real subsequent assignment");
        const auto pair = pair_plan[0];
        require(pair.at("form") == "sp1" && pair.at("group") == Json::array({0}) && pair.value("fission", false),
                "a CFG pair can fission to one GPU when another request needs capacity");
        require(pair_plan[1].at("request_id") == "i" && pair_plan[1].at("group") == Json::array({1}),
                "the waiting image consumes the released companion GPU");
        require(!policy_traits("fold-cfg2sp3").fission && policy_traits("approx-cfg2sp3").fission,
                "policy identity records the resource mechanism");
    });
    check("new policies simulate to completion", [&] {
        Json trace = Json::array();
        for (int i = 0; i < 12; ++i) trace.push_back(trace_row("v" + std::to_string(i), "video", i * 3000.0, i * 3000.0 + 60000));
        for (int i = 0; i < 12; ++i) trace.push_back(trace_row("i" + std::to_string(i), "image", i * 2500.0 + 700, i * 2500.0 + 700 + 9000));
        for (const char* policy : {"xdit", "tetriserve", "sbp-esp", "edf-bestfit", "edf-cfg2", "cfg2-first", "partition", "ditserve",
                                   "abl-no-block", "abl-no-prune", "abl-no-mindeg", "abl-first-fit", "ditserve-v1", "block-cfg2"}) {
            Json options = pcie_forms;
            const auto result = simulate_cluster({{"workload", trace}, {"profile", pcie_profile()}, {"policy", policy}, {"gpus", 8}, {"options", options}});
            require(result.at("metrics").at("completed") == 24, std::string("simulation completes: ") + policy);
        }
    });
    std::cout << checks - failures << "/" << checks << " cluster checks passed\n";
    return failures ? 1 : 0;
}
