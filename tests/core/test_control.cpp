#include "ditserve/control/dispatch.hpp"
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <set>
#include <stdexcept>

#ifndef CONTROL_BASELINE_PATH
#define CONTROL_BASELINE_PATH "tests/fixtures/control-baseline.json"
#endif

using namespace ditserve;
namespace {
int failures = 0, checks = 0;
void require(bool passed, const std::string& message) { if (!passed) throw std::runtime_error(message); }
void near(double actual, double expected) { require(std::abs(actual - expected) < 1e-9, "numerical mismatch"); }
void rejects(const std::function<void()>& action) {
    try { action(); } catch (const std::exception&) { return; }
    throw std::runtime_error("invalid input unexpectedly accepted");
}
void check(const std::string& name, const std::function<void()>& action) {
    ++checks;
    try { action(); }
    catch (const std::exception& error) { ++failures; std::cerr << name << ": " << error.what() << '\n'; }
}
void golden() {
    std::ifstream source(CONTROL_BASELINE_PATH);
    if (!source) throw std::runtime_error("missing frozen control baseline");
    Json fixture; source >> fixture;
    require(fixture.at("schema") == "ditserve.control-baseline.v1", "unknown control fixture");
    require(fixture.at("provenance").at("sources").size() == 3, "missing Python oracle provenance");
    for (const auto& example : fixture.at("cases")) check(example.at("name"), [&] {
        const auto before = example.at("rows");
        auto rows = before;
        const auto execute = [&] { return select_action(rows, example.at("now_ms"), example.at("policy"),
                                                       example.at("costs"), example.at("active")); };
        if (example.value("expect_error", false)) rejects([&] { execute(); });
        else {
            const auto actual = execute();
            if (example.contains("expected_request_id"))
                require(actual.at("request_id") == example.at("expected_request_id"), "session choice differs from frozen Python");
            else require(actual == example.at("expected"), "dispatch differs from frozen Python: " + actual.dump());
        }
        require(rows == before, "selection mutated input");
    });
}
void benchmark() {
    Json rows = Json::array({{{"request_id","old"},{"arrival_ms",0},{"deadline_ms",100},{"remaining_ms",50},
                             {"arrived",true},{"gpu_ready",true},{"status","pending"}},
                            {{"request_id","urgent"},{"arrival_ms",1},{"deadline_ms",20},{"remaining_ms",5},
                             {"arrived",true},{"gpu_ready",true},{"status","pending"}}});
    require(select_action(rows, 10, "fcfs", Json::object(), "old")["request_id"] == "old", "FCFS interrupted active work");
    for (const auto& policy : {"edf", "srtf"})
        require(select_action(rows, 10, policy, Json::object(), "old")["request_id"] == "urgent", "priority did not preempt");
    rows[1]["deadline_ms"] = 100; rows[1]["remaining_ms"] = 50;
    for (const auto& policy : {"edf", "srtf"})
        require(select_action(rows, 10, policy, Json::object(), "urgent")["request_id"] == "urgent", "priority tie switched active work");
}
void matrix() {
    const Json phase = {{"kind","online"},{"spec",{{"steps",3}}},
                        {"matrix",{{"/policy",Json::array({"edf","srtf"})},{"/spec/seed",Json::array({1,2})}}}};
    const auto variants = expand_matrix(phase);
    require(variants.size() == 4, "matrix omitted Cartesian combinations");
    std::set<std::pair<std::string,int>> combinations;
    for (const auto& cell : variants) {
        require(cell["kind"] == "online" && cell["spec"]["steps"] == 3 && !cell.contains("matrix"), "matrix changed base configuration");
        combinations.emplace(cell["policy"], cell["spec"]["seed"]);
    }
    require(combinations == std::set<std::pair<std::string,int>>{{"edf",1},{"edf",2},{"srtf",1},{"srtf",2}}, "matrix duplicated or missed a cell");
    require(phase["spec"].size() == 1, "matrix mutated source");
    require(expand_matrix({{"policy","edf"}}) == Json::array({{{"policy","edf"}}}), "empty matrix changed one-cell experiment");
    rejects([] { expand_matrix({{"matrix",{{"/policy",Json::array()}}}}); });
    rejects([] { expand_matrix({{"matrix",{{"not-a-pointer",Json::array({1})}}}}); });
    Json hundred = Json::array(); for (int i = 0; i < 101; ++i) hundred.push_back(i);
    rejects([&] { expand_matrix({{"matrix",{{"/a",hundred},{"/b",hundred}}}}); });
}
void gates() {
    const Json evidence = {{"quality",{{"passed",true}}},{"metrics",{{"rate",10},{"gain",-.1}}}};
    const Json rules = Json::array({{{"metric","/quality/passed"},{"op","eq"},{"value",true}},
                                    {{"metric","/metrics/rate"},{"op","ge"},{"value",10}}});
    const auto accepted=evaluate_gates(evidence,rules);
    require(accepted["qualified"] == true, "valid boundary evidence did not qualify");
    require(accepted["scopes"]["numeric"]["status"]=="passed" && accepted["scopes"]["semantic"]["status"]=="unverified" &&
            accepted["scopes"]["performance"]["status"]=="unverified", "numeric gates implied unmeasured quality or performance");
    const auto scoped=evaluate_gates(evidence,Json::array({{{"metric","/metrics/rate"},{"scope","performance"},{"value",5}},
                                                        {{"metric","/blind_review"},{"scope","semantic"},{"value",1}}}));
    require(scoped["scopes"]["performance"]["status"]=="passed" && scoped["scopes"]["semantic"]["status"]=="unverified" &&
            scoped["qualified"]==false,"missing semantic evidence was upgraded by performance evidence");
    const auto absent = evaluate_gates(evidence, Json::array({{{"metric","/missing"},{"op","ge"},{"value",1}}}));
    require(absent["qualified"] == false && absent["checks"][0]["status"] == "unverified", "missing evidence was treated as success");
    require(evaluate_gates(evidence, Json::array())["status"] == "unverified", "no gates implied scientific qualification");
    const auto negative = evaluate_gates(evidence, Json::array({{{"metric","/metrics/gain"},{"op","gt"},{"value",0}}}));
    require(negative["qualified"] == false && negative["checks"][0]["status"] == "failed", "negative improvement did not fail its gate");
    rejects([&] { evaluate_gates(evidence, Json::array({{{"metric","/metrics/rate"},{"op","unknown"},{"value",0}}})); });
    rejects([&] { evaluate_gates(evidence, Json::array({{{"metric","/metrics/rate"},{"scope","unknown"},{"value",0}}})); });
}
Json measured_rows() {
    return Json::array({{{"request_id","qualified"},{"status","completed"},{"arrival_ms",0},{"finish_ms",10},{"deadline_ms",15},
                         {"quality",{{"passed",true}}},{"qualification",{{"qualified",true},{"status","passed"}}},
                         {"model_ms",6},{"control_ms",1},{"communication_samples_ms",Json::array({1,3})}},
                        {{"request_id","unknown-quality"},{"status","completed"},{"arrival_ms",10},{"finish_ms",40},{"deadline_ms",50},
                         {"quality",{{"passed",false}}},{"model_ms",20},{"control_ms",2},{"communication_samples_ms",Json::array({2,4})}},
                        {{"request_id","failed"},{"status","failed"},{"arrival_ms",0},{"finish_ms",100},{"deadline_ms",50}},
                        {{"request_id","blocked"},{"status","blocked"}}});
}
void summary() {
    const auto summary = request_summary(measured_rows());
    require(summary["requests"] == 4 && summary["completed"] == 2 && summary["failed"] == 1 && summary["incomplete"] == 1,
            "summary denominator lost failed or blocked requests");
    near(summary["deadline_attainment"], .5);
    near(summary["latency_p50_ms"], 20); near(summary["latency_p95_ms"], 29);
    near(summary["communication_p50_ms"], 2.5); near(summary["communication_p99_ms"], 3.97);
    near(summary["qualified_goodput_rps"], 10); // Failed tail remains part of observed run time.
    require(summary["qualification_status"]=="partial" && summary["qualification_verified_requests"]==1,
            "partial qualification was presented as complete evidence");
    auto numerical_only=measured_rows();numerical_only[0].erase("qualification");
    const auto unverified=request_summary(numerical_only);
    require(unverified["qualified_goodput_rps"].is_null() && unverified["qualification_status"]=="unverified",
            "numerical quality.passed was treated as full qualification");
    numerical_only[0]["qualification"]={{"qualified",false},{"status","failed"}};
    near(request_summary(numerical_only)["qualified_goodput_rps"],0);
    numerical_only[0]["qualification"]={{"qualified",false},{"status","unverified"}};
    require(request_summary(numerical_only)["qualified_goodput_rps"].is_null(),"unverified false invented measured zero goodput");
    near(summary["model_ms"], 26); near(summary["control_ms"], 3);
    const auto empty = request_summary(Json::array());
    require(empty["deadline_attainment"].is_null() && empty["latency_p95_ms"].is_null(), "empty metrics invented a measured value");
}
void invalid_metrics() {
    auto rows = measured_rows(); rows[0]["finish_ms"] = -1;
    rejects([&] { request_summary(rows); });
    rows = measured_rows(); rows[0]["arrival_ms"] = 11;
    rejects([&] { request_summary(rows); });
    rows = measured_rows(); rows[1]["request_id"] = rows[0]["request_id"];
    rejects([&] { request_summary(rows); });
}
void multiple_observations() {
    auto first = measured_rows(), second = measured_rows();
    for (auto& row:first) { row["observation_id"]="online/1/0"; row["observation_ms"]=120; }
    for (auto& row:second) { row["observation_id"]="online/1/1"; row["observation_ms"]=80; }
    Json combined=first;
    combined.insert(combined.end(),second.begin(),second.end());
    const auto summary=request_summary(combined);
    require(summary["requests"]==8 && summary["completed"]==4 && summary["failed"]==2 && summary["incomplete"]==2,
            "observation aggregation lost requests");
    require(summary["observation_count"]==2,"separate cells were merged into one timing window");
    near(summary["observation_ms"],220); // First explicit drain 120; second failed tail 100.
    near(summary["qualified_goodput_rps"],2000./220.);
    near(summary["deadline_attainment"],.5);
    combined.push_back(combined.front());
    rejects([&]{request_summary(combined);});
}
void workloads() {
    Json phase = {{"worker","model"},{"generator",{{"seed",7},{"count",8},{"rate_rps",2.0},{"service_ms",10},{"slo_ms",25},
                                                    {"prompts",Json::array({"first","second"})},
                                                    {"specs",Json::array({{{"steps",2}},{{"steps",3}}})}}}};
    Json content;
    for (const auto& pattern : {"offline_all","steady","poisson","gamma","bursty"}) {
        phase["generator"]["pattern"] = pattern;
        const auto rows = study_workload(phase);
        require(rows == study_workload(phase) && rows.size() == 8, "workload generation is not deterministic");
        Json specs = Json::array(); double last = -1;
        for (const auto& row : rows) {
            const double arrival = row["arrival_ms"];
            require(arrival >= last, "arrivals went backwards"); last = arrival;
            near(row["deadline_ms"], arrival + 25);
            require(row["spec"]["request_id"] == row["request_id"], "model request identity differs from trace");
            specs.push_back(row["spec"]);
            if (std::string(pattern) == "offline_all") near(arrival, 0);
        }
        if (content.is_null()) content = specs;
        require(specs == content, "arrival pattern changed paired request content");
    }
    phase["generator"]["pattern"] = "steady";
    const auto slow = study_workload(phase);
    phase["generator"]["rate_rps"] = 4;
    const auto fast = study_workload(phase);
    near(slow[3]["arrival_ms"], 1500); near(fast[3]["arrival_ms"], 750);
    require(slow[3]["spec"] == fast[3]["spec"], "load sweep changed paired content");
    phase["generator"]["count"] = 0; require(study_workload(phase).empty(), "zero-count workload invented requests");
    const Json causal = {{"workload",Json::array({{{"request_id","first"}},{{"request_id","second"},{"after","first"},{"think_ms",17}}})}};
    require(study_workload(causal)[1]["think_ms"] == 17, "causal think time lost");
    rejects([] { study_workload({{"workload",Json::array({{{"request_id","same"}},{{"request_id","same"}}})}}); });
    rejects([] { study_workload({{"workload",Json::array({{{"request_id","a"},{"after","b"}},{{"request_id","b"},{"after","a"}}})}}); });
    rejects([] { study_workload({{"workload",Json::array({{{"arrival_ms",10},{"deadline_ms",9}}})}}); });
    rejects([] { study_workload({{"generator",{{"pattern","unsupported"}}}}); });
}
}

int main() {
    try { golden(); } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
    check("benchmark policies", benchmark);
    check("matrix", matrix);
    check("gates", gates);
    check("summary", summary);
    check("invalid metrics", invalid_metrics);
    check("multiple observations", multiple_observations);
    check("workloads", workloads);
    std::cout << checks - failures << '/' << checks << " control contracts passed\n";
    return failures ? 1 : 0;
}
