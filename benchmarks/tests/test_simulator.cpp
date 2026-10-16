#include "ditserve/benchmark/simulator.hpp"
#include "ditserve/benchmark/metrics.hpp"
#include "ditserve/benchmark/policy.hpp"
#include "ditserve/benchmark/workload.hpp"
#include "ditserve/runtime/random.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <stdexcept>

using namespace ditserve;

namespace {
void require(bool condition, const std::string& message) { if (!condition) throw std::runtime_error(message); }
void fails(const std::function<void()>& operation) {
    try { operation(); } catch (const std::exception&) { return; }
    throw std::runtime_error("invalid input unexpectedly accepted");
}
void equivalent(const Json& actual, const Json& expected, const std::string& path = "root") {
    if (actual.is_number() && expected.is_number()) {
        const double a = actual.get<double>(), b = expected.get<double>();
        require(std::abs(a - b) <= 1e-12 * std::max({1.0, std::abs(a), std::abs(b)}), path + ": numerical mismatch");
    } else if (actual.is_object() && expected.is_object()) {
        require(actual.size() == expected.size(), path + ": object size mismatch");
        for (auto it = expected.begin(); it != expected.end(); ++it) equivalent(actual.at(it.key()), it.value(), path + "." + it.key());
    } else if (actual.is_array() && expected.is_array()) {
        require(actual.size() == expected.size(), path + ": array size mismatch");
        for (std::size_t i = 0; i < actual.size(); ++i) equivalent(actual[i], expected[i], path + "[" + std::to_string(i) + "]");
    } else require(actual == expected, path + ": value mismatch");
}
RequestSpec spec(const std::string& id = "a", std::int64_t seed = 0, double threshold = 0.15, int steps = 20) {
    RequestSpec value; value.request_id = id; value.deadline_ms = 5000; value.seed = seed;
    value.cache_threshold = threshold; value.steps = steps; return value;
}
std::unique_ptr<State> finish(const Backend& backend, std::unique_ptr<State> state) {
    while (!state->is_complete()) backend.step_batch({state.get()});
    return state;
}

void checkpoints_and_batches() {
    const ReferenceBackend backend(7);
    for (const auto seed : {std::int64_t(0), std::int64_t(17), std::int64_t(-3)}) for (const double threshold : {0.0, 0.15, 1.0}) {
        const auto request = spec("checkpoint", seed, threshold);
        const auto expected = finish(backend, backend.prepare(request));
        auto state = backend.prepare(request);
        while (!state->is_complete()) {
            const auto before = backend.fingerprint(*state), path = backend.peek_path(*state);
            require(path == backend.peek_path(*state) && before == backend.fingerprint(*state), "peek mutated state");
            backend.step_batch({state.get()});
            require(state->path_history().back() == path, "peek differed from execution");
            const auto encoded = Json::parse(state->snapshot().dump());
            auto restored = backend.restore(encoded);
            require(backend.fingerprint(*state) == backend.fingerprint(*restored), "JSON roundtrip changed checkpoint");
            state = std::move(restored);
        }
        require(backend.fingerprint(*expected) == backend.fingerprint(*state), "pause/restore changed result");
    }
    std::vector<std::unique_ptr<State>> states, isolated;
    for (const auto& request : {spec("a", 11, 0, 13), spec("b", 31, 0.25, 20), spec("c", 53, 1, 9)}) {
        states.push_back(backend.prepare(request)); isolated.push_back(finish(backend, backend.prepare(request)));
    }
    bool mixed = false;
    for (;;) {
        std::vector<State*> active;
        for (auto it = states.rbegin(); it != states.rend(); ++it) if (!(*it)->is_complete()) active.push_back(it->get());
        if (active.empty()) break;
        mixed |= std::any_of(active.begin(), active.end(), [&](auto* state) { return backend.peek_path(*state) != backend.peek_path(*active.front()); });
        backend.step_batch(active);
    }
    require(mixed, "fixture did not exercise mixed compute/reuse");
    for (std::size_t i = 0; i < states.size(); ++i) require(backend.fingerprint(*states[i]) == backend.fingerprint(*isolated[i]), "batch cross-contaminated request");
    auto active = backend.prepare(spec()), completed = finish(backend, backend.prepare(spec("complete", 0, 0, 1)));
    const auto before = backend.fingerprint(*active);
    fails([&] { backend.step_batch({active.get(), completed.get()}); });
    fails([&] { backend.step_batch({active.get(), active.get()}); });
    fails([&] { backend.step_batch({active.get(), nullptr}); });
    require(before == backend.fingerprint(*active), "invalid batch partially advanced state");
    backend.step_batch({}); backend.step_batch({active.get()});
    auto snapshot = active->snapshot();
    auto restored = backend.restore(snapshot);
    const auto valid = backend.fingerprint(*active);
    snapshot["latents"][0] = 123; backend.step_batch({restored.get()});
    require(valid == backend.fingerprint(*active), "snapshot or restored state retained mutable aliases");
    snapshot = active->snapshot(); snapshot["version"] = 1;
    fails([&] { backend.restore(snapshot); });
    snapshot = active->snapshot(); snapshot["random_state"]["engine"] = "invalid";
    fails([&] { backend.restore(snapshot); });
    snapshot = active->snapshot(); snapshot["previous_probe"] = Json::array({1});
    fails([&] { backend.restore(snapshot); });
    snapshot = active->snapshot(); snapshot["path_history"] = Json::array();
    fails([&] { backend.restore(snapshot); });
    require(!active->snapshot().at("random_state").at("spare").is_null(), "odd latent width must exercise Gaussian spare");
    for (const int steps : {1, 2, 20}) {
        auto enabled = spec("fixed", 0, 0, steps), disabled = enabled; disabled.cache_enabled = false;
        const auto cached = finish(backend, backend.prepare(enabled)), uncached = finish(backend, backend.prepare(disabled));
        require(cached->snapshot()["latents"] == uncached->snapshot()["latents"] && uncached->snapshot()["cache_residual"].is_null(), "disabled cache differs from forced compute");
        require(std::all_of(cached->path_history().begin(), cached->path_history().end(), [](const auto& p) { return p == "compute"; }), "zero threshold reused cache");
    }
}

void workload_and_random() {
    Random rng(0);
    require(rng.uniform() == 0.15979336337046079, "versioned uniform sequence changed");
    equivalent(rng.normal(), 3.0176729926679866, "versioned normal first sample");
    equivalent(rng.normal(), 0.76610253847667775, "versioned normal spare");
    require(rng.bounded(7) == 6, "versioned bounded sequence changed");
    rng.normal(); auto restored = Random::restore(Json::parse(rng.snapshot().dump()));
    for (int i = 0; i < 10; ++i) require(rng.normal() == restored.normal() && rng.bounded(7) == restored.bounded(7), "RNG stream changed after restore");
    const CostModel cost;
    const Json options{{"count", 20}, {"seed", -3}, {"arrival_rate", 8}, {"pattern", "bursty"}};
    auto first_workload = generate_workload(options, cost), repeated = generate_workload(options, cost);
    const std::vector<double> golden_arrivals{4.876270674946317, 8.601885332167136, 12.604274148348349};
    const std::vector<std::int64_t> golden_seeds{444629137, 1589610063, 1863921828};
    for (std::size_t i = 0; i < golden_arrivals.size(); ++i) {
        equivalent(first_workload[i].arrival_ms, golden_arrivals[i], "versioned arrivals");
        require(first_workload[i].seed == golden_seeds[i], "versioned request seeds changed");
    }
    auto different_shapes = options; different_shapes["shapes"] = Json::array({Json::array({768, 512, 1})});
    const auto reshaped = generate_workload(different_shapes, cost);
    for (std::size_t i = 0; i < first_workload.size(); ++i) {
        require(first_workload[i].to_json() == repeated[i].to_json(), "workload not reproducible");
        require(first_workload[i].arrival_ms == reshaped[i].arrival_ms, "shape stream changed arrivals");
        require(std::fmod(first_workload[i].arrival_ms, 1000) < 200 + 1e-9, "bursty arrival outside active interval");
    }
    for (const Json& bad : {Json{{"count", true}}, Json{{"arrival_rate", 0}}, Json{{"shapes", Json::array()}}, Json{{"burst_active_fraction", 2}}})
        fails([&] { generate_workload(bad, cost); });
    fails([&] { RequestSpec::from_json(Json{{"request_id", "x"}, {"arrival_ms", 0}, {"deadline_ms", 1}, {"seed", true}}); });
    fails([&] { RuntimeConfig::from_json(Json{{"num_gpus", true}}); });
    fails([&] { RuntimeConfig::from_json(Json{{"policy", "unknown"}}); });
    fails([&] { CostModel::from_json(Json{{"provenance", "measured"}}); });
    fails([&] { Simulator().run({spec(), spec()}); });

    // A v2 fixture guards this implementation's own numerical contract. It does
    // not assert equivalence with the retired Python RNG or cache trajectory.
    const ReferenceBackend backend;
    const auto state = finish(backend, backend.prepare(spec("golden", 17, 0.25, 9)));
    require(backend.fingerprint(*state) == "2b1c4f1e3b9dd1f08147c8be20ef959da03c1df5b691b2a01cc0e3ae0fdbb987", "v2 reference fingerprint changed");
    require(state->path_history() == std::vector<std::string>{"compute", "reuse", "reuse", "reuse", "compute", "reuse", "reuse", "reuse", "compute"}, "v2 cache trajectory changed");
}

void scheduler_invariants() {
    std::vector<RequestSpec> specs;
    for (int i = 0; i < 10; ++i) { auto request = spec(std::to_string(i), i, 0.15, 15); request.arrival_ms = i * 3; specs.push_back(request); }
    std::map<std::string, std::string> reference;
    for (const auto& policy : PolicyRegistry::instance().names()) {
        RuntimeConfig config; config.policy = policy; config.max_batch = 3; config.memory_mb = 144;
        Simulator simulator(config);
        const auto run = simulator.run(specs);
        require(run["summary"]["completed"] == 10 && run["summary"]["parity_failures"] == 0, "policy failed state parity");
        require(run == simulator.run(specs), "reused Simulator retained prior run state");
        if (policy != "batch_only") for (const auto& record : run["records"]) {
            const auto id = record["request_id"].get<std::string>(), hash = record["fingerprint"].get<std::string>();
            if (reference.count(id)) require(reference[id] == hash, "policy changed cache trajectory");
            else reference[id] = hash;
        }
    }
    RuntimeConfig config; config.policy = "edf"; config.num_gpus = 2; config.max_batch = 1; config.memory_mb = 72;
    CostModel cost; cost.transfer_gbps = 1;
    auto a = spec("A"), b = spec("B", 0, 0.15, 1), c = spec("C", 0, 0.15, 1);
    a.deadline_ms = 500; b.arrival_ms = 1; b.deadline_ms = 50; c.arrival_ms = 2; c.deadline_ms = 30;
    const auto offload = Simulator(config, cost).run({a, b, c});
    std::map<std::string, double> ready;
    bool restored = false;
    for (const auto& event : offload["events"]) {
        if (event["event"] == "offload") ready[event["request_id"].get<std::string>()] = event["ready_ms"].get<double>();
        if (event["event"] == "restore") { restored = true; require(event["time_ms"].get<double>() + 1e-8 >= ready.at(event["request_id"].get<std::string>()), "restored before outbound transfer completed"); }
    }
    require(restored, "offload fixture did not restore");
    const auto workloads = generate_workload(Json{{"count", 32}, {"seed", 4}, {"arrival_rate", 100}, {"slo_scale", 20},
                                                  {"shapes", Json::array({Json::array({512,512,1}), Json::array({768,768,1})})}}, CostModel());
    for (const auto& policy : {"edf", "path_aware", "state_aware"}) {
        config.policy = policy; config.num_gpus = 4; config.max_batch = 4; config.memory_mb = 180; config.host_memory_mb = 60;
        const auto run = Simulator(config).run(workloads); const auto& summary = run["summary"];
        for (const auto& peak : summary["peak_device_state_and_workspace_mb"]) require(peak.get<double>() <= 180 + 1e-9, "GPU budget overflow");
        require(summary["peak_host_state_mb"].get<double>() <= 60 + 1e-9 && summary["resource_utilization"].get<double>() <= 1 + 1e-9, "host/capacity budget overflow");
        double charged = 0; for (const auto& record : run["records"]) charged += record["gpu_ms"].get<double>();
        equivalent(charged, summary["gpu_busy_ms"]);
        require(summary["total_arrivals"] == workloads.size() && summary["parity_failures"] == 0, "request lost or state changed");
    }
    config = RuntimeConfig(); config.verify_outputs = false;
    require(Simulator(config).run({spec()})["summary"]["slo_met"] == 0, "unknown parity counted as success");
}

// Custom policies register without touching the runtime or the built-ins.
class LatestFirstPolicy : public Policy {
public:
    PriorityKey priority(const PriorityInput& input) const override { return {-input.spec->arrival_ms, input.spec->request_id}; }
};

// An injected backend keeps ownership of its states; the simulator never needs
// to know the concrete reference state or bypass the Backend contract.
class ForwardBackend final : public Backend {
public:
    mutable int batches = 0, restores = 0;
    std::unique_ptr<State> prepare(const RequestSpec& request) const override { return reference_.prepare(request); }
    std::string peek_path(const State& state) const override { return reference_.peek_path(state); }
    void step_batch(const std::vector<State*>& states) const override { ++batches; reference_.step_batch(states); }
    std::unique_ptr<State> restore(const Json& snapshot) const override { ++restores; return reference_.restore(snapshot); }
    std::string fingerprint(const State& state) const override { return reference_.fingerprint(state); }
private:
    ReferenceBackend reference_;
};
class ForeignState final : public State {
public:
    bool is_complete() const override { return false; }
    int step_index() const override { return 0; }
    const std::vector<std::string>& path_history() const override { return history_; }
    Json snapshot() const override { return Json::object(); }
private:
    std::vector<std::string> history_;
};
void backend_extension() {
    const auto backend = std::make_shared<ForwardBackend>();
    RuntimeConfig config; config.policy="edf"; config.max_batch=1; config.memory_mb=72;
    auto long_request=spec("long"), urgent=spec("urgent",1,0.15,1);
    long_request.deadline_ms=500;urgent.arrival_ms=1;urgent.deadline_ms=50;
    const auto run=Simulator(config,CostModel(),backend).run({long_request,urgent});
    require(run["summary"]["completed"]==2 && run["summary"]["parity_failures"]==0 && backend->batches>0 && backend->restores>0,
            "injected backend did not execute and restore through its public contract");
    ReferenceBackend reference; ForeignState foreign;
    auto valid=reference.prepare(spec());const auto before=valid->snapshot();
    fails([&]{reference.peek_path(foreign);});
    fails([&]{reference.fingerprint(foreign);});
    fails([&]{reference.step_batch({valid.get(),&foreign});});
    require(valid->snapshot()==before,"foreign state rejection partially executed a batch");
}
void policy_extension() {
    PolicyRegistry::instance().add("latest_first", [] { return std::make_unique<LatestFirstPolicy>(); });
    fails([&] { PolicyRegistry::instance().add("latest_first", [] { return std::make_unique<LatestFirstPolicy>(); }); });
    RuntimeConfig config; config.policy = "latest_first";
    const auto run = Simulator(config).run({spec("x"), spec("y", 1)});
    require(run["summary"]["completed"] == 2 && run["summary"]["parity_failures"] == 0, "custom policy run failed");
    require(run["records"].front()["request_id"] == "x" || run["records"].front()["request_id"] == "y", "custom policy produced unknown records");
}

void legacy_cost_and_metrics() {
    std::ifstream file(BASELINE_PATH); require(file.good(), "missing legacy golden fixture");
    const auto baseline = Json::parse(file); const CostModel cost;
    for (const auto& item : baseline.at("cost")) {
        const auto request = RequestSpec::from_json(item.at("spec"));
        Json actual{{"spec", request.to_json()}, {"scale", cost.scale(request)}, {"state_initial", cost.state_mb(request, false)},
                    {"state_full", cost.state_mb(request)}, {"workspace", cost.workspace(request, 3)}, {"transfer", cost.transfer_ms(12)},
                    {"compute", cost.step_ms(request, "compute")}, {"reuse", cost.step_ms(request, "reuse")},
                    {"mixed_batch", cost.batch_ms({request, request}, {"compute", "reuse"}, false)},
                    {"split_batch", cost.batch_ms({request, request}, {"compute", "reuse"}, true)}, {"full_service", cost.full_service_ms(request)}};
        equivalent(actual, item, "cost");
    }
    for (const auto& item : baseline.at("metrics"))
        equivalent(summarize(item.at("records"), item.at("start_ms"), item.at("end_ms"), item.at("gpu_busy_ms"), item.at("num_gpus")), item.at("expected"), "metrics");
}
}

int main() {
    try {
        checkpoints_and_batches(); workload_and_random(); scheduler_invariants(); legacy_cost_and_metrics(); policy_extension(); backend_extension();
        std::cout << "simulation contracts passed\n"; return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
