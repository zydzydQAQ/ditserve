#include "ditserve/control/environment.hpp"
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace ditserve;
namespace {
int failures = 0, checks = 0;
void require(bool passed, const std::string& message) { if (!passed) throw std::runtime_error(message); }
void rejects(const std::function<void()>& action) {
    try { action(); } catch (const std::exception&) { return; }
    throw std::runtime_error("invalid input unexpectedly accepted");
}
void check(const std::string& name, const std::function<void()>& action) {
    ++checks;
    try { action(); }
    catch (const std::exception& error) { ++failures; std::cerr << name << ": " << error.what() << '\n'; }
}

const std::string A100 = "GPU-3ab2fedc-ccc1-007d-f4a1-b6058b3479a3", V100 = "GPU-a058066e-bde9-c534-efb1-d2a68c117681";
const std::string R0 = "GPU-00000000-0000-0000-0000-000000004090", R1 = "GPU-11111111-1111-1111-1111-111111114090";
Json gpu(int index, const std::string& uuid, const std::string& name, double memory) {
    return {{"index", index}, {"uuid", uuid}, {"name", name}, {"memory_mib", memory}};
}
Json local() { return Json::array({gpu(0, A100, "NVIDIA A100-PCIE-40GB", 40960), gpu(1, V100, "Tesla V100-PCIE-32GB", 32768)}); }
Json pair() { return Json::array({gpu(0, R0, "NVIDIA GeForce RTX 4090", 24564), gpu(1, R1, "NVIDIA GeForce RTX 4090", 24564)}); }
Json a100_env() {
    return {{"schema", "ditserve.env.v1"}, {"name", "a100-1x40g"},
            {"gpus", {{"count", 1}, {"uuids", {A100}}, {"forbid_uuids", {V100}}, {"name_contains", "A100"}, {"min_memory_mib", 40000}}}};
}
Json rtx_env() {
    return {{"schema", "ditserve.env.v1"}, {"name", "rtx4090-2x24g"},
            {"gpus", {{"count", 2}, {"name_contains", "RTX 4090"}, {"min_memory_mib", 23000}}},
            {"models_root", "/models"},
            {"models", {{"wan", {{"path", "wan2.1-t2v-1.3b"}, {"revision", "r1"}}}, {"sana", {{"path", "/abs/sana"}}}}},
            {"worker_environment", {{"OMP_NUM_THREADS", "8"}, {"NCCL_P2P_DISABLE", "1"}}},
            {"backends", {{"wan", {{"attention", "flash"}, {"memory_fraction", 0.6}}}}},
            {"cost_profile", {{"provenance", "measured"}, {"classes", Json::object()}}},
            {"serving_classes", Json::array({{{"class", "v"}, {"kind", "video"}}})}};
}
Json study() {
    return {{"schema", "ditserve.study.v1"}, {"experiment", "serving"},
            {"workers", {{"wan", {{"backend", "wan"}, {"per_gpu", true}, {"model", "wan"},
                                  {"config", {{"options", {{"execution", "block"}}}, {"memory_fraction", 0.5}}},
                                  {"environment", {{"OMP_NUM_THREADS", "4"}}}}},
                         {"sana", {{"backend", "sana"}, {"per_gpu", true}, {"model", "sana"}}},
                         {"exporter", {{"backend", "wan"}, {"gpu", 1}, {"model", "wan"}}},
                         {"fixture", {{"backend", "mock"}, {"per_gpu", true}}}}},
            {"phases", Json::array({
                {{"id", "serve"}, {"kind", "cluster"}, {"gpus", {{"video", "wan"}, {"image", "sana"}}},
                 {"profile", "@env.cost_profile"}, {"generator", {{"classes", "@env.serving_classes"}}},
                 {"warmup", Json::array({{{"worker", "wan"}}})},
                 {"matrix", {{"/generator/classes", Json::array({"@env.serving_classes"})}}}},
                {{"id", "one"}, {"kind", "cluster"}, {"gpu_count", 1}, {"gpus", {{"video", "wan"}, {"image", "sana"}}}, {"profile", "@env.cost_profile"}},
                {{"id", "numeric"}, {"kind", "operations"}, {"workers", Json::array({"wan"})}, {"actions", Json::array({{{"op", "hello"}}})}},
                {{"id", "cross"}, {"kind", "operations"}, {"min_gpus", 4}, {"workers", Json::array({"wan"})}, {"actions", Json::array({{{"op", "hello"}}})}},
                {{"id", "after"}, {"kind", "operations"}, {"requires", Json::array({"cross"})}, {"worker", "exporter"}, {"actions", Json::array({{{"op", "hello"}}})}}})}};
}
}

int main() {
    check("the local A100 is selected and the V100 never is", [] {
        auto selected = select_gpus(a100_env(), local());
        require(selected.size() == 1 && selected[0].at("uuid") == A100, "pinned A100");
        Json swapped = Json::array({gpu(0, V100, "Tesla V100-PCIE-32GB", 32768), gpu(1, A100, "NVIDIA A100-PCIE-40GB", 40960)});
        require(select_gpus(a100_env(), swapped)[0].at("uuid") == A100, "pin survives index order");
        Json unpinned = a100_env(); unpinned["gpus"].erase("uuids");
        require(select_gpus(unpinned, swapped)[0].at("uuid") == A100, "name filter skips the V100");
        rejects([] { select_gpus(a100_env(), Json::array({gpu(0, V100, "Tesla V100-PCIE-32GB", 32768)})); });
        Json forbidding = a100_env(); forbidding["gpus"]["uuids"] = Json::array({V100});
        rejects([&] { select_gpus(forbidding, local()); });
        Json two = unpinned; two["gpus"]["count"] = 2; two["gpus"].erase("name_contains"); two["gpus"]["min_memory_mib"] = 0;
        rejects([&] { select_gpus(two, local()); });  // the V100 stays forbidden
    });
    check("a 4090 pair needs both cards with enough memory", [] {
        require(select_gpus(rtx_env(), pair()).size() == 2, "two 4090s");
        rejects([] { select_gpus(rtx_env(), Json::array({pair()[0]})); });
        Json small = pair(); small[1]["memory_mib"] = 12000;
        rejects([&] { select_gpus(rtx_env(), small); });
        Json short_uuid = pair(); short_uuid[0]["uuid"] = "GPU-0";
        rejects([&] { select_gpus(rtx_env(), short_uuid); });
    });
    check("per-GPU workers, models and defaults", [] {
        const auto out = resolve_study(study(), rtx_env(), pair());
        const auto& workers = out.at("workers");
        require(workers.contains("wan.0") && workers.contains("wan.1") && !workers.contains("wan"), "per-GPU instances");
        require(workers.at("wan.1").at("gpu_uuid") == R1 && workers.at("exporter").at("gpu_uuid") == R1, "GPU binding");
        const auto& config = workers.at("wan.0").at("config");
        require(config.at("model_path") == "/models/wan2.1-t2v-1.3b" && config.at("revision") == "r1", "model under models_root");
        require(workers.at("sana.0").at("config").at("model_path") == "/abs/sana", "absolute model path kept");
        require(config.at("attention") == "flash" && config.at("memory_fraction") == 0.5 && config.at("device") == "cuda:0", "backend defaults, study wins");
        require(workers.at("wan.0").at("environment").at("OMP_NUM_THREADS") == "4" &&
                workers.at("wan.0").at("environment").at("NCCL_P2P_DISABLE") == "1", "environment defaults, study wins");
        require(!workers.at("fixture.0").contains("gpu_uuid"), "mock workers stay off the GPU");
        require(workers.at("wan.0").at("instance_of") == "wan", "instances name their template");
    });
    check("phases expand to the selected GPUs and skip what does not fit", [] {
        const auto out = resolve_study(study(), rtx_env(), pair());
        const auto& phases = out.at("phases");
        require(phases.size() == 3, "cross and its dependent are skipped");
        require(phases[0].at("gpus").size() == 2 && phases[0].at("gpus")[1] == Json({{"video", "wan.1"}, {"image", "sana.1"}}), "one slot per GPU");
        require(phases[0].at("warmup").size() == 2 && phases[0].at("warmup")[1].at("worker") == "wan.1", "warmup per instance");
        require(phases[0].at("profile").at("provenance") == "measured", "profile substituted");
        require(phases[0].at("matrix").at("/generator/classes")[0][0].at("class") == "v", "substitution inside a matrix");
        require(phases[1].at("gpus").size() == 1, "gpu_count takes the first GPUs");
        require(phases[2].at("workers") == Json::array({"wan.0", "wan.1"}), "operations ranks per GPU");
        const auto& skipped = out.at("environment").at("skipped_phases");
        require(skipped.size() == 2 && skipped[1].at("id") == "after", "dependency cascade recorded");
        require(out.at("experiment") == "rtx4090-2x24g-serving", "experiment carries the environment");
        require(out.at("environment").at("selected") == Json::array({R0, R1}), "selection recorded");
        rejects([&] { resolve_study(out, rtx_env(), pair()); });
    });
    check("bad references fail before any run", [] {
        Json missing = rtx_env(); missing["cost_profile"] = nullptr;
        rejects([&] { resolve_study(study(), missing, pair()); });
        Json unknown = study(); unknown["workers"]["wan"]["model"] = "flux";
        rejects([&] { resolve_study(unknown, rtx_env(), pair()); });
        Json far = study(); far["workers"]["exporter"]["gpu"] = 2;
        const auto partial = resolve_study(far, rtx_env(), pair());
        require(!partial.at("workers").contains("exporter") && partial.at("environment").at("unplaced_workers").size() == 1,
                "a worker beyond the selection is left out and recorded");
        Json unsafe = rtx_env(); unsafe["worker_environment"]["CUDA_VISIBLE_DEVICES"] = "0";
        rejects([&] { resolve_study(study(), unsafe, pair()); });
        Json typo = rtx_env(); typo["gpu"] = 1;
        rejects([&] { resolve_study(study(), typo, pair()); });
        rejects([&] { resolve_study(study(), rtx_env(), Json()); });  // GPU workers need an inventory
    });
    check("simulations take only substitutions", [] {
        const Json what_if = {{"experiment", "what-if"}, {"profile", "@env.cost_profile"}, {"gpus", 8}};
        const auto out = resolve_study(what_if, rtx_env(), Json());
        require(out.at("profile").at("provenance") == "measured" && out.at("experiment") == "rtx4090-2x24g-what-if", "substituted");
        require(out.at("environment").at("inventory").is_null(), "no inventory read");
    });
    std::cout << checks - failures << "/" << checks << " environment checks passed\n";
    return failures ? 1 : 0;
}
