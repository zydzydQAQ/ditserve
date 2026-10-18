#include "ditserve/control/environment.hpp"
#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

namespace ditserve {
namespace {
std::string instance(const std::string& base, int index) { return base + "." + std::to_string(index); }

Json substitute(const Json& value, const Json& env) {
    if (value.is_string()) {
        const auto text = value.get<std::string>();
        if (text.rfind("@env.", 0) != 0) return value;
        const auto key = text.substr(5);
        if (!env.contains(key) || env.at(key).is_null())
            throw std::invalid_argument("environment " + env.value("name", std::string()) + " has no " + key +
                                        (key == "cost_profile" ? "; measure it with the cost_profile study first" : ""));
        return env.at(key);
    }
    if (value.is_object()) {
        Json result = Json::object();
        for (auto it = value.begin(); it != value.end(); ++it) result[it.key()] = substitute(it.value(), env);
        return result;
    }
    if (value.is_array()) {
        Json result = Json::array();
        for (const auto& item : value) result.push_back(substitute(item, env));
        return result;
    }
    return value;
}

void check_environment(const Json& env) {
    if (!env.is_object() || env.value("schema", "") != "ditserve.env.v1") throw std::invalid_argument("environment schema must be ditserve.env.v1");
    check_keys(env, {"schema", "name", "description", "gpus", "models_root", "models", "worker_environment", "backends",
                     "cost_profile", "links", "serving_classes", "host", "provenance"});
    path_segment(env.at("name").get<std::string>(), "environment name");
    for (const auto* key : {"models", "worker_environment", "backends", "host"})
        if (env.contains(key) && !env.at(key).is_object()) throw std::invalid_argument(std::string("environment ") + key + " must be an object");
    const Json variables = env.value("worker_environment", Json::object());
    for (auto it = variables.begin(); it != variables.end(); ++it) {
        if (it.key() == "CUDA_VISIBLE_DEVICES" || it.key() == "PYTHONPATH" || it.key() == "HF_HUB_OFFLINE" || it.key() == "TRANSFORMERS_OFFLINE")
            throw std::invalid_argument("environment cannot override device, frozen sources or offline loading");
        it.value().get<std::string>();
    }
}
}

Json select_gpus(const Json& env, const Json& inventory) {
    const auto& spec = env.at("gpus");
    check_keys(spec, {"count", "uuids", "forbid_uuids", "name_contains", "min_memory_mib"});
    const int count = int_field(spec, "count", 1, 1);
    std::set<std::string> forbidden;
    for (const auto& uuid : spec.value("forbid_uuids", Json::array())) forbidden.insert(uuid.get<std::string>());
    const auto name = spec.value("name_contains", std::string());
    const double memory = number_field(spec, "min_memory_mib");
    if (!inventory.is_array()) throw std::invalid_argument("GPU inventory must be an array");
    auto problem = [&](const Json& row) -> std::string {
        const auto uuid = row.at("uuid").get<std::string>();
        if (forbidden.count(uuid)) return "is forbidden by the environment";
        if (row.at("name").get<std::string>().find(name) == std::string::npos) return "is not a " + name;
        if (number_field(row, "memory_mib") < memory) return "has less than " + std::to_string(static_cast<int>(memory)) + " MiB";
        return "";
    };
    Json selected = Json::array();
    if (spec.contains("uuids")) {
        for (const auto& uuid : spec.at("uuids")) {
            const auto found = std::find_if(inventory.begin(), inventory.end(), [&](const Json& row) { return row.at("uuid") == uuid; });
            if (found == inventory.end()) throw std::invalid_argument("pinned GPU " + uuid.get<std::string>() + " is not present");
            const auto reason = problem(*found);
            if (!reason.empty()) throw std::invalid_argument("pinned GPU " + uuid.get<std::string>() + " " + reason);
            selected.push_back(*found);
        }
    } else {
        std::vector<Json> rows(inventory.begin(), inventory.end());
        std::sort(rows.begin(), rows.end(), [](const Json& a, const Json& b) { return integer(a.at("index"), "index") < integer(b.at("index"), "index"); });
        for (const auto& row : rows)
            if (static_cast<int>(selected.size()) < count && problem(row).empty()) selected.push_back(row);
    }
    std::set<std::string> unique;
    for (const auto& row : selected) {
        const auto uuid = row.at("uuid").get<std::string>();
        if (uuid.compare(0, 4, "GPU-") != 0 || uuid.size() != 40) throw std::invalid_argument("GPU inventory requires full device UUIDs");
        if (!unique.insert(uuid).second) throw std::invalid_argument("environment selects one GPU twice");
    }
    if (static_cast<int>(selected.size()) != count)
        throw std::invalid_argument("environment " + env.value("name", std::string()) + " needs " + std::to_string(count) +
                                    " matching GPUs, found " + std::to_string(selected.size()));
    return selected;
}

Json resolve_study(const Json& study, const Json& env, const Json& inventory) {
    check_environment(env);
    if (study.contains("environment")) throw std::invalid_argument("study is already bound to an environment");
    const auto env_name = env.at("name").get<std::string>();
    const Json selected = inventory.is_null() ? Json::array() : select_gpus(env, inventory);
    const int gpus = static_cast<int>(selected.size());
    Json out = substitute(study, env);

    std::map<std::string, std::string> bases;  // per_gpu base -> backend
    Json bindings = Json::object(), models = Json::object();
    auto bind = [&](Json worker, const std::string& name, int gpu) {
        const auto backend = worker.at("backend").get<std::string>();
        const bool bound = gpu >= 0 && backend != "mock";
        if (bound) {
            if (gpu >= gpus) throw std::invalid_argument("worker " + name + " needs GPU " + std::to_string(gpu) + " of " + std::to_string(gpus));
            worker["gpu_uuid"] = selected[gpu].at("uuid");
            bindings[name] = {{"gpu", gpu}, {"uuid", selected[gpu].at("uuid")}, {"name", selected[gpu].at("name")}};
        }
        Json config = env.value("backends", Json::object()).value(backend, Json::object());
        config.merge_patch(worker.value("config", Json::object()));
        if (bound && !config.contains("device") && !worker.contains("device")) config["device"] = "cuda:0";
        if (worker.contains("model")) {
            const auto logical = worker.at("model").get<std::string>();
            const auto& entry = env.value("models", Json::object());
            if (!entry.contains(logical)) throw std::invalid_argument("environment " + env_name + " has no model " + logical);
            const auto& model = entry.at(logical);
            std::string path = model.at("path").get<std::string>();
            if (path.empty() || path.front() != '/') path = env.value("models_root", std::string("/models")) + "/" + path;
            config["model_path"] = path;
            for (const auto* key : {"model_id", "revision"}) if (model.contains(key)) config[key] = model.at(key);
            models[logical] = {{"path", path}, {"model_id", model.value("model_id", Json())}, {"revision", model.value("revision", Json())}};
            worker.erase("model");
        }
        worker["config"] = config;
        Json environment = env.value("worker_environment", Json::object());
        environment.update(worker.value("environment", Json::object()));
        if (!environment.empty()) worker["environment"] = environment;
        return worker;
    };
    Json workers = Json::object();
    const Json defined = out.value("workers", Json::object());
    for (auto it = defined.begin(); it != defined.end(); ++it) {
        Json worker = it.value();
        const auto name = path_segment(it.key(), "worker name");
        if (worker.value("per_gpu", false)) {
            if (inventory.is_null()) throw std::invalid_argument("per_gpu workers need a GPU inventory");
            worker.erase("per_gpu");
            worker["instance_of"] = name;
            bases[name] = worker.at("backend").get<std::string>();
            for (int gpu = 0; gpu < gpus; ++gpu) workers[instance(name, gpu)] = bind(worker, instance(name, gpu), gpu);
        } else if (worker.contains("gpu")) {
            if (inventory.is_null()) throw std::invalid_argument("GPU-bound workers need a GPU inventory");
            const int gpu = static_cast<int>(integer(worker.at("gpu"), "gpu"));
            if (gpu < 0) throw std::invalid_argument("gpu must be a nonnegative index into the selected GPUs");
            worker.erase("gpu");
            workers[name] = bind(worker, name, gpu);
        } else workers[name] = worker.contains("model") || worker.contains("environment") || worker.contains("config") ? bind(worker, name, -1) : worker;
    }
    if (out.contains("workers")) out["workers"] = workers;

    if (out.contains("phases")) {
        Json phases = Json::array(), skipped = Json::array();
        std::set<std::string> dropped;
        for (auto phase : out.at("phases")) {
            const auto id = phase.at("id").get<std::string>();
            std::string reason;
            const int width = phase.contains("gpu_count") ? static_cast<int>(integer(phase.at("gpu_count"), "gpu_count")) : gpus;
            if (int_field(phase, "min_gpus", 0, 0) > gpus) reason = "needs " + std::to_string(int_field(phase, "min_gpus", 0, 0)) + " GPUs";
            else if (width > gpus || width < 1) reason = "gpu_count does not fit the selected GPUs";
            for (const auto& dependency : phase.value("requires", Json::array())) {
                const auto parent = dependency.is_string() ? dependency.get<std::string>() : dependency.at("phase").get<std::string>();
                if (dropped.count(parent)) reason = "requires skipped phase " + parent;
            }
            if (!reason.empty()) { dropped.insert(id); skipped.push_back({{"id", id}, {"reason", reason}}); continue; }
            phase.erase("min_gpus"); phase.erase("gpu_count");
            auto expand = [&](const std::string& name) {
                Json names = Json::array();
                if (!bases.count(name)) names.push_back(name);
                else for (int gpu = 0; gpu < width; ++gpu) names.push_back(instance(name, gpu));
                return names;
            };
            if (phase.contains("gpus") && phase.at("gpus").is_object()) {
                const auto video = phase.at("gpus").at("video").get<std::string>(), image = phase.at("gpus").at("image").get<std::string>();
                if (!bases.count(video) || !bases.count(image)) throw std::invalid_argument("cluster slot templates must name per_gpu workers");
                Json slots = Json::array();
                for (int gpu = 0; gpu < width; ++gpu) slots.push_back({{"video", instance(video, gpu)}, {"image", instance(image, gpu)}});
                phase["gpus"] = slots;
            }
            if (phase.contains("warmup")) {
                Json warmup = Json::array();
                for (const auto& item : phase.at("warmup"))
                    for (const auto& name : expand(item.at("worker").get<std::string>())) { Json copy = item; copy["worker"] = name; warmup.push_back(copy); }
                phase["warmup"] = warmup;
            }
            if (phase.value("kind", "") == "operations") {
                Json names = Json::array();
                const auto listed = phase.contains("workers") ? phase.at("workers") : Json::array({phase.value("worker", "model")});
                for (const auto& name : listed) for (const auto& expanded : expand(name.get<std::string>())) names.push_back(expanded);
                phase.erase("worker");
                phase["workers"] = names;
            }
            phases.push_back(phase);
        }
        if (phases.empty()) throw std::invalid_argument("no phase of this study fits environment " + env_name);
        out["phases"] = phases;
        out["environment"]["skipped_phases"] = skipped;
    }
    out["experiment"] = env_name + "-" + out.at("experiment").get<std::string>();
    Json selected_uuids = Json::array();
    for (const auto& row : selected) selected_uuids.push_back(row.at("uuid"));
    out["environment"].update({{"schema", "ditserve.env-resolution.v1"}, {"name", env_name},
        {"provenance", env.value("provenance", Json())}, {"inventory", inventory}, {"selected", selected_uuids},
        {"workers", bindings}, {"models", models}});
    return out;
}
}
