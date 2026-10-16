#include "ditserve/benchmark/policy.hpp"
#include <stdexcept>

namespace ditserve {
namespace {
enum class Ranking { arrival, deadline, slack };
struct Builtin {
    const char* name;
    Ranking ranking;
    bool cache, preemption, path_groups, same_path, state_aware;
    int max_batch; // zero keeps the configured limit
};
constexpr Builtin builtins[] = {
    {"cache_only", Ranking::arrival, true, false, false, false, false, 1},
    {"batch_only", Ranking::arrival, false, false, false, false, false, 0},
    {"naive", Ranking::arrival, true, false, false, false, false, 0},
    {"edf", Ranking::deadline, true, true, false, false, false, 0},
    {"path_aware", Ranking::deadline, true, true, true, true, false, 0},
    {"state_aware", Ranking::slack, true, true, true, false, true, 0},
};

class BuiltinPolicy final : public Policy {
public:
    explicit BuiltinPolicy(Builtin config) : config_(config) {}
    PriorityKey priority(const PriorityInput& input) const override {
        const auto& spec = *input.spec;
        double key = spec.deadline_ms;
        if (config_.ranking == Ranking::arrival) key = spec.arrival_ms;
        else if (config_.ranking == Ranking::slack)
            key = spec.deadline_ms - input.now_ms - input.remaining_ms + input.restore_ms;
        return {key, spec.request_id};
    }
    bool uses_cache() const override { return config_.cache; }
    bool preemptive() const override { return config_.preemption; }
    bool group_paths() const override { return config_.path_groups; }
    bool same_path_only() const override { return config_.same_path; }
    bool state_aware() const override { return config_.state_aware; }
    int batch_limit(int requested) const override { return config_.max_batch ? config_.max_batch : requested; }
private:
    Builtin config_;
};
}

PolicyRegistry& PolicyRegistry::instance() {
    static PolicyRegistry registry = [] {
        PolicyRegistry built;
        for (const auto& config : builtins)
            built.add(config.name, [config] { return std::make_unique<BuiltinPolicy>(config); });
        return built;
    }();
    return registry;
}
void PolicyRegistry::add(const std::string& name, Factory factory) {
    if (name.empty() || !factory) throw std::invalid_argument("policy registration requires a name and factory");
    for (const auto& entry : factories_)
        if (entry.first == name) throw std::invalid_argument("duplicate policy: " + name);
    factories_.push_back({name, std::move(factory)});
}
std::unique_ptr<Policy> PolicyRegistry::create(const std::string& name) const {
    for (const auto& entry : factories_) if (entry.first == name) return entry.second();
    throw std::invalid_argument("unknown policy: " + name);
}
std::vector<std::string> PolicyRegistry::names() const {
    std::vector<std::string> result;
    for (const auto& entry : factories_) result.push_back(entry.first);
    return result;
}
}
