#include "ditserve/policy.hpp"
#include "ditserve/policies.hpp"
#include <stdexcept>

namespace ditserve {
PolicyRegistry& PolicyRegistry::instance() {
    static PolicyRegistry registry = [] {
        PolicyRegistry built;
        register_builtin_policies(built);
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
