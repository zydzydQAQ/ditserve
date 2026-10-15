#pragma once

#include "ditserve/types.hpp"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ditserve {

// Everything the scheduler offers a policy for ranking one candidate.
struct PriorityInput {
    const RequestSpec* spec = nullptr;
    double now_ms = 0;          // current simulation time
    double remaining_ms = 0;    // estimated time to finish the candidate
    double restore_ms = 0;      // cost to bring back the candidate's snapshot
};

// Lower key schedules first; request_id is the deterministic tiebreak.
using PriorityKey = std::pair<double, std::string>;

// Scheduling strategy. The event loop is fixed; policies only answer through
// these hooks, so a new strategy never touches the runtime.
class Policy {
public:
    virtual ~Policy() = default;
    virtual std::string name() const = 0;
    virtual PriorityKey priority(const PriorityInput& input) const = 0;
    virtual bool uses_cache() const { return true; }       // false: disable the feature cache
    virtual bool preemptive() const { return true; }       // false: never preempt resident requests
    virtual bool group_paths() const { return false; }     // true: order/split batches by execution path
    virtual bool same_path_only() const { return false; }  // true: never mix paths in one batch
    virtual bool state_aware() const { return false; }     // true: value-based eviction + keep/switch
    virtual int batch_limit(int requested) const { return requested; }
};

// Named policy construction. Built-ins are pre-registered; extensions call
// instance().add(name, factory) before building a Simulator.
class PolicyRegistry {
public:
    using Factory = std::function<std::unique_ptr<Policy>()>;
    static PolicyRegistry& instance();
    void add(const std::string& name, Factory factory);
    std::unique_ptr<Policy> create(const std::string& name) const;
    std::vector<std::string> names() const;
private:
    std::vector<std::pair<std::string, Factory>> factories_;
};
}
