#pragma once

#include "ditserve/policy.hpp"

namespace ditserve {

// FIFO ranking shared by the nonpreemptive family.
class FifoPolicy : public Policy {
public:
    PriorityKey priority(const PriorityInput& input) const override {
        return {input.spec->arrival_ms, input.spec->request_id};
    }
    bool preemptive() const override { return false; }
};

// Fixed cache, FIFO, batch size one, no preemption of running requests.
class CacheOnlyPolicy : public FifoPolicy {
public:
    std::string name() const override { return "cache_only"; }
    int batch_limit(int) const override { return 1; }
};

// Cache disabled, FIFO continuous batching, no preemption.
class BatchOnlyPolicy : public FifoPolicy {
public:
    std::string name() const override { return "batch_only"; }
    bool uses_cache() const override { return false; }
};

// Fixed cache, FIFO; mixed-path batches are billed at the expensive branch.
class NaivePolicy : public FifoPolicy {
public:
    std::string name() const override { return "naive"; }
};

// Earliest-deadline-first; preempts at safe step boundaries, offloads when needed.
class EdfPolicy : public Policy {
public:
    std::string name() const override { return "edf"; }
    PriorityKey priority(const PriorityInput& input) const override {
        return {input.spec->deadline_ms, input.spec->request_id};
    }
};

// EDF that only batches requests currently on the same execution path.
class PathAwarePolicy : public EdfPolicy {
public:
    std::string name() const override { return "path_aware"; }
    bool group_paths() const override { return true; }
    bool same_path_only() const override { return true; }
};

// Ranks by deadline, remaining work and restore cost; groups by path and
// evicts/keeps resident state by value.
class StateAwarePolicy : public Policy {
public:
    std::string name() const override { return "state_aware"; }
    PriorityKey priority(const PriorityInput& input) const override {
        return {input.spec->deadline_ms - input.now_ms - input.remaining_ms + input.restore_ms,
                input.spec->request_id};
    }
    bool group_paths() const override { return true; }
    bool state_aware() const override { return true; }
};

// Called once by PolicyRegistry::instance(); keeps the canonical order.
void register_builtin_policies(PolicyRegistry& registry);
}
