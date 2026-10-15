#pragma once

#include "ditserve/random.hpp"
#include <memory>
#include <optional>

namespace ditserve {

// Scheduler-facing execution state. Backends carry the real payloads;
// the scheduler only needs progress, path history and checkpoint bytes.
class State {
public:
    virtual ~State() = default;
    virtual bool is_complete() const = 0;
    virtual int step_index() const = 0;
    virtual const std::vector<std::string>& path_history() const = 0;
    virtual Json snapshot() const = 0;
};

// Control-plane backend contract. The scheduler drives execution exclusively
// through this interface, so alternative backends plug in without changes.
class Backend {
public:
    virtual ~Backend() = default;
    virtual std::unique_ptr<State> prepare(const RequestSpec& spec) const = 0;
    virtual std::string peek_path(const State& state) const = 0;
    virtual void step_batch(const std::vector<State*>& states) const = 0;
    virtual std::unique_ptr<State> restore(const Json& snapshot) const = 0;
    virtual std::string fingerprint(const State& state) const = 0;
};

// Synthetic reference executor: fixed RNG/sampling algorithms and analytic
// cache decisions. It is a reproducibility oracle, not a diffusion model.
class ReferenceState : public State {
public:
    RequestSpec spec;
    std::vector<double> latents;
    std::vector<std::vector<double>> sampler_history;
    Random random;
    std::optional<std::vector<double>> cache_features, cache_residual, previous_probe;
    double error_accumulator = 0;
    int step_index_ = 0;
    std::vector<std::string> path_history_;
    bool is_complete() const override;
    int step_index() const override { return step_index_; }
    const std::vector<std::string>& path_history() const override { return path_history_; }
    Json snapshot() const override;
    static ReferenceState restore(const Json& snapshot);
};

class ReferenceBackend : public Backend {
public:
    explicit ReferenceBackend(int latent_size = 16);
    // Concrete value API, preferred where the reference type is known.
    ReferenceState prepare_state(const RequestSpec& spec) const;
    std::string peek_path(const ReferenceState& state) const;
    void step_batch(const std::vector<ReferenceState*>& states) const;
    std::string fingerprint(const ReferenceState& state) const;
    // Type-erased Backend interface used by the scheduler.
    std::unique_ptr<State> prepare(const RequestSpec& spec) const override;
    std::string peek_path(const State& state) const override;
    void step_batch(const std::vector<State*>& states) const override;
    std::unique_ptr<State> restore(const Json& snapshot) const override;
    std::string fingerprint(const State& state) const override;
private:
    int latent_size_;
    static const ReferenceState& reference(const State& state);
    static ReferenceState& reference(State& state);
};
}
