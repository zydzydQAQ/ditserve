#pragma once

#include "ditserve/benchmark/workload.hpp"
#include <memory>

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
class ReferenceBackend final : public Backend {
public:
    explicit ReferenceBackend(int latent_size = 16);
    std::unique_ptr<State> prepare(const RequestSpec& spec) const override;
    std::string peek_path(const State& state) const override;
    void step_batch(const std::vector<State*>& states) const override;
    std::unique_ptr<State> restore(const Json& snapshot) const override;
    std::string fingerprint(const State& state) const override;
private:
    int latent_size_;
};
}
