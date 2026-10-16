#include "ditserve/benchmark/backend.hpp"
#include "ditserve/runtime/random.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <openssl/sha.h>
#include <set>
#include <sstream>
#include <stdexcept>

namespace ditserve {
namespace {
class ReferenceState final : public State {
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

struct Decision { std::string path; std::vector<double> probe; double accumulated; };

Decision decide(const ReferenceState& state) {
    if (state.is_complete()) throw std::invalid_argument("request already complete: " + state.spec.request_id);
    const double time = (state.step_index_ + 1.0) / state.spec.steps;
    const double geometry = std::log1p(static_cast<double>(state.spec.height) * state.spec.width * state.spec.frames) * 0.002;
    Decision result{"compute", {}, state.error_accumulator};
    for (std::size_t i = 0; i < state.latents.size(); ++i)
        result.probe.push_back(state.latents[i] + geometry + 0.025 * std::sin(time + i * 0.17));
    if (state.previous_probe) {
        double difference = 0, scale = 0;
        for (std::size_t i = 0; i < result.probe.size(); ++i) {
            difference += std::abs(result.probe[i] - state.previous_probe->at(i));
            scale += std::abs(state.previous_probe->at(i));
        }
        result.accumulated += difference / std::max(scale, 1e-12);
    }
    if (state.spec.cache_enabled && state.step_index_ != 0 && state.step_index_ != state.spec.steps - 1 &&
        state.cache_residual && result.accumulated < state.spec.cache_threshold) result.path = "reuse";
    return result;
}

Json optional_vector(const std::optional<std::vector<double>>& value) { return value ? Json(*value) : Json(nullptr); }
void validate_vector(const std::vector<double>& values, std::size_t size) {
    if (values.size() != size || std::any_of(values.begin(), values.end(), [](double x) { return !std::isfinite(x); }))
        throw std::invalid_argument("checkpoint vector has invalid shape or non-finite values");
}

bool ReferenceState::is_complete() const { return step_index_ >= spec.steps; }
Json ReferenceState::snapshot() const {
    return {{"version", 2}, {"spec", spec.to_json()}, {"latents", latents}, {"sampler_history", sampler_history},
            {"random_state", random.snapshot()}, {"cache_features", optional_vector(cache_features)},
            {"cache_residual", optional_vector(cache_residual)}, {"previous_probe", optional_vector(previous_probe)},
            {"error_accumulator", error_accumulator}, {"step_index", step_index_}, {"path_history", path_history_}};
}

ReferenceState ReferenceState::restore(const Json& snapshot) {
    if (integer(snapshot.at("version"), "checkpoint version") != 2) throw std::invalid_argument("unsupported checkpoint version");
    ReferenceState state;
    state.spec = RequestSpec::from_json(snapshot.at("spec"));
    state.latents = snapshot.at("latents").get<std::vector<double>>();
    if (state.latents.empty()) throw std::invalid_argument("empty checkpoint latents");
    validate_vector(state.latents, state.latents.size());
    state.sampler_history = snapshot.at("sampler_history").get<std::vector<std::vector<double>>>();
    if (state.sampler_history.size() > 2) throw std::invalid_argument("invalid sampler history length");
    for (const auto& row : state.sampler_history) validate_vector(row, state.latents.size());
    state.random = Random::restore(snapshot.at("random_state"));
    const auto restore_vector = [&](const char* name, std::optional<std::vector<double>>& destination) {
        if (!snapshot.at(name).is_null()) {
            destination = snapshot.at(name).get<std::vector<double>>();
            validate_vector(*destination, state.latents.size());
        }
    };
    restore_vector("cache_features", state.cache_features); restore_vector("cache_residual", state.cache_residual);
    restore_vector("previous_probe", state.previous_probe);
    state.error_accumulator = number(snapshot.at("error_accumulator"), "error_accumulator");
    state.step_index_ = int_field(snapshot, "step_index", 0, 0);
    state.path_history_ = snapshot.at("path_history").get<std::vector<std::string>>();
    if (state.step_index_ > state.spec.steps || state.path_history_.size() != static_cast<std::size_t>(state.step_index_) ||
        std::any_of(state.path_history_.begin(), state.path_history_.end(), [](const auto& p) { return p != "compute" && p != "reuse"; }))
        throw std::invalid_argument("checkpoint step/path history mismatch");
    if (state.sampler_history.size() != static_cast<std::size_t>(std::min(state.step_index_, 2)) ||
        (state.step_index_ > 0) != state.previous_probe.has_value() ||
        state.cache_features.has_value() != state.cache_residual.has_value() ||
        (state.spec.cache_enabled && state.step_index_ > 0) != state.cache_residual.has_value())
        throw std::invalid_argument("checkpoint missing or inconsistent execution history");
    return state;
}

const ReferenceState& reference(const State& state) {
    const auto* result = dynamic_cast<const ReferenceState*>(&state);
    if (!result) throw std::invalid_argument("state belongs to a different backend");
    return *result;
}
ReferenceState& reference(State& state) {
    auto* result = dynamic_cast<ReferenceState*>(&state);
    if (!result) throw std::invalid_argument("state belongs to a different backend");
    return *result;
}
}

ReferenceBackend::ReferenceBackend(int latent_size) : latent_size_(latent_size) {
    if (latent_size < 1) throw std::invalid_argument("latent_size must be positive");
}
std::unique_ptr<State> ReferenceBackend::prepare(const RequestSpec& spec) const {
    spec.validate();
    auto state = std::make_unique<ReferenceState>();
    state->spec = spec; state->random = Random(static_cast<std::uint64_t>(spec.seed));
    for (int i = 0; i < latent_size_; ++i) state->latents.push_back(-1.0 + 2.0 * state->random.uniform());
    return state;
}
std::string ReferenceBackend::peek_path(const State& state) const { return decide(reference(state)).path; }
void ReferenceBackend::step_batch(const std::vector<State*>& states) const {
    std::set<const State*> pointers;
    std::vector<ReferenceState*> concrete;
    std::set<std::string> ids;
    std::vector<Decision> decisions;
    for (auto* erased : states) {
        if (!erased || !pointers.insert(erased).second)
            throw std::invalid_argument("batch requires distinct non-null states");
        auto& state = reference(*erased);
        if (!ids.insert(state.spec.request_id).second)
            throw std::invalid_argument("batch requires distinct request IDs");
        concrete.push_back(&state);
        decisions.push_back(decide(state));
    }
    for (std::size_t j = 0; j < concrete.size(); ++j) {
        auto& state = *concrete[j]; const auto& decision = decisions[j];
        std::vector<double> residual;
        if (decision.path == "compute") {
            std::vector<double> features;
            const double time = (state.step_index_ + 1.0) / state.spec.steps;
            for (std::size_t i = 0; i < state.latents.size(); ++i) {
                const double feature = std::tanh(decision.probe[i] + 0.1 * std::sin(time + i * 0.31));
                features.push_back(feature); residual.push_back(0.6 * state.latents[i] + 0.35 * feature);
            }
            if (state.spec.cache_enabled) { state.cache_features = features; state.cache_residual = residual; }
            state.error_accumulator = 0;
        } else { residual = *state.cache_residual; state.error_accumulator = decision.accumulated; }
        for (std::size_t i = 0; i < state.latents.size(); ++i) {
            const double velocity = state.sampler_history.empty() ? residual[i] : 1.1 * residual[i] - 0.1 * state.sampler_history.back()[i];
            const double noise = state.random.normal() * 0.0005 * (state.spec.steps - state.step_index_) / state.spec.steps;
            state.latents[i] = state.latents[i] - 0.065 * velocity + noise;
        }
        state.sampler_history.push_back(residual);
        if (state.sampler_history.size() > 2) state.sampler_history.erase(state.sampler_history.begin());
        state.previous_probe = decision.probe; ++state.step_index_; state.path_history_.push_back(decision.path);
    }
}
std::string ReferenceBackend::fingerprint(const State& state) const {
    const std::string encoded = reference(state).snapshot().dump();
    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(encoded.data()), encoded.size(), digest);
    std::ostringstream stream;
    for (const auto byte : digest) stream << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(byte);
    return stream.str();
}

std::unique_ptr<State> ReferenceBackend::restore(const Json& snapshot) const {
    return std::make_unique<ReferenceState>(ReferenceState::restore(snapshot));
}
}
