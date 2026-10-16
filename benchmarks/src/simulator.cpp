#include "ditserve/benchmark/simulator.hpp"
#include "ditserve/benchmark/metrics.hpp"
#include "ditserve/benchmark/policy.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>

namespace ditserve {
void RuntimeConfig::validate() const {
    PolicyRegistry::instance().create(policy);
    if (num_gpus < 1 || max_batch < 1) throw std::invalid_argument("GPU and batch counts must be positive");
    number(memory_mb, "memory_mb", true); number(host_memory_mb, "host_memory_mb", true);
}
Json RuntimeConfig::to_json() const {
    return {{"policy", policy}, {"num_gpus", num_gpus}, {"max_batch", max_batch}, {"memory_mb", memory_mb},
            {"host_memory_mb", host_memory_mb}, {"verify_outputs", verify_outputs}};
}
RuntimeConfig RuntimeConfig::from_json(const Json& value) {
    check_keys(value, {"policy", "num_gpus", "max_batch", "memory_mb", "host_memory_mb", "verify_outputs"});
    RuntimeConfig result;
    result.policy = string_field(value, "policy", result.policy);
    result.num_gpus = int_field(value, "num_gpus", result.num_gpus); result.max_batch = int_field(value, "max_batch", result.max_batch);
    result.memory_mb = number_field(value, "memory_mb", result.memory_mb, true);
    result.host_memory_mb = number_field(value, "host_memory_mb", result.host_memory_mb, true);
    result.verify_outputs = bool_field(value, "verify_outputs", result.verify_outputs);
    result.validate(); return result;
}

namespace {
constexpr double epsilon = 1e-9;
struct Job {
    RequestSpec spec;
    std::unique_ptr<State> state;
    std::optional<Json> snapshot;
    std::optional<int> gpu;
    bool running = false, paused = false;
    std::optional<double> first_start_ms;
    double last_run_ms = 0, gpu_ms = 0, available_ms = 0;
    int compute_steps = 0, reuse_steps = 0, preemptions = 0, offloads = 0, restores = 0;
};
struct Worker {
    int index;
    std::optional<double> busy_until;
    std::vector<std::string> batch, paths, previous;
};
bool contains(const std::vector<Job*>& jobs, const Job* job) { return std::find(jobs.begin(), jobs.end(), job) != jobs.end(); }

// One run owns all mutable state; a Simulator can be reused without reset APIs.
class Run {
public:
    Run(const RuntimeConfig& config, const CostModel& cost, const Backend& backend, std::vector<RequestSpec> specs);
    Json execute();
private:
    const RuntimeConfig& config_;
    const CostModel& cost_;
    const Backend& backend_;
    std::unique_ptr<Policy> policy_;
    std::vector<RequestSpec> specs_;
    std::vector<std::unique_ptr<Job>> jobs_;
    std::vector<Worker> workers_;
    Json events_ = Json::array(), records_ = Json::array();
    double now_ = 0, host_mb_ = 0, peak_host_mb_ = 0, busy_ms_ = 0, transfer_ms_ = 0, transfer_mb_ = 0;
    int preemptions_ = 0, offloads_ = 0, restores_ = 0, split_batches_ = 0;
    std::vector<double> peak_gpu_mb_;
    std::vector<std::pair<double, double>> host_releases_;
    std::size_t batch_count_ = 0, batch_members_ = 0;
    void event(const std::string& kind, Json fields);
    Job* find(const std::string& id) const;
    int step(const Job& job) const;
    std::string path(const Job& job) const;
    double state_size(const Job& job) const;
    double remaining(const Job& job) const;
    std::pair<double, std::string> priority(const Job& job) const;
    std::vector<Job*> resident(const Worker& worker) const;
    void schedule(Worker& worker);
    void complete(Worker& worker);
    void finish(Job* job, const std::string& status, Json reason);
};

Run::Run(const RuntimeConfig& config, const CostModel& cost, const Backend& backend, std::vector<RequestSpec> specs)
    : config_(config), cost_(cost), backend_(backend), policy_(PolicyRegistry::instance().create(config.policy)),
      specs_(std::move(specs)), peak_gpu_mb_(config.num_gpus, 0) {
    for (auto& spec : specs_) if (!policy_->uses_cache()) spec.cache_enabled = false;
    std::stable_sort(specs_.begin(), specs_.end(), [](const auto& left, const auto& right) {
        return std::tie(left.arrival_ms, left.request_id) < std::tie(right.arrival_ms, right.request_id);
    });
    for (int i = 0; i < config.num_gpus; ++i) workers_.push_back({i, {}, {}, {}, {}});
    if (!specs_.empty()) now_ = specs_[0].arrival_ms;
}
void Run::event(const std::string& kind, Json fields) { fields["time_ms"] = now_; fields["event"] = kind; events_.push_back(std::move(fields)); }
Job* Run::find(const std::string& id) const {
    for (const auto& job : jobs_) if (job->spec.request_id == id) return job.get();
    return nullptr;
}
int Run::step(const Job& job) const { return job.state ? job.state->step_index() : job.snapshot->at("step_index").get<int>(); }
std::string Run::path(const Job& job) const {
    if (job.state) return backend_.peek_path(*job.state);
    return backend_.peek_path(*backend_.restore(*job.snapshot));
}
double Run::state_size(const Job& job) const { return cost_.state_mb(job.spec, step(job) > 0); }
double Run::remaining(const Job& job) const {
    const int observed = job.compute_steps + job.reuse_steps;
    const double fraction = observed ? static_cast<double>(job.compute_steps) / observed : 1;
    const double mean = fraction * cost_.step_ms(job.spec, "compute") + (1 - fraction) * cost_.step_ms(job.spec, "reuse");
    return std::max(0, job.spec.steps - step(job)) * mean;
}
std::pair<double, std::string> Run::priority(const Job& job) const {
    const PriorityInput input{&job.spec, now_, remaining(job), job.snapshot ? cost_.transfer_ms(state_size(job)) : 0};
    return policy_->priority(input);
}
std::vector<Job*> Run::resident(const Worker& worker) const {
    std::vector<Job*> result;
    for (const auto& job : jobs_) if (job->gpu == worker.index) result.push_back(job.get());
    return result;
}

void Run::schedule(Worker& worker) {
    std::vector<Job*> candidates, old;
    for (const auto& job : jobs_)
        if (!job->running && job->available_ms <= now_ + epsilon && (!job->gpu || job->gpu == worker.index)) candidates.push_back(job.get());
    if (candidates.empty()) return;
    std::stable_sort(candidates.begin(), candidates.end(), [&](const auto* a, const auto* b) { return priority(*a) < priority(*b); });
    for (const auto& id : worker.previous) { auto* job = find(id); if (job && !job->running) old.push_back(job); }
    if (!policy_->preemptive() && !old.empty()) {
        std::vector<Job*> ordered = old;
        for (auto* job : candidates) if (!contains(old, job)) ordered.push_back(job);
        candidates = std::move(ordered);
    }
    auto* head = candidates.front();
    const auto head_path = path(*head);
    std::vector<Job*> compatible;
    for (auto* job : candidates) if (job->spec.shape_key() == head->spec.shape_key()) compatible.push_back(job);
    if (policy_->group_paths()) {
        std::vector<Job*> same, other;
        for (auto* job : compatible) (path(*job) == head_path ? same : other).push_back(job);
        if (!policy_->same_path_only()) same.insert(same.end(), other.begin(), other.end());
        compatible = std::move(same);
    }
    std::vector<Job*> chosen;
    double reserved_states = 0;
    for (auto* job : compatible) {
        if (static_cast<int>(chosen.size()) + 1 > policy_->batch_limit(config_.max_batch)) break;
        const double reserved = reserved_states + cost_.state_mb(job->spec) + cost_.workspace(head->spec, static_cast<int>(chosen.size()) + 1);
        if (reserved <= config_.memory_mb + epsilon) { chosen.push_back(job); reserved_states += cost_.state_mb(job->spec); }
    }
    if (chosen.empty()) return;
    if (policy_->state_aware() && !old.empty() && std::none_of(old.begin(), old.end(), [&](auto* j) { return contains(chosen, j); })) {
        auto* next = chosen.front(); auto* keep = old.front();
        const double gain = priority(*keep).first - priority(*next).first;
        const double switching = cost_.transfer_ms(state_size(*keep)) + cost_.transfer_ms(state_size(*next));
        if (gain > 0 && gain < switching && next->spec.deadline_ms - now_ > remaining(*next) + switching) chosen = {keep};
    }
    double needed = cost_.workspace(chosen.front()->spec, static_cast<int>(chosen.size()));
    for (auto* job : chosen) needed += cost_.state_mb(job->spec);
    std::vector<Job*> extras, victims;
    double extras_size = 0;
    for (auto* job : resident(worker)) if (!contains(chosen, job)) { extras.push_back(job); extras_size += state_size(*job); }
    std::stable_sort(extras.begin(), extras.end(), [&](const auto* a, const auto* b) {
        return policy_->state_aware() ? std::make_pair(priority(*a).first, state_size(*a)) > std::make_pair(priority(*b).first, state_size(*b))
                                      : std::tie(a->last_run_ms, a->spec.request_id) < std::tie(b->last_run_ms, b->spec.request_id);
    });
    for (auto* job : extras) {
        if (needed + extras_size <= config_.memory_mb + epsilon) break;
        victims.push_back(job); extras_size -= state_size(*job);
    }
    double outgoing = 0, incoming = 0;
    for (auto* job : victims) outgoing += state_size(*job);
    for (auto* job : chosen) if (job->snapshot) incoming += state_size(*job);
    if (host_mb_ + outgoing - incoming > config_.host_memory_mb + epsilon || host_mb_ + outgoing > config_.host_memory_mb + epsilon) {
        victims.clear(); chosen.clear();
        for (auto* candidate : candidates) if (candidate->gpu == worker.index) {
            const double candidate_need = cost_.state_mb(candidate->spec) + cost_.workspace(candidate->spec);
            double other_size = 0;
            for (auto* job : resident(worker)) if (job != candidate) other_size += state_size(*job);
            if (candidate_need + other_size <= config_.memory_mb + epsilon) { chosen.push_back(candidate); break; }
        }
        if (chosen.empty()) return;
    }
    for (auto* job : old) if (!contains(chosen, job) && !job->paused) {
        job->paused = true; ++job->preemptions; ++preemptions_;
        event("pause", {{"request_id", job->spec.request_id}, {"gpu", worker.index}});
    }
    double transfer = 0;
    for (auto* job : victims) {
        const double size = state_size(*job), duration = cost_.transfer_ms(size);
        transfer += duration; transfer_mb_ += size;
        job->snapshot = job->state->snapshot(); job->state.reset(); job->gpu.reset(); ++job->offloads;
        job->available_ms = now_ + transfer; ++offloads_; host_mb_ += size; peak_host_mb_ = std::max(peak_host_mb_, host_mb_);
        event("offload", {{"request_id", job->spec.request_id}, {"gpu", worker.index}, {"memory_mb", size}, {"duration_ms", duration}, {"ready_ms", job->available_ms}});
    }
    for (auto* job : chosen) {
        if (job->snapshot) {
            const double size = state_size(*job), duration = cost_.transfer_ms(size);
            transfer += duration; transfer_mb_ += size;
            // Host ownership ends only when the simulated H2D transfer completes.
            host_releases_.push_back({now_ + transfer, size});
            job->state = backend_.restore(*job->snapshot); job->snapshot.reset(); ++job->restores; ++restores_;
            event("restore", {{"request_id", job->spec.request_id}, {"gpu", worker.index}, {"memory_mb", size}, {"duration_ms", duration}, {"ready_ms", now_ + transfer}});
        }
        job->gpu = worker.index; job->running = true; job->paused = false;
        if (!job->first_start_ms) job->first_start_ms = now_;
    }
    std::vector<RequestSpec> specs;
    worker.batch.clear(); worker.paths.clear();
    for (auto* job : chosen) { specs.push_back(job->spec); worker.batch.push_back(job->spec.request_id); worker.paths.push_back(path(*job)); }
    if (policy_->group_paths() && std::find(worker.paths.begin(), worker.paths.end(), "compute") != worker.paths.end() &&
        std::find(worker.paths.begin(), worker.paths.end(), "reuse") != worker.paths.end()) ++split_batches_;
    const double compute = cost_.batch_ms(specs, worker.paths, policy_->group_paths()), duration = transfer + compute;
    number(duration, "batch duration", true);
    transfer_ms_ += transfer; busy_ms_ += duration;
    for (auto* job : chosen) job->gpu_ms += duration / chosen.size();
    double occupied = 0;
    for (auto* job : resident(worker)) occupied += contains(chosen, job) ? cost_.state_mb(job->spec) : state_size(*job);
    occupied += cost_.workspace(chosen.front()->spec, static_cast<int>(chosen.size()));
    if (occupied > config_.memory_mb + 1e-7) throw std::logic_error("device state/workspace budget exceeded");
    peak_gpu_mb_[worker.index] = std::max(peak_gpu_mb_[worker.index], occupied);
    ++batch_count_; batch_members_ += chosen.size(); worker.busy_until = now_ + duration;
    event("batch", {{"gpu", worker.index}, {"requests", worker.batch}, {"paths", worker.paths}, {"duration_ms", duration},
                    {"transfer_ms", transfer}, {"compute_ms", compute}, {"memory_mb", occupied}});
}

void Run::complete(Worker& worker) {
    std::vector<Job*> jobs;
    std::vector<State*> states;
    for (const auto& id : worker.batch) { auto* job = find(id); jobs.push_back(job); states.push_back(job->state.get()); }
    backend_.step_batch(states); worker.previous = worker.batch;
    for (std::size_t i = 0; i < jobs.size(); ++i) {
        auto* job = jobs[i]; job->running = false; job->last_run_ms = now_;
        if (worker.paths[i] == "compute") ++job->compute_steps; else ++job->reuse_steps;
        event("step", {{"request_id", job->spec.request_id}, {"gpu", worker.index}, {"step", job->state->step_index()}, {"path", worker.paths[i]}});
        if (job->state->is_complete()) finish(job, "completed", nullptr);
        else if (now_ >= job->spec.deadline_ms - epsilon) finish(job, "timed_out", "deadline_at_step_boundary");
    }
    worker.busy_until.reset(); worker.batch.clear(); worker.paths.clear();
}
void Run::finish(Job* job, const std::string& status, Json reason) {
    std::unique_ptr<State> restored;
    const State* state = job->state.get();
    if (!state) { restored = backend_.restore(*job->snapshot); state = restored.get(); }
    Json fingerprint = nullptr, parity = nullptr;
    if (status == "completed") {
        fingerprint = backend_.fingerprint(*state);
        if (config_.verify_outputs) {
            auto isolated = backend_.prepare(job->spec);
            while (!isolated->is_complete()) backend_.step_batch({isolated.get()});
            parity = fingerprint == backend_.fingerprint(*isolated);
        }
    }
    if (job->snapshot) host_mb_ -= state_size(*job);
    records_.push_back({{"request_id", job->spec.request_id}, {"arrival_ms", job->spec.arrival_ms}, {"deadline_ms", job->spec.deadline_ms},
        {"status", status}, {"reason", reason}, {"completion_ms", status == "completed" ? Json(now_) : Json(nullptr)},
        {"terminal_ms", now_}, {"latency_ms", status == "completed" ? Json(now_ - job->spec.arrival_ms) : Json(nullptr)},
        {"first_start_ms", job->first_start_ms ? Json(*job->first_start_ms) : Json(nullptr)}, {"quality_ok", parity}, {"gpu_ms", job->gpu_ms},
        {"compute_steps", job->compute_steps}, {"reuse_steps", job->reuse_steps}, {"preemptions", job->preemptions}, {"offloads", job->offloads},
        {"restores", job->restores}, {"fingerprint", fingerprint}, {"cache_enabled", job->spec.cache_enabled}, {"path_history", state->path_history()}});
    event(status, {{"request_id", job->spec.request_id}, {"reason", reason}});
    jobs_.erase(std::find_if(jobs_.begin(), jobs_.end(), [&](const auto& pointer) { return pointer.get() == job; }));
}

Json Run::execute() {
    const double start = now_;
    std::size_t cursor = 0;
    while (cursor < specs_.size() || !jobs_.empty()) {
        double released = 0;
        for (const auto& release : host_releases_) if (release.first <= now_ + epsilon) released += release.second;
        host_mb_ -= released;
        host_releases_.erase(std::remove_if(host_releases_.begin(), host_releases_.end(), [&](const auto& release) { return release.first <= now_ + epsilon; }), host_releases_.end());
        // Preserve the observable ordering: completions precede same-time arrivals.
        for (auto& worker : workers_) if (worker.busy_until && *worker.busy_until <= now_ + epsilon) complete(worker);
        while (cursor < specs_.size() && specs_[cursor].arrival_ms <= now_ + epsilon) {
            auto owned = std::make_unique<Job>(); owned->spec = specs_[cursor]; owned->state = backend_.prepare(owned->spec);
            auto* job = owned.get(); jobs_.push_back(std::move(owned));
            event("arrival", {{"request_id", job->spec.request_id}});
            if (cost_.state_mb(job->spec) + cost_.workspace(job->spec) > config_.memory_mb + epsilon)
                finish(job, "rejected", "request_peak_exceeds_device_budget");
            ++cursor;
        }
        std::vector<Job*> expired;
        for (const auto& job : jobs_) if (!job->running && job->available_ms <= now_ + epsilon && job->spec.deadline_ms <= now_ + epsilon) expired.push_back(job.get());
        for (auto* job : expired) finish(job, "timed_out", "deadline");
        for (auto& worker : workers_) if (!worker.busy_until) schedule(worker);
        if (cursor >= specs_.size() && jobs_.empty()) break;
        std::vector<double> future;
        for (const auto& worker : workers_) if (worker.busy_until && *worker.busy_until > now_ + epsilon) future.push_back(*worker.busy_until);
        if (cursor < specs_.size()) future.push_back(specs_[cursor].arrival_ms);
        for (const auto& job : jobs_) {
            if (!job->running && job->spec.deadline_ms > now_ + epsilon) future.push_back(job->spec.deadline_ms);
            if (job->available_ms > now_ + epsilon) future.push_back(job->available_ms);
        }
        for (const auto& release : host_releases_) if (release.first > now_ + epsilon) future.push_back(release.first);
        if (future.empty()) throw std::logic_error("simulator made no progress");
        now_ = *std::min_element(future.begin(), future.end());
    }
    auto summary = summarize(records_, start, std::max(now_, start), busy_ms_, config_.num_gpus);
    int checked = 0, failures = 0;
    for (const auto& record : records_) { checked += !record.at("quality_ok").is_null(); failures += record.at("quality_ok") == false; }
    summary.update({{"policy", config_.policy}, {"preemptions", preemptions_}, {"offloads", offloads_}, {"restores", restores_},
        {"transfer_mb", transfer_mb_}, {"transfer_ms", transfer_ms_}, {"peak_device_state_and_workspace_mb", peak_gpu_mb_},
        {"peak_host_state_mb", peak_host_mb_}, {"mean_batch_size", batch_count_ ? static_cast<double>(batch_members_) / batch_count_ : 0},
        {"split_batches", split_batches_}, {"parity_checked", checked}, {"parity_failures", failures},
        {"quality_scope", "Exact reference-state parity against isolated execution, NOT image/video quality"}, {"research_gate_eligible", false}});
    std::sort(records_.begin(), records_.end(), [](const auto& a, const auto& b) {
        return std::make_pair(a.at("arrival_ms").template get<double>(), a.at("request_id").template get<std::string>()) <
               std::make_pair(b.at("arrival_ms").template get<double>(), b.at("request_id").template get<std::string>());
    });
    return {{"schema_version", 2}, {"core", core_version()}, {"provenance", "synthetic"}, {"config", config_.to_json()},
            {"cost_model", cost_.to_json()}, {"summary", summary}, {"records", records_}, {"events", events_}};
}
}

Simulator::Simulator(RuntimeConfig config, CostModel cost, std::shared_ptr<const Backend> backend)
    : config_(std::move(config)), cost_(std::move(cost)), backend_(std::move(backend)) { config_.validate(); cost_.validate(); }
Json Simulator::run(std::vector<RequestSpec> specs) const {
    validate_workload(specs);
    return Run(config_, cost_, *backend_, std::move(specs)).execute();
}
}
