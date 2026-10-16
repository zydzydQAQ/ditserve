#include "ditserve/control/budget.hpp"
#include "ditserve/runtime/process.hpp"
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <sys/file.h>
#include <unistd.h>

namespace ditserve {
namespace {
double finite(double value, const char* label) {
    if (!std::isfinite(value) || value < 0) throw std::invalid_argument(std::string(label) + " must be finite and nonnegative");
    return value;
}
Json limits_json(const BudgetLimits& limits) {
    if (limits.total_gpu_seconds) finite(*limits.total_gpu_seconds, "total GPU seconds");
    finite(limits.repair_reserve_gpu_seconds, "repair reserve");
    if (limits.total_gpu_seconds && limits.repair_reserve_gpu_seconds > *limits.total_gpu_seconds)
        throw std::invalid_argument("repair reserve exceeds total budget");
    if (limits.carry != "none" && limits.carry != "forward") throw std::invalid_argument("carry must be none or forward");
    for (const auto& phase : limits.phase_gpu_seconds) {
        if (phase.first.empty()) throw std::invalid_argument("empty budget phase");
        finite(phase.second, "phase GPU seconds");
    }
    std::set<std::string> phases;
    for (const auto& phase : limits.phase_order)
        if (phase.empty() || !phases.insert(phase).second) throw std::invalid_argument("phase order must be nonempty and unique");
    if (!limits.phase_gpu_seconds.empty() && !limits.phase_order.empty()) {
        if (phases.size() != limits.phase_gpu_seconds.size()) throw std::invalid_argument("phase order differs from phase limits");
        for (const auto& phase : limits.phase_gpu_seconds)
            if (!phases.count(phase.first)) throw std::invalid_argument("phase order differs from phase limits");
    }
    if (limits.carry == "forward" && !limits.phase_gpu_seconds.empty() && phases.empty())
        throw std::invalid_argument("forward carry requires explicit phase order");
    return {{"total_gpu_seconds", limits.total_gpu_seconds ? Json(*limits.total_gpu_seconds) : Json(nullptr)},
            {"phase_gpu_seconds", limits.phase_gpu_seconds}, {"phase_order", limits.phase_order},
            {"repair_reserve_gpu_seconds", limits.repair_reserve_gpu_seconds}, {"carry", limits.carry}};
}
std::string identifier() {
    std::random_device random;
    std::ostringstream value;
    value << std::hex << std::setfill('0');
    for (int i = 0; i < 4; ++i) value << std::setw(8) << random();
    return value.str();
}
bool same_charge(double left, double right) {
    return std::abs(left - right) <= std::max(1e-6, 1e-12 * std::max(left, right));
}
bool completion_status(const std::string& status) {
    static const std::set<std::string> statuses{"completed", "failed", "interrupted", "timed_out", "cancelled", "incomplete"};
    return statuses.count(status);
}
int phase_index(const BudgetLimits& limits, const std::string& phase) {
    if (limits.phase_order.empty()) return -1;
    const auto found = std::find(limits.phase_order.begin(), limits.phase_order.end(), phase);
    if (found == limits.phase_order.end()) throw std::invalid_argument("phase is absent from configured phase order");
    return static_cast<int>(found - limits.phase_order.begin());
}
}

Budget::Budget(std::filesystem::path path, BudgetLimits limits, Clock monotonic, Clock wall)
    : path_(std::filesystem::absolute(std::move(path)).lexically_normal()), limits_(std::move(limits)),
      monotonic_(monotonic ? std::move(monotonic) : Clock(monotonic_seconds)),
      wall_(wall ? std::move(wall) : Clock([] {
          return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
      })) {
    const auto encoded_limits = limits_json(limits_);
    std::filesystem::create_directories(path_.parent_path());
    const auto lock_path = path_.string() + ".lock";
    lock_ = open(lock_path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
    if (lock_ < 0) throw std::runtime_error("cannot open budget lock: " + std::string(std::strerror(errno)));
    try {
        if (flock(lock_, LOCK_EX | LOCK_NB) < 0) throw std::runtime_error("budget ledger is already locked");
        if (std::filesystem::exists(path_)) {
            std::ifstream stream(path_);
            if (!stream) throw std::runtime_error("cannot read budget ledger");
            stream >> data_;
            if (data_.value("schema_version", "") != "ditserve.budget.v1" || data_.at("limits") != encoded_limits)
                throw std::invalid_argument("budget schema or immutable allocation differs");
            if (!data_.at("active").is_null()) throw std::runtime_error("unsettled budget session; cleanup must be verified before recovery");
            if (!data_.at("sessions").is_array() || !data_.at("phase_usage").is_object())
                throw std::invalid_argument("invalid budget charge history");
            double regular = 0, repair = 0;
            int cursor = -1;
            Json phases = Json::object();
            std::set<std::string> ids;
            for (const auto& session : data_["sessions"]) {
                if (!ids.insert(session.at("session_id").get<std::string>()).second || session.at("cleanup_verified") != Json(true) ||
                    !completion_status(session.at("status").get<std::string>()))
                    throw std::invalid_argument("invalid settled budget session");
                const double charge = number(session.at("gpu_seconds"), "session GPU seconds");
                const auto devices = integer(session.at("devices"), "charged device count");
                const double elapsed = number(session.at("elapsed_seconds"), "charged elapsed seconds");
                if (devices < 0 || !same_charge(charge, elapsed * devices))
                    throw std::invalid_argument("budget device-seconds differ from actual elapsed time");
                const int index = phase_index(limits_, session.at("phase"));
                if (limits_.carry == "forward" && index < cursor)
                    throw std::invalid_argument("budget history returned to an earlier phase");
                cursor = std::max(cursor, index);
                if (session.at("repair").get<bool>()) repair += charge;
                else {
                    regular += charge;
                    const auto phase = session.at("phase").get<std::string>();
                    phases[phase] = phases.value(phase, 0.0) + charge;
                }
            }
            if (!same_charge(regular, number(data_.at("used_regular_gpu_seconds"), "regular GPU seconds")) ||
                !same_charge(repair, number(data_.at("used_repair_gpu_seconds"), "repair GPU seconds")) ||
                !same_charge(regular + repair, number(data_.at("used_gpu_seconds"), "used GPU seconds")) ||
                phases != data_["phase_usage"])
                throw std::invalid_argument("budget charge history is inconsistent");
            if (integer(data_.at("phase_cursor"), "phase cursor") != cursor)
                throw std::invalid_argument("invalid budget phase cursor");
        } else {
            data_ = {{"schema_version", "ditserve.budget.v1"}, {"allocation_id", identifier()}, {"limits", encoded_limits},
                     {"used_gpu_seconds", 0.0}, {"used_regular_gpu_seconds", 0.0}, {"used_repair_gpu_seconds", 0.0},
                     {"phase_usage", Json::object()}, {"phase_cursor", -1},
                     {"active", nullptr}, {"sessions", Json::array()}};
            save(data_);
        }
    } catch (...) { close(lock_); lock_ = -1; throw; }
}

Budget::~Budget() { if (lock_ >= 0) close(lock_); }

void Budget::commit(Json value) {
    try { save(value); }
    catch (...) { poisoned_ = true; throw; }
    data_ = std::move(value);
}

void Budget::save(const Json& value) const {
    auto temporary = (path_.parent_path() / ("." + path_.filename().string() + ".XXXXXX")).string();
    std::vector<char> name(temporary.begin(), temporary.end());
    name.push_back('\0');
    const int descriptor = mkstemp(name.data());
    if (descriptor < 0) throw std::runtime_error("cannot create atomic budget file");
    int directory = -1;
    bool closed = false;
    try {
        if (fcntl(descriptor, F_SETFD, FD_CLOEXEC) < 0) throw std::runtime_error("cannot protect budget file descriptor");
        const auto encoded = value.dump(2) + '\n';
        std::size_t position = 0;
        while (position < encoded.size()) {
            const auto count = write(descriptor, encoded.data() + position, encoded.size() - position);
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) throw std::runtime_error("cannot write budget ledger");
            position += static_cast<std::size_t>(count);
        }
        if (fsync(descriptor) < 0) throw std::runtime_error("cannot sync budget ledger");
        closed = true;
        if (close(descriptor) < 0) throw std::runtime_error("cannot close budget ledger");
        if (rename(name.data(), path_.c_str()) < 0) throw std::runtime_error("cannot replace budget ledger");
        directory = open(path_.parent_path().c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (directory < 0 || fsync(directory) < 0) throw std::runtime_error("cannot sync budget directory");
        close(directory);
    } catch (...) {
        if (!closed) close(descriptor);
        if (directory >= 0) close(directory);
        unlink(name.data());
        throw;
    }
}

double Budget::allowance(const std::string& phase, bool repair) const {
    const double total = limits_.total_gpu_seconds ? *limits_.total_gpu_seconds : std::numeric_limits<double>::infinity();
    double remaining = total - data_.at("used_gpu_seconds").get<double>();
    if (repair) return std::min(remaining, limits_.repair_reserve_gpu_seconds - data_.at("used_repair_gpu_seconds").get<double>());
    remaining = std::min(remaining, total - limits_.repair_reserve_gpu_seconds - data_.at("used_regular_gpu_seconds").get<double>());
    if (limits_.phase_gpu_seconds.empty()) return remaining;
    if (!limits_.phase_gpu_seconds.count(phase)) throw std::invalid_argument("phase has no authorized budget");
    double phase_remaining = limits_.phase_gpu_seconds.at(phase) - data_.at("phase_usage").value(phase, 0.0);
    if (limits_.carry == "forward") {
        const int index = phase_index(limits_, phase);
        for (int i = 0; i < index; ++i) {
            const auto& previous = limits_.phase_order[i];
            phase_remaining += limits_.phase_gpu_seconds.at(previous) - data_.at("phase_usage").value(previous, 0.0);
        }
    }
    return phase_remaining < remaining ? phase_remaining : remaining;
}

Json Budget::begin(const Json& identity, int devices, const std::string& phase, double watchdog_seconds, bool repair) {
    if (poisoned_) throw std::runtime_error("budget persistence failed; reopen and verify the ledger before continuing");
    if (!data_["active"].is_null()) throw std::runtime_error("a budget session is already active");
    if (!identity.is_object() || !identity.contains("config_sha256") || !identity["config_sha256"].is_string() ||
        identity["config_sha256"].get<std::string>().empty()) throw std::invalid_argument("budget identity requires config_sha256");
    if (devices < 0 || phase.empty() || finite(watchdog_seconds, "watchdog seconds") == 0)
        throw std::invalid_argument("invalid device count, phase or watchdog");
    const int index = phase_index(limits_, phase);
    if (limits_.carry == "forward" && index < data_["phase_cursor"].get<int>())
        throw std::invalid_argument("forward budget cannot return to an earlier phase");
    if (repair) {
        const auto retry = identity.value("retry_of", std::string());
        const auto previous = std::find_if(data_["sessions"].begin(), data_["sessions"].end(), [&](const Json& row) {
            return row.at("session_id") == retry && row.at("phase") == phase && row.at("status") != "completed" &&
                   row.at("identity").at("config_sha256") == identity["config_sha256"];
        });
        if (retry.empty() || previous == data_["sessions"].end())
            throw std::invalid_argument("repair requires a settled failed attempt with matching phase and config_sha256");
    }
    if (devices > 0 && allowance(phase, repair) <= 0) throw std::runtime_error("GPU budget exhausted");
    // CPU fixtures consume no GPU allowance, but still validate the phase.
    allowance(phase, repair);
    const double started = finite(monotonic_(), "monotonic clock");
    const double deadline_value = finite(started + watchdog_seconds, "watchdog deadline");
    Json next = data_;
    next["active"] = {{"session_id", identifier()}, {"identity", identity}, {"phase", phase}, {"devices", devices},
                      {"repair", repair}, {"started_at", finite(wall_(), "wall clock")}, {"started_monotonic", started},
                      {"deadline_monotonic", deadline_value}, {"supervisor_pid", getpid()}};
    next["phase_cursor"] = std::max(index, data_["phase_cursor"].get<int>());
    commit(std::move(next));
    return data_["active"];
}

double Budget::deadline() const {
    if (poisoned_) throw std::runtime_error("budget persistence failed");
    if (data_["active"].is_null()) throw std::runtime_error("no active budget session");
    const auto& active = data_["active"];
    double result = active.at("deadline_monotonic").get<double>();
    const int devices = active.at("devices").get<int>();
    if (devices > 0) result = std::min(result, active.at("started_monotonic").get<double>() +
                                      allowance(active.at("phase"), active.at("repair")) / devices);
    return result;
}

void Budget::check(double reserve_wall_seconds) const {
    if (poisoned_) throw std::runtime_error("budget persistence failed");
    finite(reserve_wall_seconds, "reservation seconds");
    if (data_["active"].is_null()) throw std::runtime_error("GPU work requires an active budget session");
    const auto& active = data_["active"];
    const double now = finite(monotonic_(), "monotonic clock");
    const double elapsed = now - active.at("started_monotonic").get<double>();
    if (elapsed < 0) throw std::runtime_error("monotonic clock reversed; budget remains unsettled");
    if (now + reserve_wall_seconds >= active.at("deadline_monotonic").get<double>())
        throw std::runtime_error("phase watchdog cannot admit the next complete work unit");
    const int devices = active.at("devices").get<int>();
    if (devices > 0 && (elapsed + reserve_wall_seconds) * devices >= allowance(active.at("phase"), active.at("repair")))
        throw std::runtime_error("GPU budget cannot admit the next complete work unit");
}

Json Budget::finish(const std::string& status, bool cleanup_verified) {
    if (poisoned_) throw std::runtime_error("budget persistence failed; preserve the unsettled session");
    if (data_["active"].is_null()) throw std::runtime_error("no active budget session");
    if (!completion_status(status)) throw std::invalid_argument("unknown budget completion status");
    if (!cleanup_verified) throw std::runtime_error("worker cleanup unverified; budget session remains unsettled");
    const double ended = finite(monotonic_(), "monotonic clock");
    const double elapsed = ended - data_["active"].at("started_monotonic").get<double>();
    if (elapsed < 0) throw std::runtime_error("monotonic clock reversed; budget session remains unsettled");
    const double charge = finite(elapsed * data_["active"].at("devices").get<int>(), "actual GPU charge");
    Json record = data_["active"];
    record.update({{"ended_at", finite(wall_(), "wall clock")}, {"ended_monotonic", ended}, {"elapsed_seconds", elapsed},
                   {"gpu_seconds", charge}, {"status", status}, {"cleanup_verified", true}});
    Json next = data_;
    next["sessions"].push_back(record);
    next["used_gpu_seconds"] = finite(next["used_gpu_seconds"].get<double>() + charge, "cumulative GPU charge");
    const std::string pool = record["repair"].get<bool>() ? "used_repair_gpu_seconds" : "used_regular_gpu_seconds";
    next[pool] = next[pool].get<double>() + charge;
    if (!record["repair"].get<bool>()) {
        const auto phase = record["phase"].get<std::string>();
        next["phase_usage"][phase] = next["phase_usage"].value(phase, 0.0) + charge;
    }
    next["active"] = nullptr;
    commit(std::move(next));
    return record;
}
}
