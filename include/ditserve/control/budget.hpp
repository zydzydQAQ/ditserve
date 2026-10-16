#pragma once

#include "ditserve/types.hpp"
#include <filesystem>
#include <functional>
#include <map>
#include <optional>

namespace ditserve {
struct BudgetLimits {
    std::optional<double> total_gpu_seconds;
    std::map<std::string, double> phase_gpu_seconds;
    std::vector<std::string> phase_order;
    double repair_reserve_gpu_seconds = 0;
    std::string carry = "none";
};

// The ledger lock is held for this object's lifetime, including GPU cleanup.
class Budget {
public:
    using Clock = std::function<double()>;
    Budget(std::filesystem::path ledger, BudgetLimits limits = {}, Clock monotonic = {}, Clock wall = {});
    ~Budget();
    Budget(const Budget&) = delete;
    Budget& operator=(const Budget&) = delete;
    Json begin(const Json& identity, int devices, const std::string& phase,
               double watchdog_seconds, bool repair = false);
    void check(double reserve_wall_seconds = 0) const;
    Json finish(const std::string& status, bool cleanup_verified);
    double deadline() const; // Earliest watchdog or funded device-time deadline.
    Json snapshot() const { return data_; }
private:
    std::filesystem::path path_;
    BudgetLimits limits_;
    Clock monotonic_, wall_;
    int lock_ = -1;
    bool poisoned_ = false;
    Json data_;
    void save(const Json& value) const;
    void commit(Json value);
    double allowance(const std::string& phase, bool repair) const;
};
}
