#pragma once

#include "ditserve/types.hpp"
#include <optional>
#include <random>

namespace ditserve {
// Distribution algorithms are fixed here; standard-library distributions vary.
class Random {
public:
    explicit Random(std::uint64_t seed = 0);
    double uniform();
    double normal();
    double exponential(double rate);
    std::uint64_t bounded(std::uint64_t limit);
    Json snapshot() const;
    static Random restore(const Json& value);
private:
    std::mt19937_64 engine_;
    std::optional<double> spare_;
};
}
