#include "ditserve/random.hpp"
#include <cmath>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace ditserve {
Random::Random(std::uint64_t seed) : engine_(seed) {}
double Random::uniform() { return static_cast<double>(engine_() >> 11) * 0x1.0p-53; }
double Random::normal() {
    if (spare_) { const double result = *spare_; spare_.reset(); return result; }
    const double radius = std::sqrt(-2.0 * std::log(1.0 - uniform()));
    const double angle = 6.283185307179586476925286766559 * uniform();
    spare_ = radius * std::sin(angle);
    return radius * std::cos(angle);
}
double Random::exponential(double rate) {
    number(rate, "rate", true);
    return -std::log(1.0 - uniform()) / rate;
}
std::uint64_t Random::bounded(std::uint64_t limit) {
    if (!limit) throw std::invalid_argument("random bound must be positive");
    const auto threshold = static_cast<std::uint64_t>(-limit) % limit;
    std::uint64_t value;
    do { value = engine_(); } while (value < threshold);
    return value % limit;
}
Json Random::snapshot() const {
    std::ostringstream stream;
    stream.imbue(std::locale::classic()); stream << engine_;
    return {{"version", "mt19937_64-box-muller.v1"}, {"engine", stream.str()}, {"spare", spare_ ? Json(*spare_) : Json(nullptr)}};
}
Random Random::restore(const Json& value) {
    if (value.at("version") != "mt19937_64-box-muller.v1") throw std::invalid_argument("unsupported RNG version");
    Random result;
    std::istringstream stream(value.at("engine").get<std::string>());
    stream.imbue(std::locale::classic());
    if (!(stream >> result.engine_)) throw std::invalid_argument("invalid RNG engine state");
    stream >> std::ws;
    if (!stream.eof()) throw std::invalid_argument("trailing RNG engine state");
    if (!value.at("spare").is_null()) {
        if (!value.at("spare").is_number() || !std::isfinite(value.at("spare").get<double>()))
            throw std::invalid_argument("invalid Gaussian spare");
        result.spare_ = value.at("spare").get<double>();
    }
    return result;
}
}
