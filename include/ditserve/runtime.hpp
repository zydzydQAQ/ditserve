#pragma once

#include "ditserve/backend.hpp"
#include "ditserve/cost.hpp"
#include <memory>

namespace ditserve {
struct RuntimeConfig {
    std::string policy = "state_aware";
    int num_gpus = 1, max_batch = 4;
    double memory_mb = 512, host_memory_mb = 65536;
    bool verify_outputs = true;
    void validate() const;
    Json to_json() const;
    static RuntimeConfig from_json(const Json& value);
};

class Simulator {
public:
    explicit Simulator(RuntimeConfig config = {}, CostModel cost = {},
                       std::shared_ptr<const Backend> backend = std::make_shared<ReferenceBackend>());
    Json run(std::vector<RequestSpec> specs) const;
private:
    RuntimeConfig config_;
    CostModel cost_;
    std::shared_ptr<const Backend> backend_;
};
}
