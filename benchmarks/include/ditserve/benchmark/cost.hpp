#pragma once
#include "ditserve/benchmark/workload.hpp"

namespace ditserve {
struct CostModel {
    std::string provenance = "synthetic", description = "Uncalibrated illustrative costs at 512x512, one frame";
    double compute_ms = 12, reuse_ms = 2, probe_ms = 0.2, batch_increment = 0.55, packing_ms = 0.05;
    double latent_mb = 8, cache_mb = 48, workspace_mb = 16, transfer_gbps = 12, transfer_setup_ms = 0.15, launch_ms = 0.1;
    void validate() const;
    static CostModel from_json(const Json& value);
    Json to_json() const;
    double scale(const RequestSpec& spec) const;
    double state_mb(const RequestSpec& spec, bool initialized = true) const;
    double workspace(const RequestSpec& spec, int batch_size = 1) const;
    double transfer_ms(double megabytes) const;
    double step_ms(const RequestSpec& spec, const std::string& path) const;
    double batch_ms(const std::vector<RequestSpec>& specs, const std::vector<std::string>& paths, bool split_paths) const;
    double full_service_ms(const RequestSpec& spec) const;
};
}
