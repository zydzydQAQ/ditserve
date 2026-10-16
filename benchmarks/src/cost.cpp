#include "ditserve/benchmark/cost.hpp"
#include <algorithm>
#include <stdexcept>

namespace ditserve {
Json CostModel::to_json() const {
    return {{"provenance", provenance}, {"description", description}, {"compute_ms", compute_ms}, {"reuse_ms", reuse_ms},
            {"probe_ms", probe_ms}, {"batch_increment", batch_increment}, {"packing_ms", packing_ms}, {"latent_mb", latent_mb},
            {"cache_mb", cache_mb}, {"workspace_mb", workspace_mb}, {"transfer_gbps", transfer_gbps},
            {"transfer_setup_ms", transfer_setup_ms}, {"launch_ms", launch_ms}};
}
void CostModel::validate() const {
    if (provenance != "synthetic") throw std::invalid_argument("cost model must be synthetic; measured profiles require calibration");
    const auto fields = to_json();
    for (auto it = fields.begin(); it != fields.end(); ++it)
        if (it.key() != "provenance" && it.key() != "description") number(it.value(), it.key());
    if (compute_ms <= 0 || reuse_ms <= 0 || transfer_gbps <= 0 || batch_increment <= 0 || batch_increment > 1)
        throw std::invalid_argument("invalid cost model time, bandwidth, or batch increment");
}
CostModel CostModel::from_json(const Json& value) {
    check_keys(value, {"provenance", "description", "compute_ms", "reuse_ms", "probe_ms", "batch_increment", "packing_ms", "latent_mb", "cache_mb", "workspace_mb", "transfer_gbps", "transfer_setup_ms", "launch_ms"});
    CostModel result;
    result.provenance = string_field(value, "provenance", result.provenance);
    result.description = value.value("description", result.description);
    for (auto entry : {std::make_pair("compute_ms", &result.compute_ms), {"reuse_ms", &result.reuse_ms}, {"probe_ms", &result.probe_ms},
         {"batch_increment", &result.batch_increment}, {"packing_ms", &result.packing_ms}, {"latent_mb", &result.latent_mb},
         {"cache_mb", &result.cache_mb}, {"workspace_mb", &result.workspace_mb}, {"transfer_gbps", &result.transfer_gbps},
         {"transfer_setup_ms", &result.transfer_setup_ms}, {"launch_ms", &result.launch_ms}})
        *entry.second = number_field(value, entry.first, *entry.second);
    result.validate(); return result;
}
double CostModel::scale(const RequestSpec& spec) const { return static_cast<double>(spec.height) * spec.width * spec.frames / (512.0 * 512); }
double CostModel::state_mb(const RequestSpec& spec, bool initialized) const { return scale(spec) * (latent_mb + (initialized && spec.cache_enabled ? cache_mb : 0)); }
double CostModel::workspace(const RequestSpec& spec, int batch_size) const { return scale(spec) * workspace_mb * batch_size; }
double CostModel::transfer_ms(double megabytes) const { return transfer_setup_ms + megabytes * 1024 * 1024 / (transfer_gbps * 1e6); }
double CostModel::step_ms(const RequestSpec& spec, const std::string& path) const {
    if (path != "compute" && path != "reuse") throw std::invalid_argument("unknown execution path");
    return scale(spec) * ((path == "compute" ? compute_ms : reuse_ms) + probe_ms) + launch_ms;
}
double CostModel::batch_ms(const std::vector<RequestSpec>& specs, const std::vector<std::string>& paths, bool split_paths) const {
    if (specs.empty() && paths.empty()) return 0;
    if (specs.empty() || specs.size() != paths.size()) throw std::invalid_argument("batch/path size mismatch");
    int compute = 0;
    for (std::size_t i = 0; i < specs.size(); ++i) {
        if (specs[i].shape_key() != specs[0].shape_key() || (paths[i] != "compute" && paths[i] != "reuse"))
            throw std::invalid_argument("batch requires homogeneous shapes/models and known paths");
        compute += paths[i] == "compute";
    }
    const int reuse = static_cast<int>(paths.size()) - compute;
    const auto group_cost = [&](int count, double unit) { return count ? scale(specs[0]) * (unit + probe_ms) * (1 + batch_increment * (count - 1)) + launch_ms : 0; };
    if (!split_paths) return group_cost(static_cast<int>(paths.size()), compute ? compute_ms : reuse_ms);
    return group_cost(compute, compute_ms) + group_cost(reuse, reuse_ms) + (compute && reuse ? packing_ms * specs.size() : 0);
}
double CostModel::full_service_ms(const RequestSpec& spec) const { return step_ms(spec, "compute") * spec.steps; }
}
