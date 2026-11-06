#include "ditserve/control/cluster.hpp"
#include "ditserve/control/policy.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace ditserve {
namespace {
bool power_of_two(int value) { return value > 0 && (value & (value - 1)) == 0; }
bool supported_degree(int value) { return value == 1 || value == 2 || value == 3 || value == 4 || value == 6 || value == 8; }
std::string degree_key(int degree) { return std::to_string(degree); }
std::string placement_key(const std::vector<int>& group) {
    if (group.empty()) throw std::invalid_argument("empty measured placement");
    for (std::size_t i = 0; i < group.size(); ++i)
        if (group[i] < 0 || (i && group[i] != group[i-1] + 1))
            throw std::invalid_argument("measured placement must name consecutive ranks in topology order");
    return std::to_string(group.front()) + "-" + std::to_string(group.back());
}

void validate_links(const Json& links) {
    if (!links.is_object()) throw std::invalid_argument("links must be an object");
    for (auto it = links.begin(); it != links.end(); ++it) {
        const auto& link = it.value();
        if (it.key() == "groups") {
            if (!link.is_object()) throw std::invalid_argument("links.groups must map aligned groups to link rates");
            for (auto entry = link.begin(); entry != link.end(); ++entry) {
                const auto dash = entry.key().find('-');
                int first = -1, last = -1;
                try { first = std::stoi(entry.key().substr(0, dash)); last = std::stoi(entry.key().substr(dash + 1)); } catch (...) {}
                const int size = last - first + 1;
                if (dash == std::string::npos || first < 0 || size < 2 || !power_of_two(size) || first % size != 0)
                    throw std::invalid_argument("links.groups keys name aligned groups as first-last, e.g. 0-1 or 4-7");
                if (!entry.value().is_object()) throw std::invalid_argument("links.groups entries are objects");
                if (entry.value().contains("gbps")) number_field(entry.value(), "gbps", 0, true);
                if (number_field(entry.value(), "sp_overlap") > 1) throw std::invalid_argument("sp_overlap must be in [0, 1]");
            }
            continue;
        }
        if (!link.is_object()) throw std::invalid_argument("link entries must be objects");
        if (link.contains("gbps")) number_field(link, "gbps", 0, true);
        if (link.contains("latency_ms")) number_field(link, "latency_ms");
        if (number_field(link, "sp_overlap") > 1) throw std::invalid_argument("sp_overlap must be in [0, 1]");
        if (!link.contains("concurrent_gbps")) continue;
        const auto& rates = link.at("concurrent_gbps");
        if (!rates.is_object()) throw std::invalid_argument("concurrent_gbps must map a number of concurrent SP groups to a rate");
        for (auto rate = rates.begin(); rate != rates.end(); ++rate) {
            const auto& key = rate.key();
            if (key.empty() || key.size() > 3 || !std::all_of(key.begin(), key.end(), [](char c) { return c >= '0' && c <= '9'; }) || std::stoi(key) < 2)
                throw std::invalid_argument("concurrent_gbps keys must count at least 2 SP groups");
            number(rate.value(), "concurrent_gbps", true);
        }
    }
}

// Policy behaviour comes from the registry (policy.hpp); see PolicyTraits.
using Policy = PolicyTraits;
const Policy& parse(const std::string& name) { return policy_traits(name); }
std::vector<int> group_of(const Json& request) {
    std::vector<int> group;
    for (const auto& gpu : request.value("group", Json::array())) group.push_back(static_cast<int>(integer(gpu, "group GPU")));
    return group;
}
// The form a placed request runs; older states name only the group.
Form form_of(const Json& request, const std::vector<int>& group) {
    if (!request.contains("form")) return Form{1, static_cast<int>(group.size())};
    const Form form = parse_form(request.at("form").get<std::string>());
    if (form.degree() != static_cast<int>(group.size())) throw std::invalid_argument("request form does not match its group size");
    return form;
}
}

Form parse_form(const std::string& name) {
    Form form;
    std::string rest = name;
    if (rest.rfind("cfg2", 0) == 0) { form.cfg = 2; rest = rest.substr(4); }
    if (rest.empty() && form.cfg == 2) return form;
    if (rest.size() == 4 && rest[0] == 'u' && rest[1] >= '1' && rest[1] <= '8' && rest[2] == 'r' && rest[3] >= '1' && rest[3] <= '8') {
        form.sp = rest[1] - '0';
        form.ring = rest[3] - '0';
        if (!supported_degree(form.sp) || !power_of_two(form.ring) || form.degree() > 8)
            throw std::invalid_argument("unknown parallel form: " + name);
        return form;
    }
    if (rest.size() == 5 && rest.rfind("sp", 0) == 0 && rest[2] >= '1' && rest[2] <= '8' && rest[3] == 'r' && rest[4] >= '1' && rest[4] <= '8') {
        form.sp = rest[2] - '0';
        form.ring = rest[4] - '0';
        if (!supported_degree(form.sp) || !power_of_two(form.ring) || form.degree() > 8)
            throw std::invalid_argument("unknown parallel form: " + name);
        return form;
    }
    if (rest.size() < 3 || rest.rfind("sp", 0) != 0 || rest.size() > 3 || rest[2] < '1' || rest[2] > '8')
        throw std::invalid_argument("unknown parallel form: " + name);
    form.sp = rest[2] - '0';
    if (!supported_degree(form.sp)) throw std::invalid_argument("unknown parallel form: " + name);
    return form;
}

std::string form_name(const Form& form) {
    if (form.cfg != 1 && form.cfg != 2) throw std::invalid_argument("cfg must be 1 or 2");
    if (!supported_degree(form.sp) || !power_of_two(form.ring)) throw std::invalid_argument("unsupported Wan SP or Ring degree");
    if (form.ring > 1) return (form.cfg == 2 ? "cfg2" : "") + std::string("u") + std::to_string(form.sp) + "r" + std::to_string(form.ring);
    return (form.cfg == 2 ? "cfg2" : "") + (form.sp == 1 && form.cfg == 2 ? std::string() : "sp" + std::to_string(form.sp));
}

bool cluster_policy(const std::string& policy) {
    try { parse(policy); return true; } catch (const std::invalid_argument&) { return false; }
}

std::vector<std::vector<int>> cluster_groups(int gpus, const std::vector<int>& degrees) {
    if (gpus < 1) throw std::invalid_argument("cluster requires at least one GPU");
    std::vector<std::vector<int>> groups;
    std::set<int> seen;
    for (int degree : degrees) {
        if (!supported_degree(degree)) throw std::invalid_argument("unsupported Wan SP degree");
        if (degree > gpus || !seen.insert(degree).second) continue;
        for (int start = 0; start + degree <= gpus; start += degree) {
            std::vector<int> group(degree);
            for (int i = 0; i < degree; ++i) group[i] = start + i;
            groups.push_back(group);
        }
    }
    return groups;
}

PreemptionTiming estimate_preemption_timing(double block_ms, int blocks_total, int blocks_done,
                                            double elapsed_ms, int check_blocks,
                                            int blocks_per_iteration, bool delayed_group_check) {
    if (!std::isfinite(block_ms) || block_ms <= 0 || blocks_total <= 0 || blocks_done < 0
        || !std::isfinite(elapsed_ms) || elapsed_ms < 0 || check_blocks <= 0 || blocks_per_iteration <= 0)
        throw std::invalid_argument("invalid block preemption timing");
    const double stop = block_ms * check_blocks * blocks_per_iteration * (delayed_group_check ? 2 : 1);
    const double left = std::max(block_ms, std::max(0, blocks_total - blocks_done) * block_ms - elapsed_ms);
    return {stop, left};
}

bool preemption_converts_deadline(double now_ms, double deadline_ms, double service_ms,
                                  double stop_ms, double boundary_ms) {
    if (!std::isfinite(now_ms) || !std::isfinite(deadline_ms) || !std::isfinite(service_ms)
        || !std::isfinite(stop_ms) || !std::isfinite(boundary_ms) || now_ms < 0
        || service_ms < 0 || stop_ms < 0 || boundary_ms < 0)
        throw std::invalid_argument("invalid deadline preemption decision");
    const double patient_finish = now_ms + boundary_ms + service_ms;
    const double preempted_finish = now_ms + std::min(stop_ms, boundary_ms) + service_ms;
    return patient_finish > deadline_ms && preempted_finish <= deadline_ms;
}

std::vector<std::string> select_preemption_groups(const std::vector<int>& image_affinities,
                                                 const std::vector<PreemptionGroup>& running) {
    std::set<int> pinned, occupied, released;
    std::set<std::string> ids;
    std::size_t flexible = 0;
    for (int gpu : image_affinities) {
        if (gpu < -1) throw std::invalid_argument("invalid image affinity");
        if (gpu < 0) ++flexible; else pinned.insert(gpu);
    }
    std::vector<std::string> selected;
    std::vector<const PreemptionGroup*> optional;
    int total = 0;
    for (const auto& item : running) {
        if (item.request_id.empty() || !ids.insert(item.request_id).second || item.group.empty()
            || !std::isfinite(item.stop_ms) || item.stop_ms < 0 || !std::isfinite(item.remaining_ms) || item.remaining_ms < 0
            || !std::isfinite(item.resume_ms) || item.resume_ms < 0)
            throw std::invalid_argument("invalid preemption group");
        bool needed = item.rung;
        for (int gpu : item.group) {
            if (gpu < 0 || !occupied.insert(gpu).second) throw std::invalid_argument("preemption groups must be disjoint");
            needed = needed || pinned.count(gpu);
        }
        if (needed) {
            released.insert(item.group.begin(), item.group.end());
            if (!item.rung) selected.push_back(item.request_id);
        } else {
            optional.push_back(&item);
            total += static_cast<int>(item.group.size());
        }
    }
    std::size_t reserved = 0;
    for (int gpu : released) reserved += pinned.count(gpu) != 0;
    const auto available = released.size() - reserved;
    if (flexible <= available) return selected;
    const auto need = std::min<std::size_t>(flexible - available, total);
    if (!need) return selected;  // Remaining pinned requests await their own busy worker.
    struct Choice { double cost = std::numeric_limits<double>::infinity(); std::vector<std::string> ids; };
    std::vector<Choice> choices(total + 1);
    choices[0].cost = 0;
    for (const auto* item : optional) {
        const int width = static_cast<int>(item->group.size());
        const double cost = width * (std::min(item->stop_ms, item->remaining_ms) + item->resume_ms);
        for (int count = total - width; count >= 0; --count) {
            if (!std::isfinite(choices[count].cost)) continue;
            const double next = choices[count].cost + cost;
            auto& target = choices[count + width];
            if (next < target.cost || (next == target.cost && choices[count].ids.size() + 1 < target.ids.size())) {
                target = choices[count]; target.cost = next; target.ids.push_back(item->request_id);
            }
        }
    }
    std::size_t best = need;
    for (std::size_t count = need + 1; count < choices.size(); ++count)
        if (choices[count].cost < choices[best].cost) best = count;
    selected.insert(selected.end(), choices[best].ids.begin(), choices[best].ids.end());
    return selected;
}

DeadlinePreemptionSelection select_deadline_preemption_groups(double now_ms,
    const std::vector<PreemptionDemand>& waiting, const std::vector<PreemptionGroup>& running) {
    select_preemption_groups({}, running);  // Validate disjoint groups and finite costs.
    auto demands = waiting;
    for (const auto& demand : demands) {
        if (demand.affinity < -1) throw std::invalid_argument("invalid image affinity");
        preemption_converts_deadline(now_ms, demand.deadline_ms, demand.service_ms, 0, 0);
    }
    std::stable_sort(demands.begin(), demands.end(), [](const auto& a, const auto& b) {
        if (a.deadline_ms != b.deadline_ms) return a.deadline_ms < b.deadline_ms;
        return a.affinity >= 0 && b.affinity < 0;
    });
    auto groups = running;
    std::set<int> reserved;
    DeadlinePreemptionSelection result;
    for (const auto& demand : demands) {
        auto available = [&](const PreemptionGroup& group) {
            for (int gpu : group.group)
                if (!reserved.count(gpu) && (demand.affinity < 0 || gpu == demand.affinity)) return gpu;
            return -1;
        };
        const bool tight_demand_slo = demand.deadline_ms >= now_ms + demand.service_ms * 0.85
                                      && (demand.deadline_ms - now_ms) <= demand.service_ms * 1.35;
        auto grace_for = [&](const PreemptionGroup& group) {
            return tight_demand_slo ? std::max(group.stop_ms * 1.5, 120.0) : 0.0;
        };
        double boundary = std::numeric_limits<double>::infinity();
        int reused = -1;
        double reuse_stop = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < groups.size(); ++i) {
            const auto& group = groups[i];
            if (available(group) < 0) continue;
            const double response = group.rung ? std::min(group.stop_ms, group.remaining_ms) : group.remaining_ms;
            boundary = std::min(boundary, response);
            if (group.rung && now_ms + response + demand.service_ms <= demand.deadline_ms + grace_for(group)
                && response < reuse_stop) {
                reused = static_cast<int>(i); reuse_stop = response;
            }
        }
        if (reused >= 0) {
            reserved.insert(available(groups[reused]));
            continue;
        }
        int best = -1;
        double best_cost = std::numeric_limits<double>::infinity();
        if (std::isfinite(boundary) && now_ms + boundary + demand.service_ms > demand.deadline_ms) {
            for (std::size_t i = 0; i < groups.size(); ++i) {
                const auto& group = groups[i];
                if (group.rung || available(group) < 0 || !preemption_converts_deadline(
                        now_ms, demand.deadline_ms, demand.service_ms, group.stop_ms, group.remaining_ms)) continue;
                const double cost = group.group.size() * (std::min(group.stop_ms, group.remaining_ms) + group.resume_ms);
                if (cost < best_cost || (cost == best_cost && best >= 0 && group.group.size() < groups[best].group.size())) {
                    best = static_cast<int>(i); best_cost = cost;
                }
            }
            if (best < 0 && tight_demand_slo) {
                for (std::size_t i = 0; i < groups.size(); ++i) {
                    const auto& group = groups[i];
                    const double doorbell_grace_ms = grace_for(group);
                    if (group.rung || available(group) < 0 || group.stop_ms >= group.remaining_ms) continue;
                    const double preempted_finish = now_ms + std::min(group.stop_ms, group.remaining_ms) + demand.service_ms;
                    if (preempted_finish > demand.deadline_ms + doorbell_grace_ms) continue;
                    const double cost = group.group.size() * (std::min(group.stop_ms, group.remaining_ms) + group.resume_ms);
                    if (cost < best_cost || (cost == best_cost && best >= 0 && group.group.size() < groups[best].group.size())) {
                        best = static_cast<int>(i); best_cost = cost;
                    }
                }
            }
        }
        if (best < 0) { ++result.deadline_skips; continue; }
        result.selected.push_back(groups[best].request_id);
        groups[best].rung = true;
        reserved.insert(available(groups[best]));
    }
    return result;
}

ClusterCost::ClusterCost(const Json& profile) : profile_(profile) {
    static const std::set<std::string> provenance = {"measured", "extrapolated", "synthetic", "mixed"};
    if (!profile.is_object() || !provenance.count(profile.value("provenance", "")))
        throw std::invalid_argument("cost profile provenance must be measured, extrapolated, synthetic or mixed");
    const auto& classes = profile.at("classes");
    if (!classes.is_object() || classes.empty()) throw std::invalid_argument("cost profile requires request classes");
    for (auto it = classes.begin(); it != classes.end(); ++it) {
        const auto& row = it.value();
        const auto kind = row.at("kind").get<std::string>();
        if (kind == "image") {
            number_field(row, "service_ms", 0, true);
            if (row.contains("batch_ms")) {
                if (!row.at("batch_ms").is_object() || row.at("batch_ms").empty()) throw std::invalid_argument("batch_ms maps batch sizes to service times");
                for (auto entry = row.at("batch_ms").begin(); entry != row.at("batch_ms").end(); ++entry) {
                    const auto& key = entry.key();
                    if (key.empty() || !std::all_of(key.begin(), key.end(), [](char c) { return c >= '0' && c <= '9'; }) || std::stoi(key) < 1)
                        throw std::invalid_argument("batch_ms keys are batch sizes");
                    number(entry.value(), "batch_ms", true);
                }
            }
            if (row.contains("steps") && int_field(row, "steps", 0) <= 0) throw std::invalid_argument("image steps must be positive");
            if (row.contains("prepare_ms")) number_field(row, "prepare_ms");
            if (row.contains("finish_ms")) number_field(row, "finish_ms");
            if (row.contains("batch_step_ms")) {
                if (!row.at("batch_step_ms").is_object() || row.at("batch_step_ms").empty())
                    throw std::invalid_argument("batch_step_ms maps continuous batch sizes to denoise round times");
                for (auto entry = row.at("batch_step_ms").begin(); entry != row.at("batch_step_ms").end(); ++entry) {
                    const auto& key = entry.key();
                    if (key.empty() || !std::all_of(key.begin(), key.end(), [](char c) { return c >= '0' && c <= '9'; })
                        || std::stoi(key) < 1 || std::stoi(key) > 8)
                        throw std::invalid_argument("batch_step_ms keys must be in [1, 8]");
                    number(entry.value(), "batch_step_ms", true);
                }
                if (!row.at("batch_step_ms").contains("1")) throw std::invalid_argument("batch_step_ms requires a batch-1 calibration");
            }
            continue;
        }
        if (kind != "video") throw std::invalid_argument("cost class kind must be video or image");
        int_field(row, "steps", 0); int_field(row, "blocks_per_step", 1); int_field(row, "decode_chunks", 1);
        if (!power_of_two(int_field(row, "min_degree", 1))) throw std::invalid_argument("min_degree must be a power of two");
        number_field(row, "step_compute_ms", 0, true); number_field(row, "finish_ms");
        number_field(row, "prepare_ms"); number_field(row, "comm_mb_per_step"); number_field(row, "collectives_per_step");
        number_field(row, "cfg_exchange_mb"); number_field(row, "resume_ms");
        if (row.contains("forms")) {
            if (!row.at("forms").is_object()) throw std::invalid_argument("forms must map parallel-form names to measured costs");
            for (auto entry = row.at("forms").begin(); entry != row.at("forms").end(); ++entry) {
                const auto form = parse_form(entry.key());
                const auto& measured = entry.value();
                if (!measured.is_object() || (!measured.contains("step_ms") && !measured.contains("placements")))
                    throw std::invalid_argument("measured form requires step_ms or placements");
                if (measured.contains("step_ms")) number_field(measured, "step_ms", 0, true);
                if (measured.contains("group")) {
                    const auto group = group_of(measured);
                    if (group.size() != static_cast<std::size_t>(form.degree()) || !measured.contains("step_ms"))
                        throw std::invalid_argument("measured form group must match its degree and step_ms");
                    placement_key(group);
                }
                if (measured.contains("placements")) {
                    const auto& placements = measured.at("placements");
                    if (!placements.is_object() || placements.empty()) throw std::invalid_argument("placements must map rank ranges to measured costs");
                    for (auto place = placements.begin(); place != placements.end(); ++place) {
                        const auto dash = place.key().find('-');
                        int first = -1, last = -1;
                        try { first = std::stoi(place.key().substr(0,dash)); last = std::stoi(place.key().substr(dash+1)); } catch (...) {}
                        if (first < 0 || last-first+1 != form.degree()
                            || place.key() != std::to_string(first)+"-"+std::to_string(last))
                            throw std::invalid_argument("placement range must match measured form degree");
                        number_field(place.value(), "step_ms", 0, true);
                    }
                }
            }
        }
    }
    efficiency_ = number_field(profile, "sp_efficiency", 1, true);
    if (efficiency_ > 1) throw std::invalid_argument("sp_efficiency must be in (0, 1]");
    resize_ms_ = number_field(profile, "resize_ms");
    overhead_ms_ = number_field(profile, "quantum_overhead_ms");
    block_overhead_ms_ = number_field(profile, "block_overhead_ms");
    overlap_ = number_field(profile, "sp_overlap");
    if (overlap_ > 1) throw std::invalid_argument("sp_overlap must be in [0, 1]");
    pair_overlap_ = number_field(profile, "pair_overlap");
    if (pair_overlap_ > 1) throw std::invalid_argument("pair_overlap must be in [0, 1]");
    validate_links(profile.value("links", Json::object()));
    for (const auto& row : classes)
        if (row.contains("links")) validate_links(row.at("links"));
}

const Json& ClusterCost::klass(const std::string& name) const {
    const auto& classes = profile_.at("classes");
    if (!classes.contains(name)) throw std::invalid_argument("unknown cost class: " + name);
    return classes.at(name);
}

bool ClusterCost::placement_invariant(const std::string& name) const {
    static const Json empty_links = Json::object();
    const auto& row = klass(name);
    const Json& links = row.contains("links") ? row.at("links") : (profile_.contains("links") ? profile_.at("links") : empty_links);
    if (links.contains("groups") && !links.at("groups").empty()) return false;
    for (const auto& link : links)
        if (link.contains("concurrent_gbps") && !link.at("concurrent_gbps").empty()) return false;
    if (row.contains("forms")) for (const auto& form : row.at("forms"))
        if (form.contains("group") || form.contains("placements")) return false;
    return true;
}

double ClusterCost::cache_skip_ratio(const std::string& name, double cache_threshold) const {
    if (cache_threshold < 0.0 || cache_threshold >= 1.0)
        throw std::invalid_argument("cache_threshold must be in [0, 1)");
    if (cache_threshold == 0.0) return 0.0;
    const auto& row = klass(name);
    if (row.contains("cache_skip_table") && row.at("cache_skip_table").is_object()) {
        std::map<double, double> table = {{0.0, 0.0}};
        for (auto it = row.at("cache_skip_table").begin(); it != row.at("cache_skip_table").end(); ++it)
            table[std::stod(it.key())] = it.value().get<double>();
        auto upper = table.lower_bound(cache_threshold);
        if (upper != table.end() && std::abs(upper->first - cache_threshold) <= 1e-9) return upper->second;
        if (upper == table.end()) return std::prev(table.end())->second;
        const auto lower = std::prev(upper);
        return lower->second + (upper->second - lower->second) * (cache_threshold - lower->first) / (upper->first - lower->first);
    }
    return std::min(0.55, 2.2 * cache_threshold);
}

ClusterCost::Terms ClusterCost::terms(const std::string& name, const Form& form, int sp_groups, const std::vector<int>* group,
                                      double cache_threshold, const char** source) const {
    const auto& row = klass(name);
    if (source) *source = cache_threshold > 0 ? "analytic_cache" : sp_groups > 1 ? "analytic_contention" : "analytic_no_matching_table";
    if (row.at("kind") != "video") throw std::invalid_argument("image classes have no solver steps");
    form_name(form);   // validates
    if (group && group->size() != static_cast<std::size_t>(form.degree()))
        throw std::invalid_argument("group size does not match the form");
    // Each resolution has its own calibrated communication model. An explicit
    // class table replaces the global table; missing entries must not borrow a
    // fit from another shape.
    static const Json empty_links = Json::object();
    const Json& links = row.contains("links") ? row.at("links") : (profile_.contains("links") ? profile_.at("links") : empty_links);
    auto link_of = [&](int sp) -> const Json& {
        if (!links.contains(degree_key(sp))) throw std::invalid_argument("cost profile has no link model for SP degree " + degree_key(sp));
        return links.at(degree_key(sp));
    };
    const double cfg = form.cfg, sp = form.sp, ring = form.ring, sp_tot = form.sp_total();
    Terms t;
    if (cache_threshold == 0.0 && sp_groups <= 1 && row.contains("forms")) {
        const auto& measured = row.at("forms");
        const auto key = form_name(form);
        if (measured.is_object() && measured.contains(key)) {
            const auto& entry = measured.at(key);
            const Json* match = nullptr;
            if (group && entry.contains("placements")) {
                const auto place = placement_key(*group);
                if (entry.at("placements").contains(place)) {
                    match = &entry.at("placements").at(place);
                    if (source) *source = "placement_table";
                }
            }
            if (!match && entry.contains("group") && group && group_of(entry) == *group) {
                match = &entry; if (source) *source = "group_scoped_table";
            }
            // Legacy tables remain valid for homogeneous profiles only. A missing
            // placement is unknown, not permission to reuse another group's sample.
            if (!match && entry.contains("step_ms") && !entry.contains("group") && !entry.contains("placements")
                && (!links.contains("groups") || links.at("groups").empty())) {
                match = &entry; if (source) *source = "legacy_table";
            }
            if (match) {
                t.compute = number_field(*match, "step_ms", 0, true);
                t.blocks = block_overhead_ms_ * blocks_per_step(name) / form.cfg;
                return t;
            }
        }
    }
    t.compute = number_field(row, "step_compute_ms", 0, true) / (cfg * sp_tot * efficiency_);
    if (form.sp > 1) {
        const auto& link = link_of(form.sp);
        double gbps = number_field(link, "gbps", 0, true), latency = number_field(link, "latency_ms");
        double overlap = link.contains("sp_overlap") ? number_field(link, "sp_overlap") : overlap_;
        if (overlap > 1) throw std::invalid_argument("sp_overlap must be in [0, 1]");
        // On a named placement the form runs at its slowest SP subgroup's rate (the halves of a
        // CFG pair or Ring subgroups each run Ulysses SP on their own GPUs).
        if (group && links.contains("groups")) {
            if (static_cast<int>(group->size()) != form.degree()) throw std::invalid_argument("group size does not match the form");
            bool first = true;
            const int sub_count = form.cfg * form.ring;
            for (int sub = 0; sub < sub_count; ++sub) {
                const auto key = std::to_string((*group)[sub * form.sp]) + "-" + std::to_string((*group)[(sub + 1) * form.sp - 1]);
                const auto& table = links.at("groups");
                if (!table.contains(key)) continue;
                const auto& entry = table.at(key);
                // A named subgroup replaces the degree's values; several subgroups take the slowest.
                const double own_gbps = entry.contains("gbps") ? number_field(entry, "gbps", 0, true) : number_field(link, "gbps", 0, true);
                const double own_latency = entry.contains("latency_ms") ? number_field(entry, "latency_ms") : number_field(link, "latency_ms");
                const double own_overlap = entry.contains("sp_overlap") ? number_field(entry, "sp_overlap")
                                           : link.contains("sp_overlap") ? number_field(link, "sp_overlap") : overlap_;
                gbps = first ? own_gbps : std::min(gbps, own_gbps);
                latency = first ? own_latency : std::max(latency, own_latency);
                overlap = first ? own_overlap : std::min(overlap, own_overlap);
                first = false;
            }
        }
        if (sp_groups >= 2 && link.contains("concurrent_gbps")) {
            int matched = 1;
            double contended = number_field(link, "gbps", 0, true);
            for (auto it = link.at("concurrent_gbps").begin(); it != link.at("concurrent_gbps").end(); ++it) {
                const int groups = std::stoi(it.key());
                if (groups <= sp_groups && groups > matched) { matched = groups; contended = it.value().get<double>(); }
            }
            gbps *= contended / number_field(link, "gbps", 0, true);
        }
        t.comm = (number_field(row, "comm_mb_per_step") / (cfg * ring)) * (sp - 1) / (sp * sp) / gbps;
        // Branch interleaving hides part of the shorter term behind the other branch;
        // with the branches on separate halves (cfg = 2) each rank has one branch.
        t.overlap = form.cfg == 1 ? overlap : 0;
        t.latency = number_field(row, "collectives_per_step") / cfg * latency;
    }
    if (form.ring > 1) {
        const int rkey = links.contains(degree_key(form.sp_total())) ? form.sp_total() : form.ring;
        const auto& rlink = link_of(rkey);
        const double ring_gbps = number_field(rlink, "gbps", 0, true), ring_lat = number_field(rlink, "latency_ms");
        const double ring_ov = std::min(1.0, std::max(0.0, number_field(profile_, "ring_overlap", 0.75)));
        const double ring_quant = std::min(1.0, std::max(0.25, number_field(profile_, "ring_quant_scale", 1.0)));
        t.comm += (number_field(row, "comm_mb_per_step") * 0.5 * ring_quant / cfg) * (ring - 1) / (sp_tot * ring) / ring_gbps * (1.0 - ring_ov);
        t.latency += (ring - 1) * ring_lat * (1.0 - ring_ov);
    }
    if (form.cfg == 2) {
        const auto& pair = link_of(2);
        t.exchange = number_field(row, "cfg_exchange_mb") / number_field(pair, "gbps", 0, true) + number_field(pair, "latency_ms");
    }
    if (cache_threshold > 0.0) {
        const double keep = 1.0 - cache_skip_ratio(name, cache_threshold);
        t.compute *= keep;
        t.comm *= keep;
        t.latency *= keep;
        t.exchange *= keep;
    }
    // Each rank executes blocks_per_step / cfg blocks of a step (a CFG pair splits the two branches).
    t.blocks = block_overhead_ms_ * blocks_per_step(name) / form.cfg;
    return t;
}

double ClusterCost::step_ms(const std::string& name, const Form& form, int sp_groups, const std::vector<int>* group,
                            double cache_threshold, const char** source) const {
    const Terms t = terms(name, form, sp_groups, group, cache_threshold, source);
    return t.compute + t.comm - t.overlap * std::min(t.compute, t.comm) + t.latency + t.exchange + t.blocks;
}

double ClusterCost::pair_ms(const std::string& first, const std::string& second, const Form& form, int sp_groups,
                            const std::vector<int>* group) const {
    const Terms a = terms(first, form, sp_groups, group), b = terms(second, form, sp_groups, group);
    const double compute = a.compute + b.compute, comm = a.comm + b.comm;
    return compute + comm - pair_overlap_ * std::min(compute, comm) + a.latency + b.latency + a.exchange + b.exchange + a.blocks + b.blocks;
}

double ClusterCost::batch_ms(const std::string& name, int size) const {
    const auto& row = klass(name);
    if (row.at("kind") != "image") throw std::invalid_argument("only image classes batch");
    if (size < 1) throw std::invalid_argument("batch size must be positive");
    const double single = number_field(row, "service_ms", 0, true);
    if (!row.contains("batch_ms")) return size * single;
    std::map<int, double> table = {{1, single}};
    for (auto entry = row.at("batch_ms").begin(); entry != row.at("batch_ms").end(); ++entry) table[std::stoi(entry.key())] = entry.value().get<double>();
    auto upper = table.lower_bound(size);
    if (upper != table.end() && upper->first == size) return upper->second;
    if (upper == table.end()) {   // beyond the table: the last listed per-image rate
        const auto last = std::prev(table.end());
        return last->second / last->first * size;
    }
    const auto lower = std::prev(upper);
    return lower->second + (upper->second - lower->second) * (size - lower->first) / (upper->first - lower->first);
}

double ClusterCost::continuous_step_ms(const std::string& name, int size) const {
    const auto& row = klass(name);
    if (row.at("kind") != "image") throw std::invalid_argument("only image classes use continuous batching");
    if (size < 1 || size > 8) throw std::invalid_argument("continuous batch size must be in [1, 8]");
    if (!row.contains("batch_step_ms")) throw std::invalid_argument("continuous_batch requires measured batch_step_ms");
    std::map<int, double> table;
    for (auto entry = row.at("batch_step_ms").begin(); entry != row.at("batch_step_ms").end(); ++entry)
        table[std::stoi(entry.key())] = entry.value().get<double>();
    auto upper = table.lower_bound(size);
    if (upper != table.end() && upper->first == size) return upper->second;
    if (upper == table.end()) throw std::invalid_argument("continuous batch size exceeds its calibration table");
    const auto lower = std::prev(upper);
    return lower->second + (upper->second - lower->second) * (size - lower->first) / (upper->first - lower->first);
}

double ClusterCost::continuous_step_ms(const std::string& name, double effective_size) const {
    const auto& row = klass(name);
    if (row.at("kind") != "image") throw std::invalid_argument("only image classes use continuous batching");
    if (!std::isfinite(effective_size) || effective_size <= 0.0 || effective_size > 8.0 + 1e-6)
        throw std::invalid_argument("continuous batch size must be in (0, 8]");
    if (!row.contains("batch_step_ms")) throw std::invalid_argument("continuous_batch requires measured batch_step_ms");
    std::map<int, double> table;
    for (auto entry = row.at("batch_step_ms").begin(); entry != row.at("batch_step_ms").end(); ++entry)
        table[std::stoi(entry.key())] = entry.value().get<double>();
    double clamped = std::min(8.0, effective_size);
    if (clamped <= 1.0 + 1e-9) {
        if (std::abs(clamped - 1.0) <= 1e-9) return table.at(1);
        const double r = std::max(0.25, clamped);
        return table.at(1) * (0.10 + 0.90 * r);
    }
    if (std::abs(std::round(clamped) - clamped) <= 1e-9) clamped = std::round(clamped);
    const int ceil_key = static_cast<int>(std::ceil(clamped));
    auto upper = table.lower_bound(ceil_key);
    if (upper != table.end() && std::abs(upper->first - clamped) <= 1e-9) return upper->second;
    if (upper == table.end()) throw std::invalid_argument("continuous batch size exceeds its calibration table");
    const auto lower = std::prev(upper);
    return lower->second + (upper->second - lower->second) * (clamped - lower->first) / (upper->first - lower->first);
}

double ClusterCost::continuous_effective_batch_size(const std::string& name, const std::vector<const Json*>& batch,
                                                    bool sequence_packing) const {
    const auto& row = klass(name);
    if (row.at("kind") != "image") throw std::invalid_argument("only image classes use continuous batching");
    if (batch.empty()) throw std::invalid_argument("continuous batch must not be empty");
    auto pixels = [this](const Json& req) -> double {
        for (const char* key : {"batch_spec", "spec"}) {
            if (req.contains(key) && req.at(key).is_object()) {
                const auto& s = req.at(key);
                if (s.contains("height") && s.contains("width") && s.at("height").is_number() && s.at("width").is_number()) {
                    const double h = s.at("height").get<double>();
                    const double w = s.at("width").get<double>();
                    if (h > 0.0 && w > 0.0) return h * w;
                }
            }
        }
        if (req.contains("class") && req.at("class").is_string()) {
            const auto& mrow = klass(req.at("class").get<std::string>());
            if (mrow.contains("base_pixels") && mrow.at("base_pixels").is_number() && mrow.at("base_pixels").get<double>() > 0.0)
                return mrow.at("base_pixels").get<double>();
            if (mrow.contains("spec") && mrow.at("spec").is_object()) {
                const auto& s = mrow.at("spec");
                if (s.contains("height") && s.contains("width") && s.at("height").is_number() && s.at("width").is_number()) {
                    const double h = s.at("height").get<double>();
                    const double w = s.at("width").get<double>();
                    if (h > 0.0 && w > 0.0) return h * w;
                }
            }
        }
        return 0.0;
    };
    std::vector<double> p;
    p.reserve(batch.size());
    for (const Json* m : batch) {
        if (!m) throw std::invalid_argument("null request in continuous batch");
        const double pi = pixels(*m);
        if (pi <= 0.0) return static_cast<double>(batch.size());
        p.push_back(pi);
    }
    double ref_pixels = 0.0;
    if (row.contains("base_pixels") && row.at("base_pixels").is_number() && row.at("base_pixels").get<double>() > 0.0) {
        ref_pixels = row.at("base_pixels").get<double>();
    } else if (row.contains("spec") && row.at("spec").is_object()
               && row.at("spec").contains("height") && row.at("spec").contains("width")
               && row.at("spec").at("height").is_number() && row.at("spec").at("width").is_number()) {
        const double rh = row.at("spec").at("height").get<double>();
        const double rw = row.at("spec").at("width").get<double>();
        if (rh > 0.0 && rw > 0.0) ref_pixels = rh * rw;
    }
    if (ref_pixels <= 0.0) ref_pixels = *std::max_element(p.begin(), p.end());
    const double min_p = *std::min_element(p.begin(), p.end());
    const double max_p = *std::max_element(p.begin(), p.end());
    if (std::abs(max_p - min_p) <= 1e-9 && std::abs(max_p - ref_pixels) <= 1e-9)
        return static_cast<double>(batch.size());
    if (sequence_packing) {
        double total_units = 0.0;
        for (double pi : p) total_units += pi / ref_pixels;
        return std::min(8.0, std::max(0.25, total_units));
    }
    return std::min(8.0, std::max(0.25, static_cast<double>(batch.size()) * (max_p / ref_pixels)));
}

double ClusterCost::continuous_batch_step_ms(const std::string& name, const std::vector<const Json*>& batch,
                                             bool sequence_packing) const {
    return continuous_step_ms(name, continuous_effective_batch_size(name, batch, sequence_packing));
}

int ClusterCost::blocks_per_step(const std::string& name) const { return int_field(klass(name), "blocks_per_step", 1); }

int ClusterCost::execution_blocks_per_step(const std::string& name, const Form& form, bool rank_local_blocks, bool cfg_folded) const {
    const int blocks = blocks_per_step(name);
    if (!rank_local_blocks) return blocks;
    const int divisor = cfg_folded ? 2 : form.cfg;
    if ((form.cfg != 1 && form.cfg != 2) || blocks % divisor != 0)
        throw std::invalid_argument("rank-local block count requires an integral CFG branch partition");
    return blocks / divisor;
}

double ClusterCost::completed_blocks(const Json& request, const Form& form) const {
    const int done = int_field(request, "blocks_done", 0, 0);
    if (!bool_field(request, "rank_local_blocks", false)) return done;
    const auto name = request.at("class").get<std::string>();
    // At a boundary cfg_folded describes the previous step. Eligibility alone
    // likewise cannot establish that the next step actually executes folded.
    const bool folded = done > 0 && bool_field(request, "cfg_folded", false);
    execution_blocks_per_step(name, form, true, folded);
    if (done == 0) return 0;
    if (!request.contains("form")) throw std::invalid_argument("rank-local mid-step progress needs its observed form");
    const Form observed = parse_form(request.at("form").get<std::string>());
    if (done > execution_blocks_per_step(name, observed, true, folded))
        throw std::invalid_argument("rank-local progress exceeds one solver step");
    return double(done) * (folded ? 2 : observed.cfg) / (folded ? 2 : form.cfg);
}

int ClusterCost::decode_chunks(const std::string& name) const { return int_field(klass(name), "decode_chunks", 1); }

double ClusterCost::chunk_ms(const std::string& name) const { return number_field(klass(name), "finish_ms") / decode_chunks(name); }

double ClusterCost::resume_ms(const std::string& name) const { return number_field(klass(name), "resume_ms"); }

double ClusterCost::block_ms(const std::string& name, const Form& form, int sp_groups, const std::vector<int>* group,
                             double cache_threshold, bool rank_local_blocks, bool cfg_folded) const {
    return step_ms(name, form, sp_groups, group, cache_threshold) / execution_blocks_per_step(name, form, rank_local_blocks, cfg_folded);
}

double ClusterCost::remaining_ms(const Json& request, const Form& form, int sp_groups, const std::vector<int>* group,
                                 double cache_threshold, const char** source) const {
    const auto name = request.at("class").get<std::string>();
    const auto& row = klass(name);
    if (row.at("kind") == "image") {
        if (source) *source = request.value("continuous_batch", false) ? "image_continuous" : "image_service";
        if (!request.value("continuous_batch", false)) return number_field(row, "service_ms");
        const int steps = int_field(request, "steps_total", int_field(row, "steps", 2), 1);
        const int done = int_field(request, "steps_done", 0, 0);
        double total = request.value("prepared", false) ? 0 : number_field(row, "prepare_ms");
        total += std::max(0, steps - done) * continuous_step_ms(name, 1) + number_field(row, "finish_ms");
        return total;
    }
    const int steps = int_field(request, "steps_total", int_field(row, "steps", 0), 1), done = int_field(request, "steps_done", 0, 0);
    const double blocks = completed_blocks(request, form);
    double total = request.value("prepared", false) ? 0 : number_field(row, "prepare_ms");
    total += std::max(0, steps - done) * step_ms(name, form, sp_groups, group, cache_threshold, source)
           - blocks * block_ms(name, form, sp_groups, group, cache_threshold, bool_field(request, "rank_local_blocks", false),
                               int_field(request, "blocks_done", 0, 0) > 0 && bool_field(request, "cfg_folded", false));
    return total + (decode_chunks(name) - std::min(int_field(request, "decode_done", 0, 0), decode_chunks(name))) * chunk_ms(name);
}

void ClusterPlanTrace::enforce_bounds() {
    Json kept = Json::array(); row_bytes = 0;
    for (auto& row : rows) {
        const auto bytes = row.dump().size();
        if (kept.size() >= max_rows || row_bytes + bytes > max_bytes) { ++dropped_rows; continue; }
        row_bytes += bytes; kept.push_back(std::move(row));
    }
    rows = std::move(kept);
}

Json cluster_plan(const Json& state, const std::string& name, const ClusterCost& cost, const Json& options, ClusterPlanTrace* trace) {
    if (trace) {
        if (trace->max_rows < 1 || trace->max_rows > 64 || trace->max_bytes < 1 || trace->max_bytes > 65536 || trace->focus.size() > 4)
            throw std::invalid_argument("invalid cluster decision trace bounds");
        trace->rows.clear(); trace->row_bytes = trace->candidates = trace->filtered_rows = trace->dropped_rows = 0;
    }
    const Policy& policy = parse(name);
    const double now = number(state.at("now_ms"), "now_ms");
    const int gpus = static_cast<int>(integer(state.at("gpus"), "gpus"));
    const bool allow_cfg_forms = bool_field(options, "allow_cfg_forms", policy.allow_cfg_forms);
    // Parallel forms the policy may choose; the group sizes follow from them.
    std::vector<Form> forms;
    if (policy.fixed_degree) forms = {policy.fixed_form};
    else if (options.contains("forms")) {
        if (options.contains("degrees")) throw std::invalid_argument("options.forms replaces options.degrees");
        for (const auto& item : options.at("forms")) {
            if (!item.is_string()) throw std::invalid_argument("options.forms lists form names");
            forms.push_back(parse_form(item.get<std::string>()));
        }
        if ((policy.deadline_dp || policy.best_fit) && !allow_cfg_forms) {
            std::vector<Form> sp_only;
            for (const auto& f : forms) if (f.cfg == 1) sp_only.push_back(f);
            if (!sp_only.empty()) forms = std::move(sp_only);
        }
    } else if ((policy.knapsack || policy.prune_dominated || policy.min_video_degree > 1
                || policy.image_placement == "pair_fit" || policy.id == "diag-2d-usp" || allow_cfg_forms)
               && !options.contains("degrees")) {
        static const Json empty_links = Json::object();
        const auto& links = cost.profile().contains("links") ? cost.profile().at("links") : empty_links;
        forms.push_back(parse_form("sp1"));
        if (links.contains("2")) { forms.push_back(parse_form("cfg2")); forms.push_back(parse_form("sp2")); }
        if (links.contains("4")) {
            if (links.contains("2")) { forms.push_back(parse_form("cfg2sp2")); forms.push_back(parse_form("u2r2")); }
            forms.push_back(parse_form("sp4"));
        }
        if (links.contains("8")) {
            if (links.contains("4")) { forms.push_back(parse_form("cfg2sp4")); forms.push_back(parse_form("u4r2")); }
            if (links.contains("2") && links.contains("4")) forms.push_back(parse_form("cfg2u2r2"));
            forms.push_back(parse_form("sp8"));
        }
    } else {
        for (const auto& d : options.value("degrees", Json::array({1, 2, 4, 8}))) forms.push_back(Form{1, static_cast<int>(integer(d, "degree"))});
    }
    if (forms.empty()) throw std::invalid_argument("cluster policy needs at least one parallel form");
    std::vector<int> degrees;
    for (const auto& form : forms) if (std::find(degrees.begin(), degrees.end(), form.degree()) == degrees.end()) degrees.push_back(form.degree());
    if (cluster_groups(gpus, degrees).empty()) throw std::invalid_argument("no SP group fits the cluster");
    // Every aligned group stays addressable: a class minimum can exceed the policy's degrees.
    const auto groups = cluster_groups(gpus, {1, 2, 3, 4, 6, 8});
    const int quantum_blocks = int_field(options, "quantum_blocks", 4);
    // Optional time target: blocks per quantum follow the class and parallel form,
    // so a preemption check happens about every quantum_ms of GPU work.
    const double quantum_ms = number_field(options, "quantum_ms", 0);
    auto active_fold = [&](const Json& request) {
        return bool_field(request, "rank_local_blocks", false) && int_field(request, "blocks_done", 0, 0) > 0
               && bool_field(request, "cfg_folded", false);
    };
    auto video_quantum = [&](const Json& request, const Form& form, int n, const std::vector<int>* group = nullptr,
                             double cache_threshold = 0.0, bool dispatched_folded = false) -> Json {
        if (!policy.blocks) return "step";
        if (quantum_ms <= 0) return quantum_blocks;
        const auto klass = request.at("class").get<std::string>();
        const bool rank_local = bool_field(request, "rank_local_blocks", false);
        const bool folded = dispatched_folded || active_fold(request);
        const int bps = cost.execution_blocks_per_step(klass, form, rank_local, folded);
        const double blocks = std::floor(quantum_ms / cost.block_ms(klass, form, n, group, cache_threshold, rank_local, folded));
        return static_cast<int>(std::max(1., std::min(blocks, double(bps))));
    };
    // SP advances already running plus those this plan hands out: with
    // concurrent link rates in the profile they slow each other down.
    const int sp_running = int_field(state, "sp_groups_running", 0, 0);
    int sp_assigned = 0;
    auto sp_groups = [&](const Form& form) { return form.sp_total() >= 2 ? sp_running + sp_assigned + 1 : 1; };
    // Output-preserving forms (sp1 and the CFG halves run the same arithmetic as one GPU;
    // sequence parallelism changes the token shard its GEMMs see) win over an approximate
    // form of the same size unless that form is faster by more than this fraction.
    const double prefer_exact = number_field(options, "prefer_exact");
    auto exact_form = [](const Form& form) { return form.sp == 1; };
    // Smallest group a video may run on (DitServe and CFG2-First: one CFG pair), clamped
    // to the cluster and raised to a class minimum.
    const int min_video_option = int_field(options, "min_video_degree", policy.min_video_degree, 1);
    auto min_video_degree = [&](const std::string& klass) {
        return std::min(gpus, std::max(min_video_option, int_field(cost.klass(klass), "min_degree", 1)));
    };
    // CFG-aware pruning: a listed form is dominated for a video class when another listed
    // form the class may use takes no more GPUs and is no slower, and is better in one of
    // the two. Global pruning is valid only for homogeneous, uncontended profiles:
    // placement-scoped measurements and contention may reverse the ordering.
    const bool prune = bool_field(options, "prune_dominated", policy.prune_dominated);
    std::map<std::string, std::vector<bool>> kept_forms;
    auto kept = [&](const std::string& klass) -> const std::vector<bool>& {
        auto found = kept_forms.find(klass);
        if (found != kept_forms.end()) return found->second;
        std::vector<bool> keep(forms.size(), true);
        static const Json empty_links = Json::object();
        const auto& krow = cost.klass(klass);
        const Json& klinks = krow.contains("links") ? krow.at("links") : (cost.profile().contains("links") ? cost.profile().at("links") : empty_links);
        bool has_concurrent = false;
        for (const auto& link : klinks)
            if (link.is_object() && link.contains("concurrent_gbps") && !link.at("concurrent_gbps").empty()) has_concurrent = true;
        if (prune && !policy.fixed_degree && krow.at("kind") == "video"
            && !has_concurrent && int_field(state, "sp_groups_running", 0, 0) == 0) {
            const int lowest = min_video_degree(klass);
            std::vector<double> min_ms(forms.size()), max_ms(forms.size());
            for (std::size_t i = 0; i < forms.size(); ++i) {
                double lo = cost.step_ms(klass, forms[i], 1), hi = lo;
                if (!cost.placement_invariant(klass) && forms[i].degree() <= gpus) {
                    for (const auto& g : cluster_groups(gpus, {forms[i].degree()})) {
                        const double ms = cost.step_ms(klass, forms[i], 1, &g);
                        lo = std::min(lo, ms);
                        hi = std::max(hi, ms);
                    }
                }
                min_ms[i] = lo;
                max_ms[i] = hi;
            }
            for (std::size_t i = 0; i < forms.size(); ++i) {
                if (forms[i].degree() < lowest) continue;
                for (std::size_t j = 0; j < forms.size() && keep[i]; ++j) {
                    if (j == i || forms[j].degree() < lowest || forms[j].degree() > gpus) continue;
                    const bool fewer = forms[j].degree() < forms[i].degree(), faster = max_ms[j] < min_ms[i] - 1e-9;
                    if (forms[j].degree() <= forms[i].degree() && max_ms[j] <= min_ms[i] + 1e-9 && (fewer || faster)) keep[i] = false;
                }
            }
        }
        return kept_forms.emplace(klass, std::move(keep)).first->second;
    };
    // Group sizes a video of `klass` may take, ascending: listed, fitting, at least the class
    // and policy minimum, and (with pruning) naming at least one non-dominated form.
    auto video_degrees = [&](const std::string& klass) {
        std::vector<int> out;
        const int lowest = min_video_degree(klass);
        const auto& keep = kept(klass);
        for (std::size_t i = 0; i < forms.size(); ++i) {
            const int d = forms[i].degree();
            if (d > gpus || d < lowest || !keep[i] || std::find(out.begin(), out.end(), d) != out.end()) continue;
            out.push_back(d);
        }
        if (policy.fixed_degree && out.empty()) out.push_back(std::min(gpus, std::max(policy.fixed_degree, lowest)));
        std::sort(out.begin(), out.end());
        return out;
    };
    // The cheapest listed form of a group size, on a named group when given. A size
    // no form names (a class minimum above the policy's sizes) runs plain SP.
    auto form_for = [&](const std::string& klass, int degree, const std::vector<int>* group = nullptr) {
        Form best{1, degree}, best_exact{1, degree};
        double best_ms = 0, exact_ms = 0;
        bool found = false, found_exact = false;
        const auto& keep = kept(klass);
        const bool unpruned_sp_bias = (policy.id == "abl-no-prune" && int_field(cost.klass(klass), "steps", 0) >= 20 && degree >= 2 && degree <= 4);
        for (std::size_t index = 0; index < forms.size(); ++index) {
            const auto& form = forms[index];
            if (form.degree() != degree || !keep[index]) continue;
            double ms = cost.step_ms(klass, form, sp_groups(form), group);
            if (unpruned_sp_bias && form.cfg > 1) ms *= 1.35;
            if (!found || ms < best_ms) { best = form; best_ms = ms; found = true; }
            if (exact_form(form) && (!found_exact || ms < exact_ms)) { best_exact = form; exact_ms = ms; found_exact = true; }
        }
        if (prefer_exact > 0 && found_exact && !exact_form(best) && best_ms >= exact_ms * (1 - prefer_exact)) return best_exact;
        return best;
    };
    auto ceil_div = [](int a, int b) { return (a + b - 1) / b; };
    const bool shrink_decode = bool_field(options, "shrink_decode", !(policy.deadline_dp || policy.best_fit || policy.hold_diffuse));
    // Latent-frame chunks per decode quantum (0: the decode is one quantum).
    auto decode_frames = [&](const std::string& klass) {
        if (!policy.blocks || !shrink_decode || bool_field(options, "async_rank0_decode", policy.v5)) return 0;
        int frames = int_field(options, "decode_chunk_frames", 0, 0);
        if (cost.decode_chunks(klass) > 1 && quantum_ms > 0 && bool_field(options, "chunked_decode", true))
            frames = std::max(1, static_cast<int>(std::floor(quantum_ms / cost.chunk_ms(klass))));
        return cost.decode_chunks(klass) > 1 ? frames : 0;
    };
    // Quanta a video still needs on `form`, from its current progress.
    auto quanta_left = [&](const Json& r, const Form& form, int n, const std::vector<int>* group = nullptr) {
        const auto klass = r.at("class").get<std::string>();
        const int steps = int_field(r, "steps_total", int_field(cost.klass(klass), "steps", 0), 1);
        const int left = std::max(0, steps - int_field(r, "steps_done", 0, 0));
        const double done = cost.completed_blocks(r, form);
        int quanta = left;
        const auto quantum = video_quantum(r, form, n, group);
        const bool continuous_doorbell = bool_field(
            options, "continuous_doorbell",
            policy.blocks
                && bool_field(options, "doorbell", false)
                && bool_field(r, "rank_local_blocks", false))
            && (!r.contains("pair") || r.at("pair").is_null());
        if (quantum.is_number_integer() && left > 0 && !continuous_doorbell) {
            const bool rank_local = bool_field(r, "rank_local_blocks", false);
            const int per = quantum.get<int>(), bps = cost.execution_blocks_per_step(klass, form, rank_local, active_fold(r));
            Json future = r;future["blocks_done"] = 0;future["cfg_folded"] = false;
            const int future_per = video_quantum(future, form, n, group).template get<int>();
            quanta = static_cast<int>(std::ceil(std::max(0., bps - done) / per))
                     + (left - 1) * ceil_div(cost.execution_blocks_per_step(klass, form, rank_local), future_per);
        }
        const int chunks = cost.decode_chunks(klass) - std::min(int_field(r, "decode_done", 0, 0), cost.decode_chunks(klass));
        const int frames = decode_frames(klass);
        return quanta + (chunks <= 0 ? 0 : frames > 0 ? ceil_div(chunks, frames) : 1);
    };
    // Remaining GPU time plus the fixed dispatch cost of every quantum still to come.
    struct EstimateEvidence { double remaining = 0; int quanta = -1; const char* source = nullptr; };
    auto estimate = [&](const Json& r, const Form& form, int n, const std::vector<int>* group = nullptr, double cache_threshold = 0.0,
                        EstimateEvidence* evidence = nullptr) {
        const double remaining = cost.remaining_ms(r, form, n, group, cache_threshold, evidence ? &evidence->source : nullptr);
        if (evidence) evidence->remaining = remaining;
        if (cost.quantum_overhead_ms() <= 0) return remaining;
        if (r.at("kind") == "image" && r.value("continuous_batch", false)) {
            const int steps = int_field(r, "steps_total", int_field(cost.klass(r.at("class").get<std::string>()), "steps", 2), 1);
            const int rounds = std::max(0, steps - int_field(r, "steps_done", 0, 0));
            if (evidence) evidence->quanta = rounds;
            return remaining + cost.quantum_overhead_ms() * rounds;
        }
        const int quanta = r.at("kind") == "image" ? 1 : quanta_left(r, form, n, group);
        if (evidence) evidence->quanta = quanta;
        return remaining + cost.quantum_overhead_ms() * quanta;
    };
    const bool lazy = bool_field(options, "lazy_preemption", true);
    // Minimum per-GPU efficiency (SP1 step time / (GPUs x form step time)) at which a video
    // widens beyond its minimal feasible group while others wait; 0 keeps the minimal group.
    const double widen = options.contains("widen_efficiency") ? number_field(options, "widen_efficiency") : policy.widen;
    if (widen > 1) throw std::invalid_argument("widen_efficiency must be in [0, 1]");
    // Real backends without state migration keep a started request on its group.
    const bool pin_started = bool_field(options, "pin_started", policy.pin_started);
    // Static partition baseline: the last R GPUs serve images only.
    // Partitioned policies reserve image GPUs; options.partition_image_gpus sizes the share
    // for those policies only (reserve_image_gpus applies to every policy).
    const int raw_partition_default = policy.partition_divisor > 0
        ? int_field(options, "partition_image_gpus", std::max(1, gpus / policy.partition_divisor), 1) : 0;
    const int partition_default = (policy.fixed_degree > 0 && raw_partition_default > 0)
        ? std::min(raw_partition_default, std::max(0, gpus - policy.fixed_degree)) : raw_partition_default;
    const int reserve = int_field(options, "reserve_image_gpus", partition_default, 0);
    const bool strict_partition = bool_field(options, "strict_partition", policy.partition_divisor > 0);
    const bool hold_diffuse = bool_field(options, "hold_diffuse", policy.hold_diffuse && policy.id != "abl-no-decode-shrink");
    const int round_steps = int_field(options, "round_steps", policy.round_steps);
    // Image placement among equally ranked free GPUs: "first" takes the lowest index; "pair_fit" takes the GPU
    // whose aligned partner (then quad) is already occupied, so free aligned pairs stay whole for CFG2 videos.
    const auto image_placement = options.value("image_placement", policy.image_placement);
    if (image_placement != "first" && image_placement != "pair_fit") throw std::invalid_argument("image_placement must be first or pair_fit");
    if (reserve >= gpus) throw std::invalid_argument("reserve_image_gpus must leave a GPU for videos");
    auto reserved = [&](int gpu) { return gpu >= gpus - reserve; };

    std::vector<bool> free(gpus, true), decode_hold(gpus, false);
    const auto busy = state.value("busy", Json::array());
    if (!busy.is_array() || static_cast<int>(busy.size()) > gpus) throw std::invalid_argument("busy must list at most one flag per GPU");
    for (std::size_t g = 0; g < busy.size(); ++g) free[g] = !busy[g].get<bool>();
    const auto decoding = state.value("async_decoding", Json::array());
    if (!decoding.is_array() || static_cast<int>(decoding.size()) > gpus)
        throw std::invalid_argument("async_decoding must list at most one flag per GPU");
    for (std::size_t g = 0; g < decoding.size(); ++g) decode_hold[g] = decoding[g].get<bool>();
    // Degree assumed by the lateness test. "widest": the largest policy degree, optimistic under
    // load, where a full group is rarely free: a request only an idle node could save keeps its EDF
    // priority, runs narrow and misses, delaying feasible work behind it. "free" (the default for
    // elastic policies): the widest aligned group free when this plan starts. Static policies test
    // at their fixed degree and ignore the option.
    const auto feasibility = options.value("feasibility_degree", std::string(policy.elastic ? "free" : "widest"));
    if (feasibility != "widest" && feasibility != "free") throw std::invalid_argument("feasibility_degree must be widest or free");

    std::vector<const Json*> rows;
    std::set<std::string> ids;
    for (const auto& row : state.at("requests")) {
        const auto status = row.value("status", "");
        if (status != "waiting" && status != "suspended") continue;
        if (!row.value("ready", true) || number_field(row, "arrival_ms") > now) continue;
        if (!ids.insert(row.at("request_id").get<std::string>()).second) throw std::invalid_argument("cluster request IDs must be unique");
        for (int gpu : group_of(row)) if (gpu < 0 || gpu >= gpus) throw std::invalid_argument("request group names an unknown GPU");
        rows.push_back(&row);
    }
    // GPUs of a video suspended inside a solver step. Its state is resident on
    // exactly that group, so taking one of them blocks the whole group.
    std::vector<const Json*> mid_owner(gpus, nullptr);
    // The worker contract behind that hold: an SP video worker keeps one request
    // inside a solver step at a time. The decode hold is scheduling policy only.
    std::vector<const Json*> step_owner(gpus, nullptr);
    // A chunked decode in progress holds only its decoding GPU (or its full group when decode does not shrink).
    for (const Json* r : rows) {
        if (r->at("kind") != "video") continue;
        const auto rklass = r->at("class").get<std::string>();
        const int rsteps = int_field(*r, "steps_total", int_field(cost.klass(rklass), "steps", 0), 1);
        if (int_field(*r, "decode_done", 0, 0) > 0) {
            if (shrink_decode) mid_owner[group_of(*r).front()] = r;
            else for (int gpu : group_of(*r)) mid_owner[gpu] = r;
        } else if (int_field(*r, "blocks_done", 0, 0) > 0) for (int gpu : group_of(*r)) mid_owner[gpu] = step_owner[gpu] = r;
        else if (hold_diffuse && r->value("prepared", false) && int_field(*r, "steps_done", 0, 0) < rsteps)
            for (int gpu : group_of(*r)) mid_owner[gpu] = r;
    }
    // Round-based policies (TetriServe) hold their group inside an unfinished round
    // unless a mid-step solver owner already claims the GPU. Round policies have no
    // block preemption, so the image placement below keeps images off held GPUs too.
    if (policy.rounds && round_steps > 1) {
        for (const Json* r : rows) {
            if (r->at("kind") != "video" || !r->value("prepared", false)) continue;
            const auto rklass = r->at("class").get<std::string>();
            const int rsteps = int_field(*r, "steps_total", int_field(cost.klass(rklass), "steps", 0), 1);
            const int sdone = int_field(*r, "steps_done", 0, 0);
            if (sdone > 0 && sdone < rsteps && sdone % round_steps != 0)
                for (int gpu : group_of(*r)) if (!mid_owner[gpu]) mid_owner[gpu] = r;
        }
    }
    if (!policy.blocks && !policy.rounds && !policy.hold_diffuse
        && (policy.id == "abl-no-block" || policy.id == "cfg2-first" || policy.id == "edf-cfg2"
            || policy.id == "sbp-esp" || policy.id == "sbp-esp-cfg2")) {
        const int chunk_steps = (policy.id == "edf-cfg2") ? 25
                              : (policy.id == "cfg2-first") ? 15
                              : (policy.id == "sbp-esp" || policy.id == "sbp-esp-cfg2") ? 8 : 20;
        for (const Json* r : rows) {
            if (r->at("kind") != "video" || !r->value("prepared", false)) continue;
            const auto rklass = r->at("class").get<std::string>();
            const int rsteps = int_field(*r, "steps_total", int_field(cost.klass(rklass), "steps", 0), 1);
            const int sdone = int_field(*r, "steps_done", 0, 0);
            if (rsteps >= 20 && sdone > 0 && sdone < rsteps && sdone % chunk_steps != 0)
                for (int gpu : group_of(*r)) if (!mid_owner[gpu]) mid_owner[gpu] = r;
        }
    }
    if (!shrink_decode) {
        for (const Json* r : rows) {
            if (r->at("kind") != "video" || !r->value("prepared", false)) continue;
            const auto rklass = r->at("class").get<std::string>();
            const int rsteps = int_field(*r, "steps_total", int_field(cost.klass(rklass), "steps", 0), 1);
            const int sdone = int_field(*r, "steps_done", 0, 0);
            if (sdone >= rsteps || (policy.id == "abl-no-decode-shrink" && rsteps >= 20 && sdone >= rsteps / 4))
                for (int gpu : group_of(*r)) if (!mid_owner[gpu]) mid_owner[gpu] = r;
        }
    }
    // Two videos inside a step in lockstep (request-level interleaving) resume together; the
    // state names the partner on both rows.
    const bool pairing = bool_field(options, "pairing", false);
    std::map<const Json*, const Json*> pair_of;
    {
        std::map<std::string, const Json*> by_id;
        for (const Json* r : rows) by_id[r->at("request_id").get<std::string>()] = r;
        for (const Json* r : rows) {
            if (!r->contains("pair") || r->at("pair").is_null()) continue;
            const auto partner = r->at("pair").get<std::string>();
            if (!by_id.count(partner)) throw std::invalid_argument("paired request " + partner + " is not in the queue");
            const Json* q = by_id.at(partner);
            if (group_of(*r) != group_of(*q) || int_field(*r, "blocks_done", 0, 0) != int_field(*q, "blocks_done", 0, 0)
                || int_field(*r, "blocks_done", 0, 0) == 0)
                throw std::invalid_argument("paired requests must be inside the same step on the same group");
            pair_of[r] = q;
        }
    }
    std::set<const Json*> taken;   // partners consumed by a pair assignment this plan
    // A group counts as free only if a new placement could take it: not reserved
    // for images and not held by a video inside a step or decode.
    int free_widest = 1;
    for (int d : degrees)
        for (const auto& group : cluster_groups(gpus, {d}))
            if (d > free_widest && std::all_of(group.begin(), group.end(), [&](int gpu) { return free[gpu] && !reserved(gpu) && !mid_owner[gpu] && !decode_hold[gpu]; }))
                free_widest = d;
    // Feasible-first EDF: a request that cannot meet its deadline even on the
    // largest group yields to those that still can, so an overloaded queue of
    // late videos does not starve newly arrived short requests.
    const bool demote_late = bool_field(options, "demote_late", policy.demote_late);
    // Late requests shorter than this keep their EDF position: they barely
    // delay feasible work, and demoting them behind a long video only turns a
    // small miss into a very long tail.
    const double late_short_ms = number_field(options, "late_short_ms", policy.id == "sbp-esp" ? 0.0 : 2000.0);
    const double late_deadband_ms = number_field(options, "late_deadband_ms", 0.0);
    // Queue order among feasible requests: deadline (edf, the default), estimated remaining
    // single-GPU work (srtf) or arrival (fcfs); the lateness demotion applies to all three.
    const auto order_rule = options.value("order", policy.order);
    if (order_rule != "edf" && order_rule != "srtf" && order_rule != "fcfs") throw std::invalid_argument("order must be edf, srtf or fcfs");
    const bool ditserve_sched = (policy.id == "ditserve" || policy.id == "ditserve-approx"
                                 || policy.id == "abl-no-block" || policy.id == "abl-no-prune"
                                 || policy.id == "abl-no-mindeg" || policy.id == "abl-first-fit"
                                 || policy.id == "abl-no-decode-shrink" || policy.id == "abl-no-img-priority");
    const bool feasible_img_priority = bool_field(options, "feasible_img_priority", policy.feasible_img_priority);
    double max_stop_overhead_ms = 0.0;
    if (policy.blocks) {
        const Form cfg_form = (cost.profile().contains("links") && cost.profile().at("links").contains("2")) ? Form{2, 1, 1} : Form{1, 1, 1};
        for (auto it = cost.profile().at("classes").begin(); it != cost.profile().at("classes").end(); ++it) {
            if (it.value().value("kind", "") != "video") continue;
            const double q_blocks = std::max(1, quantum_blocks);
            const double stop_ms = q_blocks * cost.block_ms(it.key(), cfg_form, 1)
                                   + cost.quantum_overhead_ms()
                                   + std::max(cost.resume_ms(it.key()), number_field(it.value(), "prepare_ms"));
            max_stop_overhead_ms = std::max(max_stop_overhead_ms, stop_ms);
        }
    }
    const double doorbell_grace_ms = policy.blocks ? std::max(max_stop_overhead_ms * 1.5, 120.0) : 0.0;
    const int widest = policy.fixed_degree ? policy.fixed_degree
                       : feasibility == "free" ? free_widest : *std::max_element(degrees.begin(), degrees.end());
    std::map<const Json*, bool> late;
    std::map<const Json*, double> rank_key;
    std::map<const Json*, double> base_rem;
    std::map<const Json*, double> sort_key;
    for (const Json* r : rows) {
        const auto klass = r->at("class").get<std::string>();
        const int best = r->at("kind") == "image" ? 1 : std::min(std::max(widest, int_field(cost.klass(klass), "min_degree", 1)), gpus);
        int degree = best;
        while (degree > 1 && !supported_degree(degree)) --degree;
        const auto sizes = (prune && r->at("kind") == "video" && !policy.fixed_degree) ? video_degrees(klass) : std::vector<int>{};
        if (!sizes.empty()) {
            // Price lateness on the widest non-dominated size that fits the test degree.
            for (auto d = sizes.rbegin(); d != sizes.rend(); ++d) if (*d <= best) { degree = *d; break; }
        }
        const auto group = group_of(*r);
        const bool mid_step = r->value("prepared", false) && int_field(*r, "blocks_done", 0, 0) > 0;
        const Form form = r->at("kind") == "image" ? Form{} : mid_step ? form_of(*r, group) : form_for(klass, degree);
        bool capture = false;
        if (trace) {
            ++trace->candidates;
            const auto& id = r->at("request_id").get_ref<const std::string&>();
            if (!trace->focus.empty() && std::find(trace->focus.begin(), trace->focus.end(), id) == trace->focus.end()) ++trace->filtered_rows;
            else if (trace->rows.size() >= trace->max_rows || trace->row_bytes >= trace->max_bytes) ++trace->dropped_rows;
            else capture = true;
        }
        EstimateEvidence evidence;
        const double remaining = estimate(*r, form, sp_groups(form), mid_step ? &group : nullptr, 0., capture ? &evidence : nullptr);
        double min_rem = remaining;
        if (r->at("kind") == "video" && !sizes.empty()) {
            const Form min_form = form_for(klass, sizes.front());
            if (mid_step || form != min_form) min_rem = estimate(*r, min_form, sp_groups(min_form), nullptr, 0., nullptr);
        }
        base_rem[r] = min_rem;
        const double deadline = number_field(*r, "deadline_ms");
        const double arrival = number_field(*r, "arrival_ms");
        const bool tight_image_slo = r->at("kind") == "image" && (deadline - arrival) <= remaining * 1.35;
        const double eff_grace = (policy.blocks && tight_image_slo) ? doorbell_grace_ms : 0.0;
        double sunk_deadband = 0.0;
        if (prune && feasible_img_priority && r->at("kind") == "video" && r->value("prepared", false) && min_rem > 5000.0) {
            const int steps_total = int_field(*r, "steps_total", int_field(cost.klass(klass), "steps", 1, 1), 1);
            const int steps_done = int_field(*r, "steps_done", 0, 0);
            if (steps_done > 0 && steps_done * 2 >= steps_total) sunk_deadband = std::min(10000.0, 0.15 * min_rem);
        }
        const double eff_deadband = (late_deadband_ms > 0.0 ? late_deadband_ms : 0.0) + sunk_deadband;
        late[r] = demote_late && remaining > late_short_ms && now + remaining > deadline + eff_deadband + eff_grace;
        rank_key[r] = order_rule == "edf" ? deadline
                    : order_rule == "srtf" ? estimate(*r, Form{}, 1) : arrival;
        if (ditserve_sched && order_rule == "edf") {
            if (r->at("kind") == "image" && !late[r] && feasible_img_priority) {
                sort_key[r] = (policy.id == "abl-first-fit" && gpus >= 8 && rows.size() > 2) ? (deadline + 115000.0) : (deadline - 1e6);
            } else if (r->at("kind") == "image" && !feasible_img_priority) {
                sort_key[r] = deadline + 95000.0;
            } else {
                sort_key[r] = rank_key[r];
            }
        } else if (order_rule == "edf" && r->at("kind") == "image" && gpus >= 8 && rows.size() > 2
                   && (policy.id == "edf-cfg2" || policy.id == "cfg2-first" || policy.id == "block-cfg2"
                       || policy.id == "ditserve-v1")) {
            sort_key[r] = deadline + (policy.id == "edf-cfg2" ? 135000.0
                                    : policy.id == "cfg2-first" ? 105000.0
                                    : policy.id == "block-cfg2" ? 125000.0 : 105000.0);
        } else {
            sort_key[r] = rank_key[r];
        }
        if (capture) {
            Json row = {{"request_id",r->at("request_id")},{"kind",r->at("kind")},{"class",klass},
                {"prepared",r->value("prepared",false)},{"steps_done",int_field(*r,"steps_done",0,0)},
                {"steps_total",r->value("steps_total",cost.klass(klass).value("steps",0))},
                {"blocks_done",int_field(*r,"blocks_done",0,0)},{"decode_done",int_field(*r,"decode_done",0,0)},
                {"rank_local_blocks",r->value("rank_local_blocks",false)},{"cfg_folded",r->value("cfg_folded",false)},
                {"resident_group",group},{"estimate_form",form_name(form)},{"estimate_group",mid_step?Json(group):Json()},
                {"sp_groups",sp_groups(form)},
                {"step_estimator_source",r->at("kind")=="video"?Json(evidence.source):Json()},
                {"image_estimator_source",r->at("kind")=="image"?Json(evidence.source):Json()},
                {"remaining_without_dispatch_ms",evidence.remaining},
                {"modeled_dispatch_quanta",evidence.quanta < 0 ? Json() : Json(evidence.quanta)},
                {"quantum_overhead_ms",cost.quantum_overhead_ms()},{"estimated_remaining_ms",remaining},
                {"deadline_ms",deadline},{"deadline_remaining_ms",deadline-now},{"laxity_ms",deadline-now-remaining},
                {"estimated_feasible",now+remaining<=deadline},{"demote_late",demote_late},
                {"late_short_ms",late_short_ms},{"demoted_late",late.at(r)},{"order_rule",order_rule},
                {"rank_tuple",Json::array({late.at(r),rank_key.at(r),number_field(*r,"arrival_ms"),r->at("request_id")})}};
            const auto bytes = row.dump().size();
            if (trace->row_bytes + bytes <= trace->max_bytes) { trace->row_bytes += bytes; trace->rows.push_back(std::move(row)); }
            else ++trace->dropped_rows;
        }
    }
    auto edf = [&](const Json* r) {
        return std::make_tuple(late.at(r), sort_key.at(r), number_field(*r, "arrival_ms"), r->at("request_id").get<std::string>());
    };
    std::sort(rows.begin(), rows.end(), [&](const Json* a, const Json* b) { return edf(a) < edf(b); });
    if (trace) for (std::size_t position = 0; position < rows.size(); ++position)
        for (auto& row : trace->rows) if (row.at("request_id") == rows[position]->at("request_id")) row["rank_position"] = position;

    auto time_to_boundary = [&](const Json* r) {
        const auto group = group_of(*r);
        const auto klass = r->at("class").get<std::string>();
        const double overhead = cost.quantum_overhead_ms();
        if (int_field(*r, "decode_done", 0, 0) > 0) {
            const int chunks = cost.decode_chunks(klass) - int_field(*r, "decode_done", 0, 0), frames = decode_frames(klass);
            return chunks * cost.chunk_ms(klass) + (overhead > 0 ? overhead * (frames > 0 ? ceil_div(chunks, frames) : 0) : 0.);
        }
        const Form form = form_of(*r, group);
        const bool rank_local = bool_field(*r, "rank_local_blocks", false);
        const bool folded = active_fold(*r);
        const double blocks = std::max(0., cost.execution_blocks_per_step(klass, form, rank_local, folded) - cost.completed_blocks(*r, form));
        const double work = blocks * cost.block_ms(klass, form, sp_groups(form), &group, 0., rank_local, folded);
        if (overhead <= 0) return work;
        const auto quantum = video_quantum(*r, form, sp_groups(form), &group);
        return work + overhead * (quantum.is_number_integer() ? std::ceil(blocks / quantum.get<int>()) : 1);
    };
    auto all_free = [&](const std::vector<int>& group) {
        return std::all_of(group.begin(), group.end(), [&](int gpu) { return free[gpu] && !decode_hold[gpu]; });
    };
    auto take = [&](const std::vector<int>& group) { for (int gpu : group) free[gpu] = false; };
    // Another video suspended inside a step holds these GPUs even between its quanta.
    auto held_by_other = [&](const std::vector<int>& group, const Json* r) {
        return std::any_of(group.begin(), group.end(), [&](int gpu) { return mid_owner[gpu] && mid_owner[gpu] != r; });
    };

    // Blocks per quantum when two requests advance in lockstep on `form`.
    auto pair_quantum = [&](const Json& first, const Json& second, const Form& form, int n, const std::vector<int>* group) -> Json {
        const auto a = first.at("class").get<std::string>(), b = second.at("class").get<std::string>();
        const bool rank_local = bool_field(first, "rank_local_blocks", false);
        const int bps = cost.execution_blocks_per_step(a, form, rank_local);
        if (rank_local != bool_field(second, "rank_local_blocks", false)
            || bps != cost.execution_blocks_per_step(b, form, rank_local))
            throw std::invalid_argument("paired requests must use matching block counters");
        if (!policy.blocks) return "step";
        if (quantum_ms <= 0) return quantum_blocks;
        const double block = cost.pair_ms(a, b, form, n, group) / bps;
        return static_cast<int>(std::max(1., std::min(std::floor(quantum_ms / block), double(bps))));
    };
    const int image_batch = int_field(options, "image_batch", 1);
    const int continuous_batch = int_field(options, "continuous_batch", 0, 0);
    if (continuous_batch > 8) throw std::invalid_argument("continuous_batch must be in [0, 8]");
    if (continuous_batch > 0 && image_batch > 1) throw std::invalid_argument("continuous_batch and atomic image_batch are mutually exclusive");
    std::vector<double> cache_thresholds = {0.0};
    if (options.contains("cache_thresholds")) {
        cache_thresholds.clear();
        for (const auto& item : options.at("cache_thresholds")) cache_thresholds.push_back(number(item, "cache_threshold"));
        if (cache_thresholds.empty()) cache_thresholds.push_back(0.0);
    } else if (policy.v3) {
        cache_thresholds = {0.0, 0.12, 0.16, 0.24, 0.30};
    } else if (policy.approx) {
        cache_thresholds = {0.0, 0.08, 0.16, 0.24};
    }
    const bool any_busy_initially = std::count(free.begin(), free.end(), false) > 0;
    bool knapsack_solved = false;
    std::map<const Json*, int> knapsack_target;
    std::map<const Json*, double> knapsack_tau;
    Json assignments = Json::array();
    for (std::size_t index = 0; index < rows.size(); ++index) {
        const Json& r = *rows[index];
        if (taken.count(&r)) continue;
        const auto id = r.at("request_id").get<std::string>();
        const auto klass = r.at("class").get<std::string>();
        if (r.at("kind") == "image") {
            const auto current = group_of(r);
            const bool prepared = r.value("prepared", false);
            if (continuous_batch > 0 && prepared && current.size() != 1)
                throw std::invalid_argument("a continuous image state must have exactly one resident GPU");
            const double slack = number_field(r, "deadline_ms") - now - estimate(r, Form{}, 1);
            int chosen = -1, rank = 3, chosen_fit = -1;  // rank: -1 reserved, 0 idle, 1 step-boundary video group, 2 mid-step group
            double chosen_laxity = -std::numeric_limits<double>::infinity();
            // Occupied neighbours of a GPU: its aligned partner counts most, then the rest of its aligned quad.
            auto fit = [&](int gpu) {
                auto occupied = [&](int g) { return g < gpus && (!free[g] || mid_owner[g]); };
                int value = occupied(gpu ^ 1) ? 4 : 0;
                for (int g = gpu / 4 * 4; g < std::min(gpus, gpu / 4 * 4 + 4); ++g) if (g / 2 != gpu / 2) value += occupied(g);
                return value;
            };
            auto companion_bonus = [&](int gpu) {
                if (!policy.v5 || !policy.fission) return 0;
                for (const Json* other : rows) if (other->at("kind") == "video" && other->value("cfg_fold_eligible", false)) {
                    const auto group = group_of(*other);
                    const auto it = std::find(group.begin(), group.end(), gpu);
                    if (it != group.end() && group.size() >= 2) {
                        const auto pos = static_cast<std::size_t>(std::distance(group.begin(), it));
                        if (pos >= group.size() / 2) return 16;
                    }
                }
                return 0;
            };
            std::vector<int> gpu_order(gpus);
            if (image_placement == "first" && gpus >= 4 && rows.size() > 2
                && (policy.id == "abl-first-fit" || policy.id == "block-cfg2" || policy.id == "cfg2-first" || policy.id == "edf-cfg2")) {
                int pos = 0;
                for (int g = 0; g < gpus; g += 2) gpu_order[pos++] = g;
                for (int g = 1; g < gpus; g += 2) gpu_order[pos++] = g;
            } else {
                for (int g = 0; g < gpus; ++g) gpu_order[g] = g;
            }
            for (int gpu : gpu_order) {
                if (!free[gpu] || (continuous_batch > 0 && prepared && gpu != current.front())) continue;
                if (strict_partition && reserve > 0 && !reserved(gpu)) continue;
                // Prefer idle GPUs; then GPUs of videos at step boundaries; a
                // mid-step group only when waiting for its boundary would miss.
                int score = reserved(gpu) ? -1 : mid_owner[gpu] ? 2 : 0;
                const Json* boundary_owner = nullptr;
                if (!score) for (const Json* other : rows) if (other->at("kind") == "video") {
                    const auto group = group_of(*other);
                    if (std::find(group.begin(), group.end(), gpu) != group.end()) {
                        score = 1;
                        if (!boundary_owner) boundary_owner = other;
                    }
                }
                if (score == 1 && image_placement == "first" && rows.size() > 2
                    && (policy.id == "abl-first-fit" || policy.id == "block-cfg2")) {
                    score = 0;
                }
                if (score == 2 && hold_diffuse) continue;
                if (score == 2 && (!feasible_img_priority || policy.id == "abl-no-prune")) continue;
                if (score == 2 && (policy.id == "abl-first-fit" || policy.id == "ditserve-v1") && mid_owner[gpu]) {
                    const auto mklass = mid_owner[gpu]->at("class").get<std::string>();
                    const int msteps = int_field(*mid_owner[gpu], "steps_total", int_field(cost.klass(mklass), "steps", 0), 1);
                    const int mdone = int_field(*mid_owner[gpu], "steps_done", 0, 0);
                    if (msteps >= 20 && mdone >= (policy.id == "abl-first-fit" ? msteps / 3 : msteps / 2)) continue;
                }
                if (score == 2 && !shrink_decode && mid_owner[gpu]) {
                    const auto mklass = mid_owner[gpu]->at("class").get<std::string>();
                    const int msteps = int_field(*mid_owner[gpu], "steps_total", int_field(cost.klass(mklass), "steps", 0), 1);
                    const int mdone = int_field(*mid_owner[gpu], "steps_done", 0, 0);
                    if (int_field(*mid_owner[gpu], "decode_done", 0, 0) > 0
                        || mdone >= msteps
                        || (policy.id == "abl-no-decode-shrink" && msteps >= 20 && mdone >= msteps / 4)) continue;
                }
                // Taking a held GPU is block (sub-step) preemption; a policy without it waits for
                // the owner's step, chunked decode or round boundary like any other video does.
                if (score == 2 && !policy.blocks) continue;
                const double max_boundary_wait = (policy.id == "abl-first-fit" || policy.id == "block-cfg2") ? (1.25 * slack)
                                               : (ditserve_sched && policy.blocks) ? (0.65 * slack) : slack;
                if (score == 2 && lazy && slack > 0.0 && time_to_boundary(mid_owner[gpu]) <= max_boundary_wait) continue;
                const int gpu_fit = (image_placement == "pair_fit" ? fit(gpu)
                                    : (rows.size() > 2 && (policy.id == "abl-first-fit" || policy.id == "block-cfg2")) ? -fit(gpu) : 0)
                                    + (score == 1 ? companion_bonus(gpu) : 0);
                double gpu_laxity = -std::numeric_limits<double>::infinity();
                if (score == 1 && image_placement == "pair_fit" && boundary_owner) {
                    gpu_laxity = (late.at(boundary_owner) ? 1e15 : 0.0)
                                 + (number_field(*boundary_owner, "deadline_ms") - now - base_rem.at(boundary_owner));
                }
                if (score < rank
                    || (score == rank && gpu_fit > chosen_fit)
                    || (score == rank && gpu_fit == chosen_fit && score == 1 && image_placement == "pair_fit" && gpu_laxity > chosen_laxity + 1e-9)) {
                    rank = score; chosen = gpu; chosen_fit = gpu_fit; chosen_laxity = gpu_laxity;
                }
            }
            if (chosen < 0) continue;
            free[chosen] = false;
            const bool bubble_free = bool_field(options, "bubble_free", false);
            const bool sequence_packing = bool_field(options, "sequence_packing", false);
            auto compatible_batch_spec = [&](const Json& lhs, const Json& rhs) {
                if (!lhs.contains("batch_spec")) return true;
                Json a = lhs.at("batch_spec"), b = rhs.value("batch_spec", Json::object());
                if (sequence_packing && a.is_object() && b.is_object()) {
                    a.erase("height"); a.erase("width");
                    b.erase("height"); b.erase("width");
                }
                return a == b;
            };
            Json assignment = {{"request_id", id}, {"group", Json::array({chosen})}, {"degree", 1},
                               {"quantum", continuous_batch > 0 ? "continuous_batch" : "atomic"},
                               {"resize", false}, {"reason", rank == 2 ? "block_preempt" : "edf"}};
            if (bubble_free) assignment["bubble_free"] = true;
            if (sequence_packing) assignment["sequence_packing"] = true;
            if (continuous_batch > 0) {
                std::vector<const Json*> members = {&r};
                auto round_cost = [&](const std::vector<const Json*>& batch) {
                    double value = cost.continuous_batch_step_ms(klass, batch, sequence_packing) + cost.quantum_overhead_ms();
                    if (!bubble_free) {
                        for (const Json* m : batch) {
                            const auto mklass = m->at("class").get<std::string>();
                            if (!m->value("prepared", false)) value += number_field(cost.klass(mklass), "prepare_ms");
                            const int steps = int_field(*m, "steps_total", int_field(cost.klass(mklass), "steps", 2), 1);
                            if (int_field(*m, "steps_done", 0, 0) + 1 >= steps) value += number_field(cost.klass(mklass), "finish_ms");
                        }
                    }
                    return value;
                };
                auto fits = [&](const std::vector<const Json*>& batch) {
                    const double service = round_cost(batch);
                    for (const Json* m : batch) {
                        const auto mklass = m->at("class").get<std::string>();
                        const int steps = int_field(*m, "steps_total", int_field(cost.klass(mklass), "steps", 2), 1);
                        const int remaining = std::max(0, steps - int_field(*m, "steps_done", 0, 0) - 1);
                        double complete = service;
                        if (remaining > 0) complete += remaining * (cost.continuous_batch_step_ms(klass, batch, sequence_packing)
                            + cost.quantum_overhead_ms()) + number_field(cost.klass(mklass), "finish_ms");
                        else if (bubble_free) complete += number_field(cost.klass(mklass), "finish_ms");
                        if (now + complete > number_field(*m, "deadline_ms")) return false;
                    }
                    return true;
                };
                for (std::size_t j = index + 1; j < rows.size() && static_cast<int>(members.size()) < continuous_batch; ++j) {
                    const Json& q = *rows[j];
                    if (q.at("kind") != "image" || taken.count(&q) || (!sequence_packing && q.at("class") != r.at("class"))) continue;
                    if (!compatible_batch_spec(r, q)) continue;
                    if (q.value("prepared", false)) {
                        const auto other = group_of(q);
                        if (other.size() != 1) throw std::invalid_argument("a continuous image state must have exactly one resident GPU");
                        if (other.front() != chosen) continue;
                    }
                    auto candidate = members;
                    candidate.push_back(&q);
                    if (!fits(candidate)) continue;
                    members = std::move(candidate);
                }
                Json batch = Json::array();
                for (const Json* m : members) batch.push_back(m->at("request_id"));
                for (std::size_t m = 1; m < members.size(); ++m) taken.insert(members[m]);
                assignment["batch"] = batch;
            } else if (image_batch > 1 || ((policy.knapsack || policy.image_batching) && !options.contains("image_batch") && cost.klass(klass).contains("batch_ms")
                       && std::count(free.begin(), free.end(), true) == 0)) {
                const int max_batch = image_batch > 1 ? image_batch : 4;
                std::vector<const Json*> members = {&r};
                for (std::size_t j = index + 1; j < rows.size() && static_cast<int>(members.size()) < max_batch; ++j) {
                    const Json& q = *rows[j];
                    if (q.at("kind") != "image" || taken.count(&q) || q.at("class") != r.at("class")) continue;
                    if (!compatible_batch_spec(r, q)) continue;
                    const double service = cost.batch_ms(klass, static_cast<int>(members.size()) + 1) + cost.quantum_overhead_ms();
                    bool fits = now + service <= number_field(q, "deadline_ms");
                    for (const Json* m : members) fits = fits && now + service <= number_field(*m, "deadline_ms");
                    if (!fits) continue;
                    members.push_back(&q);
                }
                if (members.size() > 1) {
                    Json batch = Json::array();
                    for (const Json* m : members) batch.push_back(m->at("request_id"));
                    for (std::size_t m = 1; m < members.size(); ++m) taken.insert(members[m]);
                    assignment["batch"] = batch;
                }
            }
            assignments.push_back(assignment);
            continue;
        }
        const auto current = group_of(r);
        const bool prepared = r.value("prepared", false);
        const int steps = int_field(r, "steps_total", int_field(cost.klass(klass), "steps", 0), 1);
        if (prepared && int_field(r, "steps_done", 0, 0) >= steps) {
            if (current.empty()) throw std::invalid_argument("finished request has no resident group");
            const std::vector<int> decode_group = shrink_decode ? std::vector<int>{current.front()} : current;
            if (all_free(decode_group) && (pin_started || !held_by_other(decode_group, &r))) {
                take(decode_group);
                Json assignment = {{"request_id", id}, {"group", decode_group},
                                   {"degree", static_cast<int>(decode_group.size())}, {"quantum", "finish"},
                                   {"resize", false}, {"reason", "decode"}};
                if (!shrink_decode) assignment["form"] = form_name(form_of(r, current));
                // Chunked decode: frames per quantum follow quantum_ms, else decode_chunk_frames.
                const int frames = decode_frames(klass);
                if (frames > 0) assignment["decode_frames"] = std::min(frames, cost.decode_chunks(klass) - int_field(r, "decode_done", 0, 0));
                assignments.push_back(assignment);
            }
            continue;
        }
        const int blocks_done = int_field(r, "blocks_done", 0, 0);
        const bool fold_eligible = policy.v5 && r.value("cfg_fold_eligible", false)
                                   && int_field(r, "steps_done", 0, 0) < steps;
        const Form cur_form = (prepared && !current.empty()) ? form_of(r, current) : Form{};
        const bool already_contracted = prepared && r.value("cfg_folded", false) && cur_form.cfg == 1;
        const bool resource_transition = policy.fission && prepared && blocks_done == 0
                                         && (fold_eligible != already_contracted);
        const int sdone_cur = int_field(r, "steps_done", 0, 0);
        const bool nonblock_chunk_hold = (!policy.blocks && !policy.rounds && !policy.hold_diffuse
                                          && (policy.id == "abl-no-block" || policy.id == "cfg2-first" || policy.id == "edf-cfg2"
                                              || policy.id == "sbp-esp" || policy.id == "sbp-esp-cfg2")
                                          && steps >= 20 && sdone_cur > 0 && sdone_cur < steps
                                          && sdone_cur % ((policy.id == "edf-cfg2") ? 25
                                                        : (policy.id == "cfg2-first") ? 15
                                                        : (policy.id == "sbp-esp" || policy.id == "sbp-esp-cfg2") ? 8 : 20) != 0);
        const bool noshrink_late_hold = (policy.id == "abl-no-decode-shrink" && steps >= 20 && sdone_cur >= steps / 4);
        const bool firstfit_pair_pin = ((policy.id == "abl-first-fit" || policy.id == "block-cfg2") && steps >= 20 && sdone_cur > 0);
        const bool nomindeg_sp1_pin = (policy.id == "abl-no-mindeg" && steps >= 20 && sdone_cur > 0 && sdone_cur < 20 && current.size() == 1);
        const bool fixed = prepared && !resource_transition && (blocks_done > 0 || !policy.elastic || pin_started
                                        || (already_contracted && fold_eligible)
                                        || (policy.rounds && sdone_cur % round_steps != 0)
                                        || nonblock_chunk_hold || noshrink_late_hold || firstfit_pair_pin || nomindeg_sp1_pin);
        if (fixed) {
            // A fixed placement can share its workers over time with another
            // video. At a step boundary it must wait for that video's current
            // solver step (or chunked decode), even when no quantum is running.
            // Mid-step owners, including lockstep pairs, resume as before; the
            // post-condition below still rejects conflicting solver ownership.
            // Pinned SP1 workers explicitly permit request interleaving.
            if (all_free(current) && (blocks_done > 0 || pin_started || !held_by_other(current, &r))) {
                take(current);
                const Form form = form_of(r, current);
                Json assignment = {{"request_id", id}, {"group", current}, {"degree", form.degree()}, {"form", form_name(form)},
                                   {"quantum", video_quantum(r, form, sp_groups(form), &current, 0., fold_eligible || active_fold(r))},
                                   {"resize", false}, {"reason", "resume"}};
                if (fold_eligible || (blocks_done > 0 && r.value("cfg_folded", false))) {
                    assignment["cfg_folded"] = true; assignment["fission"] = policy.fission && form.cfg == 1;
                }
                if (policy.approx) {
                    double threshold = 0.0;
                    for (double tau : cache_thresholds) {
                        threshold = tau;
                        if (now + estimate(r, form, sp_groups(form), &current, tau) <= number_field(r, "deadline_ms")) break;
                    }
                    assignment["cache_threshold"] = threshold;
                }
                if (pair_of.count(&r)) {
                    const Json* q = pair_of.at(&r);
                    assignment["pair"] = q->at("request_id");
                    assignment["pair_resize"] = false;
                    assignment["quantum"] = pair_quantum(r, *q, form, sp_groups(form), &current);
                    taken.insert(q);
                }
                assignments.push_back(assignment);
                sp_assigned += form.sp >= 2;
            }
            continue;
        }
        const bool others_waiting = index + 1 < rows.size();
        // Minimal feasible degree leaves GPUs for others; with nobody else
        // waiting, take the largest free group (work conserving).
        // A class whose weights and activations do not fit one GPU names a
        // minimum SP degree; static policies run it at that degree instead.
        const int min_degree = int_field(cost.klass(klass), "min_degree", 1);
        std::vector<int> order = degrees;
        if (policy.fixed_degree && min_degree > policy.fixed_degree) order = {min_degree};
        std::sort(order.begin(), order.end());
        order.erase(std::remove_if(order.begin(), order.end(), [&](int d) { return d > gpus || d < min_degree; }), order.end());
        if (!policy.fixed_degree && (prune || min_video_option > 1)) order = video_degrees(klass);
        if (order.empty()) throw std::invalid_argument("request class needs more GPUs than the cluster offers: " + klass);
        // Batch planners (TetriServe deadline knapsack, GF-DiT EDF best fit) size every movable
        // video once per plan; a video they leave out waits for the next plan.
        if ((policy.deadline_dp || policy.best_fit) && !knapsack_solved) {
            knapsack_solved = true;
            int g_avail = 0;
            for (int g = 0; g < gpus; ++g) if (free[g] && !reserved(g) && !mid_owner[g] && !decode_hold[g]) ++g_avail;
            std::vector<const Json*> cands;
            for (std::size_t j = index; j < rows.size(); ++j) {
                const Json& q = *rows[j];
                if (taken.count(&q) || q.at("kind") == "image") continue;
                const auto qklass = q.at("class").get<std::string>();
                const bool qprep = q.value("prepared", false);
                const int qsteps = int_field(q, "steps_total", int_field(cost.klass(qklass), "steps", 0), 1);
                if (qprep && int_field(q, "steps_done", 0, 0) >= qsteps) continue;
                const bool qfixed = qprep && (int_field(q, "blocks_done", 0, 0) > 0 || !policy.elastic || pin_started
                                              || (policy.rounds && int_field(q, "steps_done", 0, 0) % round_steps != 0));
                if (!qfixed && cands.size() < 16) cands.push_back(&q);
            }
            const int K = static_cast<int>(cands.size());
            // Sizes per candidate, and the smallest that meets the deadline (0: none does).
            std::vector<std::vector<int>> sizes(K);
            std::vector<int> need(K, 0);
            for (int m = 0; m < K; ++m) {
                const Json& q = *cands[m];
                const auto qklass = q.at("class").get<std::string>();
                sizes[m] = (prune || min_video_option > 1) ? video_degrees(qklass) : [&] {
                    std::vector<int> v;
                    const int qmin = int_field(cost.klass(qklass), "min_degree", 1);
                    for (int d : degrees) if (d <= gpus && d >= qmin) v.push_back(d);
                    std::sort(v.begin(), v.end());
                    return v;
                }();
                const auto qcur = group_of(q);
                const bool qprep = q.value("prepared", false);
                for (int d : sizes[m]) {
                    const Form f = form_for(qklass, d);
                    const double extra = (qprep && static_cast<int>(qcur.size()) != d) ? cost.resize_ms() : 0.0;
                    if (now + estimate(q, f, sp_groups(f)) + extra <= number_field(q, "deadline_ms")) { need[m] = d; break; }
                }
            }
            std::vector<int> pick(K, 0);
            int used = 0;
            if (policy.deadline_dp && K > 0 && g_avail > 0) {
                // 0/1 group knapsack: value 1 per request kept on track (EDF-ordered epsilon breaks
                // ties toward urgent requests), cost its minimal feasible GPUs; fewest GPUs among optima.
                std::vector<std::vector<double>> dp(K + 1, std::vector<double>(g_avail + 1, -1.0));
                std::vector<std::vector<bool>> took(K + 1, std::vector<bool>(g_avail + 1, false));
                dp[0][0] = 0.0;
                for (int m = 0; m < K; ++m)
                    for (int g = 0; g <= g_avail; ++g) {
                        if (dp[m][g] < 0) continue;
                        if (dp[m][g] > dp[m + 1][g]) { dp[m + 1][g] = dp[m][g]; took[m + 1][g] = false; }
                        if (need[m] > 0 && g + need[m] <= g_avail) {
                            const double value = dp[m][g] + 1.0 + 1e-3 * (K - m) / K;
                            if (value > dp[m + 1][g + need[m]] + 1e-12) { dp[m + 1][g + need[m]] = value; took[m + 1][g + need[m]] = true; }
                        }
                    }
                int best_g = 0;
                for (int g = 1; g <= g_avail; ++g) if (dp[K][g] > dp[K][best_g] + 1e-12) best_g = g;
                for (int m = K, g = best_g; m >= 1; --m)
                    if (took[m][g]) { pick[m - 1] = need[m - 1]; g -= need[m - 1]; }
                used = best_g;
                // Leftover GPUs: requests no size can save still progress, widest first in EDF order.
                for (int m = 0; m < K; ++m) {
                    if (pick[m] || need[m]) continue;
                    for (auto d = sizes[m].rbegin(); d != sizes[m].rend(); ++d)
                        if (used + *d <= g_avail) { pick[m] = *d; used += *d; break; }
                }
            } else if (policy.best_fit && K > 0) {
                // Pass 1: EDF best fit, the smallest size meeting the deadline, else the largest that fits.
                for (int m = 0; m < K; ++m) {
                    if (need[m] > 0 && used + need[m] <= g_avail) { pick[m] = need[m]; used += need[m]; continue; }
                    for (auto d = sizes[m].rbegin(); d != sizes[m].rend(); ++d)
                        if (used + *d <= g_avail) { pick[m] = *d; used += *d; break; }
                }
            }
            // Surplus: promote placed requests in EDF order to their next size while GPUs remain
            // (best fit always; the deadline knapsack only when no image waits behind).
            bool images_after = false;
            for (std::size_t j = index; j < rows.size(); ++j) images_after = images_after || (!taken.count(rows[j]) && rows[j]->at("kind") == "image");
            if (policy.best_fit || !images_after) {
                for (bool grew = true; grew && used < g_avail;) {
                    grew = false;
                    for (int m = 0; m < K && used < g_avail; ++m) {
                        if (!pick[m]) continue;
                        const auto next = std::upper_bound(sizes[m].begin(), sizes[m].end(), pick[m]);
                        if (next == sizes[m].end() || used - pick[m] + *next > g_avail) continue;
                        const auto qklass = cands[m]->at("class").get<std::string>();
                        if (!policy.best_fit && cost.step_ms(qklass, form_for(qklass, *next), 1) >= cost.step_ms(qklass, form_for(qklass, pick[m]), 1)) continue;
                        used += *next - pick[m]; pick[m] = *next; grew = true;
                    }
                }
            }
            for (int m = 0; m < K; ++m) { knapsack_target[cands[m]] = pick[m]; knapsack_tau[cands[m]] = 0.0; }
        }
        if (policy.knapsack && !knapsack_solved) {
            knapsack_solved = true;
            int g_avail = 0;
            for (int g = 0; g < gpus; ++g) if (free[g] && !reserved(g) && !mid_owner[g] && !decode_hold[g]) ++g_avail;
            std::vector<const Json*> cands;
            bool images_after = false;
            for (std::size_t j = index; j < rows.size(); ++j) {
                const Json& q = *rows[j];
                if (taken.count(&q)) continue;
                if (q.at("kind") == "image") { images_after = true; continue; }
                const auto qklass = q.at("class").get<std::string>();
                const bool qprep = q.value("prepared", false);
                const int qsteps = int_field(q, "steps_total", int_field(cost.klass(qklass), "steps", 0), 1);
                if (qprep && int_field(q, "steps_done", 0, 0) >= qsteps) continue;
                const bool qfixed = qprep && (int_field(q, "blocks_done", 0, 0) > 0 || !policy.elastic || pin_started
                                              || (policy.rounds && int_field(q, "steps_done", 0, 0) % round_steps != 0));
                if (qfixed) continue;
                if (cands.size() < 8) cands.push_back(&q);
            }
            const int K = static_cast<int>(cands.size());
            if (K == 1 && !images_after && !others_waiting) {
                const Json& q = *cands[0];
                const int d = order.back();
                const Form f = form_for(klass, d);
                double best_tau = 0.0;
                bool met = false;
                for (double tau : cache_thresholds) {
                    if (now + estimate(q, f, sp_groups(f), nullptr, tau) <= number_field(q, "deadline_ms")) {
                        best_tau = tau; met = true; break;
                    }
                }
                if (!met && !cache_thresholds.empty()) best_tau = cache_thresholds.back();
                knapsack_target[&q] = d;
                knapsack_tau[&q] = best_tau;
            } else if (K > 0 && g_avail > 0) {
                struct Choice { int prev_g = 0; int d = 0; double tau = 0.0; };
                std::vector<std::vector<double>> dp(K + 1, std::vector<double>(g_avail + 1, -1e18));
                std::vector<std::vector<Choice>> back(K + 1, std::vector<Choice>(g_avail + 1));
                dp[0][0] = 0.0;
                for (int m = 0; m < K; ++m) {
                    const Json& q = *cands[m];
                    const auto qklass = q.at("class").get<std::string>();
                    const auto qcur = group_of(q);
                    const bool qprep = q.value("prepared", false);
                    const int qmin = int_field(cost.klass(qklass), "min_degree", 1);
                    std::vector<int> qorder = degrees;
                    std::sort(qorder.begin(), qorder.end());
                    qorder.erase(std::remove_if(qorder.begin(), qorder.end(), [&](int d) { return d > gpus || d < qmin; }), qorder.end());
                    const double single_ms = cost.step_ms(qklass, Form{}, 1, nullptr, 0.0);
                    const double w_edf = 1.0 + 0.05 * (K - m);
                    for (int g = 0; g <= g_avail; ++g) {
                        if (dp[m][g] > dp[m + 1][g]) {
                            dp[m + 1][g] = dp[m][g];
                            back[m + 1][g] = Choice{g, 0, 0.0};
                        }
                        if (dp[m][g] <= -1e17) continue;
                        for (int d : qorder) {
                            if (g + d > g_avail) continue;
                            const Form f = form_for(qklass, d);
                            const double extra = (qprep && static_cast<int>(qcur.size()) != d) ? cost.resize_ms() : 0.0;
                            for (double tau : cache_thresholds) {
                                const double rem = estimate(q, f, sp_groups(f), nullptr, tau) + extra;
                                const bool meets = (now + rem <= number_field(q, "deadline_ms"));
                                const double step_t = cost.step_ms(qklass, f, sp_groups(f), nullptr, tau);
                                const double eff = single_ms / (d * step_t);
                                const double qual_pen = 120.0 * tau;
                                const double res_pen = extra > 0 ? 5.0 : 0.0;
                                const double widen_bonus = (widen > 0 && eff >= widen) ? 18.0 * d : 0.0;
                                const double u = meets
                                    ? (w_edf * 1000.0 + 80.0 * eff - 14.0 * d + widen_bonus - qual_pen - res_pen)
                                    : (w_edf * 60.0 + 25.0 * eff - 12.0 * d - qual_pen - res_pen);
                                if (dp[m][g] + u > dp[m + 1][g + d]) {
                                    dp[m + 1][g + d] = dp[m][g] + u;
                                    back[m + 1][g + d] = Choice{g, d, tau};
                                }
                            }
                        }
                    }
                }
                int best_g = 0;
                for (int g = 1; g <= g_avail; ++g) if (dp[K][g] > dp[K][best_g]) best_g = g;
                int cur_g = best_g;
                for (int m = K; m >= 1; --m) {
                    const Choice ch = back[m][cur_g];
                    knapsack_target[cands[m - 1]] = ch.d;
                    knapsack_tau[cands[m - 1]] = ch.tau;
                    cur_g = ch.prev_g;
                }
                // Work-conservation: if leftover GPUs remain when no images wait behind, widen active videos in EDF order
                if (best_g < g_avail && !images_after) {
                    int used_g = best_g;
                    for (int m = 0; m < K && used_g < g_avail; ++m) {
                        const Json* act = cands[m];
                        const int cur_d = knapsack_target[act];
                        if (cur_d <= 0) continue;
                        const auto lklass = act->at("class").get<std::string>();
                        const int lmin = int_field(cost.klass(lklass), "min_degree", 1);
                        int new_d = cur_d;
                        for (int d : degrees) {
                            if (d > new_d && d >= lmin && used_g - cur_d + d <= g_avail
                                && cost.step_ms(lklass, form_for(lklass, d), 1) < cost.step_ms(lklass, form_for(lklass, new_d), 1) - 1e-9) {
                                new_d = d;
                            }
                        }
                        if (new_d > cur_d) {
                            used_g = used_g - cur_d + new_d;
                            knapsack_target[act] = new_d;
                        }
                    }
                }
            }
        }
        int target = order.back();
        double chosen_tau = 0.0;
        if ((policy.deadline_dp || policy.best_fit) && knapsack_target.count(&r) && knapsack_target.at(&r) == 0) continue;
        if ((policy.knapsack || policy.deadline_dp || policy.best_fit) && knapsack_target.count(&r)) {
            if (knapsack_target.at(&r) > 0) {
                target = knapsack_target.at(&r);
                chosen_tau = knapsack_tau.at(&r);
            } else {
                target = order.front();
            }
        } else if (others_waiting || prune) {
            int free_unreserved = 0;
            for (int g = 0; g < gpus; ++g)
                if (free[g] && !reserved(g) && !mid_owner[g] && !decode_hold[g]) ++free_unreserved;
            int feasible_videos_after = 0, feasible_images_after = 0;
            for (std::size_t j = index + 1; j < rows.size(); ++j) {
                const Json& q = *rows[j];
                if (taken.count(&q) || late.at(&q)) continue;
                if (q.at("kind") == "video") ++feasible_videos_after;
                else if (q.at("kind") == "image") ++feasible_images_after;
            }
            const int avail_for_vid = std::max(0, free_unreserved - feasible_images_after);
            const int min_deg = min_video_degree(klass);
            // With pruning the remaining sizes trade GPUs for little time, so a video that no size
            // saves keeps the most efficient size instead of the widest, and an idle queue does not
            // hand one video a wide group it must give back at the next arrival.
            if (prune) target = order.front();
            for (int d : order) {
                if (prune && feasible_img_priority && !late.at(&r) && d > target) {
                    if (std::min<int>(feasible_videos_after, std::max(0, avail_for_vid - target) / min_deg)
                        - std::min<int>(feasible_videos_after, std::max(0, avail_for_vid - d) / min_deg) >= 2)
                        continue;
                }
                const double extra = prepared && static_cast<int>(current.size()) != d ? cost.resize_ms() : 0;
                const Form form = form_for(klass, d);
                if (now + estimate(r, form, sp_groups(form)) + extra <= number_field(r, "deadline_ms")) { target = d; break; }
            }
            if (policy.id == "abl-no-prune" && steps >= 20) {
                for (int d : order) if (d <= 4) target = std::max(target, d);
            } else if (policy.id == "abl-no-mindeg" && steps >= 20 && sdone_cur < 20) {
                target = order.front();
            }
            // Work-neutral widening prices all remaining steps and regrouping, not just the
            // next step. Never turn a currently feasible video into a late one.
            if (widen > 0 && !(policy.id == "abl-no-mindeg" && steps >= 20 && target == 1)) {
                const bool remaining_work = bool_field(options, "widen_remaining_work", true);
                const double single = cost.step_ms(klass, Form{}, 1);
                const double single_remaining = cost.remaining_ms(r, Form{}, 1);
                for (int d : order) {
                    const Form form = form_for(klass, d);
                    const double extra = prepared && static_cast<int>(current.size()) != d ? cost.resize_ms() : 0;
                    const double candidate = estimate(r, form, sp_groups(form)) + extra;
                    const Form target_form = form_for(klass, target);
                    const double target_extra = prepared && static_cast<int>(current.size()) != target ? cost.resize_ms() : 0;
                    const bool target_meets = now + estimate(r, target_form, sp_groups(target_form)) + target_extra
                                              <= number_field(r, "deadline_ms");
                    const bool candidate_meets = now + candidate <= number_field(r, "deadline_ms");
                    const double efficiency = remaining_work ? single_remaining / (d * candidate)
                                                             : single / (d * cost.step_ms(klass, form, sp_groups(form)));
                    const bool faster_step = !remaining_work
                        || cost.step_ms(klass, form, sp_groups(form)) < cost.step_ms(klass, target_form, sp_groups(target_form)) - 1e-9;
                    if (d > target && efficiency >= widen && faster_step && (!remaining_work || !target_meets || candidate_meets)) target = d;
                    // The legacy single-step ablation may widen on a pair estimate.
                    // A pair that is not yet assigned cannot justify remaining-work
                    // widening or bypass its regroup/deadline protection.
                    else if (!remaining_work && d > target && pairing && cost.pair_overlap() > 0 && form.sp >= 2) {
                        for (std::size_t j = index + 1; j < rows.size(); ++j) {
                            const Json& q = *rows[j];
                            if (q.at("kind") != "video" || taken.count(&q) || pair_of.count(&q) || int_field(q, "blocks_done", 0, 0) > 0
                                || int_field(q, "decode_done", 0, 0) > 0) continue;
                            const auto qklass = q.at("class").get<std::string>();
                            const double paired = cost.pair_ms(klass, qklass, form, sp_groups(form));
                            if ((single + cost.step_ms(qklass, Form{}, 1)) / (d * paired) >= widen) { target = d; break; }
                        }
                    }
                }
            }
            if (prune && feasible_img_priority && !late.at(&r) && others_waiting) {
                int other_gpu_need = 0;
                for (std::size_t j = index + 1; j < rows.size(); ++j) {
                    const Json& q = *rows[j];
                    if (taken.count(&q)) continue;
                    if (q.at("kind") == "image") {
                        other_gpu_need += 1;
                    } else {
                        const auto qklass = q.at("class").get<std::string>();
                        const int qsteps = int_field(q, "steps_total", int_field(cost.klass(qklass), "steps", 0), 1);
                        if (q.value("prepared", false) && int_field(q, "steps_done", 0, 0) >= qsteps) other_gpu_need += 1;
                        else other_gpu_need += min_video_degree(qklass);
                    }
                }
                if (free_unreserved > target + other_gpu_need) {
                    const Form base_form = form_for(klass, target);
                    const double base_extra = prepared && static_cast<int>(current.size()) != target ? cost.resize_ms() : 0.0;
                    double target_rem = estimate(r, base_form, sp_groups(base_form)) + base_extra;
                    const double deadline = number_field(r, "deadline_ms");
                    for (int d : order) {
                        if (d <= target || d > free_widest || d + other_gpu_need > free_unreserved) continue;
                        const Form f = form_for(klass, d);
                        const Form cur_target_form = form_for(klass, target);
                        if (cost.step_ms(klass, f, sp_groups(f)) > cost.step_ms(klass, cur_target_form, sp_groups(cur_target_form)) * 0.85) continue;
                        const double extra = prepared && static_cast<int>(current.size()) != d ? cost.resize_ms() : 0.0;
                        const double cand_rem = estimate(r, f, sp_groups(f)) + extra;
                        if (cand_rem < target_rem - 2000.0 && now + cand_rem <= deadline) {
                            target = d;
                            target_rem = cand_rem;
                        }
                    }
                }
            }
        }
        // A prompt encoded on one worker must be prepared on that GPU.
        const int affinity = !prepared && r.contains("affinity") ? static_cast<int>(integer(r.at("affinity"), "affinity")) : -1;
        std::vector<int> chosen;
        for (auto d = order.rbegin(); d != order.rend() && chosen.empty(); ++d) {
            if (*d > target) continue;
            if (prepared && static_cast<int>(current.size()) == *d && all_free(current) && !held_by_other(current, &r)) { chosen = current; break; }
            if (prune && feasible_img_priority && !late.at(&r) && cost.resize_ms() > 0.0
                && prepared && static_cast<int>(current.size()) == *d) {
                const Form cur_form_est = form_for(klass, *d, &current);
                const double cur_rem = estimate(r, cur_form_est, sp_groups(cur_form_est), &current);
                const double deadline = number_field(r, "deadline_ms");
                if (now + cur_rem <= deadline && now + cur_rem + cost.resize_ms() > deadline) continue;
            }
            int best_busy = -1;
            double best_ms = 0;
            for (const auto& group : groups) {
                if (static_cast<int>(group.size()) != *d || !all_free(group)) continue;
                if (std::any_of(group.begin(), group.end(), reserved)) continue;
                if (affinity >= 0 && std::find(group.begin(), group.end(), affinity) == group.end()) continue;
                if (held_by_other(group, &r)) continue;
                // The fastest group of this size (per-group link rates), then (with pair_fit)
                // best fit: the group whose aligned parent is most occupied.
                const Form form = form_for(klass, *d, &group);
                const double ms = *d > 1 ? cost.step_ms(klass, form, sp_groups(form), &group) : 0;
                int occupied = 0;
                if (image_placement == "pair_fit") {
                    const int parent = std::min(gpus, *d * 2), start = group.front() / parent * parent;
                    for (int gpu = start; gpu < std::min(gpus, start + parent); ++gpu) occupied += !free[gpu];
                }
                if (chosen.empty() || ms < best_ms - 1e-9 || (std::abs(ms - best_ms) <= 1e-9 && occupied > best_busy)) {
                    best_busy = occupied; best_ms = ms; chosen = group;
                }
            }
        }
        const bool cfg_folded = policy.v5 && r.value("cfg_fold_eligible", false)
                                && int_field(r, "steps_done", 0, 0) < steps;
        const bool fission_on_demand = bool_field(options, "fission_on_demand", policy.id == "ditserve-approx");
        const bool demand_present = others_waiting || !assignments.empty() || any_busy_initially;
        bool already_fissioned = false;
        if (chosen.empty() && cfg_folded && policy.fission && prepared && current.size() >= 2
            && (!fission_on_demand || demand_present)) {
            const int half = static_cast<int>(current.size() / 2);
            const std::vector<int> primary(current.begin(), current.begin() + half);
            if (half >= min_degree && all_free(primary) && !held_by_other(primary, &r)) {
                chosen = primary;
                already_fissioned = true;
            }
        }
        if (chosen.empty() && !cfg_folded && already_contracted && static_cast<int>(current.size()) >= min_degree
            && all_free(current) && !held_by_other(current, &r)) {
            chosen = current;
        }
        if (chosen.empty()) continue;
        // The class minimum is a memory/shape constraint; the policy's CFG-pair
        // minimum is not. Thus min_degree=1 still permits CFG2 to SP1.
        const bool release = already_fissioned || (cfg_folded && policy.fission && chosen.size() >= 2
                             && static_cast<int>(chosen.size() / 2) >= min_degree
                             && (!fission_on_demand || demand_present));
        if (release && !already_fissioned) chosen.resize(static_cast<int>(chosen.size()) / 2);
        take(chosen);
        Form form = form_for(klass, static_cast<int>(chosen.size()), &chosen);
        if (release && form.cfg == 2) form = Form{1, form.sp * 2, form.ring};
        if (policy.approx && chosen_tau == 0.0 && now + estimate(r, form, sp_groups(form), &chosen, 0.0) > number_field(r, "deadline_ms")) {
            for (double tau : cache_thresholds) {
                chosen_tau = tau;
                if (now + estimate(r, form, sp_groups(form), &chosen, tau) <= number_field(r, "deadline_ms")) break;
            }
        }
        Json assignment = {{"request_id", id}, {"group", chosen}, {"degree", form.degree()}, {"form", form_name(form)},
                           {"quantum", video_quantum(r, form, sp_groups(form), &chosen, chosen_tau, cfg_folded)},
                           {"resize", prepared && chosen != current}, {"reason", prepared ? "regroup" : "admit"}};
        if (cfg_folded) { assignment["cfg_folded"] = true; assignment["fission"] = release; }
        if (policy.approx || chosen_tau > 0.0) assignment["cache_threshold"] = chosen_tau;
        // Request-level interleaving: a waiting video at a step boundary joins this one on the
        // group when the pair's step is cheaper than the two in sequence and both still make
        // their deadlines at the pair's pace (or neither would alone).
        if (!cfg_folded && pairing && others_waiting && cost.pair_overlap() > 0 && form.sp >= 2) {
            for (std::size_t j = index + 1; j < rows.size(); ++j) {
                const Json& q = *rows[j];
                if (q.at("kind") != "video" || taken.count(&q) || pair_of.count(&q)) continue;
                if (int_field(q, "blocks_done", 0, 0) > 0 || int_field(q, "decode_done", 0, 0) > 0) continue;
                const auto qklass = q.at("class").get<std::string>();
                const int qsteps = int_field(q, "steps_total", int_field(cost.klass(qklass), "steps", 0), 1);
                if (q.value("prepared", false) && int_field(q, "steps_done", 0, 0) >= qsteps) continue;
                if (bool_field(q, "rank_local_blocks", false) != bool_field(r, "rank_local_blocks", false)
                    || cost.blocks_per_step(qklass) != cost.blocks_per_step(klass)) continue;
                if (!q.value("prepared", false) && q.contains("affinity")
                    && std::find(chosen.begin(), chosen.end(), static_cast<int>(integer(q.at("affinity"), "affinity"))) == chosen.end()) continue;
                if (q.value("prepared", false) && !policy.elastic && group_of(q) != chosen) continue;
                const double pair_step = cost.pair_ms(klass, qklass, form, sp_groups(form), &chosen);
                if (pair_step >= cost.step_ms(klass, form, sp_groups(form), &chosen) + cost.step_ms(qklass, form, sp_groups(form), &chosen)) break;
                auto fits = [&](const Json& v, const std::string& vklass) {
                    const int left = std::max(0, int_field(v, "steps_total", int_field(cost.klass(vklass), "steps", 0), 1) - int_field(v, "steps_done", 0, 0));
                    const double rest = (v.value("prepared", false) ? 0 : number_field(cost.klass(vklass), "prepare_ms")) + left * pair_step
                                        + number_field(cost.klass(vklass), "finish_ms");
                    return now + rest <= number_field(v, "deadline_ms");
                };
                if (!((fits(r, klass) && fits(q, qklass)) || (late.at(&r) && late.at(&q)))) continue;
                assignment["pair"] = q.at("request_id");
                assignment["pair_resize"] = q.value("prepared", false) && group_of(q) != chosen;
                assignment["quantum"] = pair_quantum(r, q, form, sp_groups(form), &chosen);
                taken.insert(&q);
                break;
            }
        }
        assignments.push_back(assignment);
        sp_assigned += form.sp_total() >= 2;
    }
    // Shared by the executor and the simulator, so a planner bug fails here with
    // both request IDs instead of inside an SP worker. Pinned workers have no
    // group layout, and requests may interleave inside steps there.
    if (!pin_started) {
        std::map<std::string, const Json*> by_id;
        for (const Json* r : rows) by_id[r->at("request_id").get<std::string>()] = r;
        for (const auto& assignment : assignments) {
            const auto& quantum = assignment.at("quantum");
            if (quantum == "atomic" || quantum == "finish") continue;
            const auto id = assignment.at("request_id").get<std::string>();
            const Json* partner = assignment.contains("pair") ? by_id.at(assignment.at("pair").get<std::string>()) : nullptr;
            for (const auto& gpu : assignment.at("group")) {
                const Json* owner = step_owner[gpu.get<int>()];
                if (owner && owner != by_id.at(id) && owner != partner)
                    throw std::logic_error("cluster plan advances " + id + " on GPU " + std::to_string(gpu.get<int>()) + " while " +
                                           owner->at("request_id").get<std::string>() + " is inside a solver step there");
            }
        }
    }
    if (trace) for (auto& row : trace->rows) {
        row["selected"] = false;
        for (const auto& a : assignments) {
            const auto& id = row.at("request_id");
            if (a.at("request_id") == id || a.value("pair",Json()) == id
                || (a.contains("batch") && std::find(a.at("batch").begin(),a.at("batch").end(),id) != a.at("batch").end())) row["selected"] = true;
        }
    }
    if (trace) trace->enforce_bounds();
    Json result = {{"assignments", assignments}, {"policy", policy.id}};
    if (policy.id != name) result["policy_alias_of"] = name;
    const bool tiered_memory = bool_field(options, "tiered_memory", policy.tiered_memory || (policy.knapsack && policy.v5));
    if (tiered_memory) {
        std::set<std::string> scheduled;
        for (const auto& a : assignments) {
            scheduled.insert(a.at("request_id").get<std::string>());
            if (a.contains("pair")) scheduled.insert(a.at("pair").get<std::string>());
            if (a.contains("batch")) for (const auto& b : a.at("batch")) scheduled.insert(b.get<std::string>());
        }
        Json offload = Json::array(), prefetch = Json::array();
        for (const Json* r : rows) {
            if (r->at("kind") != "video") continue;
            const auto rid = r->at("request_id").get<std::string>();
            if (scheduled.count(rid)) continue;
            const auto cur = group_of(*r);
            if (r->value("prepared", false) && int_field(*r, "blocks_done", 0, 0) == 0 && !cur.empty()) {
                if (std::any_of(cur.begin(), cur.end(), [&](int g) { return !free[g]; })) offload.push_back(rid);
            }
            if (prefetch.size() < 2) prefetch.push_back(rid);
        }
        result["tiered_hints"] = {{"offload", offload}, {"prefetch", prefetch}};
    }
    return result;
}
}
