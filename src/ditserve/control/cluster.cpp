#include "ditserve/control/cluster.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace ditserve {
namespace {
bool power_of_two(int value) { return value > 0 && (value & (value - 1)) == 0; }
std::string degree_key(int degree) { return std::to_string(degree); }

struct Policy {
    int fixed_degree = 0;    // static_spK
    bool elastic = false;    // regroup at step boundaries
    bool blocks = false;     // advance in block quanta, lazy intra-step preemption
};
Policy parse(const std::string& name) {
    Policy policy;
    if (name.rfind("static_sp", 0) == 0) {
        const auto suffix = name.substr(9);
        if (suffix != "1" && suffix != "2" && suffix != "4" && suffix != "8") throw std::invalid_argument("unknown cluster policy: " + name);
        policy.fixed_degree = std::stoi(suffix);
    } else if (name == "step_elastic") policy.elastic = true;
    else if (name == "block_elastic") policy.elastic = policy.blocks = true;
    else throw std::invalid_argument("unknown cluster policy: " + name);
    return policy;
}
std::vector<int> group_of(const Json& request) {
    std::vector<int> group;
    for (const auto& gpu : request.value("group", Json::array())) group.push_back(static_cast<int>(integer(gpu, "group GPU")));
    return group;
}
}

bool cluster_policy(const std::string& policy) {
    try { parse(policy); return true; } catch (const std::invalid_argument&) { return false; }
}

std::vector<std::vector<int>> cluster_groups(int gpus, const std::vector<int>& degrees) {
    if (gpus < 1) throw std::invalid_argument("cluster requires at least one GPU");
    std::vector<std::vector<int>> groups;
    std::set<int> seen;
    for (int degree : degrees) {
        if (!power_of_two(degree)) throw std::invalid_argument("SP degree must be a power of two");
        if (degree > gpus || !seen.insert(degree).second) continue;
        for (int start = 0; start + degree <= gpus; start += degree) {
            std::vector<int> group(degree);
            for (int i = 0; i < degree; ++i) group[i] = start + i;
            groups.push_back(group);
        }
    }
    return groups;
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
        if (kind == "image") { number_field(row, "service_ms", 0, true); continue; }
        if (kind != "video") throw std::invalid_argument("cost class kind must be video or image");
        int_field(row, "steps", 0); int_field(row, "blocks_per_step", 1); int_field(row, "decode_chunks", 1);
        if (!power_of_two(int_field(row, "min_degree", 1))) throw std::invalid_argument("min_degree must be a power of two");
        number_field(row, "step_compute_ms", 0, true); number_field(row, "finish_ms");
        number_field(row, "prepare_ms"); number_field(row, "comm_mb_per_step"); number_field(row, "collectives_per_step");
    }
    efficiency_ = number_field(profile, "sp_efficiency", 1, true);
    if (efficiency_ > 1) throw std::invalid_argument("sp_efficiency must be in (0, 1]");
    resize_ms_ = number_field(profile, "resize_ms");
}

const Json& ClusterCost::klass(const std::string& name) const {
    const auto& classes = profile_.at("classes");
    if (!classes.contains(name)) throw std::invalid_argument("unknown cost class: " + name);
    return classes.at(name);
}

double ClusterCost::step_ms(const std::string& name, int degree) const {
    const auto& row = klass(name);
    if (row.at("kind") != "video") throw std::invalid_argument("image classes have no solver steps");
    const double compute = number_field(row, "step_compute_ms", 0, true);
    if (degree == 1) return compute;
    const auto& links = profile_.value("links", Json::object());
    if (!links.contains(degree_key(degree))) throw std::invalid_argument("cost profile has no link model for SP degree " + degree_key(degree));
    const auto& link = links.at(degree_key(degree));
    const double d = degree;
    return compute / (d * efficiency_)
        + number_field(row, "comm_mb_per_step") * (d - 1) / (d * d) / number_field(link, "gbps", 0, true)
        + number_field(row, "collectives_per_step") * number_field(link, "latency_ms");
}

int ClusterCost::blocks_per_step(const std::string& name) const { return int_field(klass(name), "blocks_per_step", 1); }

int ClusterCost::decode_chunks(const std::string& name) const { return int_field(klass(name), "decode_chunks", 1); }

double ClusterCost::chunk_ms(const std::string& name) const { return number_field(klass(name), "finish_ms") / decode_chunks(name); }

double ClusterCost::block_ms(const std::string& name, int degree) const { return step_ms(name, degree) / blocks_per_step(name); }

double ClusterCost::remaining_ms(const Json& request, int degree) const {
    const auto name = request.at("class").get<std::string>();
    const auto& row = klass(name);
    if (row.at("kind") == "image") return number_field(row, "service_ms");
    const int steps = int_field(request, "steps_total", int_field(row, "steps", 0), 1), done = int_field(request, "steps_done", 0, 0);
    const int blocks = int_field(request, "blocks_done", 0, 0);
    double total = request.value("prepared", false) ? 0 : number_field(row, "prepare_ms");
    total += std::max(0, steps - done) * step_ms(name, degree) - blocks * block_ms(name, degree);
    return total + (decode_chunks(name) - std::min(int_field(request, "decode_done", 0, 0), decode_chunks(name))) * chunk_ms(name);
}

Json cluster_plan(const Json& state, const std::string& name, const ClusterCost& cost, const Json& options) {
    const Policy policy = parse(name);
    const double now = number(state.at("now_ms"), "now_ms");
    const int gpus = static_cast<int>(integer(state.at("gpus"), "gpus"));
    std::vector<int> degrees;
    for (const auto& d : options.value("degrees", Json::array({1, 2, 4, 8}))) degrees.push_back(static_cast<int>(integer(d, "degree")));
    if (policy.fixed_degree) degrees = {policy.fixed_degree};
    if (cluster_groups(gpus, degrees).empty()) throw std::invalid_argument("no SP group fits the cluster");
    // Every aligned group stays addressable: a class minimum can exceed the policy's degrees.
    const auto groups = cluster_groups(gpus, {1, 2, 4, 8});
    const int quantum_blocks = int_field(options, "quantum_blocks", 4);
    // Optional time target: blocks per quantum follow the class and SP degree,
    // so a preemption check happens about every quantum_ms of GPU work.
    const double quantum_ms = number_field(options, "quantum_ms", 0);
    auto video_quantum = [&](const std::string& klass, int degree) -> Json {
        if (!policy.blocks) return "step";
        if (quantum_ms <= 0) return quantum_blocks;
        const int blocks = static_cast<int>(std::floor(quantum_ms / cost.block_ms(klass, degree)));
        return std::max(1, std::min(blocks, cost.blocks_per_step(klass)));
    };
    const bool lazy = bool_field(options, "lazy_preemption", true);
    // Real backends without state migration keep a started request on its group.
    const bool pin_started = bool_field(options, "pin_started", false);
    // Static partition baseline: the last R GPUs serve images only.
    const int reserve = int_field(options, "reserve_image_gpus", 0, 0);
    if (reserve >= gpus) throw std::invalid_argument("reserve_image_gpus must leave a GPU for videos");
    auto reserved = [&](int gpu) { return gpu >= gpus - reserve; };

    std::vector<bool> free(gpus, true);
    const auto busy = state.value("busy", Json::array());
    if (!busy.is_array() || static_cast<int>(busy.size()) > gpus) throw std::invalid_argument("busy must list at most one flag per GPU");
    for (std::size_t g = 0; g < busy.size(); ++g) free[g] = !busy[g].get<bool>();

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
    // Feasible-first EDF: a request that cannot meet its deadline even on the
    // largest group yields to those that still can, so an overloaded queue of
    // late videos does not starve newly arrived short requests.
    const bool demote_late = bool_field(options, "demote_late", true);
    // Late requests shorter than this keep their EDF position: they barely
    // delay feasible work, and demoting them behind a long video only turns a
    // small miss into a very long tail.
    const double late_short_ms = number_field(options, "late_short_ms", 2000);
    const int widest = *std::max_element(degrees.begin(), degrees.end());
    std::map<const Json*, bool> late;
    for (const Json* r : rows) {
        const int best = r->at("kind") == "image" ? 1 : std::min(std::max(widest, int_field(cost.klass(r->at("class").get<std::string>()), "min_degree", 1)), gpus);
        int degree = best;
        while (degree > 1 && !power_of_two(degree)) --degree;
        const auto group = group_of(*r);
        if (r->value("prepared", false) && int_field(*r, "blocks_done", 0, 0) > 0) degree = static_cast<int>(group.size());
        const double remaining = cost.remaining_ms(*r, degree);
        late[r] = demote_late && remaining > late_short_ms && now + remaining > number_field(*r, "deadline_ms");
    }
    auto edf = [&](const Json* r) {
        return std::make_tuple(late.at(r), number_field(*r, "deadline_ms"), number_field(*r, "arrival_ms"), r->at("request_id").get<std::string>());
    };
    std::sort(rows.begin(), rows.end(), [&](const Json* a, const Json* b) { return edf(a) < edf(b); });

    // GPUs of a video suspended inside a solver step. Its state is resident on
    // exactly that group, so taking one of them blocks the whole group.
    std::vector<const Json*> mid_owner(gpus, nullptr);
    // A chunked decode in progress holds only its decoding GPU.
    for (const Json* r : rows) {
        if (r->at("kind") != "video") continue;
        if (int_field(*r, "decode_done", 0, 0) > 0) mid_owner[group_of(*r).front()] = r;
        else if (int_field(*r, "blocks_done", 0, 0) > 0) for (int gpu : group_of(*r)) mid_owner[gpu] = r;
    }
    auto time_to_boundary = [&](const Json* r) {
        const auto group = group_of(*r);
        const auto klass = r->at("class").get<std::string>();
        if (int_field(*r, "decode_done", 0, 0) > 0)
            return (cost.decode_chunks(klass) - int_field(*r, "decode_done", 0, 0)) * cost.chunk_ms(klass);
        return (cost.blocks_per_step(klass) - int_field(*r, "blocks_done", 0, 0)) * cost.block_ms(klass, static_cast<int>(group.size()));
    };
    auto all_free = [&](const std::vector<int>& group) {
        return std::all_of(group.begin(), group.end(), [&](int gpu) { return free[gpu]; });
    };
    auto take = [&](const std::vector<int>& group) { for (int gpu : group) free[gpu] = false; };

    Json assignments = Json::array();
    for (std::size_t index = 0; index < rows.size(); ++index) {
        const Json& r = *rows[index];
        const auto id = r.at("request_id").get<std::string>();
        const auto klass = r.at("class").get<std::string>();
        if (r.at("kind") == "image") {
            const double slack = number_field(r, "deadline_ms") - now - cost.remaining_ms(r, 1);
            int chosen = -1, rank = 3;  // -1 reserved, 0 idle, 1 step-boundary video group, 2 mid-step group
            for (int gpu = 0; gpu < gpus; ++gpu) {
                if (!free[gpu]) continue;
                // Prefer idle GPUs; then GPUs of videos at step boundaries; a
                // mid-step group only when waiting for its boundary would miss.
                int score = reserved(gpu) ? -1 : mid_owner[gpu] ? 2 : 0;
                if (!score) for (const Json* other : rows) if (other->at("kind") == "video") {
                    const auto group = group_of(*other);
                    if (std::find(group.begin(), group.end(), gpu) != group.end()) score = 1;
                }
                if (score == 2 && policy.blocks && lazy && slack >= time_to_boundary(mid_owner[gpu])) continue;
                if (score < rank) { rank = score; chosen = gpu; }
            }
            if (chosen < 0) continue;
            free[chosen] = false;
            assignments.push_back({{"request_id", id}, {"group", Json::array({chosen})}, {"degree", 1}, {"quantum", "atomic"},
                                   {"resize", false}, {"reason", rank == 2 ? "block_preempt" : "edf"}});
            continue;
        }
        const auto current = group_of(r);
        const bool prepared = r.value("prepared", false);
        const int steps = int_field(r, "steps_total", int_field(cost.klass(klass), "steps", 0), 1);
        if (prepared && int_field(r, "steps_done", 0, 0) >= steps) {
            if (current.empty()) throw std::invalid_argument("finished request has no resident group");
            if (free[current.front()]) {
                free[current.front()] = false;
                Json assignment = {{"request_id", id}, {"group", Json::array({current.front()})}, {"degree", 1}, {"quantum", "finish"},
                                   {"resize", false}, {"reason", "decode"}};
                // Chunked decode: frames per quantum follow quantum_ms, else decode_chunk_frames.
                const int chunks = cost.decode_chunks(klass), done = int_field(r, "decode_done", 0, 0);
                int frames = int_field(options, "decode_chunk_frames", 0, 0);
                if (chunks > 1 && quantum_ms > 0 && bool_field(options, "chunked_decode", true))
                    frames = std::max(1, static_cast<int>(std::floor(quantum_ms / cost.chunk_ms(klass))));
                if (chunks > 1 && frames > 0) assignment["decode_frames"] = std::min(frames, chunks - done);
                assignments.push_back(assignment);
            }
            continue;
        }
        const bool fixed = prepared && (int_field(r, "blocks_done", 0, 0) > 0 || !policy.elastic || pin_started);
        if (fixed) {
            if (all_free(current)) {
                take(current);
                assignments.push_back({{"request_id", id}, {"group", current}, {"degree", current.size()},
                                       {"quantum", video_quantum(klass, static_cast<int>(current.size()))},
                                       {"resize", false}, {"reason", "resume"}});
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
        if (order.empty()) throw std::invalid_argument("request class needs more GPUs than the cluster offers: " + klass);
        int target = order.back();
        if (others_waiting) {
            for (int d : order) {
                const double extra = prepared && static_cast<int>(current.size()) != d ? cost.resize_ms() : 0;
                if (now + cost.remaining_ms(r, d) + extra <= number_field(r, "deadline_ms")) { target = d; break; }
            }
        }
        // A prompt encoded on one worker must be prepared on that GPU.
        const int affinity = !prepared && r.contains("affinity") ? static_cast<int>(integer(r.at("affinity"), "affinity")) : -1;
        std::vector<int> chosen;
        for (auto d = order.rbegin(); d != order.rend() && chosen.empty(); ++d) {
            if (*d > target) continue;
            if (prepared && static_cast<int>(current.size()) == *d && all_free(current)) { chosen = current; break; }
            int best_busy = -1;
            for (const auto& group : groups) {
                if (static_cast<int>(group.size()) != *d || !all_free(group)) continue;
                if (std::any_of(group.begin(), group.end(), reserved)) continue;
                if (affinity >= 0 && std::find(group.begin(), group.end(), affinity) == group.end()) continue;
                if (std::any_of(group.begin(), group.end(), [&](int gpu) { return mid_owner[gpu] && mid_owner[gpu] != &r; })) continue;
                // Best fit: prefer the group whose aligned parent is most occupied.
                const int parent = std::min(gpus, *d * 2), start = group.front() / parent * parent;
                int occupied = 0;
                for (int gpu = start; gpu < std::min(gpus, start + parent); ++gpu) occupied += !free[gpu];
                if (occupied > best_busy) { best_busy = occupied; chosen = group; }
            }
        }
        if (chosen.empty()) continue;
        take(chosen);
        assignments.push_back({{"request_id", id}, {"group", chosen}, {"degree", chosen.size()},
                               {"quantum", video_quantum(klass, static_cast<int>(chosen.size()))},
                               {"resize", prepared && chosen != current}, {"reason", prepared ? "regroup" : "admit"}});
    }
    return {{"assignments", assignments}, {"policy", name}};
}
}
