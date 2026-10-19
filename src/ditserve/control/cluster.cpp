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
    int fixed_degree = 0;    // static_<form>: GPUs per video
    Form fixed_form;         // and how they run it
    bool elastic = false;    // regroup at step boundaries
    bool blocks = false;     // advance in block quanta, lazy intra-step preemption
    bool rounds = false;     // regroup only every options.round_steps steps (TetriServe-like)
};
Policy parse(const std::string& name) {
    Policy policy;
    if (name.rfind("static_", 0) == 0) {
        try { policy.fixed_form = parse_form(name.substr(7)); }
        catch (const std::invalid_argument&) { throw std::invalid_argument("unknown cluster policy: " + name); }
        policy.fixed_degree = policy.fixed_form.degree();
    } else if (name == "step_elastic") policy.elastic = true;
    else if (name == "block_elastic") policy.elastic = policy.blocks = true;
    else if (name == "round_elastic") policy.elastic = policy.rounds = true;
    else throw std::invalid_argument("unknown cluster policy: " + name);
    return policy;
}
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
    if (rest.size() < 3 || rest.rfind("sp", 0) != 0 || rest.size() > 3 || rest[2] < '1' || rest[2] > '8')
        throw std::invalid_argument("unknown parallel form: " + name);
    form.sp = rest[2] - '0';
    if (!power_of_two(form.sp)) throw std::invalid_argument("unknown parallel form: " + name);
    return form;
}

std::string form_name(const Form& form) {
    if (form.cfg != 1 && form.cfg != 2) throw std::invalid_argument("cfg must be 1 or 2");
    if (!power_of_two(form.sp)) throw std::invalid_argument("SP degree must be a power of two");
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
        number_field(row, "cfg_exchange_mb");
    }
    efficiency_ = number_field(profile, "sp_efficiency", 1, true);
    if (efficiency_ > 1) throw std::invalid_argument("sp_efficiency must be in (0, 1]");
    resize_ms_ = number_field(profile, "resize_ms");
    overhead_ms_ = number_field(profile, "quantum_overhead_ms");
    overlap_ = number_field(profile, "sp_overlap");
    if (overlap_ > 1) throw std::invalid_argument("sp_overlap must be in [0, 1]");
    const auto links = profile.value("links", Json::object());
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
        if (!link.is_object() || !link.contains("concurrent_gbps")) continue;
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

const Json& ClusterCost::klass(const std::string& name) const {
    const auto& classes = profile_.at("classes");
    if (!classes.contains(name)) throw std::invalid_argument("unknown cost class: " + name);
    return classes.at(name);
}

double ClusterCost::step_ms(const std::string& name, const Form& form, int sp_groups, const std::vector<int>* group) const {
    const auto& row = klass(name);
    if (row.at("kind") != "video") throw std::invalid_argument("image classes have no solver steps");
    form_name(form);   // validates
    const auto& links = profile_.value("links", Json::object());
    auto link_of = [&](int sp) -> const Json& {
        if (!links.contains(degree_key(sp))) throw std::invalid_argument("cost profile has no link model for SP degree " + degree_key(sp));
        return links.at(degree_key(sp));
    };
    const double cfg = form.cfg, sp = form.sp;
    double step = number_field(row, "step_compute_ms", 0, true) / (cfg * sp * efficiency_);
    if (form.sp > 1) {
        const auto& link = link_of(form.sp);
        double gbps = number_field(link, "gbps", 0, true), latency = number_field(link, "latency_ms");
        double overlap = link.contains("sp_overlap") ? number_field(link, "sp_overlap") : overlap_;
        if (overlap > 1) throw std::invalid_argument("sp_overlap must be in [0, 1]");
        // On a named placement the form runs at its slowest SP subgroup's rate (the halves of a
        // CFG pair each run SP on their own GPUs).
        if (group && links.contains("groups")) {
            if (static_cast<int>(group->size()) != form.degree()) throw std::invalid_argument("group size does not match the form");
            bool first = true;
            for (int half = 0; half < form.cfg; ++half) {
                const auto key = std::to_string((*group)[half * form.sp]) + "-" + std::to_string((*group)[(half + 1) * form.sp - 1]);
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
        const double comm = number_field(row, "comm_mb_per_step") / cfg * (sp - 1) / (sp * sp) / gbps;
        // Branch interleaving hides part of the shorter term behind the other branch;
        // with the branches on separate halves (cfg = 2) each rank has one branch.
        // The hidden fraction falls as communication approaches compute, so a link may name its own.
        const double hidden = form.cfg == 1 ? overlap * std::min(step, comm) : 0;
        step += comm - hidden + number_field(row, "collectives_per_step") / cfg * latency;
    }
    if (form.cfg == 2) {
        const auto& pair = link_of(2);
        step += number_field(row, "cfg_exchange_mb") / number_field(pair, "gbps", 0, true) + number_field(pair, "latency_ms");
    }
    return step;
}

int ClusterCost::blocks_per_step(const std::string& name) const { return int_field(klass(name), "blocks_per_step", 1); }

int ClusterCost::decode_chunks(const std::string& name) const { return int_field(klass(name), "decode_chunks", 1); }

double ClusterCost::chunk_ms(const std::string& name) const { return number_field(klass(name), "finish_ms") / decode_chunks(name); }

double ClusterCost::block_ms(const std::string& name, const Form& form, int sp_groups, const std::vector<int>* group) const {
    return step_ms(name, form, sp_groups, group) / blocks_per_step(name);
}

double ClusterCost::remaining_ms(const Json& request, const Form& form, int sp_groups, const std::vector<int>* group) const {
    const auto name = request.at("class").get<std::string>();
    const auto& row = klass(name);
    if (row.at("kind") == "image") return number_field(row, "service_ms");
    const int steps = int_field(request, "steps_total", int_field(row, "steps", 0), 1), done = int_field(request, "steps_done", 0, 0);
    const int blocks = int_field(request, "blocks_done", 0, 0);
    double total = request.value("prepared", false) ? 0 : number_field(row, "prepare_ms");
    total += std::max(0, steps - done) * step_ms(name, form, sp_groups, group) - blocks * block_ms(name, form, sp_groups, group);
    return total + (decode_chunks(name) - std::min(int_field(request, "decode_done", 0, 0), decode_chunks(name))) * chunk_ms(name);
}

Json cluster_plan(const Json& state, const std::string& name, const ClusterCost& cost, const Json& options) {
    const Policy policy = parse(name);
    const double now = number(state.at("now_ms"), "now_ms");
    const int gpus = static_cast<int>(integer(state.at("gpus"), "gpus"));
    // Parallel forms the policy may choose; the group sizes follow from them.
    std::vector<Form> forms;
    if (policy.fixed_degree) forms = {policy.fixed_form};
    else if (options.contains("forms")) {
        if (options.contains("degrees")) throw std::invalid_argument("options.forms replaces options.degrees");
        for (const auto& item : options.at("forms")) {
            if (!item.is_string()) throw std::invalid_argument("options.forms lists form names");
            forms.push_back(parse_form(item.get<std::string>()));
        }
    } else {
        for (const auto& d : options.value("degrees", Json::array({1, 2, 4, 8}))) forms.push_back(Form{1, static_cast<int>(integer(d, "degree"))});
    }
    if (forms.empty()) throw std::invalid_argument("cluster policy needs at least one parallel form");
    std::vector<int> degrees;
    for (const auto& form : forms) if (std::find(degrees.begin(), degrees.end(), form.degree()) == degrees.end()) degrees.push_back(form.degree());
    if (cluster_groups(gpus, degrees).empty()) throw std::invalid_argument("no SP group fits the cluster");
    // Every aligned group stays addressable: a class minimum can exceed the policy's degrees.
    const auto groups = cluster_groups(gpus, {1, 2, 4, 8});
    const int quantum_blocks = int_field(options, "quantum_blocks", 4);
    // Optional time target: blocks per quantum follow the class and parallel form,
    // so a preemption check happens about every quantum_ms of GPU work.
    const double quantum_ms = number_field(options, "quantum_ms", 0);
    auto video_quantum = [&](const std::string& klass, const Form& form, int n, const std::vector<int>* group = nullptr) -> Json {
        if (!policy.blocks) return "step";
        if (quantum_ms <= 0) return quantum_blocks;
        const int blocks = static_cast<int>(std::floor(quantum_ms / cost.block_ms(klass, form, n, group)));
        return std::max(1, std::min(blocks, cost.blocks_per_step(klass)));
    };
    // SP advances already running plus those this plan hands out: with
    // concurrent link rates in the profile they slow each other down.
    const int sp_running = int_field(state, "sp_groups_running", 0, 0);
    int sp_assigned = 0;
    auto sp_groups = [&](const Form& form) { return form.sp >= 2 ? sp_running + sp_assigned + 1 : 1; };
    // The cheapest listed form of a group size, on a named group when given. A size
    // no form names (a class minimum above the policy's sizes) runs plain SP.
    auto form_for = [&](const std::string& klass, int degree, const std::vector<int>* group = nullptr) {
        Form best{1, degree};
        double best_ms = 0;
        bool found = false;
        for (const auto& form : forms) {
            if (form.degree() != degree) continue;
            const double ms = cost.step_ms(klass, form, sp_groups(form), group);
            if (!found || ms < best_ms) { best = form; best_ms = ms; found = true; }
        }
        return best;
    };
    auto ceil_div = [](int a, int b) { return (a + b - 1) / b; };
    // Latent-frame chunks per decode quantum (0: the decode is one quantum).
    auto decode_frames = [&](const std::string& klass) {
        int frames = int_field(options, "decode_chunk_frames", 0, 0);
        if (cost.decode_chunks(klass) > 1 && quantum_ms > 0 && bool_field(options, "chunked_decode", true))
            frames = std::max(1, static_cast<int>(std::floor(quantum_ms / cost.chunk_ms(klass))));
        return cost.decode_chunks(klass) > 1 ? frames : 0;
    };
    // Quanta a video still needs on `form`, from its current progress.
    auto quanta_left = [&](const Json& r, const Form& form, int n, const std::vector<int>* group = nullptr) {
        const auto klass = r.at("class").get<std::string>();
        const int steps = int_field(r, "steps_total", int_field(cost.klass(klass), "steps", 0), 1);
        const int left = std::max(0, steps - int_field(r, "steps_done", 0, 0)), done = int_field(r, "blocks_done", 0, 0);
        int quanta = left;
        const auto quantum = video_quantum(klass, form, n, group);
        if (quantum.is_number_integer() && left > 0) {
            const int per = quantum.get<int>(), bps = cost.blocks_per_step(klass);
            quanta = ceil_div(bps - done, per) + (left - 1) * ceil_div(bps, per);
        }
        const int chunks = cost.decode_chunks(klass) - std::min(int_field(r, "decode_done", 0, 0), cost.decode_chunks(klass));
        const int frames = decode_frames(klass);
        return quanta + (chunks <= 0 ? 0 : frames > 0 ? ceil_div(chunks, frames) : 1);
    };
    // Remaining GPU time plus the fixed dispatch cost of every quantum still to come.
    auto estimate = [&](const Json& r, const Form& form, int n, const std::vector<int>* group = nullptr) {
        const double remaining = cost.remaining_ms(r, form, n, group);
        if (cost.quantum_overhead_ms() <= 0) return remaining;
        return remaining + cost.quantum_overhead_ms() * (r.at("kind") == "image" ? 1 : quanta_left(r, form, n, group));
    };
    const bool lazy = bool_field(options, "lazy_preemption", true);
    // Minimum per-GPU efficiency (SP1 step time / (GPUs x form step time)) at which a video
    // widens beyond its minimal feasible group while others wait; 0 keeps the minimal group.
    const double widen = number_field(options, "widen_efficiency");
    if (widen > 1) throw std::invalid_argument("widen_efficiency must be in [0, 1]");
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
    // A chunked decode in progress holds only its decoding GPU.
    for (const Json* r : rows) {
        if (r->at("kind") != "video") continue;
        if (int_field(*r, "decode_done", 0, 0) > 0) mid_owner[group_of(*r).front()] = r;
        else if (int_field(*r, "blocks_done", 0, 0) > 0) for (int gpu : group_of(*r)) mid_owner[gpu] = step_owner[gpu] = r;
    }
    // A group counts as free only if a new placement could take it: not reserved
    // for images and not held by a video inside a step or decode.
    int free_widest = 1;
    for (int d : degrees)
        for (const auto& group : cluster_groups(gpus, {d}))
            if (d > free_widest && std::all_of(group.begin(), group.end(), [&](int gpu) { return free[gpu] && !reserved(gpu) && !mid_owner[gpu]; }))
                free_widest = d;
    // Feasible-first EDF: a request that cannot meet its deadline even on the
    // largest group yields to those that still can, so an overloaded queue of
    // late videos does not starve newly arrived short requests.
    const bool demote_late = bool_field(options, "demote_late", true);
    // Late requests shorter than this keep their EDF position: they barely
    // delay feasible work, and demoting them behind a long video only turns a
    // small miss into a very long tail.
    const double late_short_ms = number_field(options, "late_short_ms", 2000);
    // Queue order among feasible requests: deadline (edf, the default), estimated remaining
    // single-GPU work (srtf) or arrival (fcfs); the lateness demotion applies to all three.
    const auto order_rule = options.value("order", std::string("edf"));
    if (order_rule != "edf" && order_rule != "srtf" && order_rule != "fcfs") throw std::invalid_argument("order must be edf, srtf or fcfs");
    const int widest = policy.fixed_degree ? policy.fixed_degree
                       : feasibility == "free" ? free_widest : *std::max_element(degrees.begin(), degrees.end());
    std::map<const Json*, bool> late;
    std::map<const Json*, double> rank_key;
    for (const Json* r : rows) {
        const auto klass = r->at("class").get<std::string>();
        const int best = r->at("kind") == "image" ? 1 : std::min(std::max(widest, int_field(cost.klass(klass), "min_degree", 1)), gpus);
        int degree = best;
        while (degree > 1 && !power_of_two(degree)) --degree;
        const auto group = group_of(*r);
        const bool mid_step = r->value("prepared", false) && int_field(*r, "blocks_done", 0, 0) > 0;
        const Form form = r->at("kind") == "image" ? Form{} : mid_step ? form_of(*r, group) : form_for(klass, degree);
        const double remaining = estimate(*r, form, sp_groups(form), mid_step ? &group : nullptr);
        late[r] = demote_late && remaining > late_short_ms && now + remaining > number_field(*r, "deadline_ms");
        rank_key[r] = order_rule == "edf" ? number_field(*r, "deadline_ms")
                    : order_rule == "srtf" ? estimate(*r, Form{}, 1) : number_field(*r, "arrival_ms");
    }
    auto edf = [&](const Json* r) {
        return std::make_tuple(late.at(r), rank_key.at(r), number_field(*r, "arrival_ms"), r->at("request_id").get<std::string>());
    };
    std::sort(rows.begin(), rows.end(), [&](const Json* a, const Json* b) { return edf(a) < edf(b); });
    // Round-based regrouping: a placed video keeps its group until it has completed a multiple of
    // round_steps steps, like TetriServe's rounds; other elastic policies regroup at every step.
    const int round_steps = int_field(options, "round_steps", 5);

    auto time_to_boundary = [&](const Json* r) {
        const auto group = group_of(*r);
        const auto klass = r->at("class").get<std::string>();
        const double overhead = cost.quantum_overhead_ms();
        if (int_field(*r, "decode_done", 0, 0) > 0) {
            const int chunks = cost.decode_chunks(klass) - int_field(*r, "decode_done", 0, 0), frames = decode_frames(klass);
            return chunks * cost.chunk_ms(klass) + (overhead > 0 ? overhead * (frames > 0 ? ceil_div(chunks, frames) : 1) : 0.);
        }
        const Form form = form_of(*r, group);
        const int blocks = cost.blocks_per_step(klass) - int_field(*r, "blocks_done", 0, 0);
        const double work = blocks * cost.block_ms(klass, form, sp_groups(form), &group);
        if (overhead <= 0) return work;
        const auto quantum = video_quantum(klass, form, sp_groups(form), &group);
        return work + overhead * (quantum.is_number_integer() ? ceil_div(blocks, quantum.get<int>()) : 1);
    };
    auto all_free = [&](const std::vector<int>& group) {
        return std::all_of(group.begin(), group.end(), [&](int gpu) { return free[gpu]; });
    };
    auto take = [&](const std::vector<int>& group) { for (int gpu : group) free[gpu] = false; };
    // Another video suspended inside a step holds these GPUs even between its quanta.
    auto held_by_other = [&](const std::vector<int>& group, const Json* r) {
        return std::any_of(group.begin(), group.end(), [&](int gpu) { return mid_owner[gpu] && mid_owner[gpu] != r; });
    };

    Json assignments = Json::array();
    for (std::size_t index = 0; index < rows.size(); ++index) {
        const Json& r = *rows[index];
        const auto id = r.at("request_id").get<std::string>();
        const auto klass = r.at("class").get<std::string>();
        if (r.at("kind") == "image") {
            const double slack = number_field(r, "deadline_ms") - now - estimate(r, Form{}, 1);
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
                const int frames = decode_frames(klass);
                if (frames > 0) assignment["decode_frames"] = std::min(frames, cost.decode_chunks(klass) - int_field(r, "decode_done", 0, 0));
                assignments.push_back(assignment);
            }
            continue;
        }
        const bool fixed = prepared && (int_field(r, "blocks_done", 0, 0) > 0 || !policy.elastic || pin_started
                                        || (policy.rounds && int_field(r, "steps_done", 0, 0) % round_steps != 0));
        if (fixed) {
            if (all_free(current)) {
                take(current);
                const Form form = form_of(r, current);
                assignments.push_back({{"request_id", id}, {"group", current}, {"degree", form.degree()}, {"form", form_name(form)},
                                       {"quantum", video_quantum(klass, form, sp_groups(form), &current)},
                                       {"resize", false}, {"reason", "resume"}});
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
        if (order.empty()) throw std::invalid_argument("request class needs more GPUs than the cluster offers: " + klass);
        int target = order.back();
        if (others_waiting) {
            for (int d : order) {
                const double extra = prepared && static_cast<int>(current.size()) != d ? cost.resize_ms() : 0;
                const Form form = form_for(klass, d);
                if (now + estimate(r, form, sp_groups(form)) + extra <= number_field(r, "deadline_ms")) { target = d; break; }
            }
            // Work-neutral widening: a wider form that keeps at least widen_efficiency of the
            // single-GPU throughput (CFG halves do; PCIe sequence parallelism does not) finishes
            // the video sooner and returns its GPUs sooner, so it costs the queue nothing.
            if (widen > 0) {
                const double single = cost.step_ms(klass, Form{}, 1);
                for (int d : order) {
                    const Form form = form_for(klass, d);
                    if (d > target && single / (d * cost.step_ms(klass, form, sp_groups(form))) >= widen) target = d;
                }
            }
        }
        // A prompt encoded on one worker must be prepared on that GPU.
        const int affinity = !prepared && r.contains("affinity") ? static_cast<int>(integer(r.at("affinity"), "affinity")) : -1;
        std::vector<int> chosen;
        for (auto d = order.rbegin(); d != order.rend() && chosen.empty(); ++d) {
            if (*d > target) continue;
            if (prepared && static_cast<int>(current.size()) == *d && all_free(current) && !held_by_other(current, &r)) { chosen = current; break; }
            int best_busy = -1;
            double best_ms = 0;
            for (const auto& group : groups) {
                if (static_cast<int>(group.size()) != *d || !all_free(group)) continue;
                if (std::any_of(group.begin(), group.end(), reserved)) continue;
                if (affinity >= 0 && std::find(group.begin(), group.end(), affinity) == group.end()) continue;
                if (held_by_other(group, &r)) continue;
                // The fastest group of this size (per-group link rates), then best fit:
                // the group whose aligned parent is most occupied.
                const Form form = form_for(klass, *d, &group);
                const double ms = *d > 1 ? cost.step_ms(klass, form, sp_groups(form), &group) : 0;
                const int parent = std::min(gpus, *d * 2), start = group.front() / parent * parent;
                int occupied = 0;
                for (int gpu = start; gpu < std::min(gpus, start + parent); ++gpu) occupied += !free[gpu];
                if (chosen.empty() || ms < best_ms - 1e-9 || (std::abs(ms - best_ms) <= 1e-9 && occupied > best_busy)) {
                    best_busy = occupied; best_ms = ms; chosen = group;
                }
            }
        }
        if (chosen.empty()) continue;
        take(chosen);
        const Form form = form_for(klass, static_cast<int>(chosen.size()), &chosen);
        assignments.push_back({{"request_id", id}, {"group", chosen}, {"degree", form.degree()}, {"form", form_name(form)},
                               {"quantum", video_quantum(klass, form, sp_groups(form), &chosen)},
                               {"resize", prepared && chosen != current}, {"reason", prepared ? "regroup" : "admit"}});
        sp_assigned += form.sp >= 2;
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
            for (const auto& gpu : assignment.at("group")) {
                const Json* owner = step_owner[gpu.get<int>()];
                if (owner && owner != by_id.at(id))
                    throw std::logic_error("cluster plan advances " + id + " on GPU " + std::to_string(gpu.get<int>()) + " while " +
                                           owner->at("request_id").get<std::string>() + " is inside a solver step there");
            }
        }
    }
    return {{"assignments", assignments}, {"policy", name}};
}
}
