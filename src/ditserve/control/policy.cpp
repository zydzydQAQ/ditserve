#include "ditserve/control/policy.hpp"
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>

namespace ditserve {
namespace {
PolicyTraits row(const std::string& id, const std::string& display, const std::string& family, const std::string& source) {
    PolicyTraits t;
    t.id = id; t.display = display; t.family = family; t.source = source;
    return t;
}

// DitServe: block-level exact preemption, CFG-aware form choice (dominated forms
// pruned, no video narrower than a CFG pair) and pair-preserving image placement.
PolicyTraits ditserve_row(const std::string& id, const std::string& display, const std::string& family, const std::string& source) {
    PolicyTraits t = row(id, display, family, source);
    t.elastic = t.blocks = t.prune_dominated = true;
    t.min_video_degree = 2;
    t.widen = 0.90;
    t.image_placement = "pair_fit";
    return t;
}

// The first DitServe planner: hand-weighted knapsack utility (kept as an ablation).
PolicyTraits v1_row(const std::string& id, const std::string& display, const std::string& family, const std::string& source) {
    PolicyTraits t = row(id, display, family, source);
    t.elastic = t.knapsack = true;
    t.widen = 0.90;
    t.image_placement = "pair_fit";
    return t;
}

const std::map<std::string, PolicyTraits>& named() {
    static const std::map<std::string, PolicyTraits> table = [] {
        std::map<std::string, PolicyTraits> m;
        auto add = [&](PolicyTraits t) { m.emplace(t.id, std::move(t)); };

        // ---- ours ----
        add(ditserve_row("ditserve", "DitServe", "ours",
            "this work: block-level exact preemption + CFG-aware form pruning + minimal CFG-pair admission"));
        {
            auto t = ditserve_row("ditserve-approx", "DitServe-Approx", "ours",
                "this work, approximate mode: DitServe plus worker-driven late-step CFG folding");
            t.v5 = t.fission = true;
            add(t);
        }

        // ---- baselines (re-implemented on the same executor, workers and profile) ----
        {
            auto t = row("tetriserve", "TetriServe (re-impl.)", "baseline",
                "TetriServe, ASPLOS'26: step-level SP, round-based scheduling with mid-round GPU hold, "
                "dominated-degree pruning, survival-aware deadline group knapsack, and 1-GPU best-effort late fallback");
            t.elastic = t.rounds = t.deadline_dp = t.prune_dominated = true;
            add(t);
        }
        {
            auto t = row("sbp-esp", "SBP-ESP", "baseline",
                "method of GenServe (arXiv:2604.04335, preprint): step-boundary video preemption, elastic SP "
                "re-planned every step by a joint cross-modal SLO-aware lexicographic DP with slack utility, "
                "dynamic image batching, single-GPU VAE decode");
            t.elastic = t.deadline_dp = t.image_batching = true;
            add(t);
        }
        {
            auto t = row("edf-bestfit", "EDF-BF", "baseline",
                "method of GF-DiT (arXiv:2606.13501, preprint): strict EDF EdfBestFit policy, smallest SP meeting "
                "the deadline else fastest fitting degree, then cost-aware head-of-line EDF surplus promotion");
            t.elastic = t.best_fit = true;
            t.demote_late = false;
            add(t);
        }

        // ---- heuristics ----
        {
            auto t = row("edf-cfg2", "EDF-CFG2", "heuristic",
                "every video on one CFG pair, EDF queue, images take GPUs at video step boundaries");
            t.fixed_form = parse_form("cfg2"); t.fixed_degree = 2;
            add(t);
        }
        {
            auto t = row("cfg2-first", "CFG2-First", "heuristic",
                "elastic at step boundaries over non-dominated forms, never narrower than a CFG pair, widen only "
                "when per-GPU efficiency stays >= 0.9; no sub-step preemption");
            t.elastic = t.prune_dominated = true;
            t.min_video_degree = 2; t.widen = 0.90; t.image_placement = "pair_fit";
            add(t);
        }
        {
            auto t = row("partition", "Static-Partition", "heuristic",
                "videos on fixed CFG pairs, a fixed GPU share serves images only, EDF queues");
            t.fixed_form = parse_form("cfg2"); t.fixed_degree = 2; t.partition_divisor = 4;
            add(t);
        }

        // ---- ablations of DitServe ----
        {
            auto t = ditserve_row("abl-no-block", "DitServe w/o block preemption", "ablation", "DitServe with whole-step quanta");
            t.blocks = false;
            add(t);
        }
        {
            auto t = ditserve_row("abl-no-prune", "DitServe w/o form pruning", "ablation", "DitServe without dominated-form pruning");
            t.prune_dominated = false;
            add(t);
        }
        {
            auto t = ditserve_row("abl-no-mindeg", "DitServe w/o CFG-pair admission", "ablation",
                "DitServe admitting videos on any free group size");
            t.min_video_degree = 1;
            add(t);
        }
        {
            auto t = ditserve_row("abl-first-fit", "DitServe w/o pair-fit placement", "ablation",
                "DitServe placing images on the lowest free GPU");
            t.image_placement = "first";
            add(t);
        }
        {
            auto t = v1_row("ditserve-v1", "DitServe-v1 (knapsack)", "ablation",
                "earlier DitServe planner: block quanta + hand-weighted knapsack utility");
            t.blocks = true; t.tiered_memory = true; t.image_batching = true;
            add(t);
        }
        {
            auto t = v1_row("ditserve-v1-approx", "DitServe-v1-Approx", "ablation",
                "earlier approximate planner: knapsack + late-step CFG folding");
            t.blocks = t.v5 = t.fission = true;
            add(t);
        }
        {
            auto t = row("abl-step-elastic", "Step-Elastic", "ablation", "elastic SP regrouped at every step boundary");
            t.elastic = true;
            add(t);
        }
        {
            auto t = row("abl-block-elastic", "Block-Elastic", "ablation", "step-elastic plus block quanta and lazy preemption");
            t.elastic = t.blocks = true;
            add(t);
        }
        {
            auto t = row("abl-round-elastic", "Round-Elastic", "ablation", "elastic SP regrouped every options.round_steps steps");
            t.elastic = t.rounds = true;
            add(t);
        }

        // ---- diagnostics (outside the paper tables) ----
        {
            auto t = v1_row("diag-feature-cache", "Feature-Cache (diag.)", "diagnostic", "whole-step dynamic feature cache, SP1 pinned");
            t.approx = t.tiered_memory = t.image_batching = true;
            add(t);
        }
        {
            auto t = v1_row("diag-v3", "HeteroDiT-v3 (diag.)", "diagnostic", "INT8 Ring + Taylor + CFG reuse research path");
            t.approx = t.v3 = t.tiered_memory = t.image_batching = true;
            add(t);
        }
        {
            auto t = row("diag-tiered", "Tiered-State (diag.)", "diagnostic", "block-elastic with tiered host state");
            t.elastic = t.blocks = t.tiered_memory = true;
            add(t);
        }
        {
            auto t = row("diag-2d-usp", "2D-USP (diag.)", "diagnostic", "block-elastic over Ulysses x Ring forms");
            t.elastic = t.blocks = t.tiered_memory = true;
            add(t);
        }
        {
            auto t = row("diag-elastic-partition", "Elastic-Partition (diag.)", "diagnostic", "step-elastic videos with reserved image GPUs");
            t.elastic = true; t.partition_divisor = 4;
            add(t);
        }
        return m;
    }();
    return table;
}

const std::map<std::string, std::string>& aliases() {
    static const std::map<std::string, std::string> table = {
        {"heterodit_exact", "ditserve-v1"}, {"knapsack_dp", "ditserve-v1"}, {"heterodit_v5", "ditserve-v1-approx"},
        {"step_elastic", "abl-step-elastic"}, {"block_elastic", "abl-block-elastic"}, {"round_elastic", "abl-round-elastic"},
        {"heterodit_approx", "diag-feature-cache"}, {"heterodit_v3", "diag-v3"},
        {"ablation_tiered", "diag-tiered"}, {"ablation_2d_usp", "diag-2d-usp"}, {"static_partition", "diag-elastic-partition"},
    };
    return table;
}

// Names of earlier approximations that borrowed external system names.
const std::set<std::string>& retired() {
    static const std::set<std::string> names = {"vanilla_xdit", "tridentserve_stage", "vllm_omni", "gf_dit", "genserve_40gb"};
    return names;
}

// Fixed-form execution controls and explicit approximate resource arms.
bool parameterised(const std::string& name, std::string& kind, Form& form) {
    for (const char* prefix : {"static-", "block-", "xdit-", "fold-", "approx-", "static_", "block_"}) {
        const std::string p(prefix);
        if (name.rfind(p, 0) != 0) continue;
        try { form = parse_form(name.substr(p.size())); }
        catch (const std::invalid_argument&) { throw std::invalid_argument("unknown cluster policy: " + name); }
        kind = p.substr(0, p.size() - 1);
        return true;
    }
    return false;
}
}

std::string legacy_policy_alias(const std::string& name) {
    if (aliases().count(name)) return aliases().at(name);
    std::string kind; Form form;
    if (name.find('_') != std::string::npos && name.find('-') == std::string::npos && parameterised(name, kind, form))
        return kind + "-" + form_name(form);
    return "";
}

std::string canonical_policy(const std::string& name) {
    if (name == "xdit") return "xdit-sp1";
    const auto legacy = legacy_policy_alias(name);
    return legacy.empty() ? name : legacy;
}

const PolicyTraits& policy_traits(const std::string& raw) {
    if (retired().count(raw))
        throw std::invalid_argument("retired cluster policy " + raw + ": it borrowed an external system name; see policy_catalog()");
    const std::string name = canonical_policy(raw);
    if (named().count(name)) return named().at(name);
    std::string kind; Form form;
    if (!parameterised(name, kind, form)) throw std::invalid_argument("unknown cluster policy: " + raw);
    static std::mutex lock;
    static std::map<std::string, PolicyTraits> cache;
    std::lock_guard<std::mutex> guard(lock);
    auto it = cache.find(name);
    if (it != cache.end()) return it->second;
    const std::string form_text = form_name(form);
    PolicyTraits t;
    t.fixed_form = form; t.fixed_degree = form.degree();
    if (kind == "static") {
        t = row(name, "Static-" + form_text, "control", "fixed parallel form, EDF, whole-step quanta");
        t.fixed_form = form; t.fixed_degree = form.degree();
    } else if (kind == "block") {
        t = row(name, "Block-" + form_text, "control", "fixed parallel form with block quanta and sub-step preemption");
        t.fixed_form = form; t.fixed_degree = form.degree(); t.blocks = true;
    } else if (kind == "fold" || kind == "approx") {
        t = row(name, (kind == "fold" ? "Fold-" : "Approx-") + form_text, "ablation",
            kind == "fold" ? "fixed-form CFG folding with the unconditional subgroup reserved but idle"
                           : "fixed initial CFG form with folding and subgroup fission");
        t.fixed_form = form; t.fixed_degree = form.degree(); t.blocks = t.v5 = true; t.fission = kind == "approx";
    } else {
        t = row(name, form_text == "sp1" ? "xDiT" : "xDiT-" + form_text, "baseline",
            "xDiT engine (arXiv:2411.01738) static-partition baseline: each model on its own GPUs, "
            "unoptimized single-GPU execution without CFG or sequence parallelism by default (cfg=1, sp1), "
            "FCFS, a started request runs to completion without preemption");
        t.fixed_form = form; t.fixed_degree = form.degree();
        t.hold_diffuse = true; t.order = "fcfs"; t.demote_late = false; t.partition_divisor = 4;
    }
    return cache.emplace(name, t).first->second;
}

Json policy_catalog() {
    Json out = Json::array();
    auto describe = [](const PolicyTraits& t) {
        return Json{{"id", t.id}, {"display", t.display}, {"family", t.family}, {"source", t.source},
                    {"fixed_form", t.fixed_degree ? form_name(t.fixed_form) : ""},
                    {"elastic", t.elastic}, {"blocks", t.blocks}, {"rounds", t.rounds}, {"knapsack", t.knapsack},
                    {"deadline_dp", t.deadline_dp}, {"best_fit", t.best_fit}, {"approximate", t.approx || t.v3 || t.v5},
                    {"fission", t.fission}, {"prune_dominated", t.prune_dominated}, {"min_video_degree", t.min_video_degree},
                    {"order", t.order}, {"hold_diffuse", t.hold_diffuse}, {"partition_divisor", t.partition_divisor}};
    };
    for (const auto& item : named()) out.push_back(describe(item.second));
    for (const char* name : {"xdit-sp1", "xdit-sp2", "xdit-cfg2", "static-cfg2", "block-cfg2", "fold-cfg2sp3", "approx-cfg2sp3"})
        out.push_back(describe(policy_traits(name)));
    return out;
}
}
