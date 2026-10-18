#include "ditserve/control/dispatch.hpp"
#include "ditserve/runtime/random.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <tuple>

namespace ditserve {
namespace {
double required(const Json& row, const char* key) { return number(row.at(key), key); }
double signed_number(const Json& value, const std::string& name) {
    if (!value.is_number()) throw std::invalid_argument(name + " must be numeric");
    const double result = value.get<double>();
    if (!std::isfinite(result)) throw std::invalid_argument(name + " must be finite");
    return result;
}
using Row = const Json*;
auto edf(Row r) {
    return std::make_tuple(number_field(*r, "deadline_ms"), number_field(*r, "arrival_ms"), r->at("request_id").get<std::string>());
}
double slack(Row r, double now) { return number_field(*r, "deadline_ms") - now - number_field(*r, "remaining_ms"); }
std::string qkey(const Json& q) { return q.is_string() ? q.get<std::string>() : q.dump(); }
double action_cost(Row r, const Json& q) {
    return number(r->at("action_ms_by_quantum").at(qkey(q)), "action cost");
}
bool threatened(const std::vector<Row>& rows, Row selected, double cost, double now) {
    return std::any_of(rows.begin(), rows.end(), [&](Row r) {
        return r != selected && slack(r, now) >= 0 && slack(r, now) < cost;
    });
}
std::pair<Json, bool> choose_quantum(Row row, const std::vector<Row>& rows,
                                   double now, const Json& q, const Json& costs) {
    if (q != "adaptive") { action_cost(row, q); return {q, false}; }
    double limit = number(costs.at("image_deadline_ms"), "image_deadline_ms", true) * .5;
    for (Row other : rows) if (other != row && slack(other, now) >= 0) limit = std::min(limit, slack(other, now));
    int best = 0;
    for (int size : {1, 4, 8}) {
        if (size != 1 && row->value("action", "") != "denoise") break;
        if (action_cost(row, size) <= limit) best = size;
    }
    return {best ? best : 1, best == 0};
}
Json decision(Row row, const Json& q, const std::string& reason, bool conflict, double now) {
    return {{"request_id", row->at("request_id")}, {"quantum", q}, {"reason", reason},
            {"predicted_deadline_conflict", conflict}, {"slack_ms", slack(row, now)}};
}
double quantile(std::vector<double> values, double p) {
    std::sort(values.begin(), values.end());
    const double position = (values.size() - 1) * p;
    const auto lower = static_cast<std::size_t>(position);
    return values[lower] + (values[std::min(lower + 1, values.size() - 1)] - values[lower]) * (position - lower);
}
double gamma(Random& rng, double shape) {
    if (shape < 1) return gamma(rng, shape + 1) * std::pow(std::max(rng.uniform(), 1e-15), 1 / shape);
    const double d = shape - 1.0 / 3, c = 1 / std::sqrt(9 * d);
    for (;;) {
        const double x = rng.normal(), root = 1 + c * x;
        if (root <= 0) continue;
        const double v = root * root * root, u = rng.uniform();
        if (u < 1 - .0331 * x*x*x*x || std::log(std::max(u, 1e-15)) < .5*x*x + d*(1-v+std::log(v))) return d * v;
    }
}
}

Json select_action(const Json& input, double now, const std::string& policy,
                   const Json& costs, const std::string& active, const Json& default_quantum) {
    number(now, "now_ms");
    if (!input.is_array()) throw std::invalid_argument("ready rows must be an array");
    const bool session = policy == "session_serial" || policy == "stage_edf" || policy == "session_slack";
    const bool benchmark = policy == "fcfs" || policy == "edf" || policy == "srtf";
    static const std::set<std::string> arms = {"edf_b1", "edf_b1_async", "edf_b2", "edf_b4", "edf_step", "guarded_b4", "guarded_adaptive", "feasible_b4"};
    if (!session && !benchmark && !arms.count(policy)) throw std::invalid_argument("unknown policy: " + policy);
    std::vector<Row> rows;
    std::set<std::string> ids;
    for (const auto& row : input) {
        const auto status = row.value("status", "");
        if (status != "pending" && !(session && status == "running")) continue;
        if (session ? !row.value("ready", false) : (!row.value("arrived", false) || !row.value("gpu_ready", false))) continue;
        if (required(row, session ? "release_ms" : "arrival_ms") > now) continue;
        const auto id = row.at("request_id").get<std::string>();
        if (id.empty() || !ids.insert(id).second) throw std::invalid_argument("ready IDs must be unique and nonempty");
        required(row, "deadline_ms");
        if (session) {
            required(row, "stage_deadline_ms"); required(row, "remaining_total_ms");
            if (number_field(row, "deadline_ms") < number_field(row, "release_ms")) throw std::invalid_argument("deadline precedes release");
        } else {
            required(row, "remaining_ms");
            if (!benchmark) {
                const double since = required(row, "ready_since_ms");
                if (since < number_field(row, "arrival_ms") || since > now) throw std::invalid_argument("invalid ready interval");
                if (required(row, "service_ms") <= 0 || (row.at("kind") != "video" && row.at("kind") != "image"))
                    throw std::invalid_argument("invalid service cost or kind");
                const auto action = row.at("action").get<std::string>();
                if (action != "prepare" && action != "denoise" && action != "decode" && action != "image")
                    throw std::invalid_argument("unknown request action");
            }
        }
        rows.push_back(&row);
    }
    if (rows.empty()) return {{"request_id", nullptr}, {"quantum", nullptr}, {"reason", "no_ready_request"},
                              {"predicted_deadline_conflict", false}, {"slack_ms", nullptr}};
    Row current = nullptr;
    for (Row r : rows) if (r->at("request_id") == active) current = r;
    if (session) {
        auto key = [&](Row r) {
            const double primary = policy == "stage_edf" ? number_field(*r, "stage_deadline_ms") :
                policy == "session_slack" ? number_field(*r, "deadline_ms") - now - number_field(*r, "remaining_total_ms") : number_field(*r, "deadline_ms");
            return std::make_tuple(primary, number_field(*r, "deadline_ms"), number_field(*r, "release_ms"), r->at("request_id").get<std::string>());
        };
        Row selected = policy == "session_serial" && current ? current : *std::min_element(rows.begin(), rows.end(), [&](Row a, Row b){ return key(a) < key(b); });
        return {{"request_id", selected->at("request_id")}, {"quantum", default_quantum}, {"reason", policy},
                {"predicted_deadline_conflict", false}, {"slack_ms", number_field(*selected, "deadline_ms") - now - number_field(*selected, "remaining_total_ms")}};
    }
    if (benchmark) {
        auto priority = [&](Row r) { return number_field(*r, policy == "fcfs" ? "arrival_ms" : policy == "edf" ? "deadline_ms" : "remaining_ms"); };
        Row best = *std::min_element(rows.begin(), rows.end(), [&](Row a, Row b) {
            return std::make_tuple(priority(a), number_field(*a, "arrival_ms"), a->at("request_id")) <
                   std::make_tuple(priority(b), number_field(*b, "arrival_ms"), b->at("request_id"));
        });
        if (current && (policy == "fcfs" || priority(current) <= priority(best))) best = current;
        return decision(best, default_quantum, policy, false, now);
    }
    const Json quantum = policy == "edf_step" ? Json("step") : policy == "guarded_adaptive" ? Json("adaptive") :
                         policy == "edf_b2" ? Json(2) : policy == "edf_b1" || policy == "edf_b1_async" ? Json(1) : Json(4);
    Row earliest = *std::min_element(rows.begin(), rows.end(), [](Row a, Row b){ return edf(a) < edf(b); });
    if (policy.compare(0, 3, "edf") == 0) { action_cost(earliest, quantum); return decision(earliest, quantum, "edf", false, now); }
    Row waiting = nullptr;
    auto wait_key = [](Row r) { return std::make_tuple(number_field(*r,"ready_since_ms") + 2.5*number_field(*r,"service_ms"), edf(r)); };
    for (Row r : rows) if (now - number_field(*r,"ready_since_ms") >= 2.5*number_field(*r,"service_ms") && (!waiting || wait_key(r) < wait_key(waiting))) waiting = r;
    if (waiting) {
        auto q = choose_quantum(waiting, rows, now, quantum, costs);
        const bool conflict = q.second || threatened(rows, waiting, action_cost(waiting,q.first),now);
        if (!conflict) return decision(waiting,q.first,"waiting_guard",false,now);
        action_cost(earliest,1); return decision(earliest,1,"deadline_conflict_edf_min_quantum",true,now);
    }
    Row candidate = nullptr;
    for (Row r : rows) if (slack(r,now) >= 0 && (!candidate || edf(r) < edf(candidate))) candidate = r;
    const std::string reason = candidate ? "feasible_edf" : "edf_all_predicted_late";
    if (!candidate) candidate = earliest;
    auto q = choose_quantum(candidate,rows,now,quantum,costs);
    const double cost = action_cost(candidate,q.first);
    if (policy.compare(0,7,"guarded") == 0) {
        std::vector<Row> videos;
        for (Row r : rows) if (r != candidate && r->at("kind") == "video" && r->value("started",false) && slack(r,now) >= 0 && slack(r,now) < cost) videos.push_back(r);
        std::sort(videos.begin(),videos.end(),[&](Row a,Row b){ return std::make_tuple(slack(a,now),edf(a)) < std::make_tuple(slack(b,now),edf(b)); });
        for (Row r : videos) {
            auto vq = choose_quantum(r,rows,now,quantum,costs);
            const double vc = action_cost(r,vq.first);
            if (!vq.second && slack(candidate,now) >= vc && !threatened(rows,r,vc,now)) return decision(r,vq.first,"video_resume_guard",false,now);
        }
    }
    if (q.second || threatened(rows,candidate,cost,now)) {
        action_cost(earliest,1); return decision(earliest,1,"deadline_conflict_edf_min_quantum",true,now);
    }
    return decision(candidate,q.first,reason,false,now);
}

Json study_workload(const Json& phase) {
    Json rows = phase.value("workload",Json::array());
    if (!rows.is_array()) throw std::invalid_argument("workload must be an array of requests");
    if (rows.empty() && phase.contains("generator")) {
        const auto& g = phase.at("generator");
        const auto seed = integer(g.value("seed",Json(7)),"seed");
        Random arrivals(seed), content(static_cast<std::uint64_t>(seed) ^ 0x5DEECE66DULL);
        const int count = int_field(g,"count",8,0);
        const auto pattern = g.value("pattern","offline_all");
        if (pattern != "offline_all" && pattern != "steady" && pattern != "poisson" && pattern != "bursty" && pattern != "gamma") throw std::invalid_argument("unknown arrival pattern");
        const double rate = number_field(g,"rate_rps",1,true), cv = number_field(g,"cv",2,true);
        const double period = number_field(g,"burst_period_ms",1000,true), fraction = number_field(g,"burst_active_fraction",.2,true);
        if (fraction > 1) throw std::invalid_argument("burst active fraction exceeds one");
        const auto prompts = g.value("prompts",Json::array({"A red ceramic kettle on a wooden table."}));
        const auto specs = g.value("specs",Json::array({g.value("spec",Json::object())}));
        if (!prompts.is_array() || prompts.empty() || !specs.is_array() || specs.empty()) throw std::invalid_argument("prompts and specs must be nonempty");
        // Optional request mixture: each class names its kind, worker, cost
        // class, spec, service estimate and SLO (absolute or a service multiple).
        const auto classes = g.value("classes",Json::array());
        if (!classes.is_array()) throw std::invalid_argument("generator classes must be an array");
        double total_weight = 0;
        for (const auto& c : classes) total_weight += number_field(c,"weight",1,true);
        double exposure = 0;
        for (int i=0;i<count;++i) {
            if (i) exposure += pattern == "steady" ? 1000/rate : pattern == "gamma" ? gamma(arrivals,1/(cv*cv))*cv*cv*1000/rate : arrivals.exponential(rate/1000/(pattern=="bursty"?fraction:1));
            const double arrival = pattern == "offline_all" ? 0 : pattern == "bursty" ? std::floor(exposure/(period*fraction))*period + std::fmod(exposure,period*fraction) : exposure;
            Json row={{"request_id","r"+std::to_string(i)}, {"arrival_ms",arrival}};
            Json spec;
            if (classes.empty()) {
                spec = specs.at(content.bounded(specs.size()));
                row["service_ms"]=number_field(g,"service_ms",100); row["slo_ms"]=number_field(g,"slo_ms",250);
            } else {
                double pick = content.uniform()*total_weight;
                std::size_t chosen = 0;
                for (; chosen+1<classes.size(); ++chosen) { pick -= number_field(classes[chosen],"weight",1,true); if (pick < 0) break; }
                const auto& c = classes[chosen];
                spec = c.value("spec",Json::object());
                const double service = number_field(c,"service_ms",100,true);
                row["service_ms"]=service; row["slo_ms"]=c.contains("slo_ms")?number_field(c,"slo_ms",0,true):service*number_field(c,"slo_scale",2.5,true);
                row["kind"]=c.at("kind"); row["class"]=c.at("class");
                if (c.contains("worker")) row["worker"]=c.at("worker");
            }
            spec["prompt"] = prompts.at(content.bounded(prompts.size())); spec["seed"] = content.bounded(1ULL<<31);
            row["spec"]=spec;
            rows.push_back(row);
        }
    }
    std::set<std::string> ids;
    for (std::size_t i=0;i<rows.size();++i) {
        auto& row=rows[i];
        row["request_id"] = row.value("request_id","r"+std::to_string(i));
        const auto id=row.at("request_id").get<std::string>();
        if (id.empty() || !ids.insert(id).second) throw std::invalid_argument("duplicate/empty workload request ID");
        row["arrival_ms"] = number_field(row,"arrival_ms");
        row["service_ms"] = number_field(row,"service_ms",phase.value("service_ms",100.));
        row["slo_ms"] = number_field(row,"slo_ms",row["service_ms"].get<double>()*2.5);
        row["deadline_ms"] = number_field(row,"deadline_ms",row["arrival_ms"].get<double>()+row["slo_ms"].get<double>());
        if (row["deadline_ms"].get<double>() < row["arrival_ms"].get<double>()) throw std::invalid_argument("deadline before arrival");
        row["spec"] = row.value("spec",phase.value("spec",Json::object()));
        row["spec"]["request_id"]=id;
        row["worker"] = row.value("worker",phase.value("worker","model"));
        row["kind"] = row.value("kind",phase.value("request_kind","video"));
        row["status"]="pending";
    }
    for (const auto& row:rows) if (row.contains("after") && (!ids.count(row.at("after").get<std::string>()) || row.at("after")==row.at("request_id"))) throw std::invalid_argument("invalid predecessor request");
    // A dependency must precede its consumer in the frozen trace. This also
    // excludes cycles without a second workflow graph implementation.
    ids.clear();
    for (const auto& row:rows) {
        if (row.contains("after") && !ids.count(row.at("after").get<std::string>())) throw std::invalid_argument("predecessor must precede request in trace");
        ids.insert(row.at("request_id").get<std::string>());
    }
    return rows;
}

Json request_summary(const Json& rows) {
    if (!rows.is_array()) throw std::invalid_argument("metric rows must be an array");
    std::size_t complete=0,failed=0,ontime=0,quality=0,qualified_evidence=0;
    double exposure=0,work=0,control=0;
    std::map<std::string,double> windows;
    std::set<std::pair<std::string,std::string>> identities;
    std::vector<double> latencies,overhead;
    for (const auto& row:rows) {
        const auto id = row.at("request_id").get<std::string>();
        const auto observation = row.value("observation_id", std::string());
        if (id.empty() || !identities.emplace(observation,id).second)
            throw std::invalid_argument("duplicate/empty request ID within one observation");
        auto& window = windows[observation];
        window = std::max(window,number_field(row,"observation_ms"));
        if (row.contains("finish_ms") && !row.at("finish_ms").is_null())
            window = std::max(window,required(row,"finish_ms"));
        work+=number_field(row,"model_ms"); control+=number_field(row,"control_ms");
        if(row.contains("communication_samples_ms")) for(const auto& n:row.at("communication_samples_ms")) overhead.push_back(number(n,"communication ms"));
        bool qualified=false;
        if(row.contains("qualification") && !row.at("qualification").is_null()) {
            const auto& evidence=row.at("qualification");
            if(!evidence.is_object() || (evidence.contains("qualified") && !evidence.at("qualified").is_boolean()))
                throw std::invalid_argument("request qualification must contain a boolean qualified value");
            if(evidence.contains("qualified") && evidence.value("status","")!="unverified") {
                ++qualified_evidence;qualified=evidence.at("qualified").get<bool>();
            }
        }
        const auto status=row.value("status","");
        if(status=="completed") {
            ++complete;
            const double finish=required(row,"finish_ms"),arrival=required(row,"arrival_ms"),deadline=required(row,"deadline_ms");
            if (finish<arrival || deadline<arrival) throw std::invalid_argument("completed request timing precedes arrival");
            latencies.push_back(finish-arrival);
            const bool timely=finish<=deadline; ontime+=timely;
            if(timely && qualified) ++quality;
        } else if(status=="failed" || status=="interrupted") ++failed;
    }
    for (const auto& window:windows) exposure+=window.second;
    if (!std::isfinite(exposure) || !std::isfinite(work) || !std::isfinite(control))
        throw std::invalid_argument("metric totals overflowed");
    auto pct=[&](std::size_t n)->Json{return rows.empty()?Json(nullptr):Json(double(n)/rows.size());};
    return {{"requests",rows.size()},{"completed",complete},{"failed",failed},{"incomplete",rows.size()-complete-failed},
            {"ontime",ontime},{"deadline_attainment",pct(ontime)},
            {"qualified_goodput_rps",qualified_evidence && exposure>0?Json(quality*1000/exposure):Json(nullptr)},
            {"qualification_status",!qualified_evidence?"unverified":qualified_evidence==rows.size()?"verified":"partial"},
            {"qualification_verified_requests",qualified_evidence},{"qualified_ontime",quality},
            {"observation_ms",exposure},{"observation_count",windows.size()},
            {"latency_p50_ms",latencies.empty()?Json(nullptr):Json(quantile(latencies,.5))},
            {"latency_p95_ms",latencies.empty()?Json(nullptr):Json(quantile(latencies,.95))},
            {"communication_p50_ms",overhead.empty()?Json(nullptr):Json(quantile(overhead,.5))},
            {"communication_p99_ms",overhead.empty()?Json(nullptr):Json(quantile(overhead,.99))},
            {"model_ms",work},{"control_ms",control},{"denominator","all frozen requests, including failed and blocked"}};
}

Json kind_summary(const Json& rows, const std::string& kind) {
    std::vector<double> latency;
    std::size_t count=0,ontime=0;
    for (const auto& row:rows) {
        if (row.value("kind","")!=kind) continue;
        ++count;
        if (row.value("status","")!="completed") continue;
        latency.push_back(required(row,"finish_ms")-required(row,"arrival_ms"));
        ontime += required(row,"finish_ms")<=required(row,"deadline_ms");
    }
    auto at=[&](double p){return latency.empty()?Json(nullptr):Json(quantile(latency,p));};
    return {{"requests",count},{"completed",latency.size()},{"ontime",ontime},
            {"attainment",count?Json(double(ontime)/count):Json(nullptr)},
            {"latency_p50_ms",at(.5)},{"latency_p95_ms",at(.95)},{"latency_p99_ms",at(.99)}};
}

Json evaluate_gates(const Json& evidence,const Json& rules) {
    if(!rules.is_array()) throw std::invalid_argument("gates must be an array");
    Json checks=Json::array(); bool qualified=!rules.empty();
    Json scopes=Json::object();
    for(const auto* scope:{"numeric","semantic","performance"})
        scopes[scope]={{"status","unverified"},{"checks",0},{"verified",0},{"passed",0}};
    for(const auto& rule:rules) {
        const auto scope=rule.value("scope","numeric");
        if(!scopes.contains(scope))throw std::invalid_argument("gate scope must be numeric, semantic or performance");
        const auto path=rule.at("metric").get<std::string>();
        const Json::json_pointer pointer(path);
        const bool present=evidence.contains(pointer) && !evidence.at(pointer).is_null();
        const auto op=rule.value("op","ge");
        if(op!="eq" && op!="ge" && op!="le" && op!="gt" && op!="lt") throw std::invalid_argument("unknown gate comparison");
        const Json actual=present?evidence.at(pointer):Json(nullptr), expected=rule.at("value");
        const double threshold=op=="eq"?0:signed_number(expected,"gate threshold");
        bool passed=false;
        if(present) {
            if(op=="eq") passed=actual==expected;
            else {const double a=signed_number(actual,path);passed=op=="ge"?a>=threshold:op=="le"?a<=threshold:op=="gt"?a>threshold:a<threshold;}
        }
        qualified=qualified&&passed;
        auto& group=scopes[scope];
        group["checks"]=group["checks"].get<int>()+1;
        group["verified"]=group["verified"].get<int>()+int(present);
        group["passed"]=group["passed"].get<int>()+int(passed);
        checks.push_back({{"metric",path},{"scope",scope},{"status",!present?"unverified":passed?"passed":"failed"},{"actual",actual},{"op",op},{"expected",expected}});
    }
    for(auto& group:scopes)if(group["checks"].get<int>()>0)
        group["status"]=group["verified"]!=group["checks"]?"unverified":group["passed"]==group["checks"]?"passed":"failed";
    return {{"qualified",qualified},{"status",rules.empty()?"unverified":qualified?"passed":"failed"},
            {"scope","configured_gates_only"},{"scopes",scopes},{"checks",checks}};
}

Json expand_matrix(const Json& phase) {
    Json base=phase; base.erase("matrix"); Json rows=Json::array({base});
    const auto matrix=phase.value("matrix",Json::object());
    if(!matrix.is_object()) throw std::invalid_argument("matrix must map JSON pointers to arrays");
    for(auto it=matrix.begin();it!=matrix.end();++it) {
        if(!it.value().is_array() || it.value().empty()) throw std::invalid_argument("matrix choices must be nonempty arrays");
        if(rows.size()*it.value().size()>10000) throw std::invalid_argument("matrix exceeds 10000 cells");
        Json next=Json::array();
        for(const auto& row:rows) for(const auto& value:it.value()) { Json item=row;item[Json::json_pointer(it.key())]=value;next.push_back(item); }
        rows=std::move(next);
    }
    return rows;
}
}
