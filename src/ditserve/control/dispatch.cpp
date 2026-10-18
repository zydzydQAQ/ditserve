#include "ditserve/control/dispatch.hpp"
#include "ditserve/runtime/random.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>

namespace ditserve {
namespace {
double required(const Json& row, const char* key) { return number(row.at(key), key); }
double signed_number(const Json& value, const std::string& name) {
    if (!value.is_number()) throw std::invalid_argument(name + " must be numeric");
    const double result = value.get<double>();
    if (!std::isfinite(result)) throw std::invalid_argument(name + " must be finite");
    return result;
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
        // Diffusion steps belong to the request, not the hardware profile.
        if (row["kind"]=="video" && row["spec"].contains("steps") && !row.contains("steps_total")) row["steps_total"]=row["spec"]["steps"];
        row["status"]="pending";
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
