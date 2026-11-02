#include "ditserve/control/cluster.hpp"
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
using namespace ditserve;
namespace {
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
void near(double a,double b){require(std::abs(a-b)<1e-8,"numeric reconstruction differs");}
Json profile(){return {{"provenance","synthetic"},{"classes",{{"video",{{"kind","video"},{"steps",50},{"blocks_per_step",60},
    {"step_compute_ms",2000},{"prepare_ms",0},{"finish_ms",100},{"comm_mb_per_step",0},{"collectives_per_step",0},
    {"forms",{{"cfg2",{{"group",{0,1}},{"step_ms",1200}}}}}}},
    {"image",{{"kind","image"},{"service_ms",20}}}}},
    {"links",{{"2",{{"gbps",20},{"latency_ms",0}}}}},{"quantum_overhead_ms",0.5}};}
Json row(std::string id,double deadline,int done=0,int block=0){return {{"request_id",id},{"class","video"},{"kind","video"},
    {"arrival_ms",0},{"deadline_ms",deadline},{"status","suspended"},{"ready",true},{"prepared",true},
    {"steps_done",done},{"blocks_done",block},{"group",{0,1}},{"form","cfg2"},{"rank_local_blocks",true}};}
Json state(Json rows,double now=100){return {{"now_ms",now},{"gpus",2},{"busy",{false,false}},{"requests",rows}};}
const Json& find(const ClusterPlanTrace& t,const char* id){for(const auto& r:t.rows)if(r.at("request_id")==id)return r;throw std::runtime_error("trace row absent");}
Json options(){return {{"quantum_blocks",30},{"demote_late",true},{"hold_diffuse",false}};}
void check(const char* name,const std::function<void()>& action,int& failures){try{action();}catch(const std::exception& e){++failures;std::cerr<<name<<": "<<e.what()<<'\n';}}
}
int main(){int failures=0;
    check("small miss is demoted with real boundary estimate",[]{
        ClusterCost cost(profile());const auto cfg=parse_form("cfg2");
        Json older=row("older",0,10), other=row("other",100000);other["steps_total"]=1;
        const double remaining=cost.remaining_ms(older,cfg)+41*0.5;
        older["deadline_ms"]=100+remaining-6.95;
        ClusterPlanTrace trace;const auto input=state({older,other});
        const auto plain=cluster_plan(input,"block-cfg2",cost,options());
        const auto observed=cluster_plan(input,"block-cfg2",cost,options(),&trace);
        require(plain==observed,"diagnostics changed assignments");
        const auto& r=find(trace,"older");near(r.at("laxity_ms"),-6.95);
        require(r.at("demoted_late")==true&&r.at("estimated_feasible")==false,"lateness predicates differ");
        require(r.at("estimate_group").is_null()&&r.at("resident_group")==Json({0,1}),"boundary must not inherit resident placement");
        require(r.at("step_estimator_source")=="analytic_no_matching_table","boundary incorrectly borrowed group sample");
        require(r.at("modeled_dispatch_quanta")==41,"dispatch quanta incorrectly counted");
        near(r.at("estimated_remaining_ms"),r.at("remaining_without_dispatch_ms").get<double>()+41*0.5);
        require(r.at("rank_position")==1&&find(trace,"other").at("rank_position")==0,"real late rank not retained");
        auto opt=options();opt["demote_late"]=false;cluster_plan(input,"block-cfg2",cost,opt,&trace);
        require(find(trace,"older").at("estimated_feasible")==false&&find(trace,"older").at("demoted_late")==false,"raw feasibility conflated with demotion option");
        opt["demote_late"]=true;opt["late_short_ms"]=1000000;cluster_plan(input,"block-cfg2",cost,opt,&trace);
        require(find(trace,"older").at("demoted_late")==false,"short-late exemption disappeared");
    },failures);
    check("actual cost branch provenance and folded progress",[]{
        auto p=profile();ClusterPlanTrace trace;
        const auto input=state({row("mid",100000,10,15)});
        cluster_plan(input,"block-cfg2",ClusterCost(p),options(),&trace);
        auto r=find(trace,"mid");require(r.at("step_estimator_source")=="group_scoped_table","scoped table not reported");
        require(r.at("estimate_group")==Json({0,1}),"midstep group missing");
        near(r.at("remaining_without_dispatch_ms"),39.5*1200+100);
        p["classes"]["video"]["forms"]["cfg2"]={{"placements",{{"0-1",{{"step_ms",1300}}}}}};
        cluster_plan(input,"block-cfg2",ClusterCost(p),options(),&trace);
        require(find(trace,"mid").at("step_estimator_source")=="placement_table","placement branch missing");
        p["classes"]["video"]["forms"]["cfg2"]={{"step_ms",1400}};
        cluster_plan(input,"block-cfg2",ClusterCost(p),options(),&trace);
        require(find(trace,"mid").at("step_estimator_source")=="legacy_table","legacy branch missing");
        p["quantum_overhead_ms"]=0;
        cluster_plan(input,"block-cfg2",ClusterCost(p),options(),&trace);
        require(find(trace,"mid").at("modeled_dispatch_quanta").is_null(),"uncomputed zero-cost quanta invented");
        auto folded=input;folded["requests"][0]["cfg_folded"]=true;
        const auto a=cluster_plan(folded,"block-cfg2",ClusterCost(p),options());
        const auto b=cluster_plan(folded,"block-cfg2",ClusterCost(p),options(),&trace);
        require(a==b&&find(trace,"mid").at("cfg_folded")==true,"folded capture changes decisions");
    },failures);
    check("all rank rules and busy snapshots preserve exact dispatch",[]{
        for(const char* rule:{"edf","srtf","fcfs"})for(const char* policy:{"block-cfg2","static-cfg2","block_elastic"})
            for(int step:{0,10,49})for(int block:{0,15})for(bool occupied:{false,true}){
                auto in=state({row("v",30000,step,block)});in["busy"]={occupied,occupied};
                auto opt=options();opt["order"]=rule;opt["forms"]={"sp1","cfg2","sp2"};
                ClusterPlanTrace trace;
                require(cluster_plan(in,policy,ClusterCost(profile()),opt)==cluster_plan(in,policy,ClusterCost(profile()),opt,&trace),
                    "trace altered policy output on fixed state");
            }
    },failures);
    check("row caps and focus cannot alter assignments",[]{
        Json requests=Json::array();for(int i=0;i<80;++i){auto r=row("v"+std::to_string(i),100000+i);r["group"]=Json::array();r["prepared"]=false;requests.push_back(r);}
        const auto in=state(requests);ClusterPlanTrace trace;trace.max_rows=2;
        const auto plain=cluster_plan(in,"block-cfg2",ClusterCost(profile()),options());
        require(plain==cluster_plan(in,"block-cfg2",ClusterCost(profile()),options(),&trace),"row cap changed assignment");
        require(trace.candidates==80&&trace.rows.size()==2&&trace.dropped_rows==78,"row omission denominator wrong");
        trace.focus={"v79"};cluster_plan(in,"block-cfg2",ClusterCost(profile()),options(),&trace);
        require(trace.rows.size()==1&&trace.filtered_rows==79&&trace.dropped_rows==0,"focus accounting wrong");
        trace.focus.clear();trace.max_rows=64;
        cluster_plan(in,"block-cfg2",ClusterCost(profile()),options(),&trace);
        std::size_t actual=0;for(const auto& r:trace.rows)actual+=r.dump().size();
        require(trace.row_bytes==actual&&actual<=trace.max_bytes,"final row bytes differ from claimed bound");
        trace.rows[0]["extra_shape"]=std::string(70000,'x');trace.enforce_bounds();
        actual=0;for(const auto& r:trace.rows)actual+=r.dump().size();
        require(trace.row_bytes==actual&&actual<=trace.max_bytes&&trace.dropped_rows>16,"caller-enriched row is not bounded");
        trace.max_bytes=1;cluster_plan(in,"block-cfg2",ClusterCost(profile()),options(),&trace);
        require(trace.rows.empty()&&trace.dropped_rows==80,"byte cap lost omitted rows");
    },failures);
    if(failures)return 1;std::cout<<"4 decision telemetry groups passed\n";return 0;
}
