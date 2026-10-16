#include "ditserve/control/study.hpp"
#include "ditserve/control/budget.hpp"
#include "ditserve/control/dispatch.hpp"
#include "ditserve/runtime/process.hpp"
#include "ditserve/benchmark/simulator.hpp"
#include "ditserve/benchmark/workload.hpp"
#include <openssl/sha.h>
#include <algorithm>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <memory>
#include <mutex>
#include <deque>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

namespace ditserve {
namespace {
namespace fs=std::filesystem;
volatile std::sig_atomic_t cancelled=0;
void cancel_handler(int) { cancelled=1; }
struct Signals {
    using Handler=void(*)(int);
    Handler interrupt,terminate;
    Signals():interrupt(std::signal(SIGINT,cancel_handler)),terminate(std::signal(SIGTERM,cancel_handler)){cancelled=0;}
    ~Signals(){std::signal(SIGINT,interrupt);std::signal(SIGTERM,terminate);}
};
struct RunLock {
    int fd=-1;
    explicit RunLock(const fs::path& path) {
        fd=::open(path.c_str(),O_CREAT|O_RDWR|O_CLOEXEC|O_NOFOLLOW,0600);
        if(fd<0 || ::flock(fd,LOCK_EX|LOCK_NB)!=0){if(fd>=0)::close(fd);throw std::runtime_error("run is locked: "+path.string());}
    }
    ~RunLock(){if(fd>=0)::close(fd);}
};
Json read(const fs::path& path) {
    std::ifstream file(path); if(!file)throw std::runtime_error("cannot read "+path.string());
    return Json::parse(file);
}
void write(const fs::path& path,const Json& value) {
    fs::create_directories(path.parent_path());
    const fs::path tmp=path.string()+".tmp";
    const int fd=::open(tmp.c_str(),O_WRONLY|O_CREAT|O_TRUNC|O_CLOEXEC|O_NOFOLLOW,0600);
    if(fd<0)throw std::runtime_error("cannot write "+tmp.string());
    const std::string text=value.dump(2)+"\n";
    std::size_t offset=0;
    while(offset<text.size()) {const auto n=::write(fd,text.data()+offset,text.size()-offset);if(n<0){::close(fd);throw std::runtime_error("write failed");}offset+=n;}
    const int synced=::fsync(fd);::close(fd);
    if(synced!=0)throw std::runtime_error("fsync failed");
    fs::rename(tmp,path);
    const int parent=::open(path.parent_path().c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC);
    if(parent>=0){::fsync(parent);::close(parent);}
}
std::string digest(const std::string& text) {
    unsigned char bytes[SHA256_DIGEST_LENGTH];SHA256(reinterpret_cast<const unsigned char*>(text.data()),text.size(),bytes);
    std::ostringstream out;for(auto b:bytes)out<<std::hex<<std::setw(2)<<std::setfill('0')<<int(b);return out.str();
}
std::string file_hash(const fs::path& path) {
    std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("cannot hash "+path.string());
    SHA256_CTX ctx;SHA256_Init(&ctx);char block[65536];
    while(in){in.read(block,sizeof block);SHA256_Update(&ctx,block,static_cast<std::size_t>(in.gcount()));}
    if(!in.eof())throw std::runtime_error("hash read failure");
    unsigned char bytes[SHA256_DIGEST_LENGTH];SHA256_Final(bytes,&ctx);
    std::ostringstream out;for(auto b:bytes)out<<std::hex<<std::setw(2)<<std::setfill('0')<<int(b);return out.str();
}
std::string segment(const std::string& value) {
    if(value.empty() || value.size()>120 || std::string("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789").find(value.front())==std::string::npos || value.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.")!=std::string::npos)
        throw std::invalid_argument("invalid path identifier: "+value);
    return value;
}
fs::path real_path(const fs::path& path) {
    const auto absolute=fs::absolute(path).lexically_normal();
    if(fs::weakly_canonical(absolute)!=absolute)throw std::invalid_argument("work/result paths cannot traverse symlinks: "+absolute.string());
    return absolute;
}
void wait_until(double deadline) {
    const double seconds=deadline-monotonic_seconds();
    if(seconds>0)std::this_thread::sleep_for(std::chrono::duration<double>(std::min(seconds,.02)));
}
std::string environment(const char* name) { const auto* v=std::getenv(name);return v?v:""; }

struct Worker {
    std::unique_ptr<WorkerProcess> process;
    std::uint64_t sequence=0;
    Json call(Json action,double deadline) {
        action["version"]=1;action["id"]=++sequence;
        return process->call(action,deadline);
    }
};
class Workers {
public:
    Workers(const Json& definitions,const Json& request,const fs::path& work):definitions_(definitions),request_(request),work_(work){}
    Worker& get(const std::string& name) {
        auto found=workers_.find(name);if(found!=workers_.end())return *found->second;
        if(!definitions_.contains(name))throw std::invalid_argument("unknown worker: "+name);
        segment(name);
        const auto& definition=definitions_.at(name);
        auto worker=std::make_unique<Worker>();
        Json config=definition.value("config",Json::object());
        config["backend"]=definition.at("backend");
        if(definition.contains("gpu_uuid")){
            config["gpu_uuid"]=definition.at("gpu_uuid");
            config["device_id"]=definition.at("gpu_uuid");
        }
        config["work_dir"]=(work_/"artifacts"/name).string();
        fs::create_directories(work_/"artifacts"/name);
        if(definition.contains("device"))config["device"]=definition["device"];
        const auto config_path=work_/"workers"/(name+".json");write(config_path,config);
        ProcessOptions options;
        options.argv={definition.value("python",request_.value("python","python3")),"-u","-m","ditserve.runtime.worker","--backend",definition.at("backend").get<std::string>(),"--config",config_path.string()};
        options.stderr_path=work_/"workers"/(name+".log");
        options.heartbeat=heartbeat_;
        const auto root=fs::path(request_.at("package_root").get<std::string>());
        const auto frozen=request_.contains("frozen_python")?fs::path(request_.at("frozen_python").get<std::string>()):work_/"sources"/"python";
        options.environment={{"PYTHONPATH",(fs::exists(frozen)?frozen:root).string()+":"+environment("PYTHONPATH")},
            {"PYTHONDONTWRITEBYTECODE","1"},{"HF_HUB_OFFLINE","1"},{"TRANSFORMERS_OFFLINE","1"}};
        options.environment["DITSERVE_ROOT"]=request_.value("root",root.parent_path().string());
        if(definition.at("backend")=="results")options.environment["DITSERVE_REPORT_TOOLS"]=(root.parent_path()/".cache"/"tools").string();
        if(definition.contains("gpu_uuid"))options.environment["CUDA_VISIBLE_DEVICES"]=definition.at("gpu_uuid").get<std::string>();
        if(definition.contains("environment"))for(auto it=definition.at("environment").begin();it!=definition.at("environment").end();++it)options.environment[it.key()]=it.value().get<std::string>();
        worker->process=std::make_unique<WorkerProcess>(options);
        auto* result=worker.get();workers_[name]=std::move(worker);return *result;
    }
    bool stop() {bool clean=true;for(auto& item:workers_)clean=item.second->process->stop()&&clean;return clean;}
    void shutdown(double deadline){for(auto& item:workers_)item.second->call({{"op","shutdown"}},deadline);}
    std::string backend(const std::string& name)const{return definitions_.at(name).at("backend").get<std::string>();}
    void heartbeat(std::function<void()> callback){heartbeat_=std::move(callback);}
    std::vector<int> pids()const{std::vector<int> p;for(const auto& item:workers_)p.push_back(item.second->process->pid());return p;}
private:
    Json definitions_,request_;fs::path work_;std::map<std::string,std::unique_ptr<Worker>> workers_;
    std::function<void()> heartbeat_;
};

Json freeze_sources(const Json& request,const fs::path& work,bool resume) {
    const auto manifest_path=work/"sources"/"manifest.json";
    const auto binary=fs::canonical("/proc/self/exe");
    if(resume) {
        auto manifest=read(manifest_path);
        if(manifest.at("core_sha256")!=file_hash(binary))throw std::runtime_error("C++ binary changed; use a new run");
        for(auto it=manifest.at("files").begin();it!=manifest.at("files").end();++it)
            if(file_hash(work/"sources"/"python"/it.key())!=it.value())throw std::runtime_error("frozen worker source changed");
        const auto native=manifest.value("native_files",Json::object());
        for(auto it=native.begin();it!=native.end();++it)
            if(file_hash(work/"sources"/"native"/it.key())!=it.value())throw std::runtime_error("frozen native source changed");
        return manifest;
    }
    Json manifest={{"core_sha256",file_hash(binary)},{"files",Json::object()}};
    const auto root=fs::path(request.at("package_root").get<std::string>());
    for(const auto& entry:fs::recursive_directory_iterator(root/"ditserve")) {
        if(!entry.is_regular_file() || entry.path().extension()!=".py" || entry.is_symlink())continue;
        const auto relative=fs::relative(entry.path(),root);const auto to=work/"sources"/"python"/relative;
        fs::create_directories(to.parent_path());fs::copy_file(entry.path(),to,fs::copy_options::none);
        manifest["files"][relative.generic_string()]=file_hash(to);
    }
    manifest["native_files"]=Json::object();
    const auto project=root.parent_path();
    for(const auto* directory:{"include","src","cmd","benchmarks"})if(fs::is_directory(project/directory))
        for(const auto& entry:fs::recursive_directory_iterator(project/directory)){
            if(!entry.is_regular_file()||entry.is_symlink()||(entry.path().extension()!=".cpp"&&entry.path().extension()!=".hpp"&&entry.path().filename()!="CMakeLists.txt"))continue;
            const auto relative=fs::relative(entry.path(),project),to=work/"sources"/"native"/relative;
            fs::create_directories(to.parent_path());fs::copy_file(entry.path(),to);
            manifest["native_files"][relative.generic_string()]=file_hash(to);
        }
    if(fs::is_regular_file(project/"CMakeLists.txt")){
        fs::create_directories(work/"sources"/"native");
        fs::copy_file(project/"CMakeLists.txt",work/"sources"/"native"/"CMakeLists.txt");
        manifest["native_files"]["CMakeLists.txt"]=file_hash(project/"CMakeLists.txt");
    }
    fs::copy_file(binary,work/"sources"/"ditserve-core");
    write(manifest_path,manifest);return manifest;
}

Json gpu_observation(const Json& request,const fs::path& work,const Json& devices,const std::vector<int>& allowed={},double deadline=0) {
    if(devices.empty())return {{"status","exclusive_observed"},{"owned",Json::array()},{"foreign",Json::array()}};
    const Json defs={{"probe",{{"backend","mock"}}}};
    Workers probe(defs,request,work/"probe");
    try {
        auto result=probe.get("probe").call({{"op","occupancy"},{"gpu_uuids",devices},{"allowed_pids",allowed},{"allowed_pgids",allowed}},deadline>0?deadline:monotonic_seconds()+15);
        probe.stop();return result;
    }catch(const std::exception& error){probe.stop();return {{"status","unknown"},{"reason",error.what()}};}
}

void add_artifacts(Json& record,const Json& result,const fs::path& work,const Json& dependencies) {
    Json artifacts=result.value("artifacts",Json::array());
    if(result.contains("output") && result["output"].is_object() && result["output"].contains("path"))artifacts.push_back(result["output"]);
    if(result.contains("latent") && result["latent"].is_object() && result["latent"].contains("path"))artifacts.push_back(result["latent"]);
    for(auto artifact:artifacts) {
        const fs::path path=artifact.at("path").get<std::string>();
        const auto absolute=fs::absolute(path).lexically_normal();
        const auto relative=absolute.lexically_relative(work);
        if(relative.empty() || *relative.begin()==".." || fs::weakly_canonical(absolute)!=absolute)throw std::runtime_error("worker artifact escapes working directory");
        if(!fs::is_regular_file(absolute))throw std::runtime_error("worker artifact is missing");
        const auto bytes=fs::file_size(absolute);
        if(artifact.contains("bytes") && artifact["bytes"]!=bytes)throw std::runtime_error("artifact size mismatch");
        artifact["path"]=relative.generic_string();artifact["bytes"]=bytes;
        artifact["dependencies"]=dependencies;artifact["kind"]=artifact.value("kind","raw");
        for(const auto& previous:record["artifacts"])if(previous.at("path")==artifact.at("path"))throw std::runtime_error("artifact path reused during run");
        record["artifacts"].push_back(artifact);
    }
}

struct Live {
    std::string worker,state,snapshot,decode_ticket;
    Json progress=Json::object();
    Json decoded=nullptr;
    std::size_t stage=0;
    std::set<std::size_t> transitions;
    bool suspended=false;
};
Json stages(const Json& row) {
    if(row.contains("stages"))return row.at("stages");
    return Json::array({{{"worker",row.at("worker")},{"spec",row.at("spec")}}});
}
double remaining(const Json& row,const Live* live,const Json& costs) {
    const auto list=stages(row);const std::size_t stage=live?live->stage:0;
    double total=0;
    for(std::size_t i=stage;i<list.size();++i) {
        const auto& spec=list[i].value("spec",row.at("spec"));
        const int steps=int_field(spec,"steps",int_field(spec,"max_new_tokens",1),1);
        const int step=live && i==stage?live->progress.value("step",0):0;
        total+=std::max(0,steps-step)*number_field(costs,"step_ms",number_field(row,"service_ms",100)/steps)+number_field(costs,"finish_ms",0);
        if(!live || i!=stage)total+=number_field(costs,"prepare_ms",0);
    }
    return total;
}

Json replay(const Json& phase,Workers& workers,Budget& budget,Json& record,const fs::path& work,const fs::path& cell_dir) {
    Json rows=study_workload(phase),costs=phase.value("costs",Json::object());
    const auto policy=phase.value("policy",phase.value("kind","")=="profile"?"fcfs":"edf");
    std::map<std::string,Live> live;
    std::map<std::string,std::string> previous_llm,encoding;
    std::map<std::string,std::string> decoding;
    std::map<std::string,Json> conversation;
    const double origin=monotonic_seconds();
    auto now=[&](){return (monotonic_seconds()-origin)*1000;};
    auto check=[&](){if(cancelled)throw std::runtime_error("run cancelled");budget.check(number_field(phase,"cleanup_reserve_seconds",0));};
    auto invoke=[&](const std::string& worker,Json action,Json& row) {
        check();const double started=monotonic_seconds();
        auto& process=workers.get(worker);
        const bool starting=process.sequence==0;
        row["worker_pids"][worker]=process.process->pid();
        const auto op=action.at("op").get<std::string>();
        row["operation_counts"][op]=row.value("operation_counts",Json::object()).value(op,0)+1;
        auto result=process.call(action,budget.deadline());
        if(result.contains("profile")){
            if(!row.contains("profiles"))row["profiles"]=Json::array();
            row["profiles"].push_back({{"worker",worker},{"state_id",action.value("state_id","")},{"operation",op},
                {"phase_ms",result.value("phase_ms",Json::object())},{"profile",result.at("profile")}});
        }
        const double elapsed=(monotonic_seconds()-started)*1000;
        const double model=number_field(result,"model_ms",result.value("elapsed_ms",0.));
        const double transport=std::max(0.,elapsed-number_field(result,"worker_operation_ms",model)-number_field(result,"load_ms"));
        row["model_ms"]=number_field(row,"model_ms")+model;
        row["model_load_ms"]=number_field(row,"model_load_ms")+number_field(result,"load_ms");
        row["control_ms"]=number_field(row,"control_ms")+std::max(0.,elapsed-model);
        row["action_ms"][op]=row.value("action_ms",Json::object()).value(op,0.)+elapsed;
        if(!row.contains("communication_samples_ms"))row["communication_samples_ms"]=Json::array();
        if(starting)row["worker_startup_ms"]=number_field(row,"worker_startup_ms")+transport;
        else {
            if(row["communication_samples_ms"].size()<128)row["communication_samples_ms"].push_back(transport);
            row["communication_samples_total"]=row.value("communication_samples_total",0)+1;
        }
        row["communication_sample_scope"]="first 128 resident actions per request, excluding process startup, model load and worker operation; includes JSON framing";
        return result;
    };
    for(auto& row:rows){row["ready_since_ms"]=row["arrival_ms"];row["quality"]={{"status","unverified"},{"passed",false}};
        row["observation_id"]=phase.at("_observation_id");row["phase"]=phase.at("id");row["attempt"]=phase.at("_attempt");row["cell"]=phase.at("_cell");row["order"]=phase.at("_order");}
    std::string active;
    std::size_t terminal=0;
    const int resident_limit=int_field(phase,"resident_limit",100000,1);
    auto make_room=[&](const std::string& incoming,const std::string& worker){
        int resident=0;for(const auto& item:live)if(item.second.worker==worker&&!item.second.suspended)++resident;
        if(resident<resident_limit)return;
        for(auto& item:live)if(item.first!=incoming&&item.second.worker==worker&&!item.second.suspended&&item.second.decode_ticket.empty()){
            auto& victim=*std::find_if(rows.begin(),rows.end(),[&](const Json& r){return r["request_id"]==item.first;});
            if(!item.second.progress.value("transfer_ready",true)){
                const auto advanced=invoke(worker,{{"op","advance"},{"state_id",item.second.state},{"quantum","step"}},victim);
                item.second.progress=advanced.at("progress");
                if(!item.second.progress.value("transfer_ready",true))throw std::runtime_error("backend did not reach a transferable boundary");
            }
            auto snap=invoke(worker,{{"op","snapshot"},{"state_id",item.second.state},{"mode",phase.value("eviction","hot")}},victim);
            item.second.snapshot=snap.value("snapshot_id",item.second.state);item.second.suspended=true;
            if(--resident<resident_limit)return;
        }
        throw std::runtime_error("no legal state can be evicted before admission");
    };
    auto prepare=[&](Live& state,Json& row,const Json& stage){
        state.worker=stage.value("worker",row.at("worker").get<std::string>());
        const auto id=row.at("request_id").get<std::string>();
        make_room(id,state.worker);
        Json spec=stage.value("spec",row.at("spec"));spec["request_id"]=id+"-s"+std::to_string(state.stage);
        Json action={{"op","prepare"},{"spec",spec}};
        if(state.stage==0 && encoding.count(id))action["encode_ticket"]=encoding[id];
        if(workers.backend(state.worker)=="mistral"){
            const auto session=row.value("session_id",id);
            if(!spec.contains("messages")){
                auto messages=conversation.count(session)?conversation[session]:Json::array();
                messages.push_back({{"role","user"},{"content",spec.at("prompt")}});spec["messages"]=messages;action["spec"]=spec;
            }
            if(previous_llm.count(session)){
                invoke(state.worker,{{"op","restore"},{"state_id",previous_llm[session]}},row);
                action["previous_state_id"]=previous_llm[session];
                previous_llm.erase(session);
            }
            conversation[session]=spec["messages"];
        }
        auto prepared=invoke(state.worker,action,row);
        if(state.stage==0)encoding.erase(id);
        state.state=prepared.at("state_id").get<std::string>();state.progress=prepared.value("progress",Json::object());
        row["prepare"]=prepared;
        row["generation"]=spec;
    };
    try{while(terminal<rows.size()) {
        check();const double instant=now();Json ready=Json::array();
        for(auto& row:rows) {
            if(row.at("status")!="pending")continue;
            if(row.contains("after") && !row.value("released",false)) {
                auto previous=std::find_if(rows.begin(),rows.end(),[&](const Json& r){return r["request_id"]==row["after"];});
                if(previous->at("status")=="failed" || previous->at("status")=="blocked") {row["status"]="blocked";row["reason"]="predecessor did not complete";++terminal;continue;}
                if(previous->at("status")!="completed")continue;
                row["arrival_ms"]=number_field(*previous,"finish_ms")+number_field(row,"think_ms");row["deadline_ms"]=number_field(row,"arrival_ms")+number_field(row,"slo_ms");row["ready_since_ms"]=row["arrival_ms"];row["released"]=true;
            }
            if(number_field(row,"arrival_ms")>instant)continue;
            const auto id=row.at("request_id").get<std::string>();auto state=live.find(id);
            if(state!=live.end()&&!state->second.decode_ticket.empty()){
                auto& current=state->second;
                auto decoded=invoke(current.worker,{{"op","decode_poll"},{"ticket",current.decode_ticket}},row);
                if(!decoded.value("ready",false))continue;
                current.decoded=decoded;current.decode_ticket.clear();decoding.erase(current.worker);
            }
            if(state!=live.end()&&state->second.progress.value("done",false)&&decoding.count(state->second.worker))continue;
            if(state==live.end()||state->second.suspended){
                const auto target=state==live.end()?stages(row).at(0).value("worker",row.at("worker").get<std::string>()):state->second.worker;
                int resident=0;bool evictable=false;
                for(const auto& item:live)if(item.second.worker==target&&!item.second.suspended){
                    ++resident;if(item.first!=id&&item.second.decode_ticket.empty())evictable=true;
                }
                if(resident>=resident_limit&&!evictable)continue;
            }
            if(state==live.end() && phase.value("text_prefetch",0)>0){
                const auto worker=stages(row).at(0).value("worker",row.at("worker").get<std::string>());
                if(!encoding.count(id)){
                    if(encoding.size()>=static_cast<std::size_t>(int_field(phase,"text_prefetch",0,0)))continue;
                    auto spec=stages(row).at(0).value("spec",row.at("spec"));spec["request_id"]=id+"-s0";
                    auto submitted=invoke(worker,{{"op","encode_start"},{"spec",spec}},row);
                    encoding[id]=submitted.at("ticket").get<std::string>();row["encode_submit_ms"]=now();
                }
                auto encoded=invoke(worker,{{"op","encode_poll"},{"ticket",encoding[id]}},row);
                if(!encoded.value("ready",false))continue;
                if(!row.contains("encode_ready_ms")){row["encode_ready_ms"]=now();row["ready_since_ms"]=row["encode_ready_ms"];}
            }
            auto view=row;
            view["arrived"]=true;view["gpu_ready"]=true;view["ready"]=true;view["release_ms"]=row["arrival_ms"];
            view["started"]=state!=live.end();view["remaining_ms"]=remaining(row,state==live.end()?nullptr:&state->second,costs);
            view["remaining_total_ms"]=view["remaining_ms"];
            const auto list=stages(row);const std::size_t si=state==live.end()?0:state->second.stage;
            double cumulative=0,total=0;
            for(std::size_t n=0;n<list.size();++n){const double cost=number_field(list[n],"cost_ms",number_field(row,"service_ms")/list.size());total+=cost;if(n<=si)cumulative+=cost;}
            view["stage_deadline_ms"]=number_field(row,"arrival_ms")+number_field(row,"slo_ms")*(total>0?cumulative/total:double(si+1)/list.size());
            view["action"]=state==live.end()?"prepare":state->second.progress.value("done",false)?"decode":"denoise";
            const double per=number_field(costs,"block_ms",number_field(costs,"step_ms",number_field(row,"service_ms",100)/int_field(row.at("spec"),"steps",1)));
            const bool atomic=view["action"]!="denoise";
            const double base=atomic?number_field(costs,view["action"]=="prepare"?"prepare_ms":"finish_ms",per):per;
            view["action_ms_by_quantum"]={{"1",base},{"2",atomic?base:2*per},{"4",atomic?base:4*per},{"8",atomic?base:8*per},{"step",atomic?base:number_field(costs,"step_ms",per)}};
            ready.push_back(view);
        }
        if(ready.empty()){if(terminal==rows.size())break;wait_until(monotonic_seconds()+.001);continue;}
        const auto selected=select_action(ready,now(),policy,costs,active,phase.value("quantum",Json(1)));
        const auto id=selected.at("request_id").get<std::string>();
        auto& row=*std::find_if(rows.begin(),rows.end(),[&](const Json& r){return r["request_id"]==id;});
        active=id;
        try {
            auto it=live.find(id);
            if(it==live.end()) {
                const auto list=stages(row);const auto& stage=list.at(0);
                Live state;
                if(!row.contains("start_ms"))row["start_ms"]=now();
                prepare(state,row,stage);live[id]=state;
            } else {
                auto& state=it->second;
                if(state.suspended) {
                    make_room(id,state.worker);
                    invoke(state.worker,{{"op","restore"},{"state_id",state.state},{"snapshot_id",state.snapshot}},row);state.suspended=false;
                }
                if(state.progress.value("done",false)) {
                    if(phase.value("decode_mode","sync")=="async"&&state.decoded.is_null()){
                        const auto started=invoke(state.worker,{{"op","decode_start"},{"state_id",state.state},{"save_latents",phase.value("save_latents",false)}},row);
                        state.decode_ticket=started.at("ticket").get<std::string>();decoding[state.worker]=id;
                        continue;
                    }
                    auto result=state.decoded.is_null()?invoke(state.worker,{{"op","finish"},{"state_id",state.state},{"save_latents",phase.value("save_latents",false)}},row):state.decoded;
                    state.decoded=nullptr;
                    add_artifacts(record,result,work,phase.value("artifact_dependencies",Json::array()));
                    if(!row.contains("outputs"))row["outputs"]=Json::array();
                    row["outputs"].push_back(result);
                    if(row.contains("reference")) {
                        auto quality=invoke(state.worker,{{"op","evaluate"},{"output",result.at("output").at("path")},{"reference",row.at("reference")}},row);
                        row["quality"]=quality;
                    }
                    if(workers.backend(state.worker)=="mistral"){
                        const auto session=row.value("session_id",id);
                        previous_llm[session]=state.state;
                        conversation[session].push_back({{"role","assistant"},{"content",result.at("output").at("text")}});
                        invoke(state.worker,{{"op","snapshot"},{"state_id",state.state},{"mode","hot"}},row);
                    }else invoke(state.worker,{{"op","release"},{"state_id",state.state}},row);
                    const auto list=stages(row);
                    if(++state.stage<list.size()) {
                        auto stage=list[state.stage];Json spec=stage.value("spec",row.at("spec"));
                        if(stage.value("prompt_from_previous",false)) {
                            if(!result.value("output",Json::object()).contains("text"))throw std::runtime_error("previous stage did not return text");
                            spec["prompt"]=result.at("output").at("text");
                        }
                        stage["spec"]=spec;state.suspended=true;
                        prepare(state,row,stage);state.suspended=false;state.transitions.clear();
                    } else {row["status"]="completed";row["finish_ms"]=now();++terminal;live.erase(it);}
                } else {
                    const auto transitions=row.value("transitions",phase.value("transitions",Json::array()));
                    for(std::size_t t=0;t<transitions.size();++t)if(!state.transitions.count(t) && state.progress.value("step",0)>=int_field(transitions[t],"at_step",0,0)) {
                        const auto& change=transitions[t];const auto target=change.value("target_worker",state.worker);
                        Json action={{"op","snapshot"},{"state_id",state.state},{"mode",change.value("mode","hot")}};
                        if(target!=state.worker) {
                            action["migration_id"]=id+"-"+std::to_string(t);action["target_device"]=change.at("target_device");
                            action["path"]=(cell_dir/"artifacts"/(id+"-migration-"+std::to_string(t)+".pt")).string();
                        }
                        auto saved=invoke(state.worker,action,row);
                        if(saved.contains("path"))add_artifacts(record,{{"output",saved}},work,Json::array({"migration_commit"}));
                        if(target==state.worker)invoke(state.worker,{{"op","restore"},{"state_id",state.state},{"snapshot_id",saved.value("snapshot_id",state.state)}},row);
                        else {
                            make_room(id,target);
                            auto restored=invoke(target,{{"op","restore"},{"path",saved.at("path")},{"sha256",saved.at("sha256")},{"state_id",state.state},{"migration_id",action["migration_id"]}},row);
                            auto released=invoke(state.worker,{{"op","migration_release"},{"migration_id",action["migration_id"]},{"acknowledgement",restored.at("acknowledgement")}},row);
                            auto committed=invoke(target,{{"op","restore"},{"state_id",restored.at("state_id")},{"migration_id",action["migration_id"]},{"receipt",released.at("receipt")}},row);
                            state.worker=target;state.state=restored.at("state_id").get<std::string>();state.progress=committed.value("progress",state.progress);
                            for(auto& artifact:record["artifacts"])if((work/artifact.at("path").get<std::string>()).string()==saved.at("path"))artifact["dependencies"]=Json::array();
                        }
                        state.transitions.insert(t);
                    }
                    Json action={{"op","advance"},{"state_id",state.state},{"quantum",selected.at("quantum")}};
                    const auto unit=phase.value("unit","step");
                    if(selected.at("quantum")!="step")action[unit=="block"?"blocks":"steps"]=selected.at("quantum");
                    auto result=invoke(state.worker,action,row);state.progress=result.at("progress");
                }
            }
            row["ready_since_ms"]=now();
        }catch(const std::exception& error) {
            row["status"]="failed";row["error"]=error.what();row["finish_ms"]=now();++terminal;
            record["failures"].push_back({{"request_id",id},{"error",error.what()}});
            // A failed worker invalidates its live states. Stop this cell rather
            // than silently restarting and fabricating successful requests.
            throw;
        }
    }}catch(...){
        for(auto& row:rows){if(row["status"]=="pending")row["status"]="interrupted";if(!row.contains("finish_ms"))row["finish_ms"]=now();row["observation_ms"]=now();}
        record["requests"].insert(record["requests"].end(),rows.begin(),rows.end());throw;
    }
    return {{"requests",rows},{"metrics",request_summary(rows)},{"elapsed_ms",now()}};
}

// A real clock consumer independent of the model call. It records dispatch
// timing, not browser/display latency. Only bounded samples are retained.
class Player {
public:
    Player(double fps,int buffer):fps_(fps),limit_(buffer+1),thread_([this]{loop();}){}
    ~Player(){stop();}
    bool room(){std::lock_guard<std::mutex> lock(mutex_);return queue_.size()+(current_?1:0)<static_cast<std::size_t>(limit_);}
    bool empty(){std::lock_guard<std::mutex> lock(mutex_);return queue_.empty()&&!current_;}
    void accept(int frames,int version,double ready){std::lock_guard<std::mutex> lock(mutex_);if(frames<=0)throw std::invalid_argument("chunk has no frames");queue_.push_back({frames,version,ready});peak_=std::max(peak_,queue_.size()+(current_?1:0));}
    void flush(){std::lock_guard<std::mutex> lock(mutex_);for(const auto& chunk:queue_)dropped_+=chunk.frames;queue_.clear();}
    Json result(){std::lock_guard<std::mutex> lock(mutex_);return {{"frames_played",played_},{"frames_dropped",dropped_},{"peak_chunks",peak_},{"late_frames",late_},{"samples",samples_},{"sample_scope","first 128 dispatched frames; all counts retained"},{"measurement_scope","wall-clock frame dispatch; not a browser or physical display"}};}
    void stop(){{std::lock_guard<std::mutex> lock(mutex_);stopping_=true;}if(thread_.joinable())thread_.join();}
private:
    struct Chunk{int frames,version;double ready;};
    double fps_;int limit_;std::mutex mutex_;std::deque<Chunk> queue_;bool current_=false,stopping_=false;
    std::uint64_t played_=0,dropped_=0,late_=0;std::size_t peak_=0;Json samples_=Json::array();std::thread thread_;
    void loop(){
        double target=0;
        for(;;){
            Chunk chunk{};
            {std::lock_guard<std::mutex> lock(mutex_);if(stopping_)return;if(!queue_.empty()){chunk=queue_.front();queue_.pop_front();current_=true;}}
            if(!chunk.frames){std::this_thread::sleep_for(std::chrono::milliseconds(1));continue;}
            target=std::max(target,chunk.ready);
            for(int i=0;i<chunk.frames;++i){
                while(monotonic_seconds()<target){{std::lock_guard<std::mutex> lock(mutex_);if(stopping_)return;}wait_until(target);}
                const double observed=monotonic_seconds();
                {std::lock_guard<std::mutex> lock(mutex_);++played_;if(observed-target>1/fps_)++late_;if(samples_.size()<128)samples_.push_back({{"version",chunk.version},{"target_monotonic_s",target},{"observed_monotonic_s",observed}});}
                target+=1/fps_;
            }
            while(monotonic_seconds()<target){{std::lock_guard<std::mutex> lock(mutex_);if(stopping_)return;}wait_until(target);}
            {std::lock_guard<std::mutex> lock(mutex_);current_=false;}
        }
    }
};

Json stream(const Json& phase,Workers& workers,Budget& budget,Json& record,const fs::path& work){
    auto rows=study_workload(phase);
    const auto policy=phase.value("prompt_policy","fifo");
    if(policy!="fifo" && policy!="latest_only" && policy!="latest_only_flush")throw std::invalid_argument("unknown prompt policy");
    const int buffer=int_field(phase,"buffer_chunks",0,0);if(buffer>2)throw std::invalid_argument("buffer_chunks must be 0, 1 or 2");
    const double fps=number(phase.value("fps",Json(16)),"fps",true),origin=monotonic_seconds();
    const int chunks=int_field(phase,"chunks",2,1);
    std::map<std::string,std::unique_ptr<Player>> players;std::map<std::string,std::string> states;
    std::map<std::string,std::deque<Json>> pending;std::map<std::string,int> produced,versions;std::map<std::string,std::size_t> cursors;
    Json events=Json::array();std::size_t cursor=0,finished=0;
    for(auto& row:rows){row["observation_id"]=phase.at("_observation_id");row["phase"]=phase.at("id");row["attempt"]=phase.at("_attempt");row["cell"]=phase.at("_cell");row["order"]=phase.at("_order");row["quality"]={{"status","unverified"},{"passed",false}};}
    try{while(finished<rows.size()){
        if(cancelled)throw std::runtime_error("run cancelled");
        budget.check();const double now=(monotonic_seconds()-origin)*1000;
        bool advanced=false;
        for(std::size_t offset=0;offset<rows.size();++offset){
            const std::size_t i=(cursor+offset)%rows.size();auto& row=rows[i];
            if(row["status"]!="pending" || number_field(row,"arrival_ms")>now)continue;
            const auto id=row.at("request_id").get<std::string>(),worker=row.at("worker").get<std::string>();
            auto& runtime=workers.get(worker);
            row["worker_pids"][worker]=runtime.process->pid();
            if(!states.count(id)){
                auto spec=row.at("spec");spec["blocks"]=chunks;spec["steps"]=chunks;
                auto prepared=runtime.call({{"op","prepare"},{"spec",spec}},budget.deadline());
                states[id]=prepared.at("state_id").get<std::string>();players[id]=std::make_unique<Player>(fps,buffer);row["start_ms"]=now;
            }
            const auto updates=row.value("updates",Json::array());
            while(cursors[id]<updates.size() && number_field(updates[cursors[id]],"arrival_ms")<=now){
                Json event=updates[cursors[id]++];event["request_id"]=id;event["status"]="pending";
                if(policy!="fifo")while(!pending[id].empty()){auto old=pending[id].front();pending[id].pop_front();old["status"]="superseded_before_encoding";events.push_back(old);}
                pending[id].push_back(event);
            }
            if(!pending[id].empty()){
                auto update=pending[id].front();pending[id].pop_front();
                const int version=update.value("version",versions[id]+1);
                runtime.call({{"op","update_prompt"},{"state_id",states[id]},{"prompt",update.at("prompt")},{"prompt_version",version}},budget.deadline());
                versions[id]=version;update["status"]="applied";update["applied_ms"]=(monotonic_seconds()-origin)*1000;update["semantic_status"]="unverified";events.push_back(update);
                if(policy=="latest_only_flush")players[id]->flush();
            }
            if(!players[id]->room())continue;
            auto result=runtime.call({{"op","advance"},{"state_id",states[id]},{"steps",1}},budget.deadline());
            const int frames=int_field(result,"frames",1,1);
            players[id]->accept(frames,versions[id],monotonic_seconds());
            add_artifacts(record,result,work,phase.value("artifact_dependencies",Json::array({"semantic_blind_review"})));
            row["model_ms"]=number_field(row,"model_ms")+number_field(result,"model_ms");
            ++produced[id];cursor=(i+1)%rows.size();advanced=true;
            if(produced[id]>=chunks){
                auto output=runtime.call({{"op","finish"},{"state_id",states[id]}},budget.deadline());
                // LongLive's finish references the last registered chunk.
                if(output.value("output_scope","")!="last_chunk; controller retains per-chunk artifacts")
                    add_artifacts(record,output,work,phase.value("artifact_dependencies",Json::array({"semantic_blind_review"})));
                row["outputs"]=Json::array({output});
                runtime.call({{"op","release"},{"state_id",states[id]}},budget.deadline());
                row["status"]="completed";row["finish_ms"]=(monotonic_seconds()-origin)*1000;++finished;
            }
            break;
        }
        if(!advanced)wait_until(monotonic_seconds()+.001);
    }
    const double drain=monotonic_seconds()+number_field(phase,"playback_drain_seconds",chunks/fps*16+1);
    while(monotonic_seconds()<drain){if(cancelled)throw std::runtime_error("run cancelled");budget.check();bool empty=true;for(auto& item:players)if(!item.second->empty())empty=false;if(empty)break;wait_until(monotonic_seconds()+.005);}
    bool drained=true;
    for(auto& row:rows)if(!players.at(row.at("request_id").get<std::string>())->empty()){
        drained=false;row["status"]="failed";row["error"]="playback drain timed out";
    }
    if(!drained)throw std::runtime_error("playback drain timed out");
    }catch(const std::exception& error){
        const double elapsed=(monotonic_seconds()-origin)*1000;
        for(auto& row:rows){
            const auto id=row.at("request_id").get<std::string>();
            if(players.count(id)){players[id]->stop();row["playback"]=players[id]->result();}
            if(row["status"]=="pending")row["status"]="interrupted";
            row["observation_ms"]=elapsed;if(!row.contains("finish_ms"))row["finish_ms"]=elapsed;
            row["interruption"]=error.what();
        }
        record["requests"].insert(record["requests"].end(),rows.begin(),rows.end());
        record["prompt_events"]=events;
        throw;
    }
    for(auto& row:rows){const auto id=row.at("request_id").get<std::string>();players[id]->stop();row["playback"]=players[id]->result();row["quality"]={{"status","unverified"},{"passed",false}};}
    for(auto& item:pending)for(auto event:item.second){event["status"]="pending_at_stop";events.push_back(event);}
    return {{"requests",rows},{"metrics",request_summary(rows)},{"prompt_events",events},{"semantic_status","unverified"},{"elapsed_ms",(monotonic_seconds()-origin)*1000}};
}

Json operations(const Json& phase,Workers& workers,Budget& budget,Json& record,const fs::path& work,const fs::path& cell_dir){
    const auto names=phase.value("workers",Json::array({phase.value("worker","model")}));
    if(!names.is_array() || names.empty())throw std::invalid_argument("action phase requires workers");
    const auto actions=phase.at("actions");if(!actions.is_array() || actions.empty())throw std::invalid_argument("SP phase requires actions");
    Json rows=Json::array();
    for(auto action:actions){
        budget.check();if(cancelled)throw std::runtime_error("run cancelled");
        std::vector<std::future<Json>> pending;
        // Create children on the controlling thread; each call then owns one
        // independent pipe while collective ranks progress concurrently.
        std::vector<Worker*> group;for(const auto& name:names)group.push_back(&workers.get(name.get<std::string>()));
        for(std::size_t rank=0;rank<group.size();++rank){
            auto message=action;message["op"]=message.value("op","distributed");message["rank"]=rank;message["world_size"]=group.size();
            if(message.contains("per_rank")){message.merge_patch(message.at("per_rank").at(rank));message.erase("per_rank");}
            for(const auto* field:{"output_path","output","reference","path"})if(message.contains(field)&&message[field].is_string()){
                fs::path path=message[field].get<std::string>();if(path.is_relative())message[field]=(cell_dir/"artifacts"/std::to_string(rank)/path).string();
            }
            pending.push_back(std::async(std::launch::async,[&,rank,message]{return group[rank]->call(message,budget.deadline());}));
        }
        Json results=Json::array();std::exception_ptr error;
        for(std::size_t rank=0;rank<pending.size();++rank)try{auto result=pending[rank].get();result["worker_pid"]=group[rank]->process->pid();add_artifacts(record,result,work,phase.value("artifact_dependencies",Json::array()));results.push_back(result);}catch(...){if(!error)error=std::current_exception();}
        if(error)std::rethrow_exception(error);
        rows.push_back({{"action",action},{"ranks",results}});
    }
    return {{"operations",rows},{"requests",Json::array()},{"metrics",{{"operations",rows.size()},{"ranks",names.size()}}}};
}

Json publish(Json& record,const Json& request,const fs::path& work){
    try{
    for(auto& artifact:record["artifacts"]){
        const auto path=work/artifact.at("path").get<std::string>();
        if(!fs::exists(path))continue; // The finalizer verifies prior deletion receipts.
        const auto hash=file_hash(path);
        if(artifact.contains("sha256") && artifact.at("sha256")!=hash)throw std::runtime_error("artifact changed before finalization");
        artifact["sha256"]=hash;
    }
    write(work/"run.json",record);
    Json definition={{"results",{{"backend","results"},{"python",request.value("report_python",request.value("python","python3"))}}}};
    Workers reporters(definition,request,work/"reporter");
    try{
        Json action={{"op","finalize"},{"work_dir",work.string()},{"prune",request.value("prune",true)}};
        if(request.contains("results_dir"))action["output"]=request.at("results_dir");
        auto result=reporters.get("results").call(action,monotonic_seconds()+120);
        reporters.stop();return result;
    }catch(const std::exception& error){reporters.stop();return {{"status","deferred"},{"reason",error.what()}};}
    }catch(const std::exception& error){write(work/"run.json",record);return {{"status","deferred"},{"reason",error.what()}};}
}

// Executes one phase's matrix while owning every child process. The caller
// keeps budget admission and settlement together and receives verified cleanup.
bool execute_cells(const Json& phase,const Json& request,const Json& devices,
                   Budget& budget,Json& record,Json& attempt) {
    const fs::path work=request.at("work_dir").get<std::string>();
    const auto id=phase.at("id").get<std::string>();
    const int attempt_number=attempt.at("attempt");
    const auto attempt_dir=work/"attempts"/id/("attempt-"+std::to_string(attempt_number));
    bool cleanup=true;std::string error;std::vector<int> attempt_pids;
    auto cells=expand_matrix(phase);
    std::vector<std::size_t> order;for(std::size_t i=0;i<cells.size();++i)order.push_back(i);
    if(phase.value("paired",false))for(std::size_t i=cells.size();i>0;--i)order.push_back(i-1);
    for(std::size_t position=0;position<order.size();++position){
        auto cell=cells[order[position]];const auto cell_dir=attempt_dir/("cell-"+std::to_string(position));
        cell["_observation_id"]=id+"/"+std::to_string(attempt_number)+"/"+std::to_string(position);
        cell["_attempt"]=attempt_number;cell["_cell"]=order[position];cell["_order"]=position<cells.size()?"forward":"reverse";
        Json definitions=request.at("config").at("workers");
        if(cell.contains("worker_config"))definitions[cell.value("worker","model")]["config"].merge_patch(cell.at("worker_config"));
        Workers workers(definitions,request,cell_dir);
        double last_observation=0;
        std::mutex heartbeat_mutex;
        workers.heartbeat([&]{
            // Collective ranks call concurrently. One observation owns the
            // probe paths, and its timestamp must not race with other ranks.
            std::lock_guard<std::mutex> guard(heartbeat_mutex);
            if(cancelled)throw std::runtime_error("run cancelled");
            budget.check();
            if(!devices.empty()&&monotonic_seconds()-last_observation>=2){
                last_observation=monotonic_seconds();
                auto observed=gpu_observation(request,cell_dir,devices,workers.pids(),std::min(budget.deadline(),monotonic_seconds()+3));
                if(observed.value("status","")!="exclusive_observed")throw std::runtime_error("GPU isolation lost: "+observed.dump());
            }
        });
        try{
            const auto kind=cell.value("kind","online");
            Json result=cell.contains("actions")?operations(cell,workers,budget,record,work,cell_dir):kind=="streaming"?stream(cell,workers,budget,record,work):replay(cell,workers,budget,record,work,cell_dir);
            result["cell"]=order[position];result["order"]=position<cells.size()?"forward":"reverse";result["status"]="completed";
            for(auto row:result.value("requests",Json::array())){row["phase"]=id;row["attempt"]=attempt_number;row["cell"]=order[position];row["order"]=result["order"];
                row["observation_id"]=id+"/"+std::to_string(attempt_number)+"/"+std::to_string(position);
                row["observation_ms"]=result.value("elapsed_ms",number_field(row,"finish_ms"));record["requests"].push_back(row);}
            attempt["cells"].push_back(result);
            workers.shutdown(std::min(budget.deadline(),monotonic_seconds()+5));
        }catch(const std::exception& failure){error=failure.what();attempt["cells"].push_back({{"cell",order[position]},{"status","failed"},{"error",error}});}
        const auto pids=workers.pids();attempt_pids.insert(attempt_pids.end(),pids.begin(),pids.end());
        cleanup=workers.stop()&&cleanup;
        if(!error.empty() || !cleanup)break;
    }
    const auto final_gpu=gpu_observation(request,attempt_dir,devices,attempt_pids);
    const auto observed_status=final_gpu.value("status","");
    const bool observed=observed_status=="exclusive_observed" || observed_status=="busy";
    cleanup=cleanup && observed && final_gpu.value("owned",Json::array()).empty();
    if(observed && !final_gpu.value("foreign",Json::array()).empty() && error.empty())
        error="GPU isolation lost before final cleanup";
    attempt["final_occupancy"]=final_gpu;
    attempt["status"]=error.empty() && cleanup?"completed":"failed";
    if(!error.empty()){attempt["error"]=error;record["failures"].push_back({{"phase",id},{"attempt",attempt_number},{"error",error}});}
    return cleanup;
}

void validate(const Json& config){
    if(config.value("schema","")!="ditserve.study.v1")throw std::invalid_argument("schema must be ditserve.study.v1");
    segment(config.at("experiment").get<std::string>());
    if(!config.at("workers").is_object() || config.at("workers").empty())throw std::invalid_argument("workers must be a nonempty object");
    for(auto it=config.at("workers").begin();it!=config.at("workers").end();++it){
        segment(it.key());const auto& worker=it.value();
        if(worker.at("backend").get<std::string>().empty())throw std::invalid_argument("backend name must be nonempty");
        if(worker.contains("config")&&!worker.at("config").is_object())throw std::invalid_argument("worker config must be an object");
        if(worker.contains("environment")){
            if(!worker.at("environment").is_object())throw std::invalid_argument("worker environment must be an object");
            for(auto env=worker.at("environment").begin();env!=worker.at("environment").end();++env){
                if(env.key()=="CUDA_VISIBLE_DEVICES"||env.key()=="PYTHONPATH"||env.key()=="HF_HUB_OFFLINE"||env.key()=="TRANSFORMERS_OFFLINE")
                    throw std::invalid_argument("worker environment cannot override device, frozen sources or offline loading");
                env.value().get<std::string>();
            }
        }
    }
    const auto phases=config.at("phases");if(!phases.is_array() || phases.empty())throw std::invalid_argument("phases must be a nonempty array");
    std::set<std::string> seen;
    const std::set<std::string> kinds={"profile","online","state","sp","streaming","session"};
    for(const auto& phase:phases){
        const auto id=segment(phase.at("id").get<std::string>());if(seen.count(id))throw std::invalid_argument("duplicate phase");
        if(!kinds.count(phase.value("kind","online")))throw std::invalid_argument("unknown executor kind");
        for(const auto& dependency:phase.value("requires",Json::array())){
            const auto parent=dependency.is_string()?dependency.get<std::string>():dependency.at("phase").get<std::string>();
            if(!seen.count(parent))throw std::invalid_argument("phase dependency must precede its consumer");
            const auto condition=dependency.is_object()?dependency.value("requires","completed"):"completed";
            if(condition!="completed"&&condition!="qualified")throw std::invalid_argument("dependency requires completed or qualified");
        }
        seen.insert(id);
        phase.value("paired",false);
        const double admission_wait=number_field(phase,"admission_wait_seconds",0);
        const double admission_poll=number_field(phase,"admission_poll_seconds",15);
        if(admission_wait>86400 || admission_poll<=0 || admission_poll>60)
            throw std::invalid_argument("admission wait must be <=24h and poll in (0,60] seconds");
        evaluate_gates(Json::object(),phase.value("gates",Json::array()));
        for(const auto& cell:expand_matrix(phase)){
            if(cell.at("id")!=id || !kinds.count(cell.value("kind","online")))throw std::invalid_argument("matrix cannot change phase identity or introduce an unknown executor");
            if(cell.contains("worker_config")&&!cell.at("worker_config").is_object())throw std::invalid_argument("worker_config must be an object");
            auto known=[&](const std::string& worker){if(!config.at("workers").contains(worker))throw std::invalid_argument("unknown worker: "+worker);};
            if(cell.contains("worker_config")){
                known(cell.value("worker","model"));
                for(const auto* field:{"device","gpu_uuid","device_id","backend","work_dir"})
                    if(cell.at("worker_config").contains(field))throw std::invalid_argument("matrix cannot change worker device or identity");
            }
            if(cell.contains("actions")){
                if(!cell.at("actions").is_array()||cell.at("actions").empty())throw std::invalid_argument("actions must be a nonempty array");
                std::set<std::string> ranks;
                for(const auto& worker:cell.value("workers",Json::array({cell.value("worker","model")}))){
                    const auto name=worker.get<std::string>();known(name);
                    if(!ranks.insert(name).second)throw std::invalid_argument("each rank requires a distinct worker");
                }
            }else{
                const auto rows=study_workload(cell);if(rows.empty())throw std::invalid_argument("phase workload must be nonempty");
                for(const auto& row:rows){
                    known(row.at("worker").get<std::string>());
                    const auto list=stages(row);
                    if(!list.is_array()||list.empty())throw std::invalid_argument("request stages must be nonempty");
                    for(const auto& stage:list){known(stage.value("worker",row.at("worker").get<std::string>()));if(stage.contains("spec")&&!stage.at("spec").is_object())throw std::invalid_argument("stage spec must be an object");}
                    double arrival=-1;int version=0;
                    for(const auto& update:row.value("updates",Json::array())){
                        const double next=number_field(update,"arrival_ms");const int next_version=int_field(update,"version",version+1,1);
                        if(next<arrival||next_version<=version)throw std::invalid_argument("prompt updates must have ordered arrivals and increasing versions");
                        update.at("prompt").get<std::string>();arrival=next;version=next_version;
                    }
                }
            }
        }
    }
}
}

Json run_study(const Json& original_request){
    Signals signals;
    auto request=original_request;const auto config=request.at("config");validate(config);
    const auto work=real_path(request.at("work_dir").get<std::string>());
    request["work_dir"]=work.string();
    const bool resume=request.value("resume",false);
    if(fs::exists(work)&&!fs::is_empty(work)&&!fs::exists(work/"run.json"))throw std::invalid_argument("nonempty work directory has no run manifest");
    fs::create_directories(work);RunLock lock(work/"run.lock");
    const auto run_path=work/"run.json";
    if(fs::exists(run_path)!=resume)throw std::invalid_argument(resume?"resume needs an existing run":"run already exists; choose a new run or --resume");
    const auto config_hash=digest(config.dump());
    Json record=resume?read(run_path):Json{{"schema","ditserve.run.v1"},{"run_id",segment(request.at("run_id").get<std::string>())},
        {"experiment",config.at("experiment")},{"config",config},{"config_sha256",config_hash},{"status","prepared"},
        {"phases",Json::array()},{"requests",Json::array()},{"failures",Json::array()},{"artifacts",Json::array()}};
    if(record.at("config_sha256")!=config_hash)throw std::invalid_argument("frozen config differs; use a new run");
    if(record.at("run_id")!=segment(request.at("run_id").get<std::string>()))throw std::invalid_argument("resume run_id differs from the frozen run");
    const auto ledger=real_path(request.at("ledger").get<std::string>());
    if(resume && (!record.contains("budget_ledger") || real_path(record.at("budget_ledger").get<std::string>())!=ledger))
        throw std::invalid_argument("resume budget ledger differs from the frozen run");
    if(resume && record.value("status","")=="completed"){
        request["frozen_python"]=(work/"sources"/"python").string();
        record["archive"]=publish(record,request,work);return record;
    }
    record["source"]=freeze_sources(request,work,resume);request["frozen_python"]=(work/"sources"/"python").string();
    write(run_path,record);
    const auto limits=config.value("budget",Json::object());BudgetLimits budget_limits;
    if(limits.contains("limit_gpu_seconds") && !limits.at("limit_gpu_seconds").is_null())budget_limits.total_gpu_seconds=number_field(limits,"limit_gpu_seconds");
    budget_limits.repair_reserve_gpu_seconds=number_field(limits,"repair_reserve_gpu_seconds");budget_limits.carry=limits.value("carry","none");
    for(const auto& phase:config.at("phases"))budget_limits.phase_order.push_back(phase.at("id").get<std::string>());
    const auto stage_limits=limits.value("stage_limits",Json::object());
    for(auto it=stage_limits.begin();it!=stage_limits.end();++it)budget_limits.phase_gpu_seconds[it.key()]=number(it.value(),"stage limit");
    const auto ledger_relative=ledger.lexically_relative(work);
    if(ledger==work || (!ledger_relative.empty() && *ledger_relative.begin()!=".."))throw std::invalid_argument("cumulative ledger must live outside the run directory");
    record["budget_ledger"]=ledger.string();
    auto budget_owner=std::make_unique<Budget>(ledger,budget_limits);auto& budget=*budget_owner;
    Json devices=Json::array();
    for(auto it=config.at("workers").begin();it!=config.at("workers").end();++it){
        const auto backend=it.value().at("backend").get<std::string>();
        const auto device=it.value().value("device",it.value().value("config",Json::object()).value("device","cpu"));
        const bool cpu=backend=="mock"||(backend=="evaluation"&&device=="cpu");
        if(!cpu && !request.value("allow_gpu",false))throw std::invalid_argument("real model execution requires --allow-gpu");
        if(!cpu){
            if(!limits.contains("limit_gpu_seconds"))throw std::invalid_argument("real runs require an explicit budget, including null for unlimited");
            const auto uuid=it.value().at("gpu_uuid").get<std::string>();
            if(uuid.compare(0,4,"GPU-")!=0 || uuid.size()!=40)throw std::invalid_argument("real workers require a full GPU UUID");
            if(std::find(devices.begin(),devices.end(),Json(uuid))==devices.end())devices.push_back(uuid);
        }
    }
    std::map<std::string,Json> completed;
    for(const auto& attempt:record["phases"])if(attempt.value("status","")=="completed")completed[attempt.at("id").get<std::string>()]=attempt;
    const auto only=request.value("phase","");bool matched=only.empty(),failed=false;
    for(const auto& phase:config.at("phases")){
        const auto id=phase.at("id").get<std::string>();if(!only.empty() && id!=only)continue;matched=true;
        if(completed.count(id))continue;
        bool ready=true;
        for(const auto& dependency:phase.value("requires",Json::array())){
            const auto parent=dependency.is_string()?dependency.get<std::string>():dependency.at("phase").get<std::string>();
            const auto condition=dependency.is_object()?dependency.value("requires","completed"):"completed";
            if(condition!="completed" && condition!="qualified")throw std::invalid_argument("dependency requires completed or qualified");
            ready=ready&&completed.count(parent) && (condition=="completed" || completed[parent].at("qualification").value("qualified",false));
        }
        if(!ready){record["status"]="incomplete";record["blocked_phase"]=id;break;}
        int attempt_number=1;for(const auto& previous:record["phases"])if(previous.at("id")==id)++attempt_number;
        const auto attempt_dir=work/"attempts"/id/("attempt-"+std::to_string(attempt_number));fs::create_directories(attempt_dir);
        Json attempt={{"id",id},{"attempt",attempt_number},{"status","prepared"},{"cells",Json::array()}};
        const double waiting_started=monotonic_seconds();
        const double waiting_limit=number_field(phase,"admission_wait_seconds",0);
        const double waiting_poll=number_field(phase,"admission_poll_seconds",15);
        const double waiting_deadline=waiting_started+waiting_limit;
        auto observation=gpu_observation(request,attempt_dir,devices,{},waiting_limit>0?std::min(waiting_deadline,waiting_started+15):0);
        int idle_observations=0;
        while(waiting_limit>0){
            if(cancelled || monotonic_seconds()>=waiting_deadline)break;
            const bool idle=observation.value("status","")=="exclusive_observed" && observation.value("foreign",Json::array()).empty();
            idle_observations=idle?idle_observations+1:0;
            if(idle_observations>=2)break;
            const double elapsed=monotonic_seconds()-waiting_started;
            record["status"]="waiting_resource";
            record["admission"]={{"phase",id},{"observation",observation},{"wait_seconds",elapsed},{"gpu_time_charged",false}};
            write(run_path,record);
            if(cancelled || elapsed>=waiting_limit)break;
            const double next=std::min(waiting_started+waiting_limit,monotonic_seconds()+waiting_poll);
            while(!cancelled && monotonic_seconds()<next)wait_until(next);
            if(cancelled)break;
            observation=gpu_observation(request,attempt_dir,devices,{},std::min(waiting_deadline,monotonic_seconds()+15));
        }
        if(waiting_limit>0)record["admission_wait_seconds"]=monotonic_seconds()-waiting_started;
        if(cancelled || (waiting_limit>0 && idle_observations<2)){
            record["status"]="blocked";
            record["admission"]={{"phase",id},{"error",cancelled?"cancelled while waiting for isolated GPU":"GPU admission wait timed out"},{"observation",observation}};
            write(run_path,record);break;
        }
        if(observation.value("status","")!="exclusive_observed" || !observation.value("foreign",Json::array()).empty()){
            record["status"]="blocked";record["admission"]=observation;write(run_path,record);break;
        }
        const double watchdog=number(phase.value("watchdog_seconds",limits.value("watchdog_seconds",Json(300))),"watchdog_seconds",true);
        Json identity={{"run_id",record["run_id"]},{"config_sha256",config_hash},{"source_sha256",digest(record["source"].dump())},{"attempt",attempt_number},{"gpu_uuids",devices}};
        if(request.contains("retry_of"))identity["retry_of"]=request.at("retry_of");
        Json session;
        try{session=budget.begin(identity,static_cast<int>(devices.size()),id,watchdog,request.value("repair",false));}
        catch(const std::exception& failure){record["status"]="blocked";record["admission"]={{"phase",id},{"error",failure.what()}};break;}
        attempt["session_id"]=session.at("session_id");attempt["status"]="running";record["status"]="running";record["budget"]=budget.snapshot();
        record["phases"].push_back(attempt);write(run_path,record);
        const bool cleanup=execute_cells(phase,request,devices,budget,record,attempt);
        try{attempt["charge"]=budget.finish(attempt.at("status").get<std::string>(),cleanup);}catch(const std::exception& failure){attempt["status"]="unsettled";attempt["cleanup_error"]=failure.what();}
        attempt["qualification"]=evaluate_gates(attempt,phase.value("gates",Json::array()));
        record["phases"].back()=attempt;record["budget"]=budget.snapshot();
        if(attempt.at("status")=="completed")completed[id]=attempt;else failed=true;
        record["status"]=failed?"failed":"incomplete";write(run_path,record);
        if(failed)break;
    }
    if(!matched)throw std::invalid_argument("unknown selected phase");
    if(completed.size()==config.at("phases").size())record["status"]="completed";
    record["metrics"]=request_summary(record["requests"]);record["budget"]=budget.snapshot();
    record["measurement_scope"]=devices.empty()?"CPU fixture; not GPU measurements":"supervised GPU experiment";
    record["retention"]["pending_dependencies"]=record["status"]=="completed"?Json::array():Json::array({"resume_or_failure_investigation"});
    budget_owner.reset();
    record["archive"]=publish(record,request,work);return record;
}

Json simulate_study(const Json& request){
    const auto work=real_path(request.at("work_dir").get<std::string>());fs::create_directories(work);RunLock lock(work/"run.lock");
    if(fs::exists(work/"run.json"))throw std::invalid_argument("simulation run already exists");
    const auto config=request.at("config");
    Json record={{"schema","ditserve.run.v1"},{"experiment","synthetic"},{"run_id",segment(request.at("run_id").get<std::string>())},
        {"config",config},{"status","completed"},{"phases",Json::array()},{"requests",Json::array()},{"failures",Json::array()},{"artifacts",Json::array()},
        {"budget",{{"active",nullptr},{"gpu_seconds",0}}},{"notice","Synthetic cost-model simulation; not measured GPU performance."}};
    Json phase=config;phase["matrix"]=config.value("matrix",Json::object());
    for(const auto& cell:expand_matrix(phase)){
        const auto cost=CostModel::from_json(cell.value("cost_model",Json::object()));
        const auto trace=cell.contains("workload")?read_workload(cell.at("workload")):generate_workload(cell.value("generator",Json::object()),cost);
        for(const auto& policy:cell.value("policies",Json::array({"cache_only","batch_only","naive","edf","path_aware","state_aware"}))){
            auto runtime=cell.value("runtime",Json::object());runtime["policy"]=policy;
            auto result=Simulator(RuntimeConfig::from_json(runtime),cost).run(trace);
            record["phases"].push_back({{"id",policy},{"status","completed"},{"config",runtime},{"result",result}});
        }
    }
    record["archive"]=publish(record,request,work);return record;
}
}
