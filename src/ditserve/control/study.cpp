#include "ditserve/control/study.hpp"
#include "ditserve/control/budget.hpp"
#include "ditserve/control/cluster.hpp"
#include "ditserve/control/policy.hpp"
#include "ditserve/control/dispatch.hpp"
#include "ditserve/runtime/process.hpp"
#include "ditserve/benchmark/cluster_sim.hpp"
#include <openssl/sha.h>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <atomic>
#include <csignal>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iomanip>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

namespace ditserve {
namespace {
namespace fs=std::filesystem;
bool cancelled(){return CancelScope::requested();}
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
std::string segment(const std::string& value){return path_segment(value,"path identifier");}
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

class ReplayCutoffReached : public std::runtime_error {
public:
    ReplayCutoffReached():std::runtime_error("replay observation cutoff reached") {}
};

// A real model/RPC fault must survive when another fanout rank merely reaches
// its deadline while peers are being joined.
void keep_failure(std::exception_ptr& prior, std::exception_ptr next) {
    if(!prior){prior=next;return;}
    try{std::rethrow_exception(prior);}
    catch(const WorkerDeadlineExceeded&){
        try{std::rethrow_exception(next);}catch(const WorkerDeadlineExceeded&){}catch(...){prior=next;}
    }catch(...){}
}

struct Worker {
    std::unique_ptr<WorkerProcess> process;
    std::uint64_t sequence=0;
    // One request/reply at a time: a cluster regroup exports from a rank whose
    // GPU another quantum or the controller's encode/release calls may be using.
    std::mutex mutex;
    Json call(Json action,double deadline) {
        std::lock_guard<std::mutex> lock(mutex);
        action["version"]=1;action["id"]=++sequence;
        return process->call(action,deadline);
    }
};
// Controller wall-clock spans are nested observations, not additive worker costs.
// Shared only by a task's rank futures; no worker synchronization is introduced.
struct ControllerTiming {
    double origin;
    std::mutex mutex;
    Json value;
    ControllerTiming(double origin_,Json identity):origin(origin_),value(std::move(identity)){
        value["schema"]="ditserve.controller-timing.v1";
        value["clock"]="controller_replay_monotonic_ms";
        value["stages"]=Json::array();
    }
    double now()const{return (monotonic_seconds()-origin)*1000;}
    void span(const std::string& stage,double begin,const std::string& status,
              const std::string& worker="",const std::string& request="",Json details=Json::object()){
        const double end=now();
        Json entry={{"stage",stage},{"start_ms",begin},{"finish_ms",end},{"elapsed_ms",end-begin},{"status",status}};
        if(!worker.empty())entry["worker"]=worker;
        if(!request.empty())entry["request_id"]=request;
        if(!details.empty())entry["worker_details"]=std::move(details);
        std::lock_guard<std::mutex> guard(mutex);value["stages"].push_back(std::move(entry));
    }
    Json call(Worker& worker,const std::string& name,const std::string& stage,Json action,double deadline){
        const auto request=action.value("state_id",std::string());const double begin=now();
        try{
            auto result=worker.call(std::move(action),deadline);
            Json details=Json::object();
            for(const auto* key:{"worker_operation_ms","load_ms","model_ms","export_ms","write_ms","read_ms","import_ms","bytes"})
                if(result.contains(key))details[key]=result.at(key);
            span(stage,begin,"completed",name,request,std::move(details));return result;
        }catch(...){span(stage,begin,"failed",name,request);throw;}
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
        // A killed GPU worker can take seconds to release a large CUDA context;
        // stop() returns as soon as the group is gone, so only slow exits wait.
        options.cleanup_timeout_seconds=definition.value("cleanup_timeout_seconds",60.0);
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
    // Concurrent, like shutdown: slow exits must not add up across a cell's workers.
    // Returns the workers whose process group could not be verified gone.
    Json stop() {
        std::vector<std::pair<std::string,std::future<bool>>> calls;
        for(auto& item:workers_)calls.emplace_back(item.first,std::async(std::launch::async,[process=item.second->process.get()]{return process->stop();}));
        Json unverified=Json::array();
        for(auto& call:calls)if(!call.second.get())unverified.push_back({{"worker",call.first},{"pid",workers_.at(call.first)->process->pid()}});
        return unverified;
    }
    // Concurrent: ranks of one NCCL world tear it down together, and a rank that
    // waits for its peers must not hold the next worker's shutdown behind it.
    void shutdown(double deadline){
        std::vector<std::future<Json>> calls;
        for(auto& item:workers_)calls.push_back(std::async(std::launch::async,[worker=item.second.get(),deadline]{return worker->call({{"op","shutdown"}},deadline);}));
        std::exception_ptr error;
        for(auto& call:calls)try{call.get();}catch(...){if(!error)error=std::current_exception();}
        if(error)std::rethrow_exception(error);
    }
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
    for(const auto* package:{"ditserve","xfuser"}) {
        const auto package_root=root/package;
        if(!fs::is_directory(package_root))throw std::invalid_argument("missing required Python package: "+package_root.string());
        for(const auto& entry:fs::recursive_directory_iterator(package_root)) {
            if(!entry.is_regular_file() || entry.path().extension()!=".py" || entry.is_symlink())continue;
            const auto relative=fs::relative(entry.path(),root);const auto to=work/"sources"/"python"/relative;
            fs::create_directories(to.parent_path());fs::copy_file(entry.path(),to,fs::copy_options::none);
            manifest["files"][relative.generic_string()]=file_hash(to);
        }
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

}  // namespace

bool isolation_holds(const Json& observation,int& unknown_streak,int tolerance) {
    const auto status=observation.value("status","");
    if(status=="unknown")return ++unknown_streak<=tolerance;
    unknown_streak=0;
    return status=="exclusive_observed"&&observation.value("foreign",Json::array()).empty();
}

namespace {
// A CPU mock worker answers occupancy checks. Heartbeats keep one per cell
// instead of spawning a process every observation; a failed probe is dropped
// and recreated on the next check.
class OccupancyProbe {
public:
    // backend: "nvidia_smi" (two subprocesses per observation) or "nvml" (direct, resident in the probe worker).
    OccupancyProbe(const Json& request,fs::path work,std::string backend="nvidia_smi")
        :request_(request),work_(std::move(work)),backend_(std::move(backend)){}
    ~OccupancyProbe(){if(probe_)probe_->stop();}
    Json observe(const Json& devices,const std::vector<int>& allowed,double deadline) {
        if(devices.empty())return {{"status","exclusive_observed"},{"owned",Json::array()},{"foreign",Json::array()}};
        try {
            if(!probe_)probe_=std::make_unique<Workers>(Json{{"probe",{{"backend","mock"}}}},request_,work_);
            return probe_->get("probe").call({{"op","occupancy"},{"occupancy_backend",backend_},{"gpu_uuids",devices},{"allowed_pids",allowed},{"allowed_pgids",allowed}},deadline>0?deadline:monotonic_seconds()+15);
        }catch(const std::exception& error){
            if(probe_){probe_->stop();probe_.reset();}
            return {{"status","unknown"},{"reason",error.what()}};
        }
    }
private:
    Json request_;fs::path work_;std::string backend_;std::unique_ptr<Workers> probe_;
};

Json gpu_observation(const Json& request,const fs::path& work,const Json& devices,const std::vector<int>& allowed={},double deadline=0) {
    return OccupancyProbe(request,work/"probe").observe(devices,allowed,deadline);
}

Json control_mps(const Json& request,const fs::path& work,const Json& mps,double deadline=0) {
    Workers controller(Json{{"control",{{"backend","mock"}}}},request,work);
    try {
        Json action=mps;action["op"]="mps_control";action["action"]="quit";
        auto result=controller.get("control").call(action,deadline>0?deadline:monotonic_seconds()+15);
        const auto unverified=controller.stop();
        if(!unverified.empty())throw std::runtime_error("MPS control worker cleanup unverified");
        return result;
    } catch(...) { controller.stop(); throw; }
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

Json engine_contract(Worker& worker,const std::string& backend,double deadline) {
    static const std::set<std::string> targets={"wan","wan_pipeline","sd35"};
    if(!targets.count(backend))return Json::object();
    auto hello=worker.call({{"op","hello"}},deadline);
    if(!hello.contains("capabilities")||!hello.at("capabilities").is_object())
        throw std::runtime_error(backend+" worker omitted engine capabilities");
    const auto capabilities=hello.at("capabilities");
    const auto engine=capabilities.value("engine",Json::object());
    if(engine.value("name","")!="xdit"||engine.value("revision","")!="16c34e7f070de65d9d2c67a6d74e0007dd8b426a")
        throw std::runtime_error(backend+" worker does not use the pinned xDiT engine");
    if(capabilities.value("model_wrapper","").empty()||!capabilities.value("quanta",Json()).is_array()
            ||!capabilities.value("migration",Json()).is_array())
        throw std::runtime_error(backend+" worker has incomplete xDiT capabilities");
    if(hello.value("loaded",false)&&capabilities.value("engine_status","")!="loaded")
        throw std::runtime_error(backend+" worker did not activate its advertised xDiT engine");
    return capabilities;
}

// Untimed preparation before the trace clock starts. Model loading, kernel
// selection and allocator growth must not be charged to the first arrivals;
// every warmup state is released before the replay begins. The items run
// concurrently, one thread each (calls on one worker serialize on its mutex):
// every worker loads its own model, and a cell of sixteen workers warmed one
// after another spent minutes doing what takes seconds side by side.
Json warmup(const Json& phase,Workers& workers,Budget& budget) {
    const auto items=phase.value("warmup",Json::array());
    // Spawned here, on the calling thread: Workers::get is not thread-safe, and
    // an unknown worker must fail before anything starts.
    std::vector<Worker*> targets;
    for(const auto& item:items)targets.push_back(&workers.get(item.at("worker").get<std::string>()));
    if(cancelled())throw std::runtime_error("run cancelled");
    budget.check();
    const double deadline=budget.deadline();
    for(const auto& item:items){
        const auto name=item.at("worker").get<std::string>();
        const auto capabilities=engine_contract(workers.get(name),workers.backend(name),deadline);
        const auto quanta=capabilities.value("quanta",Json::array());
        if(item.contains("blocks")&&std::find(quanta.begin(),quanta.end(),"block")==quanta.end())
            throw std::invalid_argument("block warmup requires a block-capable xDiT worker");
    }
    auto run=[&items,&targets,deadline](std::size_t index)->Json{
        const auto& item=items[index];
        auto& worker=*targets[index];
        const double started=monotonic_seconds();
        Json spec=item.value("spec",Json::object());spec["request_id"]="warmup-"+std::to_string(index);
        auto prepared=worker.call({{"op","prepare"},{"spec",spec}},deadline);
        const auto state=prepared.at("state_id").get<std::string>();
        const double load=number_field(prepared,"load_ms");
        auto progress=prepared.value("progress",Json::object());
        for(int calls=0;!progress.value("done",false);++calls){
            if(calls>=10000)throw std::runtime_error("warmup did not finish");
            if(cancelled())throw std::runtime_error("run cancelled");
            Json action={{"op","advance"},{"state_id",state},{"quantum",item.value("quantum",Json("step"))}};
            if(item.contains("blocks"))action["blocks"]=item.at("blocks");
            progress=worker.call(action,deadline).at("progress");
        }
        worker.call({{"op","finish"},{"state_id",state},{"output_mode","check"}},deadline);
        worker.call({{"op","release"},{"state_id",state}},deadline);
        return {{"worker",item.at("worker")},{"load_ms",load},{"elapsed_ms",(monotonic_seconds()-started)*1000}};
    };
    std::vector<std::future<Json>> pending;
    for(std::size_t index=0;index<items.size();++index)pending.push_back(std::async(std::launch::async,run,index));
    Json results=Json::array();
    std::exception_ptr error;
    for(auto& call:pending)try{results.push_back(call.get());}catch(...){if(!error)error=std::current_exception();}
    if(error)std::rethrow_exception(error);
    budget.check();
    return results;
}

Json replay(const Json& phase,Workers& workers,Budget& budget,Json& record,const fs::path& work,const fs::path& cell_dir) {
    Json rows=study_workload(phase),costs=phase.value("costs",Json::object());
    const auto policy=phase.value("policy",phase.value("kind","")=="profile"?"fcfs":"edf");
    const auto warmed=warmup(phase,workers,budget);
    // "file" keeps outputs for downstream evaluation; serving runs without
    // consumers use "check" (shape/finite only) to keep persistence off the GPU path.
    const auto output_mode=phase.value("output_mode",std::string("file"));
    std::map<std::string,Live> live;
    std::map<std::string,std::string> encoding;
    std::map<std::string,std::string> decoding;
    const double origin=monotonic_seconds();
    auto now=[&](){return (monotonic_seconds()-origin)*1000;};
    auto check=[&](){if(cancelled())throw std::runtime_error("run cancelled");budget.check(number_field(phase,"cleanup_reserve_seconds",0));};
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
        auto prepared=invoke(state.worker,action,row);
        if(state.stage==0)encoding.erase(id);
        state.state=prepared.at("state_id").get<std::string>();state.progress=prepared.value("progress",Json::object());
        row["prepare"]=prepared;
        if(prepared.contains("encoder_pool_stage")){
            row["encoder_pool_stage"]=prepared.at("encoder_pool_stage");
            if(prepared.at("encoder_pool_stage").is_object()){
                const auto& st=prepared.at("encoder_pool_stage");
                if(st.contains("cached"))row["encoder_pool_hit"]=st.at("cached");
                if(st.contains("adopted_shm"))row["shm_adopted"]=st.at("adopted_shm");
                if(st.contains("shm_adopted"))row["shm_adopted"]=st.at("shm_adopted");
                if(st.contains("stage_ms"))row["stage_ms"]=st.at("stage_ms");
            }
        }
        for(const auto* field:{"encoder_pool_hit","shm_adopted","stage_ms","encode_meta"})
            if(prepared.contains(field))row[field]=prepared.at(field);
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
            // Mixed traces name the workers with asynchronous prompt encoding;
            // other backends encode inside their own prepare/generate call.
            const auto first_worker=stages(row).at(0).value("worker",row.at("worker").get<std::string>());
            const auto prefetch_workers=phase.value("text_prefetch_workers",Json());
            const bool prefetchable=prefetch_workers.is_null()||std::find(prefetch_workers.begin(),prefetch_workers.end(),first_worker)!=prefetch_workers.end();
            const auto replay_options=phase.value("options",Json::object());
            const bool image_prefetch_enabled=(replay_options.is_object()&&replay_options.contains("image_text_prefetch"))
                ?bool_field(replay_options,"image_text_prefetch",false)
                :bool_field(phase,"image_text_prefetch",false);
            const int text_prefetch_limit=int_field(phase,"text_prefetch",0,0);
            const bool row_prefetch=(text_prefetch_limit>0&&prefetchable)||(image_prefetch_enabled&&row.value("kind",std::string())=="image");
            if(state==live.end() && row_prefetch){
                const auto& worker=first_worker;
                if(!encoding.count(id)){
                    const std::size_t cap=text_prefetch_limit>0?static_cast<std::size_t>(text_prefetch_limit):rows.size();
                    if(encoding.size()>=cap)continue;
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
                        const auto started=invoke(state.worker,{{"op","decode_start"},{"state_id",state.state},{"save_latents",phase.value("save_latents",false)},{"output_mode",output_mode}},row);
                        state.decode_ticket=started.at("ticket").get<std::string>();decoding[state.worker]=id;
                        continue;
                    }
                    auto result=state.decoded.is_null()?invoke(state.worker,{{"op","finish"},{"state_id",state.state},{"save_latents",phase.value("save_latents",false)},{"output_mode",output_mode}},row):state.decoded;
                    state.decoded=nullptr;
                    add_artifacts(record,result,work,phase.value("artifact_dependencies",Json::array()));
                    if(!row.contains("outputs"))row["outputs"]=Json::array();
                    row["outputs"].push_back(result);
                    if(row.contains("reference")) {
                        auto quality=invoke(state.worker,{{"op","evaluate"},{"output",result.at("output").at("path")},{"reference",row.at("reference")}},row);
                        row["quality"]=quality;
                    }
                    invoke(state.worker,{{"op","release"},{"state_id",state.state}},row);
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
    return {{"requests",rows},{"metrics",request_summary(rows)},{"elapsed_ms",now()},{"warmup",warmed}};
}

// Bounded opt-in diagnostics. The disabled path owns no journal or per-plan
// trace. These are controller observations and model estimates, not GPU/RPC
// measurements, an exhaustive decision chain, or a change to scheduling policy.
void validate_controller_decisions(const Json& config) {
    if (!config.is_object()) throw std::invalid_argument("controller_decisions must be an object");
    for (auto item=config.begin();item!=config.end();++item)
        if (item.key()!="enabled"&&item.key()!="max_events"&&item.key()!="max_rows"&&item.key()!="max_bytes"&&item.key()!="focus")
            throw std::invalid_argument("unknown controller_decisions field: "+item.key());
    bool_field(config,"enabled",false);
    const int events=int_field(config,"max_events",256), rows=int_field(config,"max_rows",64), bytes=int_field(config,"max_bytes",4194304);
    if(events>256||rows>64||bytes<4096||bytes>4194304)
        throw std::invalid_argument("controller_decisions limits exceed 256 events / 64 rows / 4 MiB (minimum 4096 bytes)");
    if(config.contains("focus")){
        const auto& focus=config.at("focus");std::set<std::string> seen;
        if(!focus.is_array()||focus.size()>4)throw std::invalid_argument("controller_decisions focus must have at most four IDs");
        for(const auto& value:focus){
            if(!value.is_string()||value.get_ref<const std::string&>().empty()||value.get_ref<const std::string&>().size()>128
                ||!seen.insert(value.get<std::string>()).second)throw std::invalid_argument("invalid controller_decisions focus ID");
        }
    }
}
struct ControllerDecisions {
    Json base, events=Json::array();
    std::vector<std::string> focus;
    std::size_t max_events,max_rows,max_bytes,bytes=0,reserve;
    std::uint64_t calls=0,dropped_calls=0,dropped_rows=0,filtered_rows=0,deduplicated_calls=0,failed_calls=0,omitted_context_records=0,filtered_calls=0;
    bool byte_exhausted=false;
    std::string last_idle_signature,stop_reason;
    ControllerDecisions(const Json& config,const Json& phase,const Json& record,const ClusterCost& cost,const Json& options)
        :max_events(int_field(config,"max_events",256)),max_rows(int_field(config,"max_rows",64)),max_bytes(int_field(config,"max_bytes",4194304)){
        validate_controller_decisions(config);
        if(config.contains("focus"))focus=config.at("focus").get<std::vector<std::string>>();
        base={{"schema","ditserve.controller-decisions.v1"},{"clock","controller_replay_monotonic_ms"},
            {"observation_id",phase.at("_observation_id")},{"phase",phase.at("id")},{"attempt",phase.at("_attempt")},
            {"cell",phase.at("_cell")},{"order",phase.at("_order")},{"policy",phase.at("policy")},
            {"config_sha256",record.at("config_sha256")},{"profile_sha256",digest(cost.profile().dump())},
            {"effective_options_sha256",digest(options.dump())},{"hash_encoding","sha256(nlohmann_json.dump())"},
            {"limits",{{"max_events",max_events},{"max_rows_per_event",max_rows},{"max_bytes",max_bytes},{"max_event_bytes",65536}}},
            {"focus",focus},{"scope","ranking candidates and selected assignments; estimates are not actual RPC counts or placement feasibility"},
            {"step_estimator_source_scope","nominal video step_ms coefficient only; excludes prepare/decode, completed-block subtraction and dispatch overhead; may be multiplied by zero"},
            {"occupancy_scope","controller-observed tasks and decode ownership; not measured GPU occupancy"},
            {"byte_encoding","compact UTF-8 JSON; formatted run.json may be larger"}};
        // Covers fixed report keys and all uint64 counters. The total compact
        // report (not just its event array) stays below the configured budget.
        reserve=base.dump().size()+2048;
        if(reserve>max_bytes)throw std::invalid_argument("controller_decisions identity exceeds byte budget");
    }
    bool matches(const Json& id)const{
        const auto& value=id.get_ref<const std::string&>();
        return std::find(focus.begin(),focus.end(),value)!=focus.end();
    }
    std::unique_ptr<ClusterPlanTrace> begin(bool relevant){
        ++calls;
        if(!relevant){++filtered_calls;return nullptr;}
        if(events.size()>=max_events||byte_exhausted){
            if(stop_reason.empty())stop_reason="event_limit";
            ++dropped_calls;return nullptr;
        }
        auto trace=std::make_unique<ClusterPlanTrace>();trace->max_rows=max_rows;trace->focus=focus;
        return trace;
    }
    std::string retain(Json event,const ClusterPlanTrace& trace){
        dropped_rows+=trace.dropped_rows;filtered_rows+=trace.filtered_rows;
        event["candidate_count"]=trace.candidates;event["dropped_rows"]=trace.dropped_rows;event["filtered_rows"]=trace.filtered_rows;
        event["rows"]=trace.rows;
        omitted_context_records+=event.at("omitted_owners").get<std::size_t>()+event.at("omitted_assignments").get<std::size_t>();
        // Only identical idle snapshots are suppressed. Keep rank/late/owner
        // transitions; now and time margins alone drift on every poll. Dispatch snapshots
        // are never deduplicated so retained IDs identify a single plan call.
        if(event.at("assignments").empty()){
            auto stable=event;stable.erase("now_ms");
            for(auto& row:stable["rows"]){row.erase("deadline_remaining_ms");row.erase("laxity_ms");}
            const auto signature=digest(stable.dump());
            if(signature==last_idle_signature){++deduplicated_calls;return {};}
            last_idle_signature=signature;
        }else last_idle_signature.clear();
        const auto id=base.at("observation_id").get<std::string>()+"/decision/"+std::to_string(calls);
        event["decision_id"]=id;event["plan_call"]=calls;
        const auto size=event.dump().size()+1;
        if(size>65536||reserve+bytes+size>max_bytes){
            stop_reason=size>65536?"single_event_byte_limit":"total_byte_limit";
            ++dropped_calls;byte_exhausted=true;return {};
        }
        bytes+=size;events.push_back(std::move(event));return id;
    }
    Json report()const{
        auto out=base;
        out.update({{"events",events},{"plan_calls",calls},{"dropped_plan_calls",dropped_calls},{"dropped_rows",dropped_rows},
            {"filtered_rows",filtered_rows},{"deduplicated_idle_calls",deduplicated_calls},{"retained_event_bytes",bytes},
            {"failed_plan_calls",failed_calls},{"filtered_plan_calls",filtered_calls},{"omitted_context_records",omitted_context_records},{"capture_stopped_reason",stop_reason},
            {"truncated",dropped_calls>0||dropped_rows>0||failed_calls>0||omitted_context_records>0},{"exhaustive",false}});
        return out;
    }
};
// Concurrent multi-GPU replay. Each GPU runs at most one quantum; between
// quanta the shared cluster_plan decides placement, SP degree and preemption.
// Worker calls run on std::async threads and every rank of an SP group runs
// the same quantum concurrently; all bookkeeping stays on this thread.
// options.sp_transport selects how a request reaches the ranks of a new group:
// "pinned" (the default for real backends) keeps SP1 and the first placement;
// "resume" (mock only) re-prepares ranks from step/block counters; "export"
// joins the video workers in one process group and moves the replicated
// step-boundary state through a host-staged block_export/block_import.
Json cluster(const Json& phase,Workers& workers,Budget& budget,Json& record,const fs::path& work) {
    Json rows=study_workload(phase);
    const ClusterCost cost(phase.at("profile"));
    const auto policy=phase.at("policy").get<std::string>();
    const PolicyTraits& traits=policy_traits(policy);
    Json options=phase.value("options",Json::object());
    const int continuous_batch=int_field(options,"continuous_batch",0,0);
    if(continuous_batch>8)throw std::invalid_argument("continuous_batch must be in [0, 8]");
    if(continuous_batch>0&&int_field(options,"image_batch",1)>1)throw std::invalid_argument("continuous_batch and atomic image_batch are mutually exclusive");
    const auto slots=phase.at("gpus");
    const int gpus=static_cast<int>(slots.size());
    bool fixtures=true;
    for(const auto& slot:slots)fixtures=fixtures&&workers.backend(slot.at("video").get<std::string>())=="mock";
    const auto transport=string_field(options,"sp_transport",fixtures?"resume":"pinned");
    if(transport!="pinned"&&transport!="resume"&&transport!="export")throw std::invalid_argument("sp_transport must be pinned, resume or export");
    if(transport=="resume"&&!fixtures)throw std::invalid_argument("resume transport re-prepares from counters and is mock only");
    if(transport=="export"&&gpus!=2&&gpus!=4&&gpus!=6&&gpus!=8)throw std::invalid_argument("export transport needs 2, 4, 6 or 8 GPUs");
    // Pinning lets a started request resume while another is inside a step on the
    // same worker; SP workers hold one request inside a solver step at a time.
    if(transport!="pinned"&&bool_field(options,"pin_started",false))throw std::invalid_argument("pin_started needs pinned transport");
    if(transport=="pinned"){
        if(traits.fixed_degree>1)throw std::invalid_argument("pinned cluster workers support SP1 only");
        for(const auto& form:options.value("forms",Json::array()))if(parse_form(form.get<std::string>()).degree()!=1)
            throw std::invalid_argument("pinned transport cannot execute the requested parallel forms");
        for(const auto& degree:options.value("degrees",Json::array()))if(degree!=1)
            throw std::invalid_argument("pinned transport cannot execute the requested parallel degrees");
        options.erase("forms");options["degrees"]=Json::array({1});options["pin_started"]=true;
    }
    const bool approximate=traits.approx||traits.v3;
    const bool async_decode=bool_field(options,"async_rank0_decode",traits.v5);
    if(async_decode&&transport!="export")throw std::invalid_argument("asynchronous rank-0 decode requires export transport");
    const bool tiered=bool_field(options,"tiered_memory",traits.tiered_memory);
    if(tiered&&transport!="pinned")throw std::invalid_argument("tiered memory currently requires pinned SP1 transport");
    if(!fixtures&&approximate&&transport!="pinned")throw std::invalid_argument("dynamic cache currently requires SP1 pinned whole-step execution");
    if(!fixtures&&number_field(phase.at("profile"),"ring_quant_scale",1.)!=1.)
        throw std::invalid_argument("INT8 Ring cost discounts are not supported by the real executor");
    const char* env_transfer=std::getenv("DITSERVE_TRANSFER_DIR");
    const fs::path transfer_dir=options.contains("transfer_dir")?fs::path(string_field(options,"transfer_dir",""))
                               :(env_transfer&&*env_transfer?fs::path(env_transfer):work/"transfer");
    const int doorbell_check_blocks=int_field(options,"check_blocks",1,1);
    // Continuous execution (options.doorbell): a video block quantum runs to its step end and stops early only when
    // the controller rings its GPUs' doorbells, 8-byte counters in shared files the workers map. Lazy mode
    // rings only when an early stop changes an image from late to on time, so no-op quantum round trips disappear.
    // Data-path collectives of the video workers: NCCL, or group-free collectives (GF-DiT's GFC).
    const auto collective_backend=string_field(options,"collective_backend","nccl");
    if(collective_backend!="nccl"&&collective_backend!="gfc")throw std::invalid_argument("collective_backend must be nccl or gfc");
    if(collective_backend=="gfc"&&transport!="export")throw std::invalid_argument("group-free collectives need the export transport");
    const bool doorbell=bool_field(options,"doorbell",false);
    if(doorbell&&transport!="export")throw std::invalid_argument("doorbell needs the export transport");
    struct Bells{
        fs::path dir;std::vector<std::string> path;std::vector<int> fd;std::vector<std::uint64_t> epoch;std::uint64_t rings=0,continuous=0,deadline_skips=0;
        ~Bells(){for(int f:fd)if(f>=0)::close(f);std::error_code ignored;if(!dir.empty())fs::remove_all(dir,ignored);}
        void ring(int gpu){
            const std::uint64_t value=++epoch.at(gpu);unsigned char bytes[8];
            for(int k=0;k<8;++k)bytes[k]=static_cast<unsigned char>(value>>(8*k));
            if(::pwrite(fd.at(gpu),bytes,8,0)!=8)throw std::runtime_error("doorbell write failed: "+path.at(gpu));
            ++rings;
        }
        std::uint64_t read(int gpu)const{
            unsigned char bytes[8]={0,0,0,0,0,0,0,0};
            if(::pread(fd.at(gpu),bytes,8,0)!=8)return 0;
            std::uint64_t value=0;
            for(int k=0;k<8;++k)value|=static_cast<std::uint64_t>(bytes[k])<<(8*k);
            return value;
        }
    } bells;
    if(doorbell){
        bells.dir=transfer_dir/("doorbell-"+std::to_string(::getpid())+"-"+std::to_string(static_cast<long long>(monotonic_seconds()*1e6)));
        fs::create_directories(bells.dir);
        for(int gpu=0;gpu<gpus;++gpu){
            bells.path.push_back((bells.dir/("gpu"+std::to_string(gpu))).string());
            const int f=::open(bells.path.back().c_str(),O_RDWR|O_CREAT|O_TRUNC|O_CLOEXEC,0644);
            if(f<0)throw std::runtime_error("cannot create doorbell "+bells.path.back());
            bells.fd.push_back(f);bells.epoch.push_back(0);
            const unsigned char zero[8]={0,0,0,0,0,0,0,0};
            if(::pwrite(f,zero,8,0)!=8)throw std::runtime_error("cannot initialize doorbell "+bells.path.back());
        }
    }
    const bool prefetch=phase.value("text_prefetch",0)>0;
    const bool image_prefetch=options.contains("image_text_prefetch")
        ?bool_field(options,"image_text_prefetch",false)
        :bool_field(phase,"image_text_prefetch",false);
    options["image_text_prefetch"]=image_prefetch;
    const auto output_mode=phase.value("output_mode",std::string("file"));
    if(continuous_batch>0)for(const auto& slot:slots){
        const auto name=slot.at("image").get<std::string>();
        const auto backend=workers.backend(name);
        if(backend!="mock"&&backend!="sd35")throw std::invalid_argument("continuous_batch requires an explicit-step mock or sd35 worker");
    }
    std::set<std::string> interleaved_workers;
    for(const auto& slot:slots)for(const auto* role:{"video","image"}){
        const auto name=slot.at(role).get<std::string>();
        const auto backend=workers.backend(name);
        const auto capabilities=engine_contract(workers.get(name),backend,budget.deadline());
        const bool batch_requested=int_field(options,"image_batch",1)>1||(!options.contains("image_batch")&&traits.image_batching);
        if(std::string(role)=="image"&&batch_requested&&backend!="mock"&&!capabilities.value("request_batch",false))
            throw std::invalid_argument("image batching requires a per-request batch output capability");
        if(std::string(role)=="video"&&backend=="wan"){
            if(capabilities.value("execution",std::string())=="block_interleaved")interleaved_workers.insert(name);
            const auto quanta=capabilities.value("quanta",Json::array());
            const auto migration=capabilities.value("migration",Json::array());
            const auto degrees=capabilities.value("parallel",Json::object()).value("sp_degrees",Json::array());
            const bool blocks=traits.blocks;
            if(blocks&&std::find(quanta.begin(),quanta.end(),"block")==quanta.end())
                throw std::invalid_argument("block/elastic policy requires block execution on every Wan worker");
            if(approximate&&capabilities.value("execution",std::string())!="step")
                throw std::invalid_argument("approximate policies require whole-step Wan execution; block cache is unsupported");
            if(async_decode&&!capabilities.value("async_decode",false))
                throw std::invalid_argument("asynchronous decode requires a compatible Wan attention and VAE configuration");
            if(transport=="export"&&std::find(migration.begin(),migration.end(),"state")==migration.end())
                throw std::invalid_argument("export transport requires xDiT solver-step state migration");
            if(transport=="export"&&std::find(degrees.begin(),degrees.end(),slots.size())==degrees.end())
                throw std::invalid_argument("requested Wan SP degree is not enabled on every worker");
        }
        if(transport=="export"&&std::string(role)=="video"&&backend!="mock"&&backend!="wan")
            throw std::invalid_argument("export transport requires the resumable xDiT Wan backend");
    }
    const auto warmed=warmup(phase,workers,budget);
    if(transport=="export"){
        // init_process_group is collective: every video worker joins at once, each starting at SP1 on its GPU.
        static std::atomic<int> worlds{0};
        const int port=int_field(options,"sp_port",29600,1024)+worlds++%100;
        fs::create_directories(transfer_dir);
        std::vector<std::future<Json>> ranks;
        for(int gpu=0;gpu<gpus;++gpu){
            const auto name=slots.at(gpu).at("video").get<std::string>();
            Json init={{"op","distributed"},{"action","init"},{"init_method","tcp://127.0.0.1:"+std::to_string(port)},
                       {"rank",gpu},{"world_size",gpus},{"ranks",Json::array({gpu})},{"collective_backend",collective_backend}};
            if(options.contains("gfc_max_collective_bytes"))init["gfc_max_collective_bytes"]=int_field(options,"gfc_max_collective_bytes",1,1);
            if(options.contains("gfc_copy_engine"))init["gfc_copy_engine"]=bool_field(options,"gfc_copy_engine",false);
            ranks.push_back(std::async(std::launch::async,[&workers,name,init,deadline=budget.deadline()]{return workers.get(name).call(init,deadline);}));
        }
        std::string error;
        for(auto& rank:ranks){try{rank.get();}catch(const std::exception& failure){error=failure.what();}}
        if(!error.empty())throw std::runtime_error("SP process group initialization failed: "+error);
    }
    // steady_*: block quanta that followed this request's own previous quantum on the same GPUs;
    // resumed_*: block quanta that followed another request there (the first quantum and regroups excluded).
    struct Live{
        std::string status="pending";
        bool prepared=false,offloaded=false,cfg_folded=false,cfg_fold_eligible=false,shm_staged=false;
        int steps=0,blocks=0,total=0,encode_gpu=-1,decode_gpu=-1,decode_done=0,epoch=0;
        std::vector<int> group;
        std::string form="sp1",ticket,pair,decode_ticket,last_stop_event,last_restore_event;
        std::set<int> resident;
        double steady_ms=0,resumed_ms=0,decode_started=0,last_encode_poll_ms=-1e9,last_decode_poll_ms=-1e9;
        int steady_blocks=0,resumed_blocks=0,resumes=0;
    };
    int epochs=0,transfers=0,timing_sequence=0;   // placement, packet and timing identities within this cell
    struct Task{std::size_t row;Json assignment;std::future<Json> result;double started,scheduling_started;bool after_other=false,continuous=false,rung=false;double rung_ms=-1;
                std::shared_ptr<ControllerTiming> timing;bool timing_recorded=false;std::string diagnostic_event_id;};
    std::vector<Live> live(rows.size());
    std::vector<Task> tasks;
    std::vector<bool> busy(gpus,false);
    std::vector<std::string> last_on_gpu(gpus);   // request that last ran on each GPU
    std::vector<std::vector<std::string>> stale(gpus);   // states to release once the GPU is idle
    std::vector<std::set<std::string>> stale_tickets(gpus);   // unused encode tickets to discard once the GPU is idle
    std::map<std::string,std::size_t> index;
    for(std::size_t i=0;i<rows.size();++i){
        auto& row=rows[i];
        if(!row.contains("class"))throw std::invalid_argument("cluster requests require a cost class");
        if(cost.klass(row.at("class").get<std::string>()).at("kind")!=row.at("kind"))throw std::invalid_argument("request kind differs from its cost class");
        row["quality"]={{"status","unverified"},{"passed",false}};row["degrees"]=Json::array();row["forms"]=Json::array();
        row["observation_id"]=phase.at("_observation_id");row["phase"]=phase.at("id");row["attempt"]=phase.at("_attempt");row["cell"]=phase.at("_cell");row["order"]=phase.at("_order");
        index[row.at("request_id").get<std::string>()]=i;
    }
    auto video=[&](int gpu){return slots.at(gpu).at("video").get<std::string>();};
    auto image=[&](int gpu){return slots.at(gpu).at("image").get<std::string>();};
    std::unique_ptr<ControllerDecisions> decisions;
    if(phase.contains("controller_decisions")&&bool_field(phase.at("controller_decisions"),"enabled",false))
        decisions=std::make_unique<ControllerDecisions>(phase.at("controller_decisions"),phase,record,cost,options);
    const double origin=monotonic_seconds();
    auto now=[&](){return (monotonic_seconds()-origin)*1000;};
    const bool cutoff_enabled=phase.contains("replay_cutoff");
    const double cutoff_ms=cutoff_enabled?number_field(phase.at("replay_cutoff"),"after_ms",0,true):0.;
    const double cutoff_deadline=origin+cutoff_ms/1000.;
    auto action_deadline=[&](){return cutoff_enabled?std::min(budget.deadline(),cutoff_deadline):budget.deadline();};
    auto check=[&](){
        if(cancelled())throw std::runtime_error("run cancelled");
        // A phase watchdog/budget limit remains a failure, never normal censoring.
        budget.check(number_field(phase,"cleanup_reserve_seconds",0));
        if(cutoff_enabled&&monotonic_seconds()>=cutoff_deadline)throw ReplayCutoffReached();
    };
    auto is_cutoff=[&](std::exception_ptr error){
        try{std::rethrow_exception(error);}
        catch(const ReplayCutoffReached&){return cutoff_enabled;}
        catch(const WorkerDeadlineExceeded& e){return cutoff_enabled&&e.deadline()==cutoff_deadline&&budget.deadline()>cutoff_deadline;}
        catch(...){return false;}
    };
    bool cutoff_triggered=false;
    double observation_end=-1,cutoff_observed=-1;
    std::exception_ptr replay_error;
    auto note_rpc_failure=[&](Json& row,std::exception_ptr error){
        if(!cutoff_enabled||is_cutoff(error)||row.value("status",std::string())=="completed")return;
        try{std::rethrow_exception(error);}catch(const std::exception& failure){
            row["status"]="failed";row["error"]=failure.what();row["failure_observed_ms"]=now();
            record["failures"].push_back({{"request_id",row.at("request_id")},{"error",failure.what()}});
        }
    };
    auto invoke=[&](const std::string& name,Json action,Json& row){
        try{return workers.get(name).call(std::move(action),action_deadline());}
        catch(...){note_rpc_failure(row,std::current_exception());throw;}
    };
    auto new_timing=[&](const Json& row){
        return std::make_shared<ControllerTiming>(origin,Json{
            {"event_id",phase.at("_observation_id").get<std::string>()+"/"+std::to_string(++timing_sequence)},
            {"request_id",row.at("request_id")},{"phase",phase.at("id")},{"attempt",phase.at("_attempt")},
            {"cell",phase.at("_cell")},{"order",phase.at("_order")}});
    };
    auto persist_timing=[&](Task& task,const std::string& status){
        if(task.timing_recorded)return;
        auto& value=task.timing->value;
        value["observed_ms"]=now();value["controller_status"]=status;
        // Persist only after the future settles; async telemetry may still be
        // writing this JSON object between launch and completion.
        value["scheduling_start_ms"]=task.scheduling_started;
        if(status=="completed"){
            value["step_after"]=live[task.row].steps;value["block_after"]=live[task.row].blocks;
        }
        rows[task.row]["controller_timing"].push_back(value);task.timing_recorded=true;
    };
    Json planning={{"calls",0},{"failed",0},{"total_ms",0.},{"max_ms",0.},
                   {"scope","cluster_plan call only; excludes view construction, dispatch, RPC and polling"}};
    auto save_planning=[&](){
        auto item=planning;
        for(const auto* key:{"phase","attempt","cell","order"})item[key]=rows.front().at(key);
        record["controller_planning"].push_back(std::move(item));
        if(decisions)record["controller_decisions"].push_back(decisions->report());
    };
    auto spec_of=[&](const Json& row){Json spec=row.at("spec");spec["request_id"]=row.at("request_id");return spec;};
    for(auto& row:rows){row["generation"]=spec_of(row);row["output_mode"]=row.value("output_mode",output_mode);}
    Json cb_stats={{"rounds",0},{"late_joins",0},{"early_exits",0},{"batch_sizes",Json::object()},{"memory",Json::object()},{"samples",Json::array()}};
    std::size_t terminal=0,encoders=0,image_encoders=0;
    auto record_prepare_telemetry=[&](Json& target_row,const Json& prepared){
        target_row["prepare"]=prepared;
        if(prepared.contains("encoder_pool_stage")){
            target_row["encoder_pool_stage"]=prepared.at("encoder_pool_stage");
            if(prepared.at("encoder_pool_stage").is_object()){
                const auto& st=prepared.at("encoder_pool_stage");
                if(st.contains("cached"))target_row["encoder_pool_hit"]=st.at("cached");
                if(st.contains("encoder_pool_hit"))target_row["encoder_pool_hit"]=st.at("encoder_pool_hit");
                if(st.contains("adopted_shm"))target_row["shm_adopted"]=st.at("adopted_shm");
                if(st.contains("shm_adopted"))target_row["shm_adopted"]=st.at("shm_adopted");
                if(st.contains("stage_ms"))target_row["stage_ms"]=st.at("stage_ms");
            }
        }
        for(const auto* field:{"encoder_pool_hit","shm_adopted","stage_ms","encode_meta","cpu_prompt_encode_ms","gpu_prepare_ms","negative_prompt_cache"})
            if(prepared.contains(field))target_row[field]=prepared.at(field);
        if(prepared.contains("adopted_shm")&&!target_row.contains("shm_adopted"))
            target_row["shm_adopted"]=prepared.at("adopted_shm");
        if(!target_row.contains("encoder_pool_stage")&&(target_row.contains("encode_ticket_used")||(image_prefetch&&target_row.value("kind",std::string())=="image"))){
            const bool ticket_used=target_row.value("encode_ticket_used",false);
            const bool pool_hit=target_row.value("encoder_pool_hit",target_row.value("encode_pool_cached",ticket_used));
            const bool shm_adopted=target_row.value("shm_adopted",!ticket_used&&target_row.value("encode_cross_worker",false));
            const double stage_ms=target_row.value("stage_ms",number_field(prepared,"model_ms",number_field(prepared,"worker_operation_ms")));
            target_row["encoder_pool_stage"]={{"staged",!pool_hit},{"cached",pool_hit},{"adopted_shm",shm_adopted},{"shm_adopted",shm_adopted},{"stage_ms",stage_ms},{"encode_ticket_used",ticket_used}};
            if(!target_row.contains("encoder_pool_hit"))target_row["encoder_pool_hit"]=pool_hit;
            if(!target_row.contains("shm_adopted"))target_row["shm_adopted"]=shm_adopted;
            if(!target_row.contains("stage_ms"))target_row["stage_ms"]=stage_ms;
        }
    };
    auto move_state=[&](std::size_t i,bool offload){
        auto& state=live[i];auto& row=rows[i];
        if(!state.prepared||state.group.size()!=1||state.blocks!=0||state.decode_done!=0||state.steps>=state.total)
            throw std::runtime_error("tiered move requires an unfinished SP1 solver boundary");
        const auto operation=offload?"tiered_offload":"tiered_prefetch";
        const auto timing=new_timing(row);timing->value["transition"]=offload?"tiered_offload":"tiered_restore";
        timing->value["group"]=state.group;
        Json result;
        try{result=timing->call(workers.get(video(state.group.front())),video(state.group.front()),operation,
                              {{"op",operation},{"state_id",row.at("request_id")}},action_deadline());}
        catch(...){timing->value["observed_ms"]=now();timing->value["status"]="failed";row["controller_timing"].push_back(timing->value);note_rpc_failure(row,std::current_exception());throw;}
        timing->value["observed_ms"]=now();timing->value["status"]="completed";row["controller_timing"].push_back(timing->value);
        if(!offload)state.last_restore_event=timing->value.at("event_id").get<std::string>();
        state.offloaded=offload;
        if(!row.contains("state_moves"))row["state_moves"]=Json::array();
        row["state_moves"].push_back(result);
        row["state_move_ms"]=number_field(row,"state_move_ms")+number_field(result,"ms");
    };
    auto launch=[&](const Json& assignment,const std::string* decision_id){
        check();
        bool continuous=false;
        const auto id=assignment.at("request_id").get<std::string>();const std::size_t i=index.at(id);
        auto& row=rows[i];auto& state=live[i];
        if(state.offloaded)move_state(i,false);
        std::vector<int> group;for(const auto& gpu:assignment.at("group"))group.push_back(gpu.get<int>());
        for(int gpu:group)busy[gpu]=true;
        if(!row.contains("start_ms"))row["start_ms"]=now();
        if(row.at("kind")=="image")row["gpu_group"]=group;
        const int degree=static_cast<int>(group.size());
        // The group's parallel form: CFG branches on separate halves (cfg) and SP inside each.
        const Form form=parse_form(assignment.value("form","sp"+std::to_string(degree)));
        if(form.degree()!=degree)throw std::logic_error("assignment form does not match its group");
        const auto timing=new_timing(row);
        if(decision_id&&!decision_id->empty())timing->value["decision_id"]=*decision_id;
        timing->value.update({{"selected_ms",now()},{"group_before",state.group},{"group_after",group},
            {"form_before",state.form},{"form_after",form_name(form)},{"step_before",state.steps},{"block_before",state.blocks},
            {"epoch_before",state.epoch},{"first",!state.prepared},{"transport",transport},{"quantum",assignment.at("quantum")}});
        if(!state.last_stop_event.empty()){timing->value["resume_from_stop_event_id"]=state.last_stop_event;state.last_stop_event.clear();}
        if(!state.last_restore_event.empty()){timing->value["restore_event_id"]=state.last_restore_event;state.last_restore_event.clear();}
        if(form.cfg>1&&transport=="pinned")throw std::invalid_argument("pinned cluster workers run SP1 only");
        const bool cfg_folded=assignment.value("cfg_folded",false);
        const bool fission=assignment.value("fission",traits.fission);
        if(cfg_folded&&(!traits.v5||(fission&&form.cfg!=1)))throw std::invalid_argument("CFG-fold execution form differs from its resource mode");
        state.cfg_folded=cfg_folded;
        const auto form_label=form_name(form);
        if(std::find(row["degrees"].begin(),row["degrees"].end(),degree)==row["degrees"].end())row["degrees"].push_back(degree);
        if(assignment.at("quantum")!="atomic"&&assignment.at("quantum")!="finish"&&std::find(row["forms"].begin(),row["forms"].end(),form_label)==row["forms"].end())row["forms"].push_back(form_label);
        if(assignment.at("reason")=="block_preempt")row["preempting"]=true;
        const auto quantum=assignment.at("quantum");const double deadline=action_deadline();
        // Request-level interleaving: the partner advances in lockstep on the same group.
        const std::string pair=assignment.value("pair",std::string());
        const std::size_t j=pair.empty()?rows.size():index.at(pair);
        if(j<rows.size()){
            auto& prow=rows[j];auto& pstate=live[j];
            if(!prow.contains("start_ms"))prow["start_ms"]=now();
            if(std::find(prow["degrees"].begin(),prow["degrees"].end(),degree)==prow["degrees"].end())prow["degrees"].push_back(degree);
            if(std::find(prow["forms"].begin(),prow["forms"].end(),form_label)==prow["forms"].end())prow["forms"].push_back(form_label);
            prow["paired_quanta"]=prow.value("paired_quanta",0)+1;row["paired_quanta"]=row.value("paired_quanta",0)+1;
            pstate.status="running";pstate.group=group;pstate.form=form_label;
        }
        std::function<Json()> work_unit;
        if(quantum=="continuous_batch"){
            const auto name=image(group.front());
            Json specs=Json::object(),modes=Json::object(),ids=Json::array(),tickets=Json::object();
            int new_count=0,active_count=0;
            for(const auto& member:assignment.at("batch")){
                const auto mid=member.get<std::string>();const auto m=index.at(mid);
                auto& mrow=rows[m];auto& mstate=live[m];
                ids.push_back(mid);modes[mid]=mrow.at("output_mode");
                if(mstate.prepared)++active_count;
                else{
                    ++new_count;specs[mid]=spec_of(mrow);
                    if(!mstate.ticket.empty()){
                        const bool same_worker=(mstate.encode_gpu==group.front());
                        if(same_worker)tickets[mid]=mstate.ticket;
                        mrow["encode_ticket_used"]=same_worker;
                        mrow["encode_cross_worker"]=!same_worker;
                        mstate.ticket.clear();
                    }
                }
                if(!mrow.contains("start_ms"))mrow["start_ms"]=now();
                mrow["batched"]=static_cast<int>(assignment.at("batch").size());
                mstate.status="running";mstate.group=group;mstate.form=form_label;
            }
            const bool cb_bubble_free=assignment.value("bubble_free",false);
            const bool cb_seq_pack=assignment.value("sequence_packing",false);
            work_unit=[&workers,name,ids,specs,modes,tickets,new_count,active_count,deadline,origin,cb_bubble_free,cb_seq_pack]{
                auto& w=workers.get(name);
                Json out={{"prepare",Json::array()},{"finish",Json::array()},
                          {"new_count",new_count},{"active_count",active_count}};
                for(const auto& mid:ids)if(specs.contains(mid.get<std::string>())){
                    const auto mid_str=mid.get<std::string>();
                    Json prep_req={{"op","prepare"},{"state_id",mid},{"spec",specs.at(mid_str)}};
                    if(tickets.contains(mid_str))prep_req["encode_ticket"]=tickets.at(mid_str);
                    auto prepared=w.call(prep_req,deadline);
                    prepared["state_id"]=mid;out["prepare"].push_back(prepared);
                }
                Json adv_req={{"op","advance_batch"},{"state_ids",ids}};
                if(cb_bubble_free)adv_req["bubble_free"]=true;
                if(cb_seq_pack)adv_req["sequence_packing"]=true;
                const auto advanced=w.call(adv_req,deadline);
                out["advance_batch"]=advanced;
                for(const auto& result:advanced.at("results"))if(result.at("progress").value("done",false)){
                    const auto mid=result.at("state_id").get<std::string>();
                    Json fin_req={{"op","finish"},{"state_id",mid},{"output_mode",modes.at(mid)}};
                    if(cb_bubble_free)fin_req["bubble_free"]=true;
                    auto finished=w.call(fin_req,deadline);
                    w.call({{"op","release"},{"state_id",mid}},deadline);
                    finished["completed_at_ms"]=(monotonic_seconds()-origin)*1000;
                    finished["state_id"]=mid;out["finish"].push_back(finished);
                }
                return out;
            };
        }else if(quantum=="atomic"&&assignment.contains("batch")){
            const auto name=image(group.front());
            const bool batch_bubble_free=assignment.value("bubble_free",false);
            const bool batch_seq_pack=assignment.value("sequence_packing",false);
            Json prompts=Json::array(),seeds=Json::array(),members=Json::array();auto batch_mode=output_mode;
            for(const auto& member:assignment.at("batch")){
                const auto m=index.at(member.get<std::string>());
                auto& mrow=rows[m];auto& mstate=live[m];const auto mspec=spec_of(mrow);
                prompts.push_back(mspec.value("prompt",""));seeds.push_back(mspec.value("seed",0));
                members.push_back({{"request_id",member},{"spec",mspec},{"output_mode",mrow.at("output_mode")}});
                if(!mstate.prepared&&!mstate.ticket.empty()){
                    const bool same_worker=(mstate.encode_gpu==group.front());
                    mrow["encode_ticket_used"]=false;
                    mrow["encode_cross_worker"]=!same_worker;
                    mstate.ticket.clear();
                }
                if(mrow.at("output_mode")=="file")batch_mode="file";
                if(!mrow.contains("start_ms"))mrow["start_ms"]=now();
                mrow["batched"]=static_cast<int>(assignment.at("batch").size());mstate.status="running";
            }
            work_unit=[&workers,name,prompts,seeds,members,deadline,batch_mode,batch_bubble_free,batch_seq_pack]{
                auto& w=workers.get(name);
                Json out={{"calls",Json::array()}};
                Json batch_req={{"op","generate_batch"},{"prompts",prompts},{"seeds",seeds},
                    {"requests",members},{"output_mode",batch_mode}};
                if(batch_bubble_free)batch_req["bubble_free"]=true;
                if(batch_seq_pack)batch_req["sequence_packing"]=true;
                out["finish"]=w.call(batch_req,deadline);
                const auto& results=out.at("finish").at("results");
                if(!results.is_array()||results.size()!=members.size())throw std::runtime_error("batch output count differs from requests");
                std::set<std::string> returned;
                for(const auto& result:results){
                    const auto rid=result.at("request_id").get<std::string>();
                    auto member=std::find_if(members.begin(),members.end(),[&](const Json& m){return m.at("request_id")==rid;});
                    if(member==members.end()||!returned.insert(rid).second||result.value("output_count",0)!=1||!result.contains("output"))
                        throw std::runtime_error("batch output identity differs from requests");
                    for(auto field=member->at("spec").begin();field!=member->at("spec").end();++field)
                        if(!result.at("generation").contains(field.key())||result.at("generation").at(field.key())!=field.value())
                            throw std::runtime_error("batch output generation differs from request specification");
                }
                out["model_ms"]=number_field(out["finish"].value("service",Json::object()),"total_ms",number_field(out["finish"],"model_ms"));
                return out;
            };
        }else if(quantum=="atomic"){
            const auto name=image(group.front());const auto spec=spec_of(row);const auto mode=row.at("output_mode");
            std::string encode_ticket;
            if(!state.prepared&&!state.ticket.empty()){
                const bool same_worker=(state.encode_gpu==group.front());
                if(same_worker)encode_ticket=state.ticket;
                row["encode_ticket_used"]=same_worker;
                row["encode_cross_worker"]=!same_worker;
                state.ticket.clear();
            }
            work_unit=[&workers,name,id,spec,deadline,mode,encode_ticket]{
                auto& w=workers.get(name);
                Json out={{"calls",Json::array()}};
                Json prep_req={{"op","prepare"},{"state_id",id},{"spec",spec}};
                if(!encode_ticket.empty())prep_req["encode_ticket"]=encode_ticket;
                auto prepared=w.call(prep_req,deadline);
                out["prepare"]=prepared;
                out["model_ms"]=number_field(out,"model_ms")+number_field(prepared,"model_ms");
                auto progress=prepared.at("progress");
                for(int calls=0;!progress.value("done",false);++calls){
                    if(calls>=1000)throw std::runtime_error("image did not finish");
                    auto advanced=w.call({{"op","advance"},{"state_id",id},{"quantum","step"}},deadline);
                    progress=advanced.at("progress");out["model_ms"]=number_field(out,"model_ms")+number_field(advanced,"model_ms");
                }
                out["finish"]=w.call({{"op","finish"},{"state_id",id},{"output_mode",mode}},deadline);
                w.call({{"op","release"},{"state_id",id}},deadline);
                return out;
            };
        }else if(quantum=="finish"){
            const auto name=video(group.front());
            Json action={{"op",async_decode?"decode_start":"finish"},{"state_id",id},{"output_mode",row.at("output_mode")}};
            if(async_decode)action["video_exclusion"]=true;
            if(!async_decode&&assignment.contains("decode_frames"))action["decode_frames"]=assignment.at("decode_frames");
            work_unit=[&workers,name,id,deadline,action]{
                auto& w=workers.get(name);
                if(action.at("op")=="decode_start")return Json{{"async_decode",w.call(action,deadline)}};
                Json out={{"finish",w.call(action,deadline)}};
                if(out["finish"].value("decode",Json::object()).value("done",true))w.call({{"op","release"},{"state_id",id}},deadline);
                return out;
            };
            for(int gpu:state.resident)if(gpu!=group.front())stale[gpu].push_back(id);
            state.resident={group.front()};
            state.decode_gpu=group.front();
        }else if(transport=="export"){
            // One rank owns the request (it prepares it, or already holds the replicated
            // step-boundary state); ranks entering the group import its host-staged export
            // before any rank advances. Placement changes only here, at step boundaries,
            // and every rank of a placement receives the same epoch.
            struct Staging{Json prepare;std::string id,source_name,transfer;std::vector<std::string> targets;bool release_source=false,reencode=false;};
            auto in_group=[&](int gpu){return std::find(group.begin(),group.end(),gpu)!=group.end();};
            auto stage=[&](const std::string& rid,const Json& arow,Live& astate){
                Staging st;st.id=rid;
                if(group!=astate.group)astate.epoch=++epochs;
                int source=-1;
                if(!astate.prepared){
                    if(!astate.ticket.empty()&&astate.shm_staged&&astate.encode_gpu!=group.front()){
                        stale_tickets[astate.encode_gpu].insert(astate.ticket);
                        astate.ticket.clear();
                    }
                    // Prepare where the prompt was encoded and export the prepared state into the group. Preparing
                    // on another worker without the ticket re-encodes the prompt while the group's GPUs wait.
                    const bool ticket=!astate.ticket.empty();
                    source=ticket?astate.encode_gpu:group.front();
                    st.reencode=!ticket&&!astate.shm_staged;
                    // A continuous video on the encoding GPU gives its worker back at the next check point.
                    if(ticket&&doorbell&&!in_group(source)&&busy[source])for(auto& task:tasks)if(task.continuous&&!task.rung){
                        const auto& held=task.assignment.at("group");
                        if(std::find(held.begin(),held.end(),Json(source))==held.end())continue;
                        for(const auto& gpu:held)bells.ring(gpu.get<int>());
                        task.rung=true;task.rung_ms=now();
                    }
                    st.prepare={{"op","prepare"},{"state_id",rid},{"spec",spec_of(arow)},{"sp_degree",degree}};
                    st.prepare["cfg"]=form.cfg;st.prepare["ring"]=form.ring;
                    if(ticket)st.prepare["encode_ticket"]=astate.ticket;
                }else{
                    for(int gpu:astate.resident)if(in_group(gpu)){source=gpu;break;}
                    if(source<0&&!astate.resident.empty())source=*astate.resident.begin();
                }
                for(int gpu:group)if(gpu!=source&&!astate.resident.count(gpu))st.targets.push_back(video(gpu));
                // A source leaving the group releases with its export, never through the
                // stale list, which the controller may drain before the export runs.
                st.release_source=source>=0&&!in_group(source);
                for(int gpu:astate.resident)if(gpu!=source&&!in_group(gpu))stale[gpu].push_back(rid);
                st.source_name=source>=0?video(source):std::string();
                st.transfer=(transfer_dir/(rid+"-"+std::to_string(++transfers)+".pt")).string();
                astate.resident=std::set<int>(group.begin(),group.end());astate.ticket.clear();
                return st;
            };
            const bool unprepared=!state.prepared,pair_unprepared=j<rows.size()&&!live[j].prepared;
            std::vector<Staging> stagings={stage(id,row,state)};
            if(j<rows.size())stagings.push_back(stage(pair,rows[j],live[j]));
            timing->value["transfer_targets"]=stagings[0].targets.size();
            if(unprepared)row["prompt_reencoded"]=stagings[0].reencode;
            if(pair_unprepared)rows[j]["prompt_reencoded"]=stagings[1].reencode;
            std::vector<std::pair<std::string,Json>> advances;
            for(int gpu:group){
                Json advance={{"op","advance"},{"state_id",id},{"sp_degree",degree},{"group",group},{"epoch",state.epoch}};
                advance["cfg"]=form.cfg;advance["ring"]=form.ring;advance["cfg_folded"]=cfg_folded;
                if(assignment.contains("cache_threshold"))advance["cache_threshold"]=assignment.at("cache_threshold");
                if(!pair.empty())advance["pair_state_id"]=pair;
                if(quantum=="step")advance["quantum"]="step";else advance["blocks"]=quantum;
                if(doorbell&&pair.empty()&&quantum.is_number_integer()){
                    // Run to the step end, checking this GPU's doorbell at every check_blocks boundary (default 1).
                    advance.erase("blocks");advance["quantum"]="step";
                    advance["check_blocks"]=doorbell_check_blocks;
                    advance["doorbell"]=bells.path.at(gpu);advance["doorbell_epoch"]=bells.epoch.at(gpu);
                    continuous=true;
                }
                advances.emplace_back(video(gpu),advance);
            }
            work_unit=[&workers,&bells,group,stagings,advances,deadline,timing]{
                Json out={{"ranks",Json::array()}};
                bool staged=false;
                for(std::size_t k=0;k<stagings.size();++k){
                    const auto& st=stagings[k];const auto& rid=st.id;
                    const char* prepare_key=k?"pair_prepare":"prepare";const char* transfer_key=k?"pair_transfer":"transfer";
                    if(!st.prepare.is_null()){
                        staged=true;
                        out[prepare_key]=timing->call(workers.get(st.source_name),st.source_name,"prepare",st.prepare,deadline);
                    }
                    if(!st.targets.empty()){
                        staged=true;
                        const auto exported=timing->call(workers.get(st.source_name),st.source_name,"export",
                            {{"op","block_export"},{"state_id",rid},{"path",st.transfer},{"release",st.release_source}},deadline);
                        const double imports_begin=timing->now();
                        std::vector<std::future<Json>> imports;
                        for(const auto& name:st.targets)imports.push_back(std::async(std::launch::async,[&workers,name,rid,st,exported,deadline,timing]{
                            return timing->call(workers.get(name),name,"import",{{"op","block_import"},{"state_id",rid},{"path",st.transfer},{"sha256",exported.at("sha256")}},deadline);
                        }));
                        Json imported=Json::array();std::exception_ptr error;
                        for(auto& rank:imports){try{imported.push_back(rank.get());}catch(...){keep_failure(error,std::current_exception());}}
                        timing->span("imports_all_ready",imports_begin,!error?"completed":"failed","",rid);
                        std::error_code ignored;fs::remove(st.transfer,ignored);
                        if(error)std::rethrow_exception(error);
                        out[transfer_key]={{"bytes",exported.value("bytes",0)},{"export_ms",number_field(exported,"export_ms")},
                                           {"write_ms",number_field(exported,"write_ms")},{"targets",st.targets.size()},{"imports",imported}};
                    }
                }
                if(staged&&!advances.empty()&&advances.front().second.contains("doorbell_epoch")){
                    bool rung_during_staging=false;
                    for(std::size_t r=0;r<group.size()&&r<advances.size();++r){
                        const auto want=advances[r].second.at("doorbell_epoch").get<std::uint64_t>();
                        if(bells.read(group[r])>want){rung_during_staging=true;break;}
                    }
                    if(rung_during_staging){
                        timing->span("advances_skipped_doorbell",timing->now(),"completed");
                        out["pre_advance_doorbell_stop"]=true;
                        return out;
                    }
                }
                const double advances_begin=timing->now();
                std::vector<std::future<Json>> ranks;
                for(const auto& item:advances)ranks.push_back(std::async(std::launch::async,[&workers,item,deadline,timing]{
                    return timing->call(workers.get(item.first),item.first,"advance",item.second,deadline);
                }));
                std::exception_ptr error;
                for(auto& rank:ranks){try{out["ranks"].push_back(rank.get());}catch(...){keep_failure(error,std::current_exception());}}
                timing->span("advances_all_done",advances_begin,!error?"completed":"failed");
                if(error)std::rethrow_exception(error);
                return out;
            };
        }else{
            // Ranks entering the group (first quantum or a regroup) prepare;
            // ranks leaving it release once their GPU is idle.
            std::vector<std::pair<std::string,Json>> calls;
            std::vector<std::pair<std::string,Live*>> members={{id,&state}};
            if(j<rows.size())members.emplace_back(pair,&live[j]);
            for(const auto& member:members)for(int gpu:member.second->resident)if(std::find(group.begin(),group.end(),gpu)==group.end())stale[gpu].push_back(member.first);
            for(int gpu:group){
                Json actions=Json::array();
                for(const auto& member:members){
                    const auto& mid=member.first;Live& mstate=*member.second;
                    if(mstate.resident.count(gpu))continue;
                    Json prepare={{"op","prepare"},{"state_id",mid},{"spec",spec_of(rows[index.at(mid)])},{"sp_degree",degree}};
                    prepare["cfg"]=form.cfg;prepare["ring"]=form.ring;
                    if(mstate.prepared)prepare["resume"]={{"step",mstate.steps},{"block",mstate.blocks}};
                    else if(gpu==mstate.encode_gpu&&!mstate.ticket.empty())prepare["encode_ticket"]=mstate.ticket;
                    actions.push_back(prepare);
                }
                Json advance={{"op","advance"},{"state_id",id},{"sp_degree",degree},{"quantum",quantum}};
                advance["cfg"]=form.cfg;advance["ring"]=form.ring;advance["cfg_folded"]=cfg_folded;
                if(assignment.contains("cache_threshold"))advance["cache_threshold"]=assignment.at("cache_threshold");
                if(!pair.empty())advance["pair_state_id"]=pair;
                if(quantum!="step"){advance["blocks"]=quantum;advance["quantum"]=nullptr;}
                actions.push_back(advance);
                calls.emplace_back(video(gpu),actions);
            }
            for(const auto& member:members){member.second->resident=std::set<int>(group.begin(),group.end());member.second->ticket.clear();}
            work_unit=[&workers,calls,deadline,timing]{
                const double advances_begin=timing->now();
                std::vector<std::future<Json>> ranks;
                for(const auto& item:calls)ranks.push_back(std::async(std::launch::async,[&workers,item,deadline,timing]{
                    auto& w=workers.get(item.first);Json last;
                    for(const auto& action:item.second)last=timing->call(w,item.first,action.at("op").get<std::string>(),action,deadline);
                    return last;
                }));
                Json out={{"ranks",Json::array()}};
                std::exception_ptr error;
                for(auto& rank:ranks){try{out["ranks"].push_back(rank.get());}catch(...){keep_failure(error,std::current_exception());}}
                timing->span("prepare_and_advances_all_done",advances_begin,!error?"completed":"failed");
                if(error)std::rethrow_exception(error);
                return out;
            };
        }
        state.status="running";state.group=group;state.form=form_label;
        bool after_other=false;for(int gpu:group){if(last_on_gpu[gpu]!=id)after_other=true;last_on_gpu[gpu]=id;}
        const bool first=timing->value.at("first").get<bool>();
        const bool regroup=timing->value.at("group_before")!=timing->value.at("group_after")
                            ||timing->value.at("form_before")!=timing->value.at("form_after");
        const std::string transition=row.at("kind")=="image"?"image_service":quantum=="finish"?"finish":
            first?(timing->value.value("transfer_targets",0)>0?"initial_distribution":"initial_prepare"):
            regroup?"regroup":after_other?"resident_resume":"continuation";
        timing->value["transition"]=transition;
        timing->value["epoch_after"]=state.epoch;
        if(!pair.empty())timing->value["pair_request_id"]=pair;
        // Capture before async launch; a fast task may otherwise precede its own start timestamp.
        // Async work mutates timing->value; snapshot this identity before
        // launch so diagnostics never read that shared JSON root in flight.
        std::string diagnostic_event_id;
        if(decisions)diagnostic_event_id=timing->value.at("event_id").get<std::string>();
        const double started=now();timing->value["start_ms"]=started;
        auto observed_work=[work_unit,timing]{
            timing->value["async_begin_ms"]=timing->now();
            try{auto out=work_unit();timing->value["all_done_ms"]=timing->now();timing->value["status"]="completed";return out;}
            catch(...){timing->value["all_done_ms"]=timing->now();timing->value["status"]="failed";throw;}
        };
        // Preserve the scheduler's pre-telemetry anchor after async launch.
        // The earlier observation start must not alter preemption estimates.
        Task task{i,assignment,std::async(std::launch::async,observed_work),started,now(),after_other,false,false,-1,timing,false,std::move(diagnostic_event_id)};
        tasks.push_back(std::move(task));
        tasks.back().continuous=continuous;if(continuous)++bells.continuous;
    };
    auto finish_task=[&](Task& task,const Json& out){
        auto& row=rows[task.row];auto& state=live[task.row];
        for(const auto& gpu:task.assignment.at("group"))busy[gpu.get<int>()]=false;
        const double elapsed=now()-task.started;
        const auto quantum=task.assignment.at("quantum");
        const auto key=quantum.is_string()?quantum.get<std::string>():"blocks";
        row["action_ms"][key]=row.value("action_ms",Json::object()).value(key,0.)+elapsed;
        row["quanta"]=row.value("quanta",0)+1;
        if(task.rung_ms>=0){
            bool stopped=out.value("pre_advance_doorbell_stop",false);
            for(const auto& rank:out.value("ranks",Json::array()))for(const auto& event:rank.value("events",Json::array()))
                stopped=stopped||event.value("doorbell_stop",false);
            if(stopped){
                const double observed=now();
                state.last_stop_event=task.timing->value.at("event_id").get<std::string>()+"/stop";
                row["preemption_stop_samples"].push_back({{"stop_ms",observed-task.rung_ms},{"ring_ms",task.rung_ms},
                    {"observed_ms",observed},{"event_id",state.last_stop_event},{"task_event_id",task.timing->value.at("event_id")},
                    {"group",task.assignment.at("group")},{"scope","ring_to_controller_confirmed_stop"}});
            }
        }
        if(out.contains("async_decode")){
            state.decode_ticket=out.at("async_decode").at("ticket").get<std::string>();
            state.decode_started=task.started;
            state.status="decoding";
            row["status"]="decoding";
            row["async_decode_submits"]=row.value("async_decode_submits",0)+1;
            return;
        }
        if(quantum=="finish"&&!out.at("finish").value("decode",Json::object()).value("done",true)){
            row["model_ms"]=number_field(row,"model_ms")+number_field(out.at("finish"),"model_ms");
            state.decode_done=out.at("finish").at("decode").value("frames_done",state.decode_done);state.status="suspended";
            return;
        }
        if(quantum=="continuous_batch"){
            const auto& advanced=out.at("advance_batch");
            const int size=static_cast<int>(task.assignment.at("batch").size());
            cb_stats["rounds"]=cb_stats.value("rounds",0)+1;
            const auto batch_key=std::to_string(size);
            cb_stats["batch_sizes"][batch_key]=cb_stats["batch_sizes"].value(batch_key,0)+1;
            int completed=0;
            for(const auto& member:task.assignment.at("batch")){
                const auto mid=member.get<std::string>();const auto m=index.at(mid);
                auto& mrow=rows[m];auto& mstate=live[m];
                if(m!=task.row){
                    mrow["action_ms"][key]=mrow.value("action_ms",Json::object()).value(key,0.)+elapsed;
                    mrow["quanta"]=mrow.value("quanta",0)+1;
                }
                mrow["batched"]=size;mrow["continuous_rounds"]=mrow.value("continuous_rounds",0)+1;
                mrow["continuous_ms"]=number_field(mrow,"continuous_ms")+elapsed;
                mrow["continuous_round_model_ms"]=number_field(mrow,"continuous_round_model_ms")+number_field(advanced,"model_ms");
                double model=number_field(advanced,"model_ms"),operation=number_field(advanced,"worker_operation_ms");
                for(const auto& prepared:out.at("prepare"))if(prepared.at("state_id")==mid){
                    const double prep_model=number_field(prepared,"model_ms");
                    const double prep_op=number_field(prepared,"worker_operation_ms");
                    const double prep_stage=(prepared.contains("encoder_pool_stage")&&prepared.at("encoder_pool_stage").is_object())
                        ?number_field(prepared.at("encoder_pool_stage"),"stage_ms"):number_field(prepared,"stage_ms");
                    const double prep_cb=prep_model>0.?prep_model:(prep_stage>0.?prep_stage:prep_op);
                    model+=prep_model;operation+=prep_op;
                    mrow["prepare_cb_ms"]=number_field(mrow,"prepare_cb_ms")+prep_cb;
                    mrow["prepare_cb_wall_ms"]=number_field(mrow,"prepare_cb_wall_ms")+prep_op;
                    record_prepare_telemetry(mrow,prepared);
                }
                const auto progress=std::find_if(advanced.at("results").begin(),advanced.at("results").end(),[&](const Json& result){
                    return result.at("state_id")==mid;
                });
                if(progress==advanced.at("results").end())throw std::runtime_error("continuous batch omitted a member's progress");
                mstate.steps=progress->at("progress").value("step",0);mstate.total=progress->at("progress").value("steps",0);
                mstate.prepared=true;
                bool decoded=false;
                for(const auto& finished:out.at("finish"))if(finished.at("state_id")==mid){
                    model+=number_field(finished,"model_ms");operation+=number_field(finished,"worker_operation_ms");
                    mrow["decode_cb_ms"]=number_field(mrow,"decode_cb_ms")+number_field(finished,"model_ms");
                    add_artifacts(record,finished,work,phase.value("artifact_dependencies",Json::array()));
                    if(!mrow.contains("outputs"))mrow["outputs"]=Json::array();
                    mrow["outputs"].push_back(finished);decoded=true;
                // The cutoff observes controller acceptance, never a worker or
                // async-thread completion timestamp received after the window.
                mrow["finish_ms"]=cutoff_enabled?now():finished.value("completed_at_ms",now());
                }
                if(decoded!=progress->at("progress").value("done",false))
                    throw std::runtime_error("continuous batch progress and decode completion disagree");
                mrow["model_ms"]=number_field(mrow,"model_ms")+model;
                mrow["worker_operation_ms"]=number_field(mrow,"worker_operation_ms")+operation;
                mrow["timed_quanta"]=mrow.value("timed_quanta",0)+1;
                if(decoded){
                    ++completed;mrow["status"]="completed";mstate.status="completed";mstate.group.clear();++terminal;
                }else mstate.status="suspended";
            }
            const int joins=out.value("active_count",0)>0?out.value("new_count",0):0;
            const int exits=completed<size?completed:0;
            cb_stats["late_joins"]=cb_stats.value("late_joins",0)+joins;
            cb_stats["early_exits"]=cb_stats.value("early_exits",0)+exits;
            Json sample={{"batch_size",size},{"ms",elapsed},{"model_ms",number_field(advanced,"model_ms")},
                         {"new_count",out.at("new_count")},{"active_count",out.at("active_count")},
                         {"completed",completed},{"late_joins",joins},{"early_exits",exits}};
            if(advanced.contains("memory_after")&&advanced.at("memory_after").is_object()){
                const auto& memory=advanced.at("memory_after");
                sample["memory_after"]=memory;
                if(!cb_stats["memory"].is_object())cb_stats["memory"]=Json::object();
                auto& peak=cb_stats["memory"][batch_key];
                if(!peak.is_object())peak=Json::object();
                for(const auto* field:{"peak_allocated_bytes","peak_reserved_bytes"}){
                    const auto current=number_field(memory,field);
                    peak[field]=std::max(peak.value(field,0.),current);
                }
            }
            if(cb_stats["samples"].size()<256)cb_stats["samples"].push_back(sample);
            return;
        }
        if(quantum=="atomic"&&task.assignment.contains("batch")){
            const auto& finished=out.at("finish");
            const double finished_at=now();
            for(const auto& result:finished.at("results")){
                const auto m=index.at(result.at("request_id").get<std::string>());
                auto& mrow=rows[m];auto& mstate=live[m];
                add_artifacts(record,result,work,phase.value("artifact_dependencies",Json::array()));
                mrow["outputs"]=Json::array({result});
                if(m!=task.row){
                    mrow["action_ms"][key]=mrow.value("action_ms",Json::object()).value(key,0.)+elapsed;
                    mrow["quanta"]=mrow.value("quanta",0)+1;
                }
                mrow["batch_model_ms"]=number_field(out,"model_ms");
                mrow["batch_size"]=finished.at("results").size();
                mrow["batch_request_ids"]=task.assignment.at("batch");
                mrow["model_ms"]=number_field(mrow,"model_ms")+number_field(out,"model_ms")/finished.at("results").size();
                mrow["model_time_scope"]="amortized_batch_cost";
                mrow["status"]="completed";mrow["finish_ms"]=finished_at;
                mstate.status="completed";mstate.group.clear();++terminal;
            }
            return;
        }
        if(quantum=="atomic"||quantum=="finish"){
            if(out.contains("prepare"))record_prepare_telemetry(row,out.at("prepare"));
            add_artifacts(record,out.at("finish"),work,phase.value("artifact_dependencies",Json::array()));
            if(!row.contains("outputs"))row["outputs"]=Json::array();
            row["outputs"].push_back(out.at("finish"));
            row["model_ms"]=number_field(row,"model_ms")+number_field(out,"model_ms")+number_field(out.at("finish"),"model_ms");
            row["status"]="completed";row["finish_ms"]=now();state.status="completed";state.group.clear();++terminal;
            return;
        }
        if(out.contains("prepare"))record_prepare_telemetry(row,out.at("prepare"));
        if(out.contains("pair_prepare")&&!task.assignment.value("pair",std::string()).empty())
            record_prepare_telemetry(rows[index.at(task.assignment.at("pair").get<std::string>())],out.at("pair_prepare"));
        if(out.contains("transfer")){
            const auto& transfer=out.at("transfer");
            row["transfers"]=row.value("transfers",0)+1;
            row["transfer_bytes"]=number_field(row,"transfer_bytes")+number_field(transfer,"bytes");
            row["transfer_export_ms"]=number_field(row,"transfer_export_ms")+number_field(transfer,"export_ms")+number_field(transfer,"write_ms");
        }
        if(out.value("pre_advance_doorbell_stop",false)){
            if(out.contains("prepare"))row["model_ms"]=number_field(row,"model_ms")+number_field(out.at("prepare"),"model_ms");
            state.prepared=true;
            state.status="suspended";
            const std::string pair=task.assignment.value("pair",std::string());
            if(!pair.empty()){
                auto& pstate=live[index.at(pair)];
                pstate.prepared=true;
                pstate.status="suspended";
            }
            return;
        }
        const auto& ranks=out.at("ranks");
        const auto expected_form=parse_form(task.assignment.value("form",std::string("sp1")));
        const double expected_threshold=number_field(task.assignment,"cache_threshold",0.);
        const bool expected_folded=task.assignment.value("cfg_folded",false);
        Json applied;
        for(const auto& rank:ranks){
            const auto& actual=rank.at("applied");
            if(actual.at("degree")!=expected_form.degree()||actual.at("cfg")!=expected_form.cfg||actual.at("ring")!=expected_form.ring
                    ||number_field(actual,"cache_threshold")!=expected_threshold||actual.value("cfg_folded",false)!=expected_folded)
                throw std::runtime_error("worker execution differs from planned form, cache threshold or CFG-fold mode");
            if(applied.is_null())applied=actual;
            else if(actual!=applied)throw std::runtime_error("worker ranks disagree on applied execution");
        }
        Json trace={{"planned_form",form_name(expected_form)},{"group",task.assignment.at("group")},{"planned_cache_threshold",expected_threshold},
                    {"planned_cfg_folded",expected_folded},{"applied",applied}};
        if(!row.contains("execution"))row["execution"]=Json::array();
        if(row["execution"].empty()||row["execution"].back()!=trace)row["execution"].push_back(trace);
        row["reused_forwards"]=row.value("reused_forwards",0)+applied.value("reused_forwards",0);
        double model=0;for(const auto& rank:ranks)model=std::max(model,number_field(rank,"model_ms"));
        // Wan's counters are per rank (CFG2 runs half the profile's guidance
        // blocks). Also use observed counts for non-continuous quanta: an odd
        // interleaved request executes a complete two-branch block pair.
        const bool rank_local_blocks=workers.backend(video(state.group.front()))=="wan";
        int reported=-1;
        if(task.continuous||rank_local_blocks){
            auto count=[](const Json& rank,const char* key){
                if(std::string(key)=="events"&&rank.contains("executed_units"))return int_field(rank,"executed_units",0,1);
                int value=-1;
                for(const auto& event:rank.value(key,Json::array()))if(event.contains("blocks")){
                    if(value<0)value=0;
                    value+=int_field(event,"blocks",0,1);
                }
                return value;
            };
            bool first_rank=true;
            for(const auto& rank:ranks){
                const int value=count(rank,"events");
                if(value<0&&(task.continuous||quantum.is_number_integer()))
                    throw std::runtime_error("block quantum without an executed block count");
                if(!first_rank&&value!=reported)throw std::runtime_error("worker ranks disagree on executed block count");
                if(value>=0&&!task.assignment.value("pair",std::string()).empty()&&count(rank,"pair_events")!=value)
                    throw std::runtime_error("paired workers disagree on executed block count");
                reported=value;first_rank=false;
                // Mock ranks have no collective doorbell; retain their original
                // rank-zero counter contract rather than imposing Wan consensus.
                if(!rank_local_blocks)break;
            }
        }
        // Time a quantum spends outside its slowest rank's worker operation: dispatch, RPC, queueing and the
        // controller noticing the reply. Regroup quanta (transfers) and lockstep pairs are left out.
        if(!out.contains("transfer")&&task.assignment.value("pair",std::string()).empty()){
            double operation=0;for(const auto& rank:ranks)operation=std::max(operation,number_field(rank,"worker_operation_ms"));
            row["worker_operation_ms"]=number_field(row,"worker_operation_ms")+operation;
            row["dispatch_gap_ms"]=number_field(row,"dispatch_gap_ms")+std::max(0.,elapsed-operation);
            row["timed_quanta"]=row.value("timed_quanta",0)+1;
        }
        auto settle=[&](Json& arow,Live& astate,const char* progress_key){
            const auto progress=ranks.at(0).at(progress_key);
            for(const auto& rank:ranks){
                const auto& other=rank.at(progress_key);
                if(other.value("step",-1)!=progress.value("step",-2)
                        ||other.value("cfg_folded",false)!=progress.value("cfg_folded",false)
                        ||other.value("cfg_fold_eligible",false)!=progress.value("cfg_fold_eligible",false))
                    throw std::runtime_error("SP ranks disagree on solver progress or CFG-fold state");
            }
            arow["model_ms"]=number_field(arow,"model_ms")+model;
            if(progress.contains("cfg_fold"))arow["cfg_fold"]=progress.at("cfg_fold");
            const int executed=reported>=0?reported:quantum=="step"?cost.execution_blocks_per_step(arow.at("class").get<std::string>(),expected_form,rank_local_blocks,expected_folded)-astate.blocks:quantum.get<int>();
            const int previous_step=astate.steps;
            astate.steps=progress.value("step",0);astate.total=progress.value("steps",0);astate.prepared=true;
            astate.cfg_folded=progress.value("cfg_folded",expected_folded);
            astate.cfg_fold_eligible=progress.value("cfg_fold_eligible",false);
            if(progress.value("transfer_ready",true)||astate.steps>previous_step)astate.blocks=0;
            else astate.blocks=progress.contains("block")?progress.value("block",0):astate.blocks+executed;
            astate.status="suspended";
        };
        const bool first=!state.prepared;
        double operation_ms=0;
        for(const auto& rank:ranks)operation_ms=std::max(operation_ms,number_field(rank,"worker_operation_ms"));
        if(!row.contains("advance_samples"))row["advance_samples"]=Json::array();
        row["advance_samples"].push_back({
            {"event_id",task.timing->value.at("event_id")},
            {"form",form_name(expected_form)},{"group",task.assignment.at("group")},{"cfg_folded",expected_folded},
            {"reason",task.assignment.at("reason")},{"first",first},{"transfer",out.contains("transfer")},
            {"steady",!first&&!out.contains("transfer")&&task.assignment.value("pair",std::string()).empty()},
            {"start_ms",task.started},{"finish_ms",task.started+elapsed},
            {"elapsed_ms",elapsed},{"model_ms",model},{"worker_operation_ms",operation_ms}
        });
        const int executed_main=reported>=0?reported:quantum=="step"?cost.execution_blocks_per_step(row.at("class").get<std::string>(),expected_form,rank_local_blocks,expected_folded)-state.blocks:quantum.get<int>();
        settle(row,state,"progress");
        if(!first&&executed_main>0&&!out.contains("transfer")&&task.assignment.value("pair",std::string()).empty()){
            if(task.after_other){state.resumed_ms+=model;state.resumed_blocks+=executed_main;++state.resumes;}
            else{state.steady_ms+=model;state.steady_blocks+=executed_main;}
        }
        const std::string pair=task.assignment.value("pair",std::string());
        if(!pair.empty()){
            const std::size_t j=index.at(pair);auto& prow=rows[j];auto& pstate=live[j];
            if(out.contains("pair_transfer")){
                const auto& transfer=out.at("pair_transfer");
                prow["transfers"]=prow.value("transfers",0)+1;
                prow["transfer_bytes"]=number_field(prow,"transfer_bytes")+number_field(transfer,"bytes");
                prow["transfer_export_ms"]=number_field(prow,"transfer_export_ms")+number_field(transfer,"export_ms")+number_field(transfer,"write_ms");
            }
            prow["action_ms"][key]=prow.value("action_ms",Json::object()).value(key,0.)+elapsed;
            prow["quanta"]=prow.value("quanta",0)+1;
            settle(prow,pstate,"pair_progress");
            // Inside a step the pair must resume together; at a step boundary it dissolves.
            const bool inside=state.blocks>0&&pstate.blocks>0;
            state.pair=inside?pair:std::string();pstate.pair=inside?row.at("request_id").get<std::string>():std::string();
        }else state.pair.clear();
    };
    try{while(terminal<rows.size()){
        check();
        bool progressed=false;
        for(auto it=tasks.begin();it!=tasks.end();){
            if(it->result.wait_for(std::chrono::seconds(0))!=std::future_status::ready){++it;continue;}
            if(cutoff_enabled)check();
            Json out;
            try{out=it->result.get();if(cutoff_enabled)check();}catch(const std::exception& error){
                if(is_cutoff(std::current_exception()))throw;
                auto& row=rows[it->row];row["status"]="failed";row["error"]=error.what();row["finish_ms"]=now();++terminal;
                record["failures"].push_back({{"request_id",row.at("request_id")},{"error",error.what()}});
                if(it->assignment.contains("batch"))for(const auto& member:it->assignment.at("batch")){
                    const auto m=index.at(member.get<std::string>());if(m==it->row)continue;
                    auto& mrow=rows[m];mrow["status"]="failed";mrow["error"]=error.what();mrow["finish_ms"]=now();++terminal;
                    record["failures"].push_back({{"request_id",mrow.at("request_id")},{"error",error.what()}});
                }
                if(cutoff_enabled&&it->assignment.contains("pair"))
                    note_rpc_failure(rows[index.at(it->assignment.at("pair").get<std::string>())],std::current_exception());
                throw;
            }
            try{finish_task(*it,out);}catch(...){note_rpc_failure(rows[it->row],std::current_exception());throw;}
            persist_timing(*it,"completed");it=tasks.erase(it);progressed=true;
        }
        for(std::size_t i=0;i<rows.size();++i){
            auto& state=live[i];
            if(state.status!="decoding")continue;
            const double poll_now=now();
            if(poll_now-state.last_decode_poll_ms<5.0)continue;
            state.last_decode_poll_ms=poll_now;
            auto polled=invoke(video(state.decode_gpu),{{"op","decode_poll"},{"ticket",state.decode_ticket},{"wait",false}},rows[i]);
            if(!polled.value("ready",false))continue;
            auto& row=rows[i];
            row["model_ms"]=number_field(row,"model_ms")+number_field(polled,"model_ms");
            row["decode_model_ms"]=number_field(polled,"model_ms");
            row["async_decode_wall_ms"]=now()-state.decode_started;
            row["async_decode_event"]=polled.value("event",Json::object());
            add_artifacts(record,polled,work,phase.value("artifact_dependencies",Json::array()));
            if(!row.contains("outputs"))row["outputs"]=Json::array();
            row["outputs"].push_back(polled);
            invoke(video(state.decode_gpu),{{"op","release"},{"state_id",row.at("request_id")}},row);
            row["status"]="completed";row["finish_ms"]=now();
            state.status="completed";state.group.clear();state.resident.clear();state.decode_ticket.clear();state.decode_gpu=-1;
            ++terminal;progressed=true;
        }
        const double instant=now();
        for(int gpu=0;gpu<gpus;++gpu)if(!busy[gpu])for(const auto& id:stale[gpu])invoke(video(gpu),{{"op","release"},{"state_id",id}},rows[index.at(id)]);
        for(int gpu=0;gpu<gpus;++gpu)if(!busy[gpu])stale[gpu].clear();
        auto image_worker_busy=[&](int gpu){
            for(const auto& task:tasks){
                if(rows[task.row].at("kind")!="image")continue;
                const auto& grp=task.assignment.at("group");
                if(std::find(grp.begin(),grp.end(),Json(gpu))!=grp.end())return true;
            }
            return false;
        };
        auto video_worker_busy=[&](int gpu){
            for(const auto& task:tasks){
                if(rows[task.row].at("kind")!="video")continue;
                const auto& grp=task.assignment.at("group");
                if(std::find(grp.begin(),grp.end(),Json(gpu))!=grp.end())return true;
            }
            return false;
        };
        for(int gpu=0;gpu<gpus;++gpu)if(!busy[gpu]&&!video_worker_busy(gpu)&&!stale_tickets[gpu].empty()){
            for(const auto& t:stale_tickets[gpu]){
                try{workers.get(video(gpu)).call({{"op","encode_discard"},{"ticket",t}},budget.deadline());}catch(...){}
            }
            stale_tickets[gpu].clear();
        }
        auto pick_image_encode_gpu=[&](){
            const int start=static_cast<int>(image_encoders++%gpus);
            int best=start;
            auto score=[&](int gpu){
                int inflight=0;
                for(const auto& s:live)if(s.status=="encoding"&&s.encode_gpu==gpu)++inflight;
                return std::make_tuple(image_worker_busy(gpu)?1:0,inflight,busy[gpu]?1:0);
            };
            auto best_score=score(best);
            for(int offset=1;offset<gpus;++offset){
                const int candidate=(start+offset)%gpus;
                const auto cand_score=score(candidate);
                if(cand_score<best_score){best=candidate;best_score=cand_score;}
            }
            return best;
        };
        auto pick_video_encode_gpu=[&](){
            const int start=static_cast<int>(encoders++%gpus);
            int best=start;
            auto score=[&](int gpu){
                int inflight=0;
                for(std::size_t k=0;k<live.size();++k)
                    if(live[k].status=="encoding"&&live[k].encode_gpu==gpu&&rows[k].at("kind")=="video")++inflight;
                return std::make_tuple(video_worker_busy(gpu)?1:0,inflight,busy[gpu]?1:0);
            };
            auto best_score=score(best);
            for(int offset=1;offset<gpus;++offset){
                const int candidate=(start+offset)%gpus;
                const auto cand_score=score(candidate);
                if(cand_score<best_score){best=candidate;best_score=cand_score;}
            }
            return best;
        };
        for(std::size_t i=0;i<rows.size();++i){
            auto& state=live[i];auto& row=rows[i];
            const bool is_image=row.at("kind")=="image";
            if(state.status=="pending"&&number_field(row,"arrival_ms")<=instant){
                const bool should_encode=(row.at("kind")=="video"&&prefetch)||(is_image&&image_prefetch);
                state.status=should_encode?"encoding":"waiting";
                if(state.status=="encoding"){
                    state.encode_gpu=is_image?pick_image_encode_gpu():pick_video_encode_gpu();
                    row["encode_gpu"]=state.encode_gpu;
                }
                progressed=true;
            }
            if(state.status!="encoding")continue;
            if(state.ticket.empty()&&(is_image?image_worker_busy(state.encode_gpu):video_worker_busy(state.encode_gpu))){
                const int repicked=is_image?pick_image_encode_gpu():pick_video_encode_gpu();
                if(!(is_image?image_worker_busy(repicked):video_worker_busy(repicked))){
                    state.encode_gpu=repicked;
                    row["encode_gpu"]=state.encode_gpu;
                }
            }
            if(is_image?image_worker_busy(state.encode_gpu):video_worker_busy(state.encode_gpu))continue;
            const auto encode_worker=is_image?image(state.encode_gpu):video(state.encode_gpu);
            if(state.ticket.empty()){
                state.ticket=invoke(encode_worker,{{"op","encode_start"},{"spec",spec_of(row)}},row).at("ticket").get<std::string>();
                row["encode_submit_ms"]=now();
                row["encode_worker"]=encode_worker;
                progressed=true;
            }else{
                const double poll_now=now();
                if(poll_now-state.last_encode_poll_ms<5.0)continue;
                state.last_encode_poll_ms=poll_now;
                const auto polled=invoke(encode_worker,{{"op","encode_poll"},{"ticket",state.ticket}},row);
                if(polled.value("ready",false)){
                    if(polled.value("shm_staged",false))state.shm_staged=true;
                    state.status="waiting";row["encode_ready_ms"]=now();progressed=true;
                    // CPU encoding time and, with the negative-prompt cache, whether it hit.
                    if(polled.contains("model_ms"))row["encode_model_ms"]=number_field(polled,"model_ms");
                    if(polled.contains("push_ms"))row["encode_push_ms"]=number_field(polled,"push_ms");
                    if(polled.contains("cached")&&polled.at("cached").is_boolean())row["encode_pool_cached"]=polled.at("cached");
                    if(polled.contains("negative_prompt_cache")&&polled.at("negative_prompt_cache").is_object())row["negative_prompt_cache"]=polled.at("negative_prompt_cache").value("outcome",std::string("unknown"));
                }
            }
        }
        if(std::find(busy.begin(),busy.end(),false)!=busy.end()){
            Json requests=Json::array();
            for(std::size_t i=0;i<rows.size();++i){
                const auto& state=live[i];
                if(state.status!="waiting"&&state.status!="suspended")continue;
                const auto& row=rows[i];
                Json view={{"request_id",row.at("request_id")},{"kind",row.at("kind")},{"class",row.at("class")},
                    {"arrival_ms",row.at("arrival_ms")},{"deadline_ms",row.at("deadline_ms")},{"status",state.status},{"ready",true},
                    {"prepared",state.prepared},{"steps_done",state.steps},{"blocks_done",state.blocks},{"group",state.group},{"decode_done",state.decode_done}};
                if(row.at("kind")=="video"){
                    view["rank_local_blocks"]=workers.backend(video(state.group.empty()?0:state.group.front()))=="wan";
                    view["cfg_folded"]=state.cfg_folded;
                    view["cfg_fold_eligible"]=state.cfg_fold_eligible;
                }
                if(row.at("kind")=="image"){
                    view["continuous_batch"]=continuous_batch>0;
                    Json batch_spec=spec_of(row);
                    for(const auto* field:{"request_id","prompt","negative_prompt","seed"})batch_spec.erase(field);
                    view["batch_spec"]=batch_spec;
                }
                if(state.total>0)view["steps_total"]=state.total;
                if(!state.group.empty())view["form"]=state.form;
                if(!state.pair.empty())view["pair"]=state.pair;
                // Under the export transport the encoding worker prepares the request and exports
                // it to whatever group the planner picks, so the prompt's GPU is no constraint;
                // the other transports must prepare on that GPU so its prompt state remains local.
                if(!state.prepared&&!state.ticket.empty()&&transport!="export")view["affinity"]=state.encode_gpu;
                requests.push_back(view);
            }
            if(!requests.empty()){
                Json flags=Json::array(),decode_flags=Json::array();
                for(bool flag:busy)flags.push_back(flag);
                for(int gpu=0;gpu<gpus;++gpu)decode_flags.push_back(false);
                for(const auto& state:live)if(state.status=="decoding")decode_flags[state.decode_gpu]=true;
                int sp_running=0;   // SP advances in flight contend for the host bridge
                for(const auto& task:tasks){const auto& a=task.assignment;sp_running+=parse_form(a.value("form","sp"+std::to_string(a.at("degree").get<int>()))).sp>=2&&a.at("quantum")!="atomic"&&a.at("quantum")!="finish";}
                const Json plan_input={{"now_ms",now()},{"gpus",gpus},{"busy",flags},{"async_decoding",decode_flags},
                    {"requests",requests},{"sp_groups_running",sp_running}};
                Json plan;
                bool relevant=true;
                if(decisions&&!decisions->focus.empty()){
                    relevant=false;
                    // Match the planner's actual candidate filter before any
                    // diagnostic row/byte truncation. Every assignment member
                    // (including pair/batch partners) comes from this set.
                    const double plan_now=plan_input.at("now_ms").get<double>();
                    for(const auto& candidate:requests){
                        const auto status=candidate.value("status",std::string());
                        if((status=="waiting"||status=="suspended")&&candidate.value("ready",true)
                            &&number_field(candidate,"arrival_ms")<=plan_now&&decisions->matches(candidate.at("request_id")))relevant=true;
                    }
                    for(const auto& task:tasks){
                        if(decisions->matches(rows[task.row].at("request_id")))relevant=true;
                        if(task.assignment.contains("pair")&&decisions->matches(task.assignment.at("pair")))relevant=true;
                        if(task.assignment.contains("batch"))for(const auto& id:task.assignment.at("batch"))if(decisions->matches(id))relevant=true;
                    }
                    for(std::size_t i=0;i<live.size();++i)
                        if(live[i].status=="decoding"&&decisions->matches(rows[i].at("request_id")))relevant=true;
                }
                auto trace=decisions?decisions->begin(relevant):nullptr;
                const double planning_start=now();
                auto account_plan=[&](bool failed){const double elapsed=now()-planning_start;
                    planning["calls"]=planning.at("calls").get<int>()+1;planning["failed"]=planning.at("failed").get<int>()+int(failed);
                    planning["total_ms"]=number_field(planning,"total_ms")+elapsed;
                    planning["max_ms"]=std::max(number_field(planning,"max_ms"),elapsed);};
                try{plan=cluster_plan(plan_input,policy,cost,options,trace.get());}catch(...){account_plan(true);if(decisions&&trace)++decisions->failed_calls;throw;}
                account_plan(false);
                std::string decision_id;
                if(trace){
                    Json running=Json::array(),decoding=Json::array();std::size_t omitted_owners=0;
                    for(const auto& task:tasks){
                        if(running.size()>=64){++omitted_owners;continue;}
                        Json owner={{"event_id",task.diagnostic_event_id},{"request_id",rows[task.row].at("request_id")},
                            {"group",task.assignment.at("group")},{"continuous",task.continuous},{"quantum",task.assignment.at("quantum")}};
                        if(task.assignment.contains("pair"))owner["pair"]=task.assignment.at("pair");
                        running.push_back(std::move(owner));
                    }
                    for(std::size_t i=0;i<live.size();++i)if(live[i].status=="decoding"){
                        if(decoding.size()>=64){++omitted_owners;continue;}
                        decoding.push_back({{"request_id",rows[i].at("request_id")},{"gpu",live[i].decode_gpu}});
                    }
                    for(auto& row:trace->rows){
                        const auto& spec=rows[index.at(row.at("request_id").get<std::string>())].at("spec");
                        Json shape=Json::object();
                        for(const auto* field:{"height","width","num_frames","frames","steps"})if(spec.contains(field))shape[field]=spec.at(field);
                        row["generation_shape"]=std::move(shape);
                    }
                    trace->enforce_bounds();
                    Json selected=Json::array();std::size_t omitted_assignments=0;
                    for(const auto& assignment:plan.at("assignments")){
                        if(selected.size()>=64){++omitted_assignments;continue;}
                        selected.push_back(assignment);
                    }
                    decision_id=decisions->retain({{"now_ms",plan_input.at("now_ms")},{"busy",flags},{"async_decoding",decode_flags},
                        {"sp_groups_running",sp_running},{"running_tasks",running},{"decoding",decoding},{"omitted_owners",omitted_owners},
                        {"assignments",selected},{"omitted_assignments",omitted_assignments},
                        {"control_mode",doorbell?"doorbell_may_expand_numeric_quantum_to_continuous_step":"planner_quantum"}},*trace);
                }
                if(tiered&&plan.contains("tiered_hints")){
                    std::set<std::size_t> offloaded;
                    std::set<int> assigned;
                    for(const auto& assignment:plan.at("assignments"))for(const auto& gpu:assignment.at("group"))assigned.insert(gpu.get<int>());
                    for(const auto& rid:plan.at("tiered_hints").at("offload")){
                        const auto k=index.at(rid.get<std::string>());const auto& state=live[k];
                        if(!state.offloaded&&state.status=="suspended"&&state.blocks==0&&state.decode_done==0&&state.steps<state.total
                                &&state.group.size()==1&&!busy[state.group.front()]){
                            move_state(k,true);offloaded.insert(k);
                        }
                    }
                    for(const auto& rid:plan.at("tiered_hints").at("prefetch")){
                        const auto k=index.at(rid.get<std::string>());const auto& state=live[k];
                        if(state.offloaded&&!offloaded.count(k)&&state.group.size()==1&&!busy[state.group.front()]&&!assigned.count(state.group.front()))
                            move_state(k,false);
                    }
                }
                for(const auto& assignment:plan.at("assignments")){launch(assignment,decisions?&decision_id:nullptr);progressed=true;}
            }
        }
        if(doorbell){
            std::vector<PreemptionGroup> candidates;
            const double instant=now();
            int sp_running=0;
            for(const auto& task:tasks)if(task.continuous&&parse_form(task.assignment.at("form").get<std::string>()).sp>=2)++sp_running;
            for(const auto& task:tasks)if(task.continuous){
                const auto& row=rows[task.row];const auto& state=live[task.row];
                const auto klass=row.at("class").get<std::string>();
                const auto form=parse_form(task.assignment.at("form").get<std::string>());
                const bool wan=workers.backend(video(state.group.front()))=="wan";
                // Mock progress uses the spec's full block counter. Wan
                // splits CFG branches across ranks, or omits one when folded.
                const bool folded=wan&&task.assignment.value("cfg_folded",false);
                const int rank_blocks=wan?cost.execution_blocks_per_step(klass,form,true,folded)
                                         :int_field(row.at("spec"),"blocks_per_step",1);
                const double block=cost.step_ms(klass,form,std::max(1,sp_running),&state.group)/rank_blocks;
                const int blocks_total=rank_blocks;
                int blocks_per_iteration=1;
                if(wan&&form.cfg==1&&!folded&&number_field(row.at("spec"),"guidance_scale",5.)>1.)
                    for(int gpu:state.group)if(interleaved_workers.count(video(gpu)))blocks_per_iteration=2;
                // Match advance.check_blocks and Wan's one-checkpoint-late
                // group flag. Single-rank Wan and mock poll immediately.
                const auto timing=estimate_preemption_timing(block,blocks_total,state.blocks,instant-task.scheduling_started,
                    doorbell_check_blocks,blocks_per_iteration,wan&&form.degree()>1);
                candidates.push_back({row.at("request_id").get<std::string>(),state.group,timing.stop_ms,timing.remaining_ms,cost.resume_ms(klass),task.rung});
            }
            std::vector<int> waiting;
            std::vector<PreemptionDemand> deadline_waiting;
            const bool lazy=bool_field(options,"lazy_preemption",true);
            for(std::size_t k=0;k<rows.size();++k){
                const auto& row=rows[k];const auto& state=live[k];
                if((state.status!="waiting"&&state.status!="suspended")||row.at("kind")!="image")continue;
                const int affinity=state.prepared&&!state.group.empty()?state.group.front():-1;
                if(lazy){
                    Json view={{"class",row.at("class")},{"prepared",state.prepared},{"steps_done",state.steps},
                               {"continuous_batch",continuous_batch>0}};
                    if(state.total>0)view["steps_total"]=state.total;
                    const double service=cost.remaining_ms(view,Form{},1)+cost.quantum_overhead_ms();
                    deadline_waiting.push_back({affinity,number_field(row,"deadline_ms"),service});
                }
                waiting.push_back(affinity);
            }
            if(!waiting.empty()){
                std::vector<std::string> selected;
                if(lazy){
                    const auto decision=select_deadline_preemption_groups(instant,deadline_waiting,candidates);
                    selected=decision.selected;bells.deadline_skips+=decision.deadline_skips;
                }else selected=select_preemption_groups(waiting,candidates);
                for(auto& task:tasks)if(task.continuous&&!task.rung&&std::find(selected.begin(),selected.end(),rows[task.row].at("request_id").get<std::string>())!=selected.end()){
                    for(const auto& gpu:task.assignment.at("group"))bells.ring(gpu.get<int>());
                    task.rung=true;task.rung_ms=now();
                }
            }
        }
        if(!progressed)wait_until(monotonic_seconds()+.0005);
    }}catch(...){
        const auto error=std::current_exception();
        if(!cutoff_enabled){
            for(auto& task:tasks)if(task.result.valid())task.result.wait();
            for(auto& task:tasks)persist_timing(task,"aborted");
            save_planning();
            for(auto& row:rows){if(!row.contains("status")||row["status"]=="pending")row["status"]="interrupted";if(!row.contains("finish_ms"))row["finish_ms"]=now();row["observation_ms"]=now();}
            record["requests"].insert(record["requests"].end(),rows.begin(),rows.end());throw;
        }
        cutoff_observed=now();cutoff_triggered=is_cutoff(error);
        observation_end=cutoff_triggered?cutoff_ms:std::min(cutoff_observed,cutoff_ms);
        if(!cutoff_triggered)replay_error=error;
        // RPC threads own their process/pipe state. Their replay-bound deadline
        // terminates them; never race a controller stop() against an active call.
        for(auto& task:tasks)if(task.result.valid()){
            try{task.result.get();}catch(...){
                const auto pending=std::current_exception();
                if(!is_cutoff(pending)){
                    keep_failure(replay_error,pending);
                    std::string reason;
                    try{std::rethrow_exception(pending);}catch(const std::exception& failure){reason=failure.what();}
                    auto fail=[&](std::size_t k){
                        auto& row=rows[k];if(row.value("status","")=="completed")return;
                        row["status"]="failed";row["error"]=reason;row["failure_observed_ms"]=now();
                        record["failures"].push_back({{"request_id",row.at("request_id")},{"error",reason}});
                    };
                    fail(task.row);
                    if(task.assignment.contains("batch"))for(const auto& member:task.assignment.at("batch"))fail(index.at(member.get<std::string>()));
                    if(task.assignment.contains("pair"))fail(index.at(task.assignment.at("pair").get<std::string>()));
                }
            }
        }
        for(auto& task:tasks)persist_timing(task,cutoff_triggered?"censored":"aborted");
    }
    if(cutoff_enabled){
        if(observation_end<0){
            observation_end=std::min(now(),cutoff_ms);cutoff_observed=now();
            cutoff_triggered=std::any_of(rows.begin(),rows.end(),[&](const Json& row){
                return row.value("status","")!="completed"||number_field(row,"finish_ms")>cutoff_ms;});
        }
        for(auto& row:rows){
            const auto status=row.value("status","");
            if(status!="failed"&&(status!="completed"||number_field(row,"finish_ms")>observation_end)){
                row["status"]=number_field(row,"arrival_ms")>observation_end?"not_executed":cutoff_triggered?"censored":"interrupted";
                if(row.contains("finish_ms")){row["completion_observed_after_window_ms"]=row.at("finish_ms");row.erase("finish_ms");}
            }else if(status=="failed"&&row.contains("finish_ms")&&number_field(row,"finish_ms")>observation_end){
                row["failure_observed_ms"]=row.at("finish_ms");row.erase("finish_ms");
            }
            row["observation_ms"]=observation_end;
        }
    }
    for(std::size_t i=0;i<rows.size();++i){
        auto& row=rows[i];
        if(row.value("preempting",false))row["preemption_reason"]="block_preempt";
        row.erase("preempting");
        const auto& st=live[i];
        if(st.steady_blocks<=0)continue;
        const double steady=st.steady_ms/st.steady_blocks;
        row["block_ms_steady"]=steady;row["resumes"]=st.resumes;
        if(st.resumed_blocks>0){row["block_ms_resumed"]=st.resumed_ms/st.resumed_blocks;row["resume_extra_ms"]=st.resumed_ms-st.resumed_blocks*steady;}
    }
    save_planning();
    Json result={{"requests",rows},{"metrics",request_summary(rows)},{"elapsed_ms",cutoff_enabled?observation_end:now()},{"warmup",warmed},{"controller_planning",planning},
            {"by_kind",{{"video",kind_summary(rows,"video")},{"image",kind_summary(rows,"image")}}},
            {"policy",policy},{"gpus",gpus},{"sp_transport",transport},{"real_backends_pinned_sp1",!fixtures&&transport=="pinned"},
            {"async_rank0_decode",async_decode},{"continuous_batch",cb_stats},
            {"doorbell",{{"enabled",doorbell},{"continuous_quanta",bells.continuous},{"rings",bells.rings},
                          {"deadline_skip_events",bells.deadline_skips}}}};
    if(cutoff_enabled){
        Json counts={{"planned",rows.size()},{"completed",0},{"failed",0},{"censored",0},{"not_executed",0},{"interrupted",0}};
        for(const auto& row:rows){const auto status=row.at("status").get<std::string>();counts[status]=counts.at(status).get<int>()+1;}
        auto receipt=phase.at("replay_cutoff");
        receipt.update({{"clock","controller_replay_monotonic_ms"},{"status",cutoff_triggered?"triggered":"not_reached"},
            {"observation_horizon_ms",observation_end},{"observed_at_ms",cutoff_observed},{"replay_origin_monotonic_seconds",origin},
            {"config_sha256",record.at("config_sha256")},{"observation_id",phase.at("_observation_id")},{"policy",policy},
            {"counts",counts},{"cleanup_verified",false},{"physical_stop_bound",false}});
        result["replay_cutoff"]=receipt;
        result["status"]=replay_error?"failed":cutoff_triggered?"censored":"completed";
        if(replay_error)try{std::rethrow_exception(replay_error);}catch(const std::exception& failure){result["error"]=failure.what();}
    }
    return result;
}

Json operations(const Json& phase,Workers& workers,Budget& budget,Json& record,const fs::path& work,const fs::path& cell_dir){
    const auto names=phase.value("workers",Json::array({phase.value("worker","model")}));
    if(!names.is_array() || names.empty())throw std::invalid_argument("action phase requires workers");
    const auto actions=phase.at("actions");if(!actions.is_array() || actions.empty())throw std::invalid_argument("SP phase requires actions");
    Json rows=Json::array();
    for(auto action:actions){
        budget.check();if(cancelled())throw std::runtime_error("run cancelled");
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
    const Json mps=phase.value("mps",Json::object());
    auto owned_pids=[&](const Workers& workers){
        auto pids=workers.pids();
        if(!mps.empty())pids.push_back(mps.at("server_pid").get<int>());
        return pids;
    };
    bool cleanup=true,censored=false;std::string error;std::vector<int> attempt_pids;
    auto cells=expand_matrix(phase);
    std::vector<std::size_t> order;for(std::size_t i=0;i<cells.size();++i)order.push_back(i);
    if(phase.value("paired",false))for(std::size_t i=cells.size();i>0;--i)order.push_back(i-1);
    auto preserve_unstarted=[&](std::size_t position,const std::string& reason){
        const auto& cell=cells[order[position]];
        if(!cell.contains("replay_cutoff"))return;
        const auto observation=id+"/"+std::to_string(attempt_number)+"/"+std::to_string(position);
        auto rows=study_workload(cell);
        for(auto& row:rows){
            row.update({{"phase",id},{"attempt",attempt_number},{"cell",order[position]},
                {"order",position<cells.size()?"forward":"reverse"},{"observation_id",observation},
                {"status","not_executed"},{"observation_ms",0.},{"not_executed_reason",reason}});
            record["requests"].push_back(row);
        }
        return;
    };
    for(std::size_t position=0;position<order.size();++position){
        auto cell=cells[order[position]];const auto cell_dir=attempt_dir/("cell-"+std::to_string(position));
        cell["_observation_id"]=id+"/"+std::to_string(attempt_number)+"/"+std::to_string(position);
        cell["_attempt"]=attempt_number;cell["_cell"]=order[position];cell["_order"]=position<cells.size()?"forward":"reverse";
        Json definitions=request.at("config").at("workers");
        if(cell.contains("worker_config"))definitions[cell.value("worker","model")]["config"].merge_patch(cell.at("worker_config"));
        // video_worker_options: the same option patch for every slot's video worker of a cluster cell,
        // so a matrix axis can switch a worker feature (e.g. the negative-prompt cache) per cell.
        if(cell.value("kind","online")=="cluster"&&cell.contains("video_worker_options"))
            for(const auto& slot:cell.at("gpus")){
                auto& config=definitions[slot.at("video").get<std::string>()]["config"];
                if(!config.is_object())config=Json::object();
                if(!config.contains("options")||!config.at("options").is_object())config["options"]=Json::object();
                config["options"].merge_patch(cell.at("video_worker_options"));
            }
        if(cell.value("kind","online")=="cluster"&&cell.contains("image_worker_options"))
            for(const auto& slot:cell.at("gpus")){
                auto& config=definitions[slot.at("image").get<std::string>()]["config"];
                if(!config.is_object())config=Json::object();
                if(!config.contains("options")||!config.at("options").is_object())config["options"]=Json::object();
                config["options"].merge_patch(cell.at("image_worker_options"));
            }
        // Decision D4: isolate encoder_pool /dev/shm slabs per matrix cell so no policy
        // cell inherits warm /dev/shm prompt slabs written by a previous cell or phase.
        std::set<fs::path> cell_shm_dirs;
        for(auto it=definitions.begin();it!=definitions.end();++it){
            auto& def=it.value();
            if(!def.is_object())continue;
            const std::string backend_name=def.value("backend",std::string());
            auto& cfg=def["config"];
            if(!cfg.is_object())cfg=Json::object();
            if(!cfg.contains("options")||!cfg.at("options").is_object()){
                if(backend_name!="sd35")continue;
                cfg["options"]=Json::object();
            }
            auto& opts=cfg["options"];
            if((!opts.contains("encoder_pool")||!opts.at("encoder_pool").is_object())&&backend_name=="sd35")
                opts["encoder_pool"]=Json::object();
            if(!opts.contains("encoder_pool")||!opts.at("encoder_pool").is_object())continue;
            auto& pool=opts["encoder_pool"];
            if(pool.value("isolation",std::string("cell"))=="shared"||!pool.value("per_cell_isolation",true))continue;
            const std::string raw_shm=pool.value("shm_dir",std::string("/dev/shm/ditserve-hetero-embeds"));
            const fs::path base_shm=(raw_shm.empty()||raw_shm=="auto")
                ?fs::path("/dev/shm/ditserve-hetero-embeds"):fs::path(raw_shm);
            const fs::path scoped_shm=base_shm
                /segment(request.at("run_id").get<std::string>())
                /segment(id)
                /("attempt-"+std::to_string(attempt_number))
                /("cell-"+std::to_string(position));
            pool["base_shm_dir"]=base_shm.string();
            pool["shm_dir"]=scoped_shm.string();
            pool["per_cell_isolation"]=false;
            pool["cell_scoped"]=true;
            pool["work_dir"]=cell_dir.string();
            cell_shm_dirs.insert(scoped_shm);
        }
        for(const auto& shm_path:cell_shm_dirs){
            std::error_code ignored;
            fs::remove_all(shm_path,ignored);
        }
        Workers workers(definitions,request,cell_dir);
        const auto occupancy_backend=cell.value("occupancy_backend",std::string("nvidia_smi"));
        // occupancy_thread: observe on a background thread every 2 s instead of inside a heartbeat. Every worker call
        // runs the heartbeat first and then every 0.5 s while it waits, all under one mutex, so an inline observation
        // stalls every rank's sends and replies for its whole duration.
        const bool probe_thread=cell.value("occupancy_thread",false);
        OccupancyProbe probe(request,cell_dir/"probe",occupancy_backend);
        double last_observation=0;
        int unknown_streak=0;
        Json tolerated=Json::array();
        std::mutex heartbeat_mutex;
        // The telemetry measures the guard's cost per cell (lock waits, probe durations).
        const double cell_started=monotonic_seconds();
        Json guard_telemetry={{"schema","ditserve.guard-telemetry.v1"},{"occupancy_backend",occupancy_backend},{"cadence_seconds",2.},
            {"occupancy_thread",probe_thread},
            {"probe_timeout_seconds",3.},{"heartbeat_calls",0},{"lock_wait_total_ms",0.},{"lock_wait_max_ms",0.},
            {"heartbeat_total_ms",0.},{"heartbeat_max_ms",0.},{"probe_calls",0},{"probe_total_ms",0.},{"probe_max_ms",0.},
            {"probe_not_exclusive",0},{"events",Json::array()},{"dropped_events",0}};
        auto add_duration=[&](const char* total,const char* maximum,double ms){
            guard_telemetry[total]=guard_telemetry.at(total).get<double>()+ms;
            guard_telemetry[maximum]=std::max(guard_telemetry.at(maximum).get<double>(),ms);
        };
        std::string isolation_error;          // set by an observation; the next heartbeat throws it
        double last_verdict=monotonic_seconds();
        // Called with heartbeat_mutex held.
        auto record_observation=[&](const Json& observed,double probe_started,double probe_ms){
            guard_telemetry["probe_calls"]=guard_telemetry.at("probe_calls").get<int>()+1;
            add_duration("probe_total_ms","probe_max_ms",probe_ms);
            const auto status=observed.value("status","unknown");
            if(status!="exclusive_observed")guard_telemetry["probe_not_exclusive"]=guard_telemetry.at("probe_not_exclusive").get<int>()+1;
            if(guard_telemetry.at("events").size()<1024)guard_telemetry["events"].push_back({{"at_s",probe_started-cell_started},{"probe_ms",probe_ms},{"status",status}});
            else guard_telemetry["dropped_events"]=guard_telemetry.at("dropped_events").get<int>()+1;
            last_verdict=monotonic_seconds();
            if(!isolation_error.empty())return;
            // An answer from another backend than the one configured is not evidence for this cell.
            if(status!="unknown"&&observed.value("occupancy_backend","")!=occupancy_backend)
                isolation_error="GPU occupancy probe answered with another backend: "+observed.dump();
            // A CPU-bound host can delay nvidia-smi past the probe deadline; that alone
            // proves nothing about other processes on the GPU.
            else if(!isolation_holds(observed,unknown_streak))isolation_error="GPU isolation lost: "+observed.dump();
            else if(unknown_streak)tolerated.push_back(observed);
        };
        workers.heartbeat([&]{
            // Collective ranks call concurrently. One observation owns the
            // probe paths, and its timestamp must not race with other ranks.
            const double started=monotonic_seconds();
            std::lock_guard<std::mutex> guard(heartbeat_mutex);
            guard_telemetry["heartbeat_calls"]=guard_telemetry.at("heartbeat_calls").get<int>()+1;
            add_duration("lock_wait_total_ms","lock_wait_max_ms",(monotonic_seconds()-started)*1000);
            struct Finish{std::function<void()> f;~Finish(){f();}} finish{[&]{add_duration("heartbeat_total_ms","heartbeat_max_ms",(monotonic_seconds()-started)*1000);}};
            if(cancelled())throw std::runtime_error("run cancelled");
            budget.check();
            if(!probe_thread){
                if(!devices.empty()&&monotonic_seconds()-last_observation>=2){
                    last_observation=monotonic_seconds();
                    const double probe_started=monotonic_seconds();
                    auto observed=probe.observe(devices,owned_pids(workers),std::min(budget.deadline(),monotonic_seconds()+3));
                    record_observation(observed,probe_started,(monotonic_seconds()-probe_started)*1000);
                }
            }else if(!devices.empty()&&isolation_error.empty()&&monotonic_seconds()-last_verdict>10)
                isolation_error="GPU isolation unverified: no occupancy observation for 10 s";
            if(!isolation_error.empty())throw std::runtime_error(isolation_error);
        });
        // The background prober: one observation at a time, outside the heartbeat mutex; stopped before the
        // cell's workers are counted and stopped.
        std::condition_variable probe_wakeup;bool probe_stop=false;std::thread prober;
        if(probe_thread&&!devices.empty())prober=std::thread([&]{
            for(;;){
                {std::lock_guard<std::mutex> guard(heartbeat_mutex);if(probe_stop)return;}
                const double probe_started=monotonic_seconds();
                Json observed;
                try{observed=probe.observe(devices,owned_pids(workers),std::min(budget.deadline(),monotonic_seconds()+3));}
                catch(const std::exception& failure){observed={{"status","unknown"},{"reason",failure.what()}};}
                const double probe_ms=(monotonic_seconds()-probe_started)*1000;
                std::unique_lock<std::mutex> lock(heartbeat_mutex);
                record_observation(observed,probe_started,probe_ms);
                if(probe_wakeup.wait_for(lock,std::chrono::seconds(2),[&]{return probe_stop;}))return;
            }
        });
        auto stop_prober=[&]{
            {std::lock_guard<std::mutex> guard(heartbeat_mutex);probe_stop=true;}
            probe_wakeup.notify_all();
            if(prober.joinable())prober.join();
        };
        struct StopProber{std::function<void()> f;~StopProber(){f();}} stop_prober_on_exit{stop_prober};
        auto telemetry=[&]{std::lock_guard<std::mutex> guard(heartbeat_mutex);return guard_telemetry;};
        try{
            const auto kind=cell.value("kind","online");
            Json result=cell.contains("actions")?operations(cell,workers,budget,record,work,cell_dir):kind=="cluster"?cluster(cell,workers,budget,record,work):replay(cell,workers,budget,record,work,cell_dir);
            result["cell"]=order[position];result["order"]=position<cells.size()?"forward":"reverse";
            result["status"]=cell.contains("replay_cutoff")?result.value("status",std::string("completed")):"completed";
            if(!result.contains("policy")&&cell.contains("policy"))result["policy"]=cell.at("policy");
            if(!tolerated.empty())result["unknown_observations"]=tolerated;
            result["guard_telemetry"]=telemetry();
            if(!cell_shm_dirs.empty()){
                Json shm_list=Json::array();
                for(const auto& shm_path:cell_shm_dirs)shm_list.push_back(shm_path.string());
                result["encoder_pool_isolation"]={{"isolated",true},{"shm_dirs",shm_list},{"cleaned_after_cell",cell.value("cleanup_shm_pool",true)}};
            }
            for(auto row:result.value("requests",Json::array())){row["phase"]=id;row["attempt"]=attempt_number;row["cell"]=order[position];row["order"]=result["order"];
                row["observation_id"]=id+"/"+std::to_string(attempt_number)+"/"+std::to_string(position);
                row["observation_ms"]=result.value("elapsed_ms",number_field(row,"finish_ms"));record["requests"].push_back(row);}
            attempt["cells"].push_back(result);
            if(result.at("status")=="completed")workers.shutdown(std::min(budget.deadline(),monotonic_seconds()+5));
            else if(result.at("status")=="censored")censored=true;
            else error=result.value("error",std::string("cell failed"));
        }catch(const std::exception& failure){
            error=failure.what();
            if(cell.contains("replay_cutoff")){
                // Startup/warmup failure precedes replay time zero; it is not a
                // typed drain cutoff and cannot consume the offered cohort.
                const auto obs=cell.at("_observation_id");
                const bool materialized=std::any_of(record["requests"].begin(),record["requests"].end(),
                    [&](const Json& row){return row.value("observation_id",Json())==obs;});
                if(!materialized)preserve_unstarted(position,"cell failed before replay: "+error);
            }
            if(!cell.contains("replay_cutoff"))attempt["cells"].push_back({{"cell",order[position]},
                {"status","failed"},{"error",error},{"guard_telemetry",telemetry()}});
            else if(attempt["cells"].size()<=position)attempt["cells"].push_back({{"cell",order[position]},
                {"order",cell.at("_order")},{"policy",cell.at("policy")},{"status","failed"},{"error",error},{"guard_telemetry",telemetry()}});
            else{attempt["cells"].back()["status"]="failed";attempt["cells"].back()["error"]=error;}
        }
        stop_prober();
        const auto pids=owned_pids(workers);attempt_pids.insert(attempt_pids.end(),pids.begin(),pids.end());
        const auto unverified=workers.stop();
        if(cell.value("cleanup_shm_pool",true)){
            for(const auto& shm_path:cell_shm_dirs){
                std::error_code ignored;
                fs::remove_all(shm_path,ignored);
            }
        }
        if(!unverified.empty()){cleanup=false;attempt["unverified_workers"].push_back({{"cell",order[position]},{"workers",unverified}});}
        if(cell.contains("replay_cutoff")){
            // Reusing workers/GPU allocations is permitted only after this
            // cell's process groups and owned GPU occupancy have been checked.
            const auto final_cell=gpu_observation(request,cell_dir/"cleanup",devices,pids);
            const auto status=final_cell.value("status",std::string());
            const bool verified=unverified.empty()&&(status=="exclusive_observed"||status=="busy")
                                &&final_cell.value("owned",Json::array()).empty()&&final_cell.value("foreign",Json::array()).empty();
            cleanup=cleanup&&verified;
            if(!final_cell.value("foreign",Json::array()).empty()&&error.empty())error="GPU isolation lost during cell cleanup";
            auto& saved=attempt["cells"].back();saved["cleanup_verified"]=verified;saved["final_occupancy"]=final_cell;
            if(saved.contains("replay_cutoff")){
                auto& receipt=saved["replay_cutoff"];receipt["cleanup_verified"]=verified;
                receipt["termination_finished_ms"]=(monotonic_seconds()-receipt.at("replay_origin_monotonic_seconds").get<double>())*1000.;
            }
        }
        if(!error.empty() || !cleanup){
            for(std::size_t pending=position+1;pending<order.size();++pending){
                if(!cells[order[pending]].contains("replay_cutoff"))continue;
                preserve_unstarted(pending,"earlier cell failed or cleanup was unverified");
                attempt["cells"].push_back({{"cell",order[pending]},{"order",pending<cells.size()?"forward":"reverse"},
                    {"policy",cells[order[pending]].at("policy")},{"status","not_executed"},
                    {"reason","earlier cell failed or cleanup was unverified"}});
            }
            break;
        }
    }
    if(!mps.empty())try {
        attempt["mps_control"]=control_mps(request,attempt_dir/"mps-control",mps);
    } catch(const std::exception& failure) {
        cleanup=false;attempt["mps_control"]={{"status","failed"},{"error",failure.what()}};
        if(error.empty())error="MPS cleanup failed: "+std::string(failure.what());
    }
    const auto final_gpu=gpu_observation(request,attempt_dir,devices,attempt_pids);
    const auto observed_status=final_gpu.value("status","");
    const bool observed=observed_status=="exclusive_observed" || observed_status=="busy";
    cleanup=cleanup && observed && final_gpu.value("owned",Json::array()).empty();
    if(observed && !final_gpu.value("foreign",Json::array()).empty() && error.empty())
        error="GPU isolation lost before final cleanup";
    attempt["final_occupancy"]=final_gpu;
    attempt["status"]=error.empty() && cleanup?(censored?"censored":"completed"):"failed";
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
    if(phases.size()>1)for(const auto& phase:phases)
        if(!phase.value("mps",Json::object()).empty())throw std::invalid_argument("mps currently requires a single-phase study");
    std::set<std::string> seen;
    const std::set<std::string> kinds={"profile","online","state","sp","session","cluster"};
    for(const auto& phase:phases){
        const auto id=segment(phase.at("id").get<std::string>());if(seen.count(id))throw std::invalid_argument("duplicate phase");
        if(!kinds.count(phase.value("kind","online")))throw std::invalid_argument("unknown executor kind");
        if(phase.contains("replay_cutoff")){
            const auto& cutoff=phase.at("replay_cutoff");
            if(phases.size()!=1||phase.value("kind",std::string())!="cluster"||phase.contains("actions"))
                throw std::invalid_argument("replay cutoff currently requires one cluster phase without scripted actions");
            if(!phase.value("mps",Json::object()).empty())
                throw std::invalid_argument("replay cutoff requires per-cell worker cleanup without a shared MPS server");
            if(!cutoff.is_object()||cutoff.size()!=3||cutoff.value("schema",std::string())!="ditserve.replay-cutoff.v1"
                ||cutoff.value("mode",std::string())!="cancel_and_censor"||!cutoff.contains("after_ms"))
                throw std::invalid_argument("invalid replay cutoff schema or mode");
            number_field(cutoff,"after_ms",0,true);
        }
        for(const auto& dependency:phase.value("requires",Json::array())){
            const auto parent=dependency.is_string()?dependency.get<std::string>():dependency.at("phase").get<std::string>();
            if(!seen.count(parent))throw std::invalid_argument("phase dependency must precede its consumer");
            const auto condition=dependency.is_object()?dependency.value("requires","completed"):"completed";
            if(condition!="completed"&&condition!="qualified")throw std::invalid_argument("dependency requires completed or qualified");
        }
        seen.insert(id);
        phase.value("paired",false);
        const auto mps=phase.value("mps",Json::object());
        if(!mps.is_object())throw std::invalid_argument("mps must be an object");
        if(!mps.empty()){
            for(auto it=mps.begin();it!=mps.end();++it)
                if(it.key()!="server_pid"&&it.key()!="pipe_directory"&&it.key()!="log_directory")
                    throw std::invalid_argument("unknown mps field: "+it.key());
            if(!mps.contains("server_pid")||!mps.at("server_pid").is_number_integer()||mps.at("server_pid").get<int>()<=0)
                throw std::invalid_argument("mps server_pid must be a positive integer");
            if(!mps.contains("pipe_directory")||!fs::path(mps.at("pipe_directory").get<std::string>()).is_absolute())
                throw std::invalid_argument("mps pipe_directory must be absolute");
            if(mps.contains("log_directory")&&!fs::path(mps.at("log_directory").get<std::string>()).is_absolute())
                throw std::invalid_argument("mps log_directory must be absolute");
        }
        const double admission_wait=number_field(phase,"admission_wait_seconds",0);
        const double admission_poll=number_field(phase,"admission_poll_seconds",15);
        if(admission_wait>86400 || admission_poll<=0 || admission_poll>60)
            throw std::invalid_argument("admission wait must be <=24h and poll in (0,60] seconds");
        evaluate_gates(Json::object(),phase.value("gates",Json::array()));
        for(const auto& cell:expand_matrix(phase)){
            if(cell.value("replay_cutoff",Json())!=phase.value("replay_cutoff",Json()))
                throw std::invalid_argument("matrix cannot change the frozen replay cutoff");
            if(cell.contains("replay_cutoff")&&(cell.value("kind",std::string())!="cluster"||cell.contains("actions")))
                throw std::invalid_argument("replay cutoff matrix cells require cluster execution without scripted actions");
            if(cell.at("id")!=id || !kinds.count(cell.value("kind","online")))throw std::invalid_argument("matrix cannot change phase identity or introduce an unknown executor");
            if(cell.value("mps",Json::object())!=mps)throw std::invalid_argument("matrix cannot change the MPS control identity");
            if(cell.contains("worker_config")&&!cell.at("worker_config").is_object())throw std::invalid_argument("worker_config must be an object");
            if(cell.contains("occupancy_backend")&&(!cell.at("occupancy_backend").is_string()||
                (cell.at("occupancy_backend")!="nvidia_smi"&&cell.at("occupancy_backend")!="nvml")))
                throw std::invalid_argument("occupancy_backend must be nvidia_smi or nvml");
            if(cell.contains("controller_decisions")){
                if(cell.value("kind",std::string())!="cluster"||cell.contains("actions"))
                    throw std::invalid_argument("controller_decisions requires cluster execution");
                validate_controller_decisions(cell.at("controller_decisions"));
            }
            if(cell.contains("occupancy_thread")&&!cell.at("occupancy_thread").is_boolean())throw std::invalid_argument("occupancy_thread must be a boolean");
            if(cell.contains("video_worker_options")){
                if(!cell.at("video_worker_options").is_object())throw std::invalid_argument("video_worker_options must be an object");
                if(cell.value("kind","online")!="cluster")throw std::invalid_argument("video_worker_options applies to cluster cells only");
            }
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
                for(const auto& item:cell.value("warmup",Json::array())){
                    known(item.at("worker").get<std::string>());
                    if(item.contains("spec")&&!item.at("spec").is_object())throw std::invalid_argument("warmup spec must be an object");
                }
                if(cell.contains("text_prefetch_workers")){
                    if(!cell.at("text_prefetch_workers").is_array())throw std::invalid_argument("text_prefetch_workers must be an array of worker names");
                    for(const auto& worker:cell.at("text_prefetch_workers"))known(worker.get<std::string>());
                }
                const auto rows=study_workload(cell);if(rows.empty())throw std::invalid_argument("phase workload must be nonempty");
                if(cell.value("kind","online")=="cluster"){
                    // Slots name the per-GPU video and image workers; rows name cost classes.
                    const auto& slots=cell.at("gpus");
                    if(!slots.is_array()||slots.empty())throw std::invalid_argument("cluster gpus must be a nonempty array of worker slots");
                    std::set<std::string> names;
                    for(const auto& slot:slots)for(const auto* role:{"video","image"}){
                        const auto name=slot.at(role).get<std::string>();known(name);
                        if(!names.insert(name).second)throw std::invalid_argument("each cluster slot needs its own workers");
                    }
                    std::set<bool> counter_domains;
                    for(const auto& slot:slots)counter_domains.insert(
                        config.at("workers").at(slot.at("video").get<std::string>()).at("backend")=="wan");
                    if(counter_domains.size()>1)
                        throw std::invalid_argument("cluster video slots cannot mix Wan rank-local and non-Wan block counters");
                    const ClusterCost cost(cell.at("profile"));
                    if(!cluster_policy(cell.at("policy").get<std::string>()))throw std::invalid_argument("unknown cluster policy");
                    const auto transport=string_field(cell.value("options",Json::object()),"sp_transport","default");
                    if(transport!="default"&&transport!="pinned"&&transport!="resume"&&transport!="export")
                        throw std::invalid_argument("sp_transport must be pinned, resume or export");
                    if(bool_field(cell.value("options",Json::object()),"doorbell",false)&&transport!="export")
                        throw std::invalid_argument("doorbell needs the export transport");
                    if(transport=="export"&&slots.size()!=2&&slots.size()!=4&&slots.size()!=6&&slots.size()!=8)
                        throw std::invalid_argument("export transport needs 2, 4, 6 or 8 GPUs");
                    for(const auto& row:rows)if(cost.klass(row.at("class").get<std::string>()).at("kind")!=row.at("kind"))
                        throw std::invalid_argument("request kind differs from its cost class");
                    continue;
                }
                for(const auto& row:rows){
                    known(row.at("worker").get<std::string>());
                    const auto list=stages(row);
                    if(!list.is_array()||list.empty())throw std::invalid_argument("request stages must be nonempty");
                    for(const auto& stage:list){known(stage.value("worker",row.at("worker").get<std::string>()));if(stage.contains("spec")&&!stage.at("spec").is_object())throw std::invalid_argument("stage spec must be an object");}
                }
            }
        }
    }
}
}

Json run_study(const Json& original_request){
    CancelScope signals;
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
    const auto only=request.value("phase","");bool matched=only.empty(),failed=false,censored=false;
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
        std::vector<int> admission_pids;
        if(!phase.value("mps",Json::object()).empty())admission_pids.push_back(phase.at("mps").at("server_pid").get<int>());
        auto observation=gpu_observation(request,attempt_dir,devices,admission_pids,waiting_limit>0?std::min(waiting_deadline,waiting_started+15):0);
        int idle_observations=0;
        while(waiting_limit>0){
            if(cancelled() || monotonic_seconds()>=waiting_deadline)break;
            const bool idle=observation.value("status","")=="exclusive_observed" && observation.value("foreign",Json::array()).empty();
            idle_observations=idle?idle_observations+1:0;
            if(idle_observations>=2)break;
            const double elapsed=monotonic_seconds()-waiting_started;
            record["status"]="waiting_resource";
            record["admission"]={{"phase",id},{"observation",observation},{"wait_seconds",elapsed},{"gpu_time_charged",false}};
            write(run_path,record);
            if(cancelled() || elapsed>=waiting_limit)break;
            const double next=std::min(waiting_started+waiting_limit,monotonic_seconds()+waiting_poll);
            while(!cancelled() && monotonic_seconds()<next)wait_until(next);
            if(cancelled())break;
            observation=gpu_observation(request,attempt_dir,devices,admission_pids,std::min(waiting_deadline,monotonic_seconds()+15));
        }
        if(waiting_limit>0)record["admission_wait_seconds"]=monotonic_seconds()-waiting_started;
        if(cancelled() || (waiting_limit>0 && idle_observations<2)){
            record["status"]="blocked";
            record["admission"]={{"phase",id},{"error",cancelled()?"cancelled while waiting for isolated GPU":"GPU admission wait timed out"},{"observation",observation}};
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
        if(attempt.at("status")=="censored")attempt["qualification"]={{"qualified",false},{"status","censored"},
            {"reason","finite replay observation cutoff; not a completed or capacity-qualified run"}};
        record["phases"].back()=attempt;record["budget"]=budget.snapshot();
        if(attempt.at("status")=="completed")completed[id]=attempt;
        else if(attempt.at("status")=="censored")censored=true;
        else failed=true;
        record["status"]=failed?"failed":censored?"censored":"incomplete";write(run_path,record);
        if(failed||censored)break;
    }
    if(!matched)throw std::invalid_argument("unknown selected phase");
    if(completed.size()==config.at("phases").size())record["status"]="completed";
    record["metrics"]=request_summary(record["requests"]);record["budget"]=budget.snapshot();
    record["measurement_scope"]=devices.empty()?"CPU fixture; not GPU measurements":"supervised GPU experiment";
    record["retention"]["pending_dependencies"]=record["status"]=="completed"?Json::array():Json::array({"resume_or_failure_investigation"});
    budget_owner.reset();
    record["archive"]=publish(record,request,work);return record;
}

// Illustrative defaults keep `ditserve simulate` runnable without a config.
// Every result carries the profile provenance; synthetic is never measured.
Json synthetic_profile(){
    return {{"provenance","synthetic"},{"description","Illustrative smoke-test costs; not measured."},
        {"classes",{{"video",{{"kind","video"},{"steps",8},{"blocks_per_step",8},{"prepare_ms",20},{"encode_ms",50},
                              {"step_compute_ms",400},{"comm_mb_per_step",400},{"collectives_per_step",32},{"finish_ms",150}}},
                    {"image",{{"kind","image"},{"service_ms",40}}}}},
        {"links",{{"2",{{"gbps",20},{"latency_ms",0.02}}},{"4",{{"gbps",12},{"latency_ms",0.03}}},{"8",{{"gbps",6},{"latency_ms",0.05}}}}},
        {"sp_efficiency",0.95},{"resize_ms",5}};
}

Json simulate_study(const Json& request){
    const auto work=real_path(request.at("work_dir").get<std::string>());fs::create_directories(work);RunLock lock(work/"run.lock");
    if(fs::exists(work/"run.json"))throw std::invalid_argument("simulation run already exists");
    Json config=request.at("config");
    if(!config.contains("profile"))config["profile"]=synthetic_profile();
    if(!config.contains("workload")){
        auto& generator=config["generator"];
        if(!generator.contains("classes"))generator["classes"]={
            {{"class","video"},{"kind","video"},{"weight",1},{"service_ms",3400},{"slo_scale",3}},
            {{"class","image"},{"kind","image"},{"weight",4},{"service_ms",40},{"slo_scale",5}}};
        if(!generator.contains("pattern")){generator["pattern"]="poisson";generator["rate_rps"]=generator.value("rate_rps",Json(2));}
    }
    const auto provenance=config.at("profile").value("provenance","");
    Json record={{"schema","ditserve.run.v1"},{"experiment",config.value("experiment","synthetic")},{"run_id",segment(request.at("run_id").get<std::string>())},
        {"config",config},{"status","completed"},{"phases",Json::array()},{"requests",Json::array()},{"failures",Json::array()},{"artifacts",Json::array()},
        {"budget",{{"active",nullptr},{"gpu_seconds",0}}},
        {"notice",provenance=="synthetic"?"Synthetic cost-model simulation; not measured GPU performance.":"Cost-model simulation from a "+provenance+" profile; not an execution of models."}};
    Json phase=config;phase["matrix"]=config.value("matrix",Json::object());
    const auto cells=expand_matrix(phase);
    for(std::size_t index=0;index<cells.size();++index){
        const auto& cell=cells[index];
        const auto rows=study_workload(cell.contains("workload")?Json{{"workload",cell.at("workload")}}:Json{{"generator",cell.at("generator")}});
        for(const auto& policy:cell.value("policies",Json::array({"static-sp1","abl-step-elastic","abl-block-elastic"}))){
            const auto name=policy.get<std::string>();const int gpus=cell.value("gpus",4);
            if(name.rfind("static_sp",0)==0 && std::stoi(name.substr(9))>gpus){
                record["phases"].push_back({{"id",name+"-c"+std::to_string(index)},{"status","skipped"},{"policy",policy},{"cell",index},
                    {"reason","fixed SP degree exceeds the cell's GPUs"}});
                continue;
            }
            Json input={{"workload",rows},{"profile",cell.at("profile")},{"policy",policy},{"gpus",gpus},
                {"options",cell.value("options",Json::object())},{"dispatch_ms",cell.value("dispatch_ms",0.)}};
            if(cell.contains("encode_slots"))input["encode_slots"]=cell.at("encode_slots");
            if(cell.contains("planner_profile"))input["planner_profile"]=cell.at("planner_profile");
            auto result=simulate_cluster(input);
            if(!cell.value("keep_requests",false))result.erase("requests");
            Json settings=cell;settings.erase("workload");settings.erase("matrix");settings.erase("profile");settings.erase("planner_profile");settings.erase("policies");
            record["phases"].push_back({{"id",policy.get<std::string>()+"-c"+std::to_string(index)},{"status","completed"},
                {"policy",policy},{"cell",index},{"config",settings},{"result",result}});
        }
    }
    record["archive"]=publish(record,request,work);return record;
}
}
