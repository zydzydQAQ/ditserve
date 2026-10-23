#include "ditserve/control/study.hpp"
#include "ditserve/control/budget.hpp"
#include "ditserve/control/cluster.hpp"
#include "ditserve/control/dispatch.hpp"
#include "ditserve/runtime/process.hpp"
#include "ditserve/benchmark/cluster_sim.hpp"
#include <openssl/sha.h>
#include <algorithm>
#include <chrono>
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
#include <sched.h>
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
    bool cpu_configured(const std::string& name)const{
        const auto config=definitions_.at(name).value("config",Json::object());
        return config.contains("cpu_affinity")||config.contains("cpu_threads");
    }
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

// A CPU mock worker answers occupancy checks. Heartbeats keep one per cell
// instead of spawning a process every observation; a failed probe is dropped
// and recreated on the next check.
class OccupancyProbe {
public:
    OccupancyProbe(const Json& request,fs::path work,std::string backend="nvidia_smi")
        :request_(request),work_(std::move(work)),backend_(std::move(backend)){}
    ~OccupancyProbe(){close();}
    Json observe(const Json& devices,const std::vector<int>& allowed,double deadline) {
        if(devices.empty())return {{"status","exclusive_observed"},{"owned",Json::array()},{"foreign",Json::array()}};
        try {
            if(!probe_)probe_=std::make_unique<Workers>(Json{{"probe",{{"backend","mock"}}}},request_,work_);
            return probe_->get("probe").call({{"op","occupancy"},{"occupancy_backend",backend_},{"gpu_uuids",devices},{"allowed_pids",allowed},{"allowed_pgids",allowed}},deadline>0?deadline:monotonic_seconds()+15);
        }catch(const std::exception& error){
            close();
            return {{"status","unknown"},{"reason",error.what()}};
        }
    }
private:
    void close() noexcept {
        if(!probe_)return;
        // Give the resident NVML owner its normal shutdown path. Bound it
        // independently, then retain process-group termination as the fallback.
        try{probe_->shutdown(monotonic_seconds()+1);}catch(...){}
        probe_->stop();probe_.reset();
    }
    Json request_;fs::path work_;std::string backend_;std::unique_ptr<Workers> probe_;
};

Json gpu_observation(const Json& request,const fs::path& work,const Json& devices,const std::vector<int>& allowed={},double deadline=0) {
    return OccupancyProbe(request,work/"probe").observe(devices,allowed,deadline);
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

// Worker threads note each call they make; the controlling thread folds the
// notes into the request row, so rows are never touched concurrently.
Json traced(Worker& worker,const std::string& name,const Json& action,double deadline,Json& ops){
    ops.push_back({{"worker",name},{"op",action.at("op")},{"pid",worker.process->pid()}});
    return worker.call(action,deadline);
}
void note_calls(Json& row,const Json& ops){
    for(const auto& op:ops){
        row["worker_pids"][op.at("worker").get<std::string>()]=op.at("pid");
        const auto key=op.at("op").get<std::string>();
        row["operation_counts"][key]=row.value("operation_counts",Json::object()).value(key,0)+1;
    }
}

// Preserve encoder provenance without adding worker calls or charging CPU
// encoding a second time to model_ms. Events retain every prepare (including
// regrouping); the flat fields describe the most recently observed encoding.
void append_encoding(Json& target,const Json& event){
    target["encoding_events"].push_back(event);
    for(const auto* key:{"negative_prompt_cache","cpu_prompt_encode_ms"})
        if(event.contains(key))target[key]=event.at(key);
}
void note_encoding(Json& target,const Json& response,const std::string& name,int pid,
                   const std::string& op,const Json& ticket=nullptr){
    Json event=Json::object();
    for(const auto* key:{"negative_prompt_cache","cpu_prompt_encode_ms"})
        if(response.contains(key))event[key]=response.at(key);
    // Wan encode_poll names completed CPU encoding duration model_ms.
    if(op=="encode_poll"&&response.contains("model_ms"))
        event["cpu_prompt_encode_ms"]=response.at("model_ms");
    if(event.empty())return;
    event["worker"]=name;event["pid"]=pid;event["op"]=op;
    if(!ticket.is_null())event["ticket"]=ticket;
    append_encoding(target,event);
}

// Requests of a cluster phase or simulation cell; a generator's load and
// per-class service come from the cost profile and the GPU count.
Json phase_workload(const Json& phase,const ClusterCost& cost,int gpus,Json* derived=nullptr){
    if(!phase.contains("generator"))return study_workload(phase);
    Json resolved=phase;resolved["generator"]=resolve_generator(phase.at("generator"),cost,gpus);
    if(derived)*derived=resolved["generator"]["derived"];
    return study_workload(resolved);
}

// Untimed preparation before the trace clock starts. Model loading, kernel
// selection and allocator growth must not be charged to the first arrivals;
// every warmup state is released before the replay begins.
Json warmup(const Json& phase,Workers& workers,Budget& budget) {
    Json results=Json::array();
    const auto items=phase.value("warmup",Json::array());
    for(std::size_t index=0;index<items.size();++index){
        const auto& item=items[index];
        const auto name=item.at("worker").get<std::string>();
        auto& worker=workers.get(name);
        const double started=monotonic_seconds();
        Json spec=item.value("spec",Json::object());spec["request_id"]="warmup-"+std::to_string(index);
        if(cancelled())throw std::runtime_error("run cancelled");
        budget.check();
        auto prepared=worker.call({{"op","prepare"},{"spec",spec}},budget.deadline());
        const auto state=prepared.at("state_id").get<std::string>();
        double load=number_field(prepared,"load_ms");
        auto progress=prepared.value("progress",Json::object());
        for(int calls=0;!progress.value("done",false);++calls){
            if(calls>=10000)throw std::runtime_error("warmup did not finish");
            if(cancelled())throw std::runtime_error("run cancelled");
            budget.check();
            Json action={{"op","advance"},{"state_id",state},{"quantum",item.value("quantum",Json("step"))}};
            if(item.contains("blocks"))action["blocks"]=item.at("blocks");
            progress=worker.call(action,budget.deadline()).at("progress");
        }
        const auto finished=worker.call({{"op","finish"},{"state_id",state},{"output_mode","check"}},budget.deadline());
        worker.call({{"op","release"},{"state_id",state}},budget.deadline());
        results.push_back({{"worker",name},{"worker_pid",worker.process->pid()},{"state_id",state},{"spec",spec},
            {"load_ms",load},{"elapsed_ms",(monotonic_seconds()-started)*1000}});
        note_encoding(results.back(),prepared,name,worker.process->pid(),"prepare");
        if(finished.contains("prompt_cache"))results.back()["prompt_cache"]=finished.at("prompt_cache");
    }
    return results;
}

// These opt-in contracts are checked before any workers or budget sessions start.
int prefetch_depth(const Json& phase) {
    if (!phase.contains("text_prefetch_depth")) return -1;
    if (phase.contains("text_prefetch")) throw std::invalid_argument("give text_prefetch_depth or legacy text_prefetch, not both");
    const auto& value = phase.at("text_prefetch_depth");
    if (!value.is_number_integer() || value.is_boolean()) throw std::invalid_argument("text_prefetch_depth must be 0, 1 or 2");
    const int depth = value.get<int>();
    if (depth < 0 || depth > 2) throw std::invalid_argument("text_prefetch_depth must be 0, 1 or 2");
    return depth;
}
Json real_sp_config(const Json& phase) {
    if (!phase.contains("real_sp")) return nullptr;
    const auto& value = phase.at("real_sp");
    if (!value.is_object()) throw std::invalid_argument("real_sp must be an explicit configuration object");
    check_keys(value, {"enabled", "degrees", "transport", "timeout_seconds"});
    if (value.value("enabled", false) != true || value.value("transport", std::string()) != "host_file_v1" ||
        value.value("degrees", Json::array()) != Json::array({1, 2}))
        throw std::invalid_argument("real_sp requires enabled=true, degrees=[1,2], transport=host_file_v1");
    const double timeout = number_field(value, "timeout_seconds", 60, true);
    if (timeout > 60) throw std::invalid_argument("real_sp timeout_seconds must be in (0,60]");
    const auto options = phase.value("options", Json::object());
    if (options.contains("degrees") && options.at("degrees") != value.at("degrees"))
        throw std::invalid_argument("real_sp degrees conflict with scheduler options");
    if (options.value("pin_started", false)) throw std::invalid_argument("real_sp cannot pin started requests");
    return value;
}

cpu_set_t controller_mask(const Json& value,const cpu_set_t& allowed) {
    if(!value.is_array()||value.empty())throw std::invalid_argument("controller_cpu_affinity must be a nonempty CPU array");
    cpu_set_t mask;CPU_ZERO(&mask);
    for(const auto& cpu:value){
        if(!cpu.is_number_integer()||cpu.is_boolean())throw std::invalid_argument("controller_cpu_affinity needs integer CPUs");
        const int index=cpu.get<int>();
        if(index<0||index>=CPU_SETSIZE||!CPU_ISSET(index,&allowed)||CPU_ISSET(index,&mask))
            throw std::invalid_argument("controller_cpu_affinity must contain distinct allowed CPUs");
        CPU_SET(index,&mask);
    }
    return mask;
}
class ControllerAffinity {
public:
    explicit ControllerAffinity(const Json& requested) {
        if(sched_getaffinity(0,sizeof previous_,&previous_))throw std::runtime_error("cannot read controller CPU affinity");
        const auto mask=controller_mask(requested,previous_);
        if(sched_setaffinity(0,sizeof mask,&mask))throw std::runtime_error("cannot set controller CPU affinity");
        active_=true;
        cpu_set_t actual;
        if(sched_getaffinity(0,sizeof actual,&actual)||!CPU_EQUAL(&mask,&actual)){
            sched_setaffinity(0,sizeof previous_,&previous_);active_=false;
            throw std::runtime_error("effective controller CPU affinity differs from requested mask");
        }
        Json effective=Json::array();for(int cpu=0;cpu<CPU_SETSIZE;++cpu)if(CPU_ISSET(cpu,&actual))effective.push_back(cpu);
        evidence={{"requested_affinity",requested},{"effective_affinity",effective}};
    }
    ~ControllerAffinity(){if(active_)sched_setaffinity(0,sizeof previous_,&previous_);}
    void restore(){
        if(active_&&sched_setaffinity(0,sizeof previous_,&previous_))throw std::runtime_error("cannot restore controller CPU affinity");
        active_=false;
    }
    Json evidence;
private:
    cpu_set_t previous_;
    bool active_=false;
};

// Concurrent multi-GPU replay. Each GPU runs at most one quantum; between
// quanta the shared cluster_plan decides placement, SP degree and preemption.
// Worker calls run on std::async threads and every rank of an SP group runs
// the same quantum concurrently; all bookkeeping stays on this thread.
// Real backends retain pinned SP1 by default. The explicit real_sp contract
// enables two-rank block execution with transactional state transfer.
Json cluster(const Json& phase,Workers& workers,Budget& budget,Json& record,const fs::path& work,const fs::path& cell_dir) {
    const ClusterCost cost(phase.at("profile"));
    const auto policy=phase.at("policy").get<std::string>();
    Json options=phase.value("options",Json::object());
    const auto slots=phase.at("gpus");
    const int gpus=static_cast<int>(slots.size());
    Json derived=nullptr;
    Json rows=phase_workload(phase,cost,gpus,&derived);
    const Json trace=rows;
    bool fixtures=true;
    for(const auto& slot:slots)fixtures=fixtures&&workers.backend(slot.at("video").get<std::string>())=="mock";
    const auto sp_config=real_sp_config(phase);
    const bool real_sp=!sp_config.is_null();
    if(real_sp){
        options["degrees"]=sp_config.at("degrees");options["pin_started"]=false;options["reserve_transition_union"]=true;
    }else if(!fixtures){
        if(policy.rfind("static_sp",0)==0&&policy!="static_sp1")throw std::invalid_argument("real cluster workers support SP1 only");
        options["degrees"]=Json::array({1});options["pin_started"]=true;
    }
    const int depth=prefetch_depth(phase);
    const bool prefetch=depth>=0?depth>0:phase.value("text_prefetch",0)>0;
    const int reserved_images=int_field(options,"reserve_image_gpus",0,0);
    if(reserved_images>=gpus)throw std::invalid_argument("reserve_image_gpus must leave a GPU for videos");
    const int encoder_gpus=gpus-reserved_images;
    const auto output_mode=phase.value("output_mode",std::string("file"));
    const bool dispatch_timeline=phase.value("record_dispatch_timeline",false);
    // The r014 performance screen did not qualify this candidate for default
    // use. Keep the explicit experiment path while preserving legacy dispatch.
    const bool dispatch_refresh=phase.value("dispatch_refresh",false);
    const auto warmed=warmup(phase,workers,budget);
    for(const auto& slot:slots){workers.get(slot.at("video").get<std::string>());workers.get(slot.at("image").get<std::string>());}
    auto check=[&](){if(cancelled())throw std::runtime_error("run cancelled");budget.check(number_field(phase,"cleanup_reserve_seconds",0));};
    struct Live{std::string status="pending";bool prepared=false;int steps=0,blocks=0,total=0,encode_gpu=-1,decode_done=0,epoch=0,sequence=0;std::vector<int> group;std::string ticket;std::set<int> resident;};
    struct TaskResult{Json output;double rpc_completed_ms;};
    struct Task{std::size_t row;Json assignment;std::future<TaskResult> result;double started;};
    std::vector<Live> live(rows.size());
    std::vector<Task> tasks;
    std::vector<bool> busy(gpus,false);
    std::vector<int> peak_tickets(gpus,0);
    std::vector<std::vector<std::string>> stale(gpus);   // states to release once the GPU is idle
    std::map<std::string,std::size_t> index;
    for(std::size_t i=0;i<rows.size();++i){
        auto& row=rows[i];
        if(!row.contains("class"))throw std::invalid_argument("cluster requests require a cost class");
        if(cost.klass(row.at("class").get<std::string>()).at("kind")!=row.at("kind"))throw std::invalid_argument("request kind differs from its cost class");
        row["quality"]={{"status","unverified"},{"passed",false}};row["degrees"]=Json::array();
        row["observation_id"]=phase.at("_observation_id");row["phase"]=phase.at("id");row["attempt"]=phase.at("_attempt");row["cell"]=phase.at("_cell");row["order"]=phase.at("_order");
        index[row.at("request_id").get<std::string>()]=i;
    }
    auto video=[&](int gpu){return slots.at(gpu).at("video").get<std::string>();};
    auto image=[&](int gpu){return slots.at(gpu).at("image").get<std::string>();};
    Json runtime_bindings=Json::object();
    for(const auto& slot:slots)for(const auto* role:{"video","image"}){
        const auto name=slot.at(role).get<std::string>();
        if(depth<0&&!real_sp&&!workers.cpu_configured(name))continue;
        runtime_bindings[name]=workers.get(name).call({{"op","hello"}},budget.deadline());
    }
    if(real_sp){
        const auto rendezvous="file://"+(cell_dir/"sp-rendezvous").string();
        std::vector<std::future<Json>> initialized;
        for(int rank=0;rank<gpus;++rank){
            const auto name=video(rank);
            initialized.push_back(std::async(std::launch::async,[&,name,rank,rendezvous]{
                return workers.get(name).call({{"op","distributed"},{"action","init"},{"elastic",true},
                    {"rank",rank},{"world_size",2},{"init_method",rendezvous},
                    {"timeout_seconds",sp_config.value("timeout_seconds",60.)}},budget.deadline());
            }));
        }
        std::string failure;
        for(auto& rank:initialized)try{rank.get();}catch(const std::exception& error){failure=error.what();}
        if(!failure.empty())throw std::runtime_error("elastic SP initialization failed: "+failure);
    }
    const double origin=monotonic_seconds();
    auto now=[&](){return (monotonic_seconds()-origin)*1000;};
    auto spec_of=[&](const Json& row){Json spec=row.at("spec");spec["request_id"]=row.at("request_id");return spec;};
    std::size_t terminal=0,encoders=0;
    Json last_image_arrival=nullptr;
    auto launch=[&](const Json& assignment){
        const auto id=assignment.at("request_id").get<std::string>();const std::size_t i=index.at(id);
        auto& row=rows[i];auto& state=live[i];
        std::vector<int> group;for(const auto& gpu:assignment.at("group"))group.push_back(gpu.get<int>());
        const auto reserved=assignment.value("reserved_group",assignment.at("group"));
        for(const auto& gpu:reserved){
            if(busy[gpu.get<int>()])throw std::runtime_error("dispatch overlaps an active rank reservation");
            busy[gpu.get<int>()]=true;
        }
        if(!row.contains("start_ms"))row["start_ms"]=now();
        const int degree=static_cast<int>(group.size());
        if(std::find(row["degrees"].begin(),row["degrees"].end(),degree)==row["degrees"].end())row["degrees"].push_back(degree);
        if(assignment.at("reason")=="block_preempt")row["preempting"]=true;
        const auto quantum=assignment.at("quantum");const double deadline=budget.deadline();
        std::function<Json()> work_unit;
        if(quantum=="atomic"){
            const auto name=image(group.front());const auto spec=spec_of(row);
            work_unit=[&workers,name,id,spec,deadline,output_mode]{
                auto& w=workers.get(name);
                Json out={{"ops",Json::array()}};
                auto progress=traced(w,name,{{"op","prepare"},{"state_id",id},{"spec",spec}},deadline,out["ops"]).at("progress");
                for(int calls=0;!progress.value("done",false);++calls){
                    if(calls>=1000)throw std::runtime_error("image did not finish");
                    auto advanced=traced(w,name,{{"op","advance"},{"state_id",id},{"quantum","step"}},deadline,out["ops"]);
                    progress=advanced.at("progress");out["model_ms"]=number_field(out,"model_ms")+number_field(advanced,"model_ms");
                }
                out["finish"]=traced(w,name,{{"op","finish"},{"state_id",id},{"output_mode",output_mode}},deadline,out["ops"]);
                traced(w,name,{{"op","release"},{"state_id",id}},deadline,out["ops"]);
                return out;
            };
        }else if(real_sp){
            const auto old_group=state.group;
            const bool prepared=state.prepared;
            const int epoch=state.epoch, sequence=state.sequence;
            const auto ticket=state.ticket;
            const int leader=!ticket.empty()?state.encode_gpu:group.front();
            const auto spec=spec_of(row);
            const auto transfer_id="request-"+std::to_string(i)+"-epoch-"+std::to_string(epoch)+"-q-"+std::to_string(sequence);
            const auto transfer_path=(cell_dir/"transfers"/(transfer_id+".pt")).string();
            Json decode={{"op","finish"},{"state_id",id},{"output_mode",output_mode}};
            if(assignment.contains("decode_frames"))decode["decode_frames"]=assignment.at("decode_frames");
            state.ticket.clear();
            work_unit=[&,id,group,old_group,prepared,epoch,sequence,ticket,leader,spec,transfer_id,transfer_path,quantum,deadline,decode]{
                Json out={{"ops",Json::array()},{"ranks",Json::array()}};
                std::vector<int> source=old_group;
                int next_epoch=epoch;
                if(!prepared){
                    Json prepare={{"op","prepare"},{"state_id",id},{"spec",spec}};
                    if(!ticket.empty())prepare["encode_ticket"]=ticket;
                    const auto name=video(leader);
                    auto& w=workers.get(name);
                    const auto reply=traced(w,name,prepare,deadline,out["ops"]);
                    note_encoding(out,reply,name,w.process->pid(),"prepare",prepare.value("encode_ticket",Json()));
                    source={leader};
                }
                if(source!=group){
                    std::set<int> participants(source.begin(),source.end());participants.insert(group.begin(),group.end());
                    const double began=monotonic_seconds();
                    std::vector<std::future<Json>> transfers;
                    for(int rank:participants){const auto name=video(rank);
                        transfers.push_back(std::async(std::launch::async,[&,name]{
                            Json result={{"ops",Json::array()}};
                            result["result"]=traced(workers.get(name),name,{{"op","distributed"},{"action","transition"},
                                {"state_id",id},{"transfer_id",transfer_id},{"source_group",source},{"target_group",group},
                                {"expected_epoch",epoch},{"path",transfer_path}},deadline,result["ops"]);
                            return result;
                        }));
                    }
                    std::string failure;
                    for(auto& pending:transfers)try{
                        auto value=pending.get();for(const auto& op:value.at("ops"))out["ops"].push_back(op);
                        const auto& result=value.at("result");
                        for(const auto& artifact:result.value("artifacts",Json::array()))out["transfer_artifacts"].push_back(artifact);
                        if(result.at("group")!=group||result.at("ownership_epoch")!=epoch+1)
                            throw std::runtime_error("SP transition ownership consensus failed");
                    }catch(const std::exception& error){failure=error.what();}
                    if(!failure.empty())throw std::runtime_error("SP transition failed: "+failure);
                    next_epoch=epoch+1;
                    out["transition"]={{"source_group",source},{"target_group",group},{"ownership_epoch",next_epoch},
                        {"transport","host_file_v1"},{"elapsed_ms",(monotonic_seconds()-began)*1000}};
                }
                out["ownership_epoch"]=next_epoch;
                if(quantum=="finish"){
                    const auto name=video(group.front());
                    out["finish"]=traced(workers.get(name),name,decode,deadline,out["ops"]);
                    if(out["finish"].value("decode",Json::object()).value("done",true))
                        traced(workers.get(name),name,{{"op","release"},{"state_id",id}},deadline,out["ops"]);
                }else{
                    std::vector<std::future<Json>> ranks;
                    for(int rank:group){const auto name=video(rank);
                        ranks.push_back(std::async(std::launch::async,[&,name]{
                            Json result={{"ops",Json::array()}};
                            Json action={{"op","advance"},{"state_id",id},{"expected_epoch",next_epoch},{"quantum_seq",sequence}};
                            if(quantum=="step")action["quantum"]="step";else action["blocks"]=quantum;
                            result["result"]=traced(workers.get(name),name,action,deadline,result["ops"]);return result;
                        }));
                    }
                    std::string failure;
                    for(auto& rank:ranks)try{auto value=rank.get();out["ranks"].push_back(value.at("result"));
                        for(const auto& op:value.at("ops"))out["ops"].push_back(op);
                    }catch(const std::exception& error){failure=error.what();}
                    if(!failure.empty())throw std::runtime_error(failure);
                }
                return out;
            };
            state.resident=std::set<int>(group.begin(),group.end());
        }else if(quantum=="finish"){
            const auto name=video(group.front());
            Json action={{"op","finish"},{"state_id",id},{"output_mode",output_mode}};
            if(assignment.contains("decode_frames"))action["decode_frames"]=assignment.at("decode_frames");
            work_unit=[&workers,name,id,deadline,action]{
                auto& w=workers.get(name);
                Json out={{"ops",Json::array()}};
                out["finish"]=traced(w,name,action,deadline,out["ops"]);
                // A chunked decode releases its state with the last chunk.
                if(out["finish"].value("decode",Json::object()).value("done",true))traced(w,name,{{"op","release"},{"state_id",id}},deadline,out["ops"]);
                return out;
            };
            for(int gpu:state.resident)if(gpu!=group.front())stale[gpu].push_back(id);
            state.resident={group.front()};
        }else{
            // Ranks entering the group (first quantum or a regroup) prepare;
            // ranks leaving it release once their GPU is idle.
            std::vector<std::pair<std::string,Json>> calls;
            for(int gpu:state.resident)if(std::find(group.begin(),group.end(),gpu)==group.end())stale[gpu].push_back(id);
            Json resume={{"step",state.steps},{"block",state.blocks}};
            for(int gpu:group){
                Json actions=Json::array();
                if(!state.resident.count(gpu)){
                    Json prepare={{"op","prepare"},{"state_id",id},{"spec",spec_of(row)},{"sp_degree",degree}};
                    if(state.prepared)prepare["resume"]=resume;
                    else if(gpu==state.encode_gpu&&!state.ticket.empty())prepare["encode_ticket"]=state.ticket;
                    actions.push_back(prepare);
                }
                Json advance={{"op","advance"},{"state_id",id},{"sp_degree",degree},{"quantum",quantum}};
                if(quantum!="step"){advance["blocks"]=quantum;advance["quantum"]=nullptr;}
                actions.push_back(advance);
                calls.emplace_back(video(gpu),actions);
            }
            state.resident=std::set<int>(group.begin(),group.end());state.ticket.clear();
            work_unit=[&workers,calls,deadline]{
                std::vector<std::future<Json>> ranks;
                for(const auto& item:calls)ranks.push_back(std::async(std::launch::async,[&workers,item,deadline]{
                    auto& w=workers.get(item.first);Json rank={{"ops",Json::array()}};
                    for(const auto& action:item.second){
                        rank["result"]=traced(w,item.first,action,deadline,rank["ops"]);
                        if(action.at("op")=="prepare")note_encoding(rank,rank["result"],item.first,w.process->pid(),
                            "prepare",action.value("encode_ticket",Json()));
                    }
                    return rank;
                }));
                Json out={{"ranks",Json::array()},{"ops",Json::array()}};
                std::string error;
                for(auto& rank:ranks){try{auto value=rank.get();out["ranks"].push_back(value.at("result"));
                    for(const auto& op:value.at("ops"))out["ops"].push_back(op);
                    for(const auto& event:value.value("encoding_events",Json::array()))append_encoding(out,event);
                }catch(const std::exception& failure){error=failure.what();}}
                if(!error.empty())throw std::runtime_error(error);
                return out;
            };
        }
        state.status="running";state.group=group;
        const double dispatched=now();
        tasks.push_back({i,assignment,std::async(std::launch::async,[work_unit=std::move(work_unit),origin]{
            auto output=work_unit();
            // All calls in this host dispatch have returned. This is not a
            // device timestamp and excludes the controller's later harvest.
            const double completed=(monotonic_seconds()-origin)*1000;
            return TaskResult{std::move(output),completed};
        }),dispatched});
    };
    auto finish_task=[&](Task& task,const Json& out,double rpc_completed_ms){
        const double harvested_ms=now();
        auto& row=rows[task.row];auto& state=live[task.row];
        for(const auto& gpu:task.assignment.value("reserved_group",task.assignment.at("group")))busy[gpu.get<int>()]=false;
        if(out.contains("ownership_epoch"))state.epoch=out.at("ownership_epoch");
        if(out.contains("transition"))row["transitions"].push_back(out.at("transition"));
        if(out.contains("transfer_artifacts"))add_artifacts(record,{{"artifacts",out.at("transfer_artifacts")}},work,
            phase.value("artifact_dependencies",Json::array()));
        note_calls(row,out.at("ops"));
        for(const auto& event:out.value("encoding_events",Json::array()))append_encoding(row,event);
        const double elapsed=now()-task.started;
        const auto quantum=task.assignment.at("quantum");
        const auto key=quantum.is_string()?quantum.get<std::string>():"blocks";
        if(dispatch_timeline){
            // Host-observed operation windows, not CUDA kernel intervals.
            Json event={{"start_ms",task.started},{"finish_ms",task.started+elapsed},
                {"group",task.assignment.at("group")},{"quantum",quantum},{"reason",task.assignment.at("reason")},
                {"timing_scope","host_dispatch_task"},{"rpc_completed_ms",rpc_completed_ms},
                {"harvested_ms",harvested_ms},{"harvest_delay_ms",harvested_ms-rpc_completed_ms}};
            for(const auto* key:{"chosen_quantum_ms","block_count","quantum_feedback","decode_frames"})
                if(task.assignment.contains(key))event[key]=task.assignment.at(key);
            if(quantum.is_number_integer()&&out.contains("ranks")&&!out.at("ranks").empty()){
                int executed=0;bool observed=false;
                for(const auto& detail:out.at("ranks").at(0).value("events",Json::array()))
                    if(detail.value("kind","")=="block_quantum"){executed+=detail.at("blocks").get<int>();observed=true;}
                if(observed)event["executed_blocks"]=executed;
            }
            row["dispatch_timeline"].push_back(event);
            const auto reason=task.assignment.at("reason").get<std::string>();
            row["dispatch_reason_counts"][reason]=row.value("dispatch_reason_counts",Json::object()).value(reason,0)+1;
        }
        row["action_ms"][key]=row.value("action_ms",Json::object()).value(key,0.)+elapsed;
        row["quanta"]=row.value("quanta",0)+1;
        if(quantum=="finish"&&!out.at("finish").value("decode",Json::object()).value("done",true)){
            row["model_ms"]=number_field(row,"model_ms")+number_field(out.at("finish"),"model_ms");
            state.decode_done=out.at("finish").at("decode").value("frames_done",state.decode_done);state.status="suspended";
            return;
        }
        if(quantum=="atomic"||quantum=="finish"){
            add_artifacts(record,out.at("finish"),work,phase.value("artifact_dependencies",Json::array()));
            row["model_ms"]=number_field(row,"model_ms")+number_field(out,"model_ms")+number_field(out.at("finish"),"model_ms");
            row["outputs"]=Json::array({out.at("finish")});
            row["status"]="completed";row["finish_ms"]=now();state.status="completed";state.group.clear();++terminal;
            return;
        }
        const auto& ranks=out.at("ranks");
        const auto progress=ranks.at(0).at("progress");
        for(const auto& rank:ranks){
            const auto& cursor=rank.at("progress");
            if(real_sp&&(cursor.at("group")!=state.group||cursor.at("ownership_epoch")!=state.epoch||
                cursor.at("quantum_seq")!=state.sequence+1))throw std::runtime_error("SP advance ownership/sequence mismatch");
            for(const auto* key:{"step","steps","block","branch","next_block","transfer_ready","done"})
                if(cursor.value(key,Json())!=progress.value(key,Json()))throw std::runtime_error("SP ranks disagree on cursor: "+std::string(key));
        }
        if(real_sp)++state.sequence;
        double model=0;for(const auto& rank:ranks)model=std::max(model,number_field(rank,"model_ms"));
        row["model_ms"]=number_field(row,"model_ms")+model;
        const int executed=quantum=="step"?cost.blocks_per_step(row.at("class").get<std::string>())-state.blocks:quantum.get<int>();
        const int previous_step=state.steps;
        state.steps=progress.value("step",0);state.total=progress.value("steps",0);state.prepared=true;
        if(progress.value("transfer_ready",true)||state.steps>previous_step)state.blocks=0;
        else state.blocks=progress.contains("block")?progress.value("block",0):state.blocks+executed;
        state.status="suspended";
    };
    auto harvest=[&](){
        bool progressed=false;
        for(auto it=tasks.begin();it!=tasks.end();){
            if(it->result.wait_for(std::chrono::seconds(0))!=std::future_status::ready){++it;continue;}
            try{auto out=it->result.get();finish_task(*it,out.output,out.rpc_completed_ms);}catch(const std::exception& error){
                auto& row=rows[it->row];row["status"]="failed";row["error"]=error.what();row["finish_ms"]=now();++terminal;
                record["failures"].push_back({{"request_id",row.at("request_id")},{"error",error.what()}});
                throw;
            }
            it=tasks.erase(it);progressed=true;
        }
        return progressed;
    };
    auto admit_arrival=[&](std::size_t i,double instant){
        auto& state=live[i];auto& row=rows[i];
        if(state.status!="pending"||number_field(row,"arrival_ms")>instant)return false;
        if(row.at("kind")=="image"&&(last_image_arrival.is_null()||number_field(row,"arrival_ms")>last_image_arrival.get<double>()))
            last_image_arrival=row.at("arrival_ms");
        state.status=row.at("kind")=="video"&&prefetch?(depth>=0?"encoding_wait":"encoding"):"waiting";
        if(state.status=="encoding")state.encode_gpu=static_cast<int>(encoders++%encoder_gpus);
        row["admission_ms"]=now();return true;
    };
    auto admit_arrivals=[&](double instant){
        bool progressed=false;
        for(std::size_t i=0;i<rows.size();++i)progressed=admit_arrival(i,instant)||progressed;
        return progressed;
    };
    auto note_encoder_rpc=[&](Json& row,const std::string& op,double start,double finish,bool ready,bool failed){
        // Bounded host round-trip observations include IPC and guard heartbeat,
        // not just encoder work. Keep two windows, never every poll response.
        auto& stats=row["encoder_rpc"];
        if(stats.is_null())stats={{"timing_scope","host_encoder_round_trip"},{"calls",0},
            {"total_ms",0.},{"max_ms",0.},{"failed_calls",0}};
        const double elapsed=finish-start;
        stats["calls"]=stats.at("calls").get<int>()+1;
        stats["total_ms"]=stats.at("total_ms").get<double>()+elapsed;
        stats["max_ms"]=std::max(stats.at("max_ms").get<double>(),elapsed);
        if(failed)stats["failed_calls"]=stats.at("failed_calls").get<int>()+1;
        if(op=="encode_start"||ready)stats[op=="encode_start"?"start":"ready_poll"]={
            {"start_ms",start},{"finish_ms",finish},{"failed",failed}};
    };
    auto release_stale=[&](){
        for(int gpu=0;gpu<gpus;++gpu)if(!busy[gpu]){
            for(const auto& id:stale[gpu]){
                Json ops=Json::array();traced(workers.get(video(gpu)),video(gpu),{{"op","release"},{"state_id",id}},budget.deadline(),ops);note_calls(rows[index.at(id)],ops);
            }
            stale[gpu].clear();
        }
    };
    try{while(terminal<rows.size()){
        check();
        bool progressed=harvest();
        const double instant=now();
        release_stale();
        for(std::size_t i=0;i<rows.size();++i){
            auto& state=live[i];auto& row=rows[i];
            // Keep the legacy per-row admission/encoding order for the control.
            progressed=admit_arrival(i,instant)||progressed;
            if(state.status=="encoding_wait"){
                for(int offset=0;offset<encoder_gpus;++offset){
                    const int gpu=static_cast<int>((encoders+offset)%encoder_gpus);
                    int outstanding=0;for(const auto& item:live)outstanding+=item.encode_gpu==gpu&&!item.ticket.empty();
                    if(busy[gpu]||outstanding>=depth)continue;
                    state.encode_gpu=gpu;state.status="encoding";encoders=static_cast<std::size_t>(gpu+1);break;
                }
            }
            if(state.status!="encoding"||busy[state.encode_gpu])continue;
            auto& w=workers.get(video(state.encode_gpu));Json ops=Json::array();
            const bool starting=state.ticket.empty();
            const std::string op=starting?"encode_start":"encode_poll";
            const Json action=starting?Json{{"op",op},{"spec",spec_of(row)}}:Json{{"op",op},{"ticket",state.ticket}};
            const double rpc_start=now();Json encoded;
            try{encoded=traced(w,video(state.encode_gpu),action,budget.deadline(),ops);}
            catch(...){note_encoder_rpc(row,op,rpc_start,now(),false,true);throw;}
            note_encoder_rpc(row,op,rpc_start,now(),encoded.value("ready",false),false);
            if(starting){state.ticket=encoded.at("ticket").get<std::string>();row["encode_submit_ms"]=now();
                int outstanding=0;for(const auto& item:live)outstanding+=item.encode_gpu==state.encode_gpu&&!item.ticket.empty();
                peak_tickets[state.encode_gpu]=std::max(peak_tickets[state.encode_gpu],outstanding);
                if(depth>=0&&outstanding>depth)throw std::runtime_error("prefetch depth contract exceeded");}
            else{
                if(encoded.value("ready",false)){
                    state.status="waiting";row["encode_ready_ms"]=now();progressed=true;
                    note_encoding(row,encoded,video(state.encode_gpu),w.process->pid(),"encode_poll",state.ticket);
                    if(encoded.contains("model_ms"))row["encode_ms"]=encoded.at("model_ms");
                    if(encoded.contains("outstanding_tickets"))row["encoder_tickets_at_ready"]=encoded.at("outstanding_tickets");
                }
            }
            note_calls(row,ops);
        }
        if(dispatch_refresh){
            // Encoding round trips may have outlasted another dispatch or a
            // new arrival. Plan from fresh eligibility without changing EDF.
            progressed=harvest()||progressed;
            // A newly idle rank can own stale migration replicas. Retire them
            // before it is reused, as in the original start-of-loop path.
            release_stale();
            progressed=admit_arrivals(now())||progressed;
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
                if(state.total>0)view["steps_total"]=state.total;
                else if(row.contains("steps_total"))view["steps_total"]=row.at("steps_total");
                if(!state.prepared&&!state.ticket.empty())view["affinity"]=state.encode_gpu;
                requests.push_back(view);
            }
            if(!requests.empty()){
                Json flags=Json::array();for(bool flag:busy)flags.push_back(flag);
                const auto plan=cluster_plan({{"now_ms",now()},{"gpus",gpus},{"busy",flags},{"requests",requests},
                    {"last_image_arrival_ms",last_image_arrival}},policy,cost,options);
                for(const auto& assignment:plan.at("assignments")){launch(assignment);progressed=true;}
            }
        }
        if(!progressed)wait_until(monotonic_seconds()+.0005);
    }}catch(...){
        for(auto& task:tasks)if(task.result.valid())task.result.wait();
        if(depth>=0)for(auto& state:live)if(!state.ticket.empty())try{
            workers.get(video(state.encode_gpu)).call({{"op","encode_cancel"},{"ticket",state.ticket}},
                std::min(budget.deadline(),monotonic_seconds()+1));state.ticket.clear();
        }catch(...){} // Process shutdown below remains the final owner of failed workers.
        for(auto& row:rows){if(!row.contains("status")||row["status"]=="pending")row["status"]="interrupted";if(!row.contains("finish_ms"))row["finish_ms"]=now();row["observation_ms"]=now();}
        record["requests"].insert(record["requests"].end(),rows.begin(),rows.end());throw;
    }
    for(auto& row:rows)row.erase("preempting");
    Json result={{"requests",rows},{"metrics",request_summary(rows)},{"elapsed_ms",now()},{"warmup",warmed},
            {"host_origin_monotonic_seconds",origin},
            {"by_kind",{{"video",kind_summary(rows,"video")},{"image",kind_summary(rows,"image")}}},
            {"policy",policy},{"gpus",gpus},{"real_backends_pinned_sp1",!fixtures&&!real_sp},{"real_sp",sp_config},
            {"text_prefetch_depth",depth>=0?Json(depth):Json()},{"peak_encoder_tickets",peak_tickets},
            {"dispatch_refresh",dispatch_refresh},
            {"worker_runtime",runtime_bindings},{"scheduler_options",options},{"generator",derived}};
    // Calibration: the simulator replays the same trace with the same profile,
    // policy and options; the executor's own decisions were the real ones.
    if(phase.value("simulate",false)){
        auto simulated=simulate_cluster({{"workload",trace},{"profile",phase.at("profile")},{"policy",policy},{"gpus",gpus},{"options",options}});
        result["simulated"]={{"metrics",simulated.at("metrics")},{"by_kind",simulated.at("by_kind")},{"makespan_ms",simulated.at("makespan_ms")},
                             {"options",options},{"scope","cost-model replay of this cell's trace; not a measurement"}};
    }
    return result;
}

std::set<std::string> video_outputs(const Json& phase) {
    if(phase.contains("save_videos")&&!phase.at("save_videos").is_boolean())
        throw std::invalid_argument("save_videos must be boolean");
    const auto declared=phase.value("video_outputs",Json::array());
    if(!declared.is_array())throw std::invalid_argument("video_outputs must be an array");
    std::set<std::string> paths;
    for(const auto& entry:declared){
        if(!entry.is_string())throw std::invalid_argument("video_outputs must contain relative .npy paths");
        const auto text=entry.get<std::string>();const fs::path path=text.rfind("@cell/",0)==0?text.substr(6):text;
        if(path.empty()||path.is_absolute()||path.extension()!=".npy"||path.lexically_normal()!=path)
            throw std::invalid_argument("video_outputs must contain normalized relative or @cell/ .npy paths");
        for(const auto& part:path)if(part==".."||part==".")throw std::invalid_argument("video_outputs cannot escape the rank directory");
        if(!paths.insert(text).second)throw std::invalid_argument("video_outputs cannot contain duplicates");
    }
    return paths;
}

class TemporaryVideos {
public:
    TemporaryVideos(const Json& phase,const fs::path& cell,Json& record,Workers& workers):
        declared_(video_outputs(phase)),record_(record),workers_(workers) {
        if(phase.value("save_videos",false)||declared_.empty())return;
        directory_=cell/"temporary-videos";
        fs::create_directories(cell);
        if(!fs::create_directory(directory_))throw std::runtime_error("temporary video directory already exists");
    }
    ~TemporaryVideos(){
        if(!directory_.empty()&&!closed_){
            // A timed-out worker could otherwise recreate files after cleanup.
            // Terminate/reap the phase's children before removing its workspace.
            try{workers_.stop();}catch(...){}
            cleanup();
        }
    }
    bool contains(const std::string& path)const{return !directory_.empty()&&declared_.count(path);}
    fs::path path(std::size_t rank,const std::string& name)const{
        return name.rfind("@cell/",0)==0?directory_/"shared"/name.substr(6):directory_/std::to_string(rank)/name;
    }
    void receipt(Json& result) {
        if(directory_.empty()||!result.contains("output")||!result.at("output").is_object()||!result.at("output").contains("path"))return;
        auto& output=result["output"];const fs::path path=output.at("path").get<std::string>();
        const auto relative=path.lexically_relative(directory_);
        if(relative.empty()||*relative.begin()=="..")return;
        if(fs::weakly_canonical(path)!=path||!fs::is_regular_file(path))throw std::runtime_error("temporary video is missing or outside its directory");
        const auto bytes=fs::file_size(path);
        if(output.contains("bytes")&&output.at("bytes")!=bytes)throw std::runtime_error("temporary video size mismatch");
        // Keep full-array evidence, but never register a soon-deleted video as
        // a retained artifact. evaluate still reads its full array before cleanup.
        output["sha256"]=file_hash(path);
        output["temporary_path"]=path.string();output["retained"]=false;
        output["storage"]="temporary_for_numerical_comparison";output.erase("path");
        files_.push_back({{"path",path.string()},{"sha256",output.at("sha256")},{"bytes",bytes}});
    }
    Json finish(){
        if(directory_.empty())return {{"status","not_required"}};
        cleanup();
        if(receipt_.at("status")!="removed")throw std::runtime_error("temporary video cleanup failed: "+receipt_.dump());
        return receipt_;
    }
private:
    void cleanup()noexcept{
        try{
            std::error_code error;fs::remove_all(directory_,error);
            receipt_={{"directory",directory_.string()},{"status",error?"failed":"removed"},
                {"outputs",files_},{"retained",false},{"scope","declared phase video outputs only"}};
            if(error)receipt_["error"]=error.message();
            if(!record_.contains("temporary_video_cleanup"))record_["temporary_video_cleanup"]=Json::array();
            record_["temporary_video_cleanup"].push_back(receipt_);closed_=true;
        }catch(...){} // Preserve the original execution exception; never throw from a destructor.
    }
    std::set<std::string> declared_;fs::path directory_;Json& record_;Workers& workers_;
    Json files_=Json::array(),receipt_=Json::object();bool closed_=false;
};

Json operations(const Json& phase,Workers& workers,Budget& budget,Json& record,const fs::path& work,const fs::path& cell_dir){
    const auto names=phase.value("workers",Json::array({phase.value("worker","model")}));
    if(!names.is_array() || names.empty())throw std::invalid_argument("action phase requires workers");
    const auto actions=phase.at("actions");if(!actions.is_array() || actions.empty())throw std::invalid_argument("SP phase requires actions");
    TemporaryVideos temporary_videos(phase,cell_dir,record,workers);
    Json rows=Json::array();
    for(auto action:actions){
        budget.check();if(cancelled())throw std::runtime_error("run cancelled");
        std::vector<std::future<Json>> pending;
        // Create children on the controlling thread; each call then owns one
        // independent pipe while collective ranks progress concurrently.
        std::vector<Worker*> group;for(const auto& name:names)group.push_back(&workers.get(name.get<std::string>()));
        // An action may address a subset of ranks, e.g. export on one GPU
        // and import on another; the others skip it.
        std::vector<std::size_t> called;
        if(action.contains("only_ranks")){for(const auto& rank:action.at("only_ranks")){const auto r=static_cast<std::size_t>(integer(rank,"rank"));if(r>=group.size())throw std::invalid_argument("action rank outside the phase workers");called.push_back(r);}}
        else for(std::size_t rank=0;rank<group.size();++rank)called.push_back(rank);
        for(const auto rank:called){
            auto message=action;message.erase("only_ranks");message["op"]=message.value("op","distributed");message["rank"]=rank;
            // A collective over a subset of the phase's workers names its own world size.
            if(!message.contains("world_size"))message["world_size"]=group.size();
            if(message.contains("per_rank")){message.merge_patch(message.at("per_rank").at(rank));message.erase("per_rank");}
            if(message.contains("output_path")&&message.at("output_path").is_string()&&
               temporary_videos.contains(message.at("output_path").get<std::string>())&&
               message.value("save_latents",false)&&!message.contains("latent_path")){
                auto latent=fs::path(message.at("output_path").get<std::string>());
                latent.replace_extension(".latent.npy");message["latent_path"]=latent.string();
            }
            // Relative paths are private to a rank; @cell/ names a file every rank of the cell can reach.
            for(const auto* field:{"output_path","output","reference","path","latent_path"})if(message.contains(field)&&message[field].is_string()){
                const auto text=message[field].get<std::string>();fs::path path=text;
                if(std::string(field)!="latent_path"&&std::string(field)!="path"&&temporary_videos.contains(text))
                    message[field]=temporary_videos.path(rank,text).string();
                else if(text.rfind("@cell/",0)==0)message[field]=(cell_dir/"artifacts"/"shared"/text.substr(6)).string();
                else if(path.is_relative())message[field]=(cell_dir/"artifacts"/std::to_string(rank)/path).string();
            }
            pending.push_back(std::async(std::launch::async,[&,rank,message]{return group[rank]->call(message,budget.deadline());}));
        }
        Json results=Json::array();std::exception_ptr error;
        for(std::size_t index=0;index<pending.size();++index)try{auto result=pending[index].get();result["worker_pid"]=group[called[index]]->process->pid();result["rank"]=called[index];temporary_videos.receipt(result);add_artifacts(record,result,work,phase.value("artifact_dependencies",Json::array()));results.push_back(result);}catch(...){if(!error)error=std::current_exception();}
        if(error)std::rethrow_exception(error);
        rows.push_back({{"action",action},{"ranks",results}});
    }
    return {{"operations",rows},{"requests",Json::array()},{"metrics",{{"operations",rows.size()},{"ranks",names.size()}}},
            {"temporary_video_cleanup",temporary_videos.finish()}};
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
        if(cell.contains("worker_overrides"))for(auto it=cell.at("worker_overrides").begin();it!=cell.at("worker_overrides").end();++it)
            definitions[it.key()]["config"].merge_patch(it.value());
        if(cell.at("kind")=="cluster"&&prefetch_depth(cell)>=0)for(const auto& slot:cell.at("gpus")){
            auto& options=definitions[slot.at("video").get<std::string>()]["config"]["options"];
            if(options.is_null())options=Json::object();
            options["text_prefetch_depth"]=prefetch_depth(cell);
        }
        Workers workers(definitions,request,cell_dir);
        const auto occupancy_backend=cell.value("occupancy_backend",std::string("nvidia_smi"));
        OccupancyProbe probe(request,cell_dir/"probe",occupancy_backend);
        double last_observation=0;
        std::mutex heartbeat_mutex;
        Json guard_telemetry={{"schema","ditserve.guard-telemetry.v1"},{"occupancy_backend",occupancy_backend},
            {"cadence_seconds",2.},{"probe_timeout_seconds",3.},{"fail_closed",true},
            {"scope","cell startup, warmup and execution; excludes shutdown"},
            {"heartbeat_calls",0},{"lock_wait_total_ms",0.},{"lock_wait_max_ms",0.},
            {"heartbeat_total_ms",0.},{"heartbeat_max_ms",0.},
            {"probe_calls",0},{"probe_total_ms",0.},{"probe_max_ms",0.},{"probe_errors",0},
            {"events",Json::array()},{"dropped_events",0}};
        auto duration_stat=[&](const char* total,const char* maximum,double elapsed){
            guard_telemetry[total]=guard_telemetry.at(total).get<double>()+elapsed;
            guard_telemetry[maximum]=std::max(guard_telemetry.at(maximum).get<double>(),elapsed);
        };
        workers.heartbeat([&]{
            // Collective ranks call concurrently. One observation owns the
            // probe paths, and its timestamp must not race with other ranks.
            const double started=monotonic_seconds();
            std::lock_guard<std::mutex> guard(heartbeat_mutex);
            guard_telemetry["heartbeat_calls"]=guard_telemetry.at("heartbeat_calls").get<int>()+1;
            duration_stat("lock_wait_total_ms","lock_wait_max_ms",(monotonic_seconds()-started)*1000);
            auto finish_heartbeat=[&](){duration_stat("heartbeat_total_ms","heartbeat_max_ms",(monotonic_seconds()-started)*1000);};
            try{
            if(cancelled())throw std::runtime_error("run cancelled");
            budget.check();
            if(!devices.empty()&&monotonic_seconds()-last_observation>=2){
                last_observation=monotonic_seconds();
                const double probe_start=monotonic_seconds();
                auto observed=probe.observe(devices,workers.pids(),std::min(budget.deadline(),monotonic_seconds()+3));
                const double probe_finish=monotonic_seconds(),probe_ms=(probe_finish-probe_start)*1000;
                guard_telemetry["probe_calls"]=guard_telemetry.at("probe_calls").get<int>()+1;
                duration_stat("probe_total_ms","probe_max_ms",probe_ms);
                const bool backend_matches=observed.value("occupancy_backend","")==occupancy_backend;
                const bool passed=observed.value("status","")=="exclusive_observed"&&backend_matches;
                if(!passed)guard_telemetry["probe_errors"]=guard_telemetry.at("probe_errors").get<int>()+1;
                if(guard_telemetry.at("events").size()<4096)guard_telemetry["events"].push_back({
                    {"start_monotonic_seconds",probe_start},{"finish_monotonic_seconds",probe_finish},
                    {"duration_ms",probe_ms},{"status",observed.value("status","unknown")},
                    {"occupancy_backend",observed.value("occupancy_backend","")},
                    {"owned_count",observed.value("owned",Json::array()).size()},
                    {"owned_processes",observed.value("owned",Json::array())},
                    {"foreign_count",observed.value("foreign",Json::array()).size()}});
                else guard_telemetry["dropped_events"]=guard_telemetry.at("dropped_events").get<int>()+1;
                if(!passed)throw std::runtime_error("GPU isolation lost or probe backend differs: "+observed.dump());
            }
            }catch(...){finish_heartbeat();throw;}
            finish_heartbeat();
        });
        Json controller_binding=nullptr;
        try{
            std::unique_ptr<ControllerAffinity> controller;
            if(cell.contains("controller_cpu_affinity")){
                // Spawn model workers with the original allowed mask before
                // narrowing the controller and its future dispatch threads.
                std::set<std::string> names;
                if(cell.at("kind")=="cluster"){
                    for(const auto& slot:cell.at("gpus"))for(const auto* role:{"video","image"})names.insert(slot.at(role).get<std::string>());
                    for(const auto& item:cell.value("warmup",Json::array()))names.insert(item.at("worker").get<std::string>());
                }else for(const auto& name:cell.value("workers",Json::array({cell.value("worker","model")})))names.insert(name.get<std::string>());
                for(const auto& name:names)workers.get(name).call({{"op","hello"}},budget.deadline());
                controller=std::make_unique<ControllerAffinity>(cell.at("controller_cpu_affinity"));
                controller_binding=controller->evidence;
            }
            Json result=cell.at("kind")=="operations"?operations(cell,workers,budget,record,work,cell_dir):cluster(cell,workers,budget,record,work,cell_dir);
            result["guard_telemetry"]=guard_telemetry;
            if(controller){result["controller_cpu_binding"]=controller->evidence;controller->restore();}
            result["cell"]=order[position];result["order"]=position<cells.size()?"forward":"reverse";result["status"]="completed";
            for(auto row:result.value("requests",Json::array())){row["phase"]=id;row["attempt"]=attempt_number;row["cell"]=order[position];row["order"]=result["order"];
                row["observation_id"]=id+"/"+std::to_string(attempt_number)+"/"+std::to_string(position);
                row["observation_ms"]=result.value("elapsed_ms",number_field(row,"finish_ms"));record["requests"].push_back(row);}
            attempt["cells"].push_back(result);
            workers.shutdown(std::min(budget.deadline(),monotonic_seconds()+5));
        }catch(const std::exception& failure){error=failure.what();Json failed={{"cell",order[position]},{"status","failed"},{"error",error},{"guard_telemetry",guard_telemetry}};
            if(!controller_binding.is_null())failed["controller_cpu_binding"]=controller_binding;
            attempt["cells"].push_back(failed);}
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
    // Scripted per-worker actions (numeric checks, cost profiles, SP ranks)
    // or the concurrent serving executor; nothing else runs models.
    const std::set<std::string> kinds={"operations","cluster"};
    for(const auto& phase:phases){
        const auto id=segment(phase.at("id").get<std::string>());if(seen.count(id))throw std::invalid_argument("duplicate phase");
        if(!kinds.count(phase.value("kind","")))throw std::invalid_argument("phase kind must be operations or cluster");
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
            if(cell.at("id")!=id || cell.value("kind","")!=phase.value("kind",""))throw std::invalid_argument("matrix cannot change phase identity or executor");
            if(cell.contains("record_dispatch_timeline")&&!cell.at("record_dispatch_timeline").is_boolean())
                throw std::invalid_argument("record_dispatch_timeline must be boolean");
            if(cell.contains("dispatch_refresh")&&(cell.value("kind","")!="cluster"||!cell.at("dispatch_refresh").is_boolean()))
                throw std::invalid_argument("dispatch_refresh must be boolean on a cluster phase");
            if(cell.contains("occupancy_backend")&&(!cell.at("occupancy_backend").is_string()||
                (cell.at("occupancy_backend")!="nvidia_smi"&&cell.at("occupancy_backend")!="nvml")))
                throw std::invalid_argument("occupancy_backend must be nvidia_smi or nvml");
            if(cell.contains("controller_cpu_affinity")){
                cpu_set_t allowed;
                if(sched_getaffinity(0,sizeof allowed,&allowed))throw std::runtime_error("cannot read controller CPU affinity");
                controller_mask(cell.at("controller_cpu_affinity"),allowed);
            }
            if(cell.contains("worker_config")&&!cell.at("worker_config").is_object())throw std::invalid_argument("worker_config must be an object");
            auto known=[&](const std::string& worker){if(!config.at("workers").contains(worker))throw std::invalid_argument("unknown worker: "+worker);};
            if(cell.contains("worker_config")){
                known(cell.value("worker","model"));
                for(const auto* field:{"device","gpu_uuid","device_id","backend","work_dir"})
                    if(cell.at("worker_config").contains(field))throw std::invalid_argument("matrix cannot change worker device or identity");
            }
            if(cell.contains("worker_overrides")){
                if(!cell.at("worker_overrides").is_object()||cell.contains("worker_config"))
                    throw std::invalid_argument("worker_overrides must be an object and cannot mix with worker_config");
                for(auto it=cell.at("worker_overrides").begin();it!=cell.at("worker_overrides").end();++it){
                    known(it.key());
                    if(!it.value().is_object())throw std::invalid_argument("each worker override must be a config object");
                    for(const auto* field:{"device","gpu_uuid","device_id","backend","work_dir"})
                        if(it.value().contains(field))throw std::invalid_argument("worker_overrides cannot change worker device or identity");
                }
            }
            if(cell.at("kind")=="operations"){
                video_outputs(cell);
                if(!cell.contains("actions")||!cell.at("actions").is_array()||cell.at("actions").empty())throw std::invalid_argument("operations phases need a nonempty actions array");
                std::set<std::string> ranks;
                for(const auto& worker:cell.value("workers",Json::array({cell.value("worker","model")}))){
                    const auto name=worker.get<std::string>();known(name);
                    if(!ranks.insert(name).second)throw std::invalid_argument("each rank requires a distinct worker");
                }
                continue;
            }
            for(const auto& item:cell.value("warmup",Json::array())){
                known(item.at("worker").get<std::string>());
                if(item.contains("spec")&&!item.at("spec").is_object())throw std::invalid_argument("warmup spec must be an object");
            }
            // Slots name the per-GPU video and image workers; rows name cost classes.
            const auto& slots=cell.at("gpus");
            if(!slots.is_array()||slots.empty())throw std::invalid_argument("cluster gpus must be a nonempty array of worker slots");
            std::set<std::string> names;
            for(const auto& slot:slots)for(const auto* role:{"video","image"}){
                const auto name=slot.at(role).get<std::string>();known(name);
                if(!names.insert(name).second)throw std::invalid_argument("each cluster slot needs its own workers");
            }
            prefetch_depth(cell);
            const auto sp=real_sp_config(cell);
            if(!sp.is_null()){
                if(slots.size()!=2)throw std::invalid_argument("real_sp requires exactly two GPU slots");
                std::set<std::string> devices;
                for(const auto& slot:slots){
                    const auto& definition=config.at("workers").at(slot.at("video").get<std::string>());
                    auto effective=definition.value("config",Json::object());
                    effective.merge_patch(cell.value("worker_overrides",Json::object()).value(slot.at("video").get<std::string>(),Json::object()));
                    if(definition.at("backend")!="wan"||effective.value("options",Json::object()).value("execution",std::string())!="block")
                        throw std::invalid_argument("real_sp requires Wan block workers");
                    const auto device=definition.value("gpu_uuid",std::string());
                    if(device.empty()||!devices.insert(device).second)throw std::invalid_argument("real_sp requires distinct bound GPU UUIDs");
                }
                if(cell.at("policy")!="static_sp1"&&cell.at("policy")!="static_sp2"&&cell.at("policy")!="step_elastic"&&cell.at("policy")!="block_elastic")
                    throw std::invalid_argument("real_sp supports degrees one and two only");
            }
            const ClusterCost cost(cell.at("profile"));
            const auto rows=phase_workload(cell,cost,static_cast<int>(slots.size()));if(rows.empty())throw std::invalid_argument("phase workload must be nonempty");
            if(!cluster_policy(cell.at("policy").get<std::string>()))throw std::invalid_argument("unknown cluster policy");
            cluster_plan({{"now_ms",0},{"gpus",static_cast<int>(slots.size())},{"requests",Json::array()}},cell.at("policy"),cost,cell.value("options",Json::object()));
            for(const auto& row:rows)if(cost.klass(row.at("class").get<std::string>()).at("kind")!=row.at("kind"))
                throw std::invalid_argument("request kind differs from its cost class");
        }
    }
}
}

void validate_study(const Json& config){validate(config);}

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
            observation=gpu_observation(request,attempt_dir,devices,{},std::min(waiting_deadline,monotonic_seconds()+15));
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
    if(!config.contains("profile")){
        config["profile"]=synthetic_profile();
        if(!config.contains("options"))config["options"]={{"quantum_ms",250}};
    }
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
    Json phase=config;phase["matrix"]=config.value("matrix",Json::object());phase.erase("variants");
    // Variants are merge patches applied to every cell (e.g. the policies an
    // option applies to); without them each cell runs once.
    const auto variants=config.value("variants",Json::array({Json::object()}));
    if(!variants.is_array()||variants.empty())throw std::invalid_argument("variants must be a nonempty array of objects");
    const auto cells=expand_matrix(phase);
    for(std::size_t index=0;index<cells.size();++index)for(std::size_t variant=0;variant<variants.size();++variant){
        Json cell=cells[index];cell.merge_patch(variants[variant]);
        const auto suffix="-c"+std::to_string(index)+(variants.size()>1?"-v"+std::to_string(variant):"");
        const ClusterCost cost(cell.at("profile"));const int gpus=cell.value("gpus",4);
        Json derived=nullptr;
        const auto rows=phase_workload(cell.contains("workload")?Json{{"workload",cell.at("workload")}}:Json{{"generator",cell.at("generator")}},cost,gpus,&derived);
        for(const auto& policy:cell.value("policies",Json::array({"static_sp1","step_elastic","block_elastic"}))){
            const auto name=policy.get<std::string>();
            if(name.rfind("static_sp",0)==0 && std::stoi(name.substr(9))>gpus){
                record["phases"].push_back({{"id",name+suffix},{"status","skipped"},{"policy",policy},{"cell",index},{"variant",variant},
                    {"reason","fixed SP degree exceeds the cell's GPUs"}});
                continue;
            }
            Json input={{"workload",rows},{"profile",cell.at("profile")},{"policy",policy},{"gpus",gpus},
                {"options",cell.value("options",Json::object())},{"dispatch_ms",cell.value("dispatch_ms",0.)}};
            if(cell.contains("encode_slots"))input["encode_slots"]=cell.at("encode_slots");
            auto result=simulate_cluster(input);
            if(!cell.value("keep_requests",false))result.erase("requests");
            Json settings=cell;settings.erase("workload");settings.erase("matrix");settings.erase("profile");settings.erase("policies");
            if(!derived.is_null())result["generator"]=derived;
            record["phases"].push_back({{"id",name+suffix},{"status","completed"},
                {"policy",policy},{"cell",index},{"variant",variant},{"config",settings},{"result",result}});
        }
    }
    record["archive"]=publish(record,request,work);return record;
}
}
