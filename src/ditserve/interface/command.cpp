#include "ditserve/interface/command.hpp"
#include "ditserve/control/dispatch.hpp"
#include "ditserve/control/study.hpp"
#include "ditserve/runtime/process.hpp"
#include "ditserve/benchmark/cluster_sim.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <stdexcept>
#include <unistd.h>

namespace ditserve {
namespace {
namespace fs = std::filesystem;

std::string environment(const char* key) {
    const char* value = std::getenv(key);
    return value ? value : "";
}

fs::path absolute(const std::string& path) {
    if (path.empty()) throw std::invalid_argument("path must be nonempty");
    auto result = fs::absolute(path).lexically_normal();
    if (result.filename().empty() && result != result.root_path()) result = result.parent_path();
    return result;
}

std::string segment(const std::string& value, const char* name) { return path_segment(value, name); }

Json read_json(std::istream& input) {
    constexpr std::size_t limit = 64 * 1024 * 1024;
    std::string text;
    std::array<char, 8192> buffer{};
    while (input) {
        input.read(buffer.data(), buffer.size());
        text.append(buffer.data(), static_cast<std::size_t>(input.gcount()));
        if (text.size() > limit) throw std::invalid_argument("JSON input exceeds 64 MiB");
    }
    return Json::parse(text);
}

Json read_config(const fs::path& path) {
    std::ifstream input(path);
    if (!input) throw std::invalid_argument("cannot read config: " + path.string());
    return read_json(input);
}

fs::path report_output(const fs::path& work, const fs::path& root) {
    // This only chooses a destination. The Python finalizer verifies the
    // journal checksum, work identity and source bindings before publication.
    if (fs::exists(work / ".finalization.json"))
        return absolute(string_field(read_config(work / ".finalization.json"), "destination", ""));
    const auto run = read_config(work / "run.json");
    return root / "results" / segment(string_field(run, "experiment", ""), "experiment") /
        segment(string_field(run, "run_id", ""), "run_id");
}

fs::path executable() { return fs::canonical("/proc/self/exe"); }

fs::path project_root() {
    // A relocated build or an installation takes precedence over its build host.
    for (auto path = executable().parent_path(); !path.empty(); path = path.parent_path()) {
        if (fs::is_regular_file(path / "workers/ditserve/runtime/worker.py")) return path;
        if (fs::is_regular_file(path / "share/ditserve/python/ditserve/runtime/worker.py")) return path;
        if (path == path.root_path()) break;
    }
    return DITSERVE_PROJECT_ROOT;
}

std::string configured(const Json& request, const char* key, const char* variable) {
    return request.contains(key) ? string_field(request, key, "") : environment(variable);
}

void normalize_environment(Json& request) {
    const auto discovered = project_root();
    const auto root_value = configured(request, "root", "DITSERVE_ROOT");
    const auto root = root_value.empty() ? discovered : absolute(root_value);
    request["root"] = root.string();
    auto package = configured(request, "package_root", "DITSERVE_PACKAGE_ROOT");
    if (package.empty()) {
        for (const auto& candidate : {root / "workers", discovered / "workers", discovered / "share/ditserve/python"}) {
            if (fs::is_regular_file(candidate / "ditserve/runtime/worker.py")) {
                package = candidate.string();
                break;
            }
        }
        if (package.empty()) throw std::invalid_argument("Python package not found; set --package-root or DITSERVE_PACKAGE_ROOT");
    }
    request["package_root"] = absolute(package).string();
    auto python = configured(request, "python", "DITSERVE_PYTHON");
    if (python.empty()) {
        const auto local = root / ".venv-previs/bin/python";
        python = ::access(local.c_str(), X_OK) == 0 ? local.string() : "python3";
    }
    request["python"] = python;
}

std::string new_run_id(bool synthetic) {
    std::random_device random;
    constexpr char digits[] = "0123456789abcdef";
    std::string value = synthetic ? "simulate-" : "";
    for (int index = 0; index < 12; ++index) value += digits[random() & 15];
    return value;
}

void normalize_study(const std::string& command, Json& request) {
    const bool synthetic = command == "simulate-study";
    if (request.contains("config") && request.at("config").is_string())
        request["config"] = read_config(absolute(request.at("config").get<std::string>()));
    if (synthetic && (!request.contains("config") || request.at("config").is_null()))
        request["config"] = {{"generator", {{"count", int_field(request, "count", 32, 0)},
                                             {"seed", request.value("seed", Json(7))}}}};
    if (!request.contains("config") || !request.at("config").is_object())
        throw std::invalid_argument("config must be a JSON object or config file path");
    // Simulations default to the "synthetic" experiment; a named what-if keeps its own results directory.
    const auto experiment = segment(string_field(request.at("config"), "experiment", synthetic ? "synthetic" : ""), "experiment");
    const bool resume = bool_field(request, "resume", false);
    const bool repair = bool_field(request, "repair", false);
    if (repair != request.contains("retry_of")) throw std::invalid_argument("--repair and --retry-of must be used together");
    if (resume && !request.contains("run_id") && !request.contains("work_dir"))
        throw std::invalid_argument("--resume requires --run-id or --work-dir");
    const auto inferred = resume && request.contains("work_dir")
        ? absolute(request.at("work_dir").get<std::string>()).filename().string() : new_run_id(synthetic);
    const auto run_id = segment(string_field(request, "run_id", inferred), "run_id");
    request["run_id"] = run_id;
    const fs::path root = request.at("root").get<std::string>();
    if (!request.contains("work_dir")) request["work_dir"] = (root / ".cache/results-work" / experiment / run_id).string();
    if (!request.contains("results_dir")) {
        const fs::path results = request.contains("results_root") ? absolute(request.at("results_root").get<std::string>()) : root / "results";
        request["results_dir"] = (results / experiment / run_id).string();
    }
    if (!synthetic && !request.contains("ledger")) request["ledger"] = (root / ".state" / experiment / "budget.json").string();
    for (const auto* key : {"work_dir", "results_dir", "ledger"})
        if (request.contains(key)) request[key] = absolute(request.at(key).get<std::string>()).string();
    if (!request.contains("prune")) request["prune"] = !bool_field(request, "keep_raw", false);
    bool_field(request, "prune", true);
    bool_field(request, "allow_gpu", false);
}

struct Arguments {
    std::string command;
    Json request = Json::object();
    bool rpc = false, json = false, help = false, version = false;
};

std::pair<std::string, std::string> option(const std::string& text) {
    const auto equal = text.find('=');
    return {text.substr(0, equal), equal == std::string::npos ? "" : text.substr(equal + 1)};
}

Arguments parse(int argc, char** argv) {
    Arguments result;
    std::vector<std::string> remaining;
    std::set<std::string> seen;
    for (int index = 1; index < argc; ++index) {
        const std::string token = argv[index];
        const auto parts = option(token);
        if (parts.first == "--help" || parts.first == "-h" || parts.first == "--version" || parts.first == "--json") {
            if (token.find('=') != std::string::npos) throw std::invalid_argument("flag takes no value: " + parts.first);
            if (parts.first == "--version") result.version = true;
            else if (parts.first == "--json") result.json = true;
            else result.help = true;
        }
        else if (parts.first == "--root" || parts.first == "--package-root" || parts.first == "--python") {
            if (!seen.insert(parts.first).second) throw std::invalid_argument("duplicate option: " + parts.first);
            auto value = parts.second;
            if (token.find('=') == std::string::npos) {
                if (++index == argc) throw std::invalid_argument("missing value for " + parts.first);
                value = argv[index];
            }
            if (value.empty() || value.rfind("--", 0) == 0) throw std::invalid_argument("missing value for " + parts.first);
            auto field = parts.first.substr(2); std::replace(field.begin(), field.end(), '-', '_');
            result.request[field] = value;
        } else remaining.push_back(token);
    }
    if (remaining.empty()) {
        if (!result.version && !result.help) throw std::invalid_argument("a command is required; use --help");
        return result;
    }
    result.command = remaining.front();
    remaining.erase(remaining.begin());
    if (result.help || result.version) return result;
    if (result.command == "rpc") {
        if (remaining.size() != 1) throw std::invalid_argument("usage: ditserve rpc COMMAND < request.json");
        result.command = remaining.front(); result.rpc = true;
        const std::set<std::string> commands = {"generate", "simulate", "control", "run", "simulate-study", "doctor", "report", "results"};
        if (!commands.count(result.command)) throw std::invalid_argument("unknown RPC command: " + result.command);
        return result;
    }
    std::set<std::string> allowed;
    bool path_required = false;
    if (result.command == "run") allowed = {"config", "run-id", "work-dir", "results-root", "ledger", "phase", "resume", "allow-gpu", "repair", "retry-of"};
    else if (result.command == "simulate") allowed = {"config", "run-id", "work-dir", "results-root", "count", "seed", "keep-raw"};
    else if (result.command == "doctor") {}
    else if (result.command == "report") { allowed = {"output", "keep-raw"}; path_required = true; }
    else if (result.command == "compact-results") {
        result.command = "results"; result.request["op"] = "compact";
        allowed = {"destination", "prune-raw"}; path_required = true;
    } else if (result.command == "results") {
        if (remaining.empty()) throw std::invalid_argument("results requires inspect, verify or compact");
        const auto op = remaining.front(); remaining.erase(remaining.begin());
        if (op != "inspect" && op != "verify" && op != "compact") throw std::invalid_argument("unknown results operation: " + op);
        result.request["op"] = op; path_required = true;
        if (op == "compact") allowed = {"destination", "prune-raw"};
    } else throw std::invalid_argument("unknown command: " + result.command);
    const std::set<std::string> flags = {"keep-raw", "resume", "allow-gpu", "repair", "prune-raw"};
    std::vector<std::string> positional;
    for (std::size_t index = 0; index < remaining.size(); ++index) {
        const auto token = remaining[index];
        if (token == "--") {
            positional.insert(positional.end(), remaining.begin() + index + 1, remaining.end()); break;
        }
        if (token.rfind("--", 0) != 0) { positional.push_back(token); continue; }
        const auto parts = option(token);
        auto field = parts.first.substr(2);
        if (!allowed.count(field)) throw std::invalid_argument("unknown option: " + parts.first);
        if (!seen.insert(parts.first).second) throw std::invalid_argument("duplicate option: " + parts.first);
        const bool flag = flags.count(field);
        std::replace(field.begin(), field.end(), '-', '_');
        if (flag) {
            if (token.find('=') != std::string::npos) throw std::invalid_argument("flag takes no value: " + parts.first);
            result.request[field] = true; continue;
        }
        auto value = parts.second;
        if (token.find('=') == std::string::npos) {
            if (++index == remaining.size()) throw std::invalid_argument("missing value for " + parts.first);
            value = remaining[index];
        }
        if (value.empty() || value.rfind("--", 0) == 0) throw std::invalid_argument("missing value for " + parts.first);
        if (field == "count" || field == "seed") {
            std::int64_t parsed = 0;
            const auto end = value.data() + value.size();
            const auto conversion = std::from_chars(value.data(), end, parsed);
            if (conversion.ec != std::errc() || conversion.ptr != end || parsed < 0)
                throw std::invalid_argument(parts.first + " must be a nonnegative integer");
            result.request[field] = parsed;
        } else result.request[field] = value;
    }
    if (positional.size() != static_cast<std::size_t>(path_required))
        throw std::invalid_argument(path_required ? "exactly one path is required" : "unexpected positional argument");
    if (path_required) result.request[result.command == "report" ? "work_dir" : "path"] = positional.front();
    if (result.command == "run" && !result.request.contains("config")) throw std::invalid_argument("run requires --config");
    return result;
}

void help() {
    std::cout << "ditserve " DITSERVE_VERSION " — native C++ experiment controller\n"
        "Usage: ditserve [--json] [--root PATH] [--python PROGRAM] [--package-root PATH] COMMAND\n\n"
        "  doctor\n"
        "  simulate [--config PATH] [--count N] [--seed N] [--keep-raw]\n"
        "           [--run-id ID] [--work-dir PATH] [--results-root PATH]\n"
        "  run --config PATH [--run-id ID] [--work-dir PATH] [--results-root PATH]\n"
        "      [--ledger PATH] [--phase ID] [--resume] [--allow-gpu] [--repair --retry-of ID]\n"
        "  report WORK_DIR [--output PATH] [--keep-raw]\n"
        "  results inspect|verify PATH\n"
        "  results compact ROOT [--destination PATH] [--prune-raw]\n"
        "  compact-results ROOT [--destination PATH] [--prune-raw]\n"
        "  rpc generate|simulate|control|run|simulate-study|doctor|report|results < request.json\n\n"
        "Global options work before or after commands. --help and --version need no Python.\n"
        "DITSERVE_ROOT, DITSERVE_PYTHON and DITSERVE_PACKAGE_ROOT provide defaults.\n";
}

Json pure_rpc(const std::string& command, const Json& request) {
    if (command == "control") {
        const auto op = request.at("op");
        if (op == "workload") return study_workload(request.at("phase"));
        if (op == "matrix") return expand_matrix(request.at("phase"));
        if (op == "summary") return request_summary(request.at("rows"));
        if (op == "gates") return evaluate_gates(request.at("evidence"), request.at("rules"));
        throw std::invalid_argument("unknown control operation");
    }
    if (command == "generate") {
        check_keys(request, {"generator"});
        return study_workload({{"generator", request.at("generator")}});
    }
    Json config = request;
    if (!config.contains("profile")) {
        config["profile"] = synthetic_profile();
        if (!config.contains("options")) config["options"] = {{"quantum_ms", 250}};
    }
    if (!config.contains("workload")) config["workload"] = study_workload({{"generator", config.value("generator", Json::object())}});
    config.erase("generator");
    return simulate_cluster(config);
}

struct TemporaryLog {
    fs::path directory;
    TemporaryLog() {
        auto name = (fs::temp_directory_path() / "ditserve-service-XXXXXX").string();
        if (!::mkdtemp(name.data())) throw std::runtime_error("cannot create service log directory");
        directory = name;
    }
    ~TemporaryLog() { std::error_code error; fs::remove_all(directory, error); }
};


Json service(const Json& request, const std::string& backend, Json action) {
    TemporaryLog log;
    CancelScope signals;
    const fs::path root = request.at("root").get<std::string>();
    const fs::path package = request.at("package_root").get<std::string>();
    ProcessOptions options;
    options.argv = {request.at("python"), "-u", "-m", "ditserve.runtime.worker", "--backend", backend};
    options.stderr_path = log.directory / "stderr.log";
    options.max_message_bytes = 65 * 1024 * 1024; // 64 MiB result plus protocol envelope.
    options.heartbeat = [] {
        if (CancelScope::requested()) throw std::runtime_error("service cancelled");
    };
    options.environment = {{"PYTHONPATH", package.string() + ":" + environment("PYTHONPATH")},
        {"DITSERVE_ROOT", root.string()}, {"PYTHONDONTWRITEBYTECODE", "1"},
        {"DITSERVE_REPORT_TOOLS", (package.parent_path() / ".cache/tools").string()},
        {"HF_HUB_OFFLINE", "1"}, {"TRANSFORMERS_OFFLINE", "1"}};
    try {
        WorkerProcess worker(options);
        action["version"] = 1; action["id"] = 1;
        auto result = worker.call(action, monotonic_seconds() + 120);
        if (!worker.stop()) throw std::runtime_error("service cleanup could not be verified");
        return result;
    } catch (const std::exception& error) {
        std::ifstream diagnostic(options.stderr_path, std::ios::binary | std::ios::ate);
        std::string detail;
        if (diagnostic && diagnostic.tellg() > 0) {
            const auto size = static_cast<std::streamoff>(diagnostic.tellg());
            diagnostic.seekg(std::max<std::streamoff>(0, size - 8192));
            detail.assign(std::istreambuf_iterator<char>(diagnostic), std::istreambuf_iterator<char>());
        }
        throw std::runtime_error(std::string(error.what()) + (detail.empty() ? "" : "\nworker stderr (tail):\n" + detail));
    }
}

Json dispatch(const std::string& command, Json request) {
    if (!request.is_object()) throw std::invalid_argument("request must be a JSON object");
    if (command == "generate" || command == "simulate" || command == "control") return pure_rpc(command, request);
    if (command != "run" && command != "simulate-study" && command != "doctor" && command != "report" && command != "results")
        throw std::invalid_argument("unknown RPC command: " + command);
    request = normalize_request(command, std::move(request));
    if (command == "run") return run_study(request);
    if (command == "simulate-study") return simulate_study(request);
    if (command == "doctor") {
        auto result = service(request, "environment", {{"op", "inspect"}, {"root", request.at("root")}});
        result["core_binary"] = executable().string(); result["root"] = request.at("root");
        result["core_version"] = core_version(); result["reference_simulation_ready"] = true;
        return result;
    }
    if (command == "report") {
        Json action = {{"op", "finalize"}, {"work_dir", request.at("work_dir")}, {"prune", request.at("prune")}};
        if (request.contains("output")) action["output"] = request.at("output");
        return service(request, "results", action);
    }
    Json action = {{"op", request.at("op")}};
    if (request.at("op") == "compact") {
        action["root"] = request.at("path"); action["prune_raw"] = bool_field(request, "prune_raw", false);
        if (request.contains("destination")) action["destination"] = request.at("destination");
    } else action["path"] = request.at("path");
    return service(request, "results", action);
}

bool failed(const Json& result) {
    if (!result.is_object()) return false;
    const std::set<std::string> failures = {"failed", "incomplete", "unsettled", "blocked", "deferred", "protected"};
    if (result.contains("status") && result.at("status").is_string() && failures.count(result.at("status").get<std::string>())) return true;
    for (const auto* key : {"archive", "cleanup"}) if (result.contains(key) && failed(result.at(key))) return true;
    return false;
}
}

Json normalize_request(const std::string& command, Json request) {
    if (!request.is_object()) throw std::invalid_argument("request must be a JSON object");
    normalize_environment(request);
    if (command == "run" || command == "simulate-study") normalize_study(command, request);
    else if (command == "report") {
        request["work_dir"] = absolute(string_field(request, "work_dir", "")).string();
        request["output"] = request.contains("output") ? absolute(request.at("output").get<std::string>()).string()
            : report_output(request.at("work_dir").get<std::string>(), request.at("root").get<std::string>()).string();
        if (!request.contains("prune")) request["prune"] = !bool_field(request, "keep_raw", false);
        bool_field(request, "prune", true);
    } else if (command == "results") {
        const auto op = string_field(request, "op", "");
        if (op != "inspect" && op != "verify" && op != "compact") throw std::invalid_argument("unknown results operation: " + op);
        request["path"] = absolute(string_field(request, "path", "")).string();
        if (request.contains("destination")) request["destination"] = absolute(request.at("destination").get<std::string>()).string();
        else if (op == "compact") {
            const fs::path path = request.at("path").get<std::string>();
            if (fs::exists(path / "run.json") && read_config(path / "run.json").value("schema", "") == "ditserve.run.v1")
                request["destination"] = report_output(path, request.at("root").get<std::string>()).string();
        }
        bool_field(request, "prune_raw", false);
    }
    return request;
}

int command_main(int argc, char** argv) {
    try {
        auto args = parse(argc, argv);
        if (args.version) { std::cout << "ditserve " DITSERVE_VERSION "\n"; return 0; }
        if (args.help) { help(); return 0; }
        if (args.rpc) {
            auto request = read_json(std::cin);
            if (!request.is_object()) throw std::invalid_argument("request must be a JSON object");
            if (args.command != "generate" && args.command != "simulate" && args.command != "control")
                request.update(args.request);
            args.request = std::move(request);
        }
        const auto command = !args.rpc && args.command == "simulate" ? "simulate-study" : args.command;
        const auto result = dispatch(command, args.request);
        auto display = result;
        if (!args.rpc && !args.json && (command == "run" || command == "simulate-study")) {
            display = Json::object();
            for (const auto* field : {"run_id", "status", "metrics", "archive", "failures"})
                if (result.contains(field)) display[field] = result.at(field);
        }
        std::cout << display.dump(args.rpc || args.json ? -1 : 2) << '\n';
        return failed(result) ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "ditserve: " << error.what() << '\n';
        return 2;
    }
}
}
