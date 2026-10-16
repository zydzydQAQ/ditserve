#include "ditserve/control/budget.hpp"
#include "ditserve/runtime/process.hpp"
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <signal.h>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>

using namespace ditserve;
namespace {
void require(bool passed, const std::string& message) { if (!passed) throw std::runtime_error(message); }
void fails(const std::function<void()>& action, const std::string& message) {
    try { action(); } catch (const std::exception&) { return; }
    throw std::runtime_error(message);
}
void near(double actual, double expected) { require(std::abs(actual - expected) < 1e-8, "charge mismatch"); }
Json request(int id, const std::string& op = "echo") { return {{"version", 1}, {"id", id}, {"op", op}}; }
std::string executable() {
    char name[4096];
    const auto count = readlink("/proc/self/exe", name, sizeof(name));
    if (count <= 0 || count == sizeof(name)) throw std::runtime_error("cannot find test executable");
    return std::string(name, count);
}
int child(int argc, char** argv) {
    for (std::string line; std::getline(std::cin, line);) {
        const auto value = Json::parse(line);
        const auto operation = value.at("op").get<std::string>();
        if (operation == "wait") { sleep(10); return 0; }
        if (operation == "exit") return 3;
        if (operation == "large") { std::cout << std::string(10000, 'x') << '\n' << std::flush; continue; }
        if (operation == "stderr") std::cerr << "worker diagnostic\n" << std::flush;
        Json result{{"pid", getpid()}, {"echo", value.value("payload", Json(nullptr))},
                    {"argument", argc > 2 ? argv[2] : ""},
                    {"environment", getenv("DITSERVE_TEST_ENV") ? getenv("DITSERVE_TEST_ENV") : ""}};
        if (operation == "fork") {
            int ready[2];
            if (pipe(ready)) return 4;
            const auto descendant = fork();
            if (descendant < 0) return 4;
            if (!descendant) {
                close(ready[0]); signal(SIGTERM, SIG_IGN);
                if (write(ready[1], "1", 1) != 1) _exit(5);
                close(ready[1]);
                for (;;) pause();
            }
            close(ready[1]);
            char signal;
            if (read(ready[0], &signal, 1) != 1) return 4;
            close(ready[0]);
            result["descendant"] = descendant;
        }
        Json response{{"version", 1}, {"id", value["id"]}, {"ok", true}, {"result", result}};
        if (operation == "wrong_id") response["id"] = 999;
        if (operation == "float_id") response["id"] = value["id"].get<double>();
        if (operation == "error") response = {{"version", 1}, {"id", value["id"]}, {"ok", false},
                                              {"error", {{"type", "MockFailure"}, {"message", "deliberate failure"}}}};
        const auto encoded = response.dump() + '\n';
        if (operation == "split") {
            std::cout << encoded.substr(0, 7) << std::flush;
            usleep(5000);
            std::cout << encoded.substr(7) << std::flush;
        } else std::cout << encoded << std::flush;
        if (operation == "shutdown") return 0;
    }
    return 0;
}
void processes(const std::filesystem::path& root) {
    const std::string literal = "spaces;$(not-a-command)";
    ProcessOptions options{{executable(), "--child", literal}, {{"DITSERVE_TEST_ENV", literal}}, root / "worker.log"};
    options.cleanup_timeout_seconds = .3;
    WorkerProcess worker(options);
    auto input = request(1);
    input["payload"] = std::string(131072, 'p');
    const auto first = worker.call(input, monotonic_seconds() + 3);
    require(first["echo"] == input["payload"] && first["argument"] == literal && first["environment"] == literal,
            "worker changed literal argument/environment or partial-pipe request");
    const auto second = worker.call(request(2, "split"), monotonic_seconds() + 3);
    require(first["pid"] == second["pid"], "worker was restarted between calls");
    worker.call(request(3, "stderr"), monotonic_seconds() + 3);
    require(worker.stop() && worker.stop(), "normal worker cleanup failed");
    int status;
    require(waitpid(worker.pid(), &status, WNOHANG) < 0 && errno == ECHILD, "worker was not reaped");
    std::ifstream log(options.stderr_path);
    std::string diagnostic;
    std::getline(log, diagnostic);
    require(diagnostic == "worker diagnostic", "stderr contaminated or missed protocol log");
    WorkerProcess graceful(options);
    graceful.call(request(1, "shutdown"), monotonic_seconds() + 3);
    require(graceful.stop(), "valid final response was lost at worker exit");
    for (const std::string operation : {"wrong_id", "float_id", "error", "exit", "large"}) {
        auto bounded = options;
        bounded.max_message_bytes = 1024;
        WorkerProcess invalid(bounded);
        fails([&] { invalid.call(request(1, operation), monotonic_seconds() + 3); }, operation + " was accepted");
        require(invalid.cleanup_verified(), operation + " left worker alive");
    }
    WorkerProcess timeout(options);
    const double started = monotonic_seconds();
    fails([&] { timeout.call(request(1, "wait"), started + .04); }, "worker deadline ignored");
    require(timeout.cleanup_verified() && monotonic_seconds() - started < 1, "timeout cleanup was not bounded");
    auto supervised = options;
    int heartbeats = 0;
    supervised.heartbeat = [&] { if (++heartbeats == 2) throw std::runtime_error("GPU isolation lost"); };
    WorkerProcess monitored(supervised);
    const double monitored_start = monotonic_seconds();
    fails([&] { monitored.call(request(1, "wait"), monitored_start + 10); }, "heartbeat failure was ignored");
    require(heartbeats == 2 && monitored.cleanup_verified() && monotonic_seconds() - monitored_start < 2,
            "heartbeat waited for the hung action deadline");
    heartbeats = 0;
    supervised.heartbeat = [&] { ++heartbeats; };
    WorkerProcess watched_timeout(supervised);
    const double watched_start = monotonic_seconds();
    fails([&] { watched_timeout.call(request(1, "wait"), watched_start + .04); }, "heartbeat overrode action deadline");
    require(heartbeats == 1 && watched_timeout.cleanup_verified() && monotonic_seconds() - watched_start < 1,
            "heartbeat relaxed the hard action deadline");
    WorkerProcess tree(options);
    const auto descendant = tree.call(request(1, "fork"), monotonic_seconds() + 3)["descendant"].get<int>();
    require(tree.stop(), "descendant process group cleanup failed");
    require(kill(descendant, 0) < 0 && errno == ESRCH, "worker descendant survived cleanup");
    auto missing = options;
    missing.argv = {"/not/a/worker"};
    fails([&] { WorkerProcess absent(missing); }, "missing worker executable accepted");
}
BudgetLimits limits(const std::string& carry = "none") {
    BudgetLimits value;
    value.total_gpu_seconds = 100;
    value.repair_reserve_gpu_seconds = 20;
    value.phase_gpu_seconds = {{"a", 40}, {"b", 40}};
    value.phase_order = {"a", "b"};
    value.carry = carry;
    return value;
}
void budgets(const std::filesystem::path& root) {
    double clock = 100, wall = 1000;
    auto monotonic = [&] { return clock; };
    auto realtime = [&] { return wall; };
    const Json identity{{"config_sha256", "frozen-config"}};
    const auto path = root / "budget.json";
    {
        Budget budget(path, limits(), monotonic, realtime);
        fails([&] { Budget locked(path, limits(), monotonic, realtime); }, "ledger lock not exclusive");
        const auto initial = budget.begin(identity, 2, "a", 100);
        near(budget.deadline(), clock + 20);
        clock += 10;
        budget.check(1);
        fails([&] { budget.check(11); }, "whole-work reservation exceeded phase budget");
        fails([&] { budget.finish("failed", false); }, "unverified cleanup settled budget");
        require(!budget.snapshot()["active"].is_null(), "unverified cleanup cleared active session");
        const auto failed = budget.finish("failed", true);
        near(failed["gpu_seconds"], 20);
        near(budget.snapshot()["used_gpu_seconds"], 20);
        Json retry = identity;
        retry["retry_of"] = initial["session_id"];
        auto changed = retry;
        changed["config_sha256"] = "different";
        fails([&] { budget.begin(changed, 1, "a", 100, true); }, "repair changed frozen configuration");
        fails([&] { budget.begin(identity, 1, "a", 100, true); }, "unreferenced repair admitted");
        budget.begin(retry, 1, "a", 100, true);
        clock += 3;
        budget.finish("completed", true);
        near(budget.snapshot()["used_repair_gpu_seconds"], 3);
        budget.begin(identity, 2, "a", 100);
        clock += 11;
        fails([&] { budget.check(); }, "phase overspend ignored");
        const auto overrun = budget.finish("timed_out", true);
        near(overrun["gpu_seconds"], 22);
        near(budget.snapshot()["used_regular_gpu_seconds"], 42);
        fails([&] { budget.begin(identity, 1, "a", 100); }, "exhausted phase was reset");
    }
    {
        Budget reopened(path, limits(), monotonic, realtime);
        near(reopened.snapshot()["used_gpu_seconds"], 45);
        reopened.begin(identity, 0, "b", 5);
        clock += 5;
        fails([&] { reopened.check(); }, "CPU worker escaped watchdog");
        near(reopened.finish("timed_out", true)["gpu_seconds"], 0);
    }
    auto changed = limits();
    changed.total_gpu_seconds = 101;
    fails([&] { Budget reset(path, changed, monotonic, realtime); }, "immutable allocation changed");
    {
        Budget forward(root / "forward.json", limits("forward"), monotonic, realtime);
        forward.begin(identity, 1, "a", 200);
        clock += 10;
        forward.finish("completed", true);
        forward.begin(identity, 1, "b", 200);
        forward.check(60); // 40 seconds for b plus 30 unused seconds from a.
        clock += 60;
        forward.finish("completed", true);
        fails([&] { forward.begin(identity, 1, "a", 200); }, "forward budget returned to an earlier phase");
    }
    const auto unsettled = root / "unsettled.json";
    {
        Budget active(unsettled, {}, monotonic, realtime);
        active.begin(identity, 1, "run", 5);
        clock -= 1;
        fails([&] { active.finish("failed", true); }, "reversed clock settled budget");
        clock += 1;
    }
    fails([&] { Budget active(unsettled, {}, monotonic, realtime); }, "unsettled session was reset");
    {
        BudgetLimits unlimited;
        unlimited.repair_reserve_gpu_seconds = 2;
        Budget budget(root / "unlimited.json", unlimited, monotonic, realtime);
        const auto failed = budget.begin(identity, 1, "run", 1e7);
        budget.check(1e6);
        clock += 1000;
        budget.finish("failed", true);
        Json retry = identity;
        retry["retry_of"] = failed["session_id"];
        budget.begin(retry, 1, "run", 100, true);
        fails([&] { budget.check(3); }, "unlimited regular pool leaked into repair reserve");
        clock += 3;
        budget.finish("failed", true);
        fails([&] { budget.begin(retry, 1, "run", 100, true); }, "overspent repair pool was reset");
    }
}
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--child") return child(argc, argv);
    char directory[] = "/tmp/ditserve-supervision-XXXXXX";
    if (!mkdtemp(directory)) return 1;
    try {
        processes(directory);
        budgets(directory);
        std::filesystem::remove_all(directory);
        std::cout << "supervision contracts passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << " (artifacts: " << directory << ")\n";
        return 1;
    }
}
