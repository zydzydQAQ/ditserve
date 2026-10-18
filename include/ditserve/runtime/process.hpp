#pragma once

#include "ditserve/types.hpp"
#include <filesystem>
#include <functional>
#include <map>

namespace ditserve {
double monotonic_seconds();

// Captures SIGINT/SIGTERM for one command and restores prior handlers on exit.
class CancelScope {
public:
    CancelScope();
    ~CancelScope();
    CancelScope(const CancelScope&) = delete;
    CancelScope& operator=(const CancelScope&) = delete;
    static bool requested();
private:
    void (*interrupt_)(int);
    void (*terminate_)(int);
};

struct ProcessOptions {
    std::vector<std::string> argv;
    std::map<std::string, std::string> environment;
    std::filesystem::path stderr_path;
    std::size_t max_message_bytes = 4 * 1024 * 1024;
    double cleanup_timeout_seconds = 5;
    std::function<void()> heartbeat = {}; // Bounded checks; exceptions terminate the worker group.
};

// One persistent, independently supervised process group. Never invokes a shell.
class WorkerProcess {
public:
    explicit WorkerProcess(const ProcessOptions& options);
    ~WorkerProcess();
    WorkerProcess(const WorkerProcess&) = delete;
    WorkerProcess& operator=(const WorkerProcess&) = delete;
    Json call(const Json& request, double absolute_monotonic_deadline);
    bool stop() noexcept;
    int pid() const { return pid_; }
    bool cleanup_verified() const { return cleanup_verified_; }
private:
    int pid_ = -1, input_ = -1, output_ = -1;
    bool reaped_ = false, cleanup_verified_ = false;
    std::size_t max_message_bytes_;
    double cleanup_timeout_seconds_;
    std::function<void()> heartbeat_;
    std::string buffered_;
    void reap();
};
}
