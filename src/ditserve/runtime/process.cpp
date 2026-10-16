#include "ditserve/runtime/process.hpp"
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <stdexcept>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

extern char** environ;

namespace ditserve {
namespace {
void close_fd(int& fd) { if (fd >= 0) { ::close(fd); fd = -1; } }
std::runtime_error system_error(const std::string& operation, int code = errno) {
    return std::runtime_error(operation + ": " + std::strerror(code));
}
void nonblocking(int fd) {
    const int flags = fcntl(fd, F_GETFL);
    if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) throw system_error("nonblocking pipe");
}
bool group_gone(int pid) { return kill(-pid, 0) < 0 && errno == ESRCH; }

// A closed worker pipe must produce EPIPE, not terminate the supervisor.
struct BlockSigpipe {
    sigset_t mask{}, previous{};
    bool pending = false;
    BlockSigpipe() {
        sigemptyset(&mask); sigaddset(&mask, SIGPIPE);
        if (sigprocmask(SIG_BLOCK, &mask, &previous) < 0) throw system_error("block SIGPIPE");
        sigset_t signals{};
        sigpending(&signals);
        pending = sigismember(&signals, SIGPIPE) == 1;
    }
    ~BlockSigpipe() {
        if (!pending) { const timespec zero{}; while (sigtimedwait(&mask, nullptr, &zero) < 0 && errno == EINTR) {} }
        sigprocmask(SIG_SETMASK, &previous, nullptr);
    }
};
}

double monotonic_seconds() {
    timespec value{};
    if (clock_gettime(CLOCK_MONOTONIC, &value) < 0) throw system_error("monotonic clock");
    return static_cast<double>(value.tv_sec) + static_cast<double>(value.tv_nsec) / 1e9;
}

WorkerProcess::WorkerProcess(const ProcessOptions& options)
    : max_message_bytes_(options.max_message_bytes), cleanup_timeout_seconds_(options.cleanup_timeout_seconds), heartbeat_(options.heartbeat) {
    if (options.argv.empty() || options.argv.front().empty() || options.stderr_path.empty() ||
        max_message_bytes_ == 0 || !std::isfinite(cleanup_timeout_seconds_) || cleanup_timeout_seconds_ <= 0)
        throw std::invalid_argument("worker requires argv, stderr path and positive message/cleanup limits");
    for (const auto& argument : options.argv)
        if (argument.find('\0') != std::string::npos) throw std::invalid_argument("NUL in worker argument");
    // Reap orphaned grandchildren after terminating an owned process group.
    if (prctl(PR_SET_CHILD_SUBREAPER, 1) < 0) throw system_error("enable child subreaper");
    std::map<std::string, std::string> environment;
    for (char** entry = environ; *entry; ++entry) {
        const std::string value(*entry);
        const auto equal = value.find('=');
        if (equal != std::string::npos) environment[value.substr(0, equal)] = value.substr(equal + 1);
    }
    for (const auto& item : options.environment) {
        if (item.first.empty() || item.first.find_first_of("=\0", 0, 2) != std::string::npos ||
            item.second.find('\0') != std::string::npos) throw std::invalid_argument("invalid worker environment");
        environment[item.first] = item.second;
    }
    std::vector<std::string> entries;
    for (const auto& item : environment) entries.push_back(item.first + "=" + item.second);
    std::vector<char*> argv, envp;
    for (const auto& argument : options.argv) argv.push_back(const_cast<char*>(argument.c_str()));
    for (auto& entry : entries) envp.push_back(entry.data());
    argv.push_back(nullptr); envp.push_back(nullptr);
    const auto directory = options.stderr_path.parent_path();
    if (!directory.empty()) std::filesystem::create_directories(directory);
    int incoming[2]{-1, -1}, outgoing[2]{-1, -1}, log = -1;
    posix_spawn_file_actions_t actions;
    posix_spawnattr_t attributes;
    bool actions_ready = false, attributes_ready = false;
    try {
        if (pipe2(incoming, O_CLOEXEC) < 0 || pipe2(outgoing, O_CLOEXEC) < 0) throw system_error("worker pipe");
        log = open(options.stderr_path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
        if (log < 0) throw system_error("worker stderr log");
        auto checked = [](int code) { if (code) throw system_error("configure worker spawn", code); };
        checked(posix_spawn_file_actions_init(&actions)); actions_ready = true;
        checked(posix_spawnattr_init(&attributes)); attributes_ready = true;
        checked(posix_spawn_file_actions_adddup2(&actions, incoming[0], STDIN_FILENO));
        checked(posix_spawn_file_actions_adddup2(&actions, outgoing[1], STDOUT_FILENO));
        checked(posix_spawn_file_actions_adddup2(&actions, log, STDERR_FILENO));
        for (const int fd : {incoming[0], incoming[1], outgoing[0], outgoing[1], log})
            if (fd > STDERR_FILENO) checked(posix_spawn_file_actions_addclose(&actions, fd));
        checked(posix_spawnattr_setpgroup(&attributes, 0));
        sigset_t empty{}, defaults{};
        sigemptyset(&empty); sigemptyset(&defaults);
        for (const int signal : {SIGPIPE, SIGTERM, SIGINT}) sigaddset(&defaults, signal);
        checked(posix_spawnattr_setsigmask(&attributes, &empty));
        checked(posix_spawnattr_setsigdefault(&attributes, &defaults));
        checked(posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETPGROUP | POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF));
        pid_t child = -1;
        checked(posix_spawnp(&child, argv.front(), &actions, &attributes, argv.data(), envp.data()));
        pid_ = child;
        close_fd(incoming[0]); close_fd(outgoing[1]); close_fd(log);
        input_ = incoming[1]; incoming[1] = -1;
        output_ = outgoing[0]; outgoing[0] = -1;
        nonblocking(input_); nonblocking(output_);
        posix_spawn_file_actions_destroy(&actions); actions_ready = false;
        posix_spawnattr_destroy(&attributes); attributes_ready = false;
    } catch (...) {
        for (int* fd : {&incoming[0], &incoming[1], &outgoing[0], &outgoing[1], &log}) close_fd(*fd);
        if (actions_ready) posix_spawn_file_actions_destroy(&actions);
        if (attributes_ready) posix_spawnattr_destroy(&attributes);
        stop();
        throw;
    }
}

WorkerProcess::~WorkerProcess() { stop(); }

void WorkerProcess::reap() {
    if (pid_ <= 0) return;
    int status;
    for (;;) {
        const auto child = waitpid(-pid_, &status, WNOHANG);
        if (child == pid_) reaped_ = true;
        if (child > 0) continue;
        if (child < 0 && errno == EINTR) continue;
        if (child < 0 && errno == ECHILD) reaped_ = true;
        break;
    }
}

bool WorkerProcess::stop() noexcept {
    close_fd(input_); close_fd(output_);
    if (pid_ <= 0 || cleanup_verified_) return true;
    try {
        const double begin = monotonic_seconds();
        const double deadline = begin + cleanup_timeout_seconds_;
        const double grace = begin + std::min(.25, cleanup_timeout_seconds_ / 2);
        kill(-pid_, SIGTERM);
        bool killed = false;
        do {
            reap();
            if (reaped_ && group_gone(pid_)) { cleanup_verified_ = true; return true; }
            const double now = monotonic_seconds();
            if (!killed && now >= grace) { kill(-pid_, SIGKILL); killed = true; }
            if (now >= deadline) break;
            poll(nullptr, 0, 5);
        } while (true);
        reap();
        cleanup_verified_ = reaped_ && group_gone(pid_);
    } catch (...) { cleanup_verified_ = false; }
    return cleanup_verified_;
}

Json WorkerProcess::call(const Json& request, double deadline) {
    try {
        if (input_ < 0 || output_ < 0 || reaped_) throw std::runtime_error("worker is closed");
        if (!std::isfinite(deadline)) throw std::invalid_argument("worker deadline must be finite");
        if (!request.is_object() || !request.contains("version") || !request["version"].is_number_integer() || request["version"] != Json(1) ||
            !request.contains("id") || !request["id"].is_number_integer() ||
            !request.contains("op") || !request["op"].is_string() || request["op"].get<std::string>().empty())
            throw std::invalid_argument("invalid worker request envelope");
        const std::string encoded = request.dump() + '\n';
        if (encoded.size() > max_message_bytes_) throw std::invalid_argument("worker request exceeds message limit");
        BlockSigpipe blocked;
        std::size_t sent = 0;
        double next_heartbeat = monotonic_seconds();
        for (;;) {
            double now = monotonic_seconds();
            if (now >= deadline) throw std::runtime_error("worker action timed out");
            if (heartbeat_ && now >= next_heartbeat) {
                heartbeat_();
                now = monotonic_seconds();
                next_heartbeat = now + .5;
            }
            const double remaining = deadline - now;
            if (remaining <= 0) throw std::runtime_error("worker action timed out");
            const auto newline = buffered_.find('\n');
            if (newline != std::string::npos) {
                if (sent != encoded.size()) throw std::runtime_error("unsolicited worker response");
                const auto line = buffered_.substr(0, newline);
                buffered_.erase(0, newline + 1);
                if (!buffered_.empty()) throw std::runtime_error("multiple worker responses for one request");
                const auto response = Json::parse(line);
                if (!response.is_object() || !response.contains("version") || !response["version"].is_number_integer() || response["version"] != Json(1) ||
                    !response.contains("id") || !response["id"].is_number_integer() || response["id"] != request["id"] ||
                    !response.contains("ok") || !response["ok"].is_boolean())
                    throw std::runtime_error("invalid worker response envelope");
                if (!response["ok"].get<bool>()) {
                    const auto& error = response.at("error");
                    if (!error.is_object() || !error.contains("type") || !error["type"].is_string() ||
                        !error.contains("message") || !error["message"].is_string())
                        throw std::runtime_error("invalid worker error envelope");
                    throw std::runtime_error("worker " + error["type"].get<std::string>() + ": " + error["message"].get<std::string>());
                }
                if (!response.contains("result") || !response["result"].is_object() || response.contains("error"))
                    throw std::runtime_error("invalid worker result envelope");
                return response["result"];
            }
            pollfd descriptors[2]{{output_, POLLIN, 0}, {input_, static_cast<short>(sent < encoded.size() ? POLLOUT : 0), 0}};
            const double wait = heartbeat_ ? std::min(remaining, std::max(0., next_heartbeat - monotonic_seconds())) : remaining;
            const int timeout = static_cast<int>(std::min(static_cast<double>(INT_MAX), std::ceil(wait * 1000)));
            const int ready = poll(descriptors, 2, timeout);
            if (ready < 0) { if (errno == EINTR) continue; throw system_error("poll worker"); }
            if (ready == 0) continue;
            if (descriptors[1].revents & POLLOUT) {
                const auto count = write(input_, encoded.data() + sent, encoded.size() - sent);
                if (count > 0) sent += static_cast<std::size_t>(count);
                else if (count < 0 && errno != EINTR && errno != EAGAIN) throw system_error("write worker request");
            }
            if (descriptors[0].revents & (POLLIN | POLLHUP)) {
                char buffer[8192];
                const auto count = read(output_, buffer, sizeof(buffer));
                if (count > 0) {
                    buffered_.append(buffer, static_cast<std::size_t>(count));
                    if (buffered_.size() > max_message_bytes_) throw std::runtime_error("worker response exceeds message limit");
                } else if (count == 0) throw std::runtime_error("worker exited before a complete response");
                else if (errno != EINTR && errno != EAGAIN) throw system_error("read worker response");
            }
            // A shutdown response remains valid when the child closes its pipes immediately afterwards.
            if (buffered_.find('\n') != std::string::npos) continue;
            if ((descriptors[0].revents & (POLLERR | POLLNVAL)) || (descriptors[1].revents & (POLLERR | POLLHUP | POLLNVAL)))
                throw std::runtime_error("worker pipe closed");
        }
    } catch (...) {
        stop();
        throw;
    }
}
}
