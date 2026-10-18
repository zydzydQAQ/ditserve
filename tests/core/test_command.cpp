#include "ditserve/control/study.hpp"
#include "ditserve/interface/command.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>

using namespace ditserve;
namespace fs = std::filesystem;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    const auto root = fs::temp_directory_path() / ("ditserve-command-test-" + std::to_string(::getpid()));
    try {
        fs::create_directory(root);
        Json context = {{"root", root.string()}, {"package_root", (root / "sources").string()}, {"python", "chosen-python"}};
        auto run = context;
        run["config"] = {{"experiment", "fixture"}};
        auto first = normalize_request("run", run);
        const auto id = first.at("run_id").get<std::string>();
        require(id.size() == 12, "automatic run identity");
        require(first.at("python") == "chosen-python", "explicit interpreter changed");
        require(first.at("work_dir") == (root / ".cache/results-work/fixture" / id).string(), "work default");
        require(first.at("results_dir") == (root / "results/fixture" / id).string(), "results default");
        require(first.at("ledger") == (root / ".state/fixture/budget.json").string(), "ledger default");
        require(first.at("prune") == true, "archive cleanup default");
        require(!fs::exists(root / ".state"), "normalization created a ledger");

        run["resume"] = true;
        run["work_dir"] = (root / "existing-run").string() + "/";
        run["results_root"] = (root / "published").string();
        const auto resumed = normalize_request("run", run);
        require(resumed.at("run_id") == "existing-run", "resume did not infer identity from work directory");
        require(resumed.at("results_dir") == (root / "published/fixture/existing-run").string(), "custom results root");
        run["results_dir"] = (root / "legacy-output").string();
        run["ledger"] = (root / "explicit-budget.json").string();
        require(normalize_request("run", run).at("results_dir") == run.at("results_dir"), "complete RPC path overwritten");
        require(normalize_request("run", run).at("ledger") == run.at("ledger"), "complete RPC ledger overwritten");

        auto simulation = context;
        simulation["count"] = 2; simulation["seed"] = 11; simulation["keep_raw"] = true;
        const auto synthetic = normalize_request("simulate-study", simulation);
        require(synthetic.at("run_id").get<std::string>().rfind("simulate-", 0) == 0, "simulation identity prefix");
        require(synthetic.at("config").at("generator") == Json({{"count", 2}, {"seed", 11}}), "API generator options");
        require(synthetic.at("prune") == false, "keep_raw ignored");
        const auto config_path = root / "configuration.json";
        { std::ofstream config(config_path); config << "{\"experiment\":\"from-file\"}"; }
        auto file = context; file["config"] = config_path.string();
        require(normalize_request("run", file).at("config").at("experiment") == "from-file", "config path was not loaded");

        for (const auto& extra : {Json{{"resume", true}}, Json{{"repair", true}}, Json{{"retry_of", "missing"}},
                                 Json{{"run_id", "../escape"}}, Json{{"run_id", ".hidden"}}, Json{{"run_id", "-bad"}},
                                 Json{{"run_id", "_bad"}}, Json{{"run_id", std::string(121, 'a')}}}) {
            auto invalid = context; invalid["config"] = {{"experiment", "fixture"}}; invalid.update(extra);
            bool rejected = false;
            try { normalize_request("run", invalid); } catch (const std::exception&) { rejected = true; }
            require(rejected, "invalid admission options accepted");
        }

        // Heartbeat isolation: unanswered probes are tolerated twice in a row, others are not.
        const Json exclusive = {{"status", "exclusive_observed"}, {"owned", Json::array()}, {"foreign", Json::array()}};
        const Json unknown = {{"status", "unknown"}, {"reason", "worker action timed out"}};
        const Json busy = {{"status", "busy"}, {"foreign", Json::array({Json{{"pid", 7}}})}};
        int streak = 0;
        require(isolation_holds(exclusive, streak) && streak == 0, "exclusive observation rejected");
        require(isolation_holds(unknown, streak) && isolation_holds(unknown, streak), "slow probe ended the run");
        require(!isolation_holds(unknown, streak), "three unanswered probes tolerated");
        streak = 0;
        require(isolation_holds(unknown, streak) && isolation_holds(exclusive, streak) && streak == 0,
                "an exclusive observation did not reset the streak");
        require(!isolation_holds(busy, streak), "foreign process tolerated");
        require(!isolation_holds(Json{{"status", "exclusive_observed"}, {"foreign", Json::array({1})}}, streak),
                "foreign rows tolerated");
        fs::remove_all(root);
        std::cout << "native request normalization passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::error_code ignored; fs::remove_all(root, ignored);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
