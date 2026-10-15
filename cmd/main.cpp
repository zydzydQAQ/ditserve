#include "ditserve/runtime.hpp"
#include "ditserve/workload.hpp"
#include <iostream>
#include <iterator>
#include <stdexcept>

int main(int argc, char** argv) {
    using namespace ditserve;
    try {
        if (argc != 2 || (std::string(argv[1]) != "generate" && std::string(argv[1]) != "simulate"))
            throw std::invalid_argument("usage: ditserve-core generate|simulate < request.json");
        const std::string input((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
        const Json request = Json::parse(input);
        if (!request.is_object() || integer(request.at("schema_version"), "schema_version") != 2)
            throw std::invalid_argument("schema_version must be 2");
        const CostModel cost = CostModel::from_json(request.value("cost_model", Json::object()));
        Json result;
        if (std::string(argv[1]) == "generate") {
            check_keys(request, {"schema_version", "generator", "cost_model"});
            result = Json::array();
            for (const auto& spec : generate_workload(request.value("generator", Json::object()), cost)) result.push_back(spec.to_json());
        } else {
            check_keys(request, {"schema_version", "config", "cost_model", "workload"});
            result = Simulator(RuntimeConfig::from_json(request.value("config", Json::object())), cost).run(read_workload(request.at("workload")));
        }
        std::cout << result.dump() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ditserve-core: " << error.what() << '\n';
        return 2;
    }
}
