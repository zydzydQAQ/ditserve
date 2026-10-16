#pragma once

#include "ditserve/types.hpp"

namespace ditserve {
// Shared by the public CLI and the explicit JSON RPC interface.
Json normalize_request(const std::string& command, Json request);
int command_main(int argc, char** argv);
}
