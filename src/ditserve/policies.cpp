#include "ditserve/policies.hpp"

namespace ditserve {
void register_builtin_policies(PolicyRegistry& registry) {
    registry.add("cache_only", [] { return std::make_unique<CacheOnlyPolicy>(); });
    registry.add("batch_only", [] { return std::make_unique<BatchOnlyPolicy>(); });
    registry.add("naive", [] { return std::make_unique<NaivePolicy>(); });
    registry.add("edf", [] { return std::make_unique<EdfPolicy>(); });
    registry.add("path_aware", [] { return std::make_unique<PathAwarePolicy>(); });
    registry.add("state_aware", [] { return std::make_unique<StateAwarePolicy>(); });
}
}
