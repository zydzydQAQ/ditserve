#pragma once

#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace ditserve {
using Json = nlohmann::json;

double number(const Json& value, const std::string& name, bool positive = false);
std::int64_t integer(const Json& value, const std::string& name);
double number_field(const Json& object, const char* key, double fallback = 0, bool positive = false);
int int_field(const Json& object, const char* key, int fallback, int minimum = 1);
bool bool_field(const Json& object, const char* key, bool fallback);
std::string string_field(const Json& object, const char* key, const std::string& fallback);
void check_keys(const Json& object, std::initializer_list<const char*> allowed);
// One portable file-name component: [A-Za-z0-9][A-Za-z0-9_.-]{0,119}.
std::string path_segment(const std::string& value, const std::string& name);
Json core_version();
}
