#pragma once

#include <string>
#include <type_traits>

namespace details {
    template <typename T>
    constexpr std::string get_opencl_type_name() {
        if constexpr (std::is_same_v<T, char>) {
            return "char";
        } else if constexpr (std::is_same_v<T, int>) {
            return "int";
        } else if constexpr (std::is_same_v<T, float>) {
            return "float";
        } else if constexpr (std::is_same_v<T, double>) {
            return "double";
        } else {
            static_assert(!sizeof(T), "unsupported c++ type for OpenCL");
        }
    }
}
