#pragma once

#include "config.hpp"
#include "opencl.hpp"

#include <iostream>
#include <random>
#include <string>
#include <sstream>
#include <fstream>
#include <chrono>
#include <vector>
#include <cmath>
#include <cassert>

namespace utils {
//     template<typename T>
//     void bit_reversal_permutation(T* data, size_t size) {
//         assert((size & (size - 1)) == 0);
//
//         size_t log_n = static_cast<size_t>(std::log2(size));
//
//         auto bit_reverse = [log_n](size_t i) {
//             size_t res = 0;
//             for (size_t j = 0; j < log_n; ++j) {
//                 if ((i >> j) & 1)
//                     res |= 1 << (log_n - 1 - j);
//             }
//             return res;
//         };
//
//         for (size_t i = 0; i < size; ++i) {
//             size_t rev = bit_reverse(i);
//             if (rev > i) std::swap(data[i], data[rev]);
//         }
//     }

    template <typename Iter>
    void rand_init (Iter start, Iter end, int low, int up) {
        static std::mt19937_64 mt_source;
        std::uniform_int_distribution<int> dist(low, up);
        for (Iter cur = start; cur != end; ++cur) *cur = dist(mt_source);
    }


    std::ostream& operator<< (std::ostream& out, std::vector<cl_int>& x) {
        for (auto i: x) out << i << " ";

        return out;
    }

    std::string readFile (const std::string& filename) {
        std::ifstream input(filename);
        std::stringstream stream;
        stream << input.rdbuf();
        return stream.str();
    }
}; // namespace utils

