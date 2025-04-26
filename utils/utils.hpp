#pragma once

#include "config.hpp"
#include "opencl.hpp"

#include <iostream>
#include <random>
#include <string>
#include <sstream>
#include <fstream>

namespace utils {
    template <typename Iter>
    void rand_init (Iter start, Iter end, TYPE low, TYPE up) {
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
};
