#pragma once

#include <format>
#include <iostream>

namespace bitonic
{
    namespace utils
    {
        template<typename T>
        void dump(const std::vector<T>& data)
        {
            std::cout << "\n";

            for (const auto& elem : data) std::cout << elem << " ";

            std::cout << "\n\n";
        }
    };
};
