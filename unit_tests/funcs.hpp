#pragma once

#include "ocl.hpp"

#include <gtest/gtest.h>

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>

namespace test_funcs
{
    void get_right_answer (std::vector<int>& v) { std::sort(v.begin(), v.end()); }

	void check_test (ocl::Ocl& app, const std::filesystem::path& filepath)
    {
        std::cout << "Running: " << filepath.string() << "\n";
        std::vector<int> array_gpu;
        std::ifstream in{filepath};

        size_t size;
        in >> size;

        std::istream_iterator<int> inp(in), end;

        std::copy(inp, end, std::back_inserter(array_gpu));

        std::vector<int> array_cpu = array_gpu;

        app.writeToBuffer(array_gpu.data(), size);
        app.run();
        app.readFromBuffer(array_gpu.data());
        get_right_answer(array_cpu);

        int n = array_gpu.size();
        for (int i = 0; i < n; i++) EXPECT_EQ(array_cpu[i], array_gpu[i]);
    }

	void run_test (const std::string& test_name)
	{
        std::string kernel = "kernels/bitonic_localmem.cl";
        std::string kernel_path = std::string(TEST_DATA_DIR) + kernel;

		ocl::Ocl app{kernel_path};
        std::string test_directory = "/tests";
        std::string test_path = std::string(TEST_DATA_DIR) + test_directory + test_name;

        check_test(app, test_path);
	}
}
