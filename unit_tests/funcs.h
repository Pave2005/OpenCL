#pragma once

#include "ocl.h"

#include <gtest/gtest.h>

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <cmath>

namespace test_funcs
{
	void get_result (const std::string& filename, std::vector<int>& res, const std::string& kernelpath)
    {
        std::ifstream file(filename);
        if (!file) throw std::runtime_error("file error");

        std::size_t data_size = 0;
        file >> data_size;

        std::vector<int> data = {};
        for (int i = 0; i < data_size; ++i)
        {
            int elem = 0;
            file >> elem;

            data.push_back(elem);
        }

        std::size_t new_size = std::bit_ceil(data_size);
        data.resize(new_size, std::numeric_limits<int>::max());

        OpenCL::OclApp<int> app {kernelpath, "bitonicSort", data};

        if (app.kernel().setArg(0, app.buffer()) != CL_SUCCESS)
        {
            std::cerr << "setArg(0): kernel error" << std::endl;
            exit(1);
        }

        for (int stage = 2; stage <= new_size; stage *= 2)
        {
            for (int step = stage / 2; step > 0; step /= 2)
            {
                app.kernel().setArg(1, stage);
                app.kernel().setArg(2, step);

                if (app.queue().enqueueNDRangeKernel(app.kernel(), cl::NullRange,  cl::NDRange(new_size), cl::NullRange) != CL_SUCCESS)
                {
                    std::cerr << "enqueueNDRangeKernel: kernel error" << std::endl;
                    exit(1);
                }

                app.queue().finish();
            }
        }
        // res.resize(new_size, 0); разобраться
        res.resize(data_size);

        if (app.queue().enqueueReadBuffer(app.buffer(), CL_TRUE, 0, sizeof(int) * new_size, res.data()) != CL_SUCCESS)
        {
            std::cerr << "enqueueReadBuffer: kernel error" << std::endl;
            exit(1);
        }
    }

    void get_answer (const std::string& filename, std::vector<int>& ans)
    {
        std::ifstream file(filename);
        if (!file) throw std::runtime_error("file error");

        int data_size = 0;
        file >> data_size;

        for (int i = 0; i < data_size; ++i)
        {
            int elem = 0;
            file >> elem;

            ans.push_back(elem);
        }

        std::sort(ans.begin(), ans.end());
    }

	void run_test (const std::string& test_name)
	{
		std::string test_directory = "/tests";
		std::string test_path = std::string(TEST_DATA_DIR) + test_directory + test_name;
        std::string kernel = "/kernels/bitonic_sort.cl";
        std::string kernel_path = std::string(TEST_DATA_DIR) + kernel;

        std::vector<int> res;
		get_result(test_path, res, kernel_path);

        std::vector<int> ans;
		get_answer(test_path, ans);

        for (int i = 0; i < ans.size(); i++) EXPECT_EQ(res[i], ans[i]);
	}
}
