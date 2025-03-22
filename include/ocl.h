#pragma once

#ifndef CL_HPP_TARGET_OPENCL_VERSION
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION  120
#endif

#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_ENABLE_EXCEPTIONS

#include <opencl.hpp>

#include <iostream>
#include <fstream>

namespace OpenCL
{
    struct OclWrapper final
    {
        cl::Platform  platform;
        cl::Device      device;
        cl::Context    context;
        cl::CommandQueue queue;

        OclWrapper (cl::Platform platform_ = cl::Platform::getDefault(), cl::Device device_ = cl::Device::getDefault()) :
                    platform (platform_), device (device_), context ({ device }), queue(context, device, CL_QUEUE_PROFILING_ENABLE) {}
    };

    template <typename T>
    class OclApp final
    {
        OclWrapper platform_;
        cl::Program program_;
        cl::Buffer   buffer_;
        cl::Kernel   kernel_;

        std::string read_kernel_code (const std::string& filename)
        {
            std::ifstream file(filename);
            return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        }

    public:
        OclApp (const std::string& kernel_code, const char* kernel_name, std::vector<T>& data,
                const OclWrapper& platform = OclWrapper()) : platform_(platform), program_(platform_.context, read_kernel_code(kernel_code))
                //throw
        {
            program_.build({platform_.device});
            buffer_  = cl::Buffer(platform_.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(T) * data.size(), data.data());
            platform_.queue.enqueueWriteBuffer(buffer_, CL_TRUE, 0, sizeof(T) * data.size(), data.data());
            kernel_  = cl::Kernel(program_, kernel_name);
        }

        auto read_data (std::vector<T>& data)
        {
            return platform_.queue.enqueueReadBuffer(buffer_, CL_TRUE, 0, sizeof(T) * data.size(), data.data());
        }

        auto queue    () const { return platform_.queue; }
        auto context  () const { return platform_.context; }
        auto device   () const { return platform_.device; }
        auto platform () const { return platform_.platform; }
        auto program  () const { return program_; }
        auto buffer   () const { return buffer_; }
        auto kernel   () const { return kernel_; }
    };
}
