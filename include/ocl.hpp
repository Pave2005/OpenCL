#pragma once

#ifndef CL_HPP_TARGET_OPENCL_VERSION
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION  120
#endif

#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_ENABLE_EXCEPTIONS

#include "config.hpp"
#include "utils.hpp"

#include <opencl.hpp>
#include <cassert>
#include <iostream>


#define STRINGIFY(X) #X
#define STYPE(X) STRINGIFY(X)

#define dbgs        \
    if(!LOGGING){   \
    }               \
    else std::cout


namespace ocl {
    namespace details {
        class DeviceInfo {
        private:
            cl::size_type max_lcl_mem_size_;
            cl::size_type lcl_size_;
            cl::size_type work_group_size_;
            cl::size_type glb_size_;

        public:
            DeviceInfo (const cl::Device& dev) {
                work_group_size_  = dev.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
                glb_size_ = dev.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();

                cl_int max_lcl_mem = dev.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
                max_lcl_mem_size_ = max_lcl_mem / sizeof(TYPE);

                lcl_size_ = std::bit_floor<size_t>(max_lcl_mem / sizeof(TYPE));
                if (max_lcl_mem_size_ - lcl_size_ < 1024) lcl_size_ <<= 1;

                dbgs << "Maximum work-group size: " << work_group_size_ << std::endl;
                dbgs << "Maximum global size: " << glb_size_ << std::endl;

                dbgs << "Maximum local memory size: " << max_lcl_mem_size_ << std::endl;
                dbgs << "Modified local memory size: " << lcl_size_ << std::endl;
            }

            cl::size_type get_local_mem_size () { return lcl_size_; }

            cl::size_type get_work_group_size (int elems_num) {
                if (elems_num < work_group_size_) return elems_num;

                return work_group_size_;
            }
        };
    }

    class Ocl {
    private:
        cl::Platform     platform_;
        cl::Device       device_;
        cl::Context      context_;
        cl::CommandQueue queue_;
        cl::Program      program_;
        cl::Kernel       fast_kernel_;
        cl::Kernel       slow_kernel_;
        cl::Buffer       buffer_;
        cl::size_type    elems_num_;

        details::DeviceInfo info_;

        static cl::Platform get_platform () {
            std::vector<cl::Platform> platforms;
            cl::Platform::get(&platforms);
            cl_uint gpu_id = 0;

            for (auto plat: platforms) {
                ::clGetDeviceIDs(plat(), CL_DEVICE_TYPE_GPU, 0, NULL, &gpu_id);
                if (gpu_id > 0) {
                    dbgs << plat.getInfo<CL_PLATFORM_NAME>() << std::endl;
                    return cl::Platform(plat());
                }
            }

            dbgs << "No GPU platform found, switching to a CPU one\n";
            cl_uint cpu_id = 0;
            for (auto plat: platforms) {
                ::clGetDeviceIDs(plat(), CL_DEVICE_TYPE_ALL, 0, NULL, &cpu_id);
                if (cpu_id > 0) {
                    dbgs << "Chosen platform: " << plat.getInfo<CL_PLATFORM_NAME>() << "\n";
                    return cl::Platform(plat());
                }
            }
            dbgs << "No CPU platform found, switching to a default one (on what are you running?)\n";
            return cl::Platform::getDefault();
        }

        static cl::Device get_device (const cl::Platform& platform) {
            std::vector<cl::Device> devices;

            platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

            if (devices.empty()) {
                dbgs << "No GPU device found, switching to a CPU one\n";
                platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
                if (devices.empty()) {
                    dbgs << "No CPU device found, switching to a default one (on what are you running?)\n";
                    return cl::Device::getDefault();
                }
            }

            dbgs << "Chosen device: " << devices[0].getInfo<CL_DEVICE_NAME>() << "\n";
            return devices[0];
        }

        static cl::Context create_context (const cl::Platform& platform) {
            cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, reinterpret_cast<intptr_t>(platform()), 0};
            return cl::Context(CL_DEVICE_TYPE_GPU, properties);
        }

    public:
        Ocl (const std::string& file_name): platform_(get_platform()), device_(get_device(platform_)), context_(device_),
                                        queue_(context_, device_, cl::QueueProperties::Profiling), info_(device_) {

            std::string code = std::string("#define LCL_SZ ") + std::to_string(device_.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()) + "\n"
                            + std::string("#define TYPE ") + STYPE(TYPE) + "\n"
                            + utils::readFile(file_name);

            program_ = cl::Program{context_, code};
            program_.build();
            fast_kernel_ = cl::Kernel(program_, "bitonic_fast");
            slow_kernel_ = cl::Kernel(program_, "bitonic_slow");
        }

        uint64_t run () {
            cl::size_type buf_size = elems_num_ * sizeof(TYPE);

            fast_kernel_.setArg(0, buffer_);
            slow_kernel_.setArg(0, buffer_);

            int work_group_size = device_.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();

            if (elems_num_ < work_group_size) work_group_size = elems_num_;

            cl::NDRange global_size(elems_num_);
            cl::NDRange local_size(work_group_size);

            auto start_cycle_time = std::chrono::high_resolution_clock::now();

            for (int scale = 2; scale <= elems_num_; scale <<= 1) {

                if (scale <= work_group_size) {
                    fast_kernel_.setArg(1, scale / 2);
                    fast_kernel_.setArg(2, scale);
                    int err = queue_.enqueueNDRangeKernel(fast_kernel_, cl::NullRange, global_size, local_size);
                }
                else {
                    for (int j = scale / 2; j > 0; j >>= 1) {
                        if (j <= work_group_size / 2) {
                            fast_kernel_.setArg(1, j);
                            fast_kernel_.setArg(2, scale);
                            queue_.enqueueNDRangeKernel(fast_kernel_, cl::NullRange, global_size, local_size);
                            break;
                        }
                        else {
                            slow_kernel_.setArg(1, j);
                            slow_kernel_.setArg(2, scale);
                            queue_.enqueueNDRangeKernel(slow_kernel_, cl::NullRange, global_size, local_size);
                        }
                    }
                }
            }

            auto end_cycle_time = std::chrono::high_resolution_clock::now();
            auto cycle_dur = std::chrono::duration_cast<std::chrono::nanoseconds>(end_cycle_time - start_cycle_time).count();
            return cycle_dur;
        }

        void writeToBuffer (TYPE* input, size_t size) {
            assert(size % 2 == 0);

            auto StartWriteTime = std::chrono::high_resolution_clock::now();

            elems_num_ = size;
            cl::size_type buf_size = size * sizeof(TYPE);
            buffer_ = cl::Buffer(context_, CL_MEM_READ_WRITE, buf_size);
            queue_.enqueueWriteBuffer(buffer_, CL_TRUE, 0, buf_size, input);

            auto EndWriteTime = std::chrono::high_resolution_clock::now();
            auto WriteDur = std::chrono::duration_cast<std::chrono::nanoseconds>(EndWriteTime - StartWriteTime).count();
            dbgs << "Write buffer in: " << WriteDur << "\n";
        }

        void readFromBuffer (TYPE* output) const {
            auto ReadStartTime = std::chrono::high_resolution_clock::now();

            cl::size_type buf_size = elems_num_ * sizeof(TYPE);
            queue_.enqueueReadBuffer(buffer_, CL_TRUE, 0, buf_size, output);
            queue_.finish();

            auto ReadEndTime = std::chrono::high_resolution_clock::now();
            auto DurRead = std::chrono::duration_cast<std::chrono::nanoseconds>(ReadEndTime - ReadStartTime).count();
            dbgs << "Read buffer in: " << DurRead << "\n";
        }
    }; // Ocl
}
