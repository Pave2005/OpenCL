#include "ocl.hpp"

#include <vector>
#include <thread>
#include <iostream>
#include <numeric>
#include <functional>

constexpr int THREAD_COUNT = 4;

namespace multithreading {
    namespace details {
        template <typename T>
        void process_chunk(T* data, size_t offset, size_t size, const std::string& kernel_file) {
            ocl::Ocl<T> ocl(kernel_file);
            ocl.writeToBuffer(data + offset, size);
            ocl.run();
            ocl.readFromBuffer(data + offset);
        }

        template <typename T>
        void merge_chunks(std::vector<T>& data, size_t chunk_size, size_t total_size) {
            std::vector<T> temp(data.size());

            size_t num_chunks = total_size / chunk_size;
            while (num_chunks > 1) {
                size_t new_chunk_size = chunk_size * 2;

                for (size_t i = 0; i + 1 < num_chunks; i += 2) {
                    size_t left = i * chunk_size;
                    size_t mid = left + chunk_size;
                    size_t right = std::min(mid + chunk_size, total_size);

                    std::merge(data.begin() + left, data.begin() + mid,
                               data.begin() + mid, data.begin() + right,
                               temp.begin() + left);
                }

                if (num_chunks % 2 != 0) {
                    size_t last = (num_chunks - 1) * chunk_size;
                    std::copy(data.begin() + last, data.end(), temp.begin() + last);
                }

                std::swap(data, temp);
                chunk_size *= 2;
                num_chunks = (num_chunks + 1) / 2;
            }
        }
    }

    template <typename T>
    void sort (T* data, size_t total_size, const std::string& kernel_file) {
        std::vector<std::thread> threads;

        size_t chunk_size = total_size / THREAD_COUNT;
        if (chunk_size % 2 != 0) chunk_size--;

        for (int i = 0; i < THREAD_COUNT; ++i) {
            size_t offset = i * chunk_size;
            size_t size = (i == THREAD_COUNT - 1) ? (total_size - offset) : chunk_size;

            if (size % 2 != 0) size--;

            threads.emplace_back([=] {
                details::process_chunk<T>(data, offset, size, kernel_file);
            });
        }

        for (auto&& t : threads) t.join();

        std::vector<T> data_vec(data, data + total_size);
        details::merge_chunks<T>(data_vec, chunk_size, total_size);
        std::copy(data_vec.begin(), data_vec.end(), data);
    }
}

