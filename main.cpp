#include "ocl.hpp"
#include "utils.hpp"
#include "config.hpp"

#include <vector>
#include <iostream>
#include <string>
#include <numeric>
#include <chrono>
#include <limits>
#include <bit>

#include <cmath>

constexpr size_t data_size = (1 << 26);
// constexpr size_t data_size = std::bit_floor<size_t>(std::numeric_limits<int>::max());


int main (int argc, char* argv[]) {
    ocl::Ocl app(argv[1]);
    size_t size = data_size;
    if (argc > 2) size = std::atoi(argv[2]);

    std::vector<cl_int> cl_vector(size);

    utils::rand_init(cl_vector.rbegin(), cl_vector.rend(), -100000, 100000);

    std::cout << "Data size: " << size << " elements\n";

    auto StartTime = std::chrono::high_resolution_clock::now();
    app.writeToBuffer(cl_vector.data(), size);
    uint64_t ev_time = app.run();
    app.readFromBuffer(cl_vector.data());
    auto EndTime = std::chrono::high_resolution_clock::now();

    auto Dur = std::chrono::duration_cast<std::chrono::nanoseconds>(EndTime - StartTime).count();
    std::cout << "GPU wall time measured\t(" << size << "): \t" << Dur << " ns" << std::endl;
    std::cout << "GPU pure time measured\t(" << size << "): \t" << ev_time << " ns" << std::endl;

    cl::vector<TYPE> cpu_vector(size);
    utils::rand_init(cpu_vector.begin(), cpu_vector.end(), -100000, 100000);

    StartTime = std::chrono::high_resolution_clock::now();
    std::sort(cpu_vector.begin(), cpu_vector.end());
    EndTime = std::chrono::high_resolution_clock::now();

    Dur = std::chrono::duration_cast<std::chrono::nanoseconds>(EndTime - StartTime).count();
    std::cout << "CPU time measured with\t(" << size << "): \t" << Dur << " ns" << std::endl;

    if (std::is_sorted(cl_vector.begin(), cl_vector.end())) std::cout << "sorted correctly\n";
    else                                                    std::cout << "sorted incorrectly\n";
}
