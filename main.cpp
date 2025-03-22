#include "bitonic_sort.h"

#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <limits>
#include <bit>

int main() try
{
    std::size_t data_size = 0;

    std::cin >> data_size;
    if (data_size <= 0) throw std::runtime_error("invalid size error");

    std::vector<int> data = {};
    data.reserve(data_size);

    for (int i = 0; i < data_size; ++i)
    {
        int elem = 0;

        std::cin >> elem;
        if (!std::cin.good()) throw std::runtime_error("invalid argument error");

        data.push_back(elem);
    }

    std::size_t new_size = std::bit_ceil(data_size);
    data.resize(new_size, std::numeric_limits<int>::max());

    OpenCL::OclApp<int> app {"kernels/bitonic_sort.cl", "bitonicSort", data};
    bitonic::BitonicSort<int> bsrt = {app, data};

    bsrt.sort(data);

    data.resize(data_size);

    bitonic::utils::dump(data);

}
catch(const std::exception& exception )
{
    std::cerr << exception.what() << std::endl;
    return 1;
}
catch (...)
{
    std::cerr << "Caught unknown exception" << std::endl;
    return 1;
}
