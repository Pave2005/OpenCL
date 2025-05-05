#pragma once

#include <iostream>

#define ASCENDING  1
#define DESCENDING 0

namespace bitonic_cpu {
    template<bool order = 1>
    inline void cas (int* data, size_t i, size_t j) {
        if (order == (data[i] > data[j])) std::swap(data[i], data[j]);
    }

    template<bool order = 1>
    void bitonic_merge (size_t size, int* data) {
        if (size < 2) return;

        int mid = size / 2;
        for (int i = 0; i < mid; i++) cas<order>(data, i, i + mid);

        bitonic_merge<order>(mid, data);
        bitonic_merge<order>(mid, data + mid);
    }

    template<bool order = 1>
    void bitonic_sort (size_t size, int* data) {
        if (size < 2) return;

        int mid = size / 2;
        bitonic_sort<ASCENDING>(mid, data);
        bitonic_sort<DESCENDING>(mid, data + mid);

        bitonic_merge<order>(size, data);
    }
}
