#pragma once

#include <iostream>
#include <functional>

namespace {
    template<typename T, class Compare = std::less<T>>
    inline void cas (T* data, size_t i, size_t j) {
        Compare elem_cmp;
        if (elem_cmp(data[j], data[i])) std::swap(data[i], data[j]);
    }

    template<typename T, class Compare = std::less<T>>
    void bitonic_merge (size_t size, T* data) {
        if (size < 2) return;

        int mid = size / 2;
        for (int i = 0; i < mid; i++) cas<T, Compare>(data, i, i + mid);

        bitonic_merge<T, Compare>(mid, data);
        bitonic_merge<T, Compare>(mid, data + mid);
    }
}

namespace bitonic_cpu {
    template<typename T, class Compare = std::less<T>>
    void bitonic_sort (size_t size, T* data) {
        if (size < 2) return;

        int mid = size / 2;
        bitonic_sort<T, std::less<T>>(mid, data);
        bitonic_sort<T, std::greater<T>>(mid, data + mid);

        bitonic_merge<T, Compare>(size, data);
    }
}
