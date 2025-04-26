#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <fstream>

template <typename Iter>
void rand_init (Iter start, Iter end, int low, int up) {
    static std::mt19937_64 mt_source;
    std::uniform_int_distribution<int> dist(low, up);
    for (Iter cur = start; cur != end; ++cur) *cur = dist(mt_source);
}

int main () {
    size_t data_size = 0;
    std::string filename;

    std::cin >> data_size >> filename;

    std::ofstream file(filename);

    file << data_size << "\n";

	std::vector<int>vector(data_size);
    rand_init(vector.rbegin(), vector.rend(), -100000, 100000);

    for (int i = 0; i < vector.size(); ++i) file << vector[i] << "\n";

    file.close();

}
