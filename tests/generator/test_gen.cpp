#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <fstream>
#include <exception>

int main ()
try
{
    int data_size = 0;
    std::string filename;

    std::cin >> data_size >> filename;

    std::ofstream file(filename);
    if (!file) throw std::runtime_error("file error");

    file << data_size << std::endl;

	std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(-10000, 10000);

    for (size_t id = 0; id < data_size; ++id)
    {
        file << dis(gen) << std::endl;
    }

    file.close();
}
catch (std::exception& expt)
{
    std::cout << expt.what() << std::endl;
    return 1;
}
