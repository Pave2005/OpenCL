#include "funcs.hpp"

#include <gtest/gtest.h>

#include <string>


TEST(test1, e2e)
{
	test_funcs::run_test<int>("/test1.dat");
}

TEST(test2, e2e)
{
	test_funcs::run_test<int>("/test2.dat");
}

TEST(test3, e2e)
{
	test_funcs::run_test<int>("/test3.dat");
}

TEST(test4, e2e)
{
	test_funcs::run_test<int>("/test3.dat");
}

TEST(test5, e2e)
{
	test_funcs::run_test<int>("/test3.dat");
}
