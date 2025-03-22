#include <string>

#include <gtest/gtest.h>

#include "funcs.h"

TEST(test1, e2e)
{
	test_funcs::run_test("/test1.dat");
}

TEST(test2, e2e)
{
	test_funcs::run_test("/test2.dat");
}

TEST(test3, e2e)
{
	test_funcs::run_test("/test3.dat");
}
