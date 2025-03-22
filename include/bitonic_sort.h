#pragma once

#include "ocl.h"
#include "utils.h"

namespace bitonic
{
	template <typename T>
	class BitonicSort final
	{
	private:
		OpenCL::OclApp<T> app_;
		std::vector<T>   data_;
		std::size_t  size_ = 0;

		void bitonic_sort (std::vector<T>& res)
		{
			if (app_.kernel().setArg(0, app_.buffer()) != CL_SUCCESS)
				throw std::runtime_error("setArg(0): kernel error");

			for (int block_size = 2; block_size <= size_; block_size *= 2)
			{
				for (int step = block_size / 2; step > 0; step /= 2)
				{
					app_.kernel().setArg(1, block_size);
					app_.kernel().setArg(2, step);

					if (app_.queue().enqueueNDRangeKernel(app_.kernel(), cl::NullRange,  cl::NDRange(size_), cl::NullRange) != CL_SUCCESS)
						throw std::runtime_error("enqueueNDRangeKernel: kernel error");

					app_.queue().finish();
				}
			}

			if (res.size() < size_) res.resize ( size_, 0 );

			if (app_.queue().enqueueReadBuffer(app_.buffer(), CL_TRUE, 0, sizeof(T) * size_, res.data()) != CL_SUCCESS)
				throw std::runtime_error("enqueueReadBuffer: kernel error");
		}

	public:
		BitonicSort (OpenCL::OclApp<T>& app, std::vector<T>& data) : app_(app), data_(data), size_(data_.size()) {}

		void sort (std::vector<T>& res) { bitonic_sort(res); }
	};
}
