#pragma once

#include <memory>
#include <vector>

#include <simd_vector.hpp>

namespace snn
{

    class TensorDimension
    {
        protected:

            snn::SIMDVector dimension;

            std::vector<std::shared_ptr<TensorDimension>> tensors;

        public:


    };


}