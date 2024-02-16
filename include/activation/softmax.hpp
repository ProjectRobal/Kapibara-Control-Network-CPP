#pragma once

#include "activation.hpp"

namespace snn
{
    class SoftMAX : public Activation
    {
        public:

        inline void activate(SIMDVector& vec)
        {

            number avg=vec.dot_product();
            
            vec/=avg;
            
        }
    };
}