#pragma once

#include "misc.hpp"

#include "activation.hpp"

namespace snn
{
    class SoftMax : public Activation
    {
        
        public:

        inline void activate(SIMDVector& vec)
        {
            SIMDVector v=exp(vec);

            number sum = v.reduce();

            vec = v / sum;
        }

        inline void inverse(SIMDVector& vec)
        {
            
            
            
        }
    };
}