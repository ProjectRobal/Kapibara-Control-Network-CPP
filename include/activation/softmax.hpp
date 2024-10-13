#pragma once

#include "misc.hpp"

#include "activation.hpp"

namespace snn
{
    class SoftMax
    {
        
        public:

        static inline void activate(SIMDVector& vec)
        {
            SIMDVector v=exp(vec);

            number sum = v.reduce();

            vec = v / sum;
        }

        template<size_t Size>
        static inline void activate(SIMDVectorLite<Size>& vec)
        {
            SIMDVectorLite<Size> v=exp(vec);

            number sum = v.reduce();

            vec = v / sum;
        }

        static inline void inverse(SIMDVector& vec)
        {
            
            
            
        }
    };
}