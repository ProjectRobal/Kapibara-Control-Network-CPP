#pragma once

#include "misc.hpp"

#include "activation.hpp"

namespace snn
{
    class SiLu : public Activation
    {
        
        public:

        inline void activate(SIMDVector& vec)
        {
            SIMDVector v=exp(vec);

            vec=(v/(v+1))*vec;
        }

        template<size_t Size>
        inline void activate(SIMDVectorLite<Size>& vec)
        {
            SIMDVectorLite<Size> v=exp(vec);

            vec=(v/(v+1))*vec;
        }

        inline void inverse(SIMDVector& vec)
        {
            
            
            
        }
    };
}