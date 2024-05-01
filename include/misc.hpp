#pragma once

#include "simd_vector.hpp"

#include "config.hpp"

namespace snn
{
    SIMDVector power(const SIMDVector& vec, size_t N)
        {
            SIMDVector out=vec;

            while(--N)
            {
                out=out*vec;
            }

            return out;
        }

        SIMDVector exp(const SIMDVector& vec)
        {
            size_t n=1;

            SIMDVector x1=vec;

            SIMDVector sum=x1+1;

            while(n < 20)
            {
                x1=x1*(vec/(++n));
                sum+=x1;
            };

            return sum;
            
        }

        SIMDVector abs(const SIMDVector& vec)
        {
            SIMDVector check = ((vec<0)*-2)+1;

            return vec*check;
           
            
        }
};