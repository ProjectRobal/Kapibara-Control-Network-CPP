#pragma once

#include <random>

#include "initializer.hpp"

#include "config.hpp"

namespace snn
{
    class ConstantInit : public Initializer
    {

        number value;

        public:

        ConstantInit(number value)
        {
            this->value = value;
        }

        void init(SIMDVector& vec,size_t N)
        {

            for(size_t i=0;i<N;++i)
            {
                vec.append(this->value);
            }
        }

        void init(number& n)
        {
            n = this->value;
        }
    };
}