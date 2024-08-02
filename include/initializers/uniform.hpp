#pragma once

#include <random>

#include "initializer.hpp"

#include "config.hpp"

namespace snn
{
    class UniformInit : public Initializer
    {

        std::uniform_real_distribution<number> uniform;
        std::mt19937 gen; 

        public:

        UniformInit(number a,number b)
        : uniform(a,b)
        {
            std::random_device rd; 

            this->gen = std::mt19937(rd());
        }

        void init(SIMDVector& vec,size_t N)
        {
            for(size_t i=0;i<N;++i)
            {
                vec.append(this->uniform(gen));
            }
        }

        void init(number& n)
        {
            n=this->uniform(gen);
        }
    };
}