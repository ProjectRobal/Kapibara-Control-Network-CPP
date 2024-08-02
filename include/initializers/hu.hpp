#pragma once

#include <random>
#include <cmath>

#include "initializer.hpp"

#include "config.hpp"

namespace snn
{
    class HuInit : public Initializer
    {

        std::mt19937 gen; 

        public:

        HuInit()
        {
            std::random_device rd; 

            this->gen = std::mt19937(rd());

        }

        void init(SIMDVector& vec,size_t N)
        {
            
            std::normal_distribution<number> gauss(0.f,std::sqrt(2.f/static_cast<number>(N)));

            for(size_t i=0;i<N;++i)
            {
                vec.append(gauss(gen));
            }
        }

        void init(number& n)
        {
            std::normal_distribution<number> gauss(0.f,std::sqrt(2.f));

            n=gauss(gen);
        }
    };
}