#pragma once

#include <random>

#include "initializer.hpp"

#include "config.hpp"

namespace snn
{
    class GaussInit : public Initializer
    {

        std::normal_distribution<number> gauss;

        std::mt19937 gen; 

        public:

        GaussInit(number mean,number std)
        : gauss(mean,std)
        {
            std::random_device rd; 

            this->gen = std::mt19937(rd());

        }

        void init(SIMDVector& vec,size_t N)
        {            
            for(size_t i=0;i<N;++i)
            {
                vec.append(this->gauss(gen));
            }
        }

        void init(number& n)
        {
            n=this->gauss(gen);
        }
    };
}