#pragma once

#include <algorithm>
#include <vector>
#include <array>
#include <memory>
#include <random>
#include <fstream>


#include "simd_vector.hpp"
#include "neuron.hpp"
#include "initializer.hpp"
#include "crossover.hpp"
#include "mutation.hpp"

#include "config.hpp"

#include "misc.hpp"

#include "sub_block.hpp"

namespace snn
{

    template<size_t inputSize,size_t Populus>
    class BlockKAC
    {   
        /*
            It will store each genome with a counter which indicate how many times entity was tested.
        */


        SIMDVector worker;
        number bias;

        std::array<SubBlock<Populus>,inputSize> population;
        SubBlock<Populus> biases;


        // now since each sub block gets the same reward we will just them a pointer to it.
        long double *reward;


        public:

        BlockKAC()
        : population({SubBlock<Populus>()})
        {
            this->worker = SIMDVector(0.f,inputSize);
            this->reward = new long double(0.f);
        }

        void setup()
        {

            for(auto& subpopulation : population)
            {
                subpopulation.setup(inputSize,this->reward);    
            }

            // biases.setup(inputSize,init);  

            // this->bias = this->biases.get();   

            size_t i=0;
            for(auto& subpopulation : this->population)
            {
                this->worker.set(subpopulation.get(),i);
                ++i;
            }            

        }

        void chooseWorkers()
        {
            // this->biases.chooseWorkers();

            // this->bias = this->biases.get();

            size_t i=0;
            for(auto& subpopulation : this->population)
            {                
                subpopulation.chooseWorkers();
                this->worker.set(subpopulation.get(),i);
                ++i;
            }   

        }


        void giveReward(long double reward)
        {          
            // this->biases.giveReward(reward);

            *this->reward += reward;

            // this takes a lot
            // for(auto& subpopulation : this->population)
            // {
            //     // subpopulation.giveReward(reward);
            //     std::cout.setstate(std::ios_base::failbit);
            // }
            // std::cout.clear();
        }

        number fire(SIMDVector input)
        {
            return ( this->worker*input ).reduce();// + this->bias;
        }       


        void dump(std::ofstream& out) const
        {
            this->biases.dump(out);

            for(const auto& block : this->population)
            {
                block.dump(out);
            }
        }

        void load(std::ifstream& in)
        {
            this->biases.load(in);

            for(auto& block : this->population)
            {
                block.load(in);
            }
        }

        ~BlockKAC()
        {
            delete reward;
        }
        
    };
}