#pragma once

#include <algorithm>
#include <vector>
#include <array>
#include <memory>
#include <random>
#include <fstream>


#include "simd_vector.hpp"
#include "simd_vector_lite.hpp"

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

        SIMDVectorLite<inputSize> worker;

        std::mt19937 gen; 

        std::normal_distribution<number> global;

        std::uniform_real_distribution<float> uniform;


        // weight from population with rewward
        typedef struct weight
        {
            long double weight;
            long double reward;
        } weight_t;
 
        // there will be inputsize amount of block
        typedef struct block
        {
            uint32_t id;

            weight_t weights[Populus];

        } block_t;

        block_t block[inputSize];

        // now since each sub block gets the same reward we will just them a pointer to it.
        long double reward;

        public:

        BlockKAC()
        {
            this->reward = 0.f;

            std::random_device rd;

            this->gen = std::mt19937(rd());

            this->uniform = std::uniform_real_distribution<float>(0.f,1.f);

            number std = std::sqrt(2.f/inputSize);
            this->global = std::normal_distribution<number>(0.f,std);  

            this->worker = SIMDVectorLite<inputSize>(0);

        }

        void setup()
        {
            // create file if not exits

            size_t i=0;

            for(block_t& block : this->block)
            {
                block.id = static_cast<uint32_t>(std::round(this->uniform(this->gen)*Populus));

                for( weight_t& w : block.weights )
                {
                    w.weight = this->global(this->gen);
                }

                this->worker.set(i,block.weights[block.id].weight);

                ++i;
            }

        }

        void chooseWorkers()
        {
            // this->biases.chooseWorkers();

            // select new id in files 

            // this->bias = this->biases.get();

            // size_t i=0;
            // for(auto& subpopulation : this->population)
            // {                
            //     subpopulation.chooseWorkers();
            //     this->worker.set(subpopulation.get(),i);
            //     ++i;
            // }   


            

        }


        void giveReward(long double reward) 
        {          
            this->reward += reward;
        }

        number fire(SIMDVectorLite<inputSize> input)
        {
            return ( this->worker*input ).reduce();// + this->bias;
        }       


        void dump(std::ofstream& out) const
        {
            
        }

        void load(std::ifstream& in)
        {
            
        }
        
    };
}