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

        std::shared_ptr<Mutation> mutate;

        SIMDVector worker;
        number bias;

        std::array<SubBlock<Populus,4*Populus>,inputSize> population;
        SubBlock<Populus,4*Populus> biases;


        public:

        BlockKAC(std::shared_ptr<Mutation> _mutate)
        : mutate(_mutate),
        population({SubBlock<Populus>()})
        {
            this->worker = SIMDVector(0.f,inputSize);
        }

        void setup(std::shared_ptr<Initializer> init)
        {
            for(auto& subpopulation : population)
            {
                subpopulation.setup(inputSize,init);    
            }

            biases.setup(inputSize,init);              
        }

        void chooseWorkers()
        {
            this->biases.chooseWorkers();

            this->bias = this->biases.get();

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
            this->biases.giveReward(reward);

            for(auto& subpopulation : this->population)
            {
                subpopulation.giveReward(reward);
                std::cout.setstate(std::ios_base::failbit);
            }
            std::cout.clear();
        }

        number fire(SIMDVector input)
        {
            return ( this->worker*input ).reduce() + this->bias;
        }       


        void dump(std::ofstream& out) const
        {
            // for(auto neuron : this->population)
            // {
            //     neuron->save(out);
            // }   

            // uint64_t maiting = this->mating_counter;

            // char maiting_data[sizeof(uint64_t)]={0};

            // out.write(maiting_data,sizeof(uint64_t));
        }

        void load(std::ifstream& in)
        {
            // for(auto neuron : this->population)
            // {
            //     neuron->load(in);
            // }

            // uint64_t maiting = 0;

            // char maiting_data[sizeof(uint64_t)]={0};

            // in.read(maiting_data,sizeof(uint64_t));

            // memmove(reinterpret_cast<char*>(&maiting),maiting_data,sizeof(uint64_t));

            // this->mating_counter=maiting;
        }

        
    };
}