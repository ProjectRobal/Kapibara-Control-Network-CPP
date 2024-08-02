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

namespace snn
{

    template<size_t Populus>
    class SubBlock
    {   
        /*
            It will store each genome with a counter which indicate how many times entity was tested.
        */
        struct NumberEntity
        {
            number value;
            uint8_t counter;
            long double reward;

            NumberEntity()
            {
                this->value = 0;
                this->counter = 0;
                this->reward = 0;
            }
        };

        std::shared_ptr<Mutation> mutate;

        size_t mating_counter;

        std::uniform_int_distribution<size_t> uniform;

        std::array<NumberEntity,Populus> population;

        NumberEntity* choosen;

        std::mt19937 gen; 

        public:

        SubBlock()
        :uniform(0,Populus-1),
        population({NumberEntity()}),
        choosen(NULL)
        {
            std::random_device rd;

            this->gen = std::mt19937(rd());
        }

        SubBlock(std::shared_ptr<Mutation> _mutate)
        :SubBlock()
        {
            this->mutate = _mutate;
        }

        void maiting()
        {

        }

        void setup(std::shared_ptr<Initializer> init)
        {
            for(auto& entity : population)
            {
                init->init(entity.value);
            }

        }

        void chooseWorkers()
        {

            size_t id=this->uniform(gen);

            this->choosen = &this->population[id];
        }


        void giveReward(long double reward)
        {          
            if( this->choosen != NULL )
            {
                return;
            }

            this->choosen->reward += reward;

            // perform maititng here

            this->maiting();
        }

        number get()
        {
            if( this->choosen == NULL )
            {
                return 0;
            }

            return this->choosen->value;
        }

        number operator()()
        {
            return this->get();
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