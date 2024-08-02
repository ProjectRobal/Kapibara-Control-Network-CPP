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

        std::shared_ptr<Mutation> mutate;

        size_t mating_counter;

        std::mt19937 gen; 

        std::normal_distribution<number> distribution;

        number weight;
        long double reward;

        SIMDVector past_weights;
        SIMDVector past_rewards;

        public:

        SubBlock()
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
            number mean;
            number std;

            init->init(mean);
            init->init(std);

            this->distribution = std::normal_distribution<number>(mean,std);   
        }

        void chooseWorkers()
        {
            this->past_weights.append(this->weight);
            this->past_rewards.append(this->reward);
            this->reward = 0.f;

            this->weight = this->distribution(this->gen);

        }

        void giveReward(long double reward)
        {          
            this->reward += reward;

            // this takes some time ...
            // long double _reward = this->reward*0.8;
            // for( size_t i=this->past_weights.size()-1;i>=0;--i )
            // {
            //     this->past_rewards.set(this->past_rewards[i]+_reward,i);
            //     _reward = _reward*0.8;
            // }

            this->maiting();
        }

        number get()
        {
            return this->weight;
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