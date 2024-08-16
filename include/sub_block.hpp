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

    template<size_t Populus,size_t LongPopulus=4*Populus>
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

        SIMDVector long_past_mean;
        SIMDVector long_past_var;
        SIMDVector long_past_rewards;

        public:

        SubBlock()
        {
            std::random_device rd;

            this->gen = std::mt19937(rd());

            this->weight = 0;
            this->reward = 0;
            
        }

        SubBlock(std::shared_ptr<Mutation> _mutate)
        :SubBlock()
        {
            this->mutate = _mutate;
        }

        void maiting()
        {
            // if lenght is sufficient
            if( this->past_weights.size() >= Populus )
            {
                // std::cout<<"Maiting! "<<this->past_weights.size()<<std::endl;
                this->past_weights.append(this->weight);
                this->past_rewards.append(this->reward);

                number weighted_weights = (this->past_weights*this->past_rewards).reduce();
                number reduced_rewards = this->past_rewards.reduce();

                number mean = weighted_weights / reduced_rewards;
                number std=0.f;

                // this->past_weights-=mean;

                this->past_weights*=this->past_weights;

                number var = ((this->past_weights*this->past_rewards).reduce()/reduced_rewards) - mean*mean;

                std = std::sqrt(var);

                // something wrong here:

                this->distribution = std::normal_distribution<number>(mean,std);

                this->past_weights.clear();
                this->past_rewards.clear();

                this->long_past_rewards.append(reduced_rewards);
                this->long_past_mean.append(mean);
                this->long_past_var.append(var);
            }

            if( this->long_past_rewards.size() >= LongPopulus )
            {
                // std::cout<<"Merging! "<<this->long_past_rewards.size()<<std::endl;
                number reduced_rewards = this->long_past_rewards.reduce();

                number mean = (this->long_past_mean*this->long_past_rewards).reduce() / reduced_rewards;

                number var = (this->long_past_var*this->long_past_rewards).reduce() / reduced_rewards;

                // something wrong here:

                this->distribution = std::normal_distribution<number>(mean,var);

                this->long_past_rewards.clear();
                this->long_past_mean.clear();
                this->long_past_var.clear();
            }
        }

        void setup(std::shared_ptr<Initializer> init)
        {
            number mean;
            number std;

            init->init(mean);
            std = INITIAL_STD;

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
            reward += 0.0000001;
            this->reward += std::exp(reward);

            // long double _reward = this->reward*0.8;
            // for( size_t i=0;i>std::min((size_t)5,this->past_weights.size());++i )
            // {
            //     this->past_rewards.set(this->past_rewards[this->past_weights.size()-1 - i]+_reward,i);
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