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
        SIMDVector long_past_rewards;
        SIMDVector long_past_var;

        uint8_t Ticks;

        number mean;
        number std;

        long double current_reward;

        public:

        SubBlock()
        {
            std::random_device rd;

            this->gen = std::mt19937(rd());

            this->weight = 0;
            this->reward = 0;

            this->mean = 0;
            this->std = INITIAL_STD;

            this->current_reward = 0;
            
        }

        SubBlock(std::shared_ptr<Mutation> _mutate)
        :SubBlock()
        {
            this->mutate = _mutate;
        }

        size_t find_best(const SIMDVector& vector)
        {
            size_t best_i = 0;
            for(size_t i=1;i<vector.size();++i)
            {
                if(vector[i] > vector[best_i])
                {
                    best_i = i;
                }
            }

            return best_i;
        }

        void maiting()
        {
            // if lenght is sufficient

            if( this->long_past_rewards.size() >= Populus )
            {
                // std::cout<<"Merging! "<<this->long_past_rewards.size()<<std::endl;
                number reduced_rewards = this->long_past_rewards.reduce();

                number weighted_weights = (this->long_past_mean*this->long_past_rewards).reduce();

                number mean = weighted_weights / reduced_rewards;
                number std=0.f;

                // this->past_weights-=mean;

                SIMDVector sqaure_weights = this->long_past_mean*this->long_past_mean;

                number var = ((sqaure_weights*this->long_past_rewards).reduce()/reduced_rewards) - mean*mean;

                this->std = std::sqrt(var);

                // std::cout<<"Global crossover: "<<std::endl;
                // std::cout<<"Mean: "<<mean<<std::endl;
                // std::cout<<"Var: "<<var<<std::endl;


                // something wrong here!!!!:

                this->distribution = std::normal_distribution<number>(mean,var);

                //this->long_past_rewards.clear();
                //this->long_past_mean.clear();
                this->long_past_var.clear();


            }
        }

        void setup(std::shared_ptr<Initializer> init)
        {
            init->init(this->mean);
            this->distribution = std::normal_distribution<number>(this->mean,INITIAL_STD);   
        }

        void chooseWorkers()
        {

            this->long_past_mean.append(this->weight);
            this->long_past_rewards.append(this->past_rewards.reduce());

            this->past_rewards.clear();

            size_t max_i=0;

            for(size_t i=1;i<this->long_past_mean.size();++i)
            {
                if( this->long_past_rewards[i] > max_i )
                {
                    max_i = i;
                }
            }

            this->mean = this->long_past_mean[max_i];

            this->distribution = std::normal_distribution<number>(this->mean,this->std);   

            this->weight = this->distribution(this->gen);

            this->weight = std::max(this->weight,(number)-1000.0);
            this->weight = std::min(this->weight,(number)1000.0);

        }

        void giveReward(long double reward)
        {   
            reward += 0.0000001;

            this->reward += reward;

            this->past_rewards.append(this->reward);

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