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

        std::uniform_real_distribution<float> mutation;

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

        number min_std;
        number max_std;

        long double best_reward;

        long double last_reward;
        
        long double decay;

        public:

        SubBlock()
        {
            std::random_device rd;

            this->gen = std::mt19937(rd());

            this->weight = 0;
            this->reward = 0;

            this->mean = 0;
            this->std = INITIAL_STD;

            this->min_std = MIN_STD;

            this->best_reward = -999999;

            this->mutation = std::uniform_real_distribution<float>(0.f,1.f);
            
        }

        SubBlock(std::shared_ptr<Mutation> _mutate)
        :SubBlock()
        {
            this->mutate = _mutate;
        }

        void maiting()
        {
            // if lenght is sufficient

            if( this->long_past_rewards.size() >= Populus )
            {
                snn::SIMDVector filter_reward;
                snn::SIMDVector filter_weight;

                // std::cout<<"Maiting"<<std::endl;
                // std::cout<<"Merging! "<<this->long_past_rewards.size()<<std::endl;
                // number reduced_rewards = this->long_past_rewards.reduce();

                // number weighted_weights = (this->long_past_mean*this->long_past_rewards).reduce();

                // some more professional way is welcome
                for(size_t o=0;o<4;++o)
                {

                    size_t max_i=0;

                    for(size_t i=1;i<this->long_past_mean.size()/4;++i)
                    {
                        if( this->long_past_rewards[i] > max_i )
                        {
                            max_i = i;
                        }
                    }

                    filter_reward.append(this->long_past_rewards.pop(max_i));
                    filter_weight.append(this->long_past_mean.pop(max_i));
                }

                this->mean = filter_weight.reduce()/ filter_weight.size();

                snn::SIMDVector square = filter_weight - this->mean;

                square = square*square;

                this->std = std::sqrt(square.reduce()/filter_reward.size());

                this->long_past_rewards.clear();
                this->long_past_mean.clear();

            }
        }

        void setup(size_t inputSize,std::shared_ptr<Initializer> init)
        {
            this->max_std = std::sqrt(2.f/inputSize);

            this->decay = 0.001;

            this->mean = 0.f;
            this->distribution = std::normal_distribution<number>(this->mean,INITIAL_STD);   
        }

        void chooseWorkers()
        {
            this->reward = std::exp(this->reward);
            // this->maiting();

            if( this->reward < 0 )
            {
                // this->std = std::min(2.f*this->std,(long double)this->max_std) + this->min_std;

                this->best_reward = 0.99f*this->best_reward;
            }
            else
            {
                this->std = this->min_std;
                this->best_reward = 0.6f*this->best_reward;
            }

            

            if( this->reward > this->best_reward*0.5f )
            {
                this->best_reward = this->reward;
                this->mean = this->weight;
                // this->std = this->std/2.f;

                this->past_weights.append(this->mean);
            }

            if( this->past_weights.size() > 4 )
            {
                number _mean = this->past_weights.reduce() / this->past_weights.size();
                snn::SIMDVector meaned = this->past_weights - _mean;

                meaned = meaned * meaned;

                this->std = std::sqrt(meaned.reduce()/this->past_weights.size());

                this->past_weights.clear();
            }

            number std = this->std;

            if( this->mutation(this->gen) < 0.1f )
            {
                std = this->max_std*this->mutation(this->gen) + this->min_std;
            }

            number vec_weight = 0.f;

            this->distribution = std::normal_distribution<number>(this->mean,std);  

            number n_weight =  this->distribution(this->gen);

            // we could change the coefficients
            this->weight = n_weight + vec_weight; // - std::trunc(n_weight);

            this->reward = 0;    

        }

        void giveReward(long double reward)
        {   
            reward += 0.0000001;

            this->reward += reward;

            // this->past_rewards.append(this->reward);

            //this->maiting();
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