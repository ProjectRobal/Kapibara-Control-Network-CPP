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
                // std::cout<<"Merging! "<<this->long_past_rewards.size()<<std::endl;
                // number reduced_rewards = this->long_past_rewards.reduce();

                number weighted_weights = (this->long_past_mean*this->long_past_rewards).reduce();

                // some more professional way is welcome
                for(size_t o=0;o<Populus/2;++o)
                {

                    size_t max_i=0;

                    for(size_t i=1;i<this->long_past_mean.size()/4;++i)
                    {
                        if( this->long_past_rewards[i] > max_i )
                        {
                            max_i = i;
                        }
                    }

                    this->past_rewards.append(this->long_past_rewards.pop(max_i));
                    this->past_weights.append(this->long_past_mean.pop(max_i));
                }

                // std::cout<<"Best weights: "<<this->past_weights<<std::endl;
                // std::cout<<"Best rewards: "<<this->past_rewards<<std::endl;

                // number mean = this->past_weights.reduce() / this->past_weights.size();

                // SIMDVector square = this->past_weights - mean;

                // square = square*square;
                // this->std = std::sqrt(square.reduce() / square.size());

                // std::cout<<"STD: "<<this->std<<std::endl;
                // std::cout<<"Mean: "<<this->mean<<std::endl;

                this->long_past_rewards.clear();
                this->long_past_mean.clear();

                this->long_past_rewards.extend(this->past_rewards);
                this->long_past_mean.extend(this->past_weights);

                this->past_rewards.clear();
                this->past_weights.clear();

            }
        }

        void setup(size_t inputSize,std::shared_ptr<Initializer> init)
        {
            this->max_std = std::sqrt(2.f/inputSize);

            init->init(this->mean);
            this->distribution = std::normal_distribution<number>(this->mean,INITIAL_STD);   
        }

        void chooseWorkers()
        {

            if( ( this->reward / this->best_reward ) > 0.5f )
            {
                this->long_past_mean.append(this->weight);
                this->long_past_rewards.append(std::exp(this->reward));
   
            }

            if( this->reward<0 )
            {
                this->std = std::min(std::log(-0.05f*this->reward + 1)+this->min_std,(long double)this->max_std);
            }
            else
            {
                this->std = this->min_std;
            }

            number weighted_weights = 0.f; 

            if( this->long_past_mean.size() > 1 )
            {
                weighted_weights = (this->long_past_mean*this->long_past_rewards).reduce() / this->long_past_rewards.reduce();              
            }

            number std = this->std;

            if( this->mutation(this->gen) < 0.1f )
            {
                std = this->max_std*this->mutation(this->gen) + this->min_std;
            }

            this->distribution = std::normal_distribution<number>(weighted_weights ,std);  

            number n_weight =  this->distribution(this->gen);

            // we could change the coefficients
            this->weight = n_weight - std::trunc(n_weight);

            number dBest = (this->mean - this->weight)*0.1f;

            this->weight += dBest;

            this->reward = 0;    

        }

        void giveReward(long double reward)
        {   
            reward += 0.0000001;

            this->reward += reward;

            if( this->reward > this->best_reward )
            {
                this->best_reward = this->reward;   
                this->mean = this->weight;
            }

            // this->past_rewards.append(this->reward);

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