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
            this->last_reward = 0.f;

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
            // this->maiting();

            if( this->reward < 0 )
            {
                this->std = std::min((std::log(-1000.f*this->reward + 1) + 1.f )*this->std,(long double)this->max_std) + this->min_std;
            }
            else
            {
                this->std = this->min_std;
            }

            this->past_weights.append(this->weight);
            this->past_rewards.append(std::exp(this->reward));

            // if( this->reward<0 )
            // {

            //     long double dR = this->reward - this->last_reward;

            //     if( dR < 0)
            //     {
            //         this->std = std::min(abs(dR)*this->decay + this->std,this->max_std);
            //     }
            // }
            // else
            // {
            //     this->std = this->min_std;
            // }

            // if( this->long_past_mean.size() > 1 )
            // {
            //     weighted_weights = (this->long_past_mean*this->long_past_rewards).reduce() / this->long_past_rewards.reduce();             

            //     if( this->long_past_mean.size() >= Populus )
            //     {
            //         this->mean = weighted_weights;

            //         this->long_past_mean.clear();
            //         this->long_past_rewards.clear();
            //     } 
            // }

            if( this->past_rewards.size() >= Populus )
            {

                quicksort_mask(this->past_weights,0,this->past_weights.size(),this->past_rewards);

                for(size_t i=0;i<this->past_weights.size()/2;++i)
                {
                    this->past_weights.set(0.f,i);
                    this->past_rewards.set(0.f,i);
                }

                this->mean = this->past_weights.reduce() / (this->past_weights.size()/2);

                this->std = this->min_std;

                this->past_weights.clear();
                this->past_rewards.clear();

            }

            number std = this->std;

            if( this->mutation(this->gen) < 0.1f )
            {
                std = this->max_std*this->mutation(this->gen) + this->min_std;
            }

            this->distribution = std::normal_distribution<number>(this->mean,std);  

            number n_weight =  this->distribution(this->gen);

            // number middle_weight = (this->weight*this->reward + this->mean*this->best_reward)/(this->best_reward+this->reward);

            // we could change the coefficients
            this->weight = n_weight; // - std::trunc(n_weight);

            // number dBest = (middle_weight - this->weight)*0.2f;

            // this->weight += dBest;

            // this->weight = this->weight - std::trunc(this->weight);

            this->last_reward = this->reward;

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