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

        std::uniform_real_distribution<float> uniform;

        number weight;
        long double reward;
        long double last_reward;

        size_t weight_id;

        SIMDVector pop_weights;
        SIMDVector pop_rewards;

        size_t Ticks;

        public:

        SubBlock()
        {
            std::random_device rd;

            this->gen = std::mt19937(rd());

            this->weight = 0;
            this->reward = 0;
            this->last_reward = -9999;

            this->weight_id = 0;

            this->uniform = std::uniform_real_distribution<float>(0.f,1.f);

            this->Ticks = 0;
            
        }

        SubBlock(std::shared_ptr<Mutation> _mutate)
        :SubBlock()
        {
            this->mutate = _mutate;
        }

        void setup(size_t inputSize,std::shared_ptr<Initializer> init)
        {
            this->distribution = std::normal_distribution<number>(0.f,std::sqrt(2.f/inputSize));   

            for(size_t i=0;i<Populus;++i)
            {
                this->pop_weights.append(this->distribution(this->gen));
            }

            this->pop_rewards = SIMDVector(1.f,Populus);
        }

        number get_max()
        {
            number max = 0;
            for(size_t i=1;i<this->pop_weights.size();++i)
            {
                if( pop_rewards[i] > pop_rewards[max] )
                {
                    max = i;
                }
            }

            return this->pop_weights.pop(max);
        }

        void chooseWorkers()
        {
           
            SIMDVector exp_rewards = this->pop_rewards;

            number exp_rewards_mean = exp_rewards.reduce();

            // for(size_t i=0;i<this->pop_rewards.size();++i)
            // {
            //     exp_rewards.append(std::exp(this->pop_rewards[i]));
            // }

            exp_rewards = exp_rewards / exp_rewards_mean;

            float probability = this->uniform(this->gen);

            float prob_sum = 0.f;

            for(size_t i=0;i<exp_rewards.size();++i)
            {   
                prob_sum += exp_rewards[i];

                if( prob_sum >= probability )
                {
                    this->weight_id = i;
                    break;
                }
            }

            number best_weight;

            size_t max_i=0;

            for(size_t i=1;i<exp_rewards.size();++i)
            {
                if( exp_rewards[i] > ( exp_rewards[max_i] ) )
                {
                    max_i = i;
                }
            }

            best_weight = this->pop_weights[max_i];

            snn::SIMDVector d_weight = ((this->pop_weights*-1.f) + best_weight)*0.1f;

            d_weight.set(0.f,max_i);

            this->pop_weights += d_weight;

            // think about this part:
            for(size_t i=0;i<pop_rewards.size();++i)
            {   
                float mutation_probability = this->uniform(this->gen);

                number level = 1.f - ( exp_rewards[i] );

                if( mutation_probability < level  )
                {
                    
                    this->pop_weights.set(this->distribution(this->gen),i);
                    
                }
            }
            
        }

        void giveReward(long double reward)
        {   
            // if( reward == 0 )
            // {
            //     reward = 100;
            // }

            // long double dr = (reward - this->last_reward)*20.f;

            this->pop_rewards.set(std::exp(reward),this->weight_id);

            this->last_reward = reward;

            this->Ticks++;

        }

        number get()
        {
            return this->pop_weights[this->weight_id];
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