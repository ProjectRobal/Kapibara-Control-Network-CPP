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

            this->pop_rewards = SIMDVector(0.f,Populus);
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
           
            SIMDVector exp_rewards = snn::pexp(this->pop_rewards);

            // for(size_t i=0;i<this->pop_rewards.size();++i)
            // {
            //     exp_rewards.append(std::exp(this->pop_rewards[i]));
            // }

            exp_rewards = exp_rewards / exp_rewards.reduce();

            float probability = this->uniform(this->gen);

            for(size_t i=0;i<exp_rewards.size();++i)
            {   
                probability -= exp_rewards[i];

                if( probability <= 0.f )
                {
                    this->weight_id = i;
                    break;
                }
            }
            
            // think about this part:
            if(this->Ticks>Populus*2)
            {
                for(size_t i=0;i<exp_rewards.size();++i)
                {   
                    float mutation_probability = this->uniform(this->gen);

                    if( mutation_probability <= (1.f - exp_rewards[i] ) )
                    {
                        this->pop_weights.set(this->distribution(this->gen),i);
                        this->pop_rewards.set(0.f,i);   
                    }
                }

                this->Ticks = 0;
            }
            
            // if( this->Ticks > Populus*2 )
            // {
            //     number w1 = this->get_max();
            //     number w2 = this->get_max();

            //     number mean = (w1 + w2) / 2.f;

            //     number std = std::sqrt(( (w1 - mean)*(w1 - mean) + (w2 - mean)*(w2 - mean) ) / 2.f);

            //     std::normal_distribution<number> evo = std::normal_distribution<number>(mean,std);

            //     this->pop_weights.clear();

            //     this->pop_weights.append(w1);
            //     this->pop_weights.append(w2);   

            //     for(size_t i=0;i<Populus-2;++i)
            //     {
            //         this->pop_weights.append(evo(this->gen));
            //     }

            //     for(size_t i=0;i<5;++i)
            //     {
            //         this->pop_weights.append(this->distribution(this->gen));
            //     }

            //     this->pop_rewards = SIMDVector(0.f,Populus+5);

            //     this->Ticks = 0;
            // }
        }

        void giveReward(long double reward)
        {   

            this->pop_rewards.set(0.5f*this->pop_rewards[this->weight_id]+reward,this->weight_id);

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