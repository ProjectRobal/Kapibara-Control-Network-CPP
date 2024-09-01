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
        number std;

        std::normal_distribution<number> distribution;
        std::normal_distribution<number> global;

        std::uniform_real_distribution<float> uniform;

        number weight;
        number mean;

        long double reward;
        long double last_reward;

        size_t weight_id;

        SIMDVector pop_weights;
        SIMDVector pop_rewards;

        size_t Ticks;

        number max_std;
        number min_std;

        public:

        SubBlock()
        {
            std::random_device rd;

            this->gen = std::mt19937(rd());

            this->weight = 0;
            this->reward = 0;
            this->mean = 0;

            this->weight_id = 0;

            this->uniform = std::uniform_real_distribution<float>(0.f,1.f);

            this->Ticks = 0;
            this->last_reward = -99999;
            
        }

        SubBlock(std::shared_ptr<Mutation> _mutate)
        :SubBlock()
        {
            this->mutate = _mutate;
        }

        void setup(size_t inputSize,std::shared_ptr<Initializer> init)
        {
            this->std = std::sqrt(2.f/inputSize);
            this->distribution = std::normal_distribution<number>(0.f,this->std);   
            this->global = std::normal_distribution<number>(0.f,this->std);  

            this->max_std = this->std;
            this->min_std = MIN_STD;

            this->weight = this->distribution(this->gen);

            // this->pop_rewards = SIMDVector(0.f,Populus+5);
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

            this->pop_rewards.append(this->reward);
            this->pop_weights.append(this->weight);

            if( this->pop_rewards.size() >= Populus )
            {
                snn::quicksort_mask(this->pop_weights,0,this->pop_weights.size()-1,this->pop_rewards);

                snn::SIMDVector best;

                for(size_t i=0;i<Populus/2;++i)
                {
                    best.append(this->pop_weights[this->pop_weights.size()-1 - i]);
                }

                number mean = best.reduce() / best.size();

                best = best - mean;

                best = best*best;

                number std = std::sqrt(best.reduce()/best.size());

                this->distribution = std::normal_distribution<number>(mean,std);

                this->pop_rewards.clear();
                this->pop_weights.clear();
            }


            number mutation_probability_level = std::max(std::exp(this->reward),static_cast<number>(1.f));


            number mutation_chooser = this->uniform(this->gen);

            if( mutation_chooser > mutation_probability_level )
            {
                this->weight = this->global(this->gen);
            }
            else
            {
                this->weight = this->distribution(this->gen);
            }

            
            this->reward = 0.f;
        }

        // let's tread reward not as reward rather than error
        void giveReward(long double reward)
        {   
            this->reward += reward;
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