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
    // with only two best members it converenge faster
    template<size_t Populus,size_t AmountofMembersThatPass=5>
    class SubBlock
    {   

        std::shared_ptr<Mutation> mutate;

        std::mt19937 gen; 

        std::normal_distribution<number> distribution;
        std::normal_distribution<number> global;

        std::uniform_real_distribution<float> uniform;

        number weight;
        number best_weight;

        long double reward;
        long double best_reward;

        SIMDVector pop_weights;
        SIMDVector pop_rewards;
        SIMDVector pop_tested;

        number max_std;
        number min_std;

        number std;

        size_t choosen_weight;

        size_t id;

        public:

        static size_t SubBLockId;

        SubBlock()
        {
            SubBlock::SubBLockId++;
            this->id = SubBlock::SubBLockId;

            std::random_device rd;

            this->gen = std::mt19937(rd());

            this->weight = 0;
            this->reward = 0;
            this->best_reward = -9999999.f;
            this->best_weight = 0.f;

            this->choosen_weight = 0;
            
            this->uniform = std::uniform_real_distribution<float>(0.f,1.f);
            
        }

        SubBlock(std::shared_ptr<Mutation> _mutate)
        :SubBlock()
        {
            this->mutate = _mutate;
        }

        void setup(size_t inputSize,std::shared_ptr<Initializer> init)
        {
            number std = std::sqrt(2.f/inputSize);
            this->distribution = std::normal_distribution<number>(0.f,std);   
            this->global = std::normal_distribution<number>(0.f,std);  

            this->max_std = std;
            this->min_std = MIN_STD;

            this->weight = this->distribution(this->gen);

            for(size_t i=0;i<Populus;++i)
            {
                this->pop_weights.append(this->global(this->gen));
                this->pop_rewards.append(0.f);
                this->pop_tested.append(0.f);
            }

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

        void estimate_specie(const SIMDVector& weights)
        {
            number mean = weights.reduce()/weights.size();

            SIMDVector w = weights - mean;

            w = w*w;

            number std = std::sqrt(w.reduce()/w.size());

            this->distribution = std::normal_distribution<number>(mean,std);
        }

        void chooseWorkers()
        {
            snn::quicksort_mask(this->pop_weights,0,this->pop_weights.size()-1,this->pop_rewards);

            snn::SIMDVector norm_rewards = snn::pexp(this->pop_rewards);

            norm_rewards = norm_rewards / norm_rewards.reduce();

            number tested = (this->pop_tested>=15.f).reduce();

            if(this->id == 10)
            {
                std::cout<<this->pop_rewards<<std::endl;
                std::cout<<norm_rewards<<std::endl;
                std::cout<<"Testing:"<<std::endl;
                std::cout<<this->pop_tested<<std::endl;
                std::cout<<"Tested: "<<std::endl;
                std::cout<<tested<<std::endl;
            }


            // perform crossover!
            if( tested >= Populus/2 )
            {
                snn::quicksort_mask(this->pop_weights,0,this->pop_weights.size()-1,this->pop_rewards);

                // a k best weights leave alone
                // other weights will be discarded and replaced by mutated best weight with variance calculated from best weight or just use simple mutation :)
                // rest of population will be random weights

                if( this->id == 10 )
                {
                    std::cout<<"Crossover!"<<std::endl;
                }

                snn::SIMDVector best;

                for(size_t i=0;i<5;++i)
                {
                    best.append(this->pop_weights[this->pop_weights.size()-i-1]);
                }

                std::normal_distribution<number> best_dist(0.f,this->global.stddev()*0.1f);

                for(size_t i=0;i<this->pop_rewards.size();++i)
                {
                    this->pop_rewards.set(0.f,i);
                    this->pop_tested.set(0.f,i);
                }

                // this->pop_rewards -= this->pop_rewards[this->pop_rewards.size()-1];

                for(size_t i=0;i<5;++i)
                {
                    this->pop_weights.set(best[i],i);
                }

                for(size_t i=5;i<15;++i)
                {
                    number weight = best[i % best.size()] + best_dist(this->gen);
                    this->pop_weights.set(weight,i);
                }

                for(size_t i=15;i<Populus;++i)
                {
                    this->pop_weights.set(this->global(this->gen),i);
                }
            }

            // choose weights

            number dice = this->uniform(this->gen);

            for(size_t i=norm_rewards.size()-1;i>=0;--i)
            {
                dice -= norm_rewards[i];

                if( dice <= 0 )
                {
                    this->choosen_weight = i;

                    if(this->id == 10)
                    {
                        std::cout<<"Choosen weight id: "<<this->choosen_weight<<std::endl;
                        std::cout<<" with probability "<<norm_rewards[i]<<std::endl;
                    }
                    break;
                }
            }


            
        }

        // let's tread reward not as reward rather than error
        void giveReward(long double reward)
        {   
            this->pop_rewards.set(0.5f*this->pop_rewards[this->choosen_weight]+reward,this->choosen_weight);
            this->pop_tested.set(this->pop_tested[this->choosen_weight]+1,this->choosen_weight);
        }

        number get()
        {
            return this->pop_weights[this->choosen_weight];
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

            char buffer[SERIALIZED_NUMBER_SIZE];

            serialize_number(this->distribution.mean(),buffer);

            out.write(buffer,SERIALIZED_NUMBER_SIZE);

            serialize_number(this->distribution.stddev(),buffer);

            out.write(buffer,SERIALIZED_NUMBER_SIZE);

        }

        void load(std::ifstream& in)
        {
            char buffer[SERIALIZED_NUMBER_SIZE];

            in.read(buffer,SERIALIZED_NUMBER_SIZE);

            number mean = deserialize_number<number>(buffer);

            in.read(buffer,SERIALIZED_NUMBER_SIZE);

            number std = deserialize_number<number>(buffer);

            this->distribution = std::normal_distribution<number>(mean,std);

        }

        
    };

}