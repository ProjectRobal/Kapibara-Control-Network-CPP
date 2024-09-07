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

        number max_std;
        number min_std;

        number std;

        size_t choosen_weight;

        size_t id;

        size_t Ticks;

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

            this->Ticks = 0;

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


            if(this->id == 10)
            {
                std::cout<<"Weights: "<<std::endl;
                std::cout<<this->pop_weights<<std::endl;
                std::cout<<this->pop_rewards<<std::endl;
                std::cout<<norm_rewards<<std::endl;
            }

            snn::SIMDVector best;

            for(size_t i=0;i<5;++i)
            {
                best.append(this->pop_weights[this->pop_weights.size()-i-1]);
            }

            if( this->Ticks >= 200 )
            {

                if( this->id == 10 )
                {
                    std::cout<<"Crossover!"<<std::endl;
                }

                // number reward_mean = this->pop_rewards.reduce() / this->pop_rewards.size();

                this->pop_rewards /=10.f;

                // // this->pop_rewards -= this->pop_rewards[this->pop_rewards.size()-1];

                // number best_mean = best.reduce()/best.size();

                // best = best - best_mean;

                // best = best * best;

                // std::normal_distribution<number> best_dist(best_mean,std::sqrt(best.reduce()/best.size()));

                // for(size_t i=0;i<Populus;++i)
                // {
                //     this->pop_weights.set(best_dist(this->gen),i);
                //     this->pop_rewards.set(0,i);
                // }

                // this->choosen_weight = 0;

                this->Ticks = 0;

                return;
            }

            this->Ticks++;


            // perform crossover!
            

                // a k best weights leave alone
                // other weights will be discarded and replaced by mutated best weight with variance calculated from best weight or just use simple mutation :)
                // rest of population will be random weights


            for(size_t i=0;i<Populus/2;++i)
            {
                if( this->uniform(this->gen) <= 0.1f )
                {
                    // number weight = this->pop_weights[i] + (best_mean - this->pop_weights[i])*0.25f;
                    // size_t choose = static_cast<size_t>(this->uniform(this->gen)*best.size());

                    // number weight = best[i] + best_dist(this->gen);
                    this->pop_weights.set(this->global(this->gen),i);

                }
            }

            for(size_t i=Populus/2;i<Populus;++i)
            {

                if( this->uniform(this->gen) <= 0.25f )
                {
                    number d_weight = ( best - this->pop_weights[i] ).reduce() / 5;

                    number weight = this->pop_weights[i] + d_weight * ( this->uniform(this->gen) + 0.1f );

                    this->pop_weights.set(weight,i);

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
                    // this->choosen_weight = this->pop_weights.size()-1;

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

            this->pop_rewards.set(0.4f*this->pop_rewards[this->choosen_weight]+reward,this->choosen_weight);
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