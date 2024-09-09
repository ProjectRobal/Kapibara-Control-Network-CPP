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

        std::mt19937 gen; 

        std::normal_distribution<number> global;

        std::uniform_real_distribution<float> uniform;

        SIMDVector pop_weights;
        SIMDVector pop_rewards;

        SIMDVector best_weights;

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

            this->choosen_weight = 0;
            
            this->uniform = std::uniform_real_distribution<float>(0.f,1.f);
            
        }

        SubBlock(std::shared_ptr<Mutation> _mutate)
        :SubBlock()
        {

        }

        void setup(size_t inputSize,std::shared_ptr<Initializer> init)
        {
            number std = std::sqrt(2.f/inputSize);
            this->global = std::normal_distribution<number>(0.f,std);  

            this->max_std = std;
            this->min_std = MIN_STD;

            for(size_t i=0;i<Populus;++i)
            {
                this->pop_weights.append(this->global(this->gen));
                this->pop_rewards.append(0.f);
            }

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

            for(size_t i=Populus/2;i<Populus-1;++i)
            {

                if( this->uniform(this->gen) <= 0.25f )
                {
                    // number d_weight = ( best - this->pop_weights[i] ).reduce() / 5;

                    // number weight = this->pop_weights[i] + d_weight * ( this->uniform(this->gen) + 0.1f );

                    std::normal_distribution<number> dist(0.0,this->max_std/10);

                    number mean;

                    if(this->best_weights.size()>0)
                    {
                        mean = this->best_weights.reduce() / (this->best_weights.size()+1);
                    }
                    else
                    {
                        mean = this->pop_weights[this->pop_weights.size()-1];
                    }

                    number weight = mean + dist(this->gen)*(this->uniform(this->gen)<0.5);

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
           
            if(reward>=0)
            {
                if((this->best_weights.size() == 0)||(this->best_weights[this->best_weights.size()-1]!=this->pop_weights[this->choosen_weight]))
                {
                    this->best_weights.append(this->pop_weights[this->choosen_weight]);

                    if(this->best_weights.size()>10)
                    {
                        number b_w = this->best_weights.reduce()/this->best_weights.size();

                        this->best_weights.clear();
                        this->best_weights.append(b_w);
                    }
                }
            }
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


            for(size_t i=0;i<Populus;++i)
            {
                serialize_number(this->pop_weights[i],buffer);

                out.write(buffer,SERIALIZED_NUMBER_SIZE);
            }

            for(size_t i=0;i<Populus;++i)
            {
                serialize_number(this->pop_rewards[i],buffer);

                out.write(buffer,SERIALIZED_NUMBER_SIZE);
            }

            // save rewards:
            size_t reward_count = this->best_weights.size();

            char count_buffer[sizeof(size_t)];

            memcpy(count_buffer,(char*)&reward_count,sizeof(size_t));

            out.write(count_buffer,sizeof(size_t));

            for(size_t i=0;i<reward_count;++i)
            {
                serialize_number(this->best_weights[i],buffer);

                out.write(buffer,SERIALIZED_NUMBER_SIZE);
            }

        }

        void load(std::ifstream& in)
        {
            char buffer[SERIALIZED_NUMBER_SIZE];

            for(size_t i=0;i<Populus;++i)
            {

                in.read(buffer,SERIALIZED_NUMBER_SIZE);

                number weight = deserialize_number<number>(buffer);

                this->pop_weights.set(weight,i);

            }

            for(size_t i=0;i<Populus;++i)
            {

                in.read(buffer,SERIALIZED_NUMBER_SIZE);

                number reward = deserialize_number<number>(buffer);

                this->pop_rewards.set(reward,i);
            
            }

            char count_buffer[sizeof(size_t)];

            in.read(count_buffer,sizeof(size_t));

            size_t rewards_count;

            memcpy((char*)&rewards_count,count_buffer,sizeof(size_t));

            for(size_t i=0;i<rewards_count;++i)
            {
                in.read(buffer,SERIALIZED_NUMBER_SIZE);

                number weight = deserialize_number<number>(buffer);

                this->best_weights.append(weight);
            }

        }

        
    };

}