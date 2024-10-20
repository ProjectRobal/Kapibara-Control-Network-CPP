#pragma once

#include <algorithm>
#include <vector>
#include <array>
#include <memory>
#include <random>
#include <fstream>
#include <algorithm>

#include "simd_vector.hpp"
#include "simd_vector_lite.hpp"

#include "initializer.hpp"

#include "config.hpp"

#include "misc.hpp"

namespace snn
{
    class BlockCounter
    {
        public:

        static size_t BlockID;

        BlockCounter()
        {
            this->BlockID++;
        }
    };

    template<size_t inputSize,size_t Populus>
    class BlockKAC
    {   
        /*
            It will store each genome with a counter which indicate how many times entity was tested.
        */

        SIMDVectorLite<inputSize> worker;

        std::mt19937 gen; 

        std::normal_distribution<number> global;

        std::uniform_real_distribution<float> uniform;


        // weight from population with rewward
        typedef struct weight
        {
            long double weight;
            long double reward;
            
        } weight_t;
 
        // there will be inputsize amount of block
        typedef struct block
        {
            uint32_t id;
            uint16_t swap_count;
            weight_t weights[Populus];
            long double best_weight_buffer;
            uint16_t collected_b_weight;
            long double worse_reward;

        } block_t;

        block_t block[inputSize];

        // now since each sub block gets the same reward we will just them a pointer to it.
        long double reward;
        long double last_reward;

        public:

        size_t Id;

        BlockKAC()
        {
            this->reward = 0.f;

            this->last_reward = 0.f;

            std::random_device rd;

            this->gen = std::mt19937(rd());

            this->uniform = std::uniform_real_distribution<float>(0.f,1.f);

            number std = std::sqrt(2.f/inputSize);
            this->global = std::normal_distribution<number>(0.f,std);  

            this->worker = SIMDVectorLite<inputSize>(0);

            this->Id = BlockCounter::BlockID;

            std::cout<<this->Id<<std::endl;

            BlockCounter::BlockID++;

        }

        void setup()
        {
            // create file if not exits

            std::cout<<this->Id<<std::endl;

            size_t i=0;

            for(block_t& block : this->block)
            {
                block.id = static_cast<uint32_t>(std::round(this->uniform(this->gen)*(Populus-1)));
                block.swap_count = 0;
                block.best_weight_buffer = 0;
                block.collected_b_weight = 0;
                block.worse_reward = 0;

                for( weight_t& w : block.weights )
                {
                    w.weight = this->global(this->gen);
                    w.reward = 0.f;
                }

                // this->worker.set(i,block.weights[block.id].weight);

                this->worker[i] = block.weights[block.id].weight;

                ++i;
            }
 
        }

        number give_new_weight(block_t& victim)
        {
            if( victim.collected_b_weight > 0 )
            {
                if( this->uniform(this->gen) >= 0.1f )
                {
                    return victim.best_weight_buffer / static_cast<number>(victim.collected_b_weight);
                }

                return (victim.best_weight_buffer / static_cast<number>(victim.collected_b_weight)) + this->global(this->gen);
            }

            return this->global(this->gen);
        }

        void selection(block_t& victim)
        {
            victim.id = 0;

            qsort(victim.weights,Populus,sizeof(weight_t),[](const void* a,const void* b)->int{
                return ((weight_t*)b)->reward - ((weight_t*)a)->reward;
            });

            if( this->Id == 1 )
            {
                std::cout<<"Sorted: "<<std::endl;
                std::cout<<"first: "<<victim.weights[0].reward<<" last: "<<victim.weights[Populus-1].reward<<std::endl;
            }

            float prob = 0.5f / Populus;

            for(size_t i=0;i<Populus;++i)
            {
                if( prob*(i+1) > this->uniform(this->gen) )
                {
                    victim.weights[i].weight = this->give_new_weight(victim);
                }

                victim.weights[i].reward = 0.f;
            }

            victim.worse_reward = 0.f;
        }

        void chooseWorkers()
        {
            if(this->Id==1)
            {
                std::cout<<"Choose workers"<<std::endl;
            }
            size_t iter=0;
            for(block_t& _block : this->block)
            {
                weight_t& w = _block.weights[_block.id];

                w.reward = this->reward + 0.1*w.reward + 0*(this->reward - w.reward);

                if( this->reward >=0 )
                {
                    w.reward = this->reward;
                }

                float switch_probability = 0.01;

                if( w.reward < 0 )
                {
                    switch_probability = std::min<float>(REWARD_TO_SWITCH_PROBABILITY*this->reward,0.25f);
                }
                else if( this->last_reward < 0 )
                {
                    if(this->Id == 1)
                    {
                        std::cout<<"New best weight found! "<<std::endl;
                    }
                    _block.best_weight_buffer += w.weight;
                    _block.collected_b_weight ++ ;

                    if( _block.collected_b_weight > Populus )
                    {
                        _block.best_weight_buffer = _block.best_weight_buffer / static_cast<number>(_block.collected_b_weight);

                        _block.collected_b_weight = 1;
                    }
                }

                if( this->uniform(this->gen) < switch_probability )
                {
                    if(this->Id==1)
                    {
                        std::cout<<"Swap occured!"<<std::endl;
                        std::cout<<"Swap count: "<<_block.swap_count+1<<std::endl;
                        std::cout<<"with probability "<<switch_probability<<std::endl;
                    }
                    _block.id = ( static_cast<uint32_t>(std::round(this->uniform(this->gen)*(Populus-2))) + _block.id ) % ( Populus - 1 );

                    this->worker[iter] = _block.weights[_block.id].weight;

                    _block.swap_count++;

                    if( _block.swap_count >= Populus*2 )
                    {
                        if(this->Id==1)
                        {
                            std::cout<<"Selection started!"<<std::endl;
                        }
                        this->selection(_block);
                        _block.swap_count = 0;
                    }
                }

                iter++;

            }

            this->reward = 0;
        }


        void giveReward(long double reward) 
        {          
            this->reward += ( reward + REWARD_DIFFERENCE_GAIN*(reward - this->last_reward) );

            this->last_reward = reward;
        }

        number fire(const SIMDVectorLite<inputSize>& input)
        {
            SIMDVectorLite<inputSize> res = this->worker*input;
            return ( this->worker*input ).reduce();// + this->bias;
        }       


        void dump(std::ostream& out) const
        {
            for(size_t i=0;i<inputSize;++i)
            {
                out.write((char*)&this->block[i],sizeof(block_t));
            }
        }

        void load(std::istream& in)
        {
            for(size_t i=0;i<inputSize;++i)
            {
                in.read((char*)&this->block[i],sizeof(block_t));

                this->worker[i] = this->block[i].weights[this->block[i].id].weight;
            }   
        }
        
    };
}