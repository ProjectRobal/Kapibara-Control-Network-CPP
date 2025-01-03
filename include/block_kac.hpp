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

#include "initializers/hu.hpp"

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

    template<size_t inputSize,size_t Populus,class weight_initializer=HuInit<inputSize>>
    class BlockKAC
    {   
        /*
            It will store each genome with a counter which indicate how many times entity was tested.
        */

        SIMDVectorLite<inputSize> worker;

        std::mt19937 gen; 

        weight_initializer global;

        std::uniform_real_distribution<float> uniform;


        // weight from population with rewward
        typedef struct weight
        {
            float weight;
            float reward;
            
        } weight_t;
 
        // there will be inputsize amount of block
        typedef struct block
        {
            uint32_t id;
            uint16_t swap_count;
            SIMDVectorLite<Populus> weights;
        } block_t;

        block_t block[inputSize];

        // now since each sub block gets the same reward we will just them a pointer to it.
        long double reward;

        public:

        size_t Id;

        BlockKAC()
        {
            this->reward = 0.f;

            std::random_device rd;

            this->gen = std::mt19937(rd());

            this->uniform = std::uniform_real_distribution<float>(0.f,1.f);

            this->worker = SIMDVectorLite<inputSize>(0);

            this->Id = BlockCounter::BlockID;

            // std::cout<<this->Id<<std::endl;

            BlockCounter::BlockID++;

        }

        void setup()
        {
            // create file if not exits

            // std::cout<<this->Id<<std::endl;

            size_t i=0;

            for(block_t& block : this->block)
            {
                block.id = static_cast<uint32_t>(std::round(this->uniform(this->gen)*(Populus-1)));
                block.swap_count = 0;

                for(size_t w=0;w<Populus;w++)
                {
                    block.weights[w] = this->global.init();
                }

                this->worker[i] = block.weights[block.id];

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

                return (victim.best_weight_buffer / static_cast<number>(victim.collected_b_weight)) + this->global.init();
            }

            return this->global.init();
        }

        static void parralel_swap(block_t blocks[inputSize],SIMDVectorLite<inputSize>& worker,std::uniform_real_distribution<float>& uniform,std::mt19937& gen,size_t start,size_t end,float prob)
        {
            size_t step = static_cast<size_t>(prob*((end-start) + 1)) + 1;

            start += static_cast<size_t>(std::round(uniform(gen)*step));

            while(start<end)
            {

                
                    // if(this->Id==1)
                    // {
                    //     std::cout<<"Swap occured!"<<std::endl;
                    //     std::cout<<"Swap count: "<<_block.swap_count+1<<std::endl;
                    //     std::cout<<"with probability "<<switch_probability<<std::endl;
                    // }                    _block.swap_count++;

                    blocks[start].id = ( static_cast<uint32_t>(std::round(uniform(gen)*(Populus-2))) + blocks[start].id ) % ( Populus - 1 );

                    worker[start] = blocks[start].weights[blocks[start].id];

                

                start+=step;

            }
        }

        void chooseWorkers()
        {
            // if(this->Id==1)
            // {
            //     std::cout<<"Choose workers"<<std::endl;
            // }
            size_t iter=0;

            if( this->reward == 0 )
            {
                return;
            }

            if( this->reward > 0 )
            {
                // weight swarming is kinda slow :(
                for(block_t& _block : this->block)
                {
                    number w = _block.weights[_block.id];

                    _block.weights -= w;

                }
                
                return;
            }

            float switch_probability = std::max<float>(1.0 - REWARD_TO_SWITCH_PROBABILITY*this->reward,1.0 - MAX_SWITCH_PROBABILITY);

            size_t step = static_cast<size_t>(switch_probability*inputSize) + 1;

            size_t i = static_cast<size_t>(std::round(this->uniform(this->gen)*step));

            for(;i<inputSize;i+=step)
            {

                block_t& _block = this->block[i];

                // if( this->uniform(this->gen) < switch_probability )
                // {
                    // if(this->Id==1)
                    // {
                    //     std::cout<<"Swap occured!"<<std::endl;
                    //     std::cout<<"Swap count: "<<_block.swap_count+1<<std::endl;
                    //     std::cout<<"with probability "<<switch_probability<<std::endl;
                    // }                    _block.swap_count++;

                    _block.id = ( static_cast<uint32_t>(std::round(this->uniform(this->gen)*(Populus-2))) + _block.id ) % ( Populus - 1 );

                    this->worker[iter] = _block.weights[_block.id];

                // }

                iter++;

            }

            this->reward = 0;
        }

        void giveReward(long double reward) 
        {          
            this->reward += reward;
        }

        number fire(const SIMDVectorLite<inputSize>& input)
        {
            SIMDVectorLite<inputSize> res = this->worker*input;
            return ( this->worker*input ).reduce();// + this->bias;
        }       


        void dump(std::ostream& out) const
        {
            // for(size_t i=0;i<inputSize;++i)
            // {
            //     out.write((char*)&this->block[i],sizeof(block_t));
            // }
        }

        void load(std::istream& in)
        {
            // for(size_t i=0;i<inputSize;++i)
            // {
            //     in.read((char*)&this->block[i],sizeof(block_t));

            //     this->worker[i] = this->block[i].weights[this->block[i].id].weight;
            // }   
        }
        
    };
}