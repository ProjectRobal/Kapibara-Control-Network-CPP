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

#include <simd_vector_lite.hpp>

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

 
        // there will be inputsize amount of block
        typedef struct block
        {
            uint32_t id;
            SIMDVectorLite<Populus> weights;

        } block_t;

        block_t block[inputSize];

        // now since each sub block gets the same reward we will just them a pointer to it.
        long double reward;
        long double last_reward;

        long double reward_integral;

        public:

        size_t Id;

        BlockKAC()
        {
            this->reward = 0.f;

            this->last_reward = 0.f;

            this->reward_integral = 0.f;

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

                for( size_t w=0; w < Populus ; ++w )
                {
                    block.weights[w] = this->global.init();
                }

                // this->worker.set(i,block.weights[block.id].weight);

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

        
        void chooseWorkers()
        {
            if(this->Id==1)
            {
                std::cout<<"Choose workers"<<std::endl;
            }
            
            if( this->reward == 0.0 )
            {
                return;
            }

            size_t iter=0;

            float switch_probability = 0.0;

            switch_probability = std::min<float>(REWARD_TO_SWITCH_PROBABILITY*abs(this->reward),0.25f);

            for(block_t& _block : this->block)
            {
                if( this->uniform(this->gen) < switch_probability )
                {

                    if( this->reward < 0.0 )
                    {

                        if(this->Id==1)
                        {
                            std::cout<<"Swap occured"<<std::endl;
                            std::cout<<"with probability "<<switch_probability<<std::endl;
                        }

                        SIMDVectorLite<Populus> error = _block.weights[_block.id] - _block.weights;

                        // for(size_t i=0;i<Populus;++i)
                        // {
                        //     number error = _block.weights[_block.id] - _block.weights[i];

                        //     _block.weights[i] -= 0.5*error;
                        // }

                        _block.id = ( static_cast<uint32_t>(std::round(this->uniform(this->gen)*(Populus-2))) + _block.id ) % ( Populus - 1 );


                        this->worker[iter] = _block.weights[_block.id];
                        
                    }
                    else if( this->reward > 0.0 )
                    {
                        if(this->Id == 1)
                        {
                            std::cout<<"Positive feedback! "<<std::endl;
                        }
                        
                        // move some weights towards better solutions
                        SIMDVectorLite<Populus> error = _block.weights[_block.id] - _block.weights;

                        _block.weights += 0.5*error;

                    }

                }

                iter++;

            }

            this->reward = 0;
        }

        void giveReward(long double reward) 
        {          
            // this->reward += ( reward + REWARD_DIFFERENCE_GAIN*(reward - this->last_reward) );

            this->reward += reward;

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
                char buffer[sizeof(size_t)];

                memcpy(buffer,(char*)&this->block[i].id,sizeof(size_t));

                out.write(buffer,sizeof(size_t));

                for(size_t w=0;w<Populus;++w)
                {
                    char *num_buffer = serialize_number(this->block[i].weights[w]);   

                    out.write(num_buffer,SERIALIZED_NUMBER_SIZE);

                    delete [] num_buffer;
                }

            }
        }

        void load(std::istream& in)
        {
            for(size_t i=0;i<inputSize;++i)
            {
                in.read((char*)&this->block[i].id,sizeof(size_t));

                char num_buffer[SERIALIZED_NUMBER_SIZE];

                for(size_t w=0;w<Populus;++w)
                {
                    in.read(num_buffer,SERIALIZED_NUMBER_SIZE);

                    number num = deserialize_number<number>(num_buffer);

                    this->block[i].weights[w] = num;
                }

                this->worker[i] = this->block[i].weights[this->block[i].id];
            }   
        }
        
    };
}