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

        // there will be inputsize amount of block
        typedef struct block
        {
            uint32_t id;
            number weights[Populus];
        } block_t;

        SIMDVectorLite<inputSize> best_weights;

        block_t block[inputSize];

        // now since each sub block gets the same reward we will just them a pointer to it.
        long double reward;

        size_t collectd_weights;

        public:

        size_t Id;

        BlockKAC()
        {
            this->reward = 0.f;

            std::random_device rd;

            this->gen = std::mt19937(rd());

            this->uniform = std::uniform_real_distribution<float>(0.f,1.f);

            this->worker = SIMDVectorLite<inputSize>(0);

            this->best_weights = SIMDVectorLite<inputSize>(0);

            this->Id = BlockCounter::BlockID;

            this->collectd_weights = 0;

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
                // iter = static_cast<size_t>(std::round(this->uniform(this->gen)*(inputSize/4 + 1)));

                // we only update network partially

                this->best_weights*= this->collectd_weights;

                this->best_weights += this->worker;

                this->collectd_weights++;

                this->best_weights /= this->collectd_weights;

                if( this->collectd_weights == 10 )
                {

                    this->best_weights /= this->collectd_weights;

                    this->collectd_weights = 1;
                }
                
                return;
            }

            float switch_probability = std::min<float>(REWARD_TO_SWITCH_PROBABILITY*this->reward,MAX_SWITCH_PROBABILITY);

            // size_t step = inputSize/(static_cast<size_t>(switch_probability*inputSize) + 1);

            size_t step = static_cast<size_t>(1.0/switch_probability);

            size_t start = std::min(step,inputSize-1);

            size_t i = static_cast<size_t>(std::round(this->uniform(this->gen)*start));

            uint32_t block_step = static_cast<uint32_t>(std::round(this->uniform(this->gen)*Populus));

            // size_t mutation_counter = static_cast<size_t>(std::round(this->uniform(this->gen)*9.f))+1;

            size_t mutation_counter = 5;

            size_t mutat_counter = mutation_counter;

            for(;i<inputSize;i+=step)
            {

                block_t& _block = this->block[i];

                if( mutat_counter == 0 )
                {
                    number mutation = this->global.init()/10.f; 
                    
                    if( mutat_counter == 0 || mutat_counter == 2 || mutat_counter == 4 )
                    {

                        _block.weights[_block.id] = 0.5f*_block.weights[_block.id] + 0.5f*this->best_weights[i];

                    }
                    else
                    {
                        _block.weights[_block.id] += mutation;
                    }

                    mutat_counter = mutation_counter;
                    
                }

                mutat_counter -- ;

                _block.id += (block_step+i);

                _block.id = _block.id % Populus;

                this->worker[i] = _block.weights[_block.id];

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
            // save best weights

            out.write((const char*)&this->collectd_weights,sizeof(size_t));

            for(size_t i=0;i<inputSize;++i)
            {
                char* buff = serialize_number<number>(this->best_weights[i]);

                out.write(buff,SERIALIZED_NUMBER_SIZE);

                delete [] buff;
            }

            // save blocks
            for(size_t i=0;i<inputSize;++i)
            {
                out.write((char*)&this->block[i],sizeof(block_t));
            }
        }

        void load(std::istream& in)
        {
            // load best weights
            in.read((char*)&this->collectd_weights,sizeof(size_t));

            char buff[SERIALIZED_NUMBER_SIZE];

            for(size_t i=0;i<inputSize;++i)
            {
                in.read(buff,SERIALIZED_NUMBER_SIZE);

                number num = deserialize_number<number>(buff);

                this->best_weights[i] = num;
            }


            // load blocks
            for(size_t i=0;i<inputSize;++i)
            {
                in.read((char*)&this->block[i],sizeof(block_t));

                this->worker[i] = this->block[i].weights[this->block[i].id];
            }   
        }
        
    };
}