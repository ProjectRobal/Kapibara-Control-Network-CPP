#pragma once

#include <vector>
#include <thread>

#include <evo_kan_block.hpp>

#include <simd_vector_lite.hpp>
#include <config.hpp>

namespace snn
{

    template< size_t inputSize, size_t outputSize >
    class EvoKanLayer
    {
        protected:

        EvoKan<inputSize> *blocks;

        SIMDVectorLite<outputSize> output;


        static void fire_thread(EvoKan<inputSize> *blocks,const SIMDVectorLite<inputSize>&input,SIMDVectorLite<outputSize>& output,size_t& current_id,std::mutex& mux)
        {
            size_t id = 0;

            while( true )
            {
                {
                    std::lock_guard guard(mux);

                    id = current_id;

                    current_id++;

                    if( id >= outputSize )
                    {
                        return;
                    }
                }

                output[id] = blocks[id].fire(input);
            }
        }

        static void fit_thread(EvoKan<inputSize> *blocks,const SIMDVectorLite<inputSize>&input,const SIMDVectorLite<outputSize>& output,const SIMDVectorLite<outputSize>& target,size_t& current_id,std::mutex& mux)
        {
            size_t id = 0;

            number _target;
            number _output;

            while( true )
            {
                {
                    std::lock_guard guard(mux);

                    id = current_id;

                    current_id++;

                    if( id >= outputSize )
                    {
                        return;
                    }

                    _target = target[id];
                    _output = output[id];
                }

                blocks[id].fit(input,_output,_target);
            }
        }

        public:

        EvoKanLayer(const size_t initial_spline_size = 0)
        {
            this->blocks = new EvoKan<inputSize>[outputSize](initial_spline_size);
        }

        SIMDVectorLite<outputSize> fire(const SIMDVectorLite<inputSize>& input)
        {
            size_t current_id = 0;

            std::mutex id_lock;

            std::thread threads[THREAD_COUNT];

            for(size_t i=0;i<THREAD_COUNT;++i)
            {
                threads[i] = std::thread(EvoKanLayer::fire_thread,this->blocks,std::cref(input),std::ref(this->output),std::ref(current_id),std::ref(id_lock));
            }

            for(size_t i=0;i<THREAD_COUNT;++i)
            {
                if( threads[i].joinable() )
                {
                    threads[i].join();
                }
            }

            return this->output;

        }

        void fit(const SIMDVectorLite<inputSize>& input,const SIMDVectorLite<outputSize>& target)
        {
            size_t current_id = 0;

            std::mutex id_lock;

            std::thread threads[THREAD_COUNT];

            for(size_t i=0;i<THREAD_COUNT;++i)
            {
                threads[i] = std::thread(EvoKanLayer::fit_thread,this->blocks,std::cref(input),std::cref(this->output),std::cref(target),std::ref(current_id),std::ref(id_lock));
            }

            for(size_t i=0;i<THREAD_COUNT;++i)
            {
                if( threads[i].joinable() )
                {
                    threads[i].join();
                }
            }
            
        }

        ~EvoKanLayer()
        {
            delete [] this->blocks;
        }

    }; 

}