#pragma once

#include <iostream>
#include <thread>
#include <cstdint>

#include "evo_kan_block.hpp"
#include "simd_vector_lite.hpp"

#include "config.hpp"

namespace snn
{

    class EvoKanLayerProto
    {
        public:

        virtual size_t input_size() = 0;

        virtual size_t output_size() = 0;

        virtual void save(std::ostream& out) = 0;

        virtual void load(std::istream& in) = 0;

    };

    template<size_t InputSize,size_t OutputSize>
    class EvoKanLayer : public EvoKanLayerProto
    {

        static constexpr char* signature = "EVOKAN";

        protected:

        EvoKAN<InputSize> *blocks;

        snn::SIMDVectorLite<OutputSize> last_output;

        std::mutex fire_mutex;

        std::mutex fit_mutex;

        public:

        EvoKanLayer(){

            blocks = new EvoKAN<InputSize>[OutputSize];
        }

        size_t input_size()
        {
            return InputSize;
        }

        size_t output_size()
        {
            return OutputSize;
        }

        void fire_paraller(const snn::SIMDVectorLite<InputSize>& input,size_t& free_slot)
        {

            size_t index = 0;

            {
                std::lock_guard<std::mutex> guard(fire_mutex);

                index = free_slot;
                free_slot++;
            }

            while( index < OutputSize )
            {
                this->last_output[index] = this->blocks[index].fire(input);

                {
                    std::lock_guard<std::mutex> guard(fire_mutex);

                    index = free_slot;
                    free_slot++;
                }
            }

        }

        const snn::SIMDVectorLite<OutputSize>& fire(const snn::SIMDVectorLite<InputSize>& input)
        {

            std::thread threads[4];

            size_t free_slot = 0;

            for( size_t i = 0; i < 4; ++i )
            {
                threads[i] = std::thread(&EvoKanLayer<InputSize,OutputSize>::fire_paraller,this,std::cref(input),std::ref(free_slot));
            }

            for( size_t i = 0; i < 4; ++i )
            {
                if( threads[i].joinable() )
                {
                    threads[i].join();
                }
            }


            // for( size_t i = 0; i < OutputSize ; ++i )
            // {
            //     this->last_output[i] = this->blocks[i].fire(input);
            // }

            return this->last_output;
        }

        void fit_paraller(const snn::SIMDVectorLite<InputSize>& input,const snn::SIMDVectorLite<OutputSize>& target,size_t& free_slot)
        {

            size_t index = 0;

            {
                std::lock_guard<std::mutex> guard(fire_mutex);

                index = free_slot;
                free_slot++;
            }

            while( index < OutputSize )
            {
                this->blocks[index].fit(input,this->last_output[index],target[index]);

                {
                    std::lock_guard<std::mutex> guard(fire_mutex);

                    index = free_slot;
                    free_slot++;
                }
            }

        }

        void fit(const snn::SIMDVectorLite<InputSize>& input,const snn::SIMDVectorLite<OutputSize>& target)
        {

            std::thread threads[4];

            size_t free_slot = 0;

            for( size_t i = 0; i < 4; ++i )
            {
                threads[i] = std::thread(&EvoKanLayer<InputSize,OutputSize>::fit_paraller,this,std::cref(input),std::cref(target),std::ref(free_slot));
            }

            for( size_t i = 0; i < 4; ++i )
            {
                if( threads[i].joinable() )
                {
                    threads[i].join();
                }
            }

            // for( size_t i = 0; i < OutputSize ; ++i )
            // {
            //     this->blocks[i].fit(input,this->last_output[i],target[i]);
            // }
        }

        const snn::SIMDVectorLite<OutputSize>& get_last_output()
        {
            return this->last_output;
        }

        void save(std::ostream& out)
        {

            // add a metadata

            // a layer type
            out.write(EvoKanLayer::signature,strlen(EvoKanLayer::signature));

            // a number size in bytes
            char byte_size = sizeof(number);
            out.write(&byte_size,1);

            // for( const EvoKAN<InputSize>& block : this->blocks )
            for( size_t i=0; i<OutputSize; i++ )
            {
                this->blocks[i].save(out);
            }

        }

        void load(std::istream& in)
        {
            // read metadata

            const size_t sig_size = strlen(EvoKanLayer::signature);

            char sig_buffer[sig_size] = {0};

            in.read(sig_buffer,sig_size);

            if( strncmp(sig_buffer,EvoKanLayer::signature,sig_size) != 0 )
            {
                throw std::runtime_error("Invalid file layer signature!!!");
            }

            char byte_size = 0;

            in.read(&byte_size,1);

            if( byte_size != sizeof(number) )
            {
                throw std::runtime_error("Incompatible number data type size!!!");
            }

            // for( EvoKAN<InputSize>& block : this->blocks )
            for( size_t i=0; i<OutputSize; i++ )
            {
                this->blocks[i].load(in);
            }

        }

    };

}