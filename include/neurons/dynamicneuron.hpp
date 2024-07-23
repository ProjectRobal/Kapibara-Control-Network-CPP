#pragma once

/*
 A type of neuron that can change it's input size dynamically.

*/

#include <iostream>
#include <memory>
#include <fstream>

#include "simd_vector.hpp"
#include "initializer.hpp"
#include "crossover.hpp"

#include "neuron.hpp"

#include "config.hpp"

namespace snn
{

    #define DynamicNeuronID 3
    template<size_t InitInputSize=0>
    class DynamicNeuron : public Neuron
    {
        protected:

        SIMDVector input_weights;
        number biases;

        size_t current_input_size;

        public:

        const size_t id=DynamicNeuronID;

        DynamicNeuron()
        : Neuron()
        {
            this->current_input_size = InitInputSize;
        }

        DynamicNeuron(size_t InputSize)
        : Neuron()
        {
            this->current_input_size = InputSize;
        }

        std::shared_ptr<Neuron> crossover(std::shared_ptr<Crossover> cross,const Neuron& neuron)
        {
            std::shared_ptr<DynamicNeuron> output=std::make_shared<DynamicNeuron>();

            output->input_weights=cross->cross(this->input_weights,neuron.get_weights());
            output->biases=(this->biases+neuron.get_bias())/2;

            return output;
        }

        void mutate(std::shared_ptr<Mutation> mutate)
        {
            mutate->mutate(this->input_weights);
            mutate->mutate(this->biases);

        }

        const snn::SIMDVector& get_weights() const
        {
            return this->input_weights;
        }

        void update_weights(const snn::SIMDVector& dweight)
        {
            this->input_weights+=dweight;
        }

        void set_weights(const snn::SIMDVector& dweight)
        {
            this->input_weights=dweight;
        }

        number get_bias() const
        {
            return this->biases;
        }

        void set_bias(number b)
        {
            this->biases=b;
        }

        void update_bias(number b)
        {
            this->biases+=b;
        }

        void save(std::ofstream& out) const
        {
            // save current neuron size
            char input_size_buffer[sizeof(size_t)];

            memmove(input_size_buffer,(char*)&this->current_input_size,sizeof(size_t));

            out.write(input_size_buffer,sizeof(size_t));

            for(size_t i=0;i<input_weights.size();++i)
            {
                char* data=snn::serialize_number<number>(input_weights[i]);

                out.write(data,SERIALIZED_NUMBER_SIZE);

                delete [] data;
            }
                
            // save neuron bias

            char* data=snn::serialize_number<number>(biases);

            out.write(data,SERIALIZED_NUMBER_SIZE);

            delete [] data;

            Neuron::save(out);
        }

        void load(std::ifstream& in)
        {
            this->input_weights.clear();

            // save current neuron size
            char input_size_buffer[sizeof(size_t)];

            in.read(input_size_buffer,sizeof(size_t));
            
            memmove((char*)&this->current_input_size,input_size_buffer,sizeof(size_t));

            char data[SERIALIZED_NUMBER_SIZE];

            // load weights
            for(size_t i=0;i<this->current_input_size;++i)
            {
                in.read(data,SERIALIZED_NUMBER_SIZE);

                number weight = snn::deserialize_number<number>(data);

                this->input_weights.append(weight);
            }

            // load biases
            in.read(data,SERIALIZED_NUMBER_SIZE);

            this->biases = snn::deserialize_number<number>(data);

            Neuron::load(in);
        }


        void setup(std::shared_ptr<Initializer> init)
        {
            this->input_weights.clear();
            init->init(this->input_weights,this->current_input_size);

            init->init(this->biases);
        }

        void setup(std::shared_ptr<Initializer> init,size_t count)
        {
            this->input_weights.clear();
            init->init(this->input_weights,count);

            for(size_t i=0;i<this->current_input_size-count;++i)
            {
                this->input_weights.append(0.f);
            }

            init->init(this->biases);
        }

        void add_weight(number num)
        {
            this->input_weights.append(num);
            this->current_input_size++;
        }

        void rm_weight()
        {
            if( this->current_input_size == 1 )
            {
                return;
            }
            this->input_weights.pop();
            this->current_input_size--;
        }

        void update(const SIMDVector& weight,number bias)
        {
            this->input_weights+=weight;
            this->biases+=bias;
        }

        void set_weight(number v,size_t i)
        {
            this->input_weights.set(v,i);
        }

        number get_weight(size_t i)
        {
            return this->input_weights[i];
        }

        number fire1(const SIMDVector& input)
        {
            number store=(input_weights*input).reduce();

            return store + this->biases;
        }

        size_t input_size()
        {
            return this->current_input_size;
        }

        size_t output_size()
        {
            return 1;
        }


    };
}