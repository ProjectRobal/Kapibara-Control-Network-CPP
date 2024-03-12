#pragma once

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

    #define ForwardNeuronID 2
    template<size_t InputSize>
    class ForwardNeuron : public Neuron
    {
        protected:

        SIMDVector input_weights;
        number biases;

        public:

        const size_t id=ForwardNeuronID;

        ForwardNeuron()
        : Neuron()
        {

        }

        std::shared_ptr<Neuron> crossover(std::shared_ptr<Crossover> cross,const Neuron& neuron)
        {
            const ForwardNeuron<InputSize>& forward=dynamic_cast<const ForwardNeuron<InputSize>&>(neuron);

            std::shared_ptr<ForwardNeuron<InputSize>> output=std::make_shared<ForwardNeuron<InputSize>>();

            output->input_weights=cross->cross(this->input_weights,forward.input_weights);
            output->biases=(this->biases+forward.biases)/2;

            return output;
        }

        void mutate(std::shared_ptr<Mutation> mutate)
        {
            mutate->mutate(this->input_weights);
            mutate->mutate(this->biases);

        }

        void setup(std::shared_ptr<Initializer> init)
        {
            this->input_weights.clear();
            init->init(this->input_weights,InputSize);

            init->init(this->biases);
        }

        number fire1(const SIMDVector& input)
        {
            number store=(input_weights*input).dot_product();

            return store + this->biases;
        }

        size_t input_size()
        {
            return InputSize;
        }

        size_t output_size()
        {
            return 1;
        }

        void save(std::ofstream& file) const override
        {
            // maybe the layer should specifi the size of Neuron

            NeuronHeader header=Neuron::getHeader(InputSize,1);

            file.write((char*)&header,sizeof(header));

            for(size_t i=0;i<this->input_weights.size();++i)
            {
                number num=this->input_weights[i];
                file.write((char*)&num,sizeof(number));
            }

            file.write((char*)&this->biases,sizeof(number));

        }

        bool load(std::ifstream& file) override
        {

            NeuronHeader header={0};

            file.read((char*)&header,sizeof(header));

            if(!this->validateHeader(header))
            {
                std::cerr<<"Invalid neuron header!"<<std::endl;
                return false;
            }

            if(header.input_size != InputSize)
            {
                std::cerr<<"Invalid neuron input size!"<<std::endl;
                return false;
            }

            if(header.output_size != 1)
            {
                std::cerr<<"Invalid neuron output size!"<<std::endl;
                return false;
            }

            char* num_buf = new char[sizeof(number)];
            number num;

            for(size_t i=0;i<InputSize;++i)
            {
                file.read(num_buf,sizeof(number));

                memmove((char*)&num,num_buf,sizeof(number));

                this->input_weights.append(num);

            }

            file.read(num_buf,sizeof(number));

            memmove((char*)&this->biases,num_buf,sizeof(number));


            return true;

        }

    };
}