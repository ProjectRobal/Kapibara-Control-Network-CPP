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

    #define FeedForwardNeuronID 1

    template<size_t Input,size_t Output>
    class FeedForwardNeuron : public Neuron
    {
        

        protected:

        SIMDVector input_weights;
        SIMDVector output_weights;
        SIMDVector biases;

        public:

        const size_t id=FeedForwardNeuronID;

        FeedForwardNeuron()
        : Neuron()
        {

        }

        std::shared_ptr<Neuron> crossover(std::shared_ptr<Crossover> cross,const Neuron& neuron)
        {
            const FeedForwardNeuron<Input,Output>& forward=dynamic_cast<const FeedForwardNeuron<Input,Output>&>(neuron);

            std::shared_ptr<FeedForwardNeuron<Input,Output>> output=std::make_shared<FeedForwardNeuron<Input,Output>>();

            output->input_weights=cross->cross(this->input_weights,forward.input_weights);
            output->output_weights=cross->cross(this->output_weights,forward.output_weights);
            output->biases=cross->cross(this->biases,forward.biases);

            return output;
        }

        void mutate(std::shared_ptr<Mutation> mutate)
        {
            mutate->mutate(this->input_weights);
            mutate->mutate(this->output_weights);
            mutate->mutate(this->biases);
        }

        void setup(std::shared_ptr<Initializer> init)
        {
            this->input_weights.clear();
            init->init(this->input_weights,Input);

            this->output_weights.clear();
            init->init(this->output_weights,Output);

            this->biases.clear();
            init->init(this->biases,Output);
        }

        SIMDVector fire(const SIMDVector& input)
        {
            number store=(input_weights*input).dot_product();

            return output_weights*store + this->biases;
        }

        size_t input_size()
        {
            return Input;
        }

        size_t output_size()
        {
            return Output;
        }

        void save(std::ofstream& file) const override
        {
            // maybe the layer should specifi the size of Neuron

            NeuronHeader header=Neuron::getHeader(Input,Output);

            file.write((char*)&header,sizeof(header));

            for(size_t i=0;i<this->input_weights.size();++i)
            {
                number num=this->input_weights[i];
                file.write((char*)&num,sizeof(number));
            }

            for(size_t i=0;i<this->output_weights.size();++i)
            {
                number num=this->output_weights[i];
                file.write((char*)&num,sizeof(number));
            }

            for(size_t i=0;i<this->biases.size();++i)
            {
                number num=this->biases[i];
                file.write((char*)&num,sizeof(number));
            }

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

            if(header.input_size != Input)
            {
                std::cerr<<"Invalid neuron input size!"<<std::endl;
                return false;
            }

            if(header.output_size != Output)
            {
                std::cerr<<"Invalid neuron output size!"<<std::endl;
                return false;
            }

            char* num_buf = new char[sizeof(number)];
            number num;

            for(size_t i=0;i<Input;++i)
            {
                file.read(num_buf,sizeof(number));

                memmove((char*)&num,num_buf,sizeof(number));

                this->input_weights.append(num);

            }

            for(size_t i=0;i<Output;++i)
            {
                file.read(num_buf,sizeof(number));

                memmove((char*)&num,num_buf,sizeof(number));

                this->output_weights.append(num);
                
            }

            for(size_t i=0;i<Output;++i)
            {
                file.read(num_buf,sizeof(number));

                memmove((char*)&num,num_buf,sizeof(number));

                this->biases.append(num);
                
            }

            return true;

        }

    };
}