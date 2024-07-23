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

    #define NeuronNoBiasID 2
    template<size_t InputSize>
    class NeuronNoBias : public Neuron
    {
        protected:

        SIMDVector input_weights;

        public:

        const size_t id=NeuronNoBiasID;

        NeuronNoBias()
        : Neuron()
        {

        }

        std::shared_ptr<Neuron> crossover(std::shared_ptr<Crossover> cross,const Neuron& neuron)
        {
            std::shared_ptr<NeuronNoBias<InputSize>> output=std::make_shared<NeuronNoBias<InputSize>>();

            output->input_weights=cross->cross(this->input_weights,neuron.get_weights());

            return output;
        }

        void mutate(std::shared_ptr<Mutation> mutate)
        {
            mutate->mutate(this->input_weights);

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
            return 0;
        }

        void set_bias(number b)
        {

        }

        void update_bias(number b)
        {

        }

        void save(std::ofstream& out) const
        {

            for(size_t i=0;i<input_weights.size();++i)
            {
                char* data=snn::serialize_number<number>(input_weights[i]);

                out.write(data,SERIALIZED_NUMBER_SIZE);

                delete [] data;
            }

            Neuron::save(out);
        }

        void load(std::ifstream& in)
        {
            this->input_weights.clear();

            char data[SERIALIZED_NUMBER_SIZE];

            // load weights
            for(size_t i=0;i<InputSize;++i)
            {
                in.read(data,SERIALIZED_NUMBER_SIZE);

                number weight = snn::deserialize_number<number>(data);

                this->input_weights.append(weight);
            }

            Neuron::load(in);
        }


        void setup(std::shared_ptr<Initializer> init)
        {
            this->input_weights.clear();
            init->init(this->input_weights,InputSize);

        }

        void update(const SIMDVector& weight,number bias)
        {
            this->input_weights+=weight;
        }

        number fire1(const SIMDVector& input)
        {
            number store=(input_weights*input).reduce();

            return store;
        }

        size_t input_size()
        {
            return InputSize;
        }

        size_t output_size()
        {
            return 1;
        }


    };
}