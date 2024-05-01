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

        const snn::SIMDVector& get_weights()
        {
            return this->input_weights;
        }

        void update_weights(const snn::SIMDVector& dweight)
        {
            this->input_weights+=dweight;
        }

        void setup(std::shared_ptr<Initializer> init)
        {
            this->input_weights.clear();
            init->init(this->input_weights,InputSize);

            init->init(this->biases);
        }

        void update(const SIMDVector& weight,const number& bias)
        {
            this->input_weights+=weight;
            this->biases+=bias;
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


    };
}