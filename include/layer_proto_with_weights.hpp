#pragma once

/*
    A prototype class for all layer class.
*/


#include <iostream>
#include <fstream>

#include "simd_vector.hpp"

#include "neuron.hpp"
#include "initializer.hpp"
#include "crossover.hpp"
#include "mutation.hpp"
#include "activation.hpp"

#include "nlohmann/json.hpp"



namespace snn
{

    class LayerProtoWithWeights
    {
        public:

        //virtual void setup(size_t N,std::shared_ptr<Initializer> init,std::shared_ptr<Crossover> _crossing,std::shared_ptr<Mutation> _mutate)=0;

        virtual void set_bias(number b,size_t id){}

        virtual void set_weights(const SIMDVector& vec,size_t id){}

        virtual number get_bias(size_t id) const =0;

        virtual const snn::SIMDVector& get_weights(size_t id) const =0;

        virtual ~LayerProtoWithWeights()
        {};

    };

};
