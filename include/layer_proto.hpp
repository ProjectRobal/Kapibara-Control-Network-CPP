#pragma once

/*
    A prototype class for all layer class.
*/

#include <fstream>

#include "simd_vector.hpp"

#include "neuron.hpp"
#include "initializer.hpp"
#include "crossover.hpp"
#include "mutation.hpp"
#include "activation.hpp"

namespace snn
{

    class LayerProto
    {
        public:

        virtual void setup(size_t N,std::shared_ptr<Initializer> init,std::shared_ptr<Crossover> _crossing,std::shared_ptr<Mutation> _mutate)=0;

        virtual SIMDVector fire(const SIMDVector& input)=0;

        virtual void shuttle()=0;

        virtual void applyReward(long double reward)=0;

        virtual void setInitializer(std::shared_ptr<Initializer> init)=0;

        virtual void setActivationFunction(std::shared_ptr<Activation> active)=0;

        virtual ~LayerProto()
        {};

    };

};
