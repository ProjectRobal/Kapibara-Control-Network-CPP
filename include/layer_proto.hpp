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

    class LayerProto
    {
        public:

        //virtual void setup(size_t N,std::shared_ptr<Initializer> init,std::shared_ptr<Crossover> _crossing,std::shared_ptr<Mutation> _mutate)=0;

        virtual SIMDVector fire(const SIMDVector& input)=0;

        virtual void shuttle(){};

        virtual void applyReward(long double reward){};

        virtual void setInitializer(std::shared_ptr<Initializer> init)=0;

        virtual void setActivationFunction(std::shared_ptr<Activation> active)=0;

        virtual void set_bias(number b,size_t id){}

        virtual void set_weights(const SIMDVector& vec,size_t id){}

        virtual number get_bias(size_t id) const =0;

        virtual const snn::SIMDVector& get_weights(size_t id) const =0;

        virtual size_t neuron_count() const
        {
            return 0;
        }

        virtual size_t getTypeID()
        {
            return 0;
        };

        virtual void generate_metadata(nlohmann::json& j) const=0;

        virtual int8_t load(std::ifstream& in)=0;

        virtual int8_t save(std::ofstream& out) const=0;

        virtual ~LayerProto()
        {};

    };

};
