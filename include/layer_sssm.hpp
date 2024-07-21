#pragma once

#include <cmath>

#include <vector>
#include <functional>
#include <fstream>

#include <array>

#include "block.hpp"
#include "neuron.hpp"
#include "initializer.hpp"
#include "mutation.hpp"
#include "crossover.hpp"

#include "layer_proto.hpp"

#include "layer_st.hpp"

#include "simd_vector.hpp"

#include "config.hpp"

#include "activation.hpp"
#include "activation/linear.hpp"
#include "activation/silu.hpp"

#include "layer_utils.hpp"

#include "neurons/forwardneuron.hpp"
#include "neurons/neuronnobias.hpp"

namespace snn
{
    #define LAYERSSSM 2

    template<size_t InputSize,size_t HiddenStateSize>
    class LayerSSSM
    {
        const size_t deltaRank = ceil(InputSize/16);

        std::shared_ptr<Initializer> init;
        std::shared_ptr<Activation> activation_func;

        // it will generate delta , B and C, it is not using biases
        LayerST<NeuronNoBias<InputSize>> x_proj;

        // it projects delta from dt_rank ( size of delta vector ) to the input size 
        LayerST<ForwardNeuron<deltaRank>> d_proj;

        SIMDVector hidden_state;
        
        std::array<snn::SIMDVector,InputSize> hidden_state;

        public:

        LayerSSSM()
        {
            this->activation_func=std::make_shared<Linear>();
        }

        LayerSSSM(size_t N,std::shared_ptr<Initializer> init)
        {
            this->setup(N,init);
        }

        void setInitializer(std::shared_ptr<Initializer> init)
        {
            this->init=init;
        }

        void setActivationFunction(std::shared_ptr<Activation> active)
        {
            this->activation_func=active;
        }

        void setup(size_t N,std::shared_ptr<Initializer> init)
        {
            this->activation_func=std::make_shared<Linear>();

            this->init=init;


            this->x_proj->setup(deltaRank+HiddenStateSize*2,this->init);

            // there should be diffrent initializer I think, it will use silu activation
            this->d_proj->setup(InputSize,this->init);

            this->d_proj->setActivationFunction(std::make_shared<SiLu>());
        }

        void updateWeights(const std::vector<SIMDVector>& weights,const std::vector<number>& biases)
        {
            
        }

        SIMDVector fire(const SIMDVector& input)
        {
            SIMDVector dBC = this->x_proj->fire(input);

            // split it into delta, B and C

            SIMDVector d = dBC.extract(0,this->deltaRank);

            SIMDVector B = dBC.extract(this->deltaRank,this->deltaRank+HiddenStateSize);

            SIMDVector C = dBC.extract(this->deltaRank+HiddenStateSize,this->deltaRank+HiddenStateSize*2);


            SIMDVector e_d = this->d_proj->fire(d);

            SIMDVector output;

            this->activation_func->activate(output);

            return output;
        }

        size_t getTypeID()
        {
            return 2;
        };

        void generate_metadata(nlohmann::json& j) const
        {

        }

        int8_t load(std::ifstream& in)
        {
            return -1;
        }

        int8_t save(std::ofstream& out) const
        {
            return -1;
        }

    };  
    
} // namespace snn
