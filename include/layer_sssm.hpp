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

    template<size_t InputSize,size_t deltaRank = static_cast<size_t>(ceil(InputSize/16.f))>
    class LayerSSSM : public LayerProto
    {

        std::shared_ptr<Initializer> init;
        std::shared_ptr<Activation> activation_func;

        // it will generate delta , B and C, it is not using biases
        LayerST<NeuronNoBias<InputSize>> x_proj;

        // it projects delta from dt_rank ( size of delta vector ) to the input size 
        LayerST<ForwardNeuron<deltaRank>> d_proj;
        
        std::array<snn::SIMDVector,InputSize> hidden_state;

        std::array<snn::SIMDVector,InputSize> A;

        std::array<SIMDVector,InputSize> dB;

        size_t hiddenStateSize;

        public:

        LayerSSSM()
        {
            this->activation_func=std::make_shared<Linear>();
        }

        LayerSSSM(size_t N,std::shared_ptr<Initializer> init)
        : LayerSSSM()
        {
            this->setup(N,init);
        }

        void shuttle(){}

        void applyReward(long double reward){}

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

            this->hidden_state.fill(SIMDVector(0.f,N));

            this->dB.fill(SIMDVector(0.f,N));

            SIMDVector a_vec([](size_t i)->number{

                return -static_cast<number>( i+1 );

            },N);

            this->A.fill(a_vec);


            this->x_proj.setup(deltaRank+N*2,this->init);

            // there should be diffrent initializer I think, it will use silu activation
            this->d_proj.setup(InputSize,this->init);

            this->d_proj.setActivationFunction(std::make_shared<SiLu>());

            this->hiddenStateSize=N;
        }

        void updateWeights(const std::vector<SIMDVector>& weights,const std::vector<number>& biases)
        {
            
        }

        SIMDVector fire(const SIMDVector& input)
        {
            SIMDVector dBC = this->x_proj.fire(input);

            // split it into delta, B and C

            SIMDVector d = dBC.extract(0,deltaRank);

            SIMDVector B = dBC.extract(deltaRank,deltaRank+this->hiddenStateSize);

            SIMDVector C = dBC.extract(deltaRank+this->hiddenStateSize,deltaRank+this->hiddenStateSize*2);

            SIMDVector e_d = this->d_proj.fire(d);

            std::array<SIMDVector,InputSize> dA = this->A;

            // it should be exp(dA)
            size_t x_i=0;
            for( SIMDVector& a_line : dA )
            {
                //a_line*= this->hidden_state[x_i]*e_d[x_i++];

                a_line*=e_d[x_i];

                a_line.set(1.f + a_line[x_i],x_i);

                a_line*=this->hidden_state[x_i++];
            }

            x_i=0;    
            for( SIMDVector& b_line: this->dB )
            {
                b_line = B*input[x_i]*e_d[x_i++];
            }

            SIMDVector output;

            x_i=0;
            for( SIMDVector& h_line : this->hidden_state )
            {
                h_line = dA[x_i] + this->dB[x_i];

                output.append( (h_line*C).reduce() );
            }

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
