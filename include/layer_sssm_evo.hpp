#pragma once

#include <cmath>

#include <vector>
#include <functional>
#include <fstream>

#include <array>

#include "initializer.hpp"
#include "crossover.hpp"

#include "layer_proto.hpp"

#include "layer_kac.hpp"

#include "simd_vector.hpp"

#include "config.hpp"

#include "activation.hpp"
#include "activation/linear.hpp"
#include "activation/silu.hpp"

#include "layer_utils.hpp"

#include "neurons/forwardneuron.hpp"
#include "neurons/neuronnobias.hpp"

#include "initializers/uniform.hpp"

#include "layer_proto_with_weights.hpp"

/*

 A Selective State Space Model but with evolutionary algorithm using CoSyne.

*/
namespace snn
{
    #define LAYERSSSM 2

    template<size_t InputSize,size_t N,size_t Populus,size_t deltaRank = static_cast<size_t>(ceil(InputSize/16.f))>
    class LayerSSSMEVO
    {

        std::shared_ptr<Initializer> init;
        std::shared_ptr<Activation> activation_func;

        // it will generate delta , B and C, it is not using biases
        LayerKAC<InputSize,deltaRank+N*2,Populus> x_proj;

        // it projects delta from dt_rank ( size of delta vector ) to the input size 
        LayerKAC<deltaRank,InputSize,Populus> d_proj;
        
        std::array<snn::SIMDVectorLite<N>,InputSize> hidden_state;

        std::array<snn::SIMDVectorLite<N>,InputSize> A;

        std::array<SIMDVector,InputSize> dB;

        size_t hiddenStateSize;

        size_t TicksToShuttle;

        size_t Ticks;

        public:

        LayerSSSMEVO()
        {
            this->activation_func=std::make_shared<Linear>();
        }

        LayerSSSMEVO(std::shared_ptr<Initializer> init,size_t TicksToShuttle=200,number dt_min = 0.001,number dt_max = 0.1)
        : LayerSSSMEVO()
        {
            this->setup(init,TicksToShuttle);
        }

        void shuttle()
        {
            this->Ticks++;
            
            // after some time planer weights should be shuttled
            if( this->Ticks >= this->TicksToReplacment)
            {

                this->hidden_planer.shuttle();
                this->planer.shuttle();

                this->Ticks = 0;
            }
        }

        void applyReward(long double reward)
        {
            this->x_proj.applyReward(reward);
            
            this->d_proj.applyReward(reward);
        }

        void setInitializer(std::shared_ptr<Initializer> init)
        {
            this->init=init;
        }

        void setActivationFunction(std::shared_ptr<Activation> active)
        {
            this->activation_func=active;
        }

        void setup(size_t N,std::shared_ptr<Initializer> init,size_t TicksToShuttle=200,number dt_min = 0.001,number dt_max = 0.1)
        {
            this->TicksToShuttle = TicksToShuttle;

            this->activation_func=std::make_shared<Linear>();

            this->init=init;

            this->hidden_state.fill(SIMDVector(0.f,N));

            this->dB.fill(SIMDVector(0.f,N));

            // SIMDVector a_vec([](size_t i)->number{

            //     return -static_cast<number>( i+1 );

            // },N);

            SIMDVectorLite<N> a_vec;

            this->A.fill(a_vec);


            this->x_proj.setup(deltaRank+N*2,this->init);

            number dt_init_std = std::pow(deltaRank,-0.5);
            std::shared_ptr<UniformInit> d_uniform = std::make_shared<UniformInit>(-dt_init_std,dt_init_std);

            // there should be diffrent initializer I think, it will use silu activation
            this->d_proj.setup(d_uniform);

            this->d_proj.setActivationFunction(std::make_shared<SiLu>());

            std::uniform_real_distribution<number> uniform(0.f,1.f);

            std::random_device rd; 

            // Mersenne twister PRNG, initialized with seed from previous random device instance
            std::mt19937 gen(rd()); 

            for(size_t i=0;i<InputSize;++i)
            {
                number d_bias = uniform(gen)*(std::log(dt_max)-std::log(dt_min)) + std::log(dt_min);

                this->d_proj.set_bias(d_bias,i);
            }

            this->hiddenStateSize=N;
        }

        size_t neuron_count() const
        {
            return 0;
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

                a_line*=this->hidden_state[x_i];

                x_i++;
            }

            x_i=0;    
            for( SIMDVector& b_line: this->dB )
            {
                b_line = B*input[x_i]*e_d[x_i];

                x_i++;
            }

            SIMDVector output;

            x_i=0;
            for( SIMDVector& h_line : this->hidden_state )
            {
                h_line = dA[x_i] + this->dB[x_i];

                output.append( (h_line*C).reduce() );

                x_i++;
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
