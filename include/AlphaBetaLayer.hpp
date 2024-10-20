#pragma once

#include <cstdint>

#include "simd_vector_lite.hpp"
#include "layer.hpp"
#include "layer_kac.hpp"

#include "activation/silu.hpp"
#include "activation/exp.hpp"
#include "activation/linear.hpp"

#include "initializers/uniform.hpp"

#include "config.hpp"

/*
    An recurrent network layer. Inspired by Mamba but a bit simpler. We will just use two vectors A and B for hidden state estimation:

    h(t) = _A*h(t-1) + _B*x(t)

    Where A and B are generated by linear network. Then _A = A / ( A+B ) and _B = B / ( A+B ).

    Architecture:

   | inputs | -> | KAC Layer with SiLu |  -> | Hidden State | -> | KAC Layer with SiLu |

*/

namespace snn
{
    template<size_t InputSize,size_t HiddenStateSize,size_t OutputSize,size_t Populus>
    class AlphaBetaLayer : public Layer
    {
        // resize inputs to hidden state size
        LayerKAC<InputSize,HiddenStateSize,Populus,SiLu> input_layer;

        // estimate alpha and beta vectors, use initialization with positive value
        LayerKAC<InputSize,HiddenStateSize,Populus,Linear> alpha_layer;
        LayerKAC<InputSize,HiddenStateSize,Populus,Linear> beta_layer;

        // translate hidden state to output size
        LayerKAC<HiddenStateSize,OutputSize,Populus,SiLu> output_layer;

        SIMDVectorLite<HiddenStateSize> hidden_state;

        public:

        AlphaBetaLayer()
        : hidden_state(0)
        {}

        void setup()
        {
            this->input_layer.setup();
            this->alpha_layer.setup();
            this->beta_layer.setup();
            this->output_layer.setup();
        }

        void reset()
        {
            this->hidden_state = SIMDVectorLite<HiddenStateSize>(0);
        }

        snn::SIMDVectorLite<OutputSize> fire(const snn::SIMDVectorLite<InputSize>& input)
        {
            snn::SIMDVectorLite resized = this->input_layer.fire(input);

            snn::SIMDVectorLite<HiddenStateSize> alpha = this->alpha_layer.fire(input);

            snn::SIMDVectorLite<HiddenStateSize> beta = this->beta_layer.fire(input);

            // snn::SIMDVectorLite<HiddenStateSize> ab = alpha+beta;

            this->hidden_state = alpha * this->hidden_state + beta *resized;

            snn::SIMDVectorLite<OutputSize> output = this->output_layer.fire(this->hidden_state); 

            return output;
        }

        void applyReward(long double reward)
        {

            this->input_layer.applyReward(reward);
            this->alpha_layer.applyReward(reward);
            this->beta_layer.applyReward(reward);
            this->output_layer.applyReward(reward);

        }

        void shuttle()
        {
            this->input_layer.shuttle();
            this->alpha_layer.shuttle();
            this->beta_layer.shuttle();
            this->output_layer.shuttle();
        }

        int8_t load()
        {
            if( this->input_layer.load() < 0 || this->alpha_layer.load() < 0 || this->beta_layer.load() < 0 || this->output_layer.load() < 0 )
            {
                return -1;
            }

            return 0;
        }

        int8_t save() const
        {
            if( this->input_layer.save() < 0 || this->alpha_layer.save() < 0 || this->beta_layer.save() < 0 || this->output_layer.save() < 0 )
            {
                return -1;
            }

            return 0;
        }

        int8_t load(std::istream& in)
        {
            if( this->input_layer.load(in) < 0 || this->alpha_layer.load(in) < 0 || this->beta_layer.load(in) < 0 || this->output_layer.load(in) < 0 )
            {
                return -1;
            }

            return 0;
        }

        int8_t save(std::ostream& out) const
        {
            if( this->input_layer.save(out) < 0 || this->alpha_layer.save(out) < 0 || this->beta_layer.save(out) < 0 || this->output_layer.save(out) < 0 )
            {
                return -1;
            }

            return 0;
        }
    };

}