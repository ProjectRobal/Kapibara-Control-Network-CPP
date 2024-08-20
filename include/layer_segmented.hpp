#pragma once

#include <vector>
#include <array>
#include <functional>
#include <fstream>
#include <algorithm>

#include "block_kac.hpp"
#include "neuron.hpp"
#include "initializer.hpp"
#include "mutation.hpp"
#include "crossover.hpp"

#include "layer_proto.hpp"
#include "layer_kac.hpp"

#include "simd_vector.hpp"

#include "config.hpp"

#include "initializers/hu.hpp"

#include "activation.hpp"
#include "activation/linear.hpp"
#include "activation/silu.hpp"
#include "activation/softmax.hpp"

#include "layer_utils.hpp"

/*

 A layer that is split into a sub KAC Layers and additional layer is presents and it chooses what sub layer should be active at that moment.

*/
namespace snn
{
    #define STATICLAYERID 1

    template<size_t inputSize,size_t blocksCount,size_t Populus,size_t HiddenSize=inputSize*2>
    class LayerSegmented : public LayerProto
    { 
        std::shared_ptr<Initializer> init;
        std::shared_ptr<Activation> activation_func;

        std::uniform_real_distribution<double> uniform;

        std::array<LayerKAC<inputSize,Populus>,blocksCount> blocks;

        // it will choose what block should be active now
        LayerKAC<inputSize,Populus> hidden_planer;
        LayerKAC<HiddenSize,Populus> planer;

        SiLu hidden_activation;
        SoftMax planer_activation;

        size_t current_block;

        size_t TicksToReplacment;
        size_t Ticks;

        public:

        LayerSegmented()
        {
            this->activation_func=std::make_shared<Linear>();

        }

        LayerSegmented(size_t N,std::shared_ptr<Initializer> init,std::shared_ptr<Crossover> _crossing,std::shared_ptr<Mutation> _mutate,size_t TicksToReplacment=200)
        {
            this->setup(N,init,_crossing,_mutate,TicksToReplacment);
        }

        void setInitializer(std::shared_ptr<Initializer> init)
        {
            this->init=init;
        }

        void setActivationFunction(std::shared_ptr<Activation> active)
        {
            this->activation_func=active;
        }

        void setup(size_t N,std::shared_ptr<Initializer> init,std::shared_ptr<Crossover> _crossing,std::shared_ptr<Mutation> _mutate,size_t TicksToReplacment=200)
        {
            this->current_block = 0;
            this->Ticks = 0;
            this->TicksToReplacment = TicksToReplacment;
            this->uniform=std::uniform_real_distribution<double>(0.f,1.f);
            this->activation_func=std::make_shared<Linear>();

            std::shared_ptr<HuInit> planer_init = std::make_shared<HuInit>();

            this->init=init;

            for(size_t i=0;i<blocksCount;++i)
            {
                this->blocks[i].setup(N,init,_crossing,_mutate);
            }

            this->hidden_planer.setup(HiddenSize,planer_init,_crossing,_mutate);
            this->planer.setup(blocksCount,planer_init,_crossing,_mutate);
        }

        void applyReward(long double reward)
        {
            this->hidden_planer.applyReward(reward);
            this->planer.applyReward(reward);

            this->blocks[this->current_block].applyReward(reward);
            // block will switch it's weights every step

            if( this->Ticks >= 1)
            {
                this->blocks[this->current_block].shuttle();

                this->Ticks = 0;
            }
        }

        number get_bias(size_t id) const
        {
            return 0;
        }

        const snn::SIMDVector& get_weights(size_t id) const
        {
            return SIMDVector();
        }

        size_t neuron_count(){
            return 0;
        }

        void shuttle()
        {
            // this->Ticks++;
            
            // // after some time planer weights should be shuttled
            // if( this->Ticks >= this->TicksToReplacment)
            // {

            this->hidden_planer.shuttle();
            this->planer.shuttle();

            // }
            
        }

        SIMDVector fire(const SIMDVector& input)
        {
            // choose most promenant
            // SIMDVector probs = this->hidden_planer.fire(input);
            // this->hidden_activation.activate(probs);

            // probs = this->planer.fire(probs);
            // this->planer_activation.activate(probs);

            // this->current_block = snn::get_action_id(probs);

            SIMDVector output;
            //output.reserve(this->blocks[0].outputSize())

            output = this->blocks[0].fire(input);

            this->activation_func->activate(output);

            this->Ticks ++;

            return output;
        }


        size_t getTypeID()
        {
            return STATICLAYERID;
        };

        void generate_metadata(nlohmann::json& j) const
        {
            j["input_size"]=inputSize;
            j["output_size"]=blocks.size();
        }

        int8_t load(std::ifstream& in)
        {
            hidden_planer.load(in);
            planer.load(in);

            for(auto& block : this->blocks)
            {
                if(in.good())
                {
                    block.load(in);
                }
                else
                {
                    return -1;
                }
            }

            return 0;
        }

        int8_t save(std::ofstream& out) const
        {
            hidden_planer.save(out);
            planer.save(out);
            
            for(const auto& block : this->blocks)
            {
                if(out.good())
                {
                    block.save(out);
                }
                else
                {
                    return -1;
                }
            }

            return 0;
        }        

    };  
    
} // namespace snn
