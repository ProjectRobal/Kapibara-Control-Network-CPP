#pragma once

#include <vector>
#include <functional>
#include <fstream>
#include <algorithm>

#include "block_kac.hpp"
#include "neuron.hpp"
#include "initializer.hpp"
#include "mutation.hpp"
#include "crossover.hpp"

#include "layer_proto.hpp"

#include "simd_vector.hpp"
#include "simd_vector_lite.hpp"

#include "config.hpp"

#include "activation.hpp"
#include "activation/linear.hpp"

#include "layer_utils.hpp"

#include <filesystem>

/*

 A layer that use evolutionary algorithm for learing called CoSyne.

*/
namespace snn
{
    #define STATICLAYERID 1

    template<size_t inputSize,size_t N,size_t Populus>
    class LayerKAC 
    { 
        std::vector<BlockKAC<inputSize,Populus>> blocks;
        std::shared_ptr<Activation> activation_func;

        std::uniform_real_distribution<double> uniform;
        
        public:

        LayerKAC()
        {
            this->activation_func=std::make_shared<Linear>();
        }

        void setActivationFunction(std::shared_ptr<Activation> active)
        {
            this->activation_func=active;
        }


        void setup()
        {
            this->uniform=std::uniform_real_distribution<double>(0.f,1.f);
            this->activation_func=std::make_shared<Linear>();
            this->blocks.clear();

            this->blocks.reserve(N);

            for(size_t i=0;i<N;++i)
            {
                this->blocks.push_back(BlockKAC<inputSize,Populus>());
                this->blocks.back().setup();
                // this->blocks.back().chooseWorkers();
            }
        }

        void applyReward(long double reward)
        {

            // reward/=this->blocks.size();

            for(auto& block : this->blocks)
            {
                block.giveReward(reward);
            }
        }

        size_t neuron_count(){
            return 0;
        }

        void shuttle()
        {
            
            for(auto& block : this->blocks)
            {
                block.chooseWorkers();
            }   
        }

        SIMDVectorLite<N> fire(const SIMDVectorLite<inputSize>& input)
        {
            SIMDVectorLite<N> output;
            //output.reserve(this->blocks[0].outputSize());

            size_t i=0;

            for(auto& block : this->blocks)
            {
                
                output.set(i++,block.fire(input));

            }

            this->activation_func->activate(output);

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
            
            for(const auto& block : this->blocks)
            {
                if(out.good())
                {
                    block.dump(out);
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
