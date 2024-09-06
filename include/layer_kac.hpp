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

#include "config.hpp"

#include "activation.hpp"
#include "activation/linear.hpp"

#include "layer_utils.hpp"

/*

 A layer that use evolutionary algorithm for learing called CoSyne.

*/
namespace snn
{
    #define STATICLAYERID 1

    template<size_t inputSize,size_t Populus>
    class LayerKAC : public LayerProto
    { 
        std::vector<BlockKAC<inputSize,Populus>> blocks;
        std::shared_ptr<Initializer> init;
        std::shared_ptr<Activation> activation_func;

        std::uniform_real_distribution<double> uniform;
        

        public:

        LayerKAC()
        {
            this->activation_func=std::make_shared<Linear>();
        }

        LayerKAC(size_t N,std::shared_ptr<Initializer> init,std::shared_ptr<Crossover> _crossing,std::shared_ptr<Mutation> _mutate)
        {
            this->setup(N,init,_crossing,_mutate);
        }

        void setInitializer(std::shared_ptr<Initializer> init)
        {
            this->init=init;
        }

        void setActivationFunction(std::shared_ptr<Activation> active)
        {
            this->activation_func=active;
        }


        void setup(size_t N,std::shared_ptr<Initializer> init,std::shared_ptr<Crossover> _crossing,std::shared_ptr<Mutation> _mutate)
        {
            this->uniform=std::uniform_real_distribution<double>(0.f,1.f);
            this->activation_func=std::make_shared<Linear>();
            this->blocks.clear();

            this->init=init;

            for(size_t i=0;i<N;++i)
            {
                this->blocks.push_back(BlockKAC<inputSize,Populus>(_mutate));
                this->blocks.back().setup(init);
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

        SIMDVector fire(const SIMDVector& input)
        {
            SIMDVector output;
            //output.reserve(this->blocks[0].outputSize());

            for(auto& block : this->blocks)
            {
                
                output.append(block.fire(input));

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
