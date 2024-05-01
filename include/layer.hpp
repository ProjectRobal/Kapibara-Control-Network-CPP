#pragma once

#include <vector>
#include <functional>
#include <fstream>

#include "block.hpp"
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

namespace snn
{
    #define STATICLAYERID 1

    template<class NeuronT,size_t Populus>
    class Layer : public LayerProto
    { 
        std::vector<Block<NeuronT,Populus>> blocks;
        std::shared_ptr<Initializer> init;
        std::shared_ptr<Activation> activation_func;

        public:

        Layer()
        {
            this->activation_func=std::make_shared<Linear>();
        }

        Layer(size_t N,std::shared_ptr<Initializer> init,std::shared_ptr<Crossover> _crossing,std::shared_ptr<Mutation> _mutate)
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
            this->activation_func=std::make_shared<Linear>();
            blocks.clear();

            this->init=init;

            for(size_t i=0;i<N;++i)
            {
                blocks.push_back(Block<NeuronT,Populus>(_crossing,_mutate));
                blocks.back().setup(init);
            }
        }

        void applyReward(long double reward)
        {
            reward/=this->blocks.size();

            for(auto& block : this->blocks)
            {
                block.giveReward(reward);
            }
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

                if(block.readyToMate())
                {
                    block.maiting(this->init);
                    std::cout<<"Layer maiting!"<<std::endl;
                }

            }

            this->activation_func->activate(output);

            return output;
        }

        

    };  
    
} // namespace snn
