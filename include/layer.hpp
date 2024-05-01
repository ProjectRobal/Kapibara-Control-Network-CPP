#pragma once

#include <vector>
#include <functional>
#include <fstream>
#include <algorithm>

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

        // a probability of neuron repleacment
        double E;

        std::uniform_real_distribution<double> uniform;

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
            // a low probability at start
            this->E=1.f;
            this->uniform=std::uniform_real_distribution<double>(0.f,1.f);
            this->activation_func=std::make_shared<Linear>();
            this->blocks.clear();

            this->init=init;

            for(size_t i=0;i<N;++i)
            {
                this->blocks.push_back(Block<NeuronT,Populus>(_crossing,_mutate));
                this->blocks.back().setup(init);
                this->blocks.back().chooseWorkers();
            }
        }

        void applyReward(long double reward)
        {
            /*this->E=std::max((-reward/300.f),(long double)0.f)+0.05;

            if(this->E>1.f)
            {
                this->E=1.f;
            }*/

            reward/=this->blocks.size();

            for(auto& block : this->blocks)
            {
                block.giveReward(reward);
            }
        }

        void shuttle()
        {

            std::random_device rd; 

            // Mersenne twister PRNG, initialized with seed from previous random device instance
            std::mt19937 gen(rd()); 

            
            for(auto& block : this->blocks)
            {
                if(this->uniform(gen)<=this->E)
                {
                    block.chooseWorkers();
                }
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
                    //std::cout<<"Layer maiting!"<<std::endl;
                }

            }

            this->activation_func->activate(output);

            return output;
        }

        

    };  
    
} // namespace snn
