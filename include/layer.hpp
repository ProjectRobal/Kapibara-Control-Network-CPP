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

    template<class NeuronT,size_t Working,size_t Populus>
    class Layer : public LayerProto
    { 
        std::vector<Block<NeuronT,Working,Populus>> blocks;
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
                blocks.push_back(Block<NeuronT,Working,Populus>(_crossing,_mutate));
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

        void applyRewardToSavedBlocks(long double reward)
        {
            reward/=this->blocks.size();

            for(auto& block : this->blocks)
            {
                block.giveRewardToSavedWorkers(reward);
            }
        }

        void keepWorkers()
        {
            for(auto& block : this->blocks)
            {
                
                block.keepWorkers();

            }   
        }

        void shuttle()
        {
            
            for(auto& block : this->blocks)
            {
                
                block.chooseWorkers();

            }   
        }

        std::vector<std::shared_ptr<snn::Neuron>> getWorkingNeurons()
        {
            std::vector<std::shared_ptr<snn::Neuron>> workers;

            for(auto& block : this->blocks)
            {
                auto workers_arr=block.getWorkers();

                for(auto& worker : workers_arr)
                {
                    workers.push_back(worker);
                }
            }

            return workers;
        }

        SIMDVector fire(const SIMDVector& input)
        {
            SIMDVector output;
            //output.reserve(this->blocks[0].outputSize());

            for(auto& block : this->blocks)
            {
                
                output.extend(block.fire(input));

                if(block.readyToMate())
                {
                    block.maiting(this->init);
                    std::cout<<"Layer maiting!"<<std::endl;
                }

            }

            this->activation_func->activate(output);

            return output;
        }

        void save(std::ofstream& file)
        {
            LayerHeader header;

            header.id=STATICLAYERID;
            header.input_size=blocks[0].inputSize();
            header.output_size=blocks[0].outputSize();

            file.write((char*)&header,sizeof(header));

            for(const auto& block : this->blocks)
            {
                block.save(file);
            }
        }

        bool load(std::ifstream& file)
        {
            LayerHeader header={0};

            file.read((char*)&header,sizeof(header));

            if(strcmp("KAC",header.header)!=0)
            {
                std::cerr<<"Invalid header!"<<std::endl;
                return false;
            }

            if(header.id != STATICLAYERID)
            {
                std::cerr<<"Invalid layer format!"<<std::endl;
                return false;
            }

            for(auto& block : this->blocks)
            {
                if(!block.load(file))
                {
                    std::cerr<<"Block corrupted"<<std::endl;
                    return false;
                }
            }

            return true;
        }

    };  
    
} // namespace snn
