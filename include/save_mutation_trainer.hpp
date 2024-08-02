#pragma once

#include <iostream>

#include <memory>

#include "simd_vector.hpp"


#include "layer_proto.hpp"
#include "initializer.hpp"


#include "network.hpp"


namespace snn
{
    class SaveMutationTrainer
    {

        std::shared_ptr<Network> _network;
        std::shared_ptr<Initializer> _init;

        number mse(const SIMDVector& a,const SIMDVector& b)
        {
            SIMDVector d = a-b;

            return (d*d).reduce()/d.size();
        }        

        public:

        SaveMutationTrainer(std::shared_ptr<Network> _network,std::shared_ptr<Initializer> _init)
        {
            this->_network = _network;
            this->_init = _init;
        }

        void setInitializer(std::shared_ptr<Initializer> _init)
        {
            this->_init = _init;
        }

        void fit(SIMDVector input,SIMDVector desire_output,size_t layer_id)
        {
            std::shared_ptr<LayerProto> layer = this->_network->getLayers()[layer_id];

            size_t neuron_count = layer->neuron_count();

            size_t weights_count = layer->get_weights(0).size();

            number velocity = 0.1f;
            number last_step = 0.1f;

            

            number last_errr = this->mse(this->_network->fire(input),desire_output);

            SIMDVector biases;
            SIMDVector weights;

            // copy of current layers weights
            for(size_t i=0;i<neuron_count;++i)
            {
                biases.append(layer->get_bias(i));
                weights.extend(layer->get_weights(i));
            }

            std::vector<SIMDVector> bias_population;
            std::vector<SIMDVector> weight_population;

            number best_reward = this->mse(this->_network->fire(input),desire_output);
            size_t best_population = -1;

            bias_population.push_back(biases);
            weight_population.push_back(weights);

            // generate population
            for(size_t i=0;i<2;++i)
            {
                SIMDVector mut_weights;
                SIMDVector mut_biases;

                this->_init->init(mut_weights,neuron_count*weights_count);
                this->_init->init(mut_biases,neuron_count);

                bias_population.push_back(bias_population.back()+mut_biases);
                weight_population.push_back(weight_population.back()+mut_weights);

                const SIMDVector& _weight = weight_population.back();
                const SIMDVector& _biases = bias_population.back();
                
                for(size_t n=0;n<neuron_count;++n)
                {
                    layer->set_bias(_biases[n],n);
                    layer->set_weights(_weight.extract(n*weights_count,(n+1)*weights_count),n);
                }

                number reward = this->mse(this->_network->fire(input),desire_output);

                if(reward<best_reward)
                {
                    best_reward = reward;
                    best_population = i;
                    std::cout<<"Error: "<<reward<<std::endl;
                }
            }

            if(best_population!=-1)
            {
                for(size_t n=0;n<neuron_count;++n)
                {
                    layer->set_bias(bias_population[best_population][n],n);
                    layer->set_weights(weight_population[best_population].extract(n*weights_count,(n+1)*weights_count),n);
                }
            }
            else
            {
                for(size_t n=0;n<neuron_count;++n)
                {
                    layer->set_bias(biases[n],n);
                    layer->set_weights(weights.extract(n*weights_count,(n+1)*weights_count),n);
                }
            }

        }


    };
}