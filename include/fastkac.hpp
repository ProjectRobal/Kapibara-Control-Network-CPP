#pragma once

#include <vector>
#include <functional>
#include <fstream>
#include <algorithm>
#include <random>

#include "neuron.hpp"
#include "initializer.hpp"
#include "mutation.hpp"
#include "crossover.hpp"

#include "neurons/dynamicneuron.hpp"

#include "layer_proto.hpp"

#include "simd_vector.hpp"

#include "config.hpp"

#include "activation.hpp"
#include "activation/linear.hpp"

#include "layer_utils.hpp"

namespace snn
{
    #define FASTKACID 1

    class FastKAC : public LayerProto
    { 
        std::shared_ptr<Initializer> init;
        std::shared_ptr<Activation> activation_func;
        std::shared_ptr<Mutation> mutate;

        std::vector<std::shared_ptr<DynamicNeuron<>>> hidden;

        std::vector<std::shared_ptr<DynamicNeuron<>>> outputs;

        // a vector used to hold current neurons states
        SIMDVector hidden_states;

        size_t inputSize;

        size_t highest_output_id;

        number clip_to_one(number v)
        {
            if( v > 1.f )
            {
                return 1.f;
            }
            if( v < -1.f )
            {
                return -1.f;
            }
            return v;
        }

        public:

        FastKAC()
        {
            this->activation_func=std::make_shared<Linear>();
            this->highest_output_id = 0;
        }

        FastKAC(size_t inputSize,size_t outputSize,std::shared_ptr<Initializer> init,std::shared_ptr<Mutation> _mutate,size_t initialHiddenSize=0)
        : FastKAC()
        {
            this->setup(inputSize,outputSize,init,_mutate,initialHiddenSize);
        }

        void setInitializer(std::shared_ptr<Initializer> init)
        {
            this->init=init;
        }

        void setActivationFunction(std::shared_ptr<Activation> active)
        {
            this->activation_func=active;
        }

        void setup(size_t inputSize,size_t outputSize,std::shared_ptr<Initializer> init,std::shared_ptr<Mutation> _mutate,size_t initialHiddenSize=0)
        {
            this->inputSize = inputSize;
            this->mutate = _mutate;

            this->outputs.clear();
            this->hidden.clear();
           
            this->activation_func=std::make_shared<Linear>();

            this->init=init;

            this->hidden_states = SIMDVector(0.f,inputSize+outputSize+initialHiddenSize);

            for(size_t i=0;i<outputSize;++i)
            {
                std::shared_ptr<DynamicNeuron<>> neuron = std::make_shared<DynamicNeuron<>>(inputSize+initialHiddenSize+outputSize);

                neuron->setup(init,inputSize);

                this->outputs.push_back(neuron);
            }

            for(size_t i=0;i<initialHiddenSize;++i)
            {
                std::shared_ptr<DynamicNeuron<>> neuron = std::make_shared<DynamicNeuron<>>(inputSize+initialHiddenSize+outputSize);

                neuron->setup(init,inputSize);

                this->hidden.push_back(neuron);
            }

        }

        void applyReward(long double reward)
        {
            // perform learning        

            std::random_device rd; 

            // Mersenne twister PRNG, initialized with seed from previous random device instance
            std::mt19937 gen(rd()); 

            std::uniform_int_distribution<size_t> uniform(0,this->inputSize+this->hidden.size()-1); 

            /*for(auto& neuron : this->outputs)
            {
                neuron->mutate(this->mutate);
            }

            for(auto& neuron : this->hidden)
            {
                neuron->mutate(this->mutate);
            }*/

            if(reward>0)
            {

                auto& output_neuron = this->outputs[this->highest_output_id];

                size_t node_id_to_connect = 0;

                
            }
            else if(reward<0)
            {
                auto& output_neuron = this->outputs[this->highest_output_id];

                number weight = this->hidden_states[0];
                size_t neuron_to_prune = 0;

                // nudge weights of hidden neuron
                // find the biggest hidden state

                for(size_t i=0;i<this->hidden_states.size();++i)
                {
                    if(this->hidden_states[i] > weight)
                    {
                        neuron_to_prune = i;
                        weight = this->hidden_states[i];
                    }
                }

                // ignore inputs
                if( neuron_to_prune >= this->inputSize )
                {
                    neuron_to_prune-=this->inputSize;

                    auto& hidden_neuron = this->hidden[neuron_to_prune];

                    snn::SIMDVector mix = hidden_neuron->get_weights()*this->hidden_states;

                    neuron_to_prune = 0;
                    weight = mix[0];

                    for(size_t i=0;i<mix.size();++i)
                    {
                        if(mix[i] > weight)
                        {
                            neuron_to_prune = i;  
                            weight = mix[i]; 
                        }
                    }

                    weight = hidden_neuron->get_weights()[neuron_to_prune];

                    hidden_neuron->set_weight(clip_to_one(weight*(1+reward)),neuron_to_prune);

                    hidden_states.set(hidden_neuron->fire1(hidden_states),neuron_to_prune+this->inputSize);
                }
                
                snn::SIMDVector mix = output_neuron->get_weights()*this->hidden_states;

                weight = mix[0];
                neuron_to_prune = 0;

                for(size_t i=0;i<mix.size();++i)
                {
                    if(mix[i] > weight)
                    {
                        neuron_to_prune = i;  
                        weight = mix[i]; 
                    }
                }

                weight = output_neuron->get_weights()[neuron_to_prune];

                output_neuron->set_weight(clip_to_one(weight*(1+reward)),neuron_to_prune);


            }
        }

        void shuttle()
        {
            // perform low probability mutation

        }

        SIMDVector fire(const SIMDVector& input)
        {
            SIMDVector output;
            //output.reserve(this->blocks[0].outputSize());

            for(size_t i=0;i<input.size();++i)
            {
                hidden_states.set(input[i],i);
            }

            // hidden states

            size_t h_ptr = input.size();
            for(const auto& neuron : this->hidden)
            {
                hidden_states.set(neuron->fire1(hidden_states),h_ptr);
                ++h_ptr;
            }

            // outputs
            for(const auto& neuron : this->outputs)
            {
                number out = neuron->fire1(hidden_states);
                output.append(out);
            }

            this->activation_func->activate(output);

            this->highest_output_id = 0;

            for(size_t i=0;i<output.size();++i)
            {
                hidden_states.set(output[i],h_ptr);

                if(output[i]>output[this->highest_output_id])
                {
                    this->highest_output_id=i;
                }

                ++h_ptr;
            }

            return output;
        }


        size_t getTypeID()
        {
            return FASTKACID;
        };

        void generate_metadata(nlohmann::json& j) const
        {
            j["input_size"]=this->inputSize;
            j["output_size"]=this->outputs.size();
        }

        int8_t load(std::ifstream& in)
        {

            /*for(auto& block : this->blocks)
            {
                if(in.good())
                {
                    block.load(in);
                }
                else
                {
                    return -1;
                }
            }*/

            return 0;
        }

        int8_t save(std::ofstream& out) const
        {
            
            /*for(const auto& block : this->blocks)
            {
                if(out.good())
                {
                    block.dump(out);
                }
                else
                {
                    return -1;
                }
            }*/

            return 0;
        }        

    };  
    
} // namespace snn
