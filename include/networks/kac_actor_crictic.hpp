#pragma once

/*

    This network will be made from two sub networks a fist is 

*/

#include <memory>
#include <vector>

#include <climits>

#include "simd_vector.hpp"

#include "layer_proto.hpp"
#include "layer.hpp"

#include "neurons/feedforwardneuron.hpp"
#include "neurons/forwardneuron.hpp"


namespace snn
{
    template<size_t input_size,size_t output_size,size_t population_size,size_t N>
    class ActorCriticNetwork
    {
        protected:

        std::shared_ptr<LayerProto> oracle;
        std::shared_ptr<LayerProto> oracle_out;

        std::vector<std::shared_ptr<LayerProto>> decision;

        snn::SIMDVector last_oracle_inputs;

        long double best_estimated_reward;

        size_t OptionsNumber;

        public:

        ActorCriticNetwork()
        {

        }

        ActorCriticNetwork(size_t OptionsNumber,std::shared_ptr<Initializer> init,std::shared_ptr<Crossover> _crossing,std::shared_ptr<Mutation> _mutate)
        {
            this->setup(OptionsNumber,init,_crossing,_mutate);
        }

        void addLayer(std::shared_ptr<LayerProto> layer)
        {
            this->decision.push_back(layer);
        }

        void setup(size_t OptionsNumber,std::shared_ptr<Initializer> init,std::shared_ptr<Crossover> _crossing,std::shared_ptr<Mutation> _mutate)
        {
            this->OptionsNumber=OptionsNumber;
            // Oracle will try to estimate reward for pairs of input and action, so input size is input_size+output_size and output size is just 1
            this->oracle=std::make_shared<snn::Layer<ForwardNeuron<input_size+output_size>,1,population_size>>(N,init,_crossing,_mutate);
            this->oracle_out=std::make_shared<snn::Layer<ForwardNeuron<N>,1,population_size>>(1,init,_crossing,_mutate);

            this->oracle->shuttle();
            this->oracle_out->shuttle();
        }

        void applyReward(const long double& reward)
        {
            long double _reward=-abs(reward-best_estimated_reward);
            long double output=0;

            // run this loop as long as it will return appropiate reward it is quite slow to be honset, I think I should use some kind of replay buffer
            // then we will have our short-term memory ( Replay buffer ) and long term memory ( Actor cirtic neural netowrk )
            // 
            // A working cycle will look like this:
            // (Input+Output, Reward) pair will be append to Replay buffer
            // When Replay Buffer reached size of N it will be flushed
            // In the background Actor Critic Network will learn based on Replay Buffer data ( using some kind of swarming )
            // when Replay buffer is flushed, Actor Critic Network will try to fit data
            // 
            // Decision network can retrive data from Replay buffer using aproximation so it can get data, only if samples are close enough
         
            this->oracle->shuttle();
            this->oracle_out->shuttle();

            snn::SIMDVector oracle_input =this->oracle->fire(oracle_input);

            output = this->oracle_out->fire(oracle_input)[0];

            this->oracle->applyReward(_reward);
            this->oracle_out->applyReward(_reward);
        }

        SIMDVector step(const SIMDVector& input)
        {

            this->last_oracle_inputs=input;

            this->best_estimated_reward=LONG_MIN;

            SIMDVector best_action;

            for(size_t i=0;i<this->OptionsNumber;++i)
            {
                // choose a network
                for(auto& layer : this->decision)
                {
                    layer->shuttle();
                }

                SIMDVector output=input;

                for(auto& layer : this->decision)
                {
                    output=layer->fire(output);
                }

                // estimate reward for pair of output 

                SIMDVector oracle_input=input;

                oracle_input.extend(output);

                oracle_input=this->oracle->fire(oracle_input);

                long double reward = this->oracle_out->fire(oracle_input)[0];

                for(auto& layer : this->decision)
                {
                    layer->applyReward(reward);
                }

                if(reward>this->best_estimated_reward)
                {
                    best_action=std::move(output);
                    this->best_estimated_reward=reward;

                    for(auto& layer : this->decision)
                    {
                        layer->keepWorkers();
                    }
                }

            }

            std::cout<<"Best estimated reward: "<<best_estimated_reward<<std::endl;

            return best_action;
        }


    };

};