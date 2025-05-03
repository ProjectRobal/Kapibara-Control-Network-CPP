#pragma once

#include <deque>
#include <cmath>
#include <iostream>
#include <cstdint>

#include "simd_vector_lite.hpp"
#include "layer.hpp"
#include "layer_kac.hpp"
#include "block_kac.hpp"

#include "initializers/hu.hpp"


namespace snn
{
    template<size_t InputSize,size_t ActionCount,size_t PopulationSize = 20>
    class Attention : public Layer
    {

        snn::LayerKAC<InputSize,2,PopulationSize> conv;

        // add information about position
        snn::LayerKAC<InputSize*2 + ActionCount*2,1,PopulationSize> attention;

        std::deque<snn::SIMDVectorLite<InputSize>> last_actions;

        snn::SIMDVectorLite<InputSize*2 + ActionCount*2> clear_mask;

        BlockKAC<InputSize,PopulationSize,HuInit<InputSize>> W1;

        BlockKAC<InputSize,PopulationSize,HuInit<InputSize>> W2;

        public:

        snn::SIMDVectorLite<InputSize> process(const snn::SIMDVectorLite<InputSize>& input)
        {   
            // push current input to buffer
            // this->last_actions.pop_back()

            this->last_actions.push_front(input);

            if(this->last_actions.size()>ActionCount)
            {
                this->last_actions.pop_back();
            }


            snn::SIMDVectorLite<InputSize*2 + ActionCount*2> pair(0);


            snn::SIMDVectorLite<InputSize> output(0);


            number score_sum = 0;

            size_t action_pos = 0;

            // calculate attention for each pair
            for(const auto& action : this->last_actions)
            {
                for(size_t i=0;i<InputSize;++i)
                {
                    pair[i] = action[i];
                }
                
                size_t action1_pos = 0;

                for(const auto& action1 : this->last_actions)
                {
                    for(size_t j=0;j<InputSize;++j)
                    {
                        pair[j+InputSize] = action1[j];
                    }

                    pair = pair * this->clear_mask;

                    pair[action_pos + InputSize] = 1;

                    pair[action1_pos + InputSize*2 + ActionCount] = 1;


                    snn::SIMDVectorLite<InputSize> out = this->W1.mult(action) + this->W2.mult(action1);

                    number score_out = std::exp(this->attention.fire(pair)[0]);

                    output += score_out*out;

                    score_sum += score_out;
                    
                    action1_pos++;
                }

                action_pos++;
            }


            snn::SIMDVectorLite<InputSize> calculated_attention = output/score_sum;


            return calculated_attention;
        }

        void setup()
        {
            this->clear_mask = snn::SIMDVectorLite<InputSize*2 + ActionCount*2>(0);

            // we are going to use that mask to clear information of position encoding
            for(size_t i=0;i<InputSize;++i)
            {
                this->clear_mask[i] = 1;
            }

            for(size_t i=0;i<InputSize;++i)
            {
                this->clear_mask[i+ActionCount+InputSize] = 1;
            }

            conv.setup();
            attention.setup();

            W1.setup();
            W2.setup();

        }

        void applyReward(long double reward)
        {
            conv.applyReward(reward);
            attention.applyReward(reward);

            W1.giveReward(reward);
            W2.giveReward(reward);
        }

        void shuttle()
        {
            conv.shuttle();
            attention.shuttle();

            W1.chooseWorkers();
            W2.chooseWorkers();
        }

        int8_t load()
        {
            int8_t ret = 0;

            ret = conv.load();

            if(ret!=0)
            {
                return ret;
            }

            ret = attention.load();

            if(ret!=0)
            {
                return ret;
            }

            return 0;

        }

        int8_t save() const
        {
            int8_t ret = 0;

            ret = conv.save();

            if(ret!=0)
            {
                return ret;
            }

            ret = attention.save();

            if(ret!=0)
            {
                return ret;
            }

            return 0;
        }

        int8_t load(std::istream& in)
        {
            int8_t ret = 0;

            ret = conv.load(in);

            if(ret!=0)
            {
                return ret;
            }

            ret = attention.load(in);

            if(ret!=0)
            {
                return ret;
            }

            W1.load(in);
            W2.load(in);

            return 0;
        }

        int8_t save(std::ostream& out) const
        {
            int8_t ret = 0;

            ret = conv.save(out);

            if(ret!=0)
            {
                return ret;
            }

            ret = attention.save(out);

            if(ret!=0)
            {
                return ret;
            }

            W1.dump(out);
            W2.dump(out);

            return 0;
        }

    };

}