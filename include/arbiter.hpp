#pragma once

/*
    A class responsible for saving/loading layers 

*/

#include <memory>
#include <iostream>
#include <list>
#include <cstdint>

#include "layer.hpp"

namespace snn
{

    class Arbiter
    {
        std::list<std::shared_ptr<Layer>> layers;

        public:

        void addLayer(std::shared_ptr<Layer> layer)
        {
            layers.push_back(layer);
        }

        void setup()
        {
            for(std::shared_ptr<Layer> layer : this->layers)
            {
                layer->setup();
            }
        }

        void shuttle()
        {
            for(std::shared_ptr<Layer> layer : this->layers)
            {
                layer->shuttle();
            }   
        }

        void applyReward(long double reward)
        {
            for(std::shared_ptr<Layer> layer : this->layers)
            {
                layer->applyReward(reward);
            }
        }

        int8_t save() const
        {
            for(std::shared_ptr<Layer> layer : this->layers)
            {
                int8_t ret = layer->save();

                if( ret != 0 )
                {
                    return ret;
                }
            }

            return 0;
        }

        int8_t save(std::ostream& out) const
        {
            for(std::shared_ptr<Layer> layer : this->layers)
            {
                int8_t ret = layer->save(out);

                if( ret != 0 )
                {
                    return ret;
                }
            }

            return 0;
        }

        int8_t load() const
        {
            for(std::shared_ptr<Layer> layer : this->layers)
            {
                int8_t ret = layer->load();

                if( ret != 0 )
                {
                    return ret;
                }
            }

            return 0;
        }

        int8_t load(std::istream& in) const
        {
            for(std::shared_ptr<Layer> layer : this->layers)
            {
                int8_t ret = layer->load(in);

                if( ret != 0 )
                {
                    return ret;
                }
            }

            return 0;
        }
    };

}