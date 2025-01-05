#pragma once

/*
    A class responsible for saving/loading layers 

*/

#include <memory>
#include <mutex>
#include <iostream>
#include <list>
#include <vector>
#include <cstdint>

#include <thread>

#include "layer.hpp"

namespace snn
{

    class Arbiter
    {
        std::vector<std::shared_ptr<Layer>> layers;

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

        static void shuttle_thread(const std::vector<std::shared_ptr<Layer>>& layers,size_t start,size_t end)
        {
            while(start<end)
            {
                layers[start]->shuttle();
                start++;
            }
        }

        static void concurrent_shuttle(const std::vector<std::shared_ptr<Layer>>& layers,size_t ptr,size_t& free_slot,std::mutex& mux)
        {
            while(ptr<layers.size())
            {
                layers[ptr]->shuttle();

                std::lock_guard _mux(mux);

                ptr = free_slot;

                free_slot++;

                if(free_slot>=layers.size())
                {
                    return;
                }

            }
        }

        void shuttle()
        {
            

            size_t step = this->layers.size()/USED_THREADS;
            
            if( step == 0 )
            {
                for(std::shared_ptr<Layer> layer : this->layers)
                {
                    layer->shuttle();
                }

                return;
            }

            std::thread workers[USED_THREADS];

            size_t free_slot;

            std::mutex _mux;

            _mux.lock();

            for(size_t i=0;i<USED_THREADS;++i)
            {
                workers[i] = std::thread(concurrent_shuttle,std::ref(this->layers),i,std::ref(free_slot),std::ref(_mux));
            }

            free_slot = USED_THREADS;

            _mux.unlock();

            

            for(auto& worker : workers)
            {
                worker.join();   
            }

            std::cout<<"Free slot: "<<free_slot<<std::endl;
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