#pragma once

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

        }


    };
}