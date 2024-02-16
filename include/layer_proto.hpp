#pragma once

/*
    A prototype class for all layer class.
*/

#include <fstream>

#include "simd_vector.hpp"

#include "neuron.hpp"
#include "initializer.hpp"
#include "crossover.hpp"
#include "mutation.hpp"

namespace snn
{

    class LayerProto
    {
        public:

        virtual void setup(size_t N,std::shared_ptr<Initializer> init,std::shared_ptr<Crossover> _crossing,std::shared_ptr<Mutation> _mutate)=0;

        virtual SIMDVector fire(const SIMDVector& input)=0;

        virtual void shuttle()=0;

        virtual void applyReward(long double reward)=0;

        virtual void keepWorkers()=0;

        virtual void applyRewardToSavedBlocks(long double reward)=0;

        virtual std::vector<std::shared_ptr<snn::Neuron>> getWorkingNeurons()=0;

        virtual void save(std::ofstream& file)=0;

        virtual bool load(std::ifstream& file)=0;

        virtual ~LayerProto()
        {};

    };

};
