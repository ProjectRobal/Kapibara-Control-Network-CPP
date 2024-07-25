#include <experimental/simd>
#include <iostream>
#include <string_view>
#include <cmath>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <numeric>
#include <fstream>
#include <sys/stat.h>
#include <fcntl.h>

#include "parallel.hpp"

#include "config.hpp"

#include "simd_vector.hpp"

#include "neurons/forwardneuron.hpp"

#include "crossovers/onepoint.hpp"
#include "crossovers/fastuniform.hpp"
#include "crossovers/fastonepoint.hpp"
 
#include "initializers/gauss.hpp"
#include "initializers/normalized_gauss.hpp"
#include "initializers/constant.hpp"
#include "initializers/uniform.hpp"
#include "initializers/hu.hpp"

#include "mutatiom/gauss_mutation.hpp"

#include "parallel.hpp"

#include "block.hpp"
#include "layer.hpp"

#include "layer_st.hpp"
#include "layer.hpp"
#include "layer_sssm.hpp"

#include "fastkac.hpp"

#include "activation/sigmoid.hpp"
#include "activation/relu.hpp"
#include "activation/softmax.hpp"

#include "shared_mem.hpp"

#include "proto/cartpole.pb.h"

#include "network.hpp"

#include "serializaers/network_serialize.hpp"

#include "save_mutation_trainer.hpp"


/*

 Our theory is right but only when input values are positive,
 the other problem is exploding gradient

 but it has some potential in discreate problems.

 when reward is positive the last action is fortified.
 when reward is negative that last action is dumped.

 decreasing values works fine but increasing don't work quite well.

*/

size_t get_action_id(const snn::SIMDVector& actions)
{
    std::random_device rd; 

    // Mersenne twister PRNG, initialized with seed from previous random device instance
    std::mt19937 gen(rd()); 

    std::uniform_real_distribution<number> uniform_chooser(0.f,1.f);

    number shift = 0;

    number choose = uniform_chooser(gen);

    size_t action_id = 0;

    for(size_t i=0;i<actions.size();++i)
    {
        if( choose <= actions[i] + shift )
        {
            action_id = i;
            break;
        }

        shift += actions[i];
    }

    return action_id;
}


int main(int argc,char** argv)
{

    std::shared_ptr<snn::NormalizedGaussInit> norm_gauss=std::make_shared<snn::NormalizedGaussInit>(0.f,0.01f);
    std::shared_ptr<snn::GaussInit> gauss=std::make_shared<snn::GaussInit>(0.f,0.25f);
    std::shared_ptr<snn::ConstantInit> constant=std::make_shared<snn::ConstantInit>(0.1f);
    std::shared_ptr<snn::UniformInit> uniform=std::make_shared<snn::UniformInit>(0.f,1.f);
    std::shared_ptr<snn::HuInit> hu=std::make_shared<snn::HuInit>();

    std::shared_ptr<snn::GaussMutation> mutation=std::make_shared<snn::GaussMutation>(0.f,0.01f,0.5f);
    std::shared_ptr<snn::OnePoint> cross=std::make_shared<snn::OnePoint>();

    auto first= std::make_shared<snn::LayerST<snn::ForwardNeuron<16>>>(512,hu);
    auto second= std::make_shared<snn::LayerST<snn::ForwardNeuron<512>>>(8,hu);

    first->setActivationFunction(std::make_shared<snn::SiLu>());
    second->setActivationFunction(std::make_shared<snn::SoftMax>());

    auto ssm = std::make_shared<snn::LayerSSSM<16>>(64,hu);

    ssm->setActivationFunction(std::make_shared<snn::Linear>());
    

    //layer3->setActivationFunction(relu);

    std::shared_ptr<snn::Network> network = std::make_shared<snn::Network>();

    network->addLayer(ssm);
    network->addLayer(first);
    network->addLayer(second);

    snn::SIMDVector input;

    gauss->init(input,16);

    snn::SaveMutationTrainer trainer(network,gauss);
    
    snn::SIMDVector output(0.f,8);

    output.set(1.f,2);

    for(size_t i=0;i<10;++i)
    {

        std::cout<<"Input: "<<input<<std::endl;
        std::cout<<"Output: "<<network->fire(input)<<std::endl;

        trainer.fit(input,output,1);

        std::cout<<"Output*: "<<network->fire(input)<<std::endl;

    }

    return 0;

}

