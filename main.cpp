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

#include "layer_kac.hpp"

#include "layer_st.hpp"
#include "layer.hpp"
#include "layer_sssm.hpp"
#include "layer_segmented.hpp"
#include "layer_sssm_evo.hpp"

#include "fastkac.hpp"

#include "activation/sigmoid.hpp"
#include "activation/relu.hpp"
#include "activation/softmax.hpp"

#include "shared_mem.hpp"

#include "proto/cartpole.pb.h"

#include "network.hpp"

#include "serializaers/network_serialize.hpp"


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
    std::cout<<"Starting..."<<std::endl;

    std::shared_ptr<snn::NormalizedGaussInit> norm_gauss=std::make_shared<snn::NormalizedGaussInit>(0.f,0.01f);
    std::shared_ptr<snn::GaussInit> gauss=std::make_shared<snn::GaussInit>(0.f,0.25f);
    std::shared_ptr<snn::ConstantInit> constant=std::make_shared<snn::ConstantInit>(0.1f);
    std::shared_ptr<snn::UniformInit> uniform=std::make_shared<snn::UniformInit>(0.f,1.f);
    std::shared_ptr<snn::HuInit> hu=std::make_shared<snn::HuInit>();

    std::shared_ptr<snn::GaussMutation> mutation=std::make_shared<snn::GaussMutation>(0.f,0.01f,0.5f);
    std::shared_ptr<snn::OnePoint> cross=std::make_shared<snn::OnePoint>();

    std::chrono::time_point<std::chrono::system_clock> start, end;

    // we can compress FFT data from 512 to 24 therotically
    const size_t inputSize = 38;

    start = std::chrono::system_clock::now();

    // the network will be split into layer that will be split into block an additional network will choose what block should be active in each step.
    auto first= std::make_shared<snn::LayerSegmented<inputSize,5,10>>(128,hu,cross,mutation);
    auto second= std::make_shared<snn::LayerSegmented<128,5,10>>(128,hu,cross,mutation);
    auto third= std::make_shared<snn::LayerSegmented<128,5,10>>(128,hu,cross,mutation);
    auto forth= std::make_shared<snn::LayerSegmented<128,5,10>>(64,hu,cross,mutation);

    first->setActivationFunction(std::make_shared<snn::ReLu>());
    second->setActivationFunction(std::make_shared<snn::ReLu>());
    third->setActivationFunction(std::make_shared<snn::ReLu>());
    forth->setActivationFunction(std::make_shared<snn::SoftMax>());

    auto ssm = std::make_shared<snn::LayerSSSM<inputSize>>(256,hu);

    ssm->setActivationFunction(std::make_shared<snn::Linear>());
    

    //layer3->setActivationFunction(relu);

    std::shared_ptr<snn::Network> network = std::make_shared<snn::Network>();

    // network->addLayer(ssm);
    network->addLayer(first);
    network->addLayer(second);
    network->addLayer(third);
    network->addLayer(forth);

    snn::SIMDVector input;

    gauss->init(input,inputSize);
    
    snn::SIMDVector output(0.f,8);

    output.set(1.f,2);

    end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_initialization_seconds = end - start;

    std::cout << "Finished initialization at " << elapsed_initialization_seconds.count() << std::endl;

    std::cout<<"Starting network"<<std::endl;

    

    //std::cout<<"Input: "<<input<<std::endl;
    //std::cout<<"Output: "<<network->fire(input)<<std::endl;

    for(size_t i=0;i<20;++i)
    {
        start = std::chrono::system_clock::now();

        output = network->fire(input);

        network->applyReward(2.f);

        end = std::chrono::system_clock::now();

        std::chrono::duration<double> elapsed_seconds = end - start;

        std::cout << "finished computation at " << elapsed_seconds.count() << std::endl;
        std::cout<<"Output: "<<output<<std::endl;

    }

//    trainer.fit(input,output,1);

    

    return 0;

}

