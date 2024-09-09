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

#include "network.hpp"

#include "serializaers/network_serialize.hpp"

#include "sub_block.hpp"

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

void send_fifo(const snn::SIMDVector& to_send)
{
    std::fstream fifo;

    fifo.open("fifo",std::ios::out);

    if(!fifo.good())
    {
        std::cerr<<"Cannot open fifo for writing"<<std::endl;
        return;
    }

    fifo<<to_send[0]<<";"<<to_send[1]<<std::endl;

    fifo.close();
}

snn::SIMDVector read_fifo()
{
    std::fstream fifo;

    fifo.open("fifo_in",std::ios::in);

    if(!fifo.good())
    {
        std::cerr<<"Cannot open fifo for reading"<<std::endl;
        return snn::SIMDVector();
    }

    snn::SIMDVector output;

    std::string line;

    std::getline(fifo,line);

    std::stringstream line_read(line);

    std::string num;

    while(std::getline(line_read,num,';'))
    {
        output.append(std::stof(num,NULL));
    }

    return output;

}

template<>
size_t snn::SubBlock<20,5>::SubBLockId = 0;


int main(int argc,char** argv)
{
    std::cout<<"Starting..."<<std::endl;

    std::shared_ptr<snn::NormalizedGaussInit> norm_gauss=std::make_shared<snn::NormalizedGaussInit>(0.f,0.01f);
    std::shared_ptr<snn::GaussInit> gauss=std::make_shared<snn::GaussInit>(0.f,0.25f);
    std::shared_ptr<snn::GaussInit> gauss1=std::make_shared<snn::GaussInit>(0.f,0.01f);
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
    auto first= std::make_shared<snn::LayerSegmented<inputSize,1,20>>(4,gauss1,cross,mutation);
    auto second= std::make_shared<snn::LayerSegmented<4,1,20>>(512,gauss1,cross,mutation);
    auto third= std::make_shared<snn::LayerSegmented<512,1,20>>(128,gauss1,cross,mutation);
    auto forth= std::make_shared<snn::LayerSegmented<512,1,20>>(2,gauss1,cross,mutation);

    first->setActivationFunction(std::make_shared<snn::ReLu>());
    second->setActivationFunction(std::make_shared<snn::ReLu>());
    third->setActivationFunction(std::make_shared<snn::ReLu>());
    forth->setActivationFunction(std::make_shared<snn::Linear>());

    auto ssm = std::make_shared<snn::LayerSSSM<inputSize>>(256,hu);

    ssm->setActivationFunction(std::make_shared<snn::Linear>());
    

    //layer3->setActivationFunction(relu);

    std::shared_ptr<snn::Network> network = std::make_shared<snn::Network>();

    // network->addLayer(ssm);
    network->addLayer(first);
    network->addLayer(second);
    //network->addLayer(third);
    network->addLayer(forth);

    snn::SIMDVector input;

    gauss->init(input,20);

    input = snn::SIMDVector(0,20);


    // std::cout<<input<<std::endl;

    // std::cout<<std::endl;

    // input = snn::pexp(input);

    // std::cout<<input<<std::endl;

    // std::cout<<snn::pexp(0)<<std::endl;

    // return 0;

    
    snn::SIMDVector output(0.f,8);

    output.set(1.f,2);

    end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_initialization_seconds = end - start;

    std::cout << "Finished initialization at " << elapsed_initialization_seconds.count() << std::endl;

    std::cout<<"Starting network"<<std::endl;

    long double best_reward = -9999999;


    while(true)
    {

        snn::SIMDVector cart_input = read_fifo();

        if( cart_input[5] > 0.5f)
        {

            if( cart_input[4] > best_reward )
            {
                best_reward = cart_input[4];

                std::cout<<"New best reward: "<<best_reward<<std::endl;
            }

            network->applyReward(cart_input[4]);

        }

        // I have to speed up it just a bit 
        // network->applyReward(cart_input[4]);

        //std::cout<<"From CartPole: "<<cart_input<<std::endl;

        snn::SIMDVector output = network->fire(cart_input);

        //std::cout<<"To CartPole: "<<output<<std::endl;

        send_fifo(output);

    }

    return 0;

}

