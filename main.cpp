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
// #include "layer_sssm.hpp"
// #include "layer_sssm_evo.hpp"

#include "fastkac.hpp"

#include "activation/sigmoid.hpp"
#include "activation/relu.hpp"
#include "activation/softmax.hpp"

#include "shared_mem.hpp"

#include "network.hpp"

#include "serializaers/network_serialize.hpp"

#include "sub_block.hpp"

#include "simd_vector_lite.hpp"

/*

 To save on memory we can store weights on disk and then load it to ram as a buffer.

  load 32 numbers from file.
  when execution is made load another 32 numbers in background.

  Each kac block will have line with N weights , each representing each population, plus one id with indicate what weight is choosen. In file we store
  weight with coresponding reward.
  When executing operation we will load each weight from each population.

 Each weights is going to have it's own population. We choose weight from population.
 Active weight gets reward, the lower reward is the higher probability of replacing weight,
 between 0.01 to 0.5 . 
 
 Sometimes the weights in population are going to be replace, the worst half of poplulation.
Some weights will be random some are going to be generated from collection of best weights, plus mutations.
The wieghts that achived positive rewards are collected and then used as replacment with some mutations
maybe. 
 Or to save on space we can generate new weights just using random distribution.

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

template<size_t Size>
void send_fifo(const snn::SIMDVectorLite<Size>& to_send)
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

snn::SIMDVectorLite<6> read_fifo_static()
{
    std::fstream fifo;

    fifo.open("fifo_in",std::ios::in);

    if(!fifo.good())
    {
        std::cerr<<"Cannot open fifo for reading"<<std::endl;
        return snn::SIMDVectorLite<6>();
    }

    snn::SIMDVectorLite<6> output;

    std::string line;

    std::getline(fifo,line);

    std::stringstream line_read(line);

    std::string num;

    size_t i=0;

    while(std::getline(line_read,num,';'))
    {
        output[i] = std::stof(num,NULL);

        i+=1;
    }

    return output;

}

template<>
size_t snn::SubBlock<20,5>::SubBLockId = 0;


int main(int argc,char** argv)
{
    std::cout<<"Starting..."<<std::endl;

    std::chrono::time_point<std::chrono::system_clock> start, end;

    // we can compress FFT data from 512 to 24 therotically
    const size_t inputSize = 38;

    start = std::chrono::system_clock::now();

    // the network will be split into layer that will be split into block an additional network will choose what block should be active in each step.
    auto first = snn::LayerKAC<4,1024,20>();
    auto second= snn::LayerKAC<1024,256,20>();
    // // auto third=snn::LayerKAC<512,256,20>();
    auto forth= snn::LayerKAC<256,2,20>();

    first.setup();
    second.setup();
    forth.setup();

    first.setActivationFunction(std::make_shared<snn::Linear>());
    second.setActivationFunction(std::make_shared<snn::Linear>());
    // // third->setActivationFunction(std::make_shared<snn::ReLu>());
    forth.setActivationFunction(std::make_shared<snn::Linear>());

    // auto ssm = std::make_shared<snn::LayerSSSM<inputSize>>(256,hu);

    // ssm->setActivationFunction(std::make_shared<snn::Linear>());
    

    //layer3->setActivationFunction(relu);

    // std::shared_ptr<snn::Network> network = std::make_shared<snn::Network>();

    // // network->addLayer(ssm);
    // network->addLayer(first);
    // network->addLayer(second);
    // //network->addLayer(third);
    // network->addLayer(forth);
    // std::cout<<input<<std::endl;

    // std::cout<<std::endl;

    // input = snn::pexp(input);

    // std::cout<<input<<std::endl;

    // std::cout<<snn::pexp(0)<<std::endl;

    // snn::SIMDVectorLite<36> simd1;
    // snn::SIMDVectorLite<36> simd2;

    // std::cout<<(simd1*simd2).reduce()<<std::endl;

    // // return 0;

    // char x;
    // std::cin>>x;

    // return 0;

    end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_initialization_seconds = end - start;

    std::cout << "Finished initialization at " << elapsed_initialization_seconds.count() << std::endl;

    std::cout<<"Starting network"<<std::endl;

    long double best_reward = -9999999;


    while(true)
    {

        snn::SIMDVectorLite<6> cart_input = read_fifo_static();

        if( cart_input[5] > 0.5f)
        {

            if( cart_input[4] > best_reward )
            {
                best_reward = cart_input[4];

                std::cout<<"New best reward: "<<best_reward<<std::endl;
            }

            // network->applyReward(cart_input[4]);
            first.applyReward(cart_input[4]);
            second.applyReward(cart_input[4]);
            forth.applyReward(cart_input[4]);

            first.shuttle();
            second.shuttle();
            forth.shuttle();

        }

        snn::SIMDVectorLite<4> input;

        input[0] = cart_input[0];
        input[1] = cart_input[1];
        input[2] = cart_input[2];
        input[3] = cart_input[3];

        // I have to speed up it just a bit 
        // network->applyReward(cart_input[4]);
        

        //std::cout<<"From CartPole: "<<cart_input<<std::endl;

        //snn::SIMDVector output = network->fire(cart_input);

        std::cout<<"From CartPole: "<<input<<std::endl;

        snn::SIMDVectorLite output1 = first.fire(input);
        snn::SIMDVectorLite output2 = second.fire(output1);

        snn::SIMDVectorLite output3 = forth.fire(output2);

        std::cout<<"To CartPole: "<<output3<<std::endl;

        std::cout<<std::endl;

        send_fifo(output3);

    }

    return 0;

}

