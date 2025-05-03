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


#include "config.hpp"

#include "simd_vector.hpp"

#include "layer_kac.hpp"
 
#include "initializers/gauss.hpp"
#include "initializers/constant.hpp"
#include "initializers/uniform.hpp"
#include "initializers/hu.hpp"

// #include "layer_sssm.hpp"
// #include "layer_sssm_evo.hpp"

#include "activation/sigmoid.hpp"
#include "activation/relu.hpp"
#include "activation/softmax.hpp"
#include "activation/silu.hpp"


#include "simd_vector_lite.hpp"

#include "layer_counter.hpp"

#include "arbiter.hpp"

#include "kapibara_sublayer.hpp"

#include "RResNet.hpp"

#include "attention.hpp"

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

#include "block_kac.hpp"

size_t snn::BlockCounter::BlockID = 0;

size_t snn::LayerCounter::LayerIDCounter = 0;

/*

    KapiBara input variables:

    quanterion - 4 values
    speed from encoders - 2 values
    spectogram 16x16 - 256 values
    2d points array from camera, compressed to 16x16 - 256 values
    face embeddings - 64 values when more than two faces are spotted average thier embeddings

    Total 518 values


*/



int main(int argc,char** argv)
{
    std::cout<<"Starting..."<<std::endl;

    std::chrono::time_point<std::chrono::system_clock> start, end;

    snn::Arbiter arbiter;


    snn::SIMDVectorLite<582> input;

    snn::UniformInit<0.f,1.f> uni;

    for(size_t i=0;i<input.size();++i)
    {
        input[i] = uni.init();
    }

    std::cout<<"Running network"<<std::endl;

    std::shared_ptr<snn::Attention<582,256,64,20>> attention = std::make_shared<snn::Attention<582,256,64,20>>();

    attention->setup();

    auto output1_ = attention->process(input);
    auto output2_ = attention->process(input);
    auto output3_ = attention->process(input);
    auto output4_ = attention->process(input);

    start = std::chrono::system_clock::now();


    auto output = attention->process(input);
    

    end = std::chrono::system_clock::now();

    auto pairwaise_time = static_cast<std::chrono::duration<double>>(end - start).count();

    std::cout<<"Pairwaise time: "<<pairwaise_time<<std::endl;

    std::cout<<output[0]<<std::endl;


    std::shared_ptr<snn::LayerKAC<256,4096,20>> layer1 = std::make_shared<snn::LayerKAC<256,4096,20>>();

    std::shared_ptr<snn::LayerKAC<4096,2048,20,snn::ReLu>> layer2 = std::make_shared<snn::LayerKAC<4096,2048,20,snn::ReLu>>();

    std::shared_ptr<snn::LayerKAC<2048,512,20,snn::ReLu>> layer3 = std::make_shared<snn::LayerKAC<2048,512,20,snn::ReLu>>();

    std::shared_ptr<snn::LayerKAC<512,256,20,snn::ReLu>> layer4 = std::make_shared<snn::LayerKAC<512,256,20,snn::ReLu>>();

    std::shared_ptr<snn::LayerKAC<256,64,20>> layer5 = std::make_shared<snn::LayerKAC<256,64,20>>();

    layer1->setup();
    layer2->setup();
    layer3->setup();
    layer4->setup();
    layer5->setup();


    start = std::chrono::system_clock::now();

    auto output1 = layer1->fire(output);
    auto output2 = layer2->fire(output1);
    auto output3 = layer3->fire(output2);
    auto output4 = layer4->fire(output3);
    auto output5 = layer5->fire(output4);

    for(size_t i=0;i<64;++i)
    {
        output5[i] = std::exp(output5[i]);
    }

    output5 = output5 / output5.reduce();

    end = std::chrono::system_clock::now();

    std::cout<<"Time: "<<(static_cast<std::chrono::duration<double>>(end - start)).count()+pairwaise_time<<std::endl;

    std::cout<<output5<<std::endl;

    arbiter.addLayer(attention);

    arbiter.addLayer(layer1);
    arbiter.addLayer(layer2);
    arbiter.addLayer(layer3);
    arbiter.addLayer(layer4);
    arbiter.addLayer(layer5);
    

    arbiter.applyReward(0);

    start = std::chrono::system_clock::now();

    arbiter.shuttle();

    end = std::chrono::system_clock::now();

    std::cout<<"Time: "<<(static_cast<std::chrono::duration<double>>(end - start)).count()<<std::endl;

    return 0;

    // while(true)
    // {

    //     start = std::chrono::system_clock::now();

    //     auto output = encoder->fire(input);

    //     // output = output / output.size();

    //     auto output1 = recurrent1->fire(output);

    //     auto output2 = layer1->fire(output1);

    //     // output2 = output2 / output2.size();

    //     auto output3 = recurrent2->fire(output2);

    //     auto picker = decision->fire(output3);

    //     auto out = layers[0]->fire(output3);

    //     end = std::chrono::system_clock::now();

    //     std::cout<<"Time: "<<static_cast<std::chrono::duration<double>>(end - start).count()<<std::endl;

    //     std::cout<<"Output: "<<picker<<std::endl;

    //     float x;

    //     std::cin>>x;

    //     arbiter.applyReward(x);

    //     start = std::chrono::system_clock::now();

    //     arbiter.shuttle();

    //     end = std::chrono::system_clock::now();

    //     std::cout<<"Shuttle time: "<<static_cast<std::chrono::duration<double>>(end - start).count()<<std::endl;

    // }

    
    // // the network will be split into layer that will be split into block an additional network will choose what block should be active in each step.

    // return 0;

    // auto layer0 = std::make_shared<KapiBara_SubLayer>();

    // auto recurrent0 = std::make_shared<snn::RResNet<4,256,20>>();

    

    // arbiter.addLayer(layer0);
    // arbiter.addLayer(recurrent0);

    // if( arbiter.load() == 0 )
    // {
    //     std::cout<<"Loaded networks"<<std::endl;
    // }
    // else
    // {
    //     arbiter.setup();
    // }

    // snn::Arbiter arbiter;

    // auto layer0 = std::make_shared<snn::LayerKAC<4,64,20,snn::ReLu>>();

    // auto layer1 = std::make_shared<snn::LayerKAC<64,32,20,snn::ReLu>>();

    // auto layer2 = std::make_shared<snn::LayerKAC<32,2,20,snn::Linear>>();

    // arbiter.addLayer(layer0);
    // arbiter.addLayer(layer1);
    // arbiter.addLayer(layer2);

    // arbiter.setup();

    // auto start = std::chrono::system_clock::now();

    // auto end = std::chrono::system_clock::now();

    // std::chrono::duration<double> elapsed_initialization_seconds = end - start;

    // std::cout << "Finished initialization at " << elapsed_initialization_seconds.count() << std::endl;

    // std::cout<<"Starting network"<<std::endl;

    // long double best_reward = -9999999;


    // while(true)
    // {

    //     snn::SIMDVectorLite<6> cart_input = read_fifo_static();

    //     if( cart_input[5] > 0.5f)
    //     {

    //         if(( cart_input[4] > best_reward ) || cart_input[4] >= 0 )
    //         {
    //             best_reward = cart_input[4];

    //             std::cout<<"New best reward: "<<best_reward<<std::endl;

    //             arbiter.save();
    //         }


    //         arbiter.applyReward(cart_input[4]);

    //         arbiter.shuttle();

    //     }

    //     snn::SIMDVectorLite<4> input;

    //     input[0] = cart_input[0];
    //     input[1] = cart_input[1];
    //     input[2] = cart_input[2];
    //     input[3] = cart_input[3];

    //     // snn::SIMDVectorLite rnn = recurrent0->fire(input);

    //     auto output0 = layer0->fire(input);
    //     auto output1 = layer1->fire(output0);
    //     auto output2 = layer2->fire(output1);

    //     // std::cout<<"Recurrent out: " << rnn << std::endl;

    //     send_fifo(output2);

    // }

    // return 0;

}

