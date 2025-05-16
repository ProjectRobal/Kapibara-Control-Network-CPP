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


template<size_t N>
long double cross_entropy_loss(const snn::SIMDVectorLite<N>& p1,const snn::SIMDVectorLite<N>& p2)
{
    number loss = 0.f;

    for(size_t i=0;i<N;++i)
    {
        number v = -p1[i]*std::log(p2[i]+0.000000001f);

        loss += v;
    }

    return loss;
}


int main(int argc,char** argv)
{
    std::cout<<"Starting..."<<std::endl;

    std::chrono::time_point<std::chrono::system_clock> start, end;

    snn::Arbiter arbiter;

    // identify digist on 16x16 image

    snn::SIMDVectorLite<256> input;

    snn::SIMDVectorLite<256> inputs[10];

    inputs[0] = snn::SIMDVectorLite<256>(
    {
    0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,
    0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,
    0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,
    0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,
    1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,
    1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
    1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
    1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
    1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
    1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,
    0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,
    0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,
    0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,
    0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,
    0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0
    });

    inputs[1] = snn::SIMDVectorLite<256>(
    {
    0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,
    0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,
    0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,
    0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,
    0,0,0,1,1,0,1,1,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,1,1,1,0,0,0,0,0,0,0,
    0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,
    0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,
    0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,
    0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,
    0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,
    0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,
    0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,
    0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,
    0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,
    0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0
    });

    inputs[2] = snn::SIMDVectorLite<256>(
    {
    0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
    0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
    0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
    0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
    0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0
    });

    inputs[3] = snn::SIMDVectorLite<256>(
    {
    0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
    0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
    0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
    0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
    0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
    0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
    0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0
    });

    inputs[4] = snn::SIMDVectorLite<256>(
    {
    0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,
    0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,
    0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,
    0,0,0,0,0,0,0,0,1,1,1,0,1,1,0,0,
    0,0,0,0,0,0,0,1,1,1,0,0,1,1,0,0,
    0,0,0,0,0,0,1,1,1,0,0,0,1,1,0,0,
    0,0,0,0,0,1,1,1,0,0,0,0,1,1,0,0,
    0,0,0,0,1,1,1,0,0,0,0,0,1,1,0,0,
    0,0,0,1,1,1,0,0,0,0,0,0,1,1,0,0,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0
    });

    inputs[5] = snn::SIMDVectorLite<256>(
    {
    0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,
    0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,
    0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,
    0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,
    0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,
    0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,
    0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,
    0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0
    });

    inputs[6] = snn::SIMDVectorLite<256>(
    {
    0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,
    0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,
    0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,
    0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,
    0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,
    0,1,1,0,0,1,1,1,1,1,1,1,0,0,0,0,
    0,1,1,0,1,1,0,0,0,0,0,0,1,1,0,0,
    1,1,0,0,1,0,0,0,0,0,0,0,0,1,1,0,
    1,1,0,0,1,0,0,0,0,0,0,0,0,1,1,0,
    1,1,0,0,1,0,0,0,0,0,0,0,0,1,1,0,
    1,1,0,0,1,0,0,0,0,0,0,0,0,1,1,0,
    0,1,1,0,1,1,0,0,0,0,0,0,1,1,0,0,
    0,0,1,1,0,1,1,1,1,1,1,1,1,0,0,0,
    0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0
    });

    inputs[7] = snn::SIMDVectorLite<256>(
    {
    0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,
    0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,
    0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,
    0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,
    0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,
    0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,
    0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,
    0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,
    0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,
    0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,
    0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,
    0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,
    0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0
    });

    inputs[8] = snn::SIMDVectorLite<256>(
    {
    0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,
    0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,
    0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,
    0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,
    0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,
    0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,
    0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,
    0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,
    0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,
    0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,
    0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,
    0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,
    0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,
    0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,
    0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    });

    inputs[9] = snn::SIMDVectorLite<256>(
    {
    0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,
    0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,
    0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,
    0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,
    0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,
    1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,
    1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,
    1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,
    0,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,
    0,0,1,1,0,0,0,0,0,0,0,1,1,1,0,0,
    0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,
    0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,
    0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,
    0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,
    0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0
    });

    snn::UniformInit<0.f,1.f> uni;

    for(size_t i=0;i<input.size();++i)
    {
        input[i] = uni.init();
    }

    std::cout<<"Running network"<<std::endl;

    const size_t population_size = 20;


    std::shared_ptr<snn::LayerKAC<256,512,population_size,snn::Linear>> layer1 = std::make_shared<snn::LayerKAC<256,512,population_size,snn::Linear>>();

    std::shared_ptr<snn::LayerKAC<512,256,population_size,snn::ReLu>> layer2 = std::make_shared<snn::LayerKAC<512,256,population_size,snn::ReLu>>();

    std::shared_ptr<snn::LayerKAC<256,64,population_size,snn::ReLu>> layer3 = std::make_shared<snn::LayerKAC<256,64,population_size,snn::ReLu>>();

    std::shared_ptr<snn::LayerKAC<64,10,population_size,snn::SoftMax>> layer4 = std::make_shared<snn::LayerKAC<64,10,population_size,snn::SoftMax>>();

    layer1->setup();
    layer2->setup();
    layer3->setup();
    layer4->setup();

    arbiter.addLayer(layer1);
    arbiter.addLayer(layer2);
    arbiter.addLayer(layer3);
    arbiter.addLayer(layer4);


    const size_t iterations = 1000000;

    for(size_t i=0;i<iterations;++i)
    {
        // std::cout<<i<<std::endl;

        number error = 0;

        for(size_t j=0;j<1;++j)
        {
            
            auto output1 = layer1->fire(inputs[j]);
            auto output2 = layer2->fire(output1);
            auto output3 = layer3->fire(output2);
            auto output = layer4->fire(output3);

            snn::SIMDVectorLite<10> label(0);

            label[j] = 1.f;

            number err = -cross_entropy_loss<10>(label,output);

            error += err;

            arbiter.applyReward(err*1000.f);

            arbiter.shuttle();
        }

        std::cout<<"Step: "<<i<<" error: "<<error<<std::endl;

        // std::cout<<input[0]<<std::endl;
    }
    {

    }
    

    arbiter.applyReward(0);

    arbiter.shuttle();


    return 0;
}

