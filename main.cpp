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

#include "layer_hebbian.hpp"

#include "evo_kan_block.hpp"

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
        output[i] = static_cast<number>(std::stof(num,NULL));

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
size_t max_id(const snn::SIMDVectorLite<N>& p)
{
    size_t max_i = 0;

    for( size_t i=1 ; i <N ; ++i )
    {
        if( p[i] > p[max_i] )
        {
            max_i = i;
        }
    }

    return max_i;
}


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


    const size_t size = 32;

    snn::SIMDVectorLite<size> input;

    snn::SIMDVectorLite<size> last_input;

    snn::UniformInit<-1.f,1.f> rand;

    for(size_t i=0;i<size;++i)
    {
        input[i] = rand.init();
    }

    input[0] = 2.f;

    // those hold splines for activations functions. I am going to use splines:
    // exp(-(x-x1)^2 * b)*a
    // we can treat x1 and a as coordinates in 2D space:
    // (x,y) = (x1,a)


    number target_out = 4;
    number error = 0;

    snn::EvoKAN<size,40> kan_block;

    auto out = kan_block.fire(input);

    std::cout<<"KAN output: "<<out<<std::endl;

    error = abs(out - target_out);

    std::cout<<"Error: "<<error<<std::endl;

    kan_block.fit(input,out,target_out);

    out = kan_block.fire(input);

    std::cout<<"KAN output: "<<out<<std::endl;

    error = abs(out - target_out);

    std::cout<<"Error: "<<error<<std::endl;

    kan_block.printInfo();

    for(size_t i=0;i<size;++i)
    {
        input[i] = rand.init();
    }

    last_input = input;

    std::cout<<input<<std::endl;

    target_out = 2.f;

    out = kan_block.fire(input);

    std::cout<<"Target output is "<<target_out<<std::endl;

    std::cout<<"KAN output: "<<out<<std::endl;

    error = abs(out - target_out);

    std::cout<<"Error: "<<error<<std::endl;

    kan_block.fit(input,out,target_out);

    out = kan_block.fire(input);

    std::cout<<"KAN output: "<<out<<std::endl;

    error = abs(out - target_out);

    std::cout<<"Error: "<<error<<std::endl;

    kan_block.printInfo();

    std::cout<<"KAN output: "<<out<<std::endl;

    error = abs(out - target_out);

    std::cout<<"Error: "<<error<<std::endl;

    kan_block.fit(input,out,target_out);

    out = kan_block.fire(input);

    std::cout<<"KAN output: "<<out<<std::endl;

    error = abs(out - target_out);

    std::cout<<"Error: "<<error<<std::endl;

    kan_block.printInfo();

    std::cout<<"Last input"<<std::endl;

    target_out = 4;

    out = kan_block.fire(last_input);

    std::cout<<"KAN output: "<<out<<std::endl;

    error = abs(out - target_out);

    std::cout<<"Error: "<<error<<std::endl;

    kan_block.printInfo();

    return 0;
}

