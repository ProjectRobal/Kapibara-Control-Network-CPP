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
#include "evo_kan_layer.hpp"

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

/*

    Algorithm is great but it fails when:

    - Inputs are too much correlated with each other ( are very similar to each other )
    - Outputs are small ( well I had to decrease the error threshold for points nuding )


*/


int main(int argc,char** argv)
{
    std::cout<<"Starting..."<<std::endl;

    std::chrono::time_point<std::chrono::system_clock> start, end;

    snn::Arbiter arbiter;


    const size_t size = 32;

    const size_t samples_count = 32;


    // those hold splines for activations functions. I am going to use splines:
    // exp(-(x-x1)^2 * b)*a
    // we can treat x1 and a as coordinates in 2D space:
    // (x,y) = (x1,a)

    snn::EvoKAN<4> q[2];


    const size_t iter=10000;

    number error = 0.f;

    snn::UniformInit<0.f,1.f> uniform;

    snn::SIMDVectorLite<6> fifo_input = read_fifo_static();

    snn::SIMDVectorLite<4> input;

    input[0] = fifo_input[0];
    input[1] = fifo_input[1];
    input[2] = fifo_input[2];
    input[3] = fifo_input[3];

    number reward = 0.f;

    size_t best_action_id = 0;

    const number alpha = 1.f;

    const number gamma = 0.5f; 

    number epsilon = 1.f;

    number cum_rewad = 0;

    for( size_t i = 0 ; i < iter ; ++i )
    {
        // std::cout<<"Step: "<<i<<std::endl;
        // std::cout<<fifo_input<<std::endl;

        snn::SIMDVectorLite<2> action(0);

        action[0] =  q[0].fire(input);

        action[1] = q[1].fire(input);

        if( uniform.init() > epsilon )
        {

            best_action_id = action[1] < action[0];

        }
        else
        {
            best_action_id = uniform.init() > 0.5f;
        }
        
        epsilon = 0.9f*epsilon;
        
        send_fifo<2>(action);

        fifo_input = read_fifo_static();

        snn::SIMDVectorLite<4> last_input = input;

        input[0] = fifo_input[0];
        input[1] = fifo_input[1];
        input[2] = fifo_input[2];
        input[3] = fifo_input[3];

        reward = fifo_input[4];

        // update q values

        number old_q = action[best_action_id];

        // get Q values for new state
        action[0] =  q[0].fire(input);

        action[1] = q[1].fire(input);

        snn::EvoKAN<4> &best_q = q[best_action_id];

        number max_q = std::max(action[0],action[1]);

        number new_q = old_q + alpha*( reward + gamma*max_q);

        if( fifo_input[5] > 0.5f )
        {
            std::cout<<"Rewards: "<<cum_rewad<<std::endl;

            cum_rewad = 0.f;

            new_q = 0.f;
        }

        cum_rewad += reward;


        best_q.fit(last_input,old_q,new_q);

    }


    q[0].printInfo();
    q[1].printInfo();

    return 0;
}

