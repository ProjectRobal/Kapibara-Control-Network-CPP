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

#include <opencv2/opencv.hpp>


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

#include "static_kan_block.hpp"

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

template<size_t Size>
number variance(const snn::SIMDVectorLite<Size>& input)
{
    number mean = input.reduce() / Size;

    snn::SIMDVectorLite m_mean = input - mean;

    m_mean = m_mean*m_mean;

    return m_mean.reduce() / Size;
}

/*

    Idea is simple we take image and then using convolution and KAN layer generate map of rewards in 2D space.
    
    To fit data we generate noise mask and makes convolution fit to it then that 
    noise map we give to output layer.

    I don't have better idea for now, I should probably think about some defragmentation algorithm for it.


    What can we do to speed it up?

    We can have fixed number of nodes in activations, so we can avoid using std::vectors.

    We can use uint16_t instead of floats for some speed ups, sounds like a good idea generally, but it doesn't give significant speed ups.

    I get rid of smart pointers and it gave major speed ups.

*/



int main(int argc,char** argv)
{
    std::cout<<"Starting..."<<std::endl;

    std::chrono::time_point<std::chrono::system_clock> start, end;

    snn::Arbiter arbiter;


    const size_t size = 32;

    const size_t samples_count = 32;

    
    snn::SIMDVectorLite<64> last_target(0.f);

    last_target[10] = 12.f;

    last_target[14] = -10.f;


    last_target[30] = -4.f;
    // output KAN layer, we can use small output and attach the information about current position in the reward map

    snn::StaticKAN<64,256> static_kan_block;

    // snn::StaticKAN<64,4096*2> static_kan_layer[16];


    snn::UniformInit<(number)-0.5f,(number)0.5f> noise;

    snn::UniformInit<(number)0.f,(number)1.f> chooser;

    const size_t dataset_size = 256;

    snn::SIMDVectorLite<64> dataset[dataset_size];

    number outputs[dataset_size];

    for(auto& input : dataset)
    {
        for(size_t i=0;i<64;++i)
        {
            input[i] = noise.init();
        }

    }

    for(size_t i=0;i<dataset_size;++i)
    {
        outputs[i] = noise.init()*10.f;
    }
    
    start = std::chrono::system_clock::now();

    // auto output_last = static_kan_block.fire(last_target);

    number output_last = 0;

    end = std::chrono::system_clock::now();

    std::cout<<"Elapsed: "<<std::chrono::duration<double>(end - start)<<" s"<<std::endl;

    std::cout<<output_last<<std::endl;

    snn::SIMDVectorLite<16> layer_output;

    number error = 0.f;

    for(size_t e=0;e<10000;++e)
    {
        for(size_t i=0;i<dataset_size;++i)
        {
            output_last = static_kan_block.fire(dataset[i]);


            // if( (y/2)*60 + (x/2) % 2 == 0 )
            if( chooser.init() > 0.75f)
            {
                kernel.fit(cnn_input,0.f,noise.init());
            }


            static_kan_block.fit(dataset[i],output_last,outputs[i]);

            error += abs(output_last - outputs[i]);

            // std::cout<<output_last<<" "<<outputs[i]<<std::endl;

            // std::cout<<"Fitting: "<<i<<"/"<<dataset_size<<" error: "<<std::abs(output_last - outputs[i])<<std::endl;
        }

        std::cout<<"Error: "<<error/dataset_size<<std::endl;

        error = 0.f;


    std::cout<<"Fitting done"<<std::endl;

    double total_error = 0.f;

    for(size_t i=0;i<dataset_size;++i)
    {
        number out = static_kan_block.fire(dataset[i]);

        number error = std::abs(out - outputs[i]);

        total_error += error;
    }

    std::cout<<"Total error: "<<total_error/dataset_size<<std::endl;

    
    char c;

    std::cout<<"Press any key to continue"<<std::endl;
    std::cin>>c;


    return 0;
}

