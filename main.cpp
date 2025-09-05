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

#include <experimental/simd>

#include <immintrin.h>

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

    My idea for QKAN is to aproximate spline using 256 elements look up table.

    There fore we could therotically save on computation.

    In normall CNN we use FFT for faster convolution but how to do it when 
    kernal is made of KAN function? 

    Where too fit SIMD into it, well we could use it for faster addition with reduction.

*/

template<size_t InputSize,size_t OutputSize=1,class Init=snn::UniformInit<-127.0,128.0>>
class QKAN
{
    protected:

    int8_t *table;
    int16_t *decimal;

    public:

    QKAN()
    {
        this->table = new int8_t[256*InputSize*OutputSize];
        this->decimal = new int16_t[256*InputSize*OutputSize];

        Init init;

        for(size_t i=0;i<256*InputSize*OutputSize;++i)
        {
            this->table[i] = init.init();
        }
    }


    void fire(int8_t x[],int8_t y[])
    {
        for(size_t o=0;o<OutputSize;++o)
        {
            int32_t sum = 0;         
            
            for(size_t i=0;i<InputSize;++i)
            {
                uint8_t _x = x[i] + 127;

                sum += this->table[o*256*InputSize + i*256 + _x];
            }

            y[o] = sum/InputSize;

            for(size_t i=0;i<InputSize;++i)
            {
                uint8_t _x = x[i] + 127;

                int8_t w = this->table[o*256*InputSize + i*256 + _x];

                this->decimal[o*256*InputSize + i*256 + _x] += w*y[o]/10000;

                if(this->decimal[o*256*InputSize + i*256 + _x] >= 1000)
                {
                    this->table[o*256*InputSize + i*256 + _x] ++;

                    this->decimal[o*256*InputSize + i*256 + _x] = 0;
                }
                else if(this->decimal[o*256*InputSize + i*256 + _x] <= -1000)
                {
                    this->table[o*256*InputSize + i*256 + _x] --;

                    this->decimal[o*256*InputSize + i*256 + _x] = 0;
                }
            }

        }
    }


    ~QKAN()
    {
        delete[] this->table;
    }

};



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

    snn::UniformInit<(number)-127.f,(number)128.f> noise;

    const size_t width = 224;

    const size_t input_size = width*width;

    int8_t data[input_size];

    for(size_t i=0;i<input_size;i++)
    {
        data[i] = noise.init();
    }

    start = std::chrono::system_clock::now();

    // ai = _mm512_loadu_epi8(data);

    std::cout<<"Timestamp: "<<std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start).count()<<" ms"<<std::endl;

    size_t x = 1,y = 1;

    int8_t output_img[input_size];

    // Lets test image like inference
    QKAN<3*3,1> kan;

    int8_t buffer[9];

    // Take chunk of an image

    clock_t _start = clock(); 

    for(size_t y=1;y<width-1;++y)
    {
        for(size_t x=1;x<width-1;++x)
        {
    
            int8_t output;

            buffer[0] = data[(y-1)*width + (x-1)];
            buffer[1] = data[(y-1)*width + x];
            buffer[2] = data[(y-1)*width + x+1];

            buffer[3] = data[y*width + (x-1)];
            buffer[4] = data[y*width + x];
            buffer[5] = data[y*width + x+1];

            buffer[6] = data[(y+1)*width + (x-1)];
            buffer[7] = data[(y+1)*width + x];
            buffer[8] = data[(y+1)*width + x+1];

            kan.fire(buffer,&output);

            output_img[width*(y-1)+(x-1)] = output;

        }

    }


    std::cout<<"Timestamp: "<<static_cast<double>(clock()-_start)/CLOCKS_PER_SEC * 1000<<" us"<<std::endl;

    std::cout<<(int32_t)output_img[10*width + 10]<<std::endl;
    
    return 0;

    

    const size_t dataset_size = 100;

    snn::SIMDVectorLite<1024> dataset[dataset_size];

    number outputs[dataset_size];

    for(auto& input : dataset)
    {
        for(size_t i=0;i<1024;++i)
        {
            input[i] = noise.init();
        }

    }

    for(size_t i=0;i<dataset_size;++i)
    {
        outputs[i] = noise.init()*10.f;
    }
    
    start = std::chrono::system_clock::now();



    return 0;
}

