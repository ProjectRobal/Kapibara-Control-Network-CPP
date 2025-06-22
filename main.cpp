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

    // snn::EvoKanLayer<4,16> input_layer;

    // load test image

    cv::Mat image = cv::imread("./room.jpg",cv::IMREAD_GRAYSCALE); // Replace with your image path

    // Check if the image is loaded successfully
    if (image.empty()) {
        std::cerr << "Error: Could not load image." << std::endl;
        return 1;
    }

    // Resize the image to 120x120
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(120, 120));


    // Display the resized image

    // we are going to use kernel of 2x2
    snn::SIMDVectorLite<4> cnn_input;

    snn::SIMDVectorLite<4> *cnn_input_snaphost = new snn::SIMDVectorLite<4>[60*60];


    snn::EvoKAN<4> kernel;


    // output
    snn::SIMDVectorLite<(60)*(60)> output(0.f);

    snn::SIMDVectorLite<(60)*(60)> target_output(0.f);

    
    snn::SIMDVectorLite<64> last_target(0.f);

    last_target[10] = 12.f;

    last_target[14] = -10.f;


    last_target[30] = -4.f;
    // output KAN layer, we can use small output and attach the information about current position in the reward map

    snn::EvoKanLayer<(60)*(60),64> last_layer;


    snn::UniformInit<-10.f,10.f> noise;

    snn::UniformInit<0.f,1.f> chooser;

    for(size_t i=0;i<64;i++)
    {
        last_target[i] = noise.init();
    }


    start = std::chrono::system_clock::now();
    
    // we use stride of 2
    for(int y=1;y<120;y+=2)
    {
        for(int x=1;x<120;x+=2)
        {
            cnn_input[0] = resized_image.at<u_char>(x-1,y-1)/255.f;
            cnn_input[1] = resized_image.at<u_char>(x,y-1)/255.f;

            cnn_input[2] = resized_image.at<u_char>(x-1,y)/255.f;
            cnn_input[3] = resized_image.at<u_char>(x,y)/255.f;

            // if( (y/2)*60 + (x/2) % 2 == 0 )
            if( chooser.init() > 0.75f)
            {
                kernel.fit(cnn_input,0.f,noise.init());
            }

            output[(y/2)*60 + (x/2)] = kernel.fire(cnn_input);
            // std::cout<<kernel.fire(cnn_input)<<std::endl;
        }
    }


    last_layer.fit(output,last_target);

    auto output_last = last_layer.fire(output);

    end = std::chrono::system_clock::now();


    std::cout<<"Elapsed: "<<std::chrono::duration<double>(end - start)<<" s"<<std::endl;

    std::cout<<output_last<<std::endl;

    
    char c;

    std::cout<<"Press any key to continue"<<std::endl;
    std::cin>>c;


    return 0;
}

