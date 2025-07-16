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

#include "static_kan_block.hpp"

#include "evo_kan_block.hpp"

#include "evo_kan_layer.hpp"


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



template<size_t Size>
void test_simd()
{
    snn::SIMDVectorLite<Size-1> a(1);

    assert( a.reduce() == Size-1 );

    snn::SIMDVectorLite<Size-2> b(1);

    assert( b.reduce() == Size-2 );

    snn::SIMDVectorLite<Size> x1;

    for(size_t i=0;i<Size;++i)
    {
        x1[i] = i;
        assert(x1[i] == i);
    }

    snn::SIMDVectorLite<Size> x2;

    for(size_t i=0;i<Size;++i)
    {
        x2[i] = i;
        assert(x2[i] == i);
    }

    snn::SIMDVectorLite<Size> x = x1 + x2;

    for(size_t i=0;i<Size;++i)
    {
        assert(x[i] == i+i);
    }

    x = x1-x2;

    for(size_t i=0;i<Size;++i)
    {
        assert(x[i] == i-i);
    }

    x = x1*x2;

    for(size_t i=0;i<Size;++i)
    {
        assert(x[i] == i*i);
    }

    x2[0] = 1.f;

    x = x1/x2;

    for(size_t i=1;i<Size;++i)
    {
        assert(x[i] > 0.99f && x[i] < 1.01f);
    }

}

void test_sort()
{
    std::vector<number> unsorted;

    snn::GaussInit<0.f,0.1f> init;

    number max_x = -9999999999;
    number min_x = 9999999999;

    for(size_t i=0;i<1024;++i)
    {
        number x = init.init();

        max_x = std::max(max_x,x);
        min_x = std::min(min_x,x);

        unsorted.push_back(x);
    }

    // std::sort(unsorted.begin(),unsorted.end());

    std::vector<number> sorted = unsorted;

    number to_add = init.init();

    sorted.push_back(to_add);
    
    std::chrono::time_point<std::chrono::system_clock> start, end;

    std::sort(sorted.begin(),sorted.end());

    std::vector<number> sorted2;

    start = std::chrono::system_clock::now();

    for(size_t i=0;i<unsorted.size();++i)
    {
        number x = unsorted[i];

        auto loc = std::lower_bound(sorted2.begin(),sorted2.end(),x);

        sorted2.insert(loc,x);

    }

    auto loc = std::lower_bound(sorted2.begin(),sorted2.end(),to_add);

    sorted2.insert(loc,to_add);

    end = std::chrono::system_clock::now();

    std::cout<<"Time: "<<std::chrono::duration<double>(end - start)<<" s"<<std::endl;

    // check 

    for(size_t i=0;i<sorted.size();++i)
    {   
        // std::cout<<"i: "<<i<<" left: "<<sorted[i]<<" right: "<<sorted2[i]<<std::endl;
        assert(sorted[i] == sorted2[i]);
    }

}

int main(int argc,char** argv)
{
    std::cout<<"Starting..."<<std::endl;

    std::cout<<"SIMD 19 length test"<<std::endl;
    test_simd<19>();
    std::cout<<"Passed"<<std::endl;

    std::cout<<"SIMD 32 length test"<<std::endl;
    test_simd<32>();
    std::cout<<"Passed"<<std::endl;

    std::cout<<"SIMD 33 length test"<<std::endl;
    test_simd<33>();
    std::cout<<"Passed"<<std::endl;

    std::cout<<"SIMD 96 length test"<<std::endl;
    test_simd<96>();
    std::cout<<"Passed"<<std::endl;

    std::cout<<"SIMD 100 length test"<<std::endl;
    test_simd<100>();
    std::cout<<"Passed"<<std::endl;

    std::cout<<"Sorting test"<<std::endl;
    test_sort();

    // return 0;

    snn::EvoKanLayer<64,4> kan;

    std::fstream file;

    file.open("network.neur",std::ios::in|std::ios::binary);

    kan.load(file);

    if( !file.good() )
    {
        std::cerr<<"Cannot load network splines!"<<std::endl;
    }
    else
    {
        std::cout<<"Network loaded!!!"<<std::endl;
    }

    file.close();


    std::chrono::time_point<std::chrono::system_clock> start, end;

    const size_t samples_count = 32;


    snn::UniformInit<(number)-0.5f,(number)0.5f> noise;

    snn::UniformInit<(number)0.f,(number)1.f> chooser;

    const size_t dataset_size = 1024;

    snn::SIMDVectorLite<64> dataset[dataset_size];

    snn::SIMDVectorLite<4> outputs[dataset_size];

    for(auto& input : dataset)
    {
        for(size_t i=0;i<64;++i)
        {
            input[i] = noise.init()*10.f;
        }

    }

    for(auto& output : outputs)
    {
        for(size_t i=0;i<4;++i)
        {
            output[i] = noise.init()*10.f;
        }
    }
    
    start = std::chrono::system_clock::now();

    number output_last = 0;

    end = std::chrono::system_clock::now();


    std::cout<<"Dataset:"<<std::endl;
    for(size_t e=0;e<1;++e)
    {
        for(size_t i=0;i<dataset_size;++i)
        {
            start = std::chrono::system_clock::now();
            kan.fit(dataset[i],outputs[i]);
        
            end = std::chrono::system_clock::now();

            // std::cout<<"Time: "<<std::chrono::duration<double>(end - start)<<" s"<<std::endl;
        }
    }

    std::cout<<"Test fit:"<<std::endl;

    number error = 0.f;

    for(size_t i=0;i<dataset_size;++i)
    {
        snn::SIMDVectorLite<4> output;

        snn::SIMDVectorLite<4> y = outputs[i];

        start = std::chrono::system_clock::now();

        output = kan.fire(dataset[i]);

        end = std::chrono::system_clock::now();

        // std::cout<<"Time: "<<std::chrono::duration<double>(end - start)<<" s"<<std::endl;

        error += abs( (y - output).reduce() );
    }

    std::cout<<"Error: "<<error/dataset_size<<std::endl;

    file.open("network.neur",std::ios::out|std::ios::binary);

    kan.save(file);

    if( !file.good() )
    {
        std::cerr<<"Cannot save network splines!"<<std::endl;
    }

    file.close();

    // char c;

    // std::cin>>c;

    // // Save plot

    // number x_min = -10.f;
    // number x_max = 10.f;

    // const number step = 0.01f;

    // std::fstream file;
    
    // file.open("test_plot.csv",std::ios::out);

    // snn::SIMDVectorLite<64> input;

    // while( x_min <= x_max )
    // {
    //     input[0] = x_min;

    //     number _y = kan.fire(input);

    //     file<<x_min<<";"<<_y<<std::endl;

    //     x_min += step;
    // }

    // file.close();
    

    return 0;
}

