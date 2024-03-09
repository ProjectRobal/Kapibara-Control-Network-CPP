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

#include "neurons/feedforwardneuron.hpp"

#include "crossovers/onepoint.hpp"
#include "crossovers/fastuniform.hpp"
#include "crossovers/fastonepoint.hpp"
 
#include "initializers/gauss.hpp"
#include "initializers/normalized_gauss.hpp"

#include "mutatiom/gauss_mutation.hpp"

#include "parallel.hpp"

#include "block.hpp"
#include "layer.hpp"

#include "activation/sigmoid.hpp"
#include "activation/relu.hpp"

#include "networks/kac_actor_crictic.hpp"

#include "shared_mem.hpp"

#include "proto/cartpole.pb.h"

number stddev(const snn::SIMDVector& vec)
{
    number mean=vec.dot_product();

    snn::SIMDVector omg=vec-mean;

    omg=omg*omg;

    return std::sqrt(omg.dot_product()/vec.size());

}

number evaluatePolynomial(const snn::SIMDVector& poly,const number& x)
{
    return std::pow(x,3)*poly[0] + std::pow(x,2)*poly[1] + x*poly[2] + poly[3];
}


struct CartPoleInterface
{
    uint8_t wait;

    double inputs[4]; 
    double outputs[2];
    double reward;
};


int main(int argc,char** argv)
{

    std::shared_ptr<snn::NormalizedGaussInit> norm_gauss=std::make_shared<snn::NormalizedGaussInit>(0.f,0.01f);
    std::shared_ptr<snn::GaussInit> gauss=std::make_shared<snn::GaussInit>(0.f,0.1f);

    std::shared_ptr<snn::GaussMutation> mutation=std::make_shared<snn::GaussMutation>(0.f,0.01f,0.1f);
    std::shared_ptr<snn::OnePoint> cross=std::make_shared<snn::OnePoint>();


    long double best_reward=-100;

    snn::ReLu activate;

    std::shared_ptr<snn::ReLu> relu=std::make_shared<snn::ReLu>();

    const size_t input_size=4;
    const size_t output_size=1;
    
    snn::ActorCriticNetwork<input_size,output_size,512,1> network;

    network.setup(10,32,norm_gauss,cross,mutation);

    std::shared_ptr<snn::LayerProto> layer=std::make_shared<snn::Layer<snn::FeedForwardNeuron<input_size,64>,1,32>>(16,norm_gauss,cross,mutation);
    std::shared_ptr<snn::LayerProto> layer1=std::make_shared<snn::Layer<snn::FeedForwardNeuron<64,output_size>,1,32>>(8,norm_gauss,cross,mutation);

    network.addLayer(layer);
    network.addLayer(layer1);

    // we will try to find poles in this polynomials in form of a[0]*x^3 + a[1]*x^2 + a[2] * x + a[3] = 0; 

    snn::SIMDVector inputs({0.25,0.5,0.6,0.4});  

    size_t step=1;  

    size_t maxSteps=50000;

    CartPoleInterface interf;

    interf={0};

    CartPole interface;

    interface.set_wait(1);
    interface.add_inputs(0);
    interface.add_inputs(0);
    interface.add_inputs(0);
    interface.add_inputs(0);
    interface.add_outputs(0);
    interface.add_outputs(0);
    interface.set_reward(0);


    if(access("fifo",F_OK) != 0)
    {
        if(mkfifo("fifo",0666) == -1)
        {
            std::cerr<<"Cannot create fifo"<<std::endl;
        }
    }

    if(access("fifo_in",F_OK) != 0)
    {
        if(mkfifo("fifo_in",0666) == -1)
        {
            std::cerr<<"Cannot create input fifo"<<std::endl;
        }
    }

    std::string buffer;


    std::cout<<"Starting"<<std::endl;

    int fifo=0;

    while(maxSteps--)
    {

        char start_code=0;

        fifo=open("fifo",O_RDONLY);

        read(fifo,&start_code,1);

        if( start_code == '@' )
        {

            size_t size=0;

            read(fifo,(void*)&size,8);

            char _buffer[size];

            read(fifo,_buffer,size);

            if(!interface.ParseFromArray(_buffer,size))
            {
                return -11;
            }

            std::cout<<"Recived: "<<interface.reward()<<std::endl;

            interface.set_reward(10);

            if(!interface.SerializeToString(&buffer))
            {
                return -12;
            }

            close(fifo);

            fifo=open("fifo",O_WRONLY);

            write(fifo,"@",1);
            
            size=buffer.size();

            write(fifo,(void*)&size,sizeof(size_t));
            write(fifo,buffer.c_str(),buffer.size());

            close(fifo);

            std::cout<<"Written data!"<<std::endl;
        }

        close(fifo);

        std::chrono::time_point start=std::chrono::steady_clock::now();

        number x=network.step(inputs)[0];

        long double reward=-abs(evaluatePolynomial(inputs,x));

        if(reward>best_reward)
        {
            std::cout<<"Best reward: "<<reward<<" at step: "<<step<<std::endl;
            std::cout<<"Best x: "<<x<<std::endl;
            best_reward=reward;
        }

        network.applyReward(reward);

        std::chrono::time_point end=std::chrono::steady_clock::now();

        const std::chrono::duration<double> elapsed_seconds(end-start);

        std::cout<<"Time "<<elapsed_seconds<<" s"<<std::endl;

        interface.set_reward(reward);

        std::cout<<"Reward: "<<reward<<std::endl;

        interface.set_wait(1);

        if(!interface.SerializeToString(&buffer))
        {
            return -12;
        }

        

        ++step;

    }
}

