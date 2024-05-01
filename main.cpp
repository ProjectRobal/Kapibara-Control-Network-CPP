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

#include "mutatiom/gauss_mutation.hpp"

#include "parallel.hpp"

#include "block.hpp"
#include "layer.hpp"

#include "layer_st.hpp"
#include "layer.hpp"

#include "activation/sigmoid.hpp"
#include "activation/relu.hpp"

#include "shared_mem.hpp"

#include "proto/cartpole.pb.h"

#include "replay/short_term_memory.hpp"

#include "network.hpp"

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


bool sendInterface(const CartPole& interface)
{
    int fifo=0;

    std::string buffer;

    if(!interface.SerializeToString(&buffer))
    {
        return false;
    }

    fifo=open("fifo",O_WRONLY);

    write(fifo,"@",1);
            
    size_t size=buffer.size();

    write(fifo,(void*)&size,sizeof(size_t));
    write(fifo,buffer.c_str(),buffer.size());

    close(fifo);

    return true;
}

CartPole getInterface()
{
    CartPole interface;
    bool ret=true;

    char start_code=0;
    
    int fifo=open("fifo",O_RDONLY);

    read(fifo,&start_code,1);

    if( start_code == '@' )
    {

        size_t size=0;

        read(fifo,(void*)&size,8);

        char _buffer[size];

        read(fifo,_buffer,size);

        std::cout<<"Parsing data: "<<size<<std::endl;

        if(!interface.ParseFromArray(_buffer,size))
        {
            return interface;
        }
    }

    close(fifo);

    return interface;
}


int main(int argc,char** argv)
{

    std::shared_ptr<snn::NormalizedGaussInit> norm_gauss=std::make_shared<snn::NormalizedGaussInit>(0.f,0.01f);
    std::shared_ptr<snn::GaussInit> gauss=std::make_shared<snn::GaussInit>(0.f,0.25f);

    std::shared_ptr<snn::GaussMutation> mutation=std::make_shared<snn::GaussMutation>(0.f,0.01f,0.1f);
    std::shared_ptr<snn::OnePoint> cross=std::make_shared<snn::OnePoint>();

    auto relu=std::make_shared<snn::ReLu>();

    auto first= std::make_shared<snn::Layer<snn::ForwardNeuron<128>,32>>(256,gauss,cross,mutation);
    auto layer1=std::make_shared<snn::Layer<snn::ForwardNeuron<128>,32>>(64,gauss,cross,mutation);
    auto layer2=std::make_shared<snn::Layer<snn::ForwardNeuron<128>,32>>(2,gauss,cross,mutation);

    first->setActivationFunction(relu);

    layer1->setActivationFunction(relu);

    layer2->setActivationFunction(relu);

    snn::Network network;

    network.addLayer(first);
    network.addLayer(layer1);
    network.addLayer(layer2);

    snn::SIMDVector input;

    gauss->init(input,128);

    for(size_t i=0;i<10;i++)
    {
        gauss->init(input,128);

        clock_t start=clock();

        snn::SIMDVector output=network.fire(input);

        std::cout<<"Time: "<<(double)(clock()-start)/(double)CLOCKS_PER_SEC<<" s"<<std::endl;

        std::cout<<"Output: "<<output<<std::endl;
        
        network.applyReward(10.f*(i==0));        

        input.clear();
    }

    return 0;
}

