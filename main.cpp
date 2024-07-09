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

#include "fastkac.hpp"

#include "activation/sigmoid.hpp"
#include "activation/relu.hpp"
#include "activation/softmax.hpp"

#include "shared_mem.hpp"

#include "proto/cartpole.pb.h"

#include "network.hpp"

#include "serializaers/network_serialize.hpp"

number stddev(const snn::SIMDVector& vec)
{
    number mean=vec.reduce();

    snn::SIMDVector omg=vec-mean;

    omg=omg*omg;

    return std::sqrt(omg.reduce()/vec.size());

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


// evalute a minimum of parabola
number evaluate(const snn::SIMDVector& input)
{
    return -abs((input[0]*input[0] + 10*input[1] + 2));
}

/*

 Our theory is right but only when input values are positive,
 the other problem is exploding gradient

 but it has some potential in discreate problems.

*/


int main(int argc,char** argv)
{

    std::shared_ptr<snn::NormalizedGaussInit> norm_gauss=std::make_shared<snn::NormalizedGaussInit>(0.f,0.01f);
    std::shared_ptr<snn::GaussInit> gauss=std::make_shared<snn::GaussInit>(0.f,0.25f);

    std::shared_ptr<snn::GaussMutation> mutation=std::make_shared<snn::GaussMutation>(0.f,0.01f,0.1f);
    std::shared_ptr<snn::OnePoint> cross=std::make_shared<snn::OnePoint>();

    auto relu=std::make_shared<snn::ReLu>();

    auto first= std::make_shared<snn::FastKAC>(1,2,gauss,mutation,32);

    first->setActivationFunction(std::make_shared<snn::SoftMax>());

    //layer3->setActivationFunction(relu);

    snn::Network network;

    network.addLayer(first);

    //snn::SIMDVector input;

    //gauss->init(input,16);

    //input = snn::exp(input);

    //snn::SIMDVector output = network.fire(input);

    /*std::cout<<"Input: "<<input<<std::endl;
    std::cout<<"Output: "<<output<<std::endl;

    network.applyReward(-10.f);

    output = network.fire(input);

    std::cout<<"Input: "<<input<<std::endl;
    std::cout<<"Output: "<<output<<std::endl;
    */

    //snn::NetworkSerializer::load(network,"checkpoint");

    snn::SIMDVector input;

    gauss->init(input,1);

    std::fstream file;

    file.open("log.csv",std::ios::out);

    file<<"N"<<";"<<"reward"<<std::endl;

    int target_position=40;

    int initial_position=0;

    int last_position=0;

    std::random_device rd; 

    // Mersenne twister PRNG, initialized with seed from previous random device instance
    std::mt19937 gen(rd()); 

    std::uniform_int_distribution<int> uniform(0,1);

    for(size_t i=0;i<10000;i++)
    {
        //gauss->init(input,128);

        if(uniform(gen))
        {
            target_position = -target_position;
        }

        clock_t start=clock();

        input.set(static_cast<number>(initial_position)/100.f,0);

        snn::SIMDVector output=network.fire(input);

        std::cout<<"Time: "<<(double)(clock()-start)/(double)CLOCKS_PER_SEC<<" s"<<std::endl;

        std::cout<<"Output: "<<output<<std::endl;

        number reward = 0;

        if( output[0] > output[1] )
        {
            // move left

            initial_position--;
        }
        else
        {
            // move right

            initial_position++;
        }

        if( fabs(initial_position - target_position) >= fabs(last_position - target_position) )
        {
            reward = -1;
        }
        else if( fabs(initial_position - target_position) < fabs(last_position - target_position) )
        {
            reward = 1;
        }

        last_position = initial_position;
        
        network.applyReward(reward);  

        std::cout<<"Reward: "<<reward<<std::endl;
        std::cout<<"Agent pos: "<<initial_position<<" Target pos: "<<target_position<<std::endl;

        file<<i<<";"<<reward<<std::endl;

        if( initial_position == target_position )
        {
            std::cout<<"Position reached"<<std::endl;
            break;
        }     

        //input.clear();
    }

    file.close();

    //snn::NetworkSerializer::save(network,"checkpoint");

    return 0;
    
}

