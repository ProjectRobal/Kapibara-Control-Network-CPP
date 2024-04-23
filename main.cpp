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

#include "activation/sigmoid.hpp"
#include "activation/relu.hpp"

#include "networks/kac_actor_crictic.hpp"

#include "shared_mem.hpp"

#include "proto/cartpole.pb.h"

#include "replay/short_term_memory.hpp"

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
    std::shared_ptr<snn::GaussInit> gauss=std::make_shared<snn::GaussInit>(0.f,0.1f);

    std::shared_ptr<snn::GaussMutation> mutation=std::make_shared<snn::GaussMutation>(0.f,0.01f,0.1f);
    std::shared_ptr<snn::OnePoint> cross=std::make_shared<snn::OnePoint>();


    ShortTermMemory memory(10);

    number reward_to_find=0.f;

    for(size_t i=0;i<15;i++)
    {
        snn::SIMDVector input;
        gauss->init(input,4);
        snn::SIMDVector output;
        gauss->init(output,4);

        number reward;

        gauss->init(reward);

        reward_to_find+=reward;

        //std::cout<<reward<<std::endl;

        memory.append(input,output,reward);
    }

    reward_to_find/=15;

    double estimated=0;

    snn::SIMDVector input;
    snn::SIMDVector output;

    while(estimated==0)
    {
        input.clear();
        output.clear();

        
        gauss->init(input,4);
        gauss->init(output,4);

        memory.EstimateRewardFor(input,output,estimated);

    }

    std::cout<<"Input: "<<input<<std::endl;
    std::cout<<"Output: "<<output<<std::endl;
    std::cout<<"Size: "<<memory.getSize()<<std::endl;

    std::cout<<"Estimated: "<<estimated<<std::endl;


    return 0;

    long double best_reward=-100;

    snn::ReLu activate;

    std::shared_ptr<snn::ReLu> relu=std::make_shared<snn::ReLu>();

    const size_t input_size=4;
    const size_t output_size=2;
    
    snn::ActorCriticNetwork<input_size,output_size,512,64> network;

    network.setup(10,norm_gauss,cross,mutation);

    std::shared_ptr<snn::Layer<snn::ForwardNeuron<4>,1,256>> layer=std::make_shared<snn::Layer<snn::ForwardNeuron<4>,1,256>>(16,norm_gauss,cross,mutation);
    std::shared_ptr<snn::Layer<snn::ForwardNeuron<16>,1,256>> layer1=std::make_shared<snn::Layer<snn::ForwardNeuron<16>,1,256>>(8,norm_gauss,cross,mutation);
    std::shared_ptr<snn::Layer<snn::ForwardNeuron<8>,1,256>> layer2=std::make_shared<snn::Layer<snn::ForwardNeuron<8>,1,256>>(2,norm_gauss,cross,mutation);

    layer->setActivationFunction(relu);
    layer1->setActivationFunction(relu);
    layer2->setActivationFunction(relu);

    network.addLayer(layer);
    network.addLayer(layer1);
    network.addLayer(layer2);
    
    // we will try to find poles in this polynomials in form of a[0]*x^3 + a[1]*x^2 + a[2] * x + a[3] = 0; 

    snn::SIMDVector inputs({0.25,0.5,0.6,0.4});  

    size_t step=1;  

    size_t maxSteps=50000;

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

    /*
    
    Network:    CartPole
            <-  Sends inputs
    Sends outputs ->
            <-  Sends reward
    Loop

    */

   //std::cout<<network.step(inputs)<<std::endl;

    
    while(maxSteps--)
    {

        // recive inputs

        interface=getInterface();

        if(interface.inputs_size()<4)
        {
            continue;
        }

        inputs.set(interface.inputs(0),0);
        inputs.set(interface.inputs(1),1);
        inputs.set(interface.inputs(2),2);
        inputs.set(interface.inputs(3),3);

        std::chrono::time_point start=std::chrono::steady_clock::now();

        snn::SIMDVector outputs=network.step(inputs);

        interface.set_outputs(0,outputs[0]);
        interface.set_outputs(1,outputs[1]);

        // send outputs

        sendInterface(interface);

        // get reward

        interface=getInterface();

        long double reward=interface.reward()-1;


        if(reward>best_reward)
        {
            std::cout<<"Best reward: "<<reward<<" at step: "<<step<<std::endl;
            best_reward=reward;
        }

        std::cout<<"Reward: "<<reward<<std::endl;

        network.applyReward(reward);

        std::chrono::time_point end=std::chrono::steady_clock::now();

        const std::chrono::duration<double> elapsed_seconds(end-start);

        std::cout<<"Time "<<elapsed_seconds<<" s"<<std::endl;

        //std::cout<<"Reward: "<<reward<<std::endl;

        if(!interface.SerializeToString(&buffer))
        {
            return -12;
        }        

        ++step;

    }
}

