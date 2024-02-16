#include <experimental/simd>
#include <iostream>
#include <string_view>
#include <cmath>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <numeric>
#include <fstream>

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

number stddev(const snn::SIMDVector& vec)
{
    number mean=vec.dot_product();

    snn::SIMDVector omg=vec-mean;

    omg=omg*omg;

    return std::sqrt(omg.dot_product()/vec.size());

}

 
int main()
{

    snn::SIMDVector a([](size_t x)-> number{ return x;},128);
    snn::SIMDVector b([](size_t x)-> number{ return x;},128);

    std::shared_ptr<snn::NormalizedGaussInit> norm_gauss=std::make_shared<snn::NormalizedGaussInit>(0.f,0.01f);
    std::shared_ptr<snn::GaussInit> gauss=std::make_shared<snn::GaussInit>(0.f,0.1f);

    std::shared_ptr<snn::GaussMutation> mutation=std::make_shared<snn::GaussMutation>(0.f,0.01f,0.1f);
    std::shared_ptr<snn::OnePoint> cross=std::make_shared<snn::OnePoint>();

    //std::cout<<a+b<<std::endl;

    gauss->init(a,4096);

    a.set(a[0]+0,0);

    snn::FeedForwardNeuron<128,4> test;

    test.setup(gauss);

    snn::Layer<snn::FeedForwardNeuron<4096,1>,1,32> layer(4,norm_gauss,cross,mutation);

    std::ifstream file;

    file.open("../layer.bin",std::ios::in|std::ios::binary);

    if(file.is_open())
    {
        std::cout<<"No file layer.bin"<<std::endl;

        if(!layer.load(file))
        {
            std::cerr<<"Error loading layer!"<<std::endl;
            layer.setup(4,norm_gauss,cross,mutation);
        }

        file.close();
    }


    long double best_reward=-100;

    snn::ReLu activate;

    layer.setActivationFunction(std::make_shared<snn::Sigmoid>());

    while(abs(best_reward)>0.001f)
    {
        const auto start = std::chrono::steady_clock::now();

        layer.shuttle();

        snn::SIMDVector output=layer.fire(a)*50.f;

        //long double reward=-(stddev(output)+abs((1.f-output[0])));

        long double reward=-abs(20.f-output[0]);

        if(reward>best_reward)
        {
            std::cout<<"Best reward: "<<reward<<std::endl;
            best_reward=reward;
            std::cout<<"Output: "<<output[0]<<std::endl;

            std::ofstream file;

            file.open("layer.bin",std::ios::out|std::ios::binary|std::ios::trunc);

            layer.save(file);

            file.close();
        }

        layer.applyReward(reward);

        const auto end = std::chrono::steady_clock::now();

        const std::chrono::duration<double> diff = end - start;

       //std::cout << "Time: " << std::setw(9) << diff.count() << std::endl;

    }
   
}