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

#include "networks/kac_actor_crictic.hpp"

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

 
int main()
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
    
    while(maxSteps--)
    {

        number x=network.step(inputs)[0];

        long double reward=-abs(evaluatePolynomial(inputs,x));

        if(reward>best_reward)
        {
            std::cout<<"Best reward: "<<reward<<" at step: "<<step<<std::endl;
            std::cout<<"Best x: "<<x<<std::endl;
            best_reward=reward;
        }

        network.applyReward(reward);

        ++step;
    }
}