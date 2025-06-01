#pragma once

#include <iostream>
#include <cstdint>
#include <vector>

#include <simd_vector_lite.hpp>

#include <config.hpp>

#include "initializers/uniform.hpp"

namespace snn
{

    template<size_t InputSize,size_t MaxNodes=40>
    class EvoKAN
    {
        // spline will be formed by using euqation:
        // y*exp(-(u-x)^2 * stretch)


        // x points of spline
        std::vector<snn::SIMDVectorLite<InputSize>> x;
        // y points of spline
        std::vector<snn::SIMDVectorLite<InputSize>> y;
        // stretch
        std::vector<snn::SIMDVectorLite<InputSize>> stretch;

        size_t nodes_count[InputSize];

        size_t row_count;

        snn::SIMDVectorLite<InputSize> select_probabilities;

        snn::UniformInit<0.f,1.f> uniform_init;

        public:

        EvoKAN()
        : nodes_count({0}),
        row_count(0)
        {}

        number fire(const snn::SIMDVectorLite<InputSize>& input)
        {
            number output = 0.f;

            for(size_t i=0;i<this->x.size();++i)
            {
                snn::SIMDVectorLite<InputSize> x_x = input - x[i];

                x_x = ((x_x * x_x)*stretch[i]*-1.f).exp()*y[i];

                output += x_x.reduce();
            }

            // generate fit select probablitiy for each node

            for(size_t i=0;i<InputSize;++i)
            {
                this->select_probabilities[i] = this->uniform_init.init();
            }

            number prob_mean = this->select_probabilities.reduce();

            this->select_probabilities /= prob_mean;

            return output;
        }

        void fit(const snn::SIMDVectorLite<InputSize>& input,number output,number target)
        {

            // select node to fit

            size_t node_index = 0;

            number prob = this->uniform_init.init();
            
            for(size_t i=0;i<InputSize;++i)
            {
                prob -= this->select_probabilities[i];

                if(prob <= 0.f)
                {
                    node_index = i;
                    break;
                }
            }

            // we do this if the error is too large

            number x = input[node_index];

            number y = target - output*2.f;

            number b = 100.f;

            if( nodes_count[node_index] < row_count)
            {
                const size_t i = nodes_count[node_index];

                this->x[i][node_index] = x;
                this->y[i][node_index] = y;
                this->stretch[i][node_index] = b;
                nodes_count[node_index] += 1;
            }
            else
            {
                // add new row

                this->x.push_back(snn::SIMDVectorLite<InputSize>());
                this->y.push_back(snn::SIMDVectorLite<InputSize>());
                this->stretch.push_back(snn::SIMDVectorLite<InputSize>());

                this->x[row_count][node_index] = x;
                this->y[row_count][node_index] = y;
                this->stretch[row_count][node_index] = b;

                nodes_count[node_index] += 1;
                row_count += 1;
            }

        }

        void printInfo(std::ostream& out = std::cout) const
        {
            out<<"EvoKAN Info:"<<std::endl;
            out<<"Input Size: "<<InputSize<<std::endl;
            out<<"Max Nodes: "<<MaxNodes<<std::endl;
            out<<"Row Count: "<<row_count<<std::endl;

            for(size_t i=0;i<InputSize;++i)
            {
                out<<"Node "<<i<<": "<<nodes_count[i]<<" nodes"<<std::endl;
            }
        }

    };

}