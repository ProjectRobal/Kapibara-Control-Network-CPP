#pragma once

#include <iostream>
#include <cstdint>
#include <vector>

#include <simd_vector_lite.hpp>

#include <config.hpp>

#include "initializers/uniform.hpp"
#include <ranges>

namespace snn
{

    template<size_t InputSize,size_t MaxNodes=40>
    class EvoKAN
    {
        // spline will be formed by using euqation:
        // y*exp(-(u-x)^2 * stretch)

        // y0*exp((x0-x)^2 * b)
        struct SplineNode
        {
            number x0;
            number y0;
            number b;

            SplineNode(number x,number y)
            {
                this->x0 = x;
                this->y0 = y;
                this->b = 100.f; // default stretch value
            }

            SplineNode()
            {
                this->b = 0.f;
            }

        };

        struct SplineNodePromise
        {
            private:

            const SplineNode* node;
            bool valid;

            public:

            SplineNodePromise(const SplineNode* _node)
            : node(_node)
            {
                this->valid = true;
            }

            SplineNodePromise()
            : node(nullptr)
            {
                this->valid = false;
            }

            operator bool()
            {
                return this->valid;
            }

            const SplineNode& value()
            {
                return *this->node;
            }

        };

        struct Spline
        {
            std::vector<SplineNode> nodes;

            number biggest_y;

            Spline()
            {
                this->biggest_y = 0;
            }

            void fit(number x,number output,number target)
            {

                // add new node when error is too high

                number dy = target - output;

                SplineNode new_node(x,dy);

                this->nodes.push_back(new_node);

            }

            SplineNodePromise get_node(size_t index) const
            {
                if( index >= this->nodes.size() )
                {
                    return SplineNodePromise();
                }

                return SplineNodePromise(&this->nodes[index]);
            }

            size_t length() const
            {
                return this->nodes.size();
            }
        };


        Spline splines[InputSize];

        // used in fiting process to distribute error
        snn::SIMDVectorLite<InputSize> active_values;

        snn::UniformInit<0.f,1.f> uniform_init;

        public:

        EvoKAN()
        {}

        number fire(const snn::SIMDVectorLite<InputSize>& input)
        {
            number output = 0.f;

            bool block_left = false;

            size_t iter = 0;


            do
            {

                snn::SIMDVectorLite<InputSize> x(0.f);
                snn::SIMDVectorLite<InputSize> y(0.f);
                snn::SIMDVectorLite<InputSize> b(0.f);

                size_t index = 0;
                
                block_left = false;

                for(const Spline& spline : this->splines)
                {
                    auto node = spline.get_node(iter);

                    if( node )
                    {
                        auto val = node.value();

                        x[index] = val.x0;
                        y[index] = val.y0;
                        b[index] = val.b;

                        block_left = true;
                    }

                    index ++;
                }

                snn::SIMDVectorLite<InputSize> x_x = input - x;

                x_x = ((x_x * x_x)*b*-1.f).exp()*y;

                output += x_x.reduce();

                iter++;

            }while(block_left);


            // generate fit select probablitiy for each node

            for(size_t i=0;i<InputSize;++i)
            {
                this->active_values[i] = this->uniform_init.init();
            }

            number prob_mean = this->active_values.reduce();

            this->active_values /= prob_mean;

            return output;
        }

        void fit(const snn::SIMDVectorLite<InputSize>& input,number output,number target)
        {

            number error = abs(output - target);

            // if error is too small, we do not do anything
            if( error < 0.01f )
            {
                return;
            }

            // select node to fit

            size_t node_index = 0;
            

            snn::SIMDVectorLite<InputSize> y_errors = this->active_values*target;
            snn::SIMDVectorLite<InputSize> output_errors = this->active_values*output;

            size_t index = 0;

            for(Spline& spline : this->splines)
            {
                spline.fit(input[index],output_errors[index],y_errors[index]);

                index ++;
            }

        }

        void printInfo(std::ostream& out = std::cout) const
        {
            out<<"EvoKAN Info:"<<std::endl;
            out<<"Input Size: "<<InputSize<<std::endl;

            for(size_t i=0;i<InputSize;++i)
            {
                out<<"Spline "<<i<<": "<<this->splines[i].length()<<" nodes"<<std::endl;
            }
        }

    };

}