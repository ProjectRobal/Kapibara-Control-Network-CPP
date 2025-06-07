#pragma once

#include <iostream>
#include <cstdint>
#include <vector>
#include <cmath>

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
                this->b = 1000.f; // default stretch value
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
            private:

            std::vector<SplineNode> nodes;

            number biggest_y;

            inline void fit_add_new_node(number x,number output,number target)
            {
                number dy = target - output;

                SplineNode new_node(x,dy);

                this->nodes.push_back(new_node);
            }

            inline void fit_move_towards_y(number x,number output,number target)
            {
                for( SplineNode& node : nodes )
                {
                    number error = abs( x - node.x0 );

                    if( error <= 0.00001f )
                    {
                        number dy = target - output;

                        node.y0 = ( node.y0 + dy ) / 2.f;

                    }
                }

            }

            inline void fit_stretch_node(number x,number output,number target)
            {

                number min_x_dist = abs( this->nodes[0].x0 - x );
                size_t min_i = 0;

                for( size_t i = 1; i < this->nodes.size() ; i++ )
                {
                    number x_dist = abs( this->nodes[i].x0 - x );

                    if( x_dist < min_x_dist )
                    {
                        min_i = i;
                        min_x_dist = x_dist;
                    }
                }

                SplineNode& node = this->nodes[min_i];

                if( target >= node.y0 )
                {

                    this->fit_add_new_node(x,output,target);

                    return;
                }

                node.b = std::log(target/node.y0)/( -( x - node.x0 )*( x - node.x0 ) );
            }

            public:

            Spline()
            {
                this->biggest_y = 0;
            }

            void fit(number x,number output,number target)
            {
                number error = abs(output-target);

                // add a node when there is no node
                if( this->nodes.size() == 0 )
                {
                    this->fit_add_new_node(x,output,target);

                    return;
                }

                // when error is not so big find nearest node and nudge it b value to fit target
                if( error <= 0.2f )
                {
                    this->fit_stretch_node(x,output,target);

                    return;
                }
                // when error is very close to zero, find node very close to target and move it's y towards target
                else if( error < 0.00001f )
                {   
                    this->fit_move_towards_y(x,output,target);

                    return;
                }

                // add new node when error is too high

                this->fit_add_new_node(x,output,target);

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