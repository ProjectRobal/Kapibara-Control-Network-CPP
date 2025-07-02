#pragma once

#include <iostream>
#include <cstdint>
#include <vector>
#include <array>
#include <deque>
#include <cmath>

#include <simd_vector_lite.hpp>

#include <config.hpp>

#include "initializers/uniform.hpp"
#include "initializers/gauss.hpp"
#include <ranges>

namespace snn
{

    template<size_t InputSize,size_t NodeCount,class X_INIT = DEF_X_INIT,class Y_INIT = DEF_Y_INIT>
    class StaticKAN
    {
        
        struct SplineNode
        {
            number x0;
            number y0;
            number a;

            SplineNode(number x,number y)
            {
                this->x0 = x;
                this->y0 = y;
            }

            SplineNode()
            {

            }

            void fit(number x,number y)
            {
                this->a = (y-this->y0)/((x-this->x0)*(x-this->x0));
            }

            static size_t required_size_for_serialization()
            {
                return 2*sizeof(number);
            }

            /*
                Funtction used for serialization.

                You need to provied buffer with size from required_size_for_serialization() function.
            */
            void serialize(char* buffer) const
            {
                const size_t buffer_size = 2*sizeof(number);

                memcpy(buffer,(char*)&this->x0,sizeof(number));

                memcpy(buffer+sizeof(number),(char*)&this->y0,sizeof(number));

            }

            /*
                Funtction used for deserialization.

                You need to provied buffer with size from required_size_for_serialization() function.
            */
            void deserialize(char* buffer) 
            {
                memcpy((char*)&this->x0,buffer,sizeof(number));
                memcpy((char*)&this->y0,buffer+sizeof(number),sizeof(number));
            }

            void printInfo(std::ostream& out = std::cout) const
            {
                out<<"SplineNode: x0: "<<this->x0<<" y0: "<<this->y0<<" a: "<<this->a<<std::endl;
            }


        };


        struct Spline
        {
            private:

            std::array<SplineNode*,NodeCount> nodes;

            number biggest_y;

            number min_x;
            number max_x;

            // instead of binary search we can use interporlation search
            std::pair<SplineNode*,SplineNode*> search( number x ) const
            {
                if( this->nodes.size() == 0 )
                {
                    return std::pair(nullptr,nullptr);
                }

                if( this->nodes.size() == 1 )
                {
                    return std::pair(this->nodes[0],this->nodes[0]);
                }

                auto left = this->nodes.begin();
                auto right = this->nodes.end()-1;

                if( x >= this->max_x )
                {
                    return std::pair(nullptr,*left);
                }

                if( x < this->min_x )
                {
                    return std::pair(*right,nullptr);
                }

                number range = this->max_x - this->min_x;

                number length = this->nodes.size()-1;

                size_t i = std::floor(static_cast<number>((x - this->min_x)/range)*length);

                left = this->nodes.begin() + i;
                right = left + 1;

                return std::pair(*left,*right);
            }

            public:

            number get_max() const
            {
                return this->max_x;
            }

            number get_min() const
            {
                return this->min_x;
            }

            static inline SplineNode* make_node(number x,number y)
            {
                return  new SplineNode(x,y);
            }

            static inline SplineNode* make_node()
            {
                return new SplineNode();
            }

            void initialize()
            {
                // initalize nodes with random values

                X_INIT x_init;

                Y_INIT y_init;

                std::vector<number> unique;

                this->min_x = std::numeric_limits<number>::max();
                this->max_x = std::numeric_limits<number>::lowest();

                for(auto& node : this->nodes)
                {
                    number x = x_init.init();

                    // we need to make sure that x is unique
                    while( std::find(unique.begin(),unique.end(),x) != unique.end() )
                    {
                        x = x_init.init();
                    }

                    unique.push_back(x);

                    node = Spline::make_node(x,y_init.init());

                    this->min_x = std::min(this->min_x,node->x0);
                    this->max_x = std::max(this->max_x,node->x0);
                    
                }

                // sort after initialization
                std::sort(this->nodes.begin(),this->nodes.end(),[](SplineNode* a, SplineNode* b)
                                  {
                                      return a->x0 < b->x0;
                                  });
            }


            Spline()
            : nodes({nullptr})
            {
                this->biggest_y = 0;
                this->min_x = 0;
                this->max_x = 0;

                this->initialize();
                
            }

            void fit_by_index(number i,number x,number output,number target)
            {

                number error = abs(output - target);

                // if( error < 0.75f )
                // {
                //     // if error is too small, we do not do anything
                //     return;
                // }

                number coverage = (1.f - std::exp(-error))*0.5f;

                if( i < 0 )
                {
                    // nudge first node

                    SplineNode* node = this->nodes[0];

                    number dy = node->y0 - target;
                    node->y0 -= dy*coverage;

                    node->x0 -= (x - node->x0)*0.25f;

                    return;
                }

                if( i >= this->nodes.size() )
                {
                    // nudge last node

                    SplineNode* node = this->nodes[this->nodes.size()-1];

                    number dy = node->y0 - target;
                    node->y0 -= dy*coverage;
                    
                    node->x0 -= (x - node->x0)*0.25f;

                    return;
                }



                size_t index = static_cast<size_t>(i);

                // if i is in bounds, we nudge both nodes

                SplineNode* left = this->nodes[index];

                if( left->x0 == x )
                {

                    number dy = left->y0 - target;

                    left->y0 -= dy*coverage;

                    return;
                }

                SplineNode* right = this->nodes[index+1];

                snn::SIMDVectorLite<2> dx;

                dx[0] = left->x0;
                dx[1] = right->x0;

                snn::SIMDVectorLite<2> dy;

                dy[0] = left->y0;
                dy[1] = right->y0;

                dx = dx - x;

                dx*=coverage;

                dy = dy - target;

                dy*=coverage;
                
                left->x0 -= dx[0];
                right->x0 -= dx[1];

                left->y0 -= dy[0];
                right->y0 -= dy[1];
            }

            SplineNode* get_by_index(number i) const
            {
                if( i >= this->nodes.size() || i < 0 )
                {
                    return nullptr;
                }

                SplineNode* node = this->nodes[static_cast<size_t>(i)];

                if( i == this->nodes.size() - 1 )
                {
                    node->a = 0;
                    return node;
                }


                SplineNode* right = this->nodes[static_cast<size_t>(i)+1];

                if( node->x0 != right->x0 )
                {

                    node->fit(right->x0,right->y0);

                }
                else
                {
                    node->a = 0;
                }

                return node;
            }

            size_t length() const
            {
                return NodeCount;
            }

            void save(std::ostream& out) const
            {
                const uint32_t length = this->nodes.size();

                // write total numbers of nodes
                out.write((char*)&length,sizeof(uint32_t));

                const size_t buffer_size = SplineNode::required_size_for_serialization();

                char buffer[buffer_size] = {0};

                for( const SplineNode* node : this->nodes )
                {
                    node->serialize(buffer);

                    out.write(buffer,buffer_size);
                }
                
            }

            void load(std::istream& in)
            {
                uint32_t length = 0;
                
                // read length of nodes
                in.read((char*)&length,sizeof(uint32_t));


                const size_t buffer_size = SplineNode::required_size_for_serialization();

                char buffer[buffer_size] = {0};

                for(size_t i=0;i<length;++i)
                {
                    in.read(buffer,buffer_size);

                    SplineNode* node = make_node();

                    node->deserialize(buffer);

                    this->nodes.push_back(node);
                }

                // we need to keep the order of nodes the way as they were saved
                std::reverse(this->nodes.begin(),this->nodes.end());

            }

            void printInfo(std::ostream& out = std::cout) const
            {
                out<<"Spline Info:"<<std::endl;
                out<<"Nodes: "<<this->nodes.size()<<std::endl;
                out<<"Min X: "<<this->min_x<<std::endl;
                out<<"Max X: "<<this->max_x<<std::endl;

                for(const SplineNode* node : this->nodes)
                {
                    node->printInfo(out);
                }
            }

            ~Spline()
            {
                for(size_t i=0;i<this->nodes.size();++i)
                {
                    delete this->nodes[i];
                }

            }

        };


        Spline splines[InputSize];

        // used in fiting process to distribute error
        snn::SIMDVectorLite<InputSize> active_values;

        snn::UniformInit<0.f,1.f> uniform_init;

        public:

        StaticKAN()
        {

            for(size_t i=0;i<InputSize;++i)
            {
                this->active_values[i] = 1.f;
            }

            number prob_mean = this->active_values.reduce();

            this->active_values /= prob_mean;

        }

        void print_spline_info(std::ostream& out = std::cout) const
        {
            out<<"Spline Info:"<<std::endl;
            out<<"Input Size: "<<InputSize<<std::endl;

            for(size_t i=0;i<InputSize;++i)
            {
                out<<"Spline "<<i<<": "<<this->splines[i].length()<<" nodes"<<std::endl;
            }
        }


        number fire(const snn::SIMDVectorLite<InputSize>& input)
        {
            number output = 0.f;

            snn::SIMDVectorLite<InputSize> x(0.f);
            snn::SIMDVectorLite<InputSize> y(0.f);
            snn::SIMDVectorLite<InputSize> a(0.f);

            snn::SIMDVectorLite<InputSize> max_x(0.f);
            snn::SIMDVectorLite<InputSize> min_x(0.f);

            size_t index = 0;

            for(const Spline& spline : this->splines)
            {
                max_x[index] = spline.get_max();
                min_x[index] = spline.get_min();

                index += 1;
            }

            snn::SIMDVectorLite<InputSize> range = max_x - min_x;

            snn::SIMDVectorLite<InputSize> indexes = ((input - min_x)/range)*(NodeCount-1);

            index = 0;    
            
            
            for(const Spline& spline : this->splines)
            {
                // that part takes some time, but how to retrive elements faster?
                auto node = spline.get_by_index(indexes[index]);

                if( node )
                {
                    x[index] = node->x0;
                    y[index] = node->y0;
                    a[index] = node->a;

                }

                index ++;
            }

            snn::SIMDVectorLite<InputSize> x_x = input - x;

            x_x = a*(x_x*x_x) + y;

            output += x_x.reduce();

            // generate fit select probablitiy for each node

            

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

            snn::SIMDVectorLite<InputSize> x(0.f);
            snn::SIMDVectorLite<InputSize> y(0.f);
            snn::SIMDVectorLite<InputSize> a(0.f);

            snn::SIMDVectorLite<InputSize> max_x(0.f);
            snn::SIMDVectorLite<InputSize> min_x(0.f);

            size_t index = 0;

            for(const Spline& spline : this->splines)
            {
                max_x[index] = spline.get_max();
                min_x[index] = spline.get_min();

                index += 1;
            }

            snn::SIMDVectorLite<InputSize> range = max_x - min_x;


            snn::SIMDVectorLite<InputSize> indexes = ((input - min_x)/range)*(NodeCount-1);


            index = 0;

            for(Spline& spline : this->splines)
            {
                // spline.fit(input[index],output_errors[index],y_errors[index]);

                spline.fit_by_index(indexes[index],input[index],output_errors[index],y_errors[index]);

                index ++;
            }

        }

        void printInfo(std::ostream& out = std::cout) const
        {
            out<<"EvoKAN Info:"<<std::endl;
            out<<"Input Size: "<<InputSize<<std::endl;

            this->splines[0].printInfo(out);
        }

        void save(std::ostream& out) const
        {
            // save splines
            for(const Spline& spline : this->splines)
            {
                spline.save(out);
            }

        }

        void load(std::istream& in)
        {
            for(Spline& spline : this->splines)
            {
                spline.load(in);
            }
        }

    };

}