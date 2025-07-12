#pragma once

#include <iostream>
#include <cstdint>
#include <vector>
#include <deque>
#include <cmath>

#include <simd_vector_lite.hpp>

#include <config.hpp>

#include "initializers/uniform.hpp"
#include <ranges>

namespace snn
{

    template<size_t InputSize>
    class EvoKAN
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


        };

        typedef std::shared_ptr<SplineNode> NodeRef;

        


        struct Spline
        {
            private:

            std::vector<SplineNode*> nodes;

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


            Spline()
            {
                this->biggest_y = 0;
                this->min_x = 0;
                this->max_x = 0;
                this->nodes.reserve(1024);
            }

            void fit(number x,number index,number output,number target)
            {
                number error = abs(output-target);

                if( error < 0.00001f )
                {
                    // if error is too small, we do not do anything
                    return;
                }

                // auto points = this->search(x); 
                
                if( this->nodes.size() < 2)
                {
                    // if there are no nodes, we need to create first one
                    SplineNode* node = make_node(x,target);

                    this->nodes.push_back(node);

                    if( this->nodes.size() == 0 )
                    {
                        // if this is first node, we need to set min and max x
                        this->min_x = x;
                        this->max_x = x;
                    }
                    else
                    {
                        // if this is second node, we need to set min and max x
                        this->min_x = std::min(this->min_x,x);
                        this->max_x = std::max(this->max_x,x);
                    }


                    std::sort(this->nodes.begin(),this->nodes.end(),[](SplineNode* a, SplineNode* b)
                    {
                        return a->x0 < b->x0;
                    });

                    return;
                }   

                SplineNode *near_node = this->nodes[index];

                if( (index >=0 && index < this->nodes.size()) && ( abs(near_node->x0 - x) < 0.0001f ) )
                {
                    number dy = near_node->y0 - target;

                    near_node->y0 -= 0.5*dy;

                    return;
                }

                SplineNode* node = make_node(x,target);

                this->nodes.push_back(node);

                // if( index < 0 )
                // {
                //     this->nodes.insert(this->nodes.begin(), node);
                // }
                // else if(index >= this->nodes.size())
                // {
                //     this->nodes.push_back(node);
                // }
                // else
                // {
                //     this->nodes.insert(this->nodes.begin() + index, node);
                // }

                std::sort(this->nodes.begin(),this->nodes.end(),[](SplineNode* a, SplineNode* b)
                {
                    return a->x0 < b->x0;
                });
                

                this->min_x = std::min(this->min_x,x);
                this->max_x = std::max(this->max_x,x);

            }

            SplineNode* get_by_index(number i) const
            {
                if( i >= this->nodes.size() || i < 0 )
                {
                    return nullptr;
                }

                size_t index = static_cast<size_t>(i);

                SplineNode* node = this->nodes[index];

                if( index == this->nodes.size()-1  )
                {
                    node->a = 0;

                    return node;
                }


                SplineNode* right = this->nodes[index+1];

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

            SplineNode* get_node(number x) const
            {
                
                auto nodes = this->search(x);

                if( nodes.first == nullptr || nodes.second == nullptr )
                {
                    return nullptr;
                }


                SplineNode* node = nodes.first;
                SplineNode* right = nodes.second;

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
                return this->nodes.size();
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

            ~Spline()
            {
                for(size_t i=0;i<this->nodes.size();++i)
                {
                    delete this->nodes[i];
                }

                this->nodes.clear();
            }

        };


        Spline splines[InputSize];

        // used in fiting process to distribute error
        snn::SIMDVectorLite<InputSize> active_values;

        snn::UniformInit<(number)0.f,(number)1.f> uniform_init;

        snn::SIMDVectorLite<InputSize> x_x;

        public:

        EvoKAN()
        {

            for(size_t i=0;i<InputSize;++i)
            {
                if( this->uniform_init.init() < 0.25f )
                {
                    this->active_values[i] = this->uniform_init.init();
                }
                else
                {
                    this->active_values[i] = 0.f;
                }
            }

            number prob_mean = this->active_values.reduce();

            this->active_values /= prob_mean;

        }


        number fire(const snn::SIMDVectorLite<InputSize>& input)
        {
            number output = 0.f;

            snn::SIMDVectorLite<InputSize> x(0.f);
            snn::SIMDVectorLite<InputSize> y(0.f);
            snn::SIMDVectorLite<InputSize> a(0.f);

            snn::SIMDVectorLite<InputSize> max_x(0.f);
            snn::SIMDVectorLite<InputSize> min_x(0.f);
            snn::SIMDVectorLite<InputSize> length(0.f);

            size_t index = 0;

            for(const Spline& spline : this->splines)
            {
                max_x[index] = spline.get_max();
                min_x[index] = spline.get_min();
                length[index] = static_cast<number>(spline.length());

                index++;
            }

            snn::SIMDVectorLite<InputSize> range = max_x - min_x;

            length-=1;

            snn::SIMDVectorLite<InputSize> indexes = ((input - min_x)/range)*length;

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

            this->x_x = input - x;

            x_x = a*(x_x*x_x) + y;

            output += x_x.reduce();

            // generate fit select probablitiy for each node

            for(size_t i=0;i<InputSize;++i)
            {
                if( this->uniform_init.init() < 0.25f )
                {
                    this->active_values[i] = this->uniform_init.init();
                }
                else
                {
                    this->active_values[i] = 0.f;
                }
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
                // std::cout<<"Error is too small, skipping fit"<<std::endl;
                return;
            }

            // select node to fit

            snn::SIMDVectorLite<InputSize> y_errors = this->active_values*(target);
            snn::SIMDVectorLite<InputSize> outputs = this->x_x;

            // std::cout<<"Outputs: "<<outputs<<std::endl;
            // std::cout<<"Y Errors: "<<y_errors<<" sum: "<<y_errors.reduce()<<std::endl;

            snn::SIMDVectorLite<InputSize> max_x(0.f);
            snn::SIMDVectorLite<InputSize> min_x(0.f);
            snn::SIMDVectorLite<InputSize> length(0.f);

            size_t index = 0;

            for(;index<InputSize;)
            {
                max_x[index] = splines[index].get_max();
                min_x[index] = splines[index].get_min();
                length[index] = static_cast<number>(splines[index].length());

                if( length[index] < 2)
                {
                    max_x[index] = min_x[index] + 0.01f;
                    min_x[index] = max_x[index] - 0.01f;
                }

                index += 1;
            }

            snn::SIMDVectorLite<InputSize> range = max_x - min_x;

            snn::SIMDVectorLite<InputSize> indexes = ((input - min_x)/range)*(length-1);

            index = 0;

            for(Spline& spline : this->splines)
            {
                spline.fit(input[index],indexes[index],outputs[index],outputs[index] + y_errors[index]);

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