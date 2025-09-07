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

#include <opencv2/opencv.hpp>


#include "config.hpp"

#include "initializers/gauss.hpp"
#include "initializers/constant.hpp"
#include "initializers/uniform.hpp"
#include "initializers/hu.hpp"



/*

    My idea for QKAN is to aproximate spline using 256 elements look up table.

    There fore we could therotically save on computation.

    In normall CNN we use FFT for faster convolution but how to do it when 
    kernal is made of KAN function? 

    Where too fit SIMD into it, well we could use it for faster addition with reduction.

*/

template<size_t InputSize,size_t OutputSize=1,bool Hebbian_Learning = true,class Init=snn::UniformInit<-127.0,128.0>>
class QKAN
{
    protected:

    int8_t *table;
    int16_t *decimal;
    Init init;


    void hebb_learn(int8_t x[],size_t neuron_index,int8_t y)
    {
        for(size_t i=0;i<InputSize;++i)
        {   
            // Update only handful of weights
            int8_t random = this->init.init();
            if( random <= -25 && random >= 25)
            {
                continue;
            }

            int8_t w = this->get_weight(neuron_index,i,x[i]);

            int16_t decimal = this->get_decimal(neuron_index,i,x[i]);

            decimal += w*y/10000;

            int8_t dx = decimal/1000;

            dx = dx*(w>-127)*(w<128);

            w += dx;

            // when dx is 1 ( decimal reached threshold ), set decimal to zero
            decimal = decimal*(1-abs(dx));

            this->get_decimal(neuron_index,i,x[i]) = decimal;

            this->get_weight(neuron_index,i,x[i]) = w;

        }
    }

    int8_t& get_weight(size_t neuron_index,uint8_t weight_index,int8_t input)
    {
        uint8_t _x = input + 127;

        return this->table[neuron_index*256*InputSize + weight_index*256 + _x];
    }

    int8_t get_weight(size_t neuron_index,uint8_t weight_index,int8_t input) const
    {
        uint8_t _x = input + 127;

        return this->table[neuron_index*256*InputSize + weight_index*256 + _x];
    }

    int16_t& get_decimal(size_t neuron_index,uint8_t weight_index,int8_t input)
    {
        uint8_t _x = input + 127;

        return this->decimal[neuron_index*256*InputSize + weight_index*256 + _x];
    }

    int16_t get_decimal(size_t neuron_index,uint8_t weight_index,int8_t input) const
    {
        uint8_t _x = input + 127;

        return this->decimal[neuron_index*256*InputSize + weight_index*256 + _x];
    }

    public:

    QKAN()
    {
        this->table = new int8_t[256*InputSize*OutputSize];

        if(Hebbian_Learning)
        {
            this->decimal = new int16_t[256*InputSize*OutputSize](0);
        }


        for(size_t i=0;i<256*InputSize*OutputSize;++i)
        {
            this->table[i] = this->init.init();
        }
    }


    void fire(int8_t x[],int8_t y[])
    {

        for(size_t o=0;o<OutputSize;++o)
        {
            int32_t sum = 0;         
            
            for(size_t i=0;i<InputSize;++i)
            {
                sum += this->get_weight(o,i,x[i]);
            }

            y[o] = sum/InputSize;

            if(Hebbian_Learning)
            {
                this->hebb_learn(x,o,y[o]);
            }

        }
    }

    /*
    
    A simple fitting but really? We will update only handfull of weights but waht about situations
    when weights are near the extreame?

    */
    void fit(int8_t x[],int8_t y[],int8_t target[])
    {
        for(size_t o=0;o<OutputSize;++o)
        {
            int8_t error = y[o] - target[o];

            int8_t to_update = std::min<int8_t>(4,InputSize);

            int8_t dy = error/to_update;

            for(size_t i=0;i<InputSize;++i)
            {
                int8_t& w = this->get_weight(o,i,x[i]);



            }

        }
    }


    ~QKAN()
    {
        delete[] this->table;
    }

};

void perform_convolution(QKAN<3*3,1>& kan,int8_t* input_img,int8_t* output_img,size_t width)
{
    int8_t buffer[9];

    for(size_t y=1;y<width-1;++y)
    {
        for(size_t x=1;x<width-1;++x)
        {
            int8_t output;

            buffer[0] = input_img[(y-1)*width + (x-1)];
            buffer[1] = input_img[(y-1)*width + x];
            buffer[2] = input_img[(y-1)*width + x+1];

            buffer[3] = input_img[y*width + (x-1)];
            buffer[4] = input_img[y*width + x];
            buffer[5] = input_img[y*width + x+1];

            buffer[6] = input_img[(y+1)*width + (x-1)];
            buffer[7] = input_img[(y+1)*width + x];
            buffer[8] = input_img[(y+1)*width + x+1];

            kan.fire(buffer,&output);

            output_img[(width-2)*(y-1)+(x-1)] = output;
        }

    }

}


int main(int argc,char** argv)
{
    std::cout<<"Starting..."<<std::endl;

    std::chrono::time_point<std::chrono::system_clock> start, end;

    snn::UniformInit<(number)-127.f,(number)128.f> noise;

    const size_t width = 224;

    const size_t input_size = width*width;

    int8_t data[input_size];

    for(size_t i=0;i<input_size;i++)
    {
        data[i] = noise.init();
    }

    start = std::chrono::system_clock::now();

    // ai = _mm512_loadu_epi8(data);

    std::cout<<"Timestamp: "<<std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start).count()<<" ms"<<std::endl;

    int8_t output_img[(width-2)*(width-2)];

    // Lets test image like inference
    QKAN<3*3,1> kan;

    // Take chunk of an image

    clock_t _start = clock(); 

    perform_convolution(kan,data,output_img,width);

    std::cout<<"Timestamp: "<<static_cast<double>(clock()-_start)/CLOCKS_PER_SEC * 1000<<" us"<<std::endl;

    for(size_t i=0;i<32;++i)
    {
        std::cout<<(int32_t)output_img[i]<<" ";
    }
    
    return 0;

}

